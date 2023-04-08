from src.utils import *
import torch
from keras.utils import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import os


# ref.: https://github.com/hate-alert/HateXplain
# This file contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators
# Few helper functions to convert attention vectors in 0 to 1 scale. While softmax converts all the values such that their sum lies between 0 --> 1. Sigmoid converts each value in the vector in the range 0 -> 1.

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def neg_softmax(x):
    """Compute softmax values for each sets of scores in x. Convert the exponentials to 1/exponentials"""
    e_x = np.exp(-(x - np.max(x)))
    return e_x / e_x.sum(axis=0)


def sigmoid(z):
    """Compute sigmoid values"""
    g = 1 / (1 + np.exp(-z))
    return g


# This function is used to aggregate the attentions vectors. This has a lot of options refer to the parameters explanation for understanding each parameter.
def aggregate_attention(at_mask, row, params):
    """input: attention vectors from 2/3 annotators (at_mask), row(dataframe row), params(parameters_dict)
       function: aggregate attention from different annotators.
       output: aggregated attention vector"""

    #### If the final label is normal or non-toxic then each value is represented by 1/len(sentences)
    if (row['final_label'] in ['normal', 'non-toxic']):
        at_mask_fin = [1 / len(at_mask[0]) for x in at_mask[0]]
    else:
        at_mask_fin = at_mask
        #### Else it will choose one of the options, where variance is added, mean is calculated, finally the vector is normalised.
        if (params['dataset']['type_attention'] == 'sigmoid'):
            at_mask_fin = int(params['dataset']['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = sigmoid(at_mask_fin)
        elif (params['dataset']['type_attention'] == 'softmax'):
            at_mask_fin = int(params['dataset']['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = softmax(at_mask_fin)
        elif (params['dataset']['type_attention'] == 'neg_softmax'):
            at_mask_fin = int(params['dataset']['variance']) * at_mask_fin
            at_mask_fin = np.mean(at_mask_fin, axis=0)
            at_mask_fin = neg_softmax(at_mask_fin)
        elif (params['dataset']['type_attention'] in ['raw', 'individual']):
            pass
    if params['dataset']['decay'] == True:
        at_mask_fin = decay(at_mask_fin, params)

    return at_mask_fin


##### Decay and distribution functions.To decay the attentions left and right of the attented word. This is done to decentralise the attention to a single word.
def distribute(old_distribution, new_distribution, index, left, right, params):
    window = params['dataset']['window']
    alpha = params['dataset']['alpha']
    p_value = params['dataset']['p_value']
    method = params['dataset']['method']

    reserve = alpha * old_distribution[index]
    #     old_distribution[index] = old_distribution[index] - reserve

    if method == 'additive':
        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

        for temp in range(index + 1, index + right):
            new_distribution[temp] = new_distribution[temp] + reserve / (left + right)

    if method == 'geometric':
        # First generate the geometric distributing for the left side
        temp_sum = 0.0
        newprob = []
        for temp in range(left):
            each_prob = p_value * ((1.0 - p_value) ** temp)
            newprob.append(each_prob)
            temp_sum += each_prob
            newprob = [each / temp_sum for each in newprob]

        for temp in range(index - left, index):
            new_distribution[temp] = new_distribution[temp] + reserve * newprob[-(temp - (index - left)) - 1]

        # do the same thing for right, but now the order is opposite
        temp_sum = 0.0
        newprob = []
        for temp in range(right):
            each_prob = p_value * ((1.0 - p_value) ** temp)
            newprob.append(each_prob)
            temp_sum += each_prob
            newprob = [each / temp_sum for each in newprob]
        for temp in range(index + 1, index + right):
            new_distribution[temp] = new_distribution[temp] + reserve * newprob[temp - (index + 1)]

    return new_distribution


def decay(old_distribution, params):
    window = params['dataset']['window']
    new_distribution = [0.0] * len(old_distribution)
    for index in range(len(old_distribution)):
        right = min(window, len(old_distribution) - index)
        left = min(window, index)
        new_distribution = distribute(old_distribution, new_distribution, index, left, right, params)

    if (params['dataset']['normalized']):
        norm_distribution = []
        for index in range(len(old_distribution)):
            norm_distribution.append(old_distribution[index] + new_distribution[index])
        tempsum = sum(norm_distribution)
        new_distrbution = [each / tempsum for each in norm_distribution]
    return new_distribution


# Text preprocessor for ekphrasis
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    # corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)


def custom_tokenize(sent, tokenizer, max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_sent = tokenizer.encode(
        sent,  # Sentence to encode.
        add_special_tokens=False)
    return encoded_sent


# input: text
# process: ekphrasis preprocesser + some extra processing
# output: list of tokens
def ek_extra_preprocess(text, params, tokenizer):
    remove_words = ['<allcaps>', '</allcaps>', '<hashtag>', '</hashtag>', '<elongated>', '<emphasis>', '<repeated>',
                    '\'', 's']
    word_list = text_processor.pre_process_doc(text)
    if params['dataset']['include_special']:
        pass
    else:
        word_list = list(filter(lambda a: a not in remove_words, word_list))
    sent = " ".join(word_list)
    sent = re.sub(r"[<\*>]", " ", sent)
    sub_word_list = custom_tokenize(sent, tokenizer)
    return sub_word_list


def returnMask(row, params, tokenizer):
    text_tokens = row['text']

    # Corner case
    if len(text_tokens) == 0:
        text_tokens = ['dummy']
    mask_all = row['rationales']
    mask_all_temp = mask_all
    while len(mask_all_temp) != 3:
        mask_all_temp.append([0] * len(text_tokens))

    word_mask_all = []
    word_tokens_all = []

    for mask in mask_all_temp:
        if mask[0] == -1:
            mask = [0] * len(mask)

        list_pos = []
        mask_pos = []

        flag = 0
        for i in range(0, len(mask)):
            if i == 0 and mask[i] == 0:
                list_pos.append(0)
                mask_pos.append(0)
            if flag == 0 and mask[i] == 1:
                mask_pos.append(1)
                list_pos.append(i)
                flag = 1

            elif flag == 1 and mask[i] == 0:
                flag = 0
                mask_pos.append(0)
                list_pos.append(i)
        if list_pos[-1] != len(mask):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts = []
        for i in range(len(list_pos) - 1):
            string_parts.append(text_tokens[list_pos[i]:list_pos[i + 1]])

        word_tokens = [101]
        word_mask = [0]
        for i in range(0, len(string_parts)):
            tokens = ek_extra_preprocess(" ".join(string_parts[i]), params, tokenizer)
            masks = [mask_pos[i]] * len(tokens)
            word_tokens += tokens
            word_mask += masks

        word_tokens = word_tokens[0:(int(params['dataset']['max_length']) - 2)]
        word_mask = word_mask[0:(int(params['dataset']['max_length']) - 2)]
        word_tokens.append(102)
        word_mask.append(0)
        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
    if len(mask_all) == 0:
        word_mask_all = []
    else:
        word_mask_all = word_mask_all[0:len(mask_all)]
    return word_tokens_all[0], word_mask_all


def set_name(params):
    if not params.get("output_dir", 0):
        default_output_dir = params["model"] + "_" + str(params["dataset"]['max_length'])
        data_params = params["dataset"]
        default_output_dir += '_' + data_params['type_attention'] + '_' + str(data_params['variance'])
        if data_params['decay']:
            default_output_dir += '_' + data_params['method'] + '_' + str(data_params['window']) + '_' + str(
                data_params['alpha']) + '_' + str(
                data_params['p_value'])
        loc_path = Path(__file__).parent
        params["output_dir"] = loc_path / default_output_dir
    file_name = str(params["output_dir"]) + "/data/"
    Path(file_name).mkdir(parents=True, exist_ok=True)
    file_name += 'tokenized_dataset.pickle'
    return file_name


def get_annotated_data(params):
    with open(params["dataset"]['data_file'], 'r') as fp:
        data = json.load(fp)
    dict_data = []
    for key in data:
        temp = {'post_id': key, 'text': data[key]['post_tokens']}
        final_label = []
        for i in range(1, 4):
            temp['annotatorid' + str(i)] = data[key]['annotators'][i - 1]['annotator_id']
            temp['target' + str(i)] = data[key]['annotators'][i - 1]['target']
            temp['label' + str(i)] = data[key]['annotators'][i - 1]['label']
            final_label.append(temp['label' + str(i)])

        final_label_id = max(final_label, key=final_label.count)
        temp['rationales'] = data[key]['rationales']

        if params['dataset']['class_names'] == 'Data/classes_two.npy':
            if final_label.count(final_label_id) == 1:
                temp['final_label'] = 'undecided'
            else:
                if final_label_id in ['hatespeech', 'offensive']:
                    final_label_id = 'toxic'
                else:
                    final_label_id = 'non-toxic'
                temp['final_label'] = final_label_id
        else:
            if final_label.count(final_label_id) == 1:
                temp['final_label'] = 'undecided'
            else:
                temp['final_label'] = final_label_id

        dict_data.append(temp)
    temp_read = pd.DataFrame(dict_data)
    return temp_read


def get_training_data(data, params, tokenizer):
    """
    input: data is a dataframe text ids attentions labels column only

    output: training data in the columns post_id,text, attention and labels """

    # majority = params['majority']
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    count = 0
    count_confused = 0
    for index, row in tqdm(data.iterrows(), total=len(data)):
        text = row['text']
        post_id = row['post_id']

        annotation_list = [row['label1'], row['label2'], row['label3']]
        annotation = row['final_label']

        if annotation != 'undecided':
            tokens_all, attention_masks = returnMask(row, params, tokenizer)
            attention_vector = aggregate_attention(attention_masks, row, params)
            attention_list.append(attention_vector)
            text_list.append(tokens_all)
            label_list.append(annotation)
            post_ids_list.append(post_id)
        else:
            count_confused += 1

    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label'])

    filename = set_name(params)
    training_data.to_pickle(filename)
    return training_data


# Data collection for test data
def get_test_data(tokenizer, data, params, message='text'):
    '''input: data is a dataframe text ids labels column only'''
    '''output: training data in the columns post_id,text (tokens) , attentions (normal) and labels'''
    post_ids_list = []
    text_list = []
    attention_list = []
    label_list = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        post_id = row['post_id']
        annotation = row['final_label']
        tokens_all, attention_masks = returnMask(row, params, tokenizer)
        attention_vector = aggregate_attention(attention_masks, row, params)
        attention_list.append(attention_vector)
        text_list.append(tokens_all)
        label_list.append(annotation)
        post_ids_list.append(post_id)

    # Calling DataFrame constructor after zipping
    # both lists, with columns specified
    training_data = pd.DataFrame(list(zip(post_ids_list, text_list, attention_list, label_list)),
                                 columns=['Post_id', 'Text', 'Attention', 'Label'])

    return training_data


def convert_data(test_data, params, list_dict, rational_present=True, topk=2):
    """this converts the data to be with or without the rationals based on the previous predictions"""
    """input: params -- input dict, list_dict -- previous predictions containing rationals
    rational_present -- whether to keep rational only or remove them only
    topk -- how many words to select"""

    temp_dict = {}
    for ele in list_dict:
        temp_dict[ele['annotation_id']] = ele['rationales'][0]['soft_rationale_predictions']

    test_data_modified = []

    for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
        try:
            attention = temp_dict[row['Post_id']]
        except KeyError:
            continue
        topk_indices = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        new_text = []
        new_attention = []
        if rational_present:
            new_attention = [0]
            new_text = [101]
            for i in range(len(row['Text'])):
                if (i in topk_indices):
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
            new_attention.append(0)
            new_text.append(102)
        else:
            for i in range(len(row['Text'])):
                if i not in topk_indices:
                    new_text.append(row['Text'][i])
                    new_attention.append(row['Attention'][i])
        test_data_modified.append([row['Post_id'], new_text, new_attention, row['Label']])

    df = pd.DataFrame(test_data_modified, columns=test_data.columns)
    return df


def transform_dummy_data(sentences):
    post_id_list = ['temp'] * len(sentences)
    pred_list = ['non-toxic'] * len(sentences)
    explanation_list = []
    sentences_list = []
    for i in range(len(sentences)):
        explanation_list.append([])
        sentences_list.append(sentences[i].split(" "))
    df = pd.DataFrame(list(zip(post_id_list, sentences_list, pred_list, pred_list,
                               pred_list, explanation_list, pred_list)),
                      columns=['post_id', 'text', 'label1', 'label2', 'label3', 'rationales', 'final_label'])

    return df


def collect_data(tokenizer, params):
    data_all_labelled = get_annotated_data(params)
    train_data = get_training_data(data_all_labelled, params, tokenizer)
    return train_data


def custom_att_masks(input_ids):
    attention_masks = []
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks


@conf("./params.yml", as_default=True)
def combine_features(tuple_data, is_train=False, return_loader=False, **params):
    input_ids = [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele[2] for ele in tuple_data]

    encoder = LabelEncoder()

    encoder.classes_ = np.load(params['dataset']['class_names'], allow_pickle=True)
    labels = encoder.transform(labels)

    input_ids = pad_sequences(input_ids, maxlen=int(params['dataset']['max_length']), dtype="long",
                              value=0, truncating="post", padding="post")
    att_vals = pad_sequences(att_vals, maxlen=int(params['dataset']['max_length']), dtype="float",
                             value=0.0, truncating="post", padding="post")
    att_masks = custom_att_masks(input_ids)
    if return_loader:
        dataloader = return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train)
        return dataloader

    dframe = pd.DataFrame.from_dict({"input_ids": list(input_ids), "attention": list(att_vals),
                                     "masks": list(att_masks), "labels": labels})
    return dframe


def return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(np.array(att_masks), dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals), dtype=torch.float)
    data = TensorDataset(inputs, attention, masks, labels)
    if is_train is False:
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['training']['batch_size'])
    return dataloader


def encodeData(dataframe):
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
    return tuple_new_data


@conf("./params.yml", as_default=True)
def createDatasetSplit(**params):
    tokenizer = AutoTokenizer.from_pretrained(
        params["model"],
    )
    params["dataset"]['data_file'] = dict_data_folder[str(params["dataset"]['num_classes'])]['data_file']
    params["dataset"]['class_names'] = dict_data_folder[str(params["dataset"]['num_classes'])]['class_label']
    filename = set_name(params)
    if not path.exists(filename):
        dataset = collect_data(tokenizer, params)

    if path.exists(filename[:-7]):
        with open(filename[:-7] + '/train_data.pickle', 'rb') as f:
            X_train = pickle.load(f)
        with open(filename[:-7] + '/val_data.pickle', 'rb') as f:
            X_val = pickle.load(f)
        with open(filename[:-7] + '/test_data.pickle', 'rb') as f:
            X_test = pickle.load(f)
    else:
        dataset = pd.read_pickle(filename)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)

        X_train = dataset[dataset['Post_id'].isin(post_id_dict['train'])]
        X_val = dataset[dataset['Post_id'].isin(post_id_dict['val'])]
        X_test = dataset[dataset['Post_id'].isin(post_id_dict['test'])]
        X_train = encodeData(X_train)
        X_val = encodeData(X_val)
        X_test = encodeData(X_test)
        os.mkdir(filename[:-7])
        with open(filename[:-7] + '/train_data.pickle', 'wb') as f:
            pickle.dump(X_train, f)
        with open(filename[:-7] + '/val_data.pickle', 'wb') as f:
            pickle.dump(X_val, f)
        with open(filename[:-7] + '/test_data.pickle', 'wb') as f:
            pickle.dump(X_test, f)
    return X_train, X_val, X_test
