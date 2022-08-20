from transformers import *
from Models.train import select_model
#### common utils
from Models.utils import fix_the_random, format_time
#### metric utils 
from Models.utils import softmax, return_params,get_module_logger,NumpyEncoder
from Models.dataLoader import createDatasetSplit, combine_features, encodeData
import time
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from sklearn.preprocessing import LabelEncoder
from Models.dataCollect import get_test_data, get_annotated_data
from tqdm import tqdm
import numpy as np
import argparse
import logging

dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}


def standaloneEval(params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False):
    device = torch.device("cpu")
    transformer_type = params["path_files"]
    tokenizer = AutoTokenizer.from_pretrained(transformer_type)
    train, val, test = createDatasetSplit(tokenizer, params)
    if (params['auto_weights']):
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes.npy', allow_pickle=True)
        params['weights'] = class_weight.compute_class_weight("balanced", classes=np.unique(y_test), y=y_test).astype(
            'float32')
    if (extra_data_path != None):
        params_dash = {}
        params_dash['num_classes'] = params["num_classes"]
        params_dash['data_file'] = extra_data_path
        params_dash['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)
        temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test'])]
        test_data = get_test_data(tokenizer, temp_read, params, message='text')
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    elif (use_ext_df):
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test_dataloader = combine_features(test, params, is_train=False)

    model = select_model(params)
    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    if ((extra_data_path != None) or (use_ext_df == True)):
        post_id_all = list(test_data['Post_id'])
    else:
        post_id_all = list(test['Post_id'])
    logger.info("Running eval on test data...")
    t0 = time.time()
    true_labels = []
    pred_labels = []
    logits_all = []
    input_mask_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels 
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        outputs = model(b_input_ids,
                        attention_vals=b_att_val,
                        attention_mask=b_input_mask,
                        labels=None, device=device)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        input_mask_all += list(batch[2].detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    list_dict = []
    for post_id, logits, pred, ground_truth in zip(post_id_all, logits_all_final, pred_labels, true_labels):
        temp = {}
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        ground_label = encoder.inverse_transform([ground_truth])[0]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        temp["ground_truth"] = ground_label
        temp["classification_scores"] = {"non-toxic": logits[0], "toxic": logits[1]}
        list_dict.append(temp)

    return list_dict, test_data


def get_final_dict(params, test_data, topk):
    list_dict_org, test_data = standaloneEval(params, extra_data_path=test_data, topk=topk)
    return list_dict_org



if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Which model to use')

    # Add the arguments
    my_parser.add_argument('model_to_use',
                           metavar='--model_to_use',
                           type=str,
                           help='model to use for evaluation')

    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the atention loss')

    args = my_parser.parse_args()
    logger = get_module_logger(__name__)
    model_to_use = args.model_to_use
    params = return_params(model_to_use, float(args.attention_lambda), 2)
    params['variance'] = 1
    params['num_classes'] = 2
    fix_the_random(seed_val=params['random_seed'])
    params['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']
    params['data_file'] = dict_data_folder[str(params['num_classes'])]['data_file']

    attention_info = "_Attn_" + params['att_lambda'] if params['train_att'] else ""
    path_name_explanation = 'explanations_dicts/' + model_to_use + attention_info + '_bias.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in [""]))

    final_dict = get_final_dict(params, params['data_file'], topk=5)
    path_name = model_to_use
    attention_info = "_Attn_" + params['att_lambda'] if params['train_att'] else ""
    path_name_explanation = 'explanations_dicts/' + str(path_name) + attention_info + '_bias.json'
    logger.info('Saving explanations to %s', path_name_explanation)
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_dict))