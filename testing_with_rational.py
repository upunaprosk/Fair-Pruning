from transformers import *
#### common utils
from Models.utils import fix_the_random, format_time, get_gpu, return_params
#### metric utils 
from Models.utils import softmax,get_module_logger, NumpyEncoder
from Models.train import select_model
from Models.dataLoader import combine_features, createDatasetSplit, encodeData
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
from sklearn.utils import class_weight
import json
from Models.bertModels import *
from sklearn.preprocessing import LabelEncoder
from Models.dataCollect import get_test_data, convert_data, get_annotated_data
from tqdm import tqdm
import numpy as np
import argparse

dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}

def standaloneEval_with_rational(params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False):
    #     device = torch.device("cpu")
    if torch.cuda.is_available() and params['device'] == 'cuda':
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        logger.info('No GPU available, using CPU instead.')
        device = torch.device("cpu")

    transformer_type = params["path_files"]
    tokenizer = AutoTokenizer.from_pretrained(transformer_type)

    train, val, test = createDatasetSplit(tokenizer, params)
    if (params['auto_weights']):
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        params['weights'] = class_weight.compute_class_weight("balanced", classes=np.unique(y_test), y=y_test).astype('float32')
    if (extra_data_path != None):
        params_dash = {}
        params_dash['num_classes'] = 2
        params_dash['data_file'] = extra_data_path
        params_dash['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)
        temp_read = temp_read[
            temp_read['post_id'].isin(post_id_dict['test']) & (
                temp_read['final_label'].isin(['hatespeech', 'offensive', 'toxic']))]
        test_data = get_test_data(tokenizer, temp_read, params, message='text')
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    elif (use_ext_df):
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test_dataloader = combine_features(test, params, is_train=False)
    model = select_model(params)
    if (params["device"] == 'cuda'):
        model.cuda()
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
    attention_all = []
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
        # model.zero_grad()
        outputs = model(b_input_ids,
                        attention_vals=b_att_val,
                        attention_mask=b_input_mask,
                        labels=None, device=device)
        #         m = nn.Softmax(dim=1)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        last_layer = len(model.layer_list)
        attention_vectors = np.mean(outputs[1][last_layer-1][:, :, 0, :].detach().cpu().numpy(), axis=1)
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    if (use_ext_df == False):
        # logger.info(true_labels)
        testf1 = f1_score(true_labels, pred_labels, average='macro')
        testacc = accuracy_score(true_labels, pred_labels)
        testprecision = precision_score(true_labels, pred_labels, average='macro')
        testrecall = recall_score(true_labels, pred_labels, average='macro')

        # Report the final accuracy for this validation run.
        logger.info(" Accuracy: {0:.3f}".format(testacc))
        logger.info(" Fscore: {0:.3f}".format(testf1))
        logger.info(" Precision: {0:.3f}".format(testprecision))
        logger.info(" Recall: {0:.3f}".format(testrecall))
        # logger.info(" Roc Auc: {0:.3f}".format(testrocauc))
        logger.info(" Test took: {:}".format(format_time(time.time() - t0)))

    attention_vector_final = []
    for x, y in zip(attention_all, input_mask_all):
        temp = []
        for x_ele, y_ele in zip(x, y):
            if (y_ele == 1):
                temp.append(x_ele)
        attention_vector_final.append(temp)

    list_dict = []

    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final,
                                                              pred_labels, true_labels):
        temp = {}
        encoder = LabelEncoder()

        encoder.classes_ = np.load(dict_data_folder[str(params['num_classes'])]['class_label'], allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        if params["num_classes"] == 2:
            temp["classification_scores"] = {"non-toxic": logits[0], "toxic": logits[1]}
        else:
            temp["classification_scores"] = {"hatespeech": logits[0], "normal": logits[1], "offensive": logits[2]}
        topk_indicies = sorted(range(len(attention)), key=lambda i: attention[i])[-topk:]
        temp_hard_rationales = []
        for ind in topk_indicies:
            temp_hard_rationales.append({'end_token': ind + 1, 'start_token': ind})
        temp["rationales"] = [{"docid": post_id,
                               "hard_rationale_predictions": temp_hard_rationales,
                               "soft_rationale_predictions": attention,
                               # "soft_sentence_predictions":[1.0],
                               "truth": ground_truth}]
        list_dict.append(temp)

    return list_dict, test_data


def get_final_dict_with_rational(params, test_data=None, topk=5):
    list_dict_org, test_data = standaloneEval_with_rational(params, extra_data_path=test_data, topk=topk)
    test_data_with_rational = convert_data(test_data, params, list_dict_org, rational_present=True, topk=topk)
    list_dict_with_rational, _ = standaloneEval_with_rational(params, test_data=test_data_with_rational, topk=topk,
                                                              use_ext_df=True)
    test_data_without_rational = convert_data(test_data, params, list_dict_org, rational_present=False, topk=topk)
    list_dict_without_rational, _ = standaloneEval_with_rational(params, test_data=test_data_without_rational,
                                                                 topk=topk, use_ext_df=True)
    final_list_dict = []
    for ele1, ele2, ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
        ele1['sufficiency_classification_scores'] = ele2['classification_scores']
        ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
        final_list_dict.append(ele1)
    return final_list_dict


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
                           help='required to assign the contribution of the attention loss')

    args = my_parser.parse_args()
    logger = get_module_logger(__name__)
    model_to_use = args.model_to_use

    params = return_params(model_to_use, att_lambda=float(args.attention_lambda), num_classes=2)
    params['device'] = 'cpu'
    fix_the_random(seed_val=params['random_seed'])
    params['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']
    params['data_file'] = dict_data_folder[str(params['num_classes'])]['data_file']
    final_list_dict = get_final_dict_with_rational(params, params['data_file'], topk=5)
    path_name = model_to_use
    path_name_explanation = 'explanations_dicts/' + str(path_name) + '_' + str(
        params['att_lambda']) + '_explanation_top5.json'
    logger.info('Saving explanations to %s', path_name_explanation)
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))
