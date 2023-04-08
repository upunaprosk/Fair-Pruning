from src.data_load import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import time
import json
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def encodeData(dataframe):
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
    return tuple_new_data


def standaloneEval_with_rational(params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False):
    logger = logging.getLogger()

    params['device'] = "cpu"
    logger.info('Using CPU...')
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(params["model"])
    train, val, test = createDatasetSplit(**params)
    test_dataloader = None

    params = set_output_dir(**params)
    model = AutoModelForSequenceClassification.from_pretrained(str(params["output_dir"]), output_attentions=True)
    model.eval()
    if extra_data_path is not None:
        params_dash = {}
        data_params = params["dataset"]
        params_dash['num_classes'] = data_params["num_classes"]
        params_dash["dataset"] = dict()
        params_dash["dataset"]['data_file'] = dict_data_folder[str(data_params['num_classes'])]['data_file']
        params_dash["dataset"]['class_names'] = dict_data_folder[str(data_params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)
        with open('./Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)
        temp_read = temp_read[
            temp_read['post_id'].isin(post_id_dict['test']) & (
                temp_read['final_label'].isin(['hatespeech', 'offensive', 'toxic']))]
        test_data = get_test_data(tokenizer, temp_read, params, message='text')
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, **params, is_train=False, return_loader=True)
    elif use_ext_df:
        test_extra = encodeData(test_data)
        test_dataloader = combine_features(test_extra, **params, is_train=False, return_loader=True)
    else:
        test_dataloader = combine_features(test, **params, is_train=False, return_loader=True)

    if (extra_data_path is not None) or (use_ext_df is True):
        post_id_all = list(test_data['Post_id'])
    else:
        with open('./Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)
        post_id_all = post_id_dict['test']

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
        outputs = model(b_input_ids,
                        attention_mask=b_input_mask, labels=None)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()
        last_layer = model.config.num_hidden_layers
        attention_vectors = np.mean(outputs[1][last_layer - 1][:, :, 0, :].detach().cpu().numpy(), axis=1)
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

    if use_ext_df == False:
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
            if y_ele == 1:
                temp.append(x_ele)
        attention_vector_final.append(temp)

    list_dict = []

    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final,
                                                              pred_labels, true_labels):
        temp = {}
        encoder = LabelEncoder()

        encoder.classes_ = np.load(dict_data_folder[str(params["dataset"]['num_classes'])]['class_label'],
                                   allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        if params["dataset"]["num_classes"] == 2:
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


@conf("./params.yml", as_default=True)
def get_final_dict_with_rational(topk=5, **params):
    log_level = logging.DEBUG if params["logging"] == "debug" else logging.INFO
    logger = set_logger(level=log_level)

    list_dict_org, test_data = standaloneEval_with_rational(params, extra_data_path=params['dataset']['data_file'],
                                                            topk=topk)
    print(test_data)
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

    rationales_path = Path(params["output_dir"]) / 'explanations_dicts/'
    rationales_path.mkdir(parents=True, exist_ok=True)
    attention_learning = "attn_" + str(params["training"]['att_lambda']) if params["training"]['train_att'] else ""
    if attention_learning:
        attention_learning += "_"
    rationales_path = str(rationales_path) + "/" + attention_learning + 'explanation_top5.json'
    logger.debug(f'Saving explanations to {rationales_path}')
    with open(rationales_path, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in final_list_dict))
    logger.info(f'Explanations saved to {rationales_path}')
    return


if __name__ == '__main__':
    get_final_dict_with_rational()
