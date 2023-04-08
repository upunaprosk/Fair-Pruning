from sklearn.metrics import roc_auc_score
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    set_seed,
)
from src.data_load import *
from transformers import TextClassificationPipeline
from collections import Counter, defaultdict


def generate_target_information(dataset):
    final_target_output = defaultdict(list)
    all_communities_selected = []

    for each in dataset.iterrows():
        # All the target communities tagged for this post
        all_targets = each[1]['target1'] + each[1]['target2'] + each[1]['target3']
        community_dict = dict(Counter(all_targets))
        # Select only those communities which are present more than once.
        for key in community_dict:
            if community_dict[key] > 1:
                final_target_output[each[1]['post_id']].append(key)
                all_communities_selected.append(key)
        # If no community is selected based on majority voting then we don't select any community
        if each[1]['post_id'] not in final_target_output:
            final_target_output[each[1]['post_id']].append('None')
    return final_target_output, all_communities_selected


# Function to divide the ids into positive or class class based on the method.
def bias_evaluation_metric(dataset, method, community):
    positive_ids = []
    negative_ids = []
    if method == 'subgroup':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] is None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] == 'non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    positive_ids.append(eachrow[1]['post_id'])
            else:
                pass
    elif method == 'bpsn':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] == 'non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] != 'non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
    elif method == 'bnsp':
        for eachrow in dataset.iterrows():
            if eachrow[1]['final_target_category'] == None:
                continue
            if community in eachrow[1]['final_target_category']:
                if eachrow[1]['final_label'] != 'non-toxic':
                    positive_ids.append(eachrow[1]['post_id'])
                else:
                    pass
            else:
                if eachrow[1]['final_label'] == 'non-toxic':
                    negative_ids.append(eachrow[1]['post_id'])
                else:
                    pass

    return {'positiveID': positive_ids, 'negativeID': negative_ids}


def convert_to_score(label_name, label_dict):
    if label_name == 'non-toxic':
        return 1 - label_dict[label_name]
    else:
        return label_dict[label_name]


@conf("./params.yml", as_default=True)
def standaloneEval(**params):
    log_level = logging.DEBUG if params["logging"] == "debug" else logging.INFO
    logger = set_logger(level=log_level)
    set_seed(params["seed"])
    params = set_output_dir(**params)
    logger.debug("Using parameters")
    logger.debug(params)
    params_dash = {}
    data_params = params["dataset"]
    params_dash['num_classes'] = data_params["num_classes"]
    params_dash["dataset"] = dict()
    params_dash["dataset"]['data_file'] = dict_data_folder[str(data_params['num_classes'])]['data_file']
    params_dash["dataset"]['class_names'] = dict_data_folder[str(data_params['num_classes'])]['class_label']
    temp_read = get_annotated_data(params_dash)
    with open('./Data/post_id_divisions.json', 'r') as fp:
        post_id_dict = json.load(fp)
    post_id_all = temp_read[temp_read['post_id'].isin(post_id_dict['test'])]["post_id"].values
    test_sents = [" ".join(word_list) for word_list in
                  temp_read[temp_read['post_id'].isin(post_id_dict['test'])]['text'].values]
    true_labels = temp_read[temp_read['post_id'].isin(post_id_dict['test'])]["final_label"].values
    logger.debug(f"Loading model from {str(params['output_dir'])}")
    model = AutoModelForSequenceClassification.from_pretrained(
        params["output_dir"]
    )
    logger.info(f"Model loaded {str(params['output_dir'])}")
    logger.debug(f"Example of testing sentence: {test_sents[0]}")
    tokenizer = AutoTokenizer.from_pretrained(params["model"])
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    predictions = pipe(test_sents, return_all_scores=True)
    logger.debug("Hatespeech (0), Normal (1) or Offensive (2)")
    logger.debug(f"Output predictions for the example: {str(predictions[0][0])}")
    list_dict = []
    logger.warn("For calculating bias toxic/non-toxic binary scheme is used.")
    for post_id, preds, ground_truth in zip(post_id_all, predictions, true_labels):
        logits = [i['score'] for i in preds]
        pred = np.argmax(logits)
        temp = dict()
        temp["annotation_id"] = post_id
        temp["classification"] = "toxic" if pred != 1 else "non-toxic"
        temp["ground_truth"] = "non-toxic" if ground_truth == "normal" else "toxic"
        temp["classification_scores"] = {"non-toxic": logits[1], "toxic": logits[0] + logits[2]}
        list_dict.append(temp)
    attention_info = "attn_" + str(params["training"]['att_lambda']) if params["training"]['train_att'] else ""
    biased_path = Path(params["output_dir"]) / 'explanations_dicts/'
    biased_path.mkdir(parents=True, exist_ok=True)
    path_name_explanation = str(biased_path) + "/" + attention_info + '_bias.json'
    with open(path_name_explanation, 'w') as fp:
        fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder) for i in list_dict))
    logger.info(f"Predictions saved to {path_name_explanation}")
    target_information, all_communities_selected = generate_target_information(temp_read)
    community_count_dict = Counter(all_communities_selected)
    # Remove None and Other from dictionary
    community_count_dict.pop('None')
    community_count_dict.pop('Other')
    # For the bias calculation, we are considering the top 10 communites based on their count
    list_selected_community = [community for community, value in community_count_dict.most_common(10)]
    deb_msg = ', '.join(list_selected_community)
    logger.debug(f"Calculating bias for the most frequent subgroups: {deb_msg}")
    final_target_information = {}
    for each in target_information:
        temp = list(set(target_information[each]))
        if len(temp) == 0:
            final_target_information[each] = None
        else:
            final_target_information[each] = temp
    method_list = ['subgroup', 'bpsn', 'bnsp']

    community_list = list(list_selected_community)

    final_bias_dictionary = defaultdict(lambda: defaultdict(dict))
    total_data = {}
    for i in list_dict:
        total_data[i['annotation_id']] = i

    temp_read['final_target_category'] = temp_read['post_id'].map(final_target_information)
    data_all_labelled_bias = temp_read[temp_read['post_id'].isin(post_id_dict['test'])]
    logger.info(f"Calculating bias for the most frequent subgroups...")
    for each_method in method_list:
        for each_community in community_list:
            community_data = bias_evaluation_metric(data_all_labelled_bias, each_method, each_community)
            truth_values = []
            prediction_values = []
            label_to_value = {'toxic': 1.0, 'non-toxic': 0.0}
            for each in community_data['positiveID']:
                truth_values.append(label_to_value[total_data[each]['ground_truth']])
                prediction_values.append(
                    convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

            for each in community_data['negativeID']:
                truth_values.append(label_to_value[total_data[each]['ground_truth']])
                prediction_values.append(
                    convert_to_score(total_data[each]['classification'], total_data[each]['classification_scores']))

            roc_output_value = roc_auc_score(truth_values, prediction_values)
            final_bias_dictionary[each_method][each_community] = roc_output_value
    scores_path = Path(str(biased_path) + '/bias_per_community.json')
    json_str = json.dumps(final_bias_dictionary, indent=4) + '\n'
    scores_path.write_text(json_str, encoding='utf-8')
    logger.info(f"Saved bias scores per hate communtiy to: {str(scores_path)}")
    power_value = -5
    num_communities = len(community_list)
    all_results = dict()
    for each_method in final_bias_dictionary:
        temp_value = []
        for each_community in final_bias_dictionary[each_method]:
            temp_value.append(pow(final_bias_dictionary[each_method][each_community], power_value))
        score_i = pow(np.sum(temp_value) / num_communities, 1 / power_value)
        logger.info(f"Bias scores calculated with {each_method}: {str(score_i)}")
        all_results[each_method] = score_i
        if params["training"]["report_to"] == "wandb":
            wandb.log({each_method: score_i})
    scores_path = Path(str(biased_path) + '/bias_scores.json')
    json_str = json.dumps(all_results, indent=4) + '\n'
    scores_path.write_text(json_str, encoding='utf-8')
    logger.info(f"Saved aggregated bias scores to: {str(scores_path)}")
    return


if __name__ == '__main__':
    list_dict_org = standaloneEval()
