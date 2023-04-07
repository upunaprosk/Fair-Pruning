from src.data_load import *
from src.utils import *
from tqdm import tqdm
import more_itertools as mit


def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


# Convert dataset into ERASER format: https://github.com/jayded/eraserbenchmark/blob/master/rationale_benchmark/utils.py
def get_evidence(post_id, anno_text, explanations):
    output = []

    indexes = sorted([i for i, each in enumerate(explanations) if each == 1])
    span_list = list(find_ranges(indexes))

    for each in span_list:
        if type(each) == int:
            start = each
            end = each + 1
        elif len(each) == 2:
            start = each[0]
            end = each[1] + 1
        else:
            raise ValueError(f"Error in spans aligning in data provided. Post_id: {post_id}")

        output.append({"docid": post_id,
                       "end_sentence": -1,
                       "end_token": end,
                       "start_sentence": -1,
                       "start_token": start,
                       "text": ' '.join([str(x) for x in anno_text[start:end]])})
    return output


# To use the metrics defined in ERASER, we will have to convert the dataset
def convert_to_eraser_format(dataset, method, save_split, save_path, id_division):
    final_output = []
    save_path = str(save_path) + "/"
    if save_split:
        train_fp = open(save_path + 'train.jsonl', 'w')
        val_fp = open(save_path + 'val.jsonl', 'w')
        test_fp = open(save_path + 'test.jsonl', 'w')

    for tcount, eachrow in enumerate(dataset):

        temp = {}
        post_id = eachrow[0]
        post_class = eachrow[1]
        anno_text_list = eachrow[2]
        majority_label = eachrow[1]

        if majority_label == 'normal':
            continue

        all_labels = eachrow[4]
        explanations = []
        for each_explain in eachrow[3]:
            explanations.append(list(each_explain))

        if method == 'union':
            final_explanation = [any(each) for each in zip(*explanations)]
            final_explanation = [int(each) for each in final_explanation]

        temp['annotation_id'] = post_id
        temp['classification'] = post_class
        temp['evidences'] = [get_evidence(post_id, list(anno_text_list), final_explanation)]
        temp['query'] = "What is the class?"
        temp['query_type'] = None
        final_output.append(temp)

        if save_split:
            if not os.path.exists(save_path + 'docs'):
                os.makedirs(save_path + 'docs')

            with open(save_path + 'docs/' + post_id, 'w') as fp:
                fp.write(' '.join([str(x) for x in list(anno_text_list)]))

            if post_id in id_division['train']:
                train_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['val']:
                val_fp.write(json.dumps(temp) + '\n')

            elif post_id in id_division['test']:
                test_fp.write(json.dumps(temp) + '\n')
    if save_split:
        train_fp.close()
        val_fp.close()
        test_fp.close()
    return final_output


@conf("params.yml", as_default=True)
def generate_eraser_input(**params):
    log_level = logging.DEBUG if params["logging"] == "debug" else logging.INFO
    logger = set_logger(level=log_level)
    data_params = params["dataset"]
    tokenizer = AutoTokenizer.from_pretrained(
        params["model"],
    )

    def get_training_data(data):

        final_binny_output = []
        for index, row in tqdm(data.iterrows(), total=len(data)):
            annotation = row['final_label']
            post_id = row['post_id']
            annotation_list = [row['label1'], row['label2'], row['label3']]
            if annotation != 'undecided':
                tokens_all, attention_masks = returnMask(row, params, tokenizer)
                final_binny_output.append([post_id, annotation, tokens_all, attention_masks, annotation_list])
        return final_binny_output

    params_dash = dict()
    params_dash["dataset"] = dict()
    params_dash["dataset"]['data_file'] = dict_data_folder[str(data_params['num_classes'])]['data_file']
    params_dash["dataset"]['class_names'] = dict_data_folder[str(data_params['num_classes'])]['class_label']
    data_all_labelled = get_annotated_data(params_dash)
    training_data = get_training_data(data_all_labelled)
    # The post_id_divisions file stores the train, val, test split ids. We select only the test ids.
    with open('./Data/post_id_divisions.json') as fp:
        id_division = json.load(fp)
    method = 'union'
    save_split = True
    save_path = './Data/Evaluation/'
    eraser_loc_path = Path(save_path)
    eraser_loc_path.mkdir(parents=True, exist_ok=True)
    eraser_loc_path = eraser_loc_path / params['model']
    eraser_loc_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"ERASER files would be saved to {str(eraser_loc_path)}")
    logger.info(f"Generating data for ERASER for the tokenizer type: {params['model']} with method: {method}")
    convert_to_eraser_format(training_data, method, save_split, eraser_loc_path, id_division)
    logger.info(f"ERASER files are saved to {str(eraser_loc_path)}")
    return


if __name__ == '__main__':
    list_dict_org = generate_eraser_input()
