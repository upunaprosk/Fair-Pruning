### This is run when you want to select the parameters from the parameters file
from transformers import *
from transformers import AdamW
from Models.utils import fix_the_random, format_time, save_bert_model, get_module_logger
from tqdm import tqdm
from Models.dataLoader import combine_features, createDatasetSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
import GPUtil
from sklearn.utils import class_weight
import json
from Models.bertModels import *
import time
from sklearn.preprocessing import LabelEncoder
import numpy as np
import argparse
import ast
import wandb
from transformers import (
    AutoTokenizer,
)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


### gpu selection algo
def get_gpu():
    logger.info('There are %d GPU(s) available.' % torch.cuda.device_count())
    while (1):
        tempID = []
        tempID = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.1, maxMemory=0.07, includeNan=False,
                                     excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            logger.info("Found a gpu")
            logger.info('We will use the GPU:'+ str(tempID[0]) + str(torch.cuda.get_device_name(tempID[0])))
            deviceID = tempID
            return deviceID
        else:
            time.sleep(5)
   # return flag,deviceID

##### selects the type of model
def select_model(params):
    n_classes = params["num_classes"]
    model_path = params["model"]
    if params["path_files"] != "N/A":
        model_path = params["path_files"]
    model = init_model(params).from_pretrained(
        model_path,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=n_classes,  # The number of output labels
        output_attentions=True,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
        hidden_dropout_prob=params['dropout_bert'] if params['dropout_bert'] else 0.1,
        params=params
    )
    logger.info("=======Initialized model=======")
    logger.info(model.config)
    return model


def Eval_phase(params, which_files='test', model=None, test_dataloader=None, device=None):
    if (params['is_model'] == True):
        logger.info("model previously passed")
        model.eval()
    else:
        return 1

    print("Running eval on ", which_files ,"...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables 

    true_labels = []
    pred_labels = []
    logits_all = []
    probs_all = []
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

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
        probs_step = torch.nn.functional.softmax(logits, dim=-1)
        probs_all.extend(probs_step[:, 1].tolist())
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    testf1 = f1_score(true_labels, pred_labels, average='macro')
    testacc = accuracy_score(true_labels, pred_labels)
    if (params['num_classes'] == 3):
        testrocauc = roc_auc_score(true_labels, logits_all_final, multi_class='ovo', average='macro')
    else:
        testrocauc = roc_auc_score(true_labels, probs_all)
    testprecision = precision_score(true_labels, pred_labels, average='macro')
    testrecall = recall_score(true_labels, pred_labels, average='macro')
    logger.info(" Accuracy: {0:.3f}".format(testacc))
    logger.info(" Fscore: {0:.3f}".format(testf1))
    logger.info(" Precision: {0:.3f}".format(testprecision))
    logger.info(" Recall: {0:.3f}".format(testrecall))
    logger.info(" Roc Auc: {0:.3f}".format(testrocauc))
    logger.info(" Test took: {:}".format(format_time(time.time() - t0)))

    return testf1, testacc, testprecision, testrecall, testrocauc, logits_all_final


def train_model(params, device):
    embeddings = None
    transformer_type = params["model"]
    tokenizer = AutoTokenizer.from_pretrained(transformer_type)

    train, val, test = createDatasetSplit(tokenizer, params)
    if (params['auto_weights']):
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
        params['weights'] = class_weight.compute_class_weight("balanced", classes=np.unique(y_test), y=y_test).astype(
            'float32')

    model = select_model(params)
    print("Class weights: ", params['class_names'], params['weights'])
    train_dataloader = combine_features(train, params, is_train=True)
    validation_dataloader = combine_features(val, params, is_train=False)
    test_dataloader = combine_features(test, params, is_train=False)
    if (params['logging'] == 'wandb'):
        wandb.log({"n_layers": len(model.layer_list),
                   "learning_rate":params['learning_rate'],
                   "epsilon":params['epsilon']})
        if model.remove_layers:
            wandb.log({"removed_layers": model.remove_layers})
    if (params["device"] == 'cuda'):
        model.cuda()
    optimizer = AdamW(model.parameters(),
                      lr=params['learning_rate'],  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=params['epsilon']  # args.adam_epsilon  - default is 1e-8.
                      )

    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params['epochs']

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps / 10),
                                                num_training_steps=total_steps)

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val=params['random_seed'])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    name_one = params['path_files']
    best_train_acc = 0
    best_test_acc = 0
    best_val_acc = 0

    best_val_fscore = 0
    best_test_fscore = 0
    best_train_fscore = 0

    best_val_roc_auc = 0
    best_train_roc_auc = 0
    best_test_roc_auc = 0

    best_val_precision = 0
    best_train_precision = 0
    best_test_precision = 0

    best_val_recall = 0
    best_train_recall = 0
    best_test_recall = 0

    for epoch_i in range(0, params['epochs']):
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, params['epochs']))
        logger.info('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

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
                            labels=b_labels,
                            device=device)

            # The call to `model` always returns a tuple, so we need to pull the 
            # loss value out of the tuple.

            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        if (params['logging'] == 'wandb'):
            wandb.log({"avg_train_loss": avg_train_loss})
        logger.info('avg_train_loss: {0:.3f}'.format(avg_train_loss))
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        train_fscore, train_accuracy, train_precision, train_recall, train_roc_auc, _ = Eval_phase(params, 'train',
                                                                                                   model,
                                                                                                   train_dataloader,
                                                                                                   device)
        val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, _ = Eval_phase(params, 'val', model,
                                                                                         validation_dataloader, device)
        test_fscore, test_accuracy, test_precision, test_recall, test_roc_auc, logits_all_final = Eval_phase(params,
                                                                                                             'test',
                                                                                                             model,
                                                                                                             test_dataloader,
                                                                                                             device)
        if (val_fscore > best_val_fscore):
            best_train_acc = train_accuracy
            best_test_acc = test_accuracy
            best_val_acc = val_accuracy

            best_train_fscore = train_fscore
            best_train_roc_auc = train_roc_auc
            best_train_precision = train_precision
            best_train_recall = train_recall

            best_val_fscore = val_fscore
            best_test_fscore = test_fscore
            best_val_roc_auc = val_roc_auc
            best_test_roc_auc = test_roc_auc

            best_val_precision = val_precision
            best_test_precision = test_precision
            best_val_recall = val_recall
            best_test_recall = test_recall

            save_bert_model(model, tokenizer, params)

    if params['logging'] == 'wandb':
        train_fscore, train_accuracy, train_precision, train_recall, train_roc_auc
        wandb.log({"val_fscore": best_val_fscore,
                    "test_fscore": best_test_fscore,
                    "val_rocauc": best_val_roc_auc,
                    "test_rocauc": best_test_roc_auc,
                    "val_precision": best_val_precision,
                   "test_precision": best_test_precision,
                   "val_recall": best_val_recall,
                   "test_recall": best_test_recall,
                   "test_recall": best_test_recall,
                   "train_fscore": best_train_fscore,
                   "train_accuracy": best_train_acc,
                   "train_precision":best_train_precision,
                   "train_recall": best_train_recall,
                   "train_rocauc": best_train_roc_auc,
                   "val_accuracy":  best_val_acc,
                   "test_accuracy": best_test_acc
                   })

    logger.info('best_val_fscore: {0:.3f}'.format(best_val_fscore))
    logger.info('best_test_fscore: {0:.3f}'.format(best_test_fscore))
    logger.info('best_val_rocauc: {0:.3f}'.format(best_val_roc_auc))
    logger.info('best_test_rocauc: {0:.3f}'.format(best_test_roc_auc))
    logger.info('best_val_precision: {0:.3f}'.format(best_val_precision))
    logger.info('best_test_precision: {0:.3f}'.format(best_test_precision))
    logger.info('best_val_recall: {0:.3f}'.format(best_val_recall))
    logger.info('best_test_recall: {0:.3f}'.format(best_test_recall))

    del model
    torch.cuda.empty_cache()
    return 1


params_data = {
    'include_special': False,
    'type_attention': 'softmax',
    'set_decay': 0.1,
    'majority': 2,
    'max_length': 128,
    'variance': 10,
    'window': 4,
    'alpha': 0.5,
    'p_value': 0.8,
    'method': 'additive',
    'decay': False,
    'normalized': False,
    'not_recollect': True,
}


common_hp = {
    'is_model': True,
    'logging': 'local',  ###wandb /local
    'learning_rate': 0.1,  ### learning rate 2e-5 for bert 0.001 for gru
    'epsilon': 1e-8,
    'batch_size': 16,
    'to_save': True,
    'epochs': 10,
    'auto_weights': True,
    'weights': [1.0, 1.0, 1.0],
    'model_name': 'birnnscrat',
    'random_seed': 42,
    'num_classes': 3,
    'att_lambda': 100,
    'device': 'cuda',
    'train_att': True

}

params_bert = {
    'path_files': 'bert-base-uncased',
    'what_bert': 'weighted',
    'save_only_bert': False,
    'supervised_layer_pos': 11,
    'num_supervised_heads': 1,
    'dropout_bert': 0,
    'freeze_embeddings': False,
    'remove_layers': ''
}

params_other = {
    "vocab_size": 0,
    "padding_idx": 0,
    "hidden_size": 64,
    "embed_size": 0,
    "embeddings": None,
    "drop_fc": 0.2,
    "drop_embed": 0.2,
    "drop_hidden": 0.1,
    "train_embed": False,
    "seq_model": "gru",
    "attention": "softmax"
}

for key in params_other:
    params_other[key] = 'N/A'

def Merge(dict1, dict2, dict3, dict4):
    res = {**dict1, **dict2, **dict3, **dict4}
    return res


params = Merge(params_data, common_hp, params_bert, params_other)

dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='Train a deep-learning model with the given data')

    # Add the arguments
    my_parser.add_argument('path',
                           metavar='--path_to_json',
                           type=str,
                           help='The path to json containining the parameters')

    my_parser.add_argument('use_from_file',
                           metavar='--use_from_file',
                           type=str,
                           help='whether use the parameters present here or directly use from file')

    my_parser.add_argument('attention_lambda',
                           metavar='--attention_lambda',
                           type=str,
                           help='required to assign the contribution of the atention loss')

    my_parser.add_argument('model',
                           metavar='--model',
                           type=str,
                           help="Transformer model, ex.: 'bert-base-uncased'")
    my_parser.add_argument('remove_layers',
                           metavar='--remove_layers',
                           type=str,
                           default='',
                           help="specify layer numbers to remove during finetuning e.g. 0,1,2 to remove first three layers")
    my_parser.add_argument('freeze_embeddings',
                           metavar='--freeze_embeddings',
                           type=bool,
                           default=False,
                           help="flag to freeze embeddings")
    args = my_parser.parse_args()
    logger = get_module_logger(__name__)
    params['best_params'] = False
    if (args.use_from_file == 'True'):
        with open(args.path, mode='r') as f:
            params = json.load(f)
        for key in params:
            if params[key] == 'True':
                params[key] = True
            elif params[key] == 'False':
                params[key] = False
            if (key in ['batch_size', 'num_classes', 'hidden_size', 'supervised_layer_pos', 'num_supervised_heads',
                        'random_seed', 'max_length']):
                if (params[key] != 'N/A'):
                    params[key] = int(params[key])

            if ((key == 'weights') and (params['auto_weights'] == False)):
                params[key] = ast.literal_eval(params[key])
        params['best_params'] = True
        ##### change in logging to output the results to wandb
    params['logging'] = 'local'
    # if(params['logging']=='wandb'):
    #     from api_config import project_name,api_token
    #     wandb.init(project_name,api_token=api_token)
    #     wandb.set_project(project_name)
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available() and params['device'] == 'cuda':
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        ##### You can set the device manually if you have only one gpu
        ##### comment this line if you don't want to manually set the gpu
        deviceID = get_gpu()
        torch.cuda.set_device(deviceID[0])
        ##### comment this line if you don't want to manually set the gpu
        #### parameter required is the gpu id
        # torch.cuda.set_device(0)

    else:
        logger.info('No GPU available, using CPU instead.')
        device = torch.device("cpu")

    #### Few handy keys that you can directly change.
    params['variance'] = 1
    params['epochs'] = 5
    params['to_save'] = True
    # params['num_classes']=2 #or 3 classes: hateful, offensive, neutral
    params['data_file'] = dict_data_folder[str(params['num_classes'])]['data_file']
    params['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']

    if (params['num_classes'] == 2 and (params['auto_weights'] == False)):
        params['weights'] = [1.0, 1.0]

    # for att_lambda in [0.001,0.01,0.1,1,10,100]
    params['att_lambda'] = float(args.attention_lambda)
    train_model(params, device)
