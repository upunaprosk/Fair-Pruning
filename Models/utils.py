import time
import datetime
import numpy as np
from sklearn.metrics import f1_score
import random
from sklearn.preprocessing import LabelEncoder
import os
import torch
import torch.nn as nn
import GPUtil
import json
import ast
import logging

from collections import Counter
from itertools import chain, repeat, islice


# PreProcessing Utils
def most_frequent(List):
    counter = Counter(List)
    max_freq = max(set(List), key=List.count)
    return max_freq, counter[max_freq]


def CheckForGreater(list1, val):
    return all(x > val for x in list1)


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


# Common Utils
def get_gpu(params):
    if (params['bert_tokens'] == True):
        load_allowed = 0.07
    else:
        load_allowed = 0.5

    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    while (1):
        tempID = []
        tempID = GPUtil.getAvailable(order='memory', limit=1, maxLoad=load_allowed, maxMemory=load_allowed,
                                     includeNan=False, excludeID=[], excludeUUID=[])
        if len(tempID) > 0:
            print("Found a gpu")
            print('We will use the GPU:', tempID[0], torch.cuda.get_device_name(tempID[0]))
            deviceID = tempID
            return deviceID
        else:
            time.sleep(5)


def fix_the_random(seed_val=42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


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


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}


def return_params(path_name, att_lambda, num_classes=2):
    with open("./Saved/" + path_name + '/params.json', mode='r') as f:
        params = json.load(f)
    for key in params:
        if not params[key]:
            params[key] = 'False'
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
    params['att_lambda'] = att_lambda
    if (params['att_lambda'] >= 1):
        params['att_lambda'] = int(params['att_lambda'])
    params['num_classes'] = num_classes
    params['path_files'] = "./Saved/" + path_name
    params['data_file'] = dict_data_folder[str(params['num_classes'])]['data_file']
    params['class_names'] = dict_data_folder[str(params['num_classes'])]['class_label']
    if params['num_classes'] == 2 and (params['auto_weights'] == False):
        params['weights'] = [1.0, 1.0]

    return params


########################################### EXTRA METRICS CALCULATOR

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    temp = e_x / e_x.sum(axis=0)  # only difference

    if np.isnan(temp).any() == True:
        return [0.0, 1.0, 0.0]
    else:
        return temp


def cross_entropy(input1, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * logsoftmax(input1))
    # if size_average:
    #     return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    # else:
    #     return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def masked_cross_entropy(input1, target, mask):
    cr_ent = 0
    for h in range(0, mask.shape[0]):
        cr_ent += cross_entropy(input1[h][mask[h]], target[h][mask[h]])

    return cr_ent / mask.shape[0]


###########################################   MODEL LOADING, SAVING AND SELECTION FUNCtIONS


#### load normal model (bert model is directly loaded using the pretrained method)
def load_model(model, params, use_cuda=False):
    if (params['train_att'] == True):

        if (params['att_lambda'] >= 1):
            params['att_lambda'] = int(params['att_lambda'])
        model_path = 'Saved/' + params['model_name'] + '_' + params['seq_model'] + '_' + str(
            params['hidden_size']) + '_' + str(params['num_classes']) + '_' + str(params['att_lambda']) + '.pth'
    else:
        model_path = 'Saved/' + params['model_name'] + '_' + params['seq_model'] + '_' + str(
            params['hidden_size']) + '_' + str(params['num_classes']) + '.pth'
    print(model_path)
    """Load model."""
    map_location = 'cpu'
    #     if use_cuda and torch.cuda.is_available():
    # map_location = 'cuda'
    model.load_state_dict(torch.load(model_path, map_location))
    return model


def save_normal_model(model, params):
    """Save model."""
    if params['train_att']:
        if params['att_lambda'] >= 1:
            params['att_lambda'] = int(params['att_lambda'])

        model_path = 'Saved/' + params['model_name'] + '_' + params['seq_model'] + '_' + str(
            params['hidden_size']) + '_' + str(params['num_classes']) + '_' + str(params['att_lambda']) + '.pth'
    else:
        model_path = 'Saved/' + params['model_name'] + '_' + params['seq_model'] + '_' + str(
            params['hidden_size']) + '_' + str(params['num_classes']) + '.pth'

    print("Saving model to %s" % model_path)
    torch.save(model.state_dict(), model_path)


def save_bert_model(model, tokenizer, params):
    output_dir = 'Saved/'
    embed_type = params["model"].split('-')[0]
    output_dir += embed_type
    if params["remove_layers"]:
        removed_layers = params["remove_layers"].replace(',', '_')
        output_dir += "_r"
        output_dir += removed_layers
    if (params['train_att']):
        if (params['att_lambda'] >= 1):
            params['att_lambda'] = int(params['att_lambda'])
        output_dir += "_"
        output_dir = output_dir + str(params['supervised_layer_pos']) + '_' + str(
            params['num_supervised_heads']) + '_' + str(params['num_classes']) + '_' + str(params['att_lambda']) + '/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(output_dir + "/params.json", "w") as fp:
        print("Saving params to", output_dir + "params.json")
        json.dump(params, fp, cls=NumpyEncoder)

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

def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger