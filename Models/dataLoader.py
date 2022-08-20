import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import sys, os

sys.path.append('../')
import pandas as pd
from Models.dataCollect import collect_data, set_name
from os import path
import pickle
import json

def custom_att_masks(input_ids):
    attention_masks = []

    # For each sentence...
    for sent in input_ids:
        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks


def combine_features(tuple_data, params, is_train=False):
    input_ids = [ele[0] for ele in tuple_data]
    att_vals = [ele[1] for ele in tuple_data]
    labels = [ele[2] for ele in tuple_data]

    encoder = LabelEncoder()

    encoder.classes_ = np.load(params['class_names'], allow_pickle=True)
    labels = encoder.transform(labels)

    input_ids = pad_sequences(input_ids, maxlen=int(params['max_length']), dtype="long",
                              value=0, truncating="post", padding="post")
    att_vals = pad_sequences(att_vals, maxlen=int(params['max_length']), dtype="float",
                             value=0.0, truncating="post", padding="post")
    att_masks = custom_att_masks(input_ids)
    dataloader = return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train)
    return dataloader


def return_dataloader(input_ids, labels, att_vals, att_masks, params, is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels, dtype=torch.long)
    masks = torch.tensor(np.array(att_masks), dtype=torch.uint8)
    attention = torch.tensor(np.array(att_vals), dtype=torch.float)
    data = TensorDataset(inputs, attention, masks, labels)
    if (is_train == False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=params['batch_size'])
    return dataloader
class Vocab_own():
    def __init__(self, dataframe, model):
        self.itos = {}
        self.stoi = {}
        self.vocab = {}
        self.embeddings = []
        self.dataframe = dataframe
        self.model = model

    ### load embedding given a word and unk if word not in vocab
    ### input: word
    ### output: embedding,word or embedding for unk, unk
    def load_embeddings(self, word):
        try:
            return self.model[word], word
        except KeyError:
            return self.model['unk'], 'unk'

    ### create vocab,stoi,itos,embedding_matrix
    ### input: **self
    ### output: updates class members
    def create_vocab(self):
        count = 1
        for index, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe)):
            for word in row['Text']:
                vector, word = self.load_embeddings(word)
                try:
                    self.vocab[word] += 1
                except KeyError:
                    if (word == 'unk'):
                        print(word)
                    self.vocab[word] = 1
                    self.stoi[word] = count
                    self.itos[count] = word
                    self.embeddings.append(vector)
                    count += 1
        self.vocab['<pad>'] = 1
        self.stoi['<pad>'] = 0
        self.itos[0] = '<pad>'
        self.embeddings.append(np.zeros((300,), dtype=float))
        self.embeddings = np.array(self.embeddings)


def encodeData(dataframe):
    tuple_new_data = []
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        tuple_new_data.append((row['Text'], row['Attention'], row['Label']))
    return tuple_new_data


def createDatasetSplit(tokenizer, params):
    filename = set_name(params)
    if not path.exists(filename):
        dataset = collect_data(tokenizer, params)

    if (path.exists(filename[:-7])):
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