from transformers.models.bert.modeling_bert import *
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel
from .utils import masked_cross_entropy
from transformers import AutoModel
import sys


def init_model(params):
    model_type = ''
    embed_type = params["model"].split('-')[0]
    embed_type = embed_type.split("/")[-1]
    if embed_type == "bert":
        model_type = BertPreTrainedModel

    if embed_type == "distilbert":
        model_type = DistilBertPreTrainedModel

    if embed_type == "roberta":
        model_type = RobertaPreTrainedModel

    class SC_weighted_Encoder(model_type):
        def __init__(self, config, params):
            super().__init__(config)
            self.num_labels = config.num_labels
            self.weights = params['weights']
            self.train_att = params['train_att']
            self.lam = params['att_lambda']
            self.num_sv_heads = params['num_supervised_heads']
            self.sv_layer = params['supervised_layer_pos']
            model_path = params["path_files"] if params["path_files"] != "N/A" else params["model"]
            __model__ = AutoModel.from_pretrained(model_path)
            if embed_type == "roberta":
                self.roberta = __model__
                self.embed_list = list(self.roberta.embeddings.parameters())
                self.layer_list = self.roberta.encoder.layer
            if embed_type == "bert":
                self.bert = __model__
                self.embed_list = list(self.bert.embeddings.parameters())
                self.layer_list = self.bert.encoder.layer
            if embed_type == "distilbert":
                self.distilbert = __model__
                self.embed_list = list(self.distilbert.embeddings.parameters())
                self.layer_list = self.distilbert.transformer.layer
            self.m_type = embed_type
            self.remove_layers = params['remove_layers']
            self.freeze_embeddings = params["freeze_embeddings"]
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            # self.softmax=nn.Softmax(config.num_labels)
            self.init_weights()
            if params["path_files"] == "N/A":
                self.__removelayers__()
        def __removelayers__(self):
            if self.remove_layers:
                layer_indexes = [int(x) for x in self.remove_layers.split(",")]
                layer_indexes.sort(reverse=True)
                for layer_idx in layer_indexes:
                    if layer_idx < 0:
                        print(f"Only positive indices allowed, passed {layer_idx}")
                        sys.exit(1)
                    for param in list(self.layer_list[layer_idx].parameters()):
                        param.requires_grad = False
                    del (self.layer_list[layer_idx])
                    print("Removed Layer:", layer_idx)
            if self.freeze_embeddings:
                for param in self.embed_list:
                    param.requires_grad = False
                print("Froze Embedding Layer")
            if self.m_type in {"bert", "roberta"}:
                self.config.num_hidden_layers = len(self.layer_list)
            elif self.m_type == "distilbert":
                self.config.n_layers = len(self.layer_list)

        def forward(self,
                    input_ids=None,
                    attention_mask=None,
                    attention_vals=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    device=None):

            args_fwd_step = {
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "head_mask": head_mask,
                "inputs_embeds": inputs_embeds
            }
            outputs = []
            if self.m_type == "roberta":
                outputs = self.roberta(
                    input_ids, **args_fwd_step
                )
            if self.m_type == "bert":
                outputs = self.bert(
                    input_ids, **args_fwd_step
                )
            if self.m_type == "distilbert":
                outputs = self.distilbert(
                    input_ids, **args_fwd_step
                )
            pooled_output = outputs[1]

            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            # logits = self.softmax(logits)

            outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

            if labels is not None:
                loss_funct = CrossEntropyLoss(weight=torch.tensor(self.weights).to(device))
                loss_logits = loss_funct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss_logits
                if (self.train_att):
                    loss_att = 0
                    for i in range(self.num_sv_heads):
                        attention_weights = outputs[1][self.sv_layer][:, i, 0, :]
                        loss_att += self.lam * masked_cross_entropy(attention_weights, attention_vals, attention_mask)
                    loss = loss + loss_att
                outputs = (loss,) + outputs

            return outputs

    return SC_weighted_Encoder