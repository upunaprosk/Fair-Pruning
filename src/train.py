from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    set_seed,
)
import torch.nn as nn
from sklearn.utils import class_weight
from datasets import Dataset
from datasets import load_metric
from src.data_load import *
from src.utils import *


def cross_entropy(input1, target):
    """ Cross entropy that accepts soft targets
    Args:
         input1: predictions for neural network
         target: targets, can be soft

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


def masked_cross_entropy(input1, target, mask):
    cr_ent = 0
    for h in range(0, mask.shape[0]):
        cr_ent += cross_entropy(input1[h][mask[h]], target[h][mask[h]])

    return cr_ent / mask.shape[0]


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


@conf("./params.yml", as_default=True)
def train_model(**params):
    log_level = logging.DEBUG if params["logging"] == "debug" else logging.INFO
    logger = set_logger(level=log_level)
    set_seed(params["seed"])
    num_labels = params["dataset"]["num_classes"]
    model_name_or_path = params["model"]
    logger.info(f"Initialized config from path {model_name_or_path}")
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        output_attentions=True
    )
    # default_output_dir: "../bert-base-cased_128_removed_layers_8_9_10_11_attn_softmax_5.0"
    params = set_output_dir(**params)
    training_params = params['training']
    device_ = None
    if not params.get("device", 0):
        if torch.cuda.is_available():
            device_ = torch.device("cuda")
        else:
            device_ = torch.device("cpu")
    else:
        device_ = torch.device(params["device"])
    logger.warning(
        f"Using device: {device_}")
    logger.debug(f"Training and evaluation parameters {training_params}")
    train, val, test = createDatasetSplit(**params)
    class_weights = [1] * num_labels
    if training_params['auto_weights']:
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params["dataset"]['class_names'], allow_pickle=True)
        class_weights = class_weight.compute_class_weight("balanced", classes=np.unique(y_test),
                                                          y=y_test).astype('float32')
    train_dataset = Dataset.from_pandas(combine_features(train, **params, is_train=True))
    validation_dataset = Dataset.from_pandas(combine_features(val, **params, is_train=False))
    predict_dataset = Dataset.from_pandas(combine_features(test, **params, is_train=False))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config
    )
    # Freezing
    embed_type = model_name_or_path.split('-')[0]
    embed_type = embed_type.split("/")[-1]
    embed_list = []
    layer_list = []
    if embed_type == "bert":
        embed_list = list(model.bert.embeddings.parameters())
        layer_list = model.bert.encoder.layer
    if embed_type == "distilbert":
        embed_list = list(model.distilbert.embeddings.parameters())
        layer_list = model.distilbert.transformer.layer
    if embed_type == "roberta":
        embed_list = list(model.roberta.embeddings.parameters())
        layer_list = model.roberta.encoder.layer
    if embed_type == "distilroberta":
        embed_list = list(model.roberta.embeddings.parameters())
        layer_list = model.roberta.encoder.layer

    remove_layers = training_params['remove_layers']
    freeze_layers = training_params['freeze_layers']
    freeze_embeddings = training_params["freeze_embeddings"]
    if freeze_layers:
        layer_indexes = [int(x) for x in freeze_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            for param in list(layer_list[layer_idx].parameters()):
                param.requires_grad = False
            logger.info("Layer frozen: " + str(layer_idx))
    if freeze_embeddings:
        for param in embed_list:
            param.requires_grad = False
        logger.info("Embedding layer frozen.")
    if remove_layers:
        layer_indexes = [int(x) for x in remove_layers.split(",")]
        layer_indexes.sort(reverse=True)
        for layer_idx in layer_indexes:
            for param in list(layer_list[layer_idx].parameters()):
                param.requires_grad = False
            del (layer_list[layer_idx])
            logger.info("Layer removed: " + str(layer_idx))
    if embed_type in {"bert", "roberta", "distilroberta"}:
        model.config.num_hidden_layers = len(layer_list)
    if embed_type == "distilbert":
        model.config.n_layers = len(layer_list)

    ACCURACY = load_metric("accuracy", keep_in_memory=True)
    F1 = load_metric("f1", keep_in_memory=True)

    def compute_metrics(p: EvalPrediction):

        preds = p.predictions[0]
        logits_all_final = []
        for logits in preds:
            logits_all_final.append(softmax(logits))
        preds = np.argmax(preds, axis=1)
        acc_result = ACCURACY.compute(predictions=preds, references=p.label_ids)
        f1_result = F1.compute(predictions=preds, references=p.label_ids, average='macro')
        result = {"accuracy": acc_result["accuracy"], "f1": f1_result["f1"]}
        return result

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, train_att=False, att_lambda=False,
                     num_supervised_heads=0, supervised_layer_pos=0, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Supervised attention training args
            self.class_weights = class_weights
            self.train_att = train_att
            self.lam = att_lambda
            self.num_sv_heads = num_supervised_heads
            self.sv_layer = supervised_layer_pos

        def compute_loss(self, model, inputs, return_outputs=False):
            """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.
            Subclass and override for custom behavior.
            """

            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels

            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['masks'], labels=inputs['labels'])
            attention_vals = inputs['attention']
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]
            if labels is not None:
                loss = self.label_smoother(outputs, labels)
            else:
                logits = outputs['logits']
                criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
                if not self.args.no_cuda:
                    loss = criterion(logits.float(), inputs['labels'].cuda().long())
                else:
                    loss = criterion(logits.float(), inputs['labels'].cuda().long())
                if self.train_att:
                    loss_att = 0
                    for i in range(self.num_sv_heads):
                        attention_weights = outputs.attentions[0][:, i, self.sv_layer, :]
                        loss_att += self.lam * masked_cross_entropy(attention_weights, attention_vals, inputs['masks'])
                    loss = loss + loss_att
            return (loss, outputs) if return_outputs else loss

    training_args = {}
    for pp in ["learning_rate", "report_to"]:
        training_args[pp] = training_params[pp]
    training_args["overwrite_output_dir"] = True
    training_args["per_device_train_batch_size"] = training_params["batch_size"]
    training_args["per_device_eval_batch_size"] = 1
    training_args["output_dir"] = str(params["output_dir"])
    training_args["num_train_epochs"] = training_params["epochs"]
    training_args["do_train"] = True
    training_args["do_eval"] = True
    training_args["do_predict"] = True
    training_args["remove_unused_columns"] = False
    trainer_params = {"model": model,
                      "train_dataset": train_dataset,
                      "eval_dataset": validation_dataset,
                      "compute_metrics": compute_metrics}

    logger.debug("Using balanced Cross-Entropy with class weights: ")
    logger.debug(class_weights)
    logger.debug("Embedding type: " + str(embed_type))
    params["model"] = model

    trainer = WeightedTrainer(**trainer_params,
                              class_weights=torch.from_numpy(class_weights).float().to(device_),
                              train_att=training_params["train_att"], att_lambda=training_params["att_lambda"],
                              num_supervised_heads=training_params["num_supervised_heads"],
                              supervised_layer_pos=training_params["supervised_layer_pos"],
                              args=TrainingArguments(**training_args))
    trainer.log({"removed_layers": remove_layers})
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()

    trainer.log_metrics("train", metrics)
    trainer.log(metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=validation_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.log(metrics)
    trainer.save_metrics("eval", metrics)

    _, _, metrics = trainer.predict(test_dataset=predict_dataset)
    trainer.log_metrics("test", metrics)
    trainer.log(metrics)
    trainer.save_metrics("test", metrics)
    logger.debug("Model and scores saved to: " + training_args["output_dir"])
    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    train_model()
