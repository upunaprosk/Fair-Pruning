# :scissors: Measuring Bias in Pruned Transformers  
This repository contains code for the paper [Measuring Bias in Pruned Transformers](https://link.springer.com/chapter/10.1007/978-3-031-30047-9_2) accepted to IDA 2023.

In that paper, we evaluate bias in compressed models trained on Gab and Twitter speech data and estimate to which extent these pruned models capture the relevant context when classifying the input text as hateful, offensive or neutral. Results of our experiments show that **transformer-based encoders with 70% or fewer preserved weights are prone to gender, racial, and religious identity-based bias, even if the performance loss is insignificant**. We suggest a supervised attention mechanism to counter bias amplification using ground truth per-token hate speech annotation. The proposed method allows pruning BERT, RoBERTa and their distilled versions up to 50% while preserving 90% of their initial performance according to bias and plausibility scores.

## Dependencies

* python 3.9.16
* transformers 4.27.4
* torch 2.0.0
* ekphrasis 0.5.4
* sentencepiece 0.1.97
* datasets 2.11.0
* omegaconf 2.3.0

We use _HateXplain_ dataset presented in the following paper:

Mathew, Binny, et al. "Hatexplain: A benchmark dataset for explainable hate speech detection." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 35. No. 17. 2021.

## Usage

Refer to *full example* usage which is availiable [here](https://github.com/upunaprosk/fair-pruning/blob/master/Example%20run.ipynb).

<details>
    <summary>Clone the repository and install the dependencies</summary>
    
```
git clone https://github.com/upunaprosk/fair-pruning
cd fair-pruning
python -m venv fair-pruning
source ./fair-pruning/bin/activate #Windows: fair-pruning\Scripts\activate
pip install -r ./requirements.txt
```

</details>


<details>
    <summary>Specify parameters in params.yml file</summary>
 
 
Training parameters ```remove_layers``` / ```freeze_layers``` are indices of encoder layers to be removed/frozen. The provided list should be sorted and the indices should be separated by comma.

```att_lambda``` is a value of coefficient regulating attention loss contribution to overall loss: $$\text{Loss}=L(\theta) + \lambda L_{attn}.$$
```num_supervised_heads``` is the number of supervised heads, ```supervised_layer_pos``` is index of supervised layer in *pruned language model*. 

Default **training** parameters include the following ones:

```
model: "bert-base-cased"
seed: 42
training:
  device: "gpu"
  batch_size: 16
  remove_layers: "8,9,10,11"
  freeze_layers: ""
  freeze_embeddings: False
  learning_rate: 2e-5
  epochs: 3
  auto_weights: True
  report_to: "wandb"
  train_att: True
  att_lambda: 1
  num_supervised_heads: 1
  supervised_layer_pos: 0
  
```

Use the following command to change default parameters file:

```
%%bash
cat <<__YML__ > params.yml
model: "bert-base-cased"
seed: 42
training:
  device: "gpu"
  batch_size: 16
  remove_layers: "6,7,8,9,10,11"
  freeze_layers: ""
  freeze_embeddings: False
  learning_rate: 2e-5
  epochs: 3
  auto_weights: True
  report_to: "wandb"
  train_att: True
  att_lambda: 1
  num_supervised_heads: 1
  supervised_layer_pos: 0
dataset:
  data_file: "Data/dataset.json"
  class_names: "Data/classes.npy"
  num_classes: 3
  max_length: 128
  include_special: False
  type_attention: "softmax"
  variance: 5.0
  decay: False
  window: 4.0
  alpha: 0.5
  p_value: 0.8
  method: "additive"
  normalized: False
logging: ""
__YML__
:
  
```

Parameters for **processing HateXplain** are listed [here](https://github.com/upunaprosk/fair-pruning/blob/master/Parameters_description.md).


</details>

<details>
    <summary>HateXplain usage</summary>

If you want just to use _HateXplain_ solely and generate data with rationales needed for supervised attention learning, use the following commands:

```
from src.data_load import *
train, val, test = createDatasetSplit()
train_dataset = Dataset.from_pandas(combine_features(train, is_train=True))
validation_dataset = Dataset.from_pandas(combine_features(val, is_train=False))
predict_dataset = Dataset.from_pandas(combine_features(test, is_train=False))
```
</details>

<details>
    <summary>Training</summary>
    
Training is based on [training scipt](https://github.com/upunaprosk/fair-pruning/blob/master/src/train.py) and is carries on HuggingFace Trainer class instance. 
If you have ```Out of memory``` GPU error issue during **evaluation**/**prediction** steps, consider commenting these steps ```trainer.evaluate()``` / ```trainer.predict()```, and evaluate models after training on CPU without calling ```.train()``` method. That happens due to existing Trainer [aggregating predictions on GPU issue](https://github.com/huggingface/transformers/issues/7232).    
</details>


<details>
    <summary>Bias and Explainability Evaluation</summary>
 
Bias measures include AUC measures calculated using (hate) target community annotations: Background Positive Subgroup Negative (BPSN), BNSP, Subgroup AUC. 
Explainability measures are calculated based on predicted explanations (top tokens with highest attention weights) and true explanations (top tokens marked as descision reasoning by annotators).     
</details>

## Contact

If you have any question about the project and/or code, you can reach us [here](mailto:irina.proskurina@univ-lyon2.fr).

## Cite

```
@inproceedings{proskurina2023other,
  title={The Other Side of Compression: Measuring Bias in Pruned Transformers},
  author={Proskurina, Irina and Metzler, Guillaume and Velcin, Julien},
  booktitle={Advances in Intelligent Data Analysis XXI: 21st International Symposium on Intelligent Data Analysis, IDA 2023, Louvain-la-Neuve, Belgium, April 12--14, 2023, Proceedings},
  pages={366--378},
  year={2023},
  organization={Springer}}
```
