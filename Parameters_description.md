## Parameters used for processing HateXplain dataset

Default dataset parameters listed in ```../params.yml```:
```
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
```

* **data_file** -> data file path 
* **class_names** -> path to file with class names. ```./Data/classes.npy``` has 3 class names: `offensive`, `hateful`, `neutral`. ```./Data/classes_two.npy``` file has two classes: `toxic` and `non-toxic`. 
* **num_classes** -> 2 or 3. Must go in line with a range ```class_names``` file.
* **max_length** -> This represent the maximum length of the words in case of non transformers models or subwords in case of transformers models. For all our models this is set to 128.
* **include_special** -> This can be set as *True* or *False*. This is with respect to [ekaphrasis](https://github.com/cbaziotis/ekphrasis) processing. For example ekphrasis adds a special character to the hashtags after processing. for e.g. #happyholidays will be transformed to <hashtag> happy holidays </hashtag>. If set the `include specials` to false than the begin token and ending token will be removed. 
* **type_attention** -> How the normalisation of the attention vector will happen. Three options are available currently "softmax","neg_softmax" and "sigmoid". More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py).
* **variance** -> constant multiplied with the attention vector to increase the difference between the attention to attended and non-attended tokens.More details [here](https://github.com/punyajoy/HateXplain/blob/master/Preprocess/attentionCal.py).  
~~~
variance=5
attention = [0,0,0,1,0]
attention_modified = attention * variance
attention_modified = [0,0,0,5,0]
~~~

* **decay** -> whether to decay the attentions left and right of the _attentive_ word (to decentralise the attention to a single word)


The following parameters are used if **decay** is set to **True**.
* **window** -> window = number of tokens used to re-distribute attention on right and left tokens
* **alpha** -> alpha regulates the proportion of original attention to preserve $\alpha \in (0,1)$
* **p_value** -> if ```method``` is set to ```"geometric"``` , p_value is used to generate the geometric distributing for the left and right side from the _attentive_ word
* **method** -> ```"geometric"``` or ```"additive"```: the way to generate attention distribution.
* **normalized** -> if True, generated attention values are normalised.
