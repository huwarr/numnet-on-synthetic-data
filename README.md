# NumNet+ pretraining on GenBERT data

Pretraining NumNet+ on synthetic numeric and textual data, the way GenBERT was.

## Sources

Data and pretraining scheme: [github](https://github.com/ag1988/injecting_numeracy/tree/master/pre_training).

NumNet code was taken from: [github](https://github.com/llamazing/numnet_plus).

## Requirements

`pip install -r requirements.txt`

## PRETRAINING

`cd pretrain`

### Data and pretrained RoBERTa

- Download synthetic data, already converted to DROP format.

  `mkdir synthetic_data && cd synthetic_data`
  
  `pip install gdown`
  
  Numeric synthetic data:
  
  - numeric_dataset_train.json:
  
    `gdown https://drive.google.com/uc?id=1WoCuawj3F1RRHG9RJ0Pfow597ASgsTf5`
  
  - numeric_dataset_dev.json:
  
    `gdown https://drive.google.com/uc?id=1juJczB0mQorhKOpfvE6z0_hFxP44jkUd`
  
  Textual synthetic data:
  
  - textual_dataset_train.json:
  
    `gdown https://drive.google.com/uc?id=1TXZv_za1I_zC3LZ4bg2A8IJmOLpqd7xo`
  
  - textual_dataset_dev.json:
  
    `gdown https://drive.google.com/uc?id=1p3OeXhpmbdrhba4P_onWveUVlPinyTig`

- Download RoBERTa.

  `mkdir roberta.large && cd roberta.large'
  
  `wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin'
  
  `wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json'
  
  `wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json'
  
  `wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt'
  
- Modify **config.json** from `"output_hidden_states": false` to `"output_hidden_states": true`.

### Pretrain

`cd ../..`

- Train with tag based multi-span extraction (NumNet+ v2):

  `sh pretrain.sh 345 1e-5 1e-5 5e-5 0.01 tag_mspan synthetic_data`

- Simple multi-span extraction (NumNet+):

  `sh pretrain.sh 345 1e-5 1e-5 5e-5 0.01 no synthetic_data`
