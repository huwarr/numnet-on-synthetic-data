TMSPAN=$1

# Requirements
pip install -r requirements.txt

# Download synthetic data, already converted to DROP format
mkdir synthetic_data && cd synthetic_data
pip install gdown
# Numeric
gdown https://drive.google.com/uc?id=1WoCuawj3F1RRHG9RJ0Pfow597ASgsTf5
gdown https://drive.google.com/uc?id=1juJczB0mQorhKOpfvE6z0_hFxP44jkUd
# Textual
gdown https://drive.google.com/uc?id=1TXZv_za1I_zC3LZ4bg2A8IJmOLpqd7xo
gdown https://drive.google.com/uc?id=1p3OeXhpmbdrhba4P_onWveUVlPinyTig

# Download RoBERTa
mkdir roberta.large && cd roberta.large
wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
'''
Modify config.json from "output_hidden_states": false to "output_hidden_states": true
'''
wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt

cd ../..

# Tag based multi-span extraction -- NumNet+ v2
if [ ${TMSPAN} = tag_mspan ]; then
    # Train
    sh pretrain.sh 345 5e-4 1.5e-5 5e-5 0.01 tag_mspan synthetic_data
    # Eval
    sh pretrain_eval.sh synthetic_data/synthetic_numeric_dev_drop_format.json synthetic_data/synthetic_textual_mixed_min3_max6_up0.7_dev.json prediction_num.json prediction_text.json tag_mspan ./pretrain_best.pt synthetic_data/roberta.large
    # Numeric
    echo "Evaluating numeric data..."
    python ../drop_eval.py --gold_path pretrain/synthetic_data/synthetic_numeric_dev_drop_format.json --prediction_path pretrain/prediction_num.json
    # Textual
    echo "Evaluating textual data..."
    python ../drop_eval.py --gold_path pretrain/synthetic_data/synthetic_textual_mixed_min3_max6_up0.7_dev.json --prediction_path pretrain/prediction_text.json
# Simple multi-span extraction -- NumNet+
else
    # Train
    sh pretrain.sh 345 5e-4 1.5e-5 5e-5 0.01 no synthetic_data
    # Eval
    sh pretrain_eval.sh synthetic_data/synthetic_numeric_dev_drop_format.json synthetic_data/synthetic_textual_mixed_min3_max6_up0.7_dev.json prediction_num.json prediction_text.json no ./pretrain_best.pt synthetic_data/roberta.large
    # Numeric
    echo "Evaluating numeric data..."
    python ../drop_eval.py --gold_path pretrain/synthetic_data/synthetic_numeric_dev_drop_format.json --prediction_path pretrain/prediction_num.json
    # Textual
    echo "Evaluating textual data..."
    python ../drop_eval.py --gold_path pretrain/synthetic_data/synthetic_textual_mixed_min3_max6_up0.7_dev.json --prediction_path pretrain/prediction_text.json
fi