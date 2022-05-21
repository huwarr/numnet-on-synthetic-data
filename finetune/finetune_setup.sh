TMSPAN=$1

# Requirements
# pip install -r requirements.txt

# Download DROP data
wget -O drop_dataset.zip https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip

# Tag based multi-span extraction -- NumNet+ v2
if [ ${TMSPAN} = tag_mspan ]; then
    # Train
    sh finetune.sh 345 3e-5 1.5e-5 5e-5 0.01 tag_mspan drop_dataset
    # Eval
    sh finetune_eval.sh drop_dataset/drop_dataset_dev.json prediction.json tag_mspan numnet_plus_345_LR_5e-4_BLR_1.5e-5_WD_5e-5_BWD_0.01tag_mspan/checkpoint_best.pt ../pretrain/roberta.large
    python ../drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json
# Simple multi-span extraction -- NumNet+
else
    # Train
    sh finetune.sh 345 3e-5 1.5e-5 5e-5 0.01 no drop_dataset
    # Eval
    sh finetune_eval.sh drop_dataset/drop_dataset_dev.json prediction.json no numnet_plus_345_LR_5e-4_BLR_1.5e-5_WD_5e-5_BWD_0.01/checkpoint_best.pt ../pretrain/roberta.large
    python ../drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json
fi