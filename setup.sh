TMSPAN=$1

# Requirements
pip install -r requirements.txt

# Download drop data
wget -O drop_dataset.zip https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip
unzip drop_dataset.zip

# Download GenBERT pretrained model
cd drop_dataset
pip install --upgrade --no-cache-dir gdown
gdown --folder https://drive.google.com/drive/folders/1-KmhWF4Jex4gyuz1J2VANd1ycmtsggQ3
cd ..

# Tag based multi-span extraction -- NumNet+ v2
if [ ${TMSPAN} = tag_mspan ]; then
    # Train
    sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 tag_mspan drop_dataset
    # Eval
    sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json tag_mspan numnet_plus_345_LR_5e-4_BLR_1.5e-5_WD_5e-5_BWD_0.01tag_mspan/checkpoint_best.pt drop_dataset/genbert
    python3 drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json
# Simple multi-span extraction -- NumNet+
else
    # Train
    sh train.sh 345 5e-4 1.5e-5 5e-5 0.01 no drop_dataset
    # Eval
    sh eval.sh drop_dataset/drop_dataset_dev.json prediction.json no numnet_plus_345_LR_5e-4_BLR_1.5e-5_WD_5e-5_BWD_0.01/checkpoint_best.pt drop_dataset/genbert
    python3 drop_eval.py --gold_path drop_dataset/drop_dataset_dev.json --prediction_path prediction.json
fi