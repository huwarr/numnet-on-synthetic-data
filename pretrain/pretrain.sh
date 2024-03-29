#!/bin/env bash
#SBATCH --job-name=roberta_numnet_plus
source activate numnet_venv_1_0

set -xe

SEED=$1
LR=$2 # 1e-5,   max_epochs = 5, warmup = 0.1, batch_size = 240-624
BLR=$3
WD=$4
BWD=$5
TMSPAN=$6
DATA_DIR=$7

BASE_DIR=.

CODE_DIR=${BASE_DIR}

if [ ${TMSPAN} = tag_mspan ]; then
  echo "Use tag_mspan model..."
  MODEL_CONFIG="--tag_mspan"
  # Numeric data
  CACHED_TRAIN_NUMERIC=${DATA_DIR}/tmspan_cached_roberta_numeric_train.pkl
  CACHED_DEV_NUMERIC=${DATA_DIR}/tmspan_cached_roberta_numeric_dev.pkl
  if [ \( ! -e "${CACHED_TRAIN_NUMERIC}" \)  -o \( ! -e "${CACHED_DEV_NUMERIC}" \) ]; then
  echo "Preparing cached numeric data."
  python ../prepare_roberta_data.py --data_type numeric --input_path ${DATA_DIR} --roberta_path ${DATA_DIR}/roberta.large  --output_dir ${DATA_DIR} --tag_mspan
  fi
  # Textual data
  CACHED_TRAIN_TEXTUAL=${DATA_DIR}/tmspan_cached_roberta_textual_train.pkl
  CACHED_DEV_TEXTUAL=${DATA_DIR}/tmspan_cached_roberta_textual_dev.pkl
  if [ \( ! -e "${CACHED_TRAIN_TEXTUAL}" \)  -o \( ! -e "${CACHED_DEV_TEXTUAL}" \) ]; then
  echo "Preparing cached textual data."
  python ../prepare_roberta_data.py --data_type textual --input_path ${DATA_DIR} --roberta_path ${DATA_DIR}/roberta.large --output_dir ${DATA_DIR} --tag_mspan
  fi
else
  echo "Use mspan model..."
  MODEL_CONFIG=""
  # Numeric data
  CACHED_TRAIN_NUMERIC=${DATA_DIR}/cached_roberta_numeric_train.pkl
  CACHED_DEV_NUMERIC=${DATA_DIR}/cached_roberta_numeric_dev.pkl
  if [ \( ! -e "${CACHED_TRAIN_NUMERIC}" \)  -o \( ! -e "${CACHED_DEV_NUMERIC}" \) ]; then
  echo "Preparing cached numeric data."
  python ../prepare_roberta_data.py --data_type numeric --input_path ${DATA_DIR} --roberta_path ${DATA_DIR}/roberta.large --output_dir ${DATA_DIR}
  fi
  # Textual data
  CACHED_TRAIN_TEXTUAL=${DATA_DIR}/cached_roberta_textual_train.pkl
  CACHED_DEV_TEXTUAL=${DATA_DIR}/cached_roberta_textual_dev.pkl
  if [ \( ! -e "${CACHED_TRAIN_TEXTUAL}" \)  -o \( ! -e "${CACHED_DEV_TEXTUAL}" \) ]; then
  echo "Preparing cached textual data."
  python ../prepare_roberta_data.py --data_type textual --input_path ${DATA_DIR} --roberta_path ${DATA_DIR}/roberta.large --output_dir ${DATA_DIR}
  fi
fi


SAVE_DIR=${BASE_DIR}
DATA_CONFIG="--data_dir ${DATA_DIR} --save_dir ${SAVE_DIR}"
TRAIN_CONFIG="--batch_size 128 --eval_batch_size 5 --max_epoch 5 --warmup 0.1 --optimizer adam \
              --learning_rate ${LR} --weight_decay ${WD} --seed ${SEED} --gradient_accumulation_steps 4 \
              --bert_learning_rate ${BLR} --bert_weight_decay ${BWD} --log_per_updates 100 --eps 1e-6"
BERT_CONFIG="--roberta_model ${DATA_DIR}/roberta.large"


echo "Start pretraining..."
python ${CODE_DIR}/roberta_gcn_pretrain.py \
    ${DATA_CONFIG} \
    ${TRAIN_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 5 --pre_path ${SAVE_DIR}/pretrain_best.pt --data_mode dev --dump_path_numeric ${SAVE_DIR}/dev_num.json \
             --dump_path_textual ${SAVE_DIR}/dev_text.json --inf_path_numeric ${DATA_DIR}/numeric_data_dev.json \
             --inf_path_textual ${DATA_DIR}/textual_data_dev.json"

python ${CODE_DIR}/roberta_pretrain_predict.py \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}

echo "Evaluation for numeric data..."
python ../drop_eval.py \
    --gold_path ${DATA_DIR}/numeric_data_dev.json \
    --prediction_path ${SAVE_DIR}/dev_num.json

echo "Evaluation for textual data..."
python ../drop_eval.py \
    --gold_path ${DATA_DIR}/textual_data_dev.json \
    --prediction_path ${SAVE_DIR}/dev_text.json