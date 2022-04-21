#!/bin/env bash
#SBATCH --job-name=eval_roberta_numnet_plus
source activate numnet_venv_1_0

set -xe

DATA_PATH_NUMERIC=$1
DATA_PATH_TEXTUAL=$2
DUMP_PATH_NUMERIC=$3
DUMP_PATH_TEXTUAL=$4
TMSPAN=$4
PRE_PATH=$5
PRETRAIN_PATH=$6

BERT_CONFIG="--roberta_model ${PRETRAIN_PATH}"

if [ ${TMSPAN} = tag_mspan ]; then
    echo "Use tag_mspan model..."
    MODEL_CONFIG="--gcn_steps 3 --use_gcn --tag_mspan"
else
    echo "Use mspan model..."
    MODEL_CONFIG="--gcn_steps 3 --use_gcn"
fi

echo "Starting evaluation..."
TEST_CONFIG="--eval_batch_size 32 --pre_path ${PRE_PATH} --data_mode dev --dump_path_numeric ${DUMP_PATH_NUMERIC} --dump_path_textual ${DUMP_PATH_TEXTUAL} \
             --inf_path_numeric ${DATA_PATH_NUMERIC} --inf_path_textual ${DATA_PATH_TEXTUAL}"

python roberta_predict.py \
    ${TEST_CONFIG} \
    ${BERT_CONFIG} \
    ${MODEL_CONFIG}
