import os
import sys
import inspect
# for 'import options'
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import json
import torch
import options
import argparse
from tqdm import tqdm
from mspan_roberta_gcn.inference_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from mspan_roberta_gcn.drop_roberta_dataset import DropReader
from tag_mspan_robert_gcn.drop_roberta_mspan_dataset import DropReader as TDropReader
from tag_mspan_robert_gcn.inference_batch_gen import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import (
    NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet,
)
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig


parser = argparse.ArgumentParser("Bert inference task.")
options.add_bert_args(parser)
options.add_model_args(parser)
options.add_inference_args(parser)

args = parser.parse_args()

args.cuda = torch.cuda.device_count() > 0


print("Build bert model.")
bert_model = RobertaModel(RobertaConfig().from_pretrained(args.roberta_model))
print("Build Drop model.")
if args.tag_mspan:
    network = TNumericallyAugmentedBertNet(
        bert_model,
        hidden_size=bert_model.config.hidden_size,
        dropout_prob=0.0,
        use_gcn=args.use_gcn,
        gcn_steps=args.gcn_steps,
    )
else:
    network = NumericallyAugmentedBertNet(
        bert_model,
        hidden_size=bert_model.config.hidden_size,
        dropout_prob=0.0,
        use_gcn=args.use_gcn,
        gcn_steps=args.gcn_steps,
    )

if args.cuda:
    network.cuda()
print("Load from pre path {}.".format(args.pre_path))
network.load_state_dict(torch.load(args.pre_path))

print("Load data from {} and {}.".format(args.inf_path_numeric, args.inf_path_textual))
tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)
if args.tag_mspan:
    inf_iter_num = TDropBatchGen(
        args,
        tokenizer,
        TDropReader(
            tokenizer, passage_length_limit=463, question_length_limit=46
        )._read(args.inf_path_numeric),
    )
    inf_iter_text = TDropBatchGen(
        args,
        tokenizer,
        TDropReader(
            tokenizer, passage_length_limit=463, question_length_limit=46
        )._read(args.inf_path_textual),
    )
else:
    inf_iter_num = DropBatchGen(
        args,
        tokenizer,
        DropReader(tokenizer, passage_length_limit=463, question_length_limit=46)._read(
            args.inf_path_numeric
        ),
    )
    inf_iter_text = DropBatchGen(
        args,
        tokenizer,
        DropReader(tokenizer, passage_length_limit=463, question_length_limit=46)._read(
            args.inf_path_textual
        ),
    )

print("Start inference...")
network.eval()

# Numeric data
print("Numeric data...")
result = {}
with torch.no_grad():
    for batch in tqdm(inf_iter_num):
        output_dict = network(**batch)
        for i in range(len(output_dict["question_id"])):
            result[output_dict["question_id"][i]] = output_dict["answer"][i][
                "predicted_answer"
            ]
with open(args.dump_path_numeric, "w", encoding="utf8") as f:
    json.dump(result, f)

# Textual data
print("Textual data...")
result = {}
with torch.no_grad():
    for batch in tqdm(inf_iter_text):
        output_dict = network(**batch)
        for i in range(len(output_dict["question_id"])):
            result[output_dict["question_id"][i]] = output_dict["answer"][i][
                "predicted_answer"
            ]

with open(args.dump_path_textual, "w", encoding="utf8") as f:
    json.dump(result, f)