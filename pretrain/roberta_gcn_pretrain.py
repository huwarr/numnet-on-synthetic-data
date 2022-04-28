import os
import sys
import inspect
# for 'import options'
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import json
import options
import argparse
from pprint import pprint
from tools.model import DropBertModel
from mspan_roberta_gcn.roberta_batch_gen import DropBatchGen
from mspan_roberta_gcn.mspan_roberta_gcn import NumericallyAugmentedBertNet
from tag_mspan_robert_gcn.roberta_batch_gen_tmspan import DropBatchGen as TDropBatchGen
from tag_mspan_robert_gcn.tag_mspan_roberta_gcn import (
    NumericallyAugmentedBertNet as TNumericallyAugmentedBertNet,
)
from datetime import datetime
from tools.utils import create_logger, set_environment
from transformers import RobertaTokenizer, RobertaModel


parser = argparse.ArgumentParser("Bert training task.")
options.add_bert_args(parser)
options.add_model_args(parser)
options.add_data_args(parser)
options.add_train_args(parser)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)

args.cuda = args.gpu_num > 0
args_path = os.path.join(args.save_dir, "args.json")
with open(args_path, "w") as f:
    json.dump(vars(args), f)

args.batch_size = args.batch_size // args.gradient_accumulation_steps
logger = create_logger(
    "Bert Drop Pretraining", log_file=os.path.join(args.save_dir, args.log_file)
)

pprint(args)
set_environment(args.seed, args.cuda)


def main():
    best_result = (float("-inf"), float("-inf"))
    logger.info("Loading data...")
    if not args.tag_mspan:
        train_itr_num = DropBatchGen(args, data_type='numeric', data_mode="train", tokenizer=tokenizer)
        dev_itr_num = DropBatchGen(args, data_type='numeric', data_mode="dev", tokenizer=tokenizer)
        train_itr_text = DropBatchGen(args, data_type='textual', data_mode="train", tokenizer=tokenizer)
        dev_itr_text = DropBatchGen(args, data_type='textual', data_mode="dev", tokenizer=tokenizer)
    else:
        train_itr_num = TDropBatchGen(args, data_type='numeric', data_mode="train", tokenizer=tokenizer)
        dev_itr_num = TDropBatchGen(args, data_type='numeric', data_mode="dev", tokenizer=tokenizer)
        train_itr_text = TDropBatchGen(args, data_type='textual', data_mode="train", tokenizer=tokenizer)
        dev_itr_text = TDropBatchGen(args, data_type='textual', data_mode="dev", tokenizer=tokenizer)
    
    num_train_steps = int(
        args.max_epoch * len(train_itr_text) / args.gradient_accumulation_steps
    )
    logger.info("Num update steps {}!".format(num_train_steps))

    logger.info("Build bert model.")
    bert_model = RobertaModel.from_pretrained(args.roberta_model)

    logger.info("Build Drop model.")
    if not args.tag_mspan:
        network = NumericallyAugmentedBertNet(
            bert_model,
            hidden_size=bert_model.config.hidden_size,
            dropout_prob=args.dropout,
            use_gcn=args.use_gcn,
            gcn_steps=args.gcn_steps,
        )
    else:
        network = TNumericallyAugmentedBertNet(
            bert_model,
            hidden_size=bert_model.config.hidden_size,
            dropout_prob=args.dropout,
            use_gcn=args.use_gcn,
            gcn_steps=args.gcn_steps,
        )

    logger.info("Build optimizer etc...")
    model = DropBertModel(args, network, num_train_step=num_train_steps)

    train_start = datetime.now()
    first = True
    updates_diff = 0
    # For stopping criteria
    loss_prev = float('inf')
    loss_curr = float('inf')
    for epoch in range(1, args.max_epoch + 1):
        model.avg_reset()
        if not first:
            # Shuffle training datasets
            # train_itr_num.reset()
            train_itr_text.reset()
        first = False
        logger.info("At epoch {}".format(epoch))
        # Update losses for stopping criteria
        loss_prev = loss_curr
        loss_curr = 0.
        count = 0
        # Like in GenBERT, loop iterates through textual batches
        for step, batch in enumerate(train_itr_text):
            # Textual data
            model.update(batch)
            updates_1 = model.updates
            train_loss_text = model.train_loss.avg
            train_em_text = model.em_avg.avg
            train_f1_text = model.f1_avg.avg
            
            # Numeric data
            while True:
                try:
                    batch = next(train_itr_num)  # sample next batch from numeric train data 
                    break
                except StopIteration:       # end of epoch: reset and shuffle
                    train_itr_num.reset()
            model.update(batch)
            updates_2 = model.updates
            train_loss_num = model.train_loss.avg
            train_em_num = model.em_avg.avg
            train_f1_num = model.f1_avg.avg

            updates_diff += updates_2 - updates_1
            updates = updates_2 - updates_diff

            if (
                (step + 1) % args.gradient_accumulation_steps
                == 0
                or step == 1
            ):
                logger.info(
                    "Updates[{0:6}], Textual: train loss[{1:.5f}] train em[{2:.5f}] f1[{3:.5f}], Numeric: train loss[{4:.5f}] train em[{5:.5f}] f1[{6:.5f}], remaining[{7}]".format(
                        updates,
                        train_loss_text,
                        train_em_text,
                        train_f1_text,
                        train_loss_num,
                        train_em_num,
                        train_f1_num,
                        str(
                            (datetime.now() - train_start)
                            / (step + 1)
                            * (num_train_steps - step - 1)
                        ).split(".")[0],
                    )
                )
                loss_curr += model.train_loss.sum
                count += model.train_loss.count
                model.avg_reset()

        loss_curr /= count
        total_num_text, eval_loss_text, eval_em_text, eval_f1_text = model.evaluate(dev_itr_text)
        total_num_num, eval_loss_num, eval_em_num, eval_f1_num = model.evaluate(dev_itr_num)

        logger.info(
            "Eval, Textual: {0:6} examples, result in epoch {1:.5f}, eval loss {2:.5f}, eval em {3:.5f} eval f1 {4:.5f}, Numeric: {5:6} examples, result in epoch {6:.5f}, eval loss {7:.5f}, eval em {8:.5f} eval f1 {9:.5f}.".format(
                total_num_text, eval_loss_text, eval_em_text, eval_f1_text, total_num_num, eval_loss_num, eval_em_num, eval_f1_num
            )
        )
        # Like in GenBERT pretraining scheme, update best result according to score on textual data
        if eval_f1_text > best_result[0]:
            save_prefix = os.path.join(args.save_dir, "pretrain_best")
            model.save(save_prefix, epoch)
            best_result[0] = eval_f1_text
            best_result[1] = eval_f1_num
            logger.info(
                "Best textual eval F1 {0:.5f}, with numeric eval F1 {1:.5f}, at epoch {3}".format(
                    best_result[0], best_result[1], epoch
                )
            )
        
        # Stopping criteria
        if abs(loss_curr - loss_prev) < args.eps:
            logger.info("Optimization has converged, at epoch {}".format(epoch))
            break

    logger.info(
        "done training in {} seconds!".format((datetime.now() - train_start).seconds)
    )


if __name__ == "__main__":
    main()
