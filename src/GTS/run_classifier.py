# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for quadruple classification."""
import argparse
import logging
import math
import os
import glob
import pickle
import json
import random
from pathlib import Path

import datasets
from datasets import load_dataset, load_metric
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

from data import preprocess_classifier, ClassifierDataset
from utils import DATA_DIR, CURRENT_POJ_DIR, PLM_DIR, PRED_DIR, RESOURCES_DIR, SAVE_DIR, convert_to_submission_json_format
from semeval_evaluate import tuple_f1, convert_opinion_to_tuple

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The dataset of the text classification task to train on.",
        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es",\
                "crosslingual_opener_es", "crosslingual_multibooked_ca", "crosslingual_multibooked_eu"],
    )
    parser.add_argument(
        '--plm_model_name',
        type=str,
        default="bert-base-uncased",
        choices=["bert-base-uncased", "bert-large-uncased", "xlm-roberta-large", "ernie_2.0_skep_large_en_pytorch", "roberta-large", "bert-base-multilingual-cased", "nb-bert-base", "nb-bert-large", "LaBSE"],
        help='Pretrained language model name'
    )
    parser.add_argument(
        "--negative_sampling_size",
        type=int,
        default=5,
        help="Randomly Sampling how many negative samples when there's no gold opinion."
    )
    parser.add_argument(
        "--positive_sampling_size",
        type=int,
        default=20,
        help='Sampling how many positive samples per instance(opinion). Notice that '
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size of dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--model_save_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=1, help="A seed for reproducible training.")
    parser.add_argument("--do_train",
        type=bool,
        default=False
    )
    parser.add_argument("--do_eval",
        type=bool,
        default=False
    )
    parser.add_argument("--use_last_epoch",
        type=bool,
        default=False
    )
    parser.add_argument("--save_last_epoch",
        type=bool,
        default=False
    )
    parser.add_argument("--do_predict",
        type=bool,
        default=False
    )
    parser.add_argument("--crosslingual",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    # Sanity Check

    if not args.do_train and not args.do_predict and not args.do_eval:
        raise ValueError("At leaset one of args.do_train or args.do_eval or args.do_predict should be true.")

    return args

def train(args, accelerator):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.model_save_dir:
        model_save_dir = args.model_save_dir
    else:
        model_save_dir = CURRENT_POJ_DIR / 'saved_models' / 'classify' / args.plm_model_name / args.dataset

    if args.crosslingual:
        os.makedirs(model_save_dir / "best", exist_ok=True)
        os.makedirs(model_save_dir / "last", exist_ok=True)
    else:
        os.makedirs(model_save_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    dataset_dir_path = str(DATA_DIR / args.dataset)
    logging.info(f"Loading dataset from {dataset_dir_path}")
    
    train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json'))
    eval_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))

    train_num_positive, train_num_negative, train_instances = preprocess_classifier(train_sentence_packs, args)
    logger.info(f"{train_num_positive} positive samples and {train_num_negative} negative samples in training set.")
    eval_num_positive, eval_num_negative, eval_instances = preprocess_classifier(eval_sentence_packs, args)
    logger.info(f"{eval_num_positive} negative samples and {eval_num_negative} negative samples in evaluation set.")

    plm_model_dir = PLM_DIR / args.plm_model_name

    config = AutoConfig.from_pretrained(plm_model_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(plm_model_dir, additional_special_tokens=["[unused1]"])
    model = AutoModelForSequenceClassification.from_pretrained(
        plm_model_dir,
        config=config,
    )

    train_dataset = ClassifierDataset(train_instances, tokenizer, args)
    eval_dataset = ClassifierDataset(eval_instances, tokenizer, args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=eval_dataset.collate_fn,
        shuffle=True,
    )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Get the metric function
    metric = load_metric(str(RESOURCES_DIR / 'huggingface' / 'metrics' / 'accuracy'))

    # Train!
    assert accelerator.num_processes == 1
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  batch size = {args.batch_size}")
    logger.info(f"  Dataset = {args.dataset}")
    logger.info(f"  PLM Model Name = {args.plm_model_name}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_acc = 0

    for epoch in range(args.num_train_epochs):

        model.train()
        
        epoch_loss = []
        
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            epoch_loss.append(loss.item())

        logger.info(f"Epoch {epoch+1} loss: {sum(epoch_loss):.2f} Avg step loss: {sum(epoch_loss)/len(epoch_loss):.2f}")

        model.eval()
        for step, batch in enumerate(eval_dataloader):

            with torch.no_grad():
                outputs = model(**batch)
            
            predictions = outputs.logits.argmax(dim=-1)
            # print(predictions)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(batch["labels"]),
            )

        eval_metric = metric.compute()
        acc = eval_metric['accuracy']
        logger.info(f"Epoch {epoch+1}: Acc: {acc:.2f}")

        if acc > best_acc:

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            if args.crosslingual:
                unwrapped_model.save_pretrained(model_save_dir / 'best', save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(model_save_dir / 'best', save_function=accelerator.save)
            else:
                unwrapped_model.save_pretrained(model_save_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(model_save_dir, save_function=accelerator.save)

            best_acc = acc

    if args.save_last_epoch:

        if args.crosslingual:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(model_save_dir / 'last', save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(model_save_dir / 'last')
        else:
            unwrapped_model.save_pretrained(model_save_dir, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(model_save_dir, save_function=accelerator.save)

def convert_format(quadruple):

    holder_span = [min(quadruple[0]), max(quadruple[0])] if quadruple[0] else []
    aspect_span = [min(quadruple[1]), max(quadruple[1])] if quadruple[1] else []
    expression_span = [min(quadruple[2]), max(quadruple[2])] if quadruple[2] else []

    return (holder_span, aspect_span, expression_span)

def evaluate(args, device):

    if args.model_save_dir:
        model_save_dir = args.model_save_dir
    else:
        model_save_dir = CURRENT_POJ_DIR / 'saved_models' / 'classify' / args.plm_model_name / args.dataset

    def _find_best_score_of_model(model_name, dataset_name):

        with open(SAVE_DIR / 'QA' / model_name / dataset_name / "best_score.txt", 'r') as f_score:

            best_score = f_score.readline().strip().split("\t")[-1]

        return float(best_score)

    if args.crosslingual:

        pred_file = SAVE_DIR / "extract" / "xlm-roberta-large" / args.dataset / "best_pred.pickle"

    else:
    
        dataset_model_map = {
            "opener_en": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
            "mpqa": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
            "darmstadt_unis": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
            "opener_es": ["xlm-roberta-large-squad2"],
            "multibooked_ca": ["xlm-roberta-large-squad2"],
            "multibooked_eu": ["xlm-roberta-large-squad2"],
            "norec": ["xlm-roberta-large-squad2"],
        }

        best_scores = [_find_best_score_of_model(model_name, args.dataset) for model_name in dataset_model_map[args.dataset]]

        best_choice = best_scores.index(max(best_scores))

        step2_best_model_name = dataset_model_map[args.dataset][best_choice]

        logger.info(f"It looks like the best choice is {step2_best_model_name} for {args.dataset}.")

        pred_file = SAVE_DIR / 'QA' / step2_best_model_name / args.dataset / "best_pred.pickle"
    
    eval_packs = json.load(open(DATA_DIR / args.dataset / "dev.json"))
    
    eval_sentences = {}
    for pack in eval_packs:
        eval_sentences[pack['sent_id']] = pack["text"]

    config = AutoConfig.from_pretrained(model_save_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_save_dir, additional_special_tokens=["[unused1]"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_save_dir,
        config=config,
    )

    model = model.to(device)

    model.eval()

    with open(pred_file, 'rb') as f_pickle:
        pred_quadruples = pickle.load(f_pickle)

    if args.crosslingual:

        for key in pred_quadruples.keys():
            quadtuples = []
            for tuple in pred_quadruples[key]:
                quadtuples.append((set(),)+tuple)
            pred_quadruples[key] = quadtuples

    sent_ids = []
    instances = []
    all_pred_quadruples = []

    assert len(pred_quadruples) == len(eval_packs) 

    flitered_predictions = {}

    for sent_id in eval_sentences.keys():
        if sent_id not in flitered_predictions:
            flitered_predictions[sent_id] = []
        sentence = eval_sentences[sent_id]
        quadruples = pred_quadruples[sent_id]
        all_pred_quadruples.extend(quadruples)
        if quadruples:
            for q in quadruples:
                _q = convert_format(q)
                instances.append((sentence,)+_q)
                sent_ids.append(sent_id)

    eval_dataset = ClassifierDataset(
        instances,
        tokenizer,
        args,
        mode='predict'
    )

    eval_dataloder = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
    )

    all_predictions = []

    for step, batch in enumerate(eval_dataloder):

        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1)

        predictions = predictions.to('cpu').tolist()
        all_predictions.extend(predictions)

    assert len(all_predictions) == len(all_pred_quadruples) == len(sent_ids)

    for pred, q, sent_id in zip(all_predictions, all_pred_quadruples, sent_ids):

        if pred:
            flitered_predictions[sent_id].append(q)

    pre, re, f1 = end2end_performance(flitered_predictions, eval_packs)

    logger.info("Quadtuple Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(pre, re, f1))

    predicted_json = convert_to_submission_json_format(eval_packs, flitered_predictions, mode="quadtuple")

    with open(SAVE_DIR / "classify" / args.plm_model_name / args.dataset / "best_score.txt", "w+") as f_score:
        f_score.write(f"{pre}\t{re}\t{f1}")

    with open(SAVE_DIR / "classify" / args.plm_model_name / args.dataset / "best_pred.json", "w+") as f_json:
        json.dump(predicted_json, f_json)


def end2end_performance(flitered_predictions, eval_packs):
    
    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in eval_packs])

    _, _, pre, re, f1 = tuple_f1(gold, flitered_predictions)

    return pre, re, f1

def predict(args, device):

    def _find_best_score_of_model(model_name, dataset_name):

        with open(SAVE_DIR / 'classify' / model_name / dataset_name / "best_score.txt", 'r') as f_score:

            best_score = f_score.readline().strip().split('\t')[-1]

        return float(best_score)

    if args.crosslingual:

        args.plm_model_name = "LaBSE"

        pred_file = PRED_DIR / args.dataset / 'step1_pred.pickle'

        score = _find_best_score_of_model(args.plm_model_name, args.dataset)
    
    else:

        dataset_model_map = {
            "opener_en": ["bert-base-uncased", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch", "LaBSE"],
            "mpqa": ["bert-base-uncased", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch"],
            "darmstadt_unis": ["bert-base-uncased", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch"],
            "opener_es": ["bert-base-multilingual-cased", "LaBSE"],
            "multibooked_ca": ["bert-base-multilingual-cased", "LaBSE"],
            "multibooked_eu": ["bert-base-multilingual-cased", "LaBSE"],
            "norec": ["bert-base-multilingual-cased", "nb-bert-base", "nb-bert-large", "LaBSE"],
        }

        best_scores = [_find_best_score_of_model(model_name, args.dataset) for model_name in dataset_model_map[args.dataset]]

        best_choice = best_scores.index(max(best_scores))

        args.plm_model_name = dataset_model_map[args.dataset][best_choice]

        logger.info(f"It looks like the best choice is {args.plm_model_name} for {args.dataset}.")

        pred_file = PRED_DIR / args.dataset / 'step2_pred.pickle'

    model_save_dir = CURRENT_POJ_DIR / 'saved_models' / 'classify' / args.plm_model_name / args.dataset

    test_packs = json.load(open(DATA_DIR / args.dataset / 'test.json'))

    test_sentences = {}
    for pack in test_packs:
        test_sentences[pack['sent_id']] = pack["text"]

    config = AutoConfig.from_pretrained(model_save_dir, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_save_dir, additional_special_tokens=["[unused1]"])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_save_dir,
        config=config,
    )

    model = model.to(device)

    model.eval()

    with open(pred_file, 'rb') as f_pickle:
        pred_quadruples = pickle.load(f_pickle)

    if args.crosslingual:

        for key in pred_quadruples.keys():
            quadtuples = []
            for tuple in pred_quadruples[key]:
                quadtuples.append((set(),)+tuple)
            pred_quadruples[key] = quadtuples

    sent_ids = []
    instances = []
    all_pred_quadruples = []

    assert len(pred_quadruples) == len(test_packs) 

    flitered_predictions = {}

    for sent_id in test_sentences.keys():
        if sent_id not in flitered_predictions:
            flitered_predictions[sent_id] = []
        sentence = test_sentences[sent_id]
        quadruples = pred_quadruples[sent_id]
        all_pred_quadruples.extend(quadruples)
        if quadruples:
            for q in quadruples:
                _q = convert_format(q)
                instances.append((sentence,)+_q)
                sent_ids.append(sent_id)

    test_dataset = ClassifierDataset(
        instances,
        tokenizer,
        args,
        mode='predict'
    )

    test_dataloder = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
        shuffle=False,
    )

    all_predictions = []

    for step, batch in enumerate(test_dataloder):

        batch = batch.to(device)

        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = outputs.logits.argmax(dim=-1)

        predictions = predictions.to('cpu').tolist()
        all_predictions.extend(predictions)

    assert len(all_predictions) == len(all_pred_quadruples) == len(sent_ids)

    for pred, q, sent_id in zip(all_predictions, all_pred_quadruples, sent_ids):

        if pred:
            flitered_predictions[sent_id].append(q)
    
    predicted_json = convert_to_submission_json_format(test_packs, flitered_predictions, mode="quadtuple")

    with open(PRED_DIR / args.dataset / f"step3_pred.json", "w+") as f_json:
        json.dump(predicted_json, f_json)

    with open(PRED_DIR / args.dataset / "step3.log", "w+") as f_log:
        if not args.crosslingual:
            f_log.write(f"Step 3: {args.plm_model_name} is used for prediction. Because it has highest f1 {max(best_scores)} in evaluation.")
        else:
            f_log.write(f"Step 3: f1 {score} in evaluation.")

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.do_train:
        train(args, accelerator)
    elif args.do_eval:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        evaluate(args, accelerator.device)
    elif args.do_predict:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        predict(args, accelerator.device)

if __name__ == "__main__":
    main()