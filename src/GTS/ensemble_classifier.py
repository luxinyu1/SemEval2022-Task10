import argparse
import logging
import json
import math
import torch
from torch.utils.data import DataLoader
import pickle
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from utils import DATA_DIR, PRED_DIR, SAVE_DIR
from data import preprocess_classifier, ClassifierDataset
from run_classifier import convert_format
from semeval_evaluate import tuple_f1, convert_opinion_to_tuple

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    
    parser.add_argument('--ensemble_plm_model_names',
                        nargs='+',
                        required=True)

    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The dataset of the text classification task to train on.",
        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es"],
    )

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--mode",
                        type=str,
                        default='eval',
                        choices=['eval', 'predict'])

    args = parser.parse_args()

    return args

def evaluate_end2end(args, flitered_predictions, eval_packs):
    
    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in eval_packs])

    _, _, pre, re, f1 = tuple_f1(gold, flitered_predictions)

    return pre, re, f1

def evaluate(args, device):

    def _find_best_score_of_model(model_name, dataset_name):

        with open(SAVE_DIR / 'QA' / model_name / dataset_name / "best_score.txt", 'r') as f_score:

            best_score = f_score.readline().strip().split("\t")[-1]

        return float(best_score)
    
    step2_dataset_model_map = {
        "opener_en": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
        "mpqa": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
        "darmstadt_unis": ["distilbert-base-uncased-distilled-squad", "roberta-large-squad2"],
        "opener_es": ["xlm-roberta-large-squad2"],
        "multibooked_ca": ["xlm-roberta-large-squad2"],
        "multibooked_eu": ["xlm-roberta-large-squad2"],
        "norec": ["xlm-roberta-large-squad2"],
    }

    best_scores = [_find_best_score_of_model(model_name, args.dataset) for model_name in step2_dataset_model_map[args.dataset]]

    best_choice = best_scores.index(max(best_scores))

    step2_best_model_name = step2_dataset_model_map[args.dataset][best_choice]

    logger.info(f"It looks like the best choice is {step2_best_model_name} for {args.dataset}.")

    step2_pred_file = SAVE_DIR / 'QA' / step2_best_model_name / args.dataset / "best_pred.pickle"
    eval_packs = json.load(open(DATA_DIR / args.dataset / "dev.json")) 

    with open(step2_pred_file, 'rb') as f_pickle:
        pred_quadruples = pickle.load(f_pickle)

    eval_sentences = {}
    for pack in eval_packs:
        eval_sentences[pack['sent_id']] = pack["text"]

    assert len(pred_quadruples) == len(eval_packs) 

    sent_ids = []
    instances = []
    all_pred_quadruples = []

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

    ensemble_models = args.ensemble_plm_model_names

    num_eval_steps = math.ceil(len(instances) / args.batch_size)

    ensemble_all_logits = [[] for _ in range(num_eval_steps)]

    for model in ensemble_models:

        model_save_dir = SAVE_DIR / 'classify' / model / args.dataset

        config = AutoConfig.from_pretrained(model_save_dir, num_labels=2)
        tokenizer = AutoTokenizer.from_pretrained(model_save_dir, additional_special_tokens=["[unused1]"])

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

        model = AutoModelForSequenceClassification.from_pretrained(
            model_save_dir,
            config=config,
        )

        model = model.to(device)

        model.eval()

        for step, batch in enumerate(eval_dataloder):

            batch = batch.to(device)

            with torch.no_grad():
                outputs = model(**batch)
            
            ensemble_all_logits[step].append(outputs.logits)

    all_predictions = []

    for step in range(len(ensemble_all_logits)):

        stacked_step_logits = torch.stack(ensemble_all_logits[step], dim=2)
        mean_step_logits = torch.mean(stacked_step_logits, dim=2)

        predictions = mean_step_logits.argmax(dim=-1)
        predictions = predictions.to('cpu').tolist()

        all_predictions.extend(predictions)

    assert len(all_predictions) == len(all_pred_quadruples) == len(sent_ids)

    for pred, q, sent_id in zip(all_predictions, all_pred_quadruples, sent_ids):

        if pred:
            flitered_predictions[sent_id].append(q)

    pre, re, f1 = evaluate_end2end(args, flitered_predictions, eval_packs)

    logger.info("Quadtuple Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(pre, re, f1))

def predict(args, device):

    pass

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.mode == 'eval':

        evaluate(args, device)

    elif args.mode == 'predict':

        predict(args, device)

if __name__ == "__main__":
    main()
