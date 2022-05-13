import os
import argparse
import logging
import torch
import json
import math
import pickle
from torch.utils.data import DataLoader

from progbar import progress_bar, prange
from data import preprocess, ABSADataset, get_max_sequence_len
from utils import DATA_DIR, PLM_DIR, LOG_DIR, CURRENT_POJ_DIR, PRED_DIR, SAVE_DIR, \
                OutputPreprocessor, SemEvalEvaluator, convert_to_submission_json_format

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def evaluate(args, device):

    model_names = args.ensemble_plm_model_names

    logger.info(f"Ensembling {model_names} predictions.")

    ### Very ugly code start, this code segment only want to get len(valid_dataloader)

    args.plm_model_name = 'bert-large-uncased'

    valid_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))
    train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json')) # Only to get max_sequence_length

    args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)

    valid_instances = preprocess(valid_sentence_packs, args)
    valid_dataset = ABSADataset(valid_instances, mode='valid')

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn
    ) # Only to get len(valid_dataloader)

    ensemble_all_logits = [[] for _ in range(len(valid_dataloader))]

    ### Very ugly code end

    for model_name in model_names:

        args.plm_model_name = model_name

        args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)

        valid_instances = preprocess(valid_sentence_packs, args)
        valid_dataset = ABSADataset(valid_instances, mode='valid')

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=valid_dataset.collate_fn
        )

        model_save_path = CURRENT_POJ_DIR / 'saved_models' / 'extract' /  model_name / args.dataset / 'checkpoint_best.pt'

        model = torch.load(model_save_path).to(device)

        model.eval()

        all_labels = []
        all_len_tokens = []
        all_len_plm_tokens = []
        all_token_ranges = []

        for step, batch in enumerate(progress_bar(valid_dataloader, desc="Validation", disable=args.disable_progress_bar)):

            inputs, tags, token_ranges, len_tokens, len_plm_tokens, sent_ids = batch

            inputs = {_: feature.to(device) for _, feature in inputs.items()}

            with torch.no_grad():
                logits = model(inputs) # [batch_size, hidden_size, hidden_size, label_num]

            ensemble_all_logits[step].append(logits.cpu())

            all_labels.append(tags)
            all_len_tokens.extend(len_tokens)
            all_len_plm_tokens.extend(len_plm_tokens)
            all_token_ranges.extend(token_ranges)

    all_preds = []

    for step in range(len(ensemble_all_logits)):

        stacked_step_logits = torch.stack(ensemble_all_logits[step], dim=4)
        mean_step_logits = torch.mean(stacked_step_logits, dim=4)

        preds = torch.argmax(mean_step_logits, dim=3)

        all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0).tolist()
    all_labels = torch.cat(all_labels, dim=0).tolist()

    output_preprocessor = OutputPreprocessor(
        all_preds, 
        all_len_plm_tokens, 
        all_len_tokens, 
        all_token_ranges, 
        ignore_index=-1)
    predicted_tuples = output_preprocessor.get_tuples()

    semeval_evaluator = SemEvalEvaluator(predicted_tuples, args.dataset, 'valid')
    predicted_triplets, predicted_quadtuples = semeval_evaluator.predicted_triplets, semeval_evaluator.predicted_quadtuples
    precision, recall, f1 = semeval_evaluator.get_all_triplet_metrics()

    logger.info("Triplet Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(precision, recall, f1))

    _precision, _recall, _f1 = semeval_evaluator.get_all_quadtuple_metrics()

    logger.info("Quadtuple Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(_precision, _recall, _f1))

    save_dir = SAVE_DIR / 'extract' / 'ensemble' / args.dataset

    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir / "best_pred.pickle", "wb") as f_pickle:
        pickle.dump(predicted_triplets, f_pickle)
    
    with open(save_dir / "best_pred.txt", "w+") as f_plain:
        for k in predicted_triplets.keys():
            f_plain.write(f"{k}:{predicted_triplets[k]}\n")
    
    predicted_json = convert_to_submission_json_format(valid_sentence_packs, predicted_triplets)

    with open(save_dir / "best_pred.json", "w+") as f_json:
        json.dump(predicted_json, f_json)
    
    with open(save_dir / 'best_score.txt', 'w+') as f_score:
        f_score.write(str(f1))

def predict(args, device):

    model_names = args.ensemble_plm_model_names

    logger.info(f"Ensembling {model_names} predictions.")

    args.plm_model_name = 'bert-large-uncased'

    test_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'test.json'))

    valid_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))
    train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json')) # Only to get max_sequence_length

    num_test_steps = math.ceil(len(test_sentence_packs) / args.batch_size)

    ensemble_all_logits = [[] for _ in range(num_test_steps)]

    for model_name in model_names:

        args.plm_model_name = model_name

        args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)

        test_instances = preprocess(test_sentence_packs, args, mode='predict')
        test_dataset = ABSADataset(test_instances, mode='predict')

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=test_dataset.collate_fn
        )

        model_save_path = CURRENT_POJ_DIR / 'saved_models' / 'extract' /  model_name / args.dataset / 'checkpoint_best.pt'

        model = torch.load(model_save_path).to(device)

        model.eval()

        all_len_tokens = []
        all_len_plm_tokens = []
        all_token_ranges = []

        for step, batch in enumerate(progress_bar(test_dataloader, desc="Predicting", disable=args.disable_progress_bar)):

            inputs, token_ranges, len_tokens, len_plm_tokens, sent_ids = batch

            inputs = {_: feature.to(device) for _, feature in inputs.items()}

            with torch.no_grad():
                logits = model(inputs) # [batch_size, hidden_size, hidden_size, label_num]

            ensemble_all_logits[step].append(logits.cpu())

            all_len_tokens.extend(len_tokens)
            all_len_plm_tokens.extend(len_plm_tokens)
            all_token_ranges.extend(token_ranges)
        
        del model
        torch.cuda.empty_cache()

    all_preds = []

    for step in range(len(ensemble_all_logits)):

        stacked_step_logits = torch.stack(ensemble_all_logits[step], dim=4)
        mean_step_logits = torch.mean(stacked_step_logits, dim=4)
        preds = torch.argmax(mean_step_logits, dim=3)

        all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0).tolist()

    output_preprocessor = OutputPreprocessor(
        all_preds, 
        all_len_plm_tokens, 
        all_len_tokens, 
        all_token_ranges, 
        ignore_index=-1)
    predicted_tuples = output_preprocessor.get_tuples()

    semeval_evaluator = SemEvalEvaluator(predicted_tuples, args.dataset, mode='predict')
    predicted_triplets, predicted_quadtuples = semeval_evaluator.predicted_triplets, semeval_evaluator.predicted_quadtuples

    os.makedirs(PRED_DIR / 'ensemble' / args.dataset, exist_ok=True)

    with open(PRED_DIR / 'ensemble' / args.dataset / "step1_pred.pickle", "wb") as f_pickle:
        pickle.dump(predicted_triplets, f_pickle)
    
    with open(PRED_DIR / 'ensemble' / args.dataset / "step1_pred.txt", "w+") as f_plain:
        for k in predicted_triplets.keys():
            f_plain.write(f"{k}:{predicted_triplets[k]}\n")
    
    predicted_json = convert_to_submission_json_format(test_sentence_packs, predicted_triplets)

    with open(PRED_DIR / 'ensemble' / args.dataset / "step1_pred.json", "w+") as f_json:
        json.dump(predicted_json, f_json)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es", \
                                "crosslingual_multibooked_ca", "crosslingual_multibooked_eu", "crosslingual_opener_es"])

    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--ensemble_plm_model_names',
                        nargs='+',
                        required=True)
    # parser.add_argument('--ensemble_weights',
    #                     nargs='+',
    #                     required=True)
    parser.add_argument('--disable_progress_bar',
                        type=bool,
                        default=False,
                        help='Disable progress bar.')
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument("--mode",
                        type=str,
                        default='eval',
                        choices=['eval', 'predict'])

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.mode == 'eval':

        evaluate(args, device)

    elif args.mode == 'predict':

        predict(args, device)

if __name__ == "__main__":

    main()
