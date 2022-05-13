import os
import json
import logging
import argparse
import time
import math
import torch
import torch.nn.functional as F
import pickle
import shutil
from torch.utils.data import DataLoader
# use a costomized process bar instead of raw tqdm to prevent a huge mess in docker log file
from progbar import progress_bar, prange
from transformers import AutoConfig, set_seed, get_scheduler
from data import preprocess, ABSADataset
from model import MultiInferModel

from utils import DATA_DIR, PLM_DIR, LOG_DIR, CURRENT_POJ_DIR, PRED_DIR, SAVE_DIR, \
                OutputPreprocessor, SemEvalEvaluator, convert_to_submission_json_format

from data import get_max_sequence_len

logger = logging.getLogger(__name__)


def train(args, device):

    if args.seed is not None:
        set_seed(args.seed)

    # load dataset
    train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json'))
    valid_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))

    args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)
    logger.info(f"args.max_sequence_len is set to {args.max_sequence_len}.")

    # preprocess
    train_instances = preprocess(train_sentence_packs, args)
    valid_instances = preprocess(valid_sentence_packs, args)

    train_dataset = ABSADataset(train_instances, mode='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=args.shuffle,
        collate_fn=train_dataset.collate_fn
    )
    valid_dataset = ABSADataset(valid_instances, mode='valid')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=valid_dataset.collate_fn
    )

    if args.model_save_dir:
        model_save_dir = args.model_save_dir
    else:
        model_save_dir = CURRENT_POJ_DIR / 'saved_models'

    if args.load_from_pretrained:
        if not os.path.exists(model_save_dir / 'extract' / f"{args.plm_model_name}+" / args.dataset):
            os.makedirs(model_save_dir / 'extract' / f"{args.plm_model_name}+" / args.dataset)

        model = torch.load(args.domain_pretained_load_path).to(device)
    else:
        if not os.path.exists(model_save_dir / 'extract' / args.plm_model_name / args.dataset):
            os.makedirs(model_save_dir / 'extract' / args.plm_model_name / args.dataset)

        model = MultiInferModel(args).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.plm.parameters(), 'lr': args.learning_rate},
        {'params': model.cls_linear.parameters()}
    ], lr=args.learning_rate)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    
    args.max_train_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    best_f1 = 0
    best_epoch = 0

    logger.info("***** Running training *****")
    logger.info("  Dataset name = %s", args.dataset)
    logger.info("  PLM name = %s", args.plm_model_name)
    logger.info("  Num training examples = %d", len(train_instances))
    logger.info("  Num validing examples = %d", len(valid_instances))
    logger.info("  lr = %f", args.learning_rate)
    logger.info("  lr_scheduler = %s", args.lr_scheduler)
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num Epochs = %d", args.epochs)

    for epoch in prange(int(args.epochs), desc="Epoch", disable=args.disable_progress_bar):

        epoch_loss = []

        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader, desc="Iteration", disable=args.disable_progress_bar)):

            inputs, tags = batch

            inputs = {_: feature.to(device) for _, feature in inputs.items()}
            tags = tags.to(device)

            preds = model(inputs) # [batch_size, max_length, max_length, label_num]

            preds_flatten = preds.reshape([-1, preds.shape[3]]) # [-1, label_num]

            tags_flatten = tags.reshape([-1]) # [-1]
            
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)
            loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss.append(loss.item())

        logger.info(f"Epoch {epoch+1} loss: {sum(epoch_loss):.2f} Avg step loss: {sum(epoch_loss)/len(epoch_loss):.2f}")

        predicted_tuples, _, _, f1 = evaluate(model, valid_dataloader, device, args)

        if args.load_from_pretrained:
            pred_save_dir = model_save_dir / 'extract' / f"{args.plm_model_name}+" / args.dataset
        else:
            pred_save_dir = model_save_dir / 'extract' / args.plm_model_name / args.dataset
        with open(pred_save_dir / f"epoch_{epoch+1}_pred.pickle", "wb") as f_pickle:
            pickle.dump(predicted_tuples, f_pickle)
        with open(pred_save_dir / f"epoch_{epoch+1}_pred.txt", "w+") as f_plain:
            for k in predicted_tuples.keys():
                f_plain.write(f"{k}:{predicted_tuples[k]}\n")
        
        predicted_json = convert_to_submission_json_format(valid_sentence_packs, predicted_tuples)
        with open(pred_save_dir / f"epoch_{epoch+1}_pred.json", "w+") as f_json:
            json.dump(predicted_json, f_json)

        if f1 > best_f1:
            model_save_path = pred_save_dir / 'checkpoint_best.pt'
            torch.save(model, model_save_path)
            best_f1 = f1
            best_epoch = epoch+1
        
        if epoch == int(args.epochs) - 1 and args.save_last_epoch:
            model_save_path = pred_save_dir / 'checkpoint_last.pt'
            torch.save(model, model_save_path)

    logger.info("***** Training Finished *****")

    shutil.copy(str(pred_save_dir / f"epoch_{best_epoch}_pred.txt"), str(pred_save_dir / f"best_pred.txt"))
    shutil.copy(str(pred_save_dir / f"epoch_{best_epoch}_pred.pickle"), str(pred_save_dir / f"best_pred.pickle"))
    shutil.copy(str(pred_save_dir / f"epoch_{best_epoch}_pred.json"), str(pred_save_dir / f"best_pred.json"))
    with open(pred_save_dir / "best_score.txt", "w+", encoding='utf-8') as f_score:
        f_score.write(str(best_f1))

    logger.info(f"Best Epoch: {best_epoch} | Best F1: {best_f1}")


def evaluate(model, dataloader, device, args, mode='valid'):

    model.eval()

    all_preds = []
    all_labels = []
    all_len_tokens = []
    all_len_plm_tokens = []
    all_token_ranges = []

    logger.info("***** Validation Start *****")

    for step, batch in enumerate(progress_bar(dataloader, desc="Validation", disable=args.disable_progress_bar)):

        inputs, tags, token_ranges, len_tokens, len_plm_tokens, sent_ids = batch

        inputs = {_: feature.to(device) for _, feature in inputs.items()}

        with torch.no_grad():
            preds = model(inputs) # [batch_size, hidden_size, hidden_size, label_num]
        
        preds = torch.argmax(preds, dim=3)
        all_preds.append(preds)
        all_labels.append(tags)
        all_len_tokens.extend(len_tokens)
        all_len_plm_tokens.extend(len_plm_tokens)
        all_token_ranges.extend(token_ranges)

    all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
    all_labels = torch.cat(all_labels, dim=0).cpu().tolist()

    output_preprocessor = OutputPreprocessor(
        all_preds, 
        all_len_plm_tokens, 
        all_len_tokens, 
        all_token_ranges, 
        ignore_index=-1)
    predicted_tuples = output_preprocessor.get_tuples()

    semeval_evaluator = SemEvalEvaluator(predicted_tuples, args.dataset, mode)
    predicted_triplets, predicted_quadtuples = semeval_evaluator.predicted_triplets, semeval_evaluator.predicted_quadtuples
    precision, recall, f1 = semeval_evaluator.get_all_triplet_metrics()

    logger.info("Triplet Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(precision, recall, f1))

    _precision, _recall, _f1 = semeval_evaluator.get_all_quadtuple_metrics()

    logger.info("Quadtuple Precision: {:.2f} Recall: {:.2f} F1: {:.2f}".format(_precision, _recall, _f1))

    return predicted_triplets, precision, recall, f1


def test(args, device):

    logger.info("***** Running Testing *****")

    if args.model_save_dir:
        model_save_dir = args.model_save_dir
    else:
        model_save_dir = CURRENT_POJ_DIR / 'saved_models'

    model_save_path = model_save_dir / 'extract' / args.plm_model_name / args.dataset / 'checkpoint_best.pt'
    model = torch.load(model_save_path).to(device)

    test_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'test.json'))
    test_instances = preprocess(test_sentence_packs, args)
    test_dataset = ABSADataset(test_instances, mode='valid')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=test_dataset.collate_fn
    )
    predicted_triplets, _, _, _ = evaluate(model, test_dataloader, device, args, mode='test')

    predicted_json = convert_to_submission_json_format(test_sentence_packs, predicted_triplets)

def predict(args, device):

    def _find_best_score_of_model(model_name, dataset_name):

        with open(SAVE_DIR / 'extract' / model_name / dataset_name / "best_score.txt", 'r') as f_score:

            best_score = f_score.readline().strip()

        return float(best_score)

    dataset_model_map = {
        "opener_en": ["ensemble", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch", "roberta-large"],
        "mpqa": ["ensemble", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch", "roberta-large"],
        "darmstadt_unis": ["ensemble", "bert-large-uncased", "ernie_2.0_skep_large_en_pytorch", "roberta-large"],
        "opener_es": ["xlm-roberta-large"],
        "multibooked_ca": ["xlm-roberta-large"],
        "multibooked_eu": ["xlm-roberta-large"],
        "norec": ["xlm-roberta-large", "nb-bert-large"],
        "crosslingual_multibooked_ca": ["xlm-roberta-large"],
        "crosslingual_multibooked_eu": ["xlm-roberta-large"],
        "crosslingual_opener_es": ["xlm-roberta-large"]
    }

    best_scores = [_find_best_score_of_model(model_name, args.dataset) for model_name in dataset_model_map[args.dataset]]

    best_choice = best_scores.index(max(best_scores))

    args.plm_model_name = dataset_model_map[args.dataset][best_choice]

    logger.info(f"It looks like the best choice is {args.plm_model_name} for {args.dataset}.")

    if args.plm_model_name != 'ensemble':

        # TODO: Modify this ugly code later

        train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json')) # Only to get max_sequence_length
        valid_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))

        args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)

        #

        model_save_dir = CURRENT_POJ_DIR / 'saved_models'

        if args.crosslingual:
            model_save_path = model_save_dir / 'extract' / args.plm_model_name / args.dataset / 'checkpoint_last.pt'
            model = torch.load(model_save_path).to(device)
        else:
            model_save_path = model_save_dir / 'extract' / args.plm_model_name / args.dataset / 'checkpoint_best.pt'
            model = torch.load(model_save_path).to(device)

        model.eval()

        test_instances = json.load(open(DATA_DIR / args.dataset / 'test.json'))

        predict_instances = preprocess(test_instances, args, mode='predict')
        predict_dataset = ABSADataset(predict_instances, mode='predict')

        predict_dataloader = DataLoader(
            predict_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=predict_dataset.collate_fn
        )

        all_preds = []
        all_len_tokens = []
        all_len_plm_tokens = []
        all_token_ranges = []

        logger.info("***** Running Predicting *****")
        logger.info("  Num testing examples = %d", len(test_instances))

        for step, batch in enumerate(progress_bar(predict_dataloader, desc="Predicting", disable=args.disable_progress_bar)):

            inputs, token_ranges, len_tokens, len_plm_tokens, sent_ids = batch

            inputs = {_: feature.to(device) for _, feature in inputs.items()}

            with torch.no_grad():
                preds = model(inputs) # [batch_size, hidden_size, hidden_size, label_num]
            
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_len_tokens.extend(len_tokens)
            all_len_plm_tokens.extend(len_plm_tokens)
            all_token_ranges.extend(token_ranges)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()

        output_preprocessor = OutputPreprocessor(
            all_preds, 
            all_len_plm_tokens, 
            all_len_tokens, 
            all_token_ranges, 
            ignore_index=-1)
        predicted_tuples = output_preprocessor.get_tuples()

        semeval_evaluator = SemEvalEvaluator(predicted_tuples, args.dataset, mode='predict')

        predicted_triplets, predicted_quadtuples = semeval_evaluator.predicted_triplets, semeval_evaluator.predicted_quadtuples

        os.makedirs(PRED_DIR / args.dataset, exist_ok=True)

        with open(PRED_DIR / args.dataset / "step1_pred.pickle", "wb") as f_pickle:
            pickle.dump(predicted_triplets, f_pickle)
        
        with open(PRED_DIR / args.dataset / "step1_pred.txt", "w+") as f_plain:
            for k in predicted_triplets.keys():
                f_plain.write(f"{k}:{predicted_triplets[k]}\n")
        
        predicted_json = convert_to_submission_json_format(test_instances, predicted_triplets)

        with open(PRED_DIR / args.dataset / "step1_pred.json", "w+") as f_json:
            json.dump(predicted_json, f_json)

        with open(PRED_DIR / args.dataset / "step1.log", "w+") as f_log:
            f_log.write(f"Step 1: {args.plm_model_name} is used for prediction. Because it has highest triplet f1 {max(best_scores)} in evaluation.")

    else:

        shutil.copytree(PRED_DIR / 'ensemble' / args.dataset, PRED_DIR / args.dataset, dirs_exist_ok=True)

        with open(PRED_DIR / args.dataset / "step1.log", "w+") as f_log:
            f_log.write(f"Step 1: {args.plm_model_name} is used for prediction. Because it has highest triplet f1 {max(best_scores)} in evaluation.")

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es", \
                                "crosslingual_multibooked_ca", "crosslingual_multibooked_eu", "crosslingual_opener_es", \
                                "opener_en+", "mpqa+", "darmstadt_unis+"])
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        choices=["train", "test", "predict"],
                        help='Options: train, test')
    
    parser.add_argument('--load_from_pretrained',
                        type=bool,
                        default=False)
    parser.add_argument('--domain_pretained_load_path',
                        type=str,
                        default=None)
                        
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--plm_model_name',
                        type=str,
                        default="bert-base-uncased",
                        choices=["bert-base-uncased", "xlm-roberta-large", "ernie_2.0_skep_large_en_pytorch", \
                            "bert-large-uncased", "calbert-base-uncased", "berteus-base-cased", \
                            "bert-base-multilingual-uncased-sentiment", "roberta-large", "roberta-large-bne", "all-roberta-large-v1", \
                            "nb-bert-large", "bert-large-uncased-domain-pretrained-only-europarl", "LaBSE"],
                        help='pretrained language model name')
    parser.add_argument('--max_sequence_len', 
                        type=int, 
                        default=128,
                        help='max length of a sentence')
    parser.add_argument('--model_save_dir',
                        type=str,
                        default=None)
    parser.add_argument('--batch_size',
                        type=int,
                        default=8)
    parser.add_argument('--seed',
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument('--shuffle',
                        type=bool,
                        default=True,
                        help="Whether shuffle the training set or not.")

    parser.add_argument('--nhops',
                        type=int,
                        default=1,
                        help='Inference times')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=3e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_warmup_steps",
                        type=int,
                        default=0,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="constant",
                        help="The scheduler to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='training epoch number')
    parser.add_argument('--class_num',
                        type=int,
                        default=8,
                        help='label number')

    parser.add_argument('--docker_mode',
                        type=bool,
                        default=False,
                        help='Set this to true when running in docker.')
    parser.add_argument('--disable_progress_bar',
                        type=bool,
                        default=False,
                        help='Disable progress bar.')
    parser.add_argument('--save_last_epoch',
                        type=bool,
                        default=False,
                        help='Save last epoch checkpoint.')
    parser.add_argument('--crosslingual',
                        type=bool,
                        default=False)

    args = parser.parse_args()

    # Sanity Check

    if args.load_from_pretrained and not args.domain_pretained_load_path:

        return ValueError("Please provide the path to domain pretrained model when load from pretrained!")

    return args

def main():

    args = parse_args()

    if args.docker_mode:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            filename=LOG_DIR/(str(time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))+".log"),
            filemode='w'
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.mode == 'train':
        train(args, device)
    elif args.mode == 'test':
        test(args, device)
    elif args.mode == 'predict':
        predict(args, device)

if __name__ == '__main__':

    main()