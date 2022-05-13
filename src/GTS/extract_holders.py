# import stanza
import os
import argparse
import glob
import json
import pickle
import logging
from utils import CURRENT_POJ_DIR, DATA_DIR, dataset2lang
from nltk.tokenize.simple import SpaceTokenizer

from semeval_evaluate import tuple_f1, convert_opinion_to_tuple

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

st = SpaceTokenizer()

def load_stopwords_from_file(path):

    with open(path, 'r' , encoding='utf-8') as f:
        
        stopwords = f.read().splitlines() 

    return stopwords

def get_holder_dict(sentence_pack, stops):

    holder_dict = {}

    for pack in sentence_pack:

        for opinion in pack['opinions']:

            if opinion['Source'][0]:

                holder_words = st.tokenize(opinion['Source'][0][0])

                holder_words = [w.lower() for w in holder_words if w.lower() not in stops]

                for w in holder_words:
                    if w in holder_dict: holder_dict[w] += 1
                    else: holder_dict[w] = 1

    return holder_dict

def extract_holders(sentences, holder_dict, triplets, nlp):

    # Holder could be entity, people, and PRP, POS

    quadruple = {}

    for sent_id in sentences.keys():

        sent_tokenized = st.tokenize(sentences[sent_id])
        holders_in_sent = [sent_tokenized.index(w) for w in sent_tokenized if w.lower() in holder_dict and holder_dict[w.lower()] > 1]

        if holders_in_sent:

            for h in holders_in_sent:

                for triplet in triplets[sent_id]:

                    if sent_id not in quadruple:
                        quadruple[sent_id] = [({h},)+triplet]
                    else:
                        quadruple[sent_id].append(({h},)+triplet)

                    quadruple[sent_id].append((set(),)+triplet)

        else:

            for triplet in triplets[sent_id]:

                quadruple[sent_id] = [(set(),)+triplet]

        if sent_id not in quadruple:

            quadruple[sent_id] = []

        # doc = nlp(sentences[s])
        # print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')
    
    return quadruple

def evaluate(args, quadruple, mode='dev', debug=False):

    gold_file = DATA_DIR / args.dataset / f"{mode}.json"

    with open(gold_file) as f_gold:
        gold = json.load(f_gold)

    gold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])

    all_f_tuples, all_gold_tuples, pre, re, f1 = tuple_f1(gold, quadruple)

    if debug:

        for f, g in zip(all_f_tuples, all_gold_tuples):
            print(f, '\t', g)
        
        print('\n')

    return pre, re, f1

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es"])
    parser.add_argument('--plm_model_name',
                        type=str,
                        default="bert-base-uncased",
                        choices=["bert-base-uncased", "xlm-roberta-large", "ernie_2.0_skep_large_en_pytorch", "roberta-large"],
                        help='Pretrained language model name')
    parser.add_argument('--mode',
                        type=str,
                        default='eval',
                        choices=['eval', 'predict'])
    parser.add_argument('--choose_epoch',
                        type=int,
                        default=None,
                        help='Specify one epoch.')

    parser.add_argument('--model_save_dir',
                        type=str,
                        default=None)
    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    if args.mode == 'eval':

        if args.model_save_dir:
            model_save_dir = args.model_save_dir
        else:
            model_save_dir = CURRENT_POJ_DIR / 'saved_models'

        step1_pred_file_dir = CURRENT_POJ_DIR / 'saved_models' / 'extract' / args.plm_model_name / args.dataset
        lang = dataset2lang[args.dataset]
        if not args.choose_epoch:
            step1_pred_file_list = glob.glob(str(step1_pred_file_dir / "*.pickle"))
        else:
            step1_pred_file_list = glob.glob(str(step1_pred_file_dir / f"*{args.choose_epoch}.pickle"))
        train_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'train.json'))
        stops = load_stopwords_from_file(CURRENT_POJ_DIR / f'stopwords_{lang}.txt')

        pred_save_dir = model_save_dir / 'extract' / args.plm_model_name / args.dataset / 'holder_pred'
        os.makedirs(pred_save_dir / 'best', exist_ok=True)

        holder_dict_save_dir = DATA_DIR / 'holder_dict'
        os.makedirs(holder_dict_save_dir / args.dataset, exist_ok=True)

        holder_dict = get_holder_dict(train_sentence_packs, stops)

        with open(holder_dict_save_dir/ args.dataset / "holder_dict.pickle", "wb") as f_pickle:
            pickle.dump(holder_dict, f_pickle)

        valid_sentence_packs = json.load(open(DATA_DIR / args.dataset / 'dev.json'))

        sentences = {}
        for pack in valid_sentence_packs:
            sentences[pack['sent_id']] = pack['text']
        
        nlp = None

        best_re = 0.0
        best_quadruple = {}
        best_epoch = 0

        for b in step1_pred_file_list:
            filename = b.split("/")[-1]
            with open(b, 'rb') as f_pickle:
                pred_triplets = pickle.load(f_pickle)
            quadruple = extract_holders(sentences, holder_dict, pred_triplets, nlp)
            with open(pred_save_dir / filename , 'wb') as f_pickle:
                pickle.dump(quadruple, f_pickle)
            pre, re, f1 = evaluate(args, quadruple)
            # print(pre, re, f1)
            if re > best_re:
                best_re = re
                best_quadruple = quadruple
                best_epoch = filename.split(".")[0].split("_")[1]
        
        with open(pred_save_dir / 'best' / f"best_holder_pred@{best_epoch}.pickle", "wb") as f_pickle:
            pickle.dump(best_quadruple, f_pickle)
        with open(pred_save_dir / 'best' / f"best_holder_pred@{best_epoch}.txt", "w+") as f_plain:
            for k in best_quadruple.keys():
                f_plain.write(f"{k}:{best_quadruple[k]}\n")
        _, _, _ = evaluate(args, best_quadruple, debug=False)
        
        logger.info(f"Best Pred Precision: {pre:.2f} Recall: {re:.2f} F1: {f1:.2f}")

    else:

        pass
        

if __name__ == '__main__':
    main()