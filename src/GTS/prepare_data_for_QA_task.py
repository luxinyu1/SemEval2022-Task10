import os
import nltk
import json
import glob
import pickle
import logging
import shutil
import argparse

from utils import DATA_DIR, CURRENT_POJ_DIR, QA_DATA_DIR, SAVE_DIR, PRED_DIR, dataset2lang

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

language_data_map = {
    'en': ['opener_en', 'darmstadt_unis', 'mpqa'],
    'es': ['opener_es'],
    'ca': ['multibooked_ca'],
    'eu': ['multibooked_eu'],
    'nb': ['norec'],
}

from nltk.tokenize.simple import SpaceTokenizer

st = SpaceTokenizer()

class QAInstance:

    def __init__(self, instance_id, text, tuple, lang, mode='train'):

        self.text = text
        self.tuple = tuple
        self.lang = lang

        self.mode = mode

        self.has_holder = len(tuple[0])>0

        if mode=='train':
            self.answer, self.question = self.build_qa_from_text_and_tuple()
        else:
            self.question = self.build_question_from_step1_result()

        self.error_token = "[unused0]"

        # self.context = self.add_error_token_before_text()
        self.context = text
        self.id = instance_id

    def build_qa_from_text_and_tuple(self):
        
        aspect_start_idx = int(self.tuple[1][0].split(":")[0]) if self.tuple[1] else None
        aspect_end_idx = int(self.tuple[1][0].split(":")[1]) if self.tuple[1] else None
        opinion_start_idx = int(self.tuple[2][0].split(":")[0]) if self.tuple[2] else None
        opinion_end_idx = int(self.tuple[2][0].split(":")[1]) if self.tuple[2] else None

        aspect_expression = self.text[aspect_start_idx:aspect_end_idx] if aspect_start_idx and aspect_end_idx else "empty"
        opinion_expression = self.text[opinion_start_idx:opinion_end_idx] if opinion_start_idx and opinion_end_idx else "empty"

        question = {
            "en": f"What is the holder given the aspect {aspect_expression} and the opinion {opinion_expression} ?",
            "eu": f"Zein da helburu {aspect_expression} eta {opinion_expression} iritzia emanda iritzia duenak ?",
            "ca": f"Quin és el titular de l'opinió donat l'aspecte {aspect_expression} i l'opinió {opinion_expression} ?",
            "es": f"¿Cuál es el titular de la opinión dado el aspecto {aspect_expression} y ​​la opinión {opinion_expression} ?",
            "nb": f"Hva er meningshaveren gitt aspektet {aspect_expression} og meningen {opinion_expression} ?"
        }

        if self.mode=='train':

            holder_start_idx = int(self.tuple[0][0].split(":")[0]) if self.tuple[0] else None
            holder_end_idx = int(self.tuple[0][0].split(":")[1]) if self.tuple[0] else None

            holder_expressinon = self.text[holder_start_idx:holder_end_idx] if holder_start_idx and holder_end_idx else []

            if holder_expressinon:

                answer = [{
                    "text": holder_expressinon,
                    "answer_start": holder_start_idx # Note that there's "[unused0]" and a blank space, thus we +10 <- This has been removed
                }]

            else:

                answer = []

            return answer, question[self.lang]
        
        elif self.mode=='predict':

            return question[self.lang]

    def build_question_from_step1_result(self):

        text = st.tokenize(self.text)

        # print(text)
        # print(self.tuple)

        aspect_start_idx = min(self.tuple[0]) if self.tuple[0] else None
        aspect_end_idx = max(self.tuple[0]) if self.tuple[0] else None

        opinion_start_idx = min(self.tuple[1]) if self.tuple[1] else None
        opinion_end_idx = max(self.tuple[1]) if self.tuple[1] else None

        aspect_expression = text[aspect_start_idx:aspect_end_idx+1] if aspect_start_idx and aspect_end_idx else ["empty"]
        opinion_expression = text[opinion_start_idx:opinion_end_idx+1] if opinion_start_idx and opinion_end_idx else ["empty"]

        aspect_expression = " ".join(aspect_expression)
        opinion_expression = " ".join(opinion_expression)

        question = {
            "en": f"What is the holder given the aspect {aspect_expression} and the opinion {opinion_expression} ?",
            "eu": f"Zein da helburu {aspect_expression} eta {opinion_expression} iritzia emanda iritzia duenak ?",
            "ca": f"Quin és el titular de l'opinió donat l'aspecte {aspect_expression} i l'opinió {opinion_expression} ?",
            "es": f"¿Cuál es el titular de la opinión dado el aspecto {aspect_expression} y ​​la opinión {opinion_expression} ?",
            "nb": f"Hva er meningshaveren gitt aspektet {aspect_expression} og meningen {opinion_expression} ?"
        }

        return question[self.lang]

    def add_error_token_before_text(self):

        return f"{self.error_token} {self.text}"

    def convert_to_squad_format(self):
        """
        {
            "title": "Super_Bowl_50",
            "paragraphs": [
                {
                    "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                }
                            ],
                            "question": "Which NFL team represented the AFC at Super Bowl 50?",
                            "id": "56be4db0acb8001400a502ec"
                        },
        """

        if self.mode == 'train':

            return {
                "paragraphs":[
                    {
                    "context": self.context,
                    "qas":[{
                        "answers": self.answer,
                        "question": self.question,
                        "id": self.id,
                    }]
                    }
                ],
            }
        
        else:

            return {
                "paragraphs":[
                    {
                    "context": self.context,
                    "qas":[{
                        "answers": [],
                        "question": self.question,
                        "id": self.id,
                    }]
                    }
                ],
            }

def get_instance_from_dataset(dataset, lang):

    train_instances = []
    eval_instances = []

    num_train_instance_with_holder = 0
    num_train_instance_wo_holder = 0
    num_eval_instance_with_holder = 0
    num_eval_instance_wo_holder = 0

    train_data_path = DATA_DIR / dataset

    train_sentence_packs = json.load(open(DATA_DIR / dataset / 'train.json'))
    eval_sentence_packs = json.load(open(DATA_DIR / dataset / 'dev.json'))

    for split in ['train', 'eval']:

        for pack in eval(f"{split}_sentence_packs"):

            sent_id = pack['sent_id']
            text = pack["text"]

            for n, opinion in enumerate(pack['opinions']):

                source_span = opinion['Source'][1] 
                target_span = opinion['Target'][1]
                expression_span = opinion['Polar_expression'][1]

                tuple = [source_span, target_span, expression_span]

                instance = QAInstance(f"{sent_id}_{n+1}", text, tuple, lang)

                if split == 'train':
                    if instance.has_holder:
                        num_train_instance_with_holder += 1
                    else:
                        num_train_instance_wo_holder += 1
                else:
                    if instance.has_holder:
                        num_eval_instance_with_holder += 1
                    else:
                        num_eval_instance_wo_holder += 1

                squad_json = instance.convert_to_squad_format()

                eval(f"{split}_instances").append(squad_json)

    return train_instances, eval_instances, num_train_instance_with_holder, num_train_instance_wo_holder, num_eval_instance_with_holder, num_eval_instance_wo_holder

def prepare_data_for_lang(lang):

    all_training_instances = []
    all_valid_instances = []

    datasets = language_data_map[lang]

    for dataset in datasets:

        train_instances, valid_instances, num_train_instance_with_holder, num_train_instance_wo_holder, num_eval_instance_with_holder, num_eval_instance_wo_holder = get_instance_from_dataset(dataset, lang)

        logger.info(f"Training set: with holder:{num_train_instance_with_holder} w/o_holder:{num_train_instance_wo_holder} \
                     Evaluation set: with holder:{num_eval_instance_with_holder} w/o holder:{num_eval_instance_wo_holder}")

        all_training_instances.extend(train_instances)
        all_valid_instances.extend(valid_instances)

    return all_training_instances, all_valid_instances

def prepare_data_for_dataset(dataset, lang):

    train_instances, valid_instances, num_train_instance_with_holder, num_train_instance_wo_holder, num_eval_instance_with_holder, num_eval_instance_wo_holder = get_instance_from_dataset(dataset, lang)

    logger.info(f"Training set: with holder:{num_train_instance_with_holder} w/o_holder:{num_train_instance_wo_holder} \
                    Evaluation set: with holder:{num_eval_instance_with_holder} w/o holder:{num_eval_instance_wo_holder}")

    return train_instances, valid_instances

def prepare_data_for_step2(step1_dataset, mode='eval', lang='en'):

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
    }

    best_scores = [_find_best_score_of_model(model_name, step1_dataset) for model_name in dataset_model_map[step1_dataset]]

    best_choice = best_scores.index(max(best_scores))

    logger.info(f"It looks like the best choice is {dataset_model_map[step1_dataset][best_choice]}")

    if mode=='eval':

        step1_pred_file_dir = CURRENT_POJ_DIR / 'saved_models' / 'extract' / dataset_model_map[step1_dataset][best_choice] / step1_dataset
        step1_pred_file_path = str(step1_pred_file_dir / "best_pred.pickle")

        instance_packs = json.load(open(DATA_DIR / step1_dataset / 'dev.json'))

    elif mode=='predict':

        step1_pred_file_path = PRED_DIR / step1_dataset / "step1_pred.pickle"

        instance_packs = json.load(open(DATA_DIR / step1_dataset / 'test.json'))

    sentence_packs = {s["sent_id"]: s['text'] for s in instance_packs}

    all_instances = []

    with open(step1_pred_file_path, 'rb') as f_pickle:

        pred = pickle.load(f_pickle)

    for key in pred.keys():

        text = sentence_packs[key]
        tuples = pred[key]

        for n, tuple in enumerate(tuples):

            instance = QAInstance(f"{key}?{n}", text, tuple, lang, mode)
            squad_json = instance.convert_to_squad_format()
            all_instances.append(squad_json)

    return all_instances, step1_pred_file_path

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prepare_for_training",
        action='store_true',
    )

    parser.add_argument(
        "--prepare_for_step2",
        action='store_true',
    )

    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "predict"],
    )

    parser.add_argument(
        "--step1_dataset",
        default="opener_en",
        type=str,
        choices=["darmstadt_unis", "mpqa", "multibooked_ca", "multibooked_eu", "norec", "opener_en", "opener_es"]
    )

    args = parser.parse_args()

    if not args.prepare_for_training and not args.prepare_for_step2:
        raise ValueError("At leaset one of args.prepare_for_training and args.prepare_for_step2 should be true.")

    return args

def main():

    args = parse_args()

    if args.prepare_for_training:

        for lang in ['en', 'es', 'ca', 'eu', 'nb']:

            for dataset in language_data_map[lang]:

                all_training_instances, all_valid_instances = prepare_data_for_dataset(dataset, lang)

                save_dir = QA_DATA_DIR / dataset

                os.makedirs(save_dir, exist_ok=True)

                with open(save_dir / "train.json", "w+") as f_json:

                    json.dump({
                        "data": all_training_instances
                    }, f_json)

                # with open(save_dir / "valid.json", "w+") as f_json:

                #     json.dump({
                #         "data": all_valid_instances
                #     }, f_json)

                # Very akward owing to the stupid load method of huggingface dataset :(

                shutil.copyfile(QA_DATA_DIR / "squad.py", save_dir / f"{dataset}.py")
    
    elif args.prepare_for_step2:

        lang = dataset2lang[args.step1_dataset]
        all_instances, step1_pred_file_path = prepare_data_for_step2(args.step1_dataset, args.mode, lang)

        save_dir = QA_DATA_DIR / args.step1_dataset
        os.makedirs(save_dir, exist_ok=True)

        if args.mode=="eval":

            shutil.copyfile(step1_pred_file_path, QA_DATA_DIR / args.step1_dataset / "best_pred.pickle")

            with open(save_dir / "valid.json", "w+") as f_json:

                json.dump({
                    "data": all_instances
                }, f_json)
            

        elif args.mode == 'predict':
        
            with open(save_dir / "test.json", "w+") as f_json:

                json.dump({
                    "data": all_instances
                }, f_json)

if __name__ == "__main__":
    main()