from dataclasses import dataclass
import time
from fastapi import FastAPI
import torch
import os
from pydantic import BaseModel
from typing import List, Optional
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, QuestionAnsweringPipeline
from dataclasses import dataclass
from nltk.tokenize.simple import SpaceTokenizer
st = SpaceTokenizer()

import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = 'cuda'

sys.path.append('src/GTS/')

from src.GTS.utils import CURRENT_POJ_DIR, PLM_DIR, OutputPreprocessor, id2sentiment
from src.GTS.data import ABSADataset, GTSPreprocessor

max_sequence_length = {
    "roberta-large-opener_en": 133,
    "xlm-roberta-large-opener_es": 193,
}
null_threshold = 1e-8

model_list = []
plm_model_name_list = []
tokenizer_list = []
msl = None
lang = None

@dataclass
class QuestionTemplate:

    lang: str
    aspect_term: str
    opinion_term: str

    def get_question(self):

        aspect_term = self.aspect_term
        opinion_term = self.opinion_term

        question_map = {
            "en": f"What is the holder given the aspect {aspect_term} and the opinion {opinion_term} ?",
            "eu": f"Zein da helburu {aspect_term} eta {opinion_term} iritzia emanda iritzia duenak ?",
            "ca": f"Quin és el titular de l'opinió donat l'aspecte {aspect_term} i l'opinió {opinion_term} ?",
            "es": f"¿Cuál es el titular de la opinión dado el aspecto {aspect_term} y ​​la opinión {opinion_term} ?",
            "nb": f"Hva er meningshaveren gitt aspektet {aspect_term} og meningen {opinion_term} ?"
        }

        return question_map[self.lang]

app = FastAPI()

class Item(BaseModel):
    texts: List[str] = []

@app.get("/")
def read_root():
    return {"message": "use /docs"}

@app.post("/reprepare/")
def reprepare(language:str, domain:str):
    global model_list
    del model_list
    prepare_models_and_tokenizers(language, domain)
    return {"language": language, "domain": domain}

@app.post("/api/")
def extract(item: Item):

    text_list = item.texts

    start_time = time.time()
    step1_inputs = convert_to_step1_input(text_list, tokenizer_list[0])

    step1_predicted_tuple = step1_predict(model_list[0], step1_inputs)

    step1_visualized_tuples, questions, instance_mapping = convert_to_step2_input(step1_predicted_tuple, text_list)

    new_text_list = [text_list[i] for i in instance_mapping]

    end_time = time.time()

    print(f"Step 1 costs {round(end_time - start_time, 2)} s")

    start_time = time.time()

    packed_res = step2_predict(model_list[1], questions, new_text_list, step1_visualized_tuples, instance_mapping)

    end_time = time.time()

    print(f"Step 2 costs {round(end_time - start_time, 2)} s")

    return {"res": packed_res}

def convert_to_step1_input(text_list, tokenizer):

    converted_input = []
    sent_ids = list(range(len(text_list)))
    for sent_id, text in zip(sent_ids, text_list):
        sentence_pack = {"sent_id": sent_id, "text": text}
        converted_input.append(GTSPreprocessor(
            tokenizer, 
            sentence_pack, 
            msl,
            plm_model_name_list[0],
            'predict'))

    return converted_input

def step1_predict(model, step1_inputs):

    step1_dataset = ABSADataset(step1_inputs, mode='predict')
    step1_dataloader = DataLoader(
        step1_dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=step1_dataset.collate_fn
    )

    all_preds = []
    all_len_tokens = []
    all_len_plm_tokens = []
    all_token_ranges = []

    for step, batch in enumerate(step1_dataloader):

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

    return predicted_tuples


def convert_to_step2_input(step1_predict_tuples, text_list):

    def offset_to_string(start_idx, end_idx, text):
        tokens = st.tokenize(text)
        return " ".join(tokens[start_idx:end_idx+1:])
    
    questions = []
    instance_mapping = []
    visualized_tuples = []

    for idx, (predict_tuples, text) in enumerate(zip(step1_predict_tuples, text_list)):
        v_t = []
        for tuple in predict_tuples:
            aspect_term = offset_to_string(
                tuple[0]-1,
                tuple[1]-1,
                text)
            opinion_term = offset_to_string(
                tuple[2]-1,
                tuple[3]-1,
                text)
            if not aspect_term:
                aspect_term = 'empty'
            if not opinion_term:
                opinion_term = 'empty'
            polarity = id2sentiment[tuple[4]]

            template = QuestionTemplate(lang, aspect_term, opinion_term)
            question = template.get_question()
            v_t.append((aspect_term, opinion_term, polarity))
            questions.append(question)
            instance_mapping.append(idx)

        visualized_tuples.append(v_t)

    return visualized_tuples, questions, instance_mapping

def step2_predict(pipe, questions, text_list, step1_predicted_tuple, instance_mapping):

    res = []
    for t, q in zip(text_list, questions):
        r = pipe(context=t, question=q)
        res.append(r)

    res = [res] if isinstance(res, dict) else res
    assert len(res) == len(instance_mapping)
    last_idx = -1
    tuple_idx = 0
    packed_res = []
    for idx, (r, i) in enumerate(zip(res, instance_mapping)):
        if i == last_idx:
            tuple_idx += 1
        else:
            tuple_idx = 0
            last_idx = i
        if r['score'] < null_threshold:
            holder = None
        else:
            holder = r['answer']
        aspect_term = step1_predicted_tuple[i][tuple_idx][0] if step1_predicted_tuple[i][tuple_idx][0] != 'empty' else None
        expression_term = step1_predicted_tuple[i][tuple_idx][1] if step1_predicted_tuple[i][tuple_idx][1] != 'empty' else None
        packed_res.append({
            "text": text_list[idx],
            "holder": holder,
            "aspect": aspect_term,
            "expression": expression_term,
            "polarity": step1_predicted_tuple[i][tuple_idx][2],
        })
    return packed_res

def prepare_models_and_tokenizers(languange='en', domain='hotel'):

    global model_list
    global plm_model_name_list
    global tokenizer_list
    global msl
    global lang

    lang = languange

    if lang=='en' and domain=='hotel':
        step1_plm_model_name = 'roberta-large'
        step2_plm_model_name = 'roberta-large-squad2'
    elif lang=='es' and domain=='hotel':
        step1_plm_model_name = 'xlm-roberta-large'
        step2_plm_model_name = 'xlm-roberta-large-squad2'
    
    if lang=='en' and domain=='hotel':
        dataset = 'opener_en'
    if lang=='es' and domain=='hotel':
        dataset = 'opener_es'

    model_save_dir = CURRENT_POJ_DIR / 'saved_models'
    step1_model_save_path = model_save_dir / 'extract' / step1_plm_model_name / dataset / 'checkpoint_best.pt'
    step1_model = torch.load(step1_model_save_path).to('cpu')
    step1_tokenizer = AutoTokenizer.from_pretrained(PLM_DIR / step1_plm_model_name)
    
    step1_model.to(device)
    step1_model.eval()

    step2_model_save_path = model_save_dir / 'QA' / step2_plm_model_name / dataset
    step2_model = AutoModelForQuestionAnswering.from_pretrained(step2_model_save_path)
    step2_model.eval()
    step2_tokenizer = AutoTokenizer.from_pretrained(step2_model_save_path)
    step2_pipe = QuestionAnsweringPipeline(
        model=step2_model, 
        tokenizer=step2_tokenizer,
        batch_size=8,
        device=-1)

    model_list = [step1_model, step2_pipe]
    plm_model_name_list = [step1_plm_model_name, step2_plm_model_name]
    tokenizer_list = [step1_tokenizer, step2_tokenizer]
    msl = max_sequence_length[f"{step1_plm_model_name}-{dataset}"]

prepare_models_and_tokenizers()