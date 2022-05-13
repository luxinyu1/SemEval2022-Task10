import sys
sys.path.append('./src/GTS/')
import json
import argparse
from data import GTSPreprocessor, get_max_sequence_len
from transformers import AutoTokenizer
import numpy as np

train_sentence_packs = json.load(open('./data/opener_en/train.json'))
valid_sentence_packs = json.load(open('./data/opener_en/dev.json'))

test_sentence_packs = json.load(open('./data/opener_en/test.json'))

instances = list()
plm_model_path = './pretrained_models/ernie_2.0_skep_large_en_pytorch/'
tokenizer = AutoTokenizer.from_pretrained(plm_model_path)
mode = 'train'

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.plm_model_name = 'ernie_2.0_skep_large_en_pytorch'
args.max_sequence_len = get_max_sequence_len([train_sentence_packs, valid_sentence_packs], args)

for sentence_pack in train_sentence_packs:
    instances.append(GTSPreprocessor(tokenizer, sentence_pack, args, mode))

# Long live visualization!
i = instances[695]
print(i.tokens)
tags = np.array(i.tags, dtype=np.int64)
np.savetxt('tags.txt', tags, fmt='%d', delimiter='\t')