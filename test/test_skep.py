import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("../pretrained_models/ernie_2.0_skep_large_en_pytorch/")
model = AutoModel.from_pretrained("../pretrained_models/ernie_2.0_skep_large_en_pytorch/")