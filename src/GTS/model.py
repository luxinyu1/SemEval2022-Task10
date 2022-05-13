# This code is modified based on https://github.com/NJUNLP/GTS/blob/main/code/BertModel/model.py

import torch
import torch.nn
from transformers import AutoModel
from transformers.models.auto.configuration_auto import AutoConfig
from utils import PLM_DIR

class MultiInferModel(torch.nn.Module):
    
    def __init__(self, args):

        super(MultiInferModel, self).__init__()

        self.args = args
        self.plm_model_dir = PLM_DIR / args.plm_model_name
        self.plm = AutoModel.from_pretrained(self.plm_model_dir)
        self.hidden_size = AutoConfig.from_pretrained(self.plm_model_dir).hidden_size
        self.cls_linear = torch.nn.Linear(self.hidden_size*2, args.class_num)
        self.feature_linear = torch.nn.Linear(self.hidden_size*2 + args.class_num*3, self.hidden_size*2)
        self.dropout = torch.nn.Dropout(0.1)

    def multi_hops(self, features, mask, k):

        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits = self.cls_linear(features)

        for _ in range(k):
            probs = logits
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1,-1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits = self.cls_linear(features)
            
        return logits

    def forward(self, inputs):

        outputs = self.plm(**inputs) # [batch_size, max_seq_length, hidden_size]

        plm_feature = self.dropout(outputs.last_hidden_state)
        plm_feature = plm_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        plm_feature = torch.cat([plm_feature, plm_feature.transpose(1, 2)], dim=3)

        logits = self.multi_hops(plm_feature, inputs['attention_mask'], self.args.nhops)

        return logits