from nltk.tokenize.simple import SpaceTokenizer
from numpy import empty, positive
import torch
from torch.utils.data import Dataset
import logging
from transformers import AutoTokenizer
import pickle
from utils import DATA_DIR, PLM_DIR, sentiment2id, dataset2lang
# from allennlp.predictors.predictor import Predictor
# import allennlp_models.tagging
from itertools import product
import random

from utils import CURRENT_POJ_DIR

st = SpaceTokenizer()

logger = logging.getLogger(__name__)

def get_max_sequence_len(packs, args):

    max_sequence_len = 0
    plm_model_path = PLM_DIR / args.plm_model_name
    tokenizer = AutoTokenizer.from_pretrained(plm_model_path)

    for pack in packs:
        for instance in pack:
            length = len(tokenizer.encode(instance['text']))
            if length > max_sequence_len:
                max_sequence_len = length
    
    return max_sequence_len

def preprocess(sentence_packs, args, mode="train"):
    instances = list()
    plm_model_path = PLM_DIR / args.plm_model_name 
    tokenizer = AutoTokenizer.from_pretrained(plm_model_path)
    for sentence_pack in sentence_packs:
        instances.append(GTSPreprocessor(
            tokenizer, 
            sentence_pack, 
            args.max_sequence_len, 
            args.plm_model_name, 
            mode))

    # Long live visualization!
    # i = instances[213]
    # tags = np.array(i.tags, dtype=np.int64)
    # np.savetxt('tags.txt', tags, fmt='%d', delimiter='\t')
    
    return instances

def preprocess_classifier(sentence_packs, args):
    instances = list()
    num_positive = 0
    num_negative = 0
    for sentence_pack in sentence_packs:
        instance = ClassifierPreprocessor(sentence_pack, args)
        num_positive += len(instance.positive_samples)
        num_negative += len(instance.negative_samples)
        instances.extend(instance.get_all_samples())
    return num_positive, num_negative, instances

def convert_char_offsets_to_token_offsets(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets = [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> -> [[2,4]]
    """
    token_idxs = []

    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        btoken = 0
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                btoken = i
            if e == eidx:
                token_idxs.append([btoken, i])
            if (b < bidx and bidx < e) or (b < eidx and eidx < e):
                raise Exception("it seems that this annotation is wrong!")
    
    return token_idxs

def get_token_offsets_from_text(text):
    pass

# class QAPreprocessor:

#     def __init__(self, sentence_pack, args):

#         self.args = args

#     def __len__(self, args):

#         pass


class ClassifierPreprocessor:

    def __init__(self, sentence_pack, args):

        self.args = args
        self.sentence_pack = sentence_pack
        self.sent_id = sentence_pack['sent_id']
        self.sentence = sentence_pack['text']
        self.tokens = st.tokenize(self.sentence.strip())
        self.len_tokens = len(self.tokens)
        self.token_offsets = list(st.span_tokenize(self.sentence))
        all_holder_spans, all_aspect_spans, all_opinion_spans, all_p, self.positive_samples = self.positive_sampling(self.sentence_pack)
        self.negative_samples = self.negative_sampling(self.sentence_pack, all_holder_spans, all_aspect_spans, all_opinion_spans, all_p, len(self.positive_samples))

    def positive_sampling(self, sentence_pack):

        def _generate_all_possible_spans(span):

            _all_possible_spans = []
            if span[0]:
                range_span = list(range(span[0][0], span[0][1]+1))
                for window in range(len(range_span)):
                    for i in range(len(range_span)-window):
                        _all_possible_spans.append([range_span[i], range_span[i+window]])
                return _all_possible_spans
            else:
                return span
        
        true_opinions = []
        all_p = []

        all_holder_spans = []
        all_aspect_spans = []
        all_opinion_spans = []

        for opinion in sentence_pack['opinions']:
            try:
                holder_span = convert_char_offsets_to_token_offsets(opinion["Source"][1], self.token_offsets)
                if not holder_span: holder_span = [[]]
            except:
                logger.warning("Skipping a wrong holder annotation!")
                holder_span = [[]]
            try:
                aspect_span = convert_char_offsets_to_token_offsets(opinion['Target'][1], self.token_offsets)
                if not aspect_span: aspect_span = [[]]
            except:
                logger.warning("Skipping a wrong aspect annotation!")
                aspect_span = [[]]
            try:
                opinion_span = convert_char_offsets_to_token_offsets(opinion['Polar_expression'][1], self.token_offsets)
                if not opinion_span: opinion_span = [[]]
            except:
                logger.warning("Skipping a wrong opinion annotation!")
                opinion_span = [[]]

            if holder_span[0] not in all_holder_spans:
                all_holder_spans.append(holder_span[0])
            if aspect_span[0] not in all_aspect_spans:
                all_aspect_spans.append(aspect_span[0])
            if opinion_span[0] not in all_opinion_spans:
                all_opinion_spans.append(opinion_span[0])

            all_possible_holder_spans = _generate_all_possible_spans(holder_span)
            all_possible_aspect_spans = _generate_all_possible_spans(aspect_span)
            all_possible_opinion_spans = _generate_all_possible_spans(opinion_span)

            # A trick that fuzzying the boundary and generate more positive samples
            for p in product(all_possible_holder_spans, all_possible_aspect_spans, all_possible_opinion_spans):

                all_p.append(p)

            true_opinions.append((self.sentence, holder_span[0], aspect_span[0], opinion_span[0], 1)) # 1 indicates that it is a true instance. 

            random.shuffle(all_p)

            for p in all_p:

                if ((self.sentence,)+p+(1,)) not in true_opinions and len(true_opinions) <= self.args.positive_sampling_size:

                    true_opinions.append((self.sentence,)+p+(1,))

        return all_holder_spans, all_aspect_spans, all_opinion_spans, all_p, true_opinions

    def negative_sampling(self, sentence_pack, all_holder_spans, all_aspect_spans, all_opinion_spans, all_p, positive_sample_num):

        def _generate_random_triplet():

            triplet = []

            while len(triplet) < 3:

                a = random.randint(0, self.len_tokens-1)
                b = random.randint(a, self.len_tokens-1)

                if [a, b] not in triplet:
                    triplet.append([a, b])
            
            return (triplet[0], triplet[1], triplet[2])

        def _generate_all_left_spans(given_spans):

            _all_left_spans = []
            range_span = list(range(0, self.len_tokens))
            for window in range(len(range_span)):
                for i in range(len(range_span)-window):
                    span = [range_span[i], range_span[i+window]]
                    if span not in given_spans:
                        _all_left_spans.append([range_span[i], range_span[i+window]])
            
            return _all_left_spans

        """
        Sampling stratgy:
        - Holder: If there exists holder expression, sample the triplet that don't have it and the triplets replacing holder with holder like words.
        - Aspect and opinion: If there exists aspect and opinion, swaping the aspects and opinons in the triplets and replacing the aspect and opinion with random words.
        """
        fake_opinions = []

        # TODO: Try it later
        # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
        # res = predictor.predict(sentence_pack['text'])

        swap_f = []
        all_f = []

        # mix aspect and opinion in true opinions

        # print("all_holder_spans:", all_holder_spans)
        # print("all_aspect_spans:", all_aspect_spans)
        # print("all_opinion_spans:", all_opinion_spans)
        # print("all_p:", all_p)
        # print("sentence_pack:", sentence_pack)

        # TODO: !IMPORTANT: 做对于 holder 的增强

        holder_dict_path = DATA_DIR / 'holder_dict' / self.args.dataset / 'holder_dict.pickle'

        with open(holder_dict_path, 'rb') as f_pickle:

            holder_dict = pickle.load(f_pickle)

        for t in all_p:

            if t[0]: # if there's a holder, remove the holder to make fake tuples
                all_f.append(([], t[1], t[2]))
            else: # if there's no holder check if there's holder like word in sentence to make fake tuples
                sent_tokenized = st.tokenize(sentence_pack["text"])
                holders_in_sent = [sent_tokenized.index(w) for w in sent_tokenized if w.lower() in holder_dict and holder_dict[w.lower()] > 1]
                for h in holders_in_sent:
                    if ([h, h], t[1], t[2]) not in all_p:
                        all_f.append(([h, h], t[1], t[2]))

        # print("holder:" ,all_holder_spans)
        # print("aspect:", all_aspect_spans)
        # print("opinion:", all_opinion_spans)

        if not ((len(all_holder_spans) == 0 and len(all_aspect_spans) == 0 and len(all_opinion_spans) == 0)):

            for fake_tuple in product(all_holder_spans, all_aspect_spans, all_opinion_spans):

                swap_f.append(fake_tuple)

        swap_f = [f for f in swap_f if f not in all_p]

        random.shuffle(swap_f)

        for s in swap_f:

            all_f.append(s)

            if len(all_f) > positive_sample_num:

                break

        if self.len_tokens >= 2: # TODO: reckless code, modify this later

            while len(all_f) < positive_sample_num:

                generated_triplet = _generate_random_triplet()
                if generated_triplet not in all_p:
                    all_f.append(generated_triplet)

        for f in all_f:

            fake_opinions.append((self.sentence,)+f+(0,))
        
        return fake_opinions

    def get_all_samples(self):

        return self.positive_samples + self.negative_samples

class GTSPreprocessor:

    def __init__(self,
                 tokenizer, 
                 sentence_pack, 
                 max_sequence_len, 
                 plm_model_name, 
                 mode='train'):

        self.mode = mode
        self.sentence_pack = sentence_pack
        self.tokenizer = tokenizer
        self.max_sequence_len = max_sequence_len
        self.plm_model_name = plm_model_name
        self.sent_id = sentence_pack['sent_id']
        self.sentence = sentence_pack['text']
        self.tokens = st.tokenize(self.sentence.strip())
        self.token_range = [[0, 0]]
        self.len_tokens = len(self.tokens)
        self.plm_tokens = tokenizer.encode(self.sentence)
        self.len_plm_tokens = len(self.plm_tokens)
        if mode != 'predict':
            self.aspect_tags = torch.zeros(max_sequence_len).long()
            self.opinion_tags = torch.zeros(max_sequence_len).long()
            self.tags = torch.zeros(max_sequence_len, max_sequence_len).long() # Grid
        self.initialize()

    def initialize(self):

        # -1 is the index which F.cross_entropy ignores

        # word -> subword span
        # TODO: Try to simply this in the future
        token_start = 1
        truncation = False
        for i, w in enumerate(self.tokens):

            if i != 0 and (self.plm_model_name == 'roberta-large' or self.plm_model_name == 'all-roberta-large-v1'):
                token_end = token_start + len(self.tokenizer.encode(" "+w, add_special_tokens=False)) # Fix the special tokenizeing method in roberta
            else:
                token_end = token_start + len(self.tokenizer.encode(w, add_special_tokens=False))
            
            if token_end > self.max_sequence_len-1:
                logger.info("Truncating one sample.")
                truncation = True
                break

            self.token_range.append([token_start, token_end-1])
            token_start = token_end
        
        if truncation:

            self.tokens = self.tokens[:i:]
            self.sentence = " ".join(self.tokens)
            self.len_tokens = i
            self.plm_tokens = self.tokenizer.encode(self.sentence)
            self.len_plm_tokens = len(self.plm_tokens)
            assert self.len_plm_tokens <= self.max_sequence_len

        assert self.len_plm_tokens == self.token_range[-1][-1] + 2

        if self.mode != 'predict':

            self.aspect_tags[self.len_plm_tokens:] = -1 # padding
            self.aspect_tags[0] = -1 # [CLS]
            self.aspect_tags[self.len_plm_tokens-1] = -1 # [SEP]

            self.opinion_tags[self.len_plm_tokens:] = -1 # padding
            self.opinion_tags[0] = -1
            self.opinion_tags[self.len_plm_tokens-1] = -1 # [SEP]

            token_offsets = list(st.span_tokenize(self.sentence))

            self.tags[:, :] = -1

            # print(self.tags.shape)
            for i in range(0, self.len_plm_tokens-1):
                for j in range(i, self.len_plm_tokens-1):
                    self.tags[i][j] = 0

            for opinion in self.sentence_pack['opinions']:

                # try:
                #     holder_span = convert_char_offsets_to_token_offsets(opinion["Source"][1], token_offsets)
                # except:
                #     logger.warning("Skipping a wrong holder annotation!")
                #     holder_span = []
                try:
                    aspect_span = convert_char_offsets_to_token_offsets(opinion['Target'][1], token_offsets)
                except:
                    logger.warning("Skipping a wrong aspect annotation!")
                    aspect_span = []
                try:
                    opinion_span = convert_char_offsets_to_token_offsets(opinion['Polar_expression'][1], token_offsets)
                except:
                    logger.warning("Skipping a wrong opinion annotation!")
                    opinion_span = []

                if not aspect_span and opinion_span:
                    aspect_span = [[-1, -1]]
                elif not opinion_span and aspect_span:
                    opinion_span = [[-1, -1]]
                elif not opinion_span and not aspect_span:
                    logger.warning(f"It seems we have a instance({self.sent_id}) that do not have both aspect and opinion span.")

                '''set tag for aspect'''
                for l, r in aspect_span:
                    start = self.token_range[l+1][0]
                    end = self.token_range[r+1][1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if i == 0:
                                self.tags[i][j] = 6
                            else:
                                self.tags[i][j] = 1
                    for i in range(l, r+1):
                        set_tag = 1 if i == l else 2
                        al, ar = self.token_range[i+1]
                        self.aspect_tags[al] = set_tag
                        self.aspect_tags[al+1:ar+1] = -1
                        '''mask positions of sub words'''
                        self.tags[al+1:ar+1, :] = -1
                        self.tags[:, al+1:ar+1] = -1

                '''set tag for opinion'''
                for l, r in opinion_span:
                    start = self.token_range[l+1][0]
                    end = self.token_range[r+1][1]
                    for i in range(start, end+1):
                        for j in range(i, end+1):
                            if i == 0:
                                self.tags[i][j] = 7
                            else:
                                self.tags[i][j] = 2
                    for i in range(l, r+1):
                        set_tag = 1 if i == l else 2
                        pl, pr = self.token_range[i+1]
                        self.opinion_tags[pl] = set_tag
                        self.opinion_tags[pl+1:pr+1] = -1
                        self.tags[pl+1:pr+1, :] = -1
                        self.tags[:, pl+1:pr+1] = -1

                for al, ar in aspect_span:
                    for pl, pr in opinion_span:
                        for i in range(al, ar+1):
                            for j in range(pl, pr+1):
                                sal, sar = self.token_range[i+1]
                                spl, spr = self.token_range[j+1]
                                self.tags[sal:sar+1, spl:spr+1] = -1
                                if i > j:
                                    self.tags[spl][sal] = sentiment2id[opinion['Polarity'].lower()]
                                else:
                                    self.tags[sal][spl] = sentiment2id[opinion['Polarity'].lower()]


class ABSADataset(Dataset):

    def __init__(self, instances, mode='train'):

        self.mode = mode
        self.instances = instances
        self.tokenizer = instances[0].tokenizer
        self.max_sequence_len = instances[0].max_sequence_len

    def __len__(self):

        return len(self.instances)

    def __getitem__(self, idx):

        sentence = self.instances[idx].sentence

        if self.mode == 'train':

            tags = self.instances[idx].tags

            return sentence, tags

        elif self.mode == 'valid':

            tags = self.instances[idx].tags

            sent_id = self.instances[idx].sent_id
            token_range = self.instances[idx].token_range
            len_tokens = self.instances[idx].len_tokens
            len_plm_tokens = self.instances[idx].len_plm_tokens

            return sentence, tags, token_range, len_tokens, len_plm_tokens, sent_id

        elif self.mode == 'predict':

            sent_id = self.instances[idx].sent_id
            token_range = self.instances[idx].token_range
            len_tokens = self.instances[idx].len_tokens
            len_plm_tokens = self.instances[idx].len_plm_tokens

            return sentence, token_range, len_tokens, len_plm_tokens, sent_id

        else:

            raise ValueError("Illegal dataset mode.")

    def collate_fn(self, data):
        
        sentences = [_[0] for _ in data]
        inputs = self.tokenizer(
            sentences,
            padding='max_length',
            max_length = self.max_sequence_len,
            return_tensors='pt',
        )

        if self.mode == 'train':

            tags = torch.stack([_[1] for _ in data], dim=0)
            
            return inputs, tags

        elif self.mode == 'valid':

            tags = torch.stack([_[1] for _ in data], dim=0)

            token_ranges = [_[2] for _ in data]
            len_tokens = [_[3] for _ in data]
            len_plm_tokens = [_[4] for _ in data]
            sent_ids = [_[5] for _ in data]

            return inputs, tags, token_ranges, len_tokens, len_plm_tokens, sent_ids

        elif self.mode == 'predict':

            token_ranges = [_[1] for _ in data]
            len_tokens = [_[2] for _ in data]
            len_plm_tokens = [_[3] for _ in data]
            sent_ids = [_[4] for _ in data]

            return inputs, token_ranges, len_tokens, len_plm_tokens, sent_ids

class ClassifierDataset(Dataset):

    def __init__(self, instances, tokenizer, args, mode='train'):

        self.args = args
        self.instances = instances
        self.tokenizer = tokenizer
        self.mode = mode
    
    def __len__(self):

        return len(self.instances)

    def __getitem__(self, idx):

        sample = self.instances[idx]
        return sample

    def span_to_text(self, sentences, spans):

        seperate_token = '[PAD]'
        empty_token = '[unused1]'

        text = []

        for sentence, span in zip(sentences, spans):

            # print(sentence)

            space_tokenized_sentence = st.tokenize(sentence)

            holder = ' '.join(space_tokenized_sentence[span[0][0]:span[0][1]+1]) if span[0] else empty_token
            aspect = ' '.join(space_tokenized_sentence[span[1][0]:span[1][1]+1]) if span[1] else empty_token
            opinion = ' '.join(space_tokenized_sentence[span[2][0]:span[2][1]+1]) if span[2] else empty_token

            # print(holder)
            # print(aspect)
            # print(opinion)

            texts_list = [holder, aspect, opinion]

            text.append(seperate_token.join(texts_list))

        return text
        
    def collate_fn(self, data):

        sentences = [_[0] for _ in data]
        spans = [(_[1], _[2], _[3]) for _ in data]

        span2texts = self.span_to_text(sentences, spans)
        
        encodings = self.tokenizer(
            text=sentences,
            text_pair=span2texts,
            truncation=True,
            padding='longest',
            max_length=self.args.max_length,
            return_tensors='pt',
        )

        if self.mode == 'train':

            labels = torch.stack([torch.tensor(_[-1], dtype=torch.long) for _ in data], dim=0)
            return {**encodings, "labels": labels}
        
        elif self.mode == 'predict':

            return encodings