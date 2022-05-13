from pathlib import Path
import json

REPO_DIR = Path(__file__).resolve().parent.parent.parent

LOG_DIR = REPO_DIR / 'logs'
DATA_DIR = REPO_DIR / 'data'
RESOURCES_DIR = REPO_DIR / 'resources'
CURRENT_POJ_DIR = REPO_DIR / 'src' / 'GTS'
QA_DATA_DIR = CURRENT_POJ_DIR / 'QA_data'
SAVE_DIR = CURRENT_POJ_DIR / 'saved_models'
PLM_DIR = REPO_DIR / 'pretrained_models'
PRED_DIR = CURRENT_POJ_DIR / 'predictions'

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
id2sentiment = {3: 'Negative', 4: 'Neutral', 5: 'Positive'}
dataset2lang = {
    'darmstadt_unis': 'en',
    'mpqa': 'en',
    'multibooked_ca': 'ca',
    'multibooked_eu': 'eu',
    'norec': 'nb',
    'opener_en': 'en',
    'opener_es': 'es'
}

from nltk.tokenize.simple import SpaceTokenizer

st = SpaceTokenizer()

def convert_to_submission_json_format(sentence_pack, tuples, mode='triplet'):

    pred_json = []
    assert len(sentence_pack) == len(tuples)

    if mode=='triplet':

        for pack, tuple_sent_id in zip(sentence_pack, tuples):

            sentence = pack['text']
            tokenized = st.tokenize(sentence)
            word_spans = list(st.span_tokenize(sentence))

            opinions = []

            assert pack['sent_id'] == tuple_sent_id

            for t in tuples[tuple_sent_id]:

                aspect_expression = [" ".join(tokenized[min(t[0]):max(t[0])+1])] if t[0] else []
                polar_expression = [" ".join(tokenized[min(t[1]):max(t[1])+1])] if t[1] else []

                aspect_span = [f"{word_spans[min(t[0])][0]}:{word_spans[max(t[0])][1]}"] if t[0] else []
                polar_span = [f"{word_spans[min(t[1])][0]}:{word_spans[max(t[1])][1]}"] if t[1] else []

                opinions.append(
                    {
                        'Source': [[], []],
                        'Target': [aspect_expression, aspect_span],
                        'Polar_expression': [polar_expression, polar_span],
                        'Polarity': t[-1]
                    }
                )

            pred_json.append(
                {
                    "sent_id": tuple_sent_id,
                    "text": sentence,
                    "opinions": opinions,
                }
            )
    
    elif mode=="quadtuple":

        for pack, tuple_sent_id in zip(sentence_pack, tuples):

            sentence = pack['text']
            tokenized = st.tokenize(sentence)
            word_spans = list(st.span_tokenize(sentence))

            opinions = []

            assert pack['sent_id'] == tuple_sent_id

            for t in tuples[tuple_sent_id]:

                holder_expressinon = [" ".join(tokenized[min(t[0]):max(t[0])+1])] if t[0] else []
                aspect_expression = [" ".join(tokenized[min(t[1]):max(t[1])+1])] if t[1] else []
                polar_expression = [" ".join(tokenized[min(t[2]):max(t[2])+1])] if t[2] else []

                holder_span = [f"{word_spans[min(t[0])][0]}:{word_spans[max(t[0])][1]}"] if t[0] else []
                aspect_span = [f"{word_spans[min(t[1])][0]}:{word_spans[max(t[1])][1]}"] if t[1] else []
                polar_span = [f"{word_spans[min(t[2])][0]}:{word_spans[max(t[2])][1]}"] if t[2] else []

                opinions.append(
                    {
                        'Source': [holder_expressinon, holder_span],
                        'Target': [aspect_expression, aspect_span],
                        'Polar_expression': [polar_expression, polar_span],
                        'Polarity': t[-1]
                    }
                )

            pred_json.append(
                {
                    "sent_id": tuple_sent_id,
                    "text": sentence,
                    "opinions": opinions,
                }
            )

    return pred_json

class OutputPreprocessor:

    def __init__(self, predictions, bert_lengths, sen_lengths, tokens_ranges, ignore_index=-1):

        self.predictions = predictions
        self.bert_lengths = bert_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = ignore_index
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        # Only care about the elements in diagonal
        spans = []
        start = -1
        for i in range(length+1): # [CLS] is the token at index 0
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length])
        return spans

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        triplets = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 8
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                        # if tags[i][j] != -1:
                        #     tag_num[int(tags[i][j])] += 1
                        # if tags[j][i] != -1:
                        #     tag_num[int(tags[j][i])] += 1
                if sum(tag_num[3:]) == 0: continue
                sentiment = -1
                # count the tag num
                if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                    sentiment = 5
                elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                    sentiment = 4
                elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                    sentiment = 3
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                triplets.append([al, ar, pl, pr, sentiment])
        return triplets

    def get_tuples(self):

        predicted_tuples = []

        for i in range(self.data_num):
            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            implict_aspect = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 6)
            predicted_aspect_spans.extend(implict_aspect)
            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            implict_opinion = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 7)
            predicted_opinion_spans.extend(implict_opinion)
            predicted_tuples.append(self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i]))

        return predicted_tuples

class SemEvalEvaluator:

    def __init__(self, predicted_tuples, dataset_name, mode="valid"):

        self.mode = mode

        if mode == "valid":
            gold_path = DATA_DIR / dataset_name / 'dev.json'
            self.sent_ids, self.golden_triplets, self.golden_quadtuples = self.load_gold_from_file(gold_path)
            self.predicted_triplets, self.predicted_quadtuples = self.convert_to_submission_tuple_format(predicted_tuples)
        elif mode == "test":
            gold_path = DATA_DIR / dataset_name / 'test.json'
            self.sent_ids, self.golden_triplets, self.golden_quadtuples = self.load_gold_from_file(gold_path)
            self.predicted_triplets, self.predicted_quadtuples = self.convert_to_submission_tuple_format(predicted_tuples)
        elif mode == "predict":
            gold_path = DATA_DIR / dataset_name / 'test.json'
            self.sent_ids = self.load_gold_from_file(gold_path)
            self.predicted_triplets, self.predicted_quadtuples = self.convert_to_submission_tuple_format(predicted_tuples)
        else:
            raise ValueError("Evaluation mode invalid!")

    def convert_opinion_to_triplet(self, sentence):
        text = sentence["text"]
        opinions = sentence["opinions"]
        opinion_tuples = []
        token_offsets = list(st.span_tokenize(text))

        if len(opinions) > 0:
            for opinion in opinions:
                target_char_idxs = opinion["Target"][1]
                exp_char_idxs = opinion["Polar_expression"][1]
                polarity = opinion["Polarity"]

                target = self.convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
                exp = self.convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
                opinion_tuples.append((target, exp, polarity))
        return opinion_tuples

    def convert_opinion_to_quadtuples(self, sentence):
        text = sentence["text"]
        opinions = sentence["opinions"]
        opinion_tuples = []
        token_offsets = list(st.span_tokenize(text))
        #
        if len(opinions) > 0:
            for opinion in opinions:
                holder_char_idxs = opinion["Source"][1]
                target_char_idxs = opinion["Target"][1]
                exp_char_idxs = opinion["Polar_expression"][1]
                polarity = opinion["Polarity"]
                #
                holder = self.convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
                target = self.convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
                exp = self.convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
                opinion_tuples.append((holder, target, exp, polarity))
        return opinion_tuples

    def convert_char_offsets_to_token_idxs(self, char_offsets, token_offsets):
        """
        char_offsets: list of str
        token_offsets: list of tuples

        >>> text = "I think the new uni ( ) is a great idea"
        >>> char_offsets = ["8:19"]
        >>> token_offsets =
        [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

        >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
        >>> (2,3,4)
        """
        token_idxs = []
        #
        for char_offset in char_offsets:
            bidx, eidx = char_offset.split(":")
            bidx, eidx = int(bidx), int(eidx)
            intoken = False
            for i, (b, e) in enumerate(token_offsets):
                if b == bidx:
                    intoken = True
                if intoken:
                    token_idxs.append(i)
                if e == eidx:
                    intoken = False
        return set(token_idxs)

    def load_gold_from_file(self, gold_path):

        with open(gold_path) as infile:
            gold = json.load(infile)

        sent_ids = [s["sent_id"] for s in gold] # It is orderly

        if self.mode=='valid' or self.mode=='test':
        
            golden_triplets = dict([(s["sent_id"], self.convert_opinion_to_triplet(s)) for s in gold])
            golden_quadtuples = dict([(s["sent_id"], self.convert_opinion_to_quadtuples(s)) for s in gold])

            return sent_ids, golden_triplets, golden_quadtuples

        elif self.mode=='predict':

            return sent_ids

    def convert_to_submission_tuple_format(self, tuples):

        assert len(self.sent_ids) == len(tuples)

        all_converted_triplets = {}
        all_converted_quadtuples = {}

        for sent_id, i in zip(self.sent_ids, tuples):

            converted_triplets = []
            converted_quadtuples = []

            for tuple in i:

                aspect_start_idx, aspect_end_idx = tuple[0], tuple[1]
                opinion_start_idx, opinion_end_idx  = tuple[2], tuple[3]

                aspect_range_set = set(range(aspect_start_idx-1, aspect_end_idx)) if set(range(aspect_start_idx-1, aspect_end_idx)) != set([-1]) else set()
                opinion_range_set = set(range(opinion_start_idx-1, opinion_end_idx)) if set(range(opinion_start_idx-1, opinion_end_idx)) != set([-1]) else set()
            
                converted_triplets.append((aspect_range_set, opinion_range_set, id2sentiment[tuple[4]]))
                converted_quadtuples.append((set(), aspect_range_set, opinion_range_set, id2sentiment[tuple[4]]))

            all_converted_triplets[sent_id] = converted_triplets
            all_converted_quadtuples[sent_id] = converted_quadtuples

        return all_converted_triplets, all_converted_quadtuples

    def sent_triplets_in_list(self, sent_tuple1, list_of_sent_tuples, keep_polarity=True):
        target1, exp1, pol1 = sent_tuple1
        if len(target1) == 0:
            target1 = frozenset(["_"])
        for target2, exp2, pol2 in list_of_sent_tuples:
            if len(target2) == 0:
                target2 = frozenset(["_"])
            if (
                len(target1.intersection(target2)) > 0
                and len(exp1.intersection(exp2)) > 0
            ):
                if keep_polarity:
                    if pol1 == pol2:
                        return True
                else:
                    return True
        return False

    def sent_quadtuples_in_list(self, sent_tuple1, list_of_sent_tuples, keep_polarity=True):
        holder1, target1, exp1, pol1 = sent_tuple1
        if len(holder1) == 0:
            holder1 = frozenset(["_"])
        if len(target1) == 0:
            target1 = frozenset(["_"])
        for holder2, target2, exp2, pol2 in list_of_sent_tuples:
            if len(holder2) == 0:
                holder2 = frozenset(["_"])
            if len(target2) == 0:
                target2 = frozenset(["_"])
            if (
                len(holder1.intersection(holder2)) > 0
                and len(target1.intersection(target2)) > 0
                and len(exp1.intersection(exp2)) > 0
            ):
                if keep_polarity:
                    if pol1 == pol2:
                        return True
                else:
                    return True
        return False


    def triplet_weighted_score(self, sent_tuple1, list_of_sent_tuples):
        best_overlap = 0
        target1, exp1, pol1 = sent_tuple1
        if len(target1) == 0:
            target1 = frozenset(["_"])
        for target2, exp2, pol2 in list_of_sent_tuples:
            if len(target2) == 0:
                target2 = frozenset(["_"])
            if (
                len(target2.intersection(target1)) > 0
                and len(exp2.intersection(exp1)) > 0
            ):
                target_overlap = len(target2.intersection(target1)) / len(target1)
                exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
                overlap = (target_overlap + exp_overlap) / 2
                if overlap > best_overlap:
                    best_overlap = overlap
        return best_overlap

    def quadtuple_weighted_score(self, sent_tuple1, list_of_sent_tuples):
        best_overlap = 0
        holder1, target1, exp1, pol1 = sent_tuple1
        if len(holder1) == 0:
            holder1 = frozenset(["_"])
        if len(target1) == 0:
            target1 = frozenset(["_"])
        for holder2, target2, exp2, pol2 in list_of_sent_tuples:
            if len(holder2) == 0:
                holder2 = frozenset(["_"])
            if len(target2) == 0:
                target2 = frozenset(["_"])
            if (
                len(holder2.intersection(holder1)) > 0
                and len(target2.intersection(target1)) > 0
                and len(exp2.intersection(exp1)) > 0
            ):
                holder_overlap = len(holder2.intersection(holder1)) / len(holder1)
                target_overlap = len(target2.intersection(target1)) / len(target1)
                exp_overlap = len(exp2.intersection(exp1)) / len(exp1)
                overlap = (holder_overlap + target_overlap + exp_overlap) / 3
                if overlap > best_overlap:
                    best_overlap = overlap
        return best_overlap

    def triplet_precision(self, gold, pred, keep_polarity=True, weighted=True):
        """
        Weighted true positives / (true positives + false positives)
        """
        weighted_tp = []
        tp = []
        fp = []
        #
        for sent_idx in pred.keys():
            ptuples = pred[sent_idx]
            gtuples = gold[sent_idx]
            for stuple in ptuples:
                if self.sent_triplets_in_list(stuple, gtuples, keep_polarity):
                    if weighted:
                        #sc = weighted_score(stuple, gtuples)
                        #if sc != 1:
                            #print(sent_idx)
                            #print(sc)
                            #print()
                        weighted_tp.append(self.triplet_weighted_score(stuple, gtuples))
                        tp.append(1)
                    else:
                        weighted_tp.append(1)
                        tp.append(1)
                else:
                    #print(sent_idx)
                    fp.append(1)
        #print("weighted tp: {}".format(sum(weighted_tp)))
        #print("tp: {}".format(sum(tp)))
        #print("fp: {}".format(sum(fp)))
        return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


    def triplet_recall(self, gold, pred, keep_polarity=True, weighted=True):
        """
        Weighted true positives / (true positives + false negatives)
        """
        weighted_tp = []
        tp = []
        fn = []
        #
        assert len(gold) == len(pred)
        #
        for sent_idx in pred.keys():
            ptuples = pred[sent_idx]
            gtuples = gold[sent_idx]
            for stuple in gtuples:
                if self.sent_triplets_in_list(stuple, ptuples, keep_polarity):
                    if weighted:
                        weighted_tp.append(self.triplet_weighted_score(stuple, ptuples))
                        tp.append(1)
                    else:
                        weighted_tp.append(1)
                        tp.append(1)
                else:
                    fn.append(1)
        return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

    def quadtuple_precision(self, gold, pred, keep_polarity=True, weighted=True):
        """
        Weighted true positives / (true positives + false positives)
        """
        weighted_tp = []
        tp = []
        fp = []
        #
        for sent_idx in pred.keys():
            ptuples = pred[sent_idx]
            gtuples = gold[sent_idx]
            for stuple in ptuples:
                if self.sent_quadtuples_in_list(stuple, gtuples, keep_polarity):
                    if weighted:
                        weighted_tp.append(self.quadtuple_weighted_score(stuple, gtuples))
                        tp.append(1)
                    else:
                        weighted_tp.append(1)
                        tp.append(1)
                else:
                    fp.append(1)
        return sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

    def quadtuple_recall(self, gold, pred, keep_polarity=True, weighted=True):
        """
        Weighted true positives / (true positives + false negatives)
        """
        weighted_tp = []
        tp = []
        fn = []
        #
        assert len(gold) == len(pred)
        #
        for sent_idx in pred.keys():
            ptuples = pred[sent_idx]
            gtuples = gold[sent_idx]
            for stuple in gtuples:
                if self.sent_quadtuples_in_list(stuple, ptuples, keep_polarity):
                    if weighted:
                        weighted_tp.append(self.quadtuple_weighted_score(stuple, ptuples))
                        tp.append(1)
                    else:
                        weighted_tp.append(1)
                        tp.append(1)
                else:
                    fn.append(1)
        return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

    def get_all_triplet_metrics(self, keep_polarity=True, weighted=True):
        # print(self.golden_tuples)
        # print(self.predicted_tuples)
        prec = self.triplet_precision(self.golden_triplets, self.predicted_triplets, keep_polarity, weighted)
        rec = self.triplet_recall(self.golden_triplets, self.predicted_triplets, keep_polarity, weighted)
        return prec, rec, 2 * (prec * rec) / (prec + rec + 0.00000000000000001)

    def get_all_quadtuple_metrics(self, keep_polarity=True, weighted=True):

        prec = self.quadtuple_precision(self.golden_quadtuples, self.predicted_quadtuples, keep_polarity, weighted)
        rec = self.quadtuple_recall(self.golden_quadtuples, self.predicted_quadtuples, keep_polarity, weighted)
        return prec, rec, 2 * (prec * rec) / (prec + rec + 0.00000000000000001)