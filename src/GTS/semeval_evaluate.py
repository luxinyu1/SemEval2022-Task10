#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function, division

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
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
    return frozenset(token_idxs)


def convert_opinion_to_tuple(sentence):
    text = sentence["text"]
    opinions = sentence["opinions"]
    opinion_tuples = []
    token_offsets = list(tk.span_tokenize(text))
    #
    if len(opinions) > 0:
        for opinion in opinions:
            holder_char_idxs = opinion["Source"][1]
            target_char_idxs = opinion["Target"][1]
            exp_char_idxs = opinion["Polar_expression"][1]
            polarity = opinion["Polarity"]
            #
            holder = convert_char_offsets_to_token_idxs(holder_char_idxs, token_offsets)
            target = convert_char_offsets_to_token_idxs(target_char_idxs, token_offsets)
            exp = convert_char_offsets_to_token_idxs(exp_char_idxs, token_offsets)
            opinion_tuples.append((holder, target, exp, polarity))
    return opinion_tuples

# def error_analysis(sent_tuple1, list_of_sent_tuples):

#     # error_type
#     # 0: holder error
#     # 1: aspect error
#     # 2: expression error
#     # 3: polarity error

#     error = [0, 0, 0, 0]

#     holder1, target1, exp1, pol1 = sent_tuple1
#     if len(holder1) == 0:
#         holder1 = frozenset(["_"])
#     if len(target1) == 0:
#         target1 = frozenset(["_"])
#     for holder2, target2, exp2, pol2 in list_of_sent_tuples:
#         if len(holder2) == 0:
#             holder2 = frozenset(["_"])
#         if len(target2) == 0:
#             target2 = frozenset(["_"])
#         if (
#             len(holder1.intersection(holder2)) > 0
#             and len(target1.intersection(target2)) > 0
#             and len(exp1.intersection(exp2)) > 0
#         ):
#             if keep_polarity:
#                 if pol1 != pol2:
#                     error[3] = 
#             else:
#                 return True



def sent_tuples_in_list(sent_tuple1, list_of_sent_tuples, keep_polarity=True):
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
                    # print(holder1, target1, exp1, pol1)
                    # print(holder2, target2, exp2, pol2)
                    return True
            else:
                return True

    return False


def weighted_score(sent_tuple1, list_of_sent_tuples):
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


def tuple_precision(gold, pred, keep_polarity=True, weighted=True):
    """
    Weighted true positives / (true positives + false positives)
    """
    weighted_tp = []
    tp = []
    fp = []
    all_f_tuples = []
    all_gold_tuples = []
    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]
        all_gold_tuples.append(gtuples)
        f_tuple = []
        for stuple in ptuples:
            if sent_tuples_in_list(stuple, gtuples, keep_polarity):
                if weighted:
                    #sc = weighted_score(stuple, gtuples)
                    #if sc != 1:
                        #print(sent_idx)
                        #print(sc)
                        #print()
                    weighted_tp.append(weighted_score(stuple, gtuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fp.append(1)
                f_tuple.append(stuple)
        all_f_tuples.append(f_tuple)
    #print("weighted tp: {}".format(sum(weighted_tp)))
    #print("tp: {}".format(sum(tp)))
    #print("fp: {}".format(sum(fp)))
    return all_f_tuples, all_gold_tuples, sum(weighted_tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def tuple_recall(gold, pred, keep_polarity=True, weighted=True):
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
            if sent_tuples_in_list(stuple, ptuples, keep_polarity):
                if weighted:
                    weighted_tp.append(weighted_score(stuple, ptuples))
                    tp.append(1)
                else:
                    weighted_tp.append(1)
                    tp.append(1)
            else:
                fn.append(1)
    return sum(weighted_tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def tuple_f1(gold, pred, keep_polarity=True, weighted=True):
    all_f_tuples, all_gold_tuples, prec = tuple_precision(gold, pred, keep_polarity, weighted)
    rec = tuple_recall(gold, pred, keep_polarity, weighted)
    #print("prec: {}".format(prec))
    #print("rec: {}".format(rec))
    return all_f_tuples, all_gold_tuples, prec, rec, 2 * (prec * rec) / (prec + rec + 0.00000000000000001)