from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
from collections import defaultdict
from metrics.rouge.evaluator import EvaluateTool
import math

# CONSTANTS
EXTRACTIVENESS_KEYS_ORDERED = ['normal', 'high', 'fully']

def get_topic_values(topic, prediction):
    return [get_topic_value(x, y) for x, y in zip(topic, prediction)]

def get_topic_value(topic, prediction):
    topic_scores = []
    tokens = nltk.word_tokenize(topic)
    cnt_all = 0
    cnt_hit = 0
    for token in tokens:
        if not token.isalpha():
            continue
        cnt_all += 1
        if token.lower() in prediction.lower():
            cnt_hit += 1

    if cnt_all == 0:
        return 0

    topic_scores.append(1.0 * cnt_hit / cnt_all)
    return sum(topic_scores)/len(topic_scores)

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])

def get_topic_score(data):
    topic_scores = []
    gold_scores = []
    for sample in data:
        if len(sample['topic']) == 0:
            continue
        topic_scores.append(get_topic_value(sample['topic'], sample['prediction']))
        gold_scores.append(get_topic_value(sample['topic'], sample['summary']))
    abs_score = sum(topic_scores) / len(topic_scores)
    relative_score = cal_diff(gold_scores, topic_scores)
    return relative_score

def cal_diff(gold_list, pred_list, relative=True):
    diffs = []
    for gold, pred in zip(gold_list, pred_list):
        diff = math.fabs(gold - pred)
        if relative:
            diff /= gold if gold else 0.1
        diffs.append(diff)
    ret = sum(diffs) / len(diffs)
    return ret