from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
import math
from collections import defaultdict
from metrics.rouge.evaluator import EvaluateTool

# CONSTANTS
EXTRACTIVENESS_KEYS_ORDERED = ['normal', 'high', 'fully']
METRICS = ['rouge-2', 'rouge-3']

def get_extractiveness_values(target, source, metrics=None):
    if metrics is None:
        metrics = ['rouge-1','rouge-2']

    rouge_evaluator = EvaluateTool()
    target_values = []
    for pred, source in zip(target, source):
        # check if precision is correct
        cur_rouge_score = rouge_evaluator.evaluate_list_fast([pred], [source], metrics=metrics)
        cur_rouge_avg = []
        for x in cur_rouge_score:
            all_metrics_scores = [x[metric]['p'] for metric in metrics]
            cur_rouge_avg.append(sum(all_metrics_scores)/len(all_metrics_scores))
        target_values.append(cur_rouge_avg[0])
    return target_values

def get_bucket(bucket, evaluator=None, print_out=False):
    ret = dict()
    for key, val in bucket.items():
        if evaluator is not None:
            val = evaluator(val)
        ret[key] = sum(val) / len(val)
        if print_out:
            print(key, sum(val) / len(val), len(val))
            print()
    return ret

def cal_inner(bucket_results, key_dict):
    inner_score = 0
    for i in range(1,len(key_dict)):
        inner_score += bucket_results[key_dict[i]] - bucket_results[key_dict[i-1]]
    return inner_score

def cal_intra(bucket_gold, bucket_pred):
    diffs = []

    for key in EXTRACTIVENESS_KEYS_ORDERED:
        length = len(bucket_gold[key])
        for i in range(length):
            gold = bucket_gold[key][i]
            pred = bucket_pred[key][i]
            diff = math.fabs(gold - pred)
            diffs.append(diff)

    ret = sum(diffs)/len(diffs)
    return ret

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])

def cal_diff(gold_list, pred_list, relative=True):
    diffs = []
    for gold, pred in zip(gold_list, pred_list):
        diff = math.fabs(gold - pred)
        if relative:
            diff /= (gold + 0.1)
        diffs.append(diff)
    ret = sum(diffs) / len(diffs)
    return ret

def get_ext_score(samples):
    bucket_gold = []
    for sample in samples:
        bucket_gold.append(get_extractiveness_values([sample['summary']], [sample['text_in']], metrics=METRICS)[0])
    bucket_pred = []
    for sample in samples:
        bucket_pred.append(get_extractiveness_values([sample['prediction']], [sample['text_in']], metrics=METRICS)[0])

    score = cal_diff(bucket_gold, bucket_pred)

    return score

def get_ext_value_one_sample(sample):
    return get_extractiveness_values([sample['prediction']], [sample['text_in']], metrics=METRICS)[0]