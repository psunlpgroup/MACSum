from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
import math
from collections import defaultdict

# CONSTANTS
LENGTH_KEYS_ORDERED = ['short', 'normal', 'long']

def get_length_values(target):
    target_values = [float(len(word_tokenize(x))) for x in target]
    return target_values

def get_bucket(bucket, evaluator, print_out=False, align=False):
    ret = dict()
    max_length = 100000 if not align else min([len(val) for key, val in bucket.items()])

    for key, val in bucket.items():
        if align:
            val = val[:max_length]
        val = evaluator(val)
        ret[key] = sum(val) / len(val)
        if print_out:
            print(val)
            print(key, sum(val) / len(val), len(val))
            print()
    return ret

def cal_inner(bucket_results, key_dict):
    inner_score = 0
    for i in range(1,len(key_dict)):
        inner_score += bucket_results[key_dict[i]] - bucket_results[key_dict[i-1]]
    return inner_score

def cal_intra(bucket_gold, bucket_pred, relative=True):
    diffs = []

    for key in LENGTH_KEYS_ORDERED:
        length = len(bucket_gold[key])
        for i in range(length):
            gold = bucket_gold[key][i]
            pred = bucket_pred[key][i]
            diff = math.fabs(gold - pred) / gold
            diffs.append(diff)

    ret = sum(diffs)/len(diffs)
    return ret

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])

def get_length_score(samples, ):
    bucket_len = defaultdict(list)
    bucket_gold = defaultdict(list)
    for sample in samples:
        bucket_len[sample['length']].append(get_length_values([sample['prediction']])[0])
        bucket_gold[sample['length']].append(get_length_values([sample['summary']])[0])

    intra_score = cal_intra(bucket_gold, bucket_len)
    return intra_score

def get_length_value_one_sample(sample):
    return get_length_values([sample['prediction']])[0]
