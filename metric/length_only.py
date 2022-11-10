from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
import math
from collections import defaultdict

# CONSTANTS
LENGTH_KEYS_ORDERED = ['short', 'normal', 'long']
DEV_SIZE = 554
folder = "/data/yfz5488/PCS/output/BART-large_cnndm_prompt_2022-08-25"

def get_length_values(target):
    target_values = [float(len(word_tokenize(x))) for x in target]
    return target_values

def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

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

if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    for data_path in path_dict:
        with open(data_path,'r',encoding='utf-8') as file:
            # print(data_path.split('/')[-2:])
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch),2)
            if int(epoch) > 200:
                continue
            data = json.load(file)

            # eval length intra

            dev_intra_score = get_length_score(data[:DEV_SIZE])
            test_intra_score = get_length_score(data[DEV_SIZE:])
            results_dict[epoch] = (dev_intra_score, test_intra_score)
            print(epoch, dev_intra_score, test_intra_score)

    results = order_results(results_dict)
    print("epoch dev test")
    for x, y in results:
        print(x, y[0], y[1])