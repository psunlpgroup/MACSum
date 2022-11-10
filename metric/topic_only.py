from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
from collections import defaultdict
from rouge.evaluator import EvaluateTool
import math

# CONSTANTS
EXTRACTIVENESS_KEYS_ORDERED = ['normal', 'high', 'fully']
METRICS = ['rouge-2', 'rouge-3']
folder = "/data/yfz5488/PCS/output/BART-large_cnndm_prompt_2022-08-25"

DEV_SIZE = 554

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

def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

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


if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    for data_path in path_dict:

        with open(data_path,'r',encoding='utf-8') as file:
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch),2)
            if int(epoch) > 200:
                continue
            data = json.load(file)

            dev_score = get_topic_score(data[:DEV_SIZE])
            test_score = get_topic_score(data[DEV_SIZE:])
            results_dict[epoch] = (dev_score,test_score)
            print(epoch, dev_score, test_score)

    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y[0], y[1])