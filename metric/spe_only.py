from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
import math
from collections import defaultdict
from nltk.corpus import stopwords
import multiprocessing
from tqdm import tqdm

st_words = stopwords.words('english')
from rouge.evaluator import EvaluateTool

# CONSTANTS
SPECIFICITY_KEYS_ORDERED = ['normal', 'high']
folder = "/data/yfz5488/PCS/output/BART-large_cnndm_prompt_2022-08-25"
# from third_party.speciteller.python3_code.speciteller import get_paragraph_specificity
import nltk

nltk.download('averaged_perceptron_tagger')

SPE_METRICS = 'weighted'


def get_specificity_values(target, metrics=SPE_METRICS, multicore=False):
    if multicore: # some bugs in this options
        # Get all cores
        cores = multiprocessing.cpu_count()
        # start a pool
        pool = multiprocessing.Pool(processes=cores)
        tasks = [(x, SPE_METRICS) for x in target]
        # do parallel calculate
        data = pool.starmap(get_specificity_value, tasks)
        data = [x for x in data if x is not None]

        return data
    else:
        return [get_specificity_value(x) for x in target]


def get_specificity_value(target, metrics=SPE_METRICS):
    # return get_paragraph_specificity(target, metric=metrics)
    num_sent = len(nltk.sent_tokenize(target))
    target = nltk.word_tokenize(target.lower())
    target = [x for x in target if x not in st_words]
    target_pos = nltk.pos_tag(target)
    # ret_1 =  len([x for x, y in target_pos if y == 'NN'])/len(target_pos)
    # ret_2 =  len([x for x, y in target_pos if y == 'NN' and x != y])/len(target_pos)

    tot = len(target_pos)
    nn_words = [x for x, y in target_pos if y == 'NN']
    vb_words = [x for x, y in target_pos if y == 'VB']
    vbg_words = [x for x, y in target_pos if y == 'VBG']
    cd_words = [x for x, y in target_pos if y == 'CD']
    nn = len(nn_words)
    vb = len(vb_words)
    cd = len(cd_words)
    vbg = len(vbg_words)

    # metrics = (nn + cd * 2)/(1+vb) 6.6 7.7
    # metrics = cd 0.62 0.83
    # metrics = nn 6.99 7.97
    # metrics = vbg 0.64 0.70
    # metrics = vb 0.4 0.409
    # metrics = tot 21 25
    # metrics = 0.1 * vbg + 0.2 * tot + 0.3 * nn + 0.4 * cd
    metrics = (0.1 * vbg + 0.2 * tot + 0.3 * nn + 0.4 * cd) / num_sent
    # print(nn_words)
    # print(vb_words)
    # print(cd_words)
    # print(metrics)
    return metrics


def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

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
    for i in range(1, len(key_dict)):
        inner_score += bucket_results[key_dict[i]] - bucket_results[key_dict[i - 1]]
    return inner_score


def cal_intra(bucket_gold, bucket_pred, relative=True):
    diffs = []

    for key in SPECIFICITY_KEYS_ORDERED:
        length = len(bucket_gold[key])
        for i in range(length):
            gold = bucket_gold[key][i]
            pred = bucket_pred[key][i]
            diff = math.fabs(gold - pred)
            if relative:
                diff /= gold
            diffs.append(diff)

    ret = sum(diffs) / len(diffs)
    return ret


def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x: x[0])


def get_spe(data):
    bucket_len = defaultdict(list)
    bucket_gold = defaultdict(list)
    for i, sample in enumerate(data):
        bucket_len[sample['specificity']].append(get_specificity_value(sample['prediction']))
        bucket_gold[sample['specificity']].append(get_specificity_value(sample['summary']))

    intra_score = cal_intra(bucket_gold, bucket_len)
    return intra_score

def get_spe_value_one_sample(sample):
    return get_specificity_value(sample['prediction'])

if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    for data_path in path_dict:
        with open(data_path, 'r', encoding='utf-8') as file:
            # print(data_path.split('/')[-2:])
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch), 2)

            if int(epoch) > 200:
                continue
            data = json.load(file)

            # eval specificity
            dev_spe = get_spe(data[:554])
            test_spe = get_spe(data[554:])
            results_dict[epoch] = (dev_spe, test_spe)
            print(epoch, dev_spe, test_spe)

    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y[0], y[1])
