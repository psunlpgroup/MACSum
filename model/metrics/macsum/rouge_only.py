from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
from collections import defaultdict
from metrics.rouge.evaluator import EvaluateTool
rouge_evaluator = EvaluateTool()

# CONSTANTS
EXTRACTIVENESS_KEYS_ORDERED = ['normal', 'high', 'fully']
METRICS = ['rouge-2', 'rouge-3']
FAST = True

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])

def get_rouge_avg(data):
    ret = get_rouge_separate(data)

    return sum(ret)/len(ret)


def get_rouge_separate(data):
    rouge1_score = []
    rouge2_score = []
    rougeL_score = []

    rouge1_p = []
    rouge2_p = []
    rougeL_p = []

    rouge1_r = []
    rouge2_r = []
    rougeL_r = []

    # anyrouge cannnot use one single sample as input
    start = 0
    ending = len(data)
    # print("total samples for rouge:", ending)
    bsz = len(data)
    while start < ending:
        end = min(ending, start + bsz)
        # print(start,'to', end)

        hypo = [x['prediction'] for x in data[start:end]]
        gold = [x['summary'] for x in data[start:end]]
        if not FAST:
            rouge_scores = rouge_evaluator.evaluate_list(hypo, gold)

            rouge1_score.append(rouge_scores["f1"][0])
            rouge2_score.append(rouge_scores["f1"][1])
            rougeL_score.append(rouge_scores["f1"][2])

            rouge1_p.append(rouge_scores["precision"][0])
            rouge2_p.append(rouge_scores["precision"][1])
            rougeL_p.append(rouge_scores["precision"][2])

            rouge1_r.append(rouge_scores["recall"][0])
            rouge2_r.append(rouge_scores["recall"][1])
            rougeL_r.append(rouge_scores["recall"][2])
        else:
            rouge_scores = rouge_evaluator.evaluate_list_fast(hypo, gold)
            rouge1_score += [rouge_score["rouge-1"]['f'] for rouge_score in rouge_scores]
            rouge2_score += [rouge_score["rouge-2"]['f'] for rouge_score in rouge_scores]
            rougeL_score += [rouge_score["rouge-l"]['f'] for rouge_score in rouge_scores]

        start = end
    ret = sum(rouge1_score) / len(rouge1_score), \
          sum(rouge2_score) / len(rouge2_score), \
          sum(rougeL_score) / len(rougeL_score)

    return ret
