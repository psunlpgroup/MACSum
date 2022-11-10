from tqdm import tqdm
import nltk
from nltk import word_tokenize
import json
import os
from collections import defaultdict
from rouge.evaluator import EvaluateTool
rouge_evaluator = EvaluateTool()

# CONSTANTS
EXTRACTIVENESS_KEYS_ORDERED = ['normal', 'high', 'fully']
METRICS = ['rouge-2', 'rouge-3']

folder = "/data/yfz5488/PCS/output/BART-large_cnndm_prompt_2022-08-25"
DEV_SIZE = 554
FAST = True
def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

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

if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    for data_path in path_dict:
        topic_scores = []
        with open(data_path,'r',encoding='utf-8') as file:
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch),2)
            if int(epoch) > 200 or int(epoch) % 10 != 0:
                continue
            data = json.load(file)

            dev_rouge = get_rouge_avg(data[:DEV_SIZE])
            test_rouge = get_rouge_avg(data[DEV_SIZE:])
            # print('epoch:', epoch)
            # print('file rouge:', len(rouge1_score))
            # print("rouge1 score", sum(rouge1_score) / len(rouge1_score))
            # print("rouge2 score", sum(rouge2_score) / len(rouge2_score))
            # print("rougeL score", sum(rougeL_score) / len(rougeL_score))

            results_dict[epoch] = (dev_rouge, test_rouge)
            print(epoch, results_dict[epoch])



    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y[0],y[1])