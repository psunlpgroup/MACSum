from ext_only import get_ext_score
from length_only import get_length_score
from rouge_only import get_rouge_avg
from spe_only import get_spe
from topic_only import get_topic_score
import os
import json

folder = "/home/yfz5488/PCS/output/BART-large_cnndm_async_concat_2022-10-27"
def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])


if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    best_dev_score = 10000
    best_epoch = 0
    for data_path in path_dict:
        with open(data_path, 'r', encoding='utf-8') as file:
            # print(data_path.split('/')[-2:])
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch), 2)

            if int(epoch) > 200 or int(epoch) % 5 != 0 :
                continue
            data = json.load(file)

            # eval specificity
            dev_scores = [get_length_score(data[:554]), get_ext_score(data[:554]), get_spe(data[:554]),
                          get_topic_score(data[:554]), 1 - get_rouge_avg(data[:554])]
            test_scores = [get_length_score(data[554:]), get_ext_score(data[554:]), get_spe(data[554:]),
                          get_topic_score(data[554:]), 1 - get_rouge_avg(data[554:])]

            if sum(dev_scores) < best_dev_score:
                best_dev_score = sum(dev_scores)
                best_epoch = epoch
                print("best epoch refreeh!", best_epoch)

            results_dict[epoch] = (dev_scores, test_scores)
            print(epoch, sum(dev_scores), sum(test_scores))

    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y[0], y[1])