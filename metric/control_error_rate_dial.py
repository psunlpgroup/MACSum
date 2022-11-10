from ext_only import get_ext_score
from length_only import get_length_score
from rouge_only import get_rouge_avg
from spe_only import get_spe
from topic_only import get_topic_score
from speaker_only import get_speaker_scores
import os
import json
DEV_SIZE = 324

##### https://stackoverflow.com/questions/27750608/error-installing-nltk-supporting-packages-nltk-download ####
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import nltk
nltk.download('averaged_perceptron_tagger')
############

folder = "/home/yfz5488/PCS/output/BART-large_qmsum_async_concat_2022-10-27"

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
            dev_data = data[DEV_SIZE:]
            test_data = data[:DEV_SIZE]
            dev_scores = [get_length_score(dev_data), get_ext_score(dev_data), get_spe(dev_data),
                          get_topic_score(dev_data), get_speaker_scores(dev_data), 1 - get_rouge_avg(dev_data)]
            test_scores = [get_length_score(test_data), get_ext_score(test_data), get_spe(test_data),
                           get_topic_score(test_data), get_speaker_scores(test_data), 1 - get_rouge_avg(test_data)]

            if sum(dev_scores) < best_dev_score:
                best_dev_score = sum(dev_scores)
                best_epoch = epoch
                print("best epoch refresh!", best_epoch)

            results_dict[epoch] = (dev_scores, test_scores)
            print(epoch, sum(dev_scores), sum(test_scores))

    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y[0], y[1])