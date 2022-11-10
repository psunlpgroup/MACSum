from tqdm import tqdm
import nltk
from nltk import word_tokenize, stem
from nltk.corpus import stopwords
import json
import string
import os
from collections import defaultdict
from rouge.evaluator import EvaluateTool
import math

folder = "/data/yfz5488/PCS/output/BART-large_qmsum_async_2022-08-25"
DEV_SIZE = 324
stop_words = set(stopwords.words('english'))
stop_words.add("disfmarker")
stop_words.add("vocalsound")
punctuation = set(x for x in string.punctuation)
stop_words |= punctuation

def get_speaker_values(speakers, sources, predictions):
    return [get_speaker_value(x,y,z) for x,y,z in zip(speakers, sources, predictions)]

def get_speaker_value(speakers, source, prediction):
    speakers = [x.strip() for x in speakers.replace(';',',').replace(':', ',').split(',') if len(x.strip())]
    cnt_speaker = 0
    cnt_non_speaker = 0
    source = source.lower().split(':')
    related = ""
    non_related = ""

    for speaker in speakers:
        for i, sentence in enumerate(source):
            if speaker.lower() in sentence.lower():
                if i != len(source) - 1:
                    related += source[i+1]
            else:
                if i != len(source)-1:
                    non_related+= source[i+1]

    related = [x.lower() for x in word_tokenize(related) if x.lower() not in stop_words]
    non_related = [x.lower() for x in word_tokenize(non_related) if x.lower() not in stop_words]
    prediction = [x.lower() for x in word_tokenize(prediction.lower()) if x.lower() not in stop_words]

    for token in prediction:
        if token in non_related:
            cnt_non_speaker+=1
        if token in related:
            cnt_speaker+=1

    opt1 = (cnt_speaker-cnt_non_speaker)/len(prediction)
    opt2 = cnt_speaker/len(prediction)
    return opt2
def load_folder():
    path_dict = []
    for path_name in os.listdir(folder):
        if 'predictions_eval' in path_name:
            path_dict.append(os.path.join(folder, path_name))
    return path_dict

def order_results(results_dict):
    return sorted(results_dict.items(), key=lambda x:x[0])

def get_speaker_scores(data):
    speaker_scores = []
    speaker_gold = []
    for sample in data:
        if len(sample['speaker']) == 0:
            continue
        speaker_scores.append(get_speaker_value(sample['speaker'], sample['text_in'], sample['prediction']))
        speaker_gold.append(get_speaker_value(sample['speaker'], sample['text_in'], sample['summary']))
    speaker_scores = sum([math.fabs(x-y)/(x+0.1) for x, y in zip(speaker_gold, speaker_scores)]) / len(speaker_scores)
    return speaker_scores


if __name__ == '__main__':
    path_dict = load_folder()
    results_dict = {}
    for data_path in path_dict:

        with open(data_path,'r',encoding='utf-8') as file:
            epoch = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            epoch = round(float(epoch), 2)
            # if int(epoch) != 100:
            #     continue
            # epoch = data_path
            if int(epoch) > 200 or int(epoch) % 20 != 0:
                continue

            data = json.load(file)
            dev_data = data[DEV_SIZE:]
            test_data = data[:DEV_SIZE]

            results_dict[epoch] = (get_speaker_scores(dev_data), get_speaker_scores(test_data))
            print(epoch, results_dict[epoch])

    print('\n\n')
    results = order_results(results_dict)
    for x, y in results:
        print(x, y)