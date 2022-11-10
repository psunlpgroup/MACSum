from tqdm import tqdm
import nltk
from nltk import word_tokenize, stem
from nltk.corpus import stopwords
import json
import string
import os
from collections import defaultdict
from metrics.rouge.evaluator import EvaluateTool
import math

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
