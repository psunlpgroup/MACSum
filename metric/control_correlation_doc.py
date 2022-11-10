import os
from length_only import get_length_value_one_sample
from ext_only import get_ext_value_one_sample
from spe_only import get_spe_value_one_sample
import json
from rouge_only import get_rouge_separate

length_dic = {"short": 0, "normal": 1, "long": 2}
ext_dic = {"normal": 0, "high": 1, "fully": 2}
spe_dic = {"normal": 0, "high": 1}
DEV_SIZE = 554

pathes = ["/data/yfz5488/PCS/output/BART-large_cnndm_async_load_2022-08-05/predictions_eval_150.2805049088359.json", ]


def cal_cv(pre_sample, cur_sample, type):
    if type == 'len':
        pre_len = length_dic[pre_sample['length']]
        cur_len = length_dic[cur_sample['length']]
        pre_score = get_length_value_one_sample(pre_sample)
        cur_score = get_length_value_one_sample(cur_sample)
        return (pre_score - cur_score) / (pre_len - cur_len)

    if type == 'ext':
        pre_len = ext_dic[pre_sample['extractiveness']]
        cur_len = ext_dic[cur_sample['extractiveness']]
        pre_score = get_ext_value_one_sample(pre_sample)
        cur_score = get_ext_value_one_sample(cur_sample)
        return (pre_score - cur_score) / (pre_len - cur_len)

    if type == 'spe':
        pre_len = ext_dic[pre_sample['specificity']]
        cur_len = ext_dic[cur_sample['specificity']]
        pre_score = get_ext_value_one_sample(pre_sample)
        cur_score = get_ext_value_one_sample(cur_sample)
        return (pre_score - cur_score) / (pre_len - cur_len)

    return NotImplementedError()


def avg(nums: list):
    return sum(nums) / len(nums)


if __name__ == '__main__':
    for path in pathes:
        data = json.load(open(path))
        data = data[DEV_SIZE:] # test set is the second part of 1101 samples
        print(path, "samples:", len(data))
        # first check the rouge scores
        print("ROUGE R1/2/L are:", get_rouge_separate(data))
        len_cvs = []
        ext_cvs = []
        spe_cvs = []
        for i, sample in enumerate(data):
            if i == 0:
                continue
            previous_doc = data[i - 1]['text_in']
            previous_tpk = data[i - 1]['topic']
            previous_len = data[i - 1]['length']
            previous_ext = data[i - 1]['extractiveness']
            previous_spe = data[i - 1]['specificity']
            cur_doc = sample['text_in']
            cur_tpk = sample['topic']
            cur_len = sample['length']
            cur_ext = sample['extractiveness']
            cur_spe = sample['specificity']
            if previous_tpk != cur_tpk or previous_doc != cur_doc:
                continue

            # eval CVs
            if previous_len != cur_len:
                len_cvs.append(cal_cv(sample, data[i - 1], type="len"))
            if previous_ext != cur_ext:
                ext_cvs.append(cal_cv(sample, data[i - 1], type='ext'))
            if previous_spe != cur_spe:
                spe_cvs.append(cal_cv(sample, data[i - 1], type='spe'))

        # cal avg
        print("Control Variance:")
        print("Length:", avg(len_cvs))
        print("Extractiveness:", avg(ext_cvs))
        print("Specificity:", avg(spe_cvs))
