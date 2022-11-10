import os
from metrics.rouge.ThirdParty.ROUGE import pyrouge
from nltk import sent_tokenize
from third_party.rouge.rouge import Rouge
import shutil

def make_html_safe(s):
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    return s


def rouge(ref, hyp, log_path):
    assert len(ref) == len(hyp)
    ref_dir = os.path.join(log_path, 'reference')
    cand_dir = os.path.join(log_path, 'candidate')
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)
    if not os.path.exists(cand_dir):
        os.makedirs(cand_dir)
    for i in range(len(ref)):
        with open(os.path.join(ref_dir, "%06d_reference.txt" % i), 'w', encoding='utf-8') as f:
            tokenized_ref = sent_tokenize(ref[i])
            tokenized_ref = '\n'.join(tokenized_ref)
            f.write(make_html_safe(tokenized_ref) + '\n')
        with open(os.path.join(cand_dir, "%06d_candidate.txt" % i), 'w', encoding='utf-8') as f:
            tokenized_cand = sent_tokenize(hyp[i])
            tokenized_cand = '\n'.join(tokenized_cand)
            f.write(make_html_safe(tokenized_cand) + '\n')

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = ref_dir
    r.system_dir = cand_dir
    rouge_results = r.convert_and_evaluate()
    scores = r.output_to_dict(rouge_results)
    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    # print("F_measure: %s Recall: %s Precision: %s\n"
    #       % (str(f_score), str(recall), str(precision)))

    # remember to delete folder
    # with open(ref_dir + "rougeScore", 'w+', encoding='utf-8') as f:
    #     f.write("F_measure: %s Recall: %s Precision: %s\n"
    #             % (str(f_score), str(recall), str(precision)))

    # print("deleting {}".format(ref_dir))
    shutil.rmtree(ref_dir)
    shutil.rmtree(cand_dir)

    return {"f1":f_score[:], "recall": recall[:], "precision": precision[:]}


def readline_aslist(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip().replace('\n',''))
    return data


from collections import defaultdict

def compute_exact_match(pred, gold):
    return pred == gold['seq_out']

class EvaluateTool(object):
    def __init__(self, args=None):
        self.args = args

    def evaluate_list(self, preds, golds):
        ref = golds
        hypo = preds

        log_path = "./test_log/"
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        results = rouge(ref, hypo, log_path)
        shutil.rmtree(log_path)

        return results

    def evaluate_list_fast(self, pred, golds, metrics=None):
        if metrics is None:
            metrics = ['rouge-1', 'rouge-2','rouge-L']
        rouge_scorer = Rouge(metrics=metrics)
        scores = rouge_scorer.get_scores(pred, golds)
        return scores

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            if len(pred) == 0:
                score_avg = 0
            else:
                rouges = self.evaluate_list_fast([pred], [gold['seq_out']])
                score_avg = (rouges[0]['rouge-1']['f'] + rouges[0]['rouge-2']['f'] + rouges[0]['rouge-l']['f']) / 3
            eval_dict["exact_match"] += score_avg
        for key in eval_dict:
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict

    def evaluate_old(self,preds, golds, section):
        eval_dict = defaultdict(float)
        golds = [gold['seq_out'] for gold in golds]
        rouge_scores = self.evaluate_list(preds, golds)
        eval_dict["exact_match"] = (rouge_scores["f1"][0] + rouge_scores["f1"][1] + rouge_scores["f1"][2])/3
        return eval_dict