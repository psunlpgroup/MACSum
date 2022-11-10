# encoding=utf8
from collections import defaultdict

def compute_exact_match(pred, gold):
    return pred == gold['seq_out']

class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        for pred, gold in zip(preds, golds):
            eval_dict["exact_match"] += compute_exact_match(pred, gold)
        for key in eval_dict:
            eval_dict[key] = eval_dict[key] / len(golds) if len(golds) else 0
        return eval_dict
