from collections import defaultdict
from .ext_only import get_ext_score
from .length_only import get_length_score
from .rouge_only import get_rouge_avg
from .spe_only import get_spe_score
from .topic_only import get_topic_score
from .speaker_only import get_speaker_scores

def compute_CER(preds, golds):
    data = []
    for sample, pred in zip(golds, preds):
        sample['prediction'].append(pred)
        data.append(sample)
    scores = [get_length_score(data), get_ext_score(data), get_spe_score(data),
              get_topic_score(data), 1 - get_rouge_avg(data)]

    # For macdial only
    if len(data) and 'speaker' in data[0].keys():
        scores.append(get_speaker_scores(data))

    score = sum(scores) / len(scores)
    return score


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        eval_dict = defaultdict(float)
        eval_dict["exact_match"] = compute_CER(preds, golds)

        return eval_dict
