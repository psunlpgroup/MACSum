import json
import datasets
import os

_DESCRIPTION = """"""
_CITATION = """"""

class BucketCNNDM(datasets.GeneratorBasedBuilder):

    def _info(self):
        features=datasets.Features(
            {
                "article": datasets.Value("string"),
                "length": datasets.Value("string"),
                "extractiveness": datasets.Value("string"),
                "specificity": datasets.Value("string"),
                "topic": datasets.Value("string"),
                "summary": datasets.Value("string"),
                # we do not need relevant turns yet
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        filepath = "./MACSum/macdoc_flatten/"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": filepath+'train.json'}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": filepath+'val.json'}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": filepath+'test.json'}),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            reader = json.load(f)
            for example_idx, example in enumerate(reader):
                ret = {
                    "article": example["article"],
                    "length": example["length"],
                    "extractiveness": example["extractiveness"],
                    "specificity": example["specificity"],
                    "topic": example["topic"],
                    "summary": example["summary"],
                }
                yield example_idx, ret
