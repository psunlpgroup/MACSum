from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        train_dataset = MACCNNDMDataset(self.args, raw_datasets['train'])
        test_dataset = MACCNNDMDataset(self.args, raw_datasets['test'])
        dev_dataset = MACCNNDMDataset(self.args, raw_datasets['validation'])

        return train_dataset, dev_dataset, test_dataset

class MACCNNDMDataset(Dataset):

    def __init__(self, args, raw_datasets):
        self.args = args

        self.extended_data = []
        for raw_data in raw_datasets:
            out_control = f" Length : {raw_data['length']} ; Extractiveness : {raw_data['extractiveness']} ; Specificity : {raw_data['specificity']}"
            raw_data.update({"text_in": raw_data['article'],
                             "seq_out": raw_data['summary'],
                             "struct_in": out_control,
                             "topic": raw_data['topic'],
                             "specificity": raw_data["specificity"],
                             "extractiveness": raw_data["extractiveness"],
                             "length": raw_data['length']})
            self.extended_data.append(raw_data)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
