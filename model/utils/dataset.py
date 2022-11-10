import torch
from torch.utils.data import Dataset

CONTROL_IDS = ['len_short','len_normal','len_long','ext_fully','ext_normal','ext_high','spe_normal','spe_high']
class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " <\s> "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                seq_in = raw_item["text_in"]  # QMSum, currently we set QMSum and CNNDM the same, i.e. we do not use ||
                if self.args.model.knowledge_usage in [None, "none"]:
                    seq_in = raw_item["text_in"]  # do noting
                elif self.args.model.knowledge_usage == 'concatenate':
                    seq_in = "{} ; {}".format(raw_item["struct_in"], raw_item["text_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
                # # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                # index = raw_item["text_in"].index(self.conv_sep)
                # if self.args.model.knowledge_usage == 'concatenate':
                #     # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                #     seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                #                                                                   raw_item["struct_in"],
                #                                                                   raw_item["text_in"][index + len(self.conv_sep):])
                # elif self.args.model.knowledge_usage == 'separate':
                #     # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                #     seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                #                                        raw_item["text_in"][index + len(self.conv_sep):])
                # else:
                #     raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage in [None, "none"]:
                    seq_in = raw_item["text_in"]
                elif self.args.model.knowledge_usage == 'concatenate':
                    seq_in = "{} ; {}".format(raw_item["struct_in"], raw_item["text_in"])

                    # ablation study:
                    # length_dict = {"short":0, "normal":1, "long":2}
                    # seq_in = seq_in.replace(f"Length : {raw_item['length']}", f"{length_dict[raw_item['length']]}")
                    # seq_in = seq_in.replace(f"Length : {raw_item['length']}", f"")
                elif self.args.model.knowledge_usage == 'separate':
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage in [None, "None"]:
                seq_in = ""  # do noting
            elif self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "output control attributes"
                seq_in = "{}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # Concatenate description.
        if self.args.model.use_description and self.args.model.concatenate_description and len(raw_item["topic"]):
            if "speaker" in raw_item.keys():
                seq_in = "Topic : {} ; Speaker : {} ; {}".format(raw_item["topic"], raw_item["speaker"], seq_in) # QMSum
            else:
                seq_in = "Topic : {} ; {}".format(raw_item["topic"], seq_in) # CNNDM has topic in it


        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        # Map the control attributes to ids
        control_ids = [CONTROL_IDS.index(f"len_{raw_item['length']}"),
                       CONTROL_IDS.index(f"ext_{raw_item['extractiveness']}"),
                       CONTROL_IDS.index(f"spe_{raw_item['specificity']}")]

        input_ids = tokenized_question_and_schemas.data["input_ids"] + control_ids
        item = {
            'input_ids': torch.LongTensor(input_ids), # Generation cannot pass other info??
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
            'control_ids': control_ids
        }

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description: # CNNDM does not map_description
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate': # CNNDM does not use knowledge
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.tokenizer.model_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)
