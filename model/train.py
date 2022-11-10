import logging
import os
import math

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import utils.tool
from utils.configue import Configure
from utils.dataset import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)


def main() -> None:
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)
    print(args.bert)
    if 'checkpoint-???' in args.bert.location:
        args.bert.location = get_last_checkpoint(
            os.path.dirname(args.bert.location.model_name_or_path))
        logger.info(f"Resolve model_name_or_path to {args.bert.location.model_name_or_path}")

    if "wandb" in training_args.report_to and training_args.local_rank <= 0:
        import wandb

        init_args = {}
        if "MLFLOW_EXPERIMENT_ID" in os.environ:
            init_args["group"] = os.environ["MLFLOW_EXPERIMENT_ID"]
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "uni-frame-for-knowledge-tabular-tasks"),
            name=training_args.run_name,
            # entity='sgt',
            **init_args,
        )
        wandb.config.update(training_args, allow_val_change=True)

    # Detect last checkpoint, but only used when not overwrite past dir
    last_checkpoint = None
    if os.path.isdir(
            training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    os.makedirs(training_args.output_dir, exist_ok=True)

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    if not args.arg_paths:
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                         name=args.dataset.dataset_version,
                                                                         cache_dir=args.dataset.data_store_path,
                                                                         )
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root=args.dataset.data_store_path)
    else:
        cache_root = os.path.join(training_args.output_dir, 'cache')
        os.makedirs(cache_root, exist_ok=True)
        # load different tasks here
        meta_tuning_data = {}
        for task, arg_path in args.arg_paths:
            task_args = Configure.Get(arg_path)
            task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
                path=task_args.dataset.loader_path,
                cache_dir=task_args.dataset.data_store_path)
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(task_raw_datasets_split, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        # combine them using meta_tuning constructor, now dataset is flatten to a list of data, only arg_path remains
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    model_tokenizer = model.tokenizer

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")
    print("train size:",len(seq2seq_train_dataset))
    print("dev size:", len(seq2seq_eval_dataset))
    if seq2seq_test_dataset is not None:
        print("test size:", len(seq2seq_test_dataset))
    print("data loaded!")
    if args.model.prefix_len in [None, False]:
        print("Length control in prefix is disabled.")

    # We wrap the "string" seq2seq data into "tokenized tensor".
    # In these TokenizedDataset, combine the info in listed dataset, such as description to the front
    # But it does not mention arg_path and prompts? Yes, it has no multiprefix
    train_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset) if seq2seq_train_dataset else None
    eval_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset) if seq2seq_eval_dataset else None
    test_dataset = TokenizedDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None

    # Set async learning
    optimizer = None
    scheduler = None
    if args.lr is not None and args.lr.lr_async is True:
        # Set optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        prefix_decay = ['wte', 'control_trans']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        (not any(nd in n for nd in no_decay)) and (not any(nd in n for nd in prefix_decay))],
             'weight_decay': args.lr.weight_decay_lm,
             'lr': args.lr.lm_lr},
            {'params': [p for n, p in model.named_parameters() if
                        (not any(nd in n for nd in no_decay)) and (any(nd in n for nd in prefix_decay))],
             'weight_decay': args.lr.weight_decay_prefix},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': args.lr.lm_lr}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr.prefix_lr, eps=training_args.adam_epsilon)
        # Set scheduler
        num_update_steps_per_epoch = len(train_dataset) // training_args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_training_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=num_training_steps
        )

    # Initialize our Trainer
    trainer = EvaluateFriendlySeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        optimizers=(optimizer, scheduler)
    )
    print('Trainer build successfully.')


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    else:
        # we directly load the parameters from hub
        assert isinstance(training_args.resume_from_checkpoint, str)
        if 'facebook/bart' in training_args.resume_from_checkpoint:
            trainer.model.from_pretrained(training_args.resume_from_checkpoint)
        else:
            state_dict = torch.load(os.path.join(training_args.resume_from_checkpoint, transformers.WEIGHTS_NAME), map_location="cpu")
            trainer.model.load_state_dict(state_dict, strict=True)
            # release memory
            del state_dict

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="eval"
        )
        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=test_dataset if test_dataset else eval_dataset,
            test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
            metric_key_prefix="predict"
        )
        metrics = predict_results.metrics
        max_predict_samples = len(test_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(test_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
