from squad import squad_convert_examples_to_features, SquadV1Processor, SquadV2Processor
from torch.utils.data import Subset
import numpy as np
import os
import random
import torch
import logging

logger = logging.getLogger(__name__)


def pad_to_max_len(input_ids, masks=None):
    max_len = 384
    masks = torch.tensor([[1]*len(input_id)+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    input_ids = torch.tensor([input_id+[0]*(max_len-len(input_id)) for input_id in input_ids], dtype=torch.long)
    return input_ids, masks


def load_and_cache_examples(args, tokenizer, path, evaluate=False, output_examples=False):
    
    if 'quac' in path.lower() and not args.version_2_with_negative:
        identifier = 'quac'
        train_file = os.path.join(path, "train_v0.2.json")
        predict_file = os.path.join(path, "val_v0.2.json")
    if 'squad' in path.lower() and not args.version_2_with_negative:
        identifier = 'squad'
        train_file = os.path.join(path, "train-v1.1.json")
        predict_file = os.path.join(path, "dev-v1.1.json")
    if 'web' in path.lower() and not args.version_2_with_negative:
        identifier = 'web'
        train_file = os.path.join(path, "squad-web-train.json")
        predict_file = os.path.join(path, "squad-web-dev.json")
    if 'wiki' in path.lower() and not args.version_2_with_negative:
        identifier = 'wiki'
        train_file = os.path.join(path, "squad-wikipedia-train.json")
        predict_file = os.path.join(path, "squad-wikipedia-dev.json")

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name.split("/"))).pop(),
            str(args.max_seq_length),
            identifier,
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, filename=predict_file)
        else:
            examples = processor.get_train_examples(args.data_dir, filename=train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.n_workers,
        )

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.data_ratio != 1:
        logger.info("Reducing dataset to %s", args.data_ratio)
        portion = int(args.data_ratio*len(dataset))
        dataset = Subset(dataset, list(range(portion)))
        examples = examples[:portion]
        features = features[:portion]
        assert len(dataset) == len(examples) == len(features)
    
    if output_examples:
        return dataset, examples, features
    return dataset
