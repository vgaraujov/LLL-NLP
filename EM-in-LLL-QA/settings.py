from transformers import AutoTokenizer, BertConfig
from modeling_bert import BertForQuestionAnswering
import argparse
import datetime
import logging
import os
import shutil
import torch
import random
import numpy as np

model_classes = {
    'bert': (BertConfig, BertForQuestionAnswering, AutoTokenizer),
}


def set_device_id(args):
    args.device_id = torch.cuda.current_device() # not for multi-gpu
    torch.cuda.set_device(args.device_id)

def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # gpu vars
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

def parse_train_args():
    parser = argparse.ArgumentParser("Train Lifelong Language Learning")

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type selected in the list: " + ", ".join(model_classes.keys()))
    parser.add_argument("--n_neighbors", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_ratio", type=float, default=1)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay_interval", type=int, default=100)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=["ag_news_csv"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mem_capacity", type=float, default=1)

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--do_lower_case", 
        default=True,
        type=bool,
        help="Set this flag if you are using an uncased model."
    )

    args = parser.parse_args()

    if args.debug:
        args.logging_steps = 1
        args.data_ratio = 0.1
        args.output_dir = "output_debug"
        args.overwrite = True

    set_device_id(args)
    set_seed(args)

    memory_size = torch.cuda.get_device_properties(args.device_id).total_memory / 1e6
    if args.batch_size <= 0:
        args.batch_size = int(memory_size * 0.38)

    if os.path.exists(args.output_dir):
        if args.overwrite:
            choice = 'y'
        else:
            choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)
    return args


def parse_test_args():
    parser = argparse.ArgumentParser("Test Lifelong Language Learning")

    parser.add_argument("--adapt_lambda", type=float, default=1e-3)
    parser.add_argument("--adapt_lr", type=float, default=5e-3)
    parser.add_argument("--adapt_steps", type=int, default=30)
    parser.add_argument("--no_fp16_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output0")

    args = parser.parse_args()
    set_device_id(args)
    return args


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
          last = self.last
        except AttributeError:
          last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.3f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True

def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())

