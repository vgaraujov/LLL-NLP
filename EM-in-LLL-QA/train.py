from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import logging
import numpy as np
import os
import pickle
import torch
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

from memory import Memory
from settings import parse_train_args, model_classes, init_logging
from utils import load_and_cache_examples


def query_neighbors(task_id, args, memory, test_dataset):

    eval_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=eval_sampler, batch_size=args.batch_size, num_workers=args.n_workers)
    
    q_input_ids, q_masks, q_token_type_ids, q_labelss, q_labelse = [], [], [], [], []
    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.cuda() for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        with torch.no_grad():
            cur_q_input_ids, cur_q_masks, cur_q_token_type_ids, cur_q_labelss, cur_q_labelse = memory.query(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        q_input_ids.extend(cur_q_input_ids)
        q_masks.extend(cur_q_masks)
        q_token_type_ids.extend(cur_q_token_type_ids)
        q_labelss.extend(cur_q_labelss)
        q_labelse.extend(cur_q_labelse)
        if (step+1) % args.logging_steps == 0:
            logging.info("Queried {} examples".format(len(q_masks)))
    pickle.dump(q_input_ids, open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'wb'))
    pickle.dump(q_masks, open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'wb'))
    pickle.dump(q_masks, open(os.path.join(args.output_dir, 'q_token_type_ids-{}'.format(task_id)), 'wb'))
    pickle.dump(q_labelss, open(os.path.join(args.output_dir, 'q_labelss-{}'.format(task_id)), 'wb'))
    pickle.dump(q_labelse, open(os.path.join(args.output_dir, 'q_labelse-{}'.format(task_id)), 'wb'))


def train_task(args, model, memory, train_dataset):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=args.n_workers)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataset)//10)

    model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0

    def update_parameters(loss):
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    for step, batch in enumerate(train_dataloader):
        model.train()
        batch = tuple(t.cuda() for t in batch)
        n_inputs = batch[0].shape[0]
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start_positions": batch[3],
            "end_positions": batch[4],
        }
        memory.add(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"], inputs["start_positions"], inputs["end_positions"])
        loss = model(**inputs)[0]
        update_parameters(loss)
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info("progress: {:.2f} , step: {} , lr: {:.2E} , avg batch size: {:.1f} , avg loss: {:.3f}".format(
                tot_n_inputs/len(train_dataset), step+1, scheduler.get_lr()[0], tot_n_inputs//(step+1), tot_epoch_loss/tot_n_inputs))

        if args.replay_interval >= 1 and (step+1) % args.replay_interval == 0:
            torch.cuda.empty_cache()
            del loss, batch, inputs
            batch = memory.sample(tot_n_inputs // (step + 1))
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }
            loss = model(**inputs)[0]
            update_parameters(loss)


    logger.info("Finsih training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters
    assert tot_n_inputs == len(train_dataset)


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info("args: " + str(args))

    logger.info("Initializing main {} model".format(args.model_name))
    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model_config = config_class.from_pretrained(args.model_name)
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
    memory = Memory(args)

    for task_id, task in enumerate(args.tasks):
        logger.info("Start parsing {} train data...".format(task))
        args.data_dir = None
        train_dataset = load_and_cache_examples(args, tokenizer, task, evaluate=False, output_examples=False)

        logger.info("Start training {}...".format(task))
        train_task(args, model, memory, train_dataset)
        model_save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(task_id))
        torch.save(model.state_dict(), model_save_path)
        pickle.dump(memory, open(os.path.join(args.output_dir, 'memory-{}'.format(task_id)), 'wb'))


    del model
    memory.build_tree()

    for task_id, task in enumerate(args.tasks):
        logger.info("Start parsing {} test data...".format(task))
        args.data_dir = None
        test_dataset, _, _ = load_and_cache_examples(args, tokenizer, task, evaluate=True, output_examples=True)
        logger.info("Start querying {}...".format(task))
        query_neighbors(task_id, args, memory, test_dataset)


if __name__ == "__main__":
    main()