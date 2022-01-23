from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import copy
import logging
import numpy as np
import os
import pickle
import torch
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

from settings import parse_test_args, model_classes, init_logging
from utils import load_and_cache_examples


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def local_adapt(inputs, feature_indices, features, tmp_model, q_inputs, args, org_params):

    optimizer = optim.SGD(tmp_model.parameters(), lr=args.adapt_lr, momentum=0.9)

    tmp_model.zero_grad()
    for step in range(args.adapt_steps):
        tmp_model.train()
        params = torch.cat([torch.reshape(param, [-1]) for param in tmp_model.parameters()], 0)
        loss = tmp_model(**q_inputs)[0] \
            + args.adapt_lambda * torch.sum((org_params - params)**2)
        loss.backward()
        optimizer.step()
        tmp_model.zero_grad()

    with torch.no_grad():
        tmp_model.eval()
        outputs = tmp_model(**inputs)
        loss = 0
        
        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

#             all_results.append(result)
        
        torch.cuda.empty_cache()
        return loss, result


def test_task(task_id, task, args, model, tokenizer):

    args.data_dir = None
    dataset, examples, features = load_and_cache_examples(args, tokenizer, task, evaluate=True, output_examples=True)
    
    if not args.no_fp16_test:
        model = model.half()

    def update_metrics(loss, cur_loss):
        return cur_loss + loss

    cur_loss = 0
    all_results = []
    if args.adapt_steps >= 1:
        with torch.no_grad():
            org_params = torch.cat([torch.reshape(param, [-1]) for param in model.parameters()], 0)

        q_input_ids = pickle.load(open(os.path.join(args.output_dir, 'q_input_ids-{}'.format(task_id)), 'rb'))
        q_masks = pickle.load(open(os.path.join(args.output_dir, 'q_masks-{}'.format(task_id)), 'rb'))
        q_token_type_ids = pickle.load(open(os.path.join(args.output_dir, 'q_token_type_ids-{}'.format(task_id)), 'rb'))
        q_labelss = pickle.load(open(os.path.join(args.output_dir, 'q_labelss-{}'.format(task_id)), 'rb'))
        q_labelse = pickle.load(open(os.path.join(args.output_dir, 'q_labelse-{}'.format(task_id)), 'rb'))
    
        for i in range(len(dataset)):
            example = dataset[i]
            inputs = {
                "input_ids": torch.tensor(np.expand_dims(example[0], 0), dtype=torch.long).cuda(),
                "attention_mask": torch.tensor(np.expand_dims(example[1], 0), dtype=torch.long).cuda(),
                "token_type_ids": torch.tensor(np.expand_dims(example[2], 0), dtype=torch.long).cuda(),
            }
            feature_indices = torch.tensor(np.expand_dims(example[3], 0), dtype=torch.long).cuda()
            q_inputs = {
                "input_ids": q_input_ids[i].cuda().detach(),
                "attention_mask": q_masks[i].cuda().detach(),
                "token_type_ids": q_token_type_ids[i].cuda().detach(),
                "start_positions": q_labelss[i].cuda().detach(),
                "end_positions": q_labelse[i].cuda().detach(),
            }

            loss, result = local_adapt(inputs, feature_indices, features, copy.deepcopy(model), q_inputs, args, org_params)
            cur_loss = update_metrics(loss, cur_loss)
            all_results.append(result)

            if (i+1) % args.logging_steps == 0:
                logging.info("Local adapted {}/{} examples, test loss: {:.3f}".format(
                    i+1, len(dataset), cur_loss/(i+1)))
    else:
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batch_size)#, num_workers=args.n_workers)

        tot_n_inputs = 0
        for step, batch in enumerate(eval_dataloader):
            model.eval()
            batch = tuple(t.cuda() for t in batch)
            n_inputs = batch[0].shape[0]
            tot_n_inputs += n_inputs

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                feature_indices = batch[3]

                outputs = model(**inputs)
                loss = 0

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

            cur_loss = update_metrics(loss*n_inputs, cur_loss)
            if (step+1) % args.logging_steps == 0:
                logging.info("Tested {}/{} examples , test loss: {:.3f}".format(
                    tot_n_inputs, len(dataset), cur_loss/tot_n_inputs))
        assert tot_n_inputs == len(dataset)
                             
    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}_N{}.json".format(task_id, args.adapt_steps))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_N{}.json".format(task_id, args.adapt_steps))
    output_null_log_odds_file = None

    # Hardcode some mandatory parameters first
    args.verbose_logging = False
    args.null_score_diff_threshold = 0.0
    args.n_best_size = 20
    args.max_answer_length = 30
    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        False,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    results = squad_evaluate(examples, predictions)

    logger.info("test loss: {:.3f} , test F1: {:.3f}, test EM: {:.3f}".format(
        cur_loss / len(dataset), results['f1'], results['exact']))
    return results


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    assert train_args.output_dir == args.output_dir
    args.__dict__.update(train_args.__dict__)
    init_logging(os.path.join(args.output_dir, 'log_test.txt'))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model_config = config_class.from_pretrained(args.model_name, hidden_dropout_prob=0, attention_probs_dropout_prob=0)
    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.tasks)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).cuda()

    avg_f1 = 0
    avg_em = 0
    for task_id, task in enumerate(args.tasks):
        logger.info("Start testing {}...".format(task))
        task_results = test_task(task_id, task, args, model, tokenizer)
        avg_f1 += task_results['f1'] / len(args.tasks)
        avg_em += task_results['exact'] / len(args.tasks)
    logger.info("Average F1: {:.3f}, Average EM: {:.3f}".format(avg_f1, avg_em))


if __name__ == "__main__":
    main()
