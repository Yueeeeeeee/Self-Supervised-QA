# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on SQuAD."""

'''The code is based on BERT in Pytorch https://github.com/huggingface/transformers'''

from abc import ABCMeta
import argparse
import collections
from collections import defaultdict
import json
import math
import os
import random
import pickle
import sys
from tqdm import tqdm, trange
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from pathlib import Path
from collections import Counter

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.modeling import BertForQuestionAnsweringQC4QA

from pytorch_pretrained_bert.optimization import BertAdam
from utils.ConfigLogger import config_logger
from utils.evaluate import f1_score, exact_match_score, metric_max_over_ground_truths
from utils.BERTRandomSampler import BERTRandomSampler

PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))

from data_utils import *
from qada_utils import *


def prediction_stage(args, device, tokenizer, logger, debug=False):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(args.output_dir, args.output_model_file)
    model_state_dict = torch.load(output_model_file)
    model = BertForQuestionAnswering.from_pretrained(args.bert_model, state_dict=model_state_dict, args=args)
    model.to(device)
    # Read prediction samples
    read_limit = None
    if debug:
        read_limit = 200
    logger.info("***** Reading Prediction Samples *****")
    eval_features, eval_examples = read_features_and_examples(args, args.predict_file, tokenizer, logger,
            use_simple_feature=False, read_examples=True, limit=read_limit)
    acc, f1 = evaluation_stage(args, eval_examples, eval_features, device, model, logger)
    logger.info('***** Prediction Performance *****')
    logger.info('EM is %.5f, F1 is %.5f', acc, f1)


def evaluate_acc_and_f1(predictions, raw_data, logger, threshold=-1, all_probs=None):
    f1 = exact_match = total = 0
    eval_threshold = True
    if threshold is None or all_probs is None:
        eval_threshold = False
    for sample in raw_data:
        if (sample.qas_id not in predictions) or (eval_threshold and sample.qas_id not in all_probs):
            message = 'Unanswered question ' + sample.qas_id + ' will receive score 0.'
            logger.warn(message)
            continue
        if not eval_threshold or (eval_threshold and all_probs[sample.qas_id] >= threshold):
            ground_truths = sample.orig_answers
            prediction = predictions[sample.qas_id]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)
            total += 1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def keep_high_prob_samples(all_probs, all_features, prob_threshold, removed_feature_index, all_indices,
        keep_generated=False):
    new_train_features = []
    for i, feature in enumerate(all_features):
        if keep_generated:
            if feature.example_index not in removed_feature_index and all_probs[feature.example_index] > prob_threshold:
                feature.start_position, feature.end_position = all_indices[i][0] = all_indices[i][1]
                new_train_features.append(feature)
                removed_feature_index.add(feature.example_index)
        else:
            if all_probs[feature.example_index] > prob_threshold:
                feature.start_position, feature.end_position = all_indices[i][0], all_indices[i][1]
                new_train_features.append(feature)
    return new_train_features, removed_feature_index


def compare_performance(args, best_acc, best_f1, acc, f1, model, logger):
    if not (best_f1 is None or best_acc is None):
        if best_acc < acc:
            logger.info('Current model BEATS previous best model, previous best is EM = %.5F, F1 = %.5f',
                best_acc, best_f1)
            best_acc, best_f1 = acc, f1
            logger.info('Current best model has been saved!')
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, args.output_model_file))
        else:
            logger.info('Current model CANNOT beat previous best model, previous best is EM = %.5F, F1 = %.5f',
                best_acc, best_f1)
    else:
        best_acc, best_f1 = acc, f1
    return best_acc, best_f1


def evaluation_stage(args, eval_examples, eval_features, device, model, logger, generate_prob_th=0.6,
        removed_feature_index=None, global_step=None, best_acc=None, best_f1=None, generate_label=False):
    if not global_step:
        logger.info("***** Running Evaluation Stage *****")
    else:
        logger.info("***** Running Predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

    model.eval()
    all_results = []
    logger.info("Start evaluating")
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
             batch_start_logits, batch_end_logits, _ = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))

    if global_step:
        prediction_file_name = 'predictions_' + str(global_step) + '.json'
        nbest_file_name = 'nbest_predictions_' + str(global_step) + '.json'
        output_prediction_file = os.path.join(args.output_dir, prediction_file_name)
        output_nbest_file = os.path.join(args.output_dir, nbest_file_name)
    else:
        output_prediction_file = os.path.join(args.output_dir, 'predictions.json')
        output_nbest_file = os.path.join(args.output_dir, 'nbest_predictions.json')
    all_predictions, all_probs, all_indices = write_predictions(args, eval_examples, eval_features, all_results,
        args.n_best_size, args.max_answer_length,
        args.do_lower_case, output_prediction_file,
        output_nbest_file, args.verbose_logging, logger, args.output_prediction)
    if generate_label:
        return keep_high_prob_samples(all_probs, eval_features, generate_prob_th, removed_feature_index, all_indices,
                keep_generated=args.keep_previous_generated)
    else:
        acc, f1 = evaluate_acc_and_f1(all_predictions, eval_examples, logger)
        logger.info('Current EM is %.5f, F1 is %.5f', acc, f1)
        if not (best_f1 is None or best_acc is None):
            best_acc, best_f1 = compare_performance(args, best_acc, best_f1, acc, f1, model, logger)
            return best_acc, best_f1
        else:
            return acc, f1


def generate_self_training_samples(args, train_examples, train_features, device, model, removed_feature_index,
        new_generated_train_features, generate_prob_th, logger):
    logger.info('***** Generating training data for this epoch *****')
    if args.keep_previous_generated:
        train_features_removed_previous = []
        for index in range(len(train_features)):
            if index not in removed_feature_index:
                train_features_removed_previous.append(train_features[index])
    else:
        train_features_removed_previous = train_features
    cur_train_features, removed_feature_index = \
        evaluation_stage(args, train_examples, train_features_removed_previous, device, model, logger,
            removed_feature_index=removed_feature_index, generate_label=True, generate_prob_th=generate_prob_th)
    if len(cur_train_features) == 0:
        logger.info("  No new training samples were generated, training procedure ends")
        return None, None
    if args.keep_previous_generated:
        new_generated_train_features.extend(cur_train_features)
    else:
        new_generated_train_features = cur_train_features
    return new_generated_train_features, removed_feature_index


def get_bert_model_parameters(model):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    return optimizer_grouped_parameters


def qc_clustering_stage(args, device, model, source_train_features, target_features):
    all_input_ids = [f.input_ids for f in source_train_features]
    all_input_mask = [f.input_mask for f in source_train_features]
    all_segment_ids = [f.segment_ids for f in source_train_features]
    source_length = len(all_input_ids)

    all_input_ids = all_input_ids + [f.input_ids for f in target_features]
    all_input_mask = all_input_mask + [f.input_mask for f in target_features]
    all_segment_ids = all_segment_ids + [f.segment_ids for f in target_features]

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=args.predict_batch_size)

    model.eval()
    sent_reps = []
    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, segment_ids = batch
            sequence_output = model.bert(input_ids, segment_ids, input_masks, output_all_encoded_layers=False)[0]
            sequence_output = sequence_output[:, 0]
            sent_reps += sequence_output.cpu().numpy().tolist()
    
    from sklearn.cluster import KMeans
    sent_reps = np.array(sent_reps)
    train_indices = np.random.choice(source_length, min(args.max_cluster_samples, source_length), replace=False)
    train_samples = sent_reps[train_indices]
    kmeans = KMeans(n_clusters=args.num_clusters).fit(train_samples)
    cluster_results = kmeans.predict(sent_reps)
    
    for idx, f in enumerate(source_train_features):
        f.q_type = cluster_results[idx]
    for idx2, f in enumerate(target_features):
        f.q_type = cluster_results[idx+idx2+1]  # as we concat source and target samples

    return source_train_features, target_features


def comb_adversarial_training_stage(args, target_train_features, target_train_examples, source_train_features,
            eval_features, eval_examples, removed_feature_index, new_generated_train_features, model, epoch,
            device, best_acc, best_f1, logger):
    
    def sample_source_batch(source_data, source_q_type_dict, sample_pointer, q_types):
        output_idx = []
        for q_type in q_types.tolist():
            next_q_idx = sample_pointer[q_type] % len(source_q_type_dict[q_type])
            output_idx.append(source_q_type_dict[q_type][next_q_idx])
            sample_pointer[q_type] += 1

        input_ids_source, input_masks_source, segment_ids_source, start_positions_source, end_positions_source, \
            q_types_source = [], [], [], [], [], []
        for idx in output_idx:
            input_ids_source.append(source_data[idx][0].unsqueeze(0))
            input_masks_source.append(source_data[idx][1].unsqueeze(0))
            segment_ids_source.append(source_data[idx][2].unsqueeze(0))
            start_positions_source.append(source_data[idx][3].unsqueeze(0))
            end_positions_source.append(source_data[idx][4].unsqueeze(0))
            q_types_source.append(source_data[idx][5].unsqueeze(0))
        
        return torch.vstack(input_ids_source), torch.vstack(input_masks_source), torch.vstack(segment_ids_source), \
            torch.cat(start_positions_source, -1), torch.cat(end_positions_source, -1), torch.cat(q_types_source, -1)

    # Generate self-training samples
    new_generated_train_features, removed_feature_index = generate_self_training_samples(args, target_train_examples,
        target_train_features, device, model, removed_feature_index, new_generated_train_features, args.generate_prob_th,
        logger)
    if new_generated_train_features is None:
        sys.exit()
    
    logger.info('\n')
    logger.info('====================  Start Adversarial Training Stage  ====================')
    all_input_ids = torch.tensor([f.input_ids for f in new_generated_train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in new_generated_train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in new_generated_train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in new_generated_train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in new_generated_train_features], dtype=torch.long)
    all_q_types = torch.tensor([f.q_type for f in new_generated_train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
        all_start_positions, all_end_positions, all_q_types)
    
    source_input_ids = torch.tensor([f.input_ids for f in source_train_features], dtype=torch.long)
    source_input_mask = torch.tensor([f.input_mask for f in source_train_features], dtype=torch.long)
    source_segment_ids = torch.tensor([f.segment_ids for f in source_train_features], dtype=torch.long)
    source_start_positions = torch.tensor([f.start_position for f in source_train_features], dtype=torch.long)
    source_end_positions = torch.tensor([f.end_position for f in source_train_features], dtype=torch.long)
    source_q_types = []
    source_q_type_dict = {x:[] for x in range(args.num_clusters)}
    for idx, f in enumerate(source_train_features):
        source_q_types.append(f.q_type)
        source_q_type_dict[f.q_type].append(idx)
    source_q_types = torch.tensor(source_q_types, dtype=torch.long)
    for key in source_q_type_dict.keys():
        random.shuffle(source_q_type_dict[key])
    sample_pointer = [0] * args.num_clusters
    source_data = TensorDataset(source_input_ids, source_input_mask, source_segment_ids, source_start_positions, 
        source_end_positions, source_q_types)

    train_sampler = BERTRandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    
    data_len = len(new_generated_train_features)
    logger.info("  Num split examples = %d", data_len)
    logger.info("  Batch size = %d", args.train_batch_size)
    num_train_steps = math.ceil(data_len / args.train_batch_size / args.gradient_accumulation_steps)
    if num_train_steps == 0 and data_len > 0:
        num_train_steps = 1
    t_total = num_train_steps
    logger.info("  Num steps = %d", num_train_steps)

    loss_sum = 0
    optimizer_grouped_parameters = get_bert_model_parameters(model)
    optimizer = BertAdam(optimizer_grouped_parameters,
            lr=args.adapt_learning_rate,
            warmup=args.warmup_proportion,
            t_total=t_total)
    global_step = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        model.train()
        batch_source = sample_source_batch(source_data, source_q_type_dict, sample_pointer, batch[-1])
        batch = tuple(t.to(device) for t in batch)
        batch_source = tuple(t.to(device) for t in batch_source)
        input_ids, input_masks, segment_ids, start_positions, end_positions, q_types = batch
        input_ids_source, input_masks_source, segment_ids_source, start_positions_source, end_positions_source, q_types_source = batch_source
        input_ids = torch.cat((input_ids, input_ids_source), 0)
        input_masks = torch.cat((input_masks, input_masks_source), 0)
        segment_ids = torch.cat((segment_ids, segment_ids_source), 0)
        start_positions = torch.cat((start_positions, start_positions_source), 0)
        end_positions = torch.cat((end_positions, end_positions_source), 0)
        q_types = torch.cat((q_types, q_types_source), 0)

        loss = model.forward_ours(input_ids, segment_ids, input_masks, start_positions,
                end_positions, q_types, lambda_c=args.lambda_c)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.adapt_learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
    final_acc, final_f1 = None, None
    if epoch == args.num_train_epochs - 1:
        final_acc, final_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
            global_step=global_step, best_acc=None, best_f1=None, logger=logger)
        best_acc, best_f1 = compare_performance(args, best_acc, best_f1, final_acc, final_f1, model, logger)
    else:
        best_acc, best_f1 = evaluation_stage(args, eval_examples, eval_features, device, model,
            global_step=global_step, best_acc=best_acc, best_f1=best_f1, logger=logger)
    return best_acc, best_f1, final_acc, final_f1


def prepare_model(args, device):
    input_model_file = os.path.join(args.input_dir, args.input_model_file)
    model_state_dict = torch.load(input_model_file)
    model = BertForQuestionAnsweringQC4QA.from_pretrained(args.bert_model, state_dict=model_state_dict, args=args)
    model.to(device)
    return model


def adaptation_stage(args, tokenizer, device, logger, debug=False):
    model = prepare_model(args, device)
    best_acc, best_f1 = 0, 0
    
    read_limit = None
    if debug:
        read_limit = 50

    ## Read target training examples
    logger.info("***** Reading Target Unlabeled Training Samples *****")
    train_features, train_examples = read_features_and_examples(args, args.target_train_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)

    ## Read source training examples
    logger.info("***** Reading Source Training Samples *****")
    source_train_features, _ = read_features_and_examples(args, args.source_train_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)

    # Read evaluation samples
    logger.info("***** Reading Evaluation Samples *****")
    eval_features, eval_examples = read_features_and_examples(args, args.target_predict_file, tokenizer, logger,
        use_simple_feature=False, read_examples=True, limit=read_limit)
    
    ## perform unsupervised clustering
    logger.info("***** Clustering Question Types *****")
    source_train_features, train_features = qc_clustering_stage(args, device, model, source_train_features, train_features)

    removed_feature_index = set()
    new_generated_train_features = []
    final_acc, final_f1 = 0.0, 0.0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info('\n')
        logger.info(' ###########  Start Training Epoch %d  ###########', epoch + 1)
        logger.info('\n')
        best_acc, best_f1, final_acc, final_f1 = comb_adversarial_training_stage(args, train_features, train_examples,
                source_train_features, eval_features, eval_examples, removed_feature_index, new_generated_train_features,
                model, epoch, device, best_acc, best_f1, logger)
        logger.info('\n')
        logger.info(' ###########  End Training Epoch %d  ###########', epoch + 1)
        logger.info('\n')

    # Save the final trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, args.output_model_file + '.final')
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info('The final model has been save')
    logger.info('*** The Training Stage is Ended ***')
    logger.info('\n\nBest EM is %.5f. Best F1 is %.5f', best_acc, best_f1)
    logger.info('\n\nFinal EM is %.5f. Best F1 is %.5f', final_acc, final_f1)


def main(debug=False):
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--output_model_file", default=None, type=str, required=True,
        help="The model file which will be saved after training, it cannot be empty.")

    ## Other parameters
    parser.add_argument("--input_dir", default=None, type=str,
                        help="The output directory where the pretrained model will be loaded.")
    parser.add_argument("--input_model_file", default=None, type=str,
        help="The model file which will be loaded before training, it cannot be empty.")
    parser.add_argument("--train_file", default='../data/squad/train-v1.1_classified.json', type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default='../data/squad/dev-v1.1.json', type=str,
                        help="SQuAD json for predictions/Dev. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--target_train_file", default=None, type=str, help="Train file in target domain")
    parser.add_argument("--target_predict_file", default=None, type=str, help="Dev file in target domain")
    parser.add_argument("--source_train_file", default='../data/squad/train-v1.1_classified.json', type=str, help="Train file in source domain")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=40, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_adaptation", action='store_true', help="Whether to run unsupervised domain adaptation.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--keep_previous_generated", action='store_true', help="Whether to keep the generated"+
            "samples in previous epochs, if not every epoch it will generate new samples from whole target domain")
    parser.add_argument("--train_batch_size", default=12, type=int, help="Total batch size for training.")
    # batch size of 12 and accumulation step of 3 --> each time compute a source batch of 4 and target batch of 4
    parser.add_argument("--predict_batch_size", default=12, type=int, help="Total batch size for predictions.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers for data loader.")
    parser.add_argument("--evaluation_interval", default=2000, type=int, help="Batch interval to run evaluation.")
    parser.add_argument("--loss_logging_interval", default=500, type=int, help="Batch interval to run evaluation.")
    parser.add_argument("--adapt_learning_rate", default=2e-5, type=float,
        help="The initial learning rate for Adam in self-training in adaptation.")
    parser.add_argument("--num_train_epochs", default=4.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--logger_path', type=str, default='bert', help='The path to save log of current program')
    parser.add_argument('--use_simple_feature', type=bool, default=False,
                        help='Whether to use the feature of simplified version, >=1 means use simple feature')
    parser.add_argument('--generate_prob_th', type=float, default=0.6,
        help='The probability threshold for generating training samples in self-training')
    parser.add_argument("--use_BN", default=True, help="Whether to use Batch Normalization in the output layer.")
    parser.add_argument("--output_prediction", action='store_true', help="Whether to output the prediction json file.")

    parser.add_argument('--lambda_c', default=0.0, type=float, help='QC4QA: scaling facotor for QC4QA loss')
    parser.add_argument('--num_clusters', default=5, type=int, help='QC4QA: number for question type clusters')
    parser.add_argument('--max_cluster_samples', default=10000, type=int, help='QC4QA: number for clustering training')

    args = parser.parse_args()
    logger = config_logger(args.logger_path)
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_predict and not args.do_adaptation:
        raise ValueError("At least one of `do_adaptation` or `do_predict` must be True.")

    if args.do_adaptation:
        if not args.source_train_file or not args.target_predict_file or not args.target_train_file:
            raise ValueError(
                "If `do_adaptation` is True, then `source_train_file` , 'target_train_file' "
                "and 'target_predict_file' must be specified.")
        if not args.input_dir or not args.input_model_file:
            raise ValueError("If `do_adaptation` is True, then `input_dir` and `input_model_file` must be specified.")

    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if (not os.path.exists(args.output_dir)) and args.do_adaptation:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    if args.do_adaptation:
        adaptation_stage(args, tokenizer, device, logger, debug=debug)

    if args.do_predict:
        prediction_stage(args, device, tokenizer, logger, debug=debug)


if __name__ == "__main__":
    main(debug=False)