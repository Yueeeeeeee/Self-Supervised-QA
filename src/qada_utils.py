import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import json
import numpy as np
import random
from collections import defaultdict
from abc import *


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def InfiniteSampling(n):
    # i = 0
    i = n - 1
    order = np.random.permutation(n)
    while True:
        yield order[i]
        i += 1
        if i >= n:
            np.random.seed()
            order = np.random.permutation(n)
            i = 0


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        return iter(InfiniteSampling(self.num_samples))

    def __len__(self):
        return 2 ** 31


class DirichletBatcher(metaclass=ABCMeta):
    def __init__(self, args, lengths_1_order, lengths_2_order, 
                 joint_synonym_matrix, dirichlet_ratio,
                 dirichlet_cache, require_coeff=True):
        self.args = args
        self.lengths_1_order = lengths_1_order
        self.lengths_2_order = lengths_2_order
        self.joint_synonym_matrix = joint_synonym_matrix
        self.dirichlet_ratio = dirichlet_ratio
        self.dirichlet_cache = dirichlet_cache
        self.require_coeff = require_coeff
    
    def permute(self, in_tensor, chunk_ids, permutation):
        out_tensor = []
        for chunk in permutation:
            for idx in range(len(in_tensor)):
                if chunk_ids[idx] == chunk:
                    out_tensor.append(in_tensor[idx])
        return torch.tensor(out_tensor)
    
    def batching_fn(self, batch):
        ids, masks, segments, starts, ends, all_synonyms, all_coeffs = [], [], [], [], [], [], []
        for i in range(len(batch)):
            synonym_ids = torch.zeros(len(batch[i][0]), self.args.max_synonyms)
            synonym_coeffs = torch.zeros(len(batch[i][0]), self.args.max_synonyms)
            for j in range(len(batch[i][0])):
                token = batch[i][0][j]
                mask_id = batch[i][1][j]
                segment_id = batch[i][2][j]
                if np.random.rand() > self.dirichlet_ratio or segment_id == 1 or mask_id == 0: 
                    synonym_ids[j, 0] = token
                    synonym_coeffs[j, 0] = 1
                else:
                    synonyms, coeffs = sample_dirichlet_synonyms(
                            self.args, self.lengths_1_order, self.lengths_2_order, self.joint_synonym_matrix,
                            self.dirichlet_cache, token, i*j, self.require_coeff)
                    synonym_ids[j] = synonyms
                    synonym_coeffs[j] = coeffs
            ids.append(batch[i][0])
            masks.append(batch[i][1])
            segments.append(batch[i][2])
            starts.append(batch[i][3])
            ends.append(batch[i][4])
            all_synonyms.append(synonym_ids)
            all_coeffs.append(synonym_coeffs)
        
        return torch.stack(ids), torch.stack(masks), torch.stack(segments), torch.stack(starts), \
            torch.stack(ends), torch.stack(all_synonyms).long(), torch.stack(all_coeffs)


def sample_dirichlet_synonyms(args, lengths_1_order, lengths_2_order, joint_synonym_matrix, 
                              dirichlet_cache, token, sample_idx, require_coeff=True):
    synonyms = joint_synonym_matrix[token]
    num_synonym_1_order = lengths_1_order[token]
    num_synonym_2_order = lengths_2_order[token]
    if require_coeff:
        coeffs = dirichlet_cache[(num_synonym_1_order.item(), \
            num_synonym_2_order.item())][sample_idx%(args.cache_size**2)]
    else:
        coeffs = torch.zeros(joint_synonym_matrix.shape[1])
    
    return synonyms, coeffs


def build_dirichlet_coeff_cache(args, lengths_1_order, lengths_2_order, joint_synonym_matrix):
    dirichlet_cache = {}
    for num_synonym_1_order in np.unique(lengths_1_order):
        if num_synonym_1_order == 0:
            continue
        for num_synonym_2_order in np.unique(lengths_2_order):
            alphas = [args.alpha] * 1 + [args.alpha * args.decay] * (num_synonym_1_order - 1) \
                + [args.alpha * args.decay * args.decay] * num_synonym_2_order
            alphas = alphas[:args.max_synonyms]
            dirichlet = np.random.dirichlet(alphas, args.cache_size**2).astype(np.float32)
            zeros = np.zeros((args.cache_size**2,
                joint_synonym_matrix.shape[1]-dirichlet.shape[1])).astype(np.float32)
            dirichlet_cache[(num_synonym_1_order, num_synonym_2_order)] = \
                torch.tensor(np.concatenate((dirichlet, zeros), axis=1))
    
    return dirichlet_cache


def build_joint_synonym_matrix(args, tokenizer=None):
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(
                        args.bert_model, do_lower_case=args.do_lower_case)
    
    synonym_dict_1_order, synonym_dict_2_order = build_2_order_synonym_dict(args, tokenizer)
    lengths_1_order = np.ones(len(tokenizer.vocab)).astype(int)  # as we include self
    lengths_2_order = np.zeros(len(tokenizer.vocab)).astype(int)
    joint_synonym_matrix = np.zeros((len(tokenizer.vocab), args.max_synonyms)).astype(int)
    joint_synonym_matrix[:, 0] = np.arange(len(tokenizer.vocab))
    
    for key, value in synonym_dict_1_order.items():
        lengths_1_order[key] = max(1, min(len(value), args.max_synonyms))
        joint_synonym_matrix[key, :len(value)] = np.array(value)[:args.max_synonyms].astype(int)
    for key, value in synonym_dict_2_order.items():
        lengths_2_order[key] = min(len(value), args.max_synonyms)
        start_pos = lengths_1_order[key]
        joint_synonym_matrix[key, start_pos:start_pos+len(value)] = \
            np.array(value)[:args.max_synonyms-start_pos].astype(int)
    
    lengths_1_order = torch.tensor(lengths_1_order).long()
    lengths_2_order = torch.tensor(lengths_2_order).long()
    joint_synonym_matrix = torch.tensor(joint_synonym_matrix).long()
    
    return lengths_1_order, lengths_2_order, joint_synonym_matrix


def build_2_order_synonym_dict(args, tokenizer):
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained(
                        args.bert_model, do_lower_case=args.do_lower_case)
    
    synonym_dict_1_order = defaultdict(list)
    synonym_dict = json.load(open(args.synonym_file, 'r'))
    existing_tokens = list(tokenizer.vocab.keys())
    for key in list(synonym_dict.keys()):
        if key in existing_tokens:
            tokenized_key = tokenizer.convert_tokens_to_ids(key)[0]
            synonym_dict_1_order[tokenized_key].append(tokenized_key)  # include self
            if len(synonym_dict[key]) == 0:
                continue
            cur_synonyms = synonym_dict[key]
            for synonym in cur_synonyms:
                if synonym in existing_tokens:
                    synonym_dict_1_order[tokenized_key].append(
                        tokenizer.convert_tokens_to_ids(synonym)[0])
    
    synonym_dict_2_order = defaultdict(list)
    for token in list(synonym_dict_1_order.keys()):
        synonyms_1_order = synonym_dict_1_order[token]
        for synonym in synonyms_1_order:
            synonym_dict_2_order[token] += synonym_dict_1_order[synonym].copy()
        synonym_dict_2_order[token] = list(set(synonym_dict_2_order[token]))
        if token in synonym_dict_2_order[token]:
            synonym_dict_2_order[token].remove(token)
        for synonym in synonyms_1_order:
            if synonym in synonym_dict_2_order[token]:
                synonym_dict_2_order[token].remove(synonym)
    
    return synonym_dict_1_order, synonym_dict_2_order