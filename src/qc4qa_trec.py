# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division

import io
import os
import sys
import logging
import json
import torch
import numpy as np
from infersent import InferSent
# import tensorflow as tf  # in case you use tf1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import tensorflow_hub as hub
tf.logging.set_verbosity(0)

# Set PATHs
PATH_SENTEVAL = './SentEval'  # specify SentEval root if not installed
PATH_TO_DATA = ''  # not necessary for inference
PATH_TO_W2V = 'glove.840B.300d.txt'  # path to your glove file
MODEL_PATH = 'infersent1.pkl'  # path to your infersent model
V = 1 # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import SentEval
sys.path.insert(0, PATH_SENTEVAL)
import senteval
from senteval.tools.classifier import MLP

# tensorflow session
session = tf.Session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SentEval prepare and batcher
def prepare(params, samples):
    params['infersent'].build_vocab([' '.join(s) for s in samples], tokenize=False)

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings1 = params['infersent'].encode(sentences, bsize=params['classifier']['batch_size'], tokenize=False)

    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings2 = params['google_use'](batch)
    return np.concatenate((embeddings1, embeddings2), axis=-1)

def make_embed_fn(module):
  with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    embed = hub.Module(module)
    embeddings = embed(sentences)
    session = tf.train.MonitoredSession()
  return lambda x: session.run(embeddings, {sentences: x})

def loadFile(fpath):
  qa_data = []
  qa_ids = []
  # tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
  #             'HUM': 3, 'LOC': 4, 'NUM': 5}
  with io.open(fpath, 'r', encoding='utf-8') as f:
    for example in f:
      if "header" in json.loads(example):
        continue
      paragraph = json.loads(example)
      for qa in paragraph['qas']:
        qa_data.append(qa['question'].split())
        qa_ids.append(qa['qid'])
  return qa_data, qa_ids

def loadSQuAD(fpath):
  qa_data = []
  qa_ids = []
  # tgt2idx = {'ABBR': 0, 'DESC': 1, 'ENTY': 2,
  #             'HUM': 3, 'LOC': 4, 'NUM': 5}
  with io.open(fpath, 'r', encoding='utf-8') as f:
    input_data = json.load(f)["data"]
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      for qa in paragraph["qas"]:
        qa_data.append(qa['question'].split())
        qa_ids.append(qa['id'])
  return qa_data, qa_ids

def getEmbeddings(qa_data, params):
  out_embeds = []
  for ii in range(0, len(qa_data), params['classifier']['batch_size']):
    batch = qa_data[ii:ii + params['classifier']['batch_size']]
    embeddings = batcher(params, batch)
    out_embeds.append(embeddings)
  return np.vstack(out_embeds)

def updateFile(fpath, q_type, q_ids):
  paragraphs = []
  with io.open(fpath, 'r', encoding='utf-8') as f:
    for example in f:
      if "header" in json.loads(example):
        continue
      paragraph = json.loads(example)
      paragraphs.append(paragraph)
    
  total_idx = 0
  for paragraph in paragraphs:
    for qa in paragraph['qas']:
      if qa['qid'] == q_ids[total_idx]:
        qa['q_type'] = q_type[total_idx]
        total_idx += 1
      else:
        print('Can not match qid:', q_ids[total_idx])
  
  with open(fpath[:-6]+'_classified.jsonl', 'w') as f:
    for sample in paragraphs:
      f.write(json.dumps(sample)+'\n')
  f.close()

def updateSQuAD(fpath, q_type, q_ids):
  with io.open(fpath, 'r', encoding='utf-8') as f:
    input_data = json.load(f)["data"]
  
  total_idx = 0
  for entry in input_data:
    for paragraph in entry["paragraphs"]:
      for qa in paragraph['qas']:
        if qa['id'] == q_ids[total_idx]:
          qa['q_type'] = q_type[total_idx]
          total_idx += 1
        else:
          print('Can not match qid:', q_ids[total_idx])
  
  with open(fpath[:-5]+'_classified.json', 'w') as f:
    f.write(json.dumps({"data": input_data})+'\n')
  f.close()

# Start TF session and load Google Universal Sentence Encoder
encoder = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder-large/3")

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 512, 'optim': 'rmsprop', 'batch_size': 16,
                                 'tenacity': 5, 'epoch_size': 4}
params_senteval['google_use'] = encoder

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
  # MRQA style input
  file_path = '../data/mrqa/train/HotpotQA.jsonl'
  all_qs, all_q_ids = loadFile(file_path)
  # SQuAD style input
  # file_path = '../data/squad/train-v1.1.json'
  # all_qs, all_q_ids = loadSQuAD(file_path)
  
  q_type = []
  print('File:', file_path)
  print(len(all_qs))

  params_model = {'bsize': 16, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
  model = InferSent(params_model)
  model.load_state_dict(torch.load(MODEL_PATH))
  model.set_w2v_path(PATH_TO_W2V)
  params_senteval['infersent'] = model.cuda().eval()

  clf = MLP(params_senteval['classifier'], inputdim=4096+512, nclasses=6, batch_size=16)
  clf.model.load_state_dict(torch.load('qc4qa_model.pth'))  # load classifier
  clf.model.eval()

  with torch.no_grad():  
    for i in range(0, len(all_qs), 1000):  # batching here
      qs = all_qs[i:1000+i]
      prepare(params_senteval, qs)
      embeds = getEmbeddings(qs, params_senteval)
      
      out = clf.predict(embeds)
      q_type += np.array(out).squeeze().astype(int).tolist()

    # MRQA style update
    updateFile(file_path, q_type, all_q_ids)
    # SQuAD sytle update
    # updateSQuAD(file_path, q_type, all_q_ids)