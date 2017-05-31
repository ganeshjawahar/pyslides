'''
Main script to launch models
'''

import sys
import os
import math
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from data import read_config, read_data, init_seed_data, get_minibatch
from models.seq2seq_vanilla import Seq2Seq
from evaluate import evaluate_model_by_bleu, evaluate_model_by_perplexity
from utils import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument(
  "--config",
  help="path to json config",
  required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)

# Set the random seed manually for reproducibility.
torch.manual_seed(config['training']['seed'])
init_seed_data(config['training']['seed'])

print 'reading data...'
word_map, train_data, val_data, test_data = read_data(config)

weight_mask = torch.ones(len(word_map['targ_word2id']))
if torch.cuda.is_available():
  print 'using GPU...'
  weight_mask = weight_mask.cuda()
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
model = Seq2Seq( 
  src_emb_dim = config['model']['dim_word_src'],
  targ_emb_dim = config['model']['dim_word_trg'],
  src_vocab_size = len(word_map['src_word2id']),
  targ_vocab_size = len(word_map['targ_word2id']),
  src_hid_dim = config['model']['dim_rnn_src'], 
  targ_hid_dim = config['model']['dim_rnn_trg'],
  batch_size = config['data']['batch_size'],
  bidirectional = config['model']['bidirectional'],
  pad_token_src = word_map['src_word2id']['<pad>'],
  pad_token_targ = word_map['targ_word2id']['<pad>'],
  num_layers_src = config['model']['n_layers_src'],
  num_layers_targ = config['model']['n_layers_trg'],
  dropout = config['model']['dropout']              
)
optimizer = optim.SGD(model.parameters(), lr=config['training']['lrate'])
if torch.cuda.is_available():
  model = model.cuda()
  loss_criterion = loss_criterion.cuda()
  config.loss_criterion = loss_criterion

# create save directory if it does not exist
if not os.path.exists(config['data']['save_dir']):
  os.makedirs(config['data']['save_dir'])

print 'training...'
best_val_perp, best_test_perp, best_test_bleu = 0, 0, 0
for epoch in xrange(config['training']['max_epochs']):
  losses = []
  num_batches = int(math.ceil(len(train_data['src_data'])/config['data']['batch_size']))
  bar = ProgressBar('Train', max=num_batches)  
  for data_idx in xrange(0, len(train_data['src_data']), config['data']['batch_size']):
    cur_batch = get_minibatch(word_map, train_data, data_idx, config)
    decoder_logit = model(cur_batch['source_input'], cur_batch['target_input'])
    optimizer.zero_grad()
    loss = loss_criterion(
      decoder_logit.contiguous().view(-1, len(word_map['targ_word2id'])),
      cur_batch['target_output'].view(-1)
    )
    losses.append(loss.data[0])
    loss.backward()
    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    torch.nn.utils.clip_grad_norm(model.parameters(), config['training']['grad_clip'])
    optimizer.step()

    # update the progress
    bar.next()

    if config['training']['test_run']:
      break
  bar.finish()

  train_perp = np.exp(np.mean(losses))
  val_perp = evaluate_model_by_perplexity(
    model, word_map, val_data, 
    loss_criterion, config
  )
  if val_perp > best_val_perp:
    test_perp = evaluate_model_by_perplexity(
      model, word_map, test_data, 
      loss_criterion, config
    )
    test_bleu = evaluate_model_by_bleu(
      model, word_map, test_data,
      config
    )
    best_test_perp = test_perp
    best_test_bleu = test_bleu[0]
  print('Epoch: %d; Train-Perp: %.3f; Val-Perp: %.3f; Best-Val-Perp: %.3f; Best-Test-Perf: %.3f; Best-Test-Bleu: %.3f;' % (epoch, train_perp, val_perp, best_val_perp, best_test_perp, best_test_bleu))

  if config['training']['test_run']:
    break


