'''
Utilities for working with data
'''

import json
import sys
import random
import operator
import torch
from torch.autograd import Variable

def init_seed_data(seed):
  random.seed(seed)

def read_config(file_path):
  '''
  read JSON config.
  '''
  json_obj = json.load(open(file_path, 'r'))
  return json_obj

def construct_vocab(config):
  '''
  construct a vocabulary from training file
  '''
  src_data, src_vocab = [], {}
  targ_data, targ_vocab = [], {}
  r = open(config['data']['train_file'], 'r')
  for line in r:
    content = line.strip().split('\t')
    targ_slide_info = content[1].split('$$$')
    
    # handle source tokens
    source = targ_slide_info[2]
    for word in source.split():
      if word not in src_vocab:
        src_vocab[word] = 1
      else:
        src_vocab[word] = src_vocab[word] + 1
    src_data.append(source.split())

    # handle target tokens
    target = targ_slide_info[1]
    for word in target.split():
      if word not in targ_vocab:
        targ_vocab[word] = 1
      else:
        targ_vocab[word] = targ_vocab[word] + 1
    targ_data.append(target.split())
  r.close()
  print('read %d & %d unique source and target tokens' % (len(src_vocab), len(targ_vocab)))
  print('read %d lines from file %s' % (len(src_data), config['data']['train_file']))

  # create source vocab
  # Discard start, end, pad and unk tokens if present already
  if '<start>' in src_vocab:
    del src_vocab['<start>']
  if '<end>' in src_vocab:
    del src_vocab['<end>']
  if '<pad>' in src_vocab:
    del src_vocab['<pad>']
  if '<unk>' in src_vocab:
    del src_vocab['<unk>']

  src_word2id = {
    '<start>' : 0,
    '<end>': 1,
    '<pad>': 2,
    '<unk>': 3
  }

  src_id2word = {
    0: '<start>',
    1: '<end>',
    2: '<pad>',
    3: '<unk>'
  }

  sorted_word2id = sorted(src_vocab.items(), key=operator.itemgetter(1), reverse=True)
  sorted_words = [word[0] for word in sorted_word2id[:config['data']['n_words_src']]]
  for idx, word in enumerate(sorted_words):
    src_word2id[word] = idx + 4
    src_id2word[idx + 4] = word

  # create target vocab
  # Discard start, end, pad and unk tokens if present already
  if '<start>' in targ_vocab:
    del targ_vocab['<start>']
  if '<end>' in targ_vocab:
    del targ_vocab['<end>']
  if '<pad>' in targ_vocab:
    del targ_vocab['<pad>']
  if '<unk>' in targ_vocab:
    del targ_vocab['<unk>']

  targ_word2id = {
    '<start>' : 0,
    '<end>': 1,
    '<pad>': 2,
    '<unk>': 3
  }

  targ_id2word = {
    0: '<start>',
    1: '<end>',
    2: '<pad>',
    3: '<unk>'
  }

  sorted_word2id = sorted(targ_vocab.items(), key=operator.itemgetter(1), reverse=True)
  sorted_words = [word[0] for word in sorted_word2id[:config['data']['n_words_trg']]]
  for idx, word in enumerate(sorted_words):
    targ_word2id[word] = idx + 4
    targ_id2word[idx + 4] = word

  word_map = {
    'src_word2id': src_word2id,
    'src_id2word': src_id2word, 
    'targ_word2id': targ_word2id, 
    'targ_id2word': targ_id2word 
  }
  train_master = {
    'src_data': src_data,         
    'targ_data': targ_data, 
  }

  return word_map, train_master

def read_file(file, config):
  '''
  Loads the dataset to memory
  '''
  src_data, targ_data = [], []
  r = open(file, 'r')
  for line in r:
    content = line.strip().split('\t')
    targ_slide_info = content[1].split('$$$')
    source = targ_slide_info[2]
    src_data.append(source.split())
    target = targ_slide_info[1]
    targ_data.append(target.split())
  r.close()
  print('read %d lines from file %s' % (len(src_data), file))
  data_master = {
    'src_data': src_data,
    'targ_data': targ_data
  }
  return data_master

def read_data(config):
  '''
  read data from files
  '''
  print 'constructing vocabulary...'
  word_map, train_master = construct_vocab(config)
  dev_master = read_file(config['data']['dev_file'], config)
  test_master = read_file(config['data']['test_file'], config)
  return word_map, train_master, dev_master, test_master

def get_minibatch(word_map, data, data_idx, config):
  '''
  prepare minibatch
  '''
  source_lines = [ ['<start>'] + line + ['<end>'] 
                   for line in data['src_data'][data_idx:(data_idx+config['data']['batch_size']) ] ]
  lens = [len(line) for line in source_lines]
  local_source_max_len = max(lens)
  word2id = word_map['src_word2id']
  source_input = [
    [word2id[w] if w in word2id else word2id['<unk>'] for w in line[:-1]] +
    [word2id['<pad>']] * (local_source_max_len - len(line))
    for line in source_lines
  ]
  source_mask = [
    ([1] * (l-1)) + ([0] * (local_source_max_len - l))
    for l in lens
  ]
  source_input_tensors = Variable(torch.LongTensor(source_input))
  source_mask_tensors = Variable(torch.FloatTensor(source_mask))
  
  target_lines = [ ['<start>'] + line + ['<end>'] 
                   for line in data['targ_data'][data_idx: (data_idx+config['data']['batch_size']) ] ]
  lens = [len(line) for line in target_lines]
  local_targ_max_len = max(lens)
  word2id = word_map['targ_word2id']
  target_input = [
    [word2id[w] if w in word2id else word2id['<unk>'] for w in line[:-1]] +
    [word2id['<pad>']] * (local_targ_max_len - len(line))
    for line in target_lines
  ]
  target_output = [
    [word2id[w] if w in word2id else word2id['<unk>'] for w in line[1:]] +
    [word2id['<pad>']] * (local_targ_max_len - len(line))
    for line in target_lines
  ]
  target_mask = [
    ([1] * (l-1)) + ([0] * (local_targ_max_len - l))
    for l in lens
  ]
  target_input_tensors = Variable(torch.LongTensor(target_input))
  target_output_tensors = Variable(torch.LongTensor(target_output))
  target_mask_tensors = Variable(torch.FloatTensor(target_mask))

  # ship tensors to gpu
  if torch.cuda.is_available():
    source_input_tensors = source_input_tensors.cuda()
    source_mask_tensors = source_mask_tensors.cuda()
    target_input_tensors = target_input_tensors.cuda()
    target_output_tensors = target_output_tensors.cuda()
    target_mask_tensors = target_mask_tensors.cuda()

  minibatch = {
    'source_input': source_input_tensors,
    'source_mask': source_mask_tensors,
    'target_input': target_input_tensors,
    'target_output': target_output_tensors,
    'target_mask': target_mask_tensors
  }

  return minibatch







