'''
Evaluate the model
'''

import sys
import subprocess
import numpy as np
from collections import Counter
import torch
from torch.autograd import Variable
from data import get_minibatch

def bleu_stats(hypothesis, reference):
  """Compute statistics for BLEU."""
  stats = []
  stats.append(len(hypothesis))
  stats.append(len(reference))
  for n in xrange(1, 5):
      s_ngrams = Counter(
          [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
      )
      r_ngrams = Counter(
          [tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)]
      )
      stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
      stats.append(max([len(hypothesis) + 1 - n, 0]))
  return stats

def bleu(stats):
  """Compute BLEU given n-gram statistics."""
  if len(filter(lambda x: x == 0, stats)) > 0:
      return 0
  (c, r) = stats[:2]
  log_bleu_prec = sum(
      [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
  ) / 4.
  return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
  """Get validation BLEU score for dev set."""
  stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
  for hyp, ref in zip(hypotheses, reference):
      stats += np.array(bleu_stats(hyp, ref))
  return 100 * bleu(stats)

def get_bleu_moses(hypotheses, references, tmp_dir):
  '''
  get BLEU score with moses bleu score
  '''
  with open(tmp_dir + 'tmp_hypotheses.txt', 'w') as f:
    for hypothesis in hypotheses:
      f.write(' '.join(hypothesis) + '\n')

  with open(tmp_dir + 'tmp_reference.txt', 'w') as f:
    for reference in references:
      f.write(' '.join(reference) + '\n')
  
  hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
  pipe = subprocess.Popen(
    ['perl', 'multi-bleu.perl', '-lc', tmp_dir + 'tmp_reference.txt'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
  )
  pipe.stdin.write(hypothesis_pipe)
  pipe.stdin.close()
  score_in_str = pipe.stdout.read()
  scores = [] 
  scores.append(float(score_in_str.split(',')[0].split('=')[1].strip()))
  ngram_scores = score_in_str.split()[3].split('/')
  scores.append(float(ngram_scores[0]))
  scores.append(float(ngram_scores[1]))
  scores.append(float(ngram_scores[2]))
  scores.append(float(ngram_scores[3]))
  return scores

def decode_minibatch(config, model, pred_targ, cur_batch):
  '''
  decode a minibatch greedily
  '''
  for mid in xrange(config['data']['max_trg_length']):
    decoder_logit = model(cur_batch['source_input'], pred_targ)
    word_probs = model.decode(decoder_logit)
    decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
    next_preds = Variable(
      torch.from_numpy(decoder_argmax[:, -1])
    )
    if torch.cuda.is_available():
      next_preds = next_preds.cuda()
    pred_targ = torch.cat((pred_targ, next_preds.unsqueeze(1)), 1)
  return pred_targ


def evaluate_model_by_bleu(model, word_map, data, config):
  '''
  evaluate model using bleu score
  '''
  preds, ground_truths = [], []
  for data_idx in xrange(0, len(data['src_data']), config['data']['batch_size']):
    cur_batch = get_minibatch(word_map, data, data_idx, config)
    
    # initialize target with <s> for every title
    pred_targ = Variable(torch.LongTensor(
      [
        [word_map['targ_word2id']['<start>']]
        for i in xrange(cur_batch['target_input'].size(0))
      ]
    ))
    if torch.cuda.is_available():
      pred_targ = pred_targ.cuda()

    # decode a minibatch greedily
    pred_targ = decode_minibatch(
      config, model, 
      pred_targ, cur_batch, 
    )

    # copy minibatch outputs to cpu (if its in gpu) and convert ids to words
    pred_targ = pred_targ.data.cpu().numpy()
    pred_targ = [
      [word_map['targ_id2word'][x] for x in line]
      for line in pred_targ
    ]

    # do the same for gold sentences
    gold_targ = cur_batch['target_output'].data.cpu().numpy()
    gold_targ = [
      [word_map['targ_id2word'][x] for x in line]
      for line in gold_targ
    ]

    # process outputs
    for title_pred, title_gold in zip(pred_targ, gold_targ):
      if '<end>' in title_pred:
        index = title_pred.index('<end>')
      else:
        index = len(title_pred)
      preds.append(title_pred[1:index+1])
      if '<end>' in title_gold:
        index = title_gold.index('<end>')
      else:
        index = len(title_gold)
      ground_truths.append(title_gold[:index])

    if config['training']['test_run']:
      break
      
  return get_bleu_moses(preds, ground_truths, config['data']['save_dir'])

def evaluate_model_by_perplexity(model, word_map, data, loss_criterion, config):
  '''
  evaluate model using perplexity score
  '''
  losses = []
  for data_idx in xrange(0, len(data['src_data']), config['data']['batch_size']):
    cur_batch = get_minibatch(word_map, data, data_idx, config)
    source_input = Variable(cur_batch['source_input'].data, volatile=True)
    target_input = Variable(cur_batch['target_input'].data, volatile=True)
    target_output = Variable(cur_batch['target_output'].data, volatile=True)
    decoder_logit = model(source_input, target_input)
    loss = loss_criterion(
      decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
      target_output.view(-1)
    )
    losses.append(loss.data[0])
    if config['training']['test_run']:
      break
  return np.exp(np.mean(losses))




  