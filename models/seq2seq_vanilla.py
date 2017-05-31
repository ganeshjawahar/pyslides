'''
Sequence to Sequence - Vanilla Model
'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np

class Seq2Seq(nn.Module):
  '''
  Container module with encoder, decoder, embeddings
  '''

  def __init__(self, 
      src_emb_dim,
      targ_emb_dim,
      src_vocab_size,
      targ_vocab_size,
      src_hid_dim,
      targ_hid_dim,
      batch_size,
      bidirectional,
      pad_token_src,
      pad_token_targ,
      num_layers_src,
      num_layers_targ,
      dropout
    ):
    '''
    initialize model
    '''
    super(Seq2Seq, self).__init__()
    self.src_emb_dim = src_emb_dim
    self.targ_emb_dim = targ_emb_dim
    self.src_vocab_size = src_vocab_size
    self.targ_vocab_size = targ_vocab_size
    self.src_hid_dim = src_hid_dim
    self.targ_hid_dim = targ_hid_dim
    self.batch_size = batch_size
    self.bidirectional = bidirectional
    self.pad_token_src = pad_token_src
    self.pad_token_targ = pad_token_targ
    self.num_layers_src = num_layers_src
    self.num_layers_targ = num_layers_targ
    self.dropout = dropout
    self.num_directions = 2 if self.bidirectional else 1

    # word embeddings
    self.src_embedding = nn.Embedding(
      self.src_vocab_size,
      self.src_emb_dim,
      self.pad_token_src
    )
    self.targ_embedding =  nn.Embedding(
      self.targ_vocab_size,
      self.targ_emb_dim,
      self.pad_token_targ
    )

    # encoder
    self.encoder = nn.LSTM(
      self.src_emb_dim,
      self.src_hid_dim,
      self.num_layers_src,
      bidirectional = self.bidirectional,
      batch_first = True,
      dropout = self.dropout
    )

    # decoder
    self.decoder = nn.LSTM(
      self.targ_emb_dim,
      self.targ_hid_dim,
      self.num_layers_targ,
      batch_first = True,
      dropout = self.dropout
    )

    self.encoder2decoder = nn.Linear(
      self.src_hid_dim * self.num_directions,
      self.targ_hid_dim
    )

    self.decoder2vocab = nn.Linear(self.targ_hid_dim, self.targ_vocab_size)
    if torch.cuda.is_available():
      self.decoder2vocab = self.decoder2vocab.cuda()

    self.init_weights()

  def init_weights(self):
    '''
    initialize weights
    '''
    initrange = 0.1
    self.src_embedding.weight.data.uniform_(-initrange, initrange)
    self.targ_embedding.weight.data.uniform_(-initrange, initrange)
    self.encoder2decoder.bias.data.fill_(0)
    self.decoder2vocab.bias.data.fill_(0)

  def get_state(self, input):
    '''
    Get cell states and hidden states
    '''
    batch_size = input.size(0) if self.encoder.batch_first else input.size(1)
    h0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.src_hid_dim
    ))
    c0_encoder = Variable(torch.zeros(
      self.encoder.num_layers * self.num_directions,
      batch_size,
      self.src_hid_dim
    ))
    if torch.cuda.is_available():
      h0_encoder = h0_encoder.cuda()
      c0_encoder = c0_encoder.cuda()
    return h0_encoder, c0_encoder

  def forward(self, source_input, target_input):
    '''
    forward prop. input through the network
    '''
    src_emb = self.src_embedding(source_input)
    targ_emb = self.targ_embedding(target_input)

    self.h0_encoder, self.c0_encoder = self.get_state(source_input)

    src_h, (src_h_t, src_c_t) = self.encoder(
      src_emb, (self.h0_encoder, self.c0_encoder)
    )

    if self.bidirectional:
      h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
      c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
    else:
      h_t = src_h_t[-1]
      c_t = src_c_t[-1]

    decoder_h_state = nn.Tanh()(self.encoder2decoder(h_t))
    decoder_c_state = nn.Tanh()(self.encoder2decoder(c_t))

    targ_h, (_, _) = self.decoder(
      targ_emb,
      (
         decoder_h_state.view(
           self.decoder.num_layers,
           decoder_h_state.size(0),
           decoder_h_state.size(1)
         ),
         decoder_c_state.view(
           self.decoder.num_layers,
           decoder_c_state.size(0),
           decoder_c_state.size(1)
         )
      )
    )

    targ_h_reshape = targ_h.contiguous().view(
      targ_h.size(0) * targ_h.size(1),
      targ_h.size(2)
    )

    decoder_logit = self.decoder2vocab(targ_h_reshape)
    decoder_logit = decoder_logit.view(
      targ_h.size(0),
      targ_h.size(1),
      decoder_logit.size(1)
    )

    return decoder_logit

  def decode(self, logits):
    '''
    return probability distribution over words
    '''
    logits_reshape = logits.view(-1, self.targ_vocab_size)
    word_probs = F.softmax(logits_reshape)
    word_probs = word_probs.view(
      logits.size()[0],
      logits.size()[1],
      logits.size()[2]
    )
    return word_probs






 
    