# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from .bilstm import BiLSTM
from .dropout import SharedDropout, IndependentDropout
class CHAR_LSTM_D(nn.Module):

    def __init__(self, n_chars, n_embed, n_out, pad_index=0):
        super(CHAR_LSTM_D, self).__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)
        # the lstm layer
        #self.lstm = nn.LSTM(input_size=n_embed,
                            #hidden_size=n_out//2,
                            #batch_first=True,
                            #bidirectional=True)
        self.lstm = BiLSTM(input_size=int(n_embed), hidden_size=int(n_out/2), num_layers=1, dropout=0.33, sflag=False, isd=True)
        self.lstm_dropout = SharedDropout(p=0.33)
        self.emb_dropout = IndependentDropout(p=0.33)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.n_chars}, {self.n_embed}, "
        s += f"n_out={self.n_out}, "
        s += f"pad_index={self.pad_index}"
        s += ')'

        return s

    def forward(self, x):
        mask = x.ne(self.pad_index)
        lens = mask.sum(dim=1)
        x = pack_padded_sequence(self.emb_dropout(self.embed(x))[0], lens, True, False)
        
        x, _ = self.lstm(x)
        
        pad_seq,_ = pad_packed_sequence(x,batch_first=True)
        
        #pad_seq = self.lstm_dropout(pad_seq)
        
        hidden = pad_seq[torch.arange(len(pad_seq)), lens-1]
        #hidden = torch.cat(torch.unbind(hidden), dim=-1)

        return hidden
