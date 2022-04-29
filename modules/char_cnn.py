# -*- coding: utf-8 -*-
  
import torch
import torch.nn as nn


class CHAR_CNN(nn.Module):
    
    def __init__(self, n_chars, n_embed, n_out, pad_index=0):
        super(CHAR_CNN, self).__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index

        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_chars,
                                  embedding_dim=n_embed)

        #conv layer 1
        self.conv1 = nn.Conv1d(n_embed, n_out, 3, padding=1)

        #conv layer 2
        self.conv2 = nn.Conv1d(n_out, n_out, 3, padding=1)
        
        #activation
        self.activation = nn.ReLU()

        #maxpooling
        self.pooling1 = nn.MaxPool1d(3,stride=1,padding=1)
        self.pooling2 = nn.MaxPool1d(20)
    
    def forward(self, x):
        mask = x.ne(self.pad_index)
        
        #batch_size*20*50
        embx = self.embed(x)
        embx[~mask] = 0
        #batch_size*50*20
        embx = embx.transpose(-1,-2)

        L1 = self.activation(self.conv1(embx))
        L1 = self.pooling1(L1)
        L2 = self.activation(self.conv2(L1))
        L = L2.transpose(-1,-2)
        L[~mask] = -float('inf')
        L = L.transpose(-1,-2)

        hidden = self.pooling2(L)
        return hidden.squeeze()


