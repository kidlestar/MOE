# -*- coding: utf-8 -*-

from modules.dropout import SharedDropout
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import PackedSequence


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def prange(self, n: int):
        ret: List[int] = []
        for i in range(n):
            ret.append(i)
        return ret

    def rprange(self, n: int):
        ret: List[int] = []
        for i in range(n):
            ret.append(n-i-1)
        return ret

    def layer_forward(self, x, hx: List[torch.Tensor], index: int, batch_sizes: List[int], reverse=False):
        hx_0 = hx_i = hx
        hx_n: List[List[torch.Tensor]] = [] 
        ohx_n : List[torch.Tensor] = []
        output: List[torch.Tensor] = []
        steps = self.rprange(len(x)) if reverse else self.prange(len(x))
        hid_mask = torch.empty(hx_0[0].size())
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
                        for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            if not reverse: 
                for i, model in enumerate(self.f_cells):
                    if i == index: 
                        thx_i = (hx_i[0],hx_i[1])
                        hx_i = [h for h in model(x[t], thx_i)]
                #hx_i = [h for h in model(x[t], hx_i) for i, model in enumerate(self.f_cells) if i==index]
            else: 
                for i, model in enumerate(self.b_cells):
                    if i == index: 
                        thx_i = (hx_i[0],hx_i[1])
                        hx_i = [h for h in model(x[t], thx_i)]
                #hx_i = [h for h in model(x[t], hx_i) for i, model in enumerate(self.b_cells) if i==index]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            ohx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            ohx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, ohx_n

    def forward(self, x, batch_sizes: List[int], sorted_indices: Optional[torch.Tensor], unsorted_indices: Optional[torch.Tensor]):
        #x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
        h, c = ih, ih
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            tx = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                tx = [i * mask[:len(i)] for i in tx]
            x_f, (h_f, c_f) = self.layer_forward(x=tx,
                                                 hx=[h[i, 0], c[i, 0]],
                                                 index=i,
                                                 batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=tx,
                                                 hx=[h[i, 1], c[i, 1]],
                                                 index=i,
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x,
                           batch_sizes,
                           sorted_indices,
                           unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, unsorted_indices)

        return x, hx

