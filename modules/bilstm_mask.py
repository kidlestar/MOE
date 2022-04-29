# -*- coding: utf-8 -*-

from modules.dropout import SharedDropout

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils import spectral_norm

class BiLSTMM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, sflag=True, isd = True):
        super(BiLSTMM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.isd = isd

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for layer in range(self.num_layers):
            if sflag:
                self.f_cells.append(spectral_norm(spectral_norm(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size),'weight_hh'),'weight_ih'))
                self.b_cells.append(spectral_norm(spectral_norm(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size),'weight_hh'),'weight_ih'))
            else:
                self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
                self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))

            input_size = hidden_size * 2

        self.reset_parameters()

    def reset_parameters(self):
        for i in self.parameters():
            # apply orthogonal_ to weight
            if len(i.shape) > 1:
                nn.init.orthogonal_(i)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(i)

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False, mask=None):
        h, c = hx
        init_h, init_c = h, c
        output, seq_len = [], len(x)
        steps = reversed(range(seq_len)) if reverse else range(seq_len)
        hid_mask = None
        if self.training and self.isd:
            if mask is None: hid_mask = SharedDropout.get_mask(h, self.dropout)
            else: hid_mask = mask.to(h.device)

        for t in steps:
            last_batch_size, batch_size = len(h), batch_sizes[t]
            if last_batch_size < batch_size:
                h = torch.cat((h, init_h[last_batch_size:batch_size]))
                c = torch.cat((c, init_c[last_batch_size:batch_size]))
            else:
                h = h[:batch_size]
                c = c[:batch_size]
            h, c = cell(input=x[t], hx=(h, c))
            output.append(h)
            if self.training and self.isd:
                h = h * hid_mask[:batch_size]
        if reverse:
            output.reverse()
        output = torch.cat(output)

        return output, hid_mask

    def forward(self, x, hx=None, masks=None):
        x, batch_sizes, sorted_indice, unsorted_indice = x
        batch_size = batch_sizes[0]

        if hx is None:
            init = x.new_zeros(batch_size, self.hidden_size)
            hx = (init, init)
        if masks is None:
            rmasks = {}
            rmasks['outer'] = []
            rmasks['forw'] = []
            rmasks['back'] = []
        for layer in range(self.num_layers):
            if self.training and self.isd:
                if masks is None:
                    mask = SharedDropout.get_mask(x[:batch_size], self.dropout)
                    mask = torch.cat([mask[:batch_size]
                                  for batch_size in batch_sizes])
                    rmasks['outer'].append(mask)
                else:
                    mask = masks['outer'][layer].to(x.device)
                x *= mask
            x = torch.split(x, batch_sizes.tolist())
            if masks is None:
                f_output, fmask = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False)
                b_output, bmask = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True)
                rmasks['forw'].append(fmask)
                rmasks['back'].append(bmask)
            else:
                f_output, fmask = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.f_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=False,mask=masks['forw'][layer])
                b_output, bmask = self.layer_forward(x=x,
                                          hx=hx,
                                          cell=self.b_cells[layer],
                                          batch_sizes=batch_sizes,
                                          reverse=True,mask=masks['back'][layer])
            x = torch.cat([f_output, b_output], -1)
        x = PackedSequence(x, batch_sizes, None, None)
        if masks is None: masks = rmasks
        return x, masks

