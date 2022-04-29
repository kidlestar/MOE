# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first
        self.mask = None

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask
    

class SharedDropout_inf(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout_inf, self).__init__()

        self.p = p
        self.batch_first = batch_first
        self.mask = None

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.batch_first:
                #mask = self.get_mask(x[:, 0], self.p)
                mask = self.get_mask(x, self.p)
            else:
                #mask = self.get_mask(x[0], self.p)
                mask = self.get_mask(x, self.p)
            #idx of all inf place
            idx_all = (mask.sum(-1))==0
            nmask = mask.new_zeros(mask.size())
            nmask[idx_all] = 1
            idxx = nmask==1
            idx = mask==0
            #x = x.transpose(-1,-2)
            x[idx] = -1e32
            x[idxx] = 0
            #x = x.transpose(-1,-2)
            #x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask


class MSharedDropout_(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(MSharedDropout_, self).__init__()

        self.p = p
        self.batch_first = batch_first
        self.mask = None

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x, mask=None):
        if self.training:
            if mask is None:
                if self.batch_first:
                    mask = self.get_mask(x[:, 0], self.p)
                else:
                    mask = self.get_mask(x[0], self.p)
            x *= mask.to(x.device).unsqueeze(1) if self.batch_first else mask
        return x, mask

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask




class SharedDropout_convex(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout_convex, self).__init__()

        self.p = p
        self.batch_first = batch_first
        self.mask = None

    def extra_repr(self):
        info = f"p={self.p}"
        if self.batch_first:
            info += f", batch_first={self.batch_first}"

        return info

    def forward(self, x):
        if self.training:
            if self.mask is not None: mask = self.mask
            else:
                if self.batch_first:
                    mask = self.get_mask(x[:, 0], self.p)
                else:
                    mask = self.get_mask(x[0], self.p)
                self.mask_fix = mask
                self.mask = mask
            #print(x.size())
            #print(mask.size())
            x = x * (mask.unsqueeze(1) if self.batch_first else mask)

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_full(x.shape, 1 - p)
        mask = torch.bernoulli(mask) / (1 - p)

        return mask

    def set(self,index):
        if index is None: 
            self.mask = None
            self.mask_fix = None
        else: 
            self.mask = self.mask_fix[index]


"""
class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, x, y, eps=1e-12):
        if self.training:
            x_mask = torch.bernoulli(x.new_full(x.shape[:2], 1 - self.p))
            y_mask = torch.bernoulli(y.new_full(y.shape[:2], 1 - self.p))
            scale = 3.0 / (2.0 * x_mask + y_mask + eps)
            x_mask *= scale
            y_mask *= scale
            x *= x_mask.unsqueeze(dim=-1)
            y *= y_mask.unsqueeze(dim=-1)

        return x, y
"""

class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items


class MIndependentDropout_(nn.Module):

    def __init__(self, p=0.5):
        super(MIndependentDropout_, self).__init__()

        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items, masks = None):
        if self.training:
            if masks is None:
                masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
                total = sum(masks)
                scale = len(items) / total.max(torch.ones_like(total))
                masks = [mask * scale for mask in masks]
            items = [item * mask.to(item.device).unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items, masks


