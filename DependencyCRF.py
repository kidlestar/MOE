import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributions.utils import lazy_property

from torch.autograd import grad

from alg import inside2, inside2g, eisner, inside, insideg


class DependencyCRF1:

    #initialization
    def __init__(self, N, s_arc, mask_sent, mask_tree):
        self.N = N
        self.s_arc = s_arc
        self.mask_sent = mask_sent
        self.mask_tree = mask_tree
        tmask_sent = mask_sent.clone()
        tmask_sent[:,0] = False
        self.tmask_sent = tmask_sent

    @lazy_property
    def partition(self):
        if self.s_arc.requires_grad: logz = inside(self.s_arc, self.tmask_sent)
        else: logz = insideg(self.s_arc, self.tmask_sent)
        return logz.squeeze()


    def log_prob_(self, arc):
        
        E = self.s_arc[:,1:,1:].transpose(-1,-2)
        idx = torch.arange(E.size(1))
        E[:,idx,idx] = self.s_arc[:,1:,0]
        s_arc = (E*arc).sum(dim=(-1,-2))
        
        return s_arc - self.partition


    def log_prob(self, arc):
        
        arc_seq = arc[self.tmask_sent]
        arc_mask = self.tmask_sent

        lens_arc = arc_mask.sum(1)

        s_arc = self.s_arc[arc_mask].gather(-1, arc_seq.unsqueeze(-1)).squeeze()

        s_arc = pad_sequence(torch.split(s_arc, lens_arc.tolist()),batch_first=True).sum(1)
        return s_arc - self.partition

    def prob(self):
        s_arc = self.s_arc.detach().requires_grad_()
        logz  = inside(scores=s_arc, mask=self.tmask_sent).sum()
        probs, = grad(logz, s_arc, retain_graph=False)
        return probs

    @lazy_property
    def argmax(self):
        s_arc = self.s_arc.detach()
        with torch.no_grad():
            s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
            preds = eisner(s_arc, self.tmask_sent)
            cptree = preds.new_zeros(preds.size(0),preds.size(1),preds.size(1))
            cptree.scatter_(1, preds.unsqueeze(-2), 1)
            ptree = cptree[:,1:,1:]
            idx = torch.arange(ptree.size(1))
            ptree[:,idx,idx] = cptree[:,0,1:]
            ptree[~self.mask_tree] = 0
            return ptree

class DependencyCRF:

    #initialization
    def __init__(self, N, s_arc, s_sib, mask_sent, mask_tree):
        self.N = N
        self.s_arc = s_arc
        self.s_sib = s_sib
        self.mask_sent = mask_sent
        self.mask_tree = mask_tree
        tmask_sent = mask_sent.clone()
        tmask_sent[:,0] = False
        self.tmask_sent = tmask_sent
    
    @lazy_property
    def partition(self):
        if self.s_arc.requires_grad: logz = inside2(scores=(self.s_arc, self.s_sib), mask=self.tmask_sent)
        else: logz = inside2g(scores=(self.s_arc, self.s_sib), mask=self.tmask_sent)
        return logz.squeeze()

 
    def log_prob(self, arc, sib):
        tsib_seq = sib[:,1:]
        tsib_mask = tsib_seq.gt(0)
        arc_seq, sib_seq = arc[self.tmask_sent], sib[self.tmask_sent]
        arc_mask, sib_mask = self.tmask_sent, sib_seq.gt(0)
        sib_seq = sib_seq[sib_mask]

        lens_arc = arc_mask.sum(1)
        lens_sib = tsib_mask.sum(1)

        s_sib = self.s_sib[self.tmask_sent][torch.arange(len(arc_seq)), arc_seq]
        s_arc = self.s_arc[arc_mask].gather(-1, arc_seq.unsqueeze(-1)).squeeze()
        s_sib = s_sib[sib_mask].gather(-1, sib_seq.unsqueeze(-1)).squeeze()

        s_arc = pad_sequence(torch.split(s_arc, lens_arc.tolist()),batch_first=True).sum(1)
        s_sib = pad_sequence(torch.split(s_sib, lens_sib.tolist()),batch_first=True).sum(1)

        return s_sib + s_arc - self.partition

    def prob(self):
        s_arc = self.s_arc.detach().requires_grad_()
        s_sib = self.s_sib.detach().requires_grad_()
        logz  = inside2(scores=(s_arc, s_sib), mask=self.tmask_sent).sum()
        probs, = grad(logz, s_arc, retain_graph=False)
        return probs
    
    @lazy_property
    def argmax(self):
        s_arc = self.s_arc.detach()
        with torch.no_grad():
            s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
            preds = eisner(s_arc, self.tmask_sent)
            cptree = preds.new_zeros(preds.size(0),preds.size(1),preds.size(1))
            cptree.scatter_(1, preds.unsqueeze(-2), 1)
            ptree = cptree[:,1:,1:]
            idx = torch.arange(ptree.size(1))
            ptree[:,idx,idx] = cptree[:,0,1:]
            ptree[~self.mask_tree] = 0
            return ptree



