import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import numpy as np

import threading

from DependencyCRF import DependencyCRF, DependencyCRF1

class model:

    def __init__(self, N, net, cnet, optimizer, scheduler, coptimizer=None, cscheduler=None):

        self.N = N
        self.net = net
        self.cnet = cnet
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.coptimizer = coptimizer
        self.cscheduler = cscheduler
        self.devices = net.devices
        self.T = 1.0



    def cal_log_prob(self, results, arc, sib, tree, deprel, lengths, mask_sent, mask_deprel):
        list_logp_arc = []
        list_logp_rel = []
        threads = []
        for i in range(self.N):
            #(s_arc, logp_rel) = results[i]
            (s_arc, s_sib, logp_rel) = results[i]
            #if i==0: device = self.devices[0]
            #else: device = self.devices[1]
            device = self.devices[i+1]
            
            list_logp_arc.append(DependencyCRF(self.N, s_arc, s_sib, mask_sent.to(device), None).log_prob(arc.to(device), sib.to(device)).to(self.devices[0]).unsqueeze(0))
            
            #list_logp_arc.append(DependencyCRF1(self.N, s_arc, mask_sent.to(device), None).log_prob(arc.to(device)).to(self.devices[0]).unsqueeze(0))
            logRel = logp_rel.transpose(1,2)[tree.transpose(-1,-2)==1]
            logRel = logRel[torch.arange(len(logRel)), deprel[mask_deprel]]
            pad_logRel = pad_sequence(torch.split(logRel, lengths.tolist()),batch_first=True)
            list_logp_rel.append(pad_logRel.sum(-1).to(self.devices[0]).unsqueeze(0))
        logp_arc = torch.cat(list_logp_arc)
        logp_rel = torch.cat(list_logp_rel)
        return logp_arc, logp_rel

    def cal_coe(self, logp):
        #with deterministic annealing
        logp = logp * self.T
        mlogp = torch.logsumexp(logp, 0)
        log_coe = logp - mlogp
        #log_coe = log_coe - torch.logsumexp(log_coe,0)
        coe = torch.exp(log_coe)
        return coe

    def ecal_log_prob(self, results, arc_logc=None, rel_logc=None, mask_sent=None):
        list_prob = []
        list_logp_rel = []
        for i in range(self.N):
            #(s_arc, logp_rel) = results[i]
            (s_arc, s_sib, logp_rel) = results[i]
            #if i==0: device = self.devices[0]
            #else: device = self.devices[1]
            device = self.devices[i+1]
            #list_prob.append(DependencyCRF1(self.N, s_arc, mask_sent.to(device), None).prob().to(self.devices[0]).unsqueeze(0))
            list_prob.append(DependencyCRF(self.N, s_arc, s_sib, mask_sent.to(device), None).prob().to(self.devices[0]).unsqueeze(0))
            
            list_logp_rel.append(logp_rel.to(self.devices[0]).unsqueeze(0))
        prob = torch.cat(list_prob)
        logprob = torch.log(prob + 1e-45)
        #average log prob
        log_aprob = torch.log(prob.mean(0)+1e-45)
        #with coefficient
        vprob = (torch.exp(arc_logc.unsqueeze(-1).unsqueeze(-1))*prob).sum(0)
        log_vprob = torch.log(vprob+1e-45)

        logp_rel = torch.cat(list_logp_rel)
        #average log prob
        alogp_rel = logp_rel + np.log(1.0/self.N)
        alogp_rel = torch.logsumexp(alogp_rel, 0)
        #with coefficient
        vlogp_rel = logp_rel + rel_logc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        vlogp_rel = torch.logsumexp(vlogp_rel,0)

        log_aprob.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))
        log_vprob.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))

        #return logprob, logp_rel, log_aprob, alogp_rel
        return logprob, logp_rel, log_aprob, log_vprob, alogp_rel, vlogp_rel


    def cal_tree_rel(self, logp_arc, logp_rel, idx, mask_sent, mask_tree, isp = False):
        #make joint prediction
        maxlogp_rel = logp_rel.max(-1)[0]
        mlogp_rel = maxlogp_rel.new_zeros(maxlogp_rel.size(0), maxlogp_rel.size(1)+1, maxlogp_rel.size(2)+1)
        mlogp_rel[:,1:,1:] = maxlogp_rel.transpose(-1,-2)
        mlogp_rel[:,1:,0] = maxlogp_rel[:,idx,idx]
        mlogp_arc = logp_arc + mlogp_rel

        x = DependencyCRF(self.N, mlogp_arc, None, mask_sent.to(self.devices[0]), mask_tree.to(self.devices[0])).argmax
        if isp:
            logpr = (maxlogp_rel * x).sum(dim=(-1,-2))
            logpa = DependencyCRF1(self.N, mlogp_arc, mask_sent.to(self.devices[0]), mask_tree.to(self.devices[0])).log_prob_(x).detach()
            logp = logpr + logpa
        rel = logp_rel.argmax(-1)
        if isp: return x, rel, logp
        else: return x, rel


    def train(self, sent, bsent, char, deprel, tree, sib, arc, lengths, mask_tree):
        mask_sent = sent!=0
        mask_deprel = deprel!=0
        with torch.no_grad():
            self.net.eval()
            results = self.net.forward(sent, char, lengths, bsent)
        
        self.cnet.train()
        (arc_logc, rel_logc) = self.cnet(sent, char, lengths)
        self.net.train()
        tresults = self.net.forward(sent, char, lengths, bsent)

        #calculation of logp_arc and logp_rel without dropout
        with torch.no_grad(): logp_arc, logp_rel = self.cal_log_prob(results, arc, sib, tree, deprel, lengths, mask_sent, mask_deprel)

        tlogp_arc, tlogp_rel = self.cal_log_prob(tresults, arc, sib, tree, deprel, lengths, mask_sent, mask_deprel)
        
        #calculation of coefficient of combination
        
        with torch.no_grad():
            acoe = self.cal_coe(logp_arc + arc_logc)

            lacoe = self.cal_coe(logp_arc) #+ arc_logc)

            rcoe = self.cal_coe(logp_rel + rel_logc)

            lrcoe = self.cal_coe(logp_rel) #+ rel_logc)
        
        lengths = lengths.to(self.devices[0])
        loss_arc = - (((acoe*tlogp_arc).sum(1)) / ((acoe*lengths).sum(1))).mean()
        loss_acoe = -(lacoe*arc_logc).sum(0).mean()/self.N

        
        loss_rel = - (((rcoe*tlogp_rel).sum(1)) / ((rcoe*lengths).sum(1))).mean()
        loss_rcoe = -(lrcoe*rel_logc).sum(0).mean()/self.N
        
        meanLoss = loss_acoe + loss_rcoe + loss_arc + loss_rel

        meanLoss.backward()

        parameters = []
        #parameters += list(self.cnet.parameters())
        for i in range(self.N):
            parameters += list(self.net.models[i].parameters())
        total_norm = nn.utils.clip_grad_norm_(parameters, 1)
        print('norm used of arcs: ' + str(total_norm.detach()))
        if torch.isnan(total_norm):
            print(loss_anlr)
            print(loss_rnlr)
            print(loss_anlr_)
            print(loss_rnlr_)
            exit()
        #self.optimizer.step()
        #self.scheduler.step()
            
        self.coptimizer.step()
        self.cscheduler.step()
        #self.T += 2.0/50/200
        print('value of temperature: ' + str(self.T))
        #if self.T > 10: self.T = 1 
        
        print('arc loss: ' + str(loss_arc.item()))
        print('rel loss: ' + str(loss_rel.item()))
        #print('reg arc: ' + str(loss_anlr_.item()))
        #print('reg rel: ' + str(loss_rnlr_.item()))
        print('arc_coe loss: ' + str(loss_acoe.item()))
        print('rel_coe loss: ' + str(loss_rcoe.item()))
        
        print('temperature: ' + str(self.T))
        print('calculated coefficient of combination of first sentence without dropout')
        print(lacoe[:,0].detach().cpu().numpy())
        print('probability of every model of first sentence without dropout')
        print(torch.exp(logp_arc[:,0].detach()).cpu().numpy())
        print('real coefficient of combination of first sentence with dropout')
        print(torch.exp(arc_logc.detach())[:,0].cpu().numpy())
        torch.cuda.empty_cache()
        return meanLoss.item(), loss_arc.item(), loss_rel.item(),loss_acoe.item(), loss_rcoe.item()


    def evaluate(self, sent, bsent, char, lengths, mask_tree):
        mask_sent = sent!=0
        idx = torch.arange(lengths.max())
        self.net.eval()
        self.cnet.eval()
        with torch.no_grad():
            results = self.net.forward(sent, char, lengths, bsent)
            (arc_logc, rel_logc) = self.cnet(sent, char, lengths)
        
        logprob, logp_rel, alogp_arc, vlogp_arc, alogp_rel, vlogp_rel = self.ecal_log_prob(results, arc_logc, rel_logc, mask_sent)
        #logprob, logp_rel, alogp_arc, alogp_rel = self.ecal_log_prob(results, None, None, mask_sent)
        x, rel = self.cal_tree_rel(vlogp_arc, vlogp_rel, idx, mask_sent, mask_tree)
        ax, arel = self.cal_tree_rel(alogp_arc, alogp_rel, idx, mask_sent, mask_tree)
        ox = []
        orel = []
        ologp = []
        for i in range(self.N):
            tx, trel ,logp = self.cal_tree_rel(logprob[i], logp_rel[i], idx, mask_sent, mask_tree, True)
            ox.append(tx)
            orel.append(trel)
            ologp.append(logp.unsqueeze(0))
        return ox, orel, x, ax, rel, arel
