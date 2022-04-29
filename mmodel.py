import torch
import torch.nn as nn

import numpy as np
import copy

from parser import DEPSODBERT, DEPSOD, DEPLIN, DEPWEK, DEPLINBERT

import threading

class mmodel():

    def __init__(self, N, devices, config, VOCAB_SIZE, CHAR_SIZE,DEPREL_SIZE, xembed=None, bert=None):

        self.N = N
        self.devices = devices
        self._lock = threading.Lock()
        #initialize networks
        self.models = []
        self.results = {}
        for i in range(N):

            #model = DEPWEK(devices[i], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, copy.deepcopy(xembed)).to(devices[i])
            #model = DEPLIN(devices[i+1], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, copy.deepcopy(xembed)).to(devices[i+1])
            #model = DEPLINBERT(devices[i+1], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, copy.deepcopy(bert), copy.deepcopy(xembed)).to(devices[i+1])
            model = DEPSOD(devices[i+1], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, copy.deepcopy(xembed)).to(devices[i+1])
            #model = DEPSODBERT(devices[i+1], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, copy.deepcopy(bert), copy.deepcopy(xembed)).to(devices[i+1])
            
            self.models.append(model)
    
    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()
    
    def worker(self, i, sent, char, lengths):
        ret = self.models[i](sent, char, lengths)
        self.results[i] = ret

    def forward(self, sent, char, lengths, bsent=None):
        #threads = [threading.Thread(target=self.worker, args=(i, sent, char, lengths)) for i in range(self.N)]
        results = [model(sent, char, lengths) for model in self.models]

        return results


