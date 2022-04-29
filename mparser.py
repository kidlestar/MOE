import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from modules import (MLP_static, BiLSTM, CHAR_LSTM, SharedDropout, IndependentDropout)


import time


class COEModel(nn.Module):

    def __init__(self, N, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=None):
        
        super(COEModel,self).__init__()
        self.N = N
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        self.pretrained = xembed
        self.INPUT_SIZE = config.wemb_size + config.n_feat_embed

        
        #Embedding
        self.word_emb = nn.Embedding(VOCAB_SIZE+1,config.wemb_size)
        nn.init.zeros_(self.word_emb.weight)
        
        #char embedding
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE+1, n_embed=config.n_char_embed, n_out=config.n_feat_embed)

        self.embed_dropout = IndependentDropout(p=config.embed_dropout)
        #BILSTM
        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=3,dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)
        self.ALSTM = nn.LSTM(input_size=config.hidden_size*2,hidden_size=config.hidden_size,num_layers=1,batch_first=True,bidirectional=True) 
        self.RLSTM = nn.LSTM(input_size=config.hidden_size*2,hidden_size=config.hidden_size,num_layers=1,batch_first=True,bidirectional=True)

        #self.m = nn.MaxPool1d(8, stride=8)
        self.m = nn.AdaptiveMaxPool1d(config.hidden_size)

        self.mlp_arc_c = MLP_static(n_in=config.hidden_size,n_hidden=N)
        
        self.mlp_rel_c = MLP_static(n_in=config.hidden_size,n_hidden=N)



    def forward(self, sent, char, lengths):
        sent = sent.to(self.device)
        char = char.to(self.device)
        lengths = lengths.to(self.device)
        mask_sent = sent!=0
        ext_mask = sent>=self.VOCAB_SIZE
        ext_sent = sent.masked_fill(ext_mask, 1)
        embSent = self.word_emb(ext_sent)
        if self.pretrained is not None:
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(char[mask_sent])
        embChar = pad_sequence(feat_embed.split((lengths+1).tolist()), True)
        embSent, embChar = self.embed_dropout(embSent,embChar)

        emb = torch.cat((embSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        features, _ = self.lstm(pack_emb)
        pad_emb,_ = pad_packed_sequence(features,batch_first=True,total_length=lengths.max()+1)
        lemb = self.lstm_dropout(pad_emb)
        arc_seq = pack_padded_sequence(lemb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        aoutput, (arc_n, c_n) = self.ALSTM(arc_seq)
        rel_seq = pack_padded_sequence(lemb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        routput, (rel_n, c_n) = self.RLSTM(rel_seq)
      
        if lengths.max()+1>100:
            aoutput,_ = pad_packed_sequence(aoutput, batch_first=True,total_length=lengths.max()+1)
            routput,_ = pad_packed_sequence(routput, batch_first=True,total_length=lengths.max()+1)
            aoutput = aoutput[:,:100,:]
            routput = routput[:,:100,:]
        else:
            aoutput,_ = pad_packed_sequence(aoutput, batch_first=True,total_length=100)
            routput,_ = pad_packed_sequence(routput, batch_first=True,total_length=100)


        aoutput = aoutput.view(aoutput.size(0),-1).unsqueeze(0)
        routput = routput.view(routput.size(0),-1).unsqueeze(0)

        #arc_n = self.m(arc_n)
        #rel_n = self.m(rel_n)
        arc_n = self.m(aoutput)
        rel_n = self.m(routput)

        #coefficient calculation
        arc_c = self.mlp_arc_c(arc_n.squeeze(0))
        rel_c = self.mlp_rel_c(rel_n.squeeze(0))

        arc_logc = arc_c - torch.logsumexp(arc_c,-1).unsqueeze(-1)
        arc_logc = arc_logc.transpose(0,1)

        rel_logc = rel_c - torch.logsumexp(rel_c,-1).unsqueeze(-1)
        rel_logc = rel_logc.transpose(0,1)

        return (arc_logc, rel_logc)
