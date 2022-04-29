import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from modules import (MLP, Biaffine, Triaffine, BiLSTM, CHAR_LSTM, SharedDropout, IndependentDropout)


#weak learner
class DEPWEK(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=None):

        super(DEPWEK,self).__init__()
        #initialization of parameters
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        self.INPUT_SIZE = config.wemb_size + config.n_feat_embed
        self.pretrained = xembed

        #create network
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.mlp_arc_d = MLP(n_in=self.INPUT_SIZE,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=self.INPUT_SIZE,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=self.INPUT_SIZE,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=self.INPUT_SIZE,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)

        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)



    def forward(self, sent, char, lengths):
        sent = sent.to(self.device)
        char = char.to(self.device)
        lengths = lengths.to(self.device)
        mask_sent = sent!=0
        ext_mask = sent>=self.VOCAB_SIZE
        ext_sent = sent.masked_fill(ext_mask, 1)
        embSent = self.word_embed(ext_sent)
        if self.pretrained is not None:
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(char[mask_sent])
        embChar = pad_sequence(feat_embed.split((lengths+1).tolist()), True)
        embSent, embChar = self.embed_dropout(embSent,embChar)

        #BiLSTM
        lemb = torch.cat((embSent,embChar),2)

        #arc score calculation
        arc_h = self.mlp_arc_h(lemb)
        arc_d = self.mlp_arc_d(lemb)
        s_arc = self.arc_attn(arc_d,arc_h)
        s_arc.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))

        #rel score calculation
        rel_h = self.mlp_rel_h(lemb)
        rel_d = self.mlp_rel_d(lemb)
        s_rel = self.rel_attn(rel_d,rel_h).permute(0,3,2,1)
        e_rel = s_rel[:,1:,1:,:]
        idx = torch.arange(0,lengths.max(),dtype=torch.long,device=self.device)
        e_rel[:,idx,idx,:] = s_rel[:,0,1:,:]
        logp_rel = e_rel - torch.logsumexp(e_rel,-1).unsqueeze(-1)

        return (s_arc, logp_rel)


#first order model
class DEPLIN(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=None):

        super(DEPLIN,self).__init__()
        #initialization of parameters
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        self.INPUT_SIZE = config.wemb_size + config.n_feat_embed
        self.pretrained = xembed

        #create network
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=config.lstm_layer,dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        self.mlp_arc_d = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)

        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)



    def forward(self, sent, char, lengths):
        sent = sent.to(self.device)
        char = char.to(self.device)
        lengths = lengths.to(self.device)
        mask_sent = sent!=0
        ext_mask = sent>=self.VOCAB_SIZE
        ext_sent = sent.masked_fill(ext_mask, 1)
        embSent = self.word_embed(ext_sent)
        if self.pretrained is not None:
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(char[mask_sent])
        embChar = pad_sequence(feat_embed.split((lengths+1).tolist()), True)
        embSent, embChar = self.embed_dropout(embSent,embChar)

        #BiLSTM
        emb = torch.cat((embSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        features, _ = self.lstm(pack_emb)
        pad_emb,_ = pad_packed_sequence(features,batch_first=True,total_length=lengths.max()+1)
        lemb = self.lstm_dropout(pad_emb)

        #arc score calculation
        arc_h = self.mlp_arc_h(lemb)
        arc_d = self.mlp_arc_d(lemb)
        s_arc = self.arc_attn(arc_d,arc_h)
        s_arc.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))

        #rel score calculation
        rel_h = self.mlp_rel_h(lemb)
        rel_d = self.mlp_rel_d(lemb)
        s_rel = self.rel_attn(rel_d,rel_h).permute(0,3,2,1)
        e_rel = s_rel[:,1:,1:,:]
        idx = torch.arange(0,lengths.max(),dtype=torch.long,device=self.device)
        e_rel[:,idx,idx,:] = s_rel[:,0,1:,:]
        logp_rel = e_rel - torch.logsumexp(e_rel,-1).unsqueeze(-1)

        return (s_arc, logp_rel)

#second order model
class DEPSOD(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=None):
        
        super(DEPSOD,self).__init__()
        #initialization of parameters
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        self.INPUT_SIZE = config.wemb_size + config.n_feat_embed
        self.pretrained = xembed

        #create network
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=config.lstm_layer,dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        self.mlp_arc_d = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_s = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_d = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_h = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)

        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=config.sib_mlp_size, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)



    def forward(self, sent, char, lengths):
        sent = sent.to(self.device)
        char = char.to(self.device)
        lengths = lengths.to(self.device)
        mask_sent = sent!=0
        ext_mask = sent>=self.VOCAB_SIZE
        ext_sent = sent.masked_fill(ext_mask, 1)
        embSent = self.word_embed(ext_sent)
        if self.pretrained is not None: 
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(char[mask_sent])
        embChar = pad_sequence(feat_embed.split((lengths+1).tolist()), True) 
        embSent, embChar = self.embed_dropout(embSent,embChar)

        #BiLSTM
        emb = torch.cat((embSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        features, _ = self.lstm(pack_emb)
        pad_emb,_ = pad_packed_sequence(features,batch_first=True,total_length=lengths.max()+1)
        lemb = self.lstm_dropout(pad_emb)

        #arc score calculation
        arc_h = self.mlp_arc_h(lemb)
        arc_d = self.mlp_arc_d(lemb)
        s_arc = self.arc_attn(arc_d,arc_h)
        s_arc.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))

        #sib score calculation
        sib_s = self.mlp_sib_s(lemb)
        sib_d = self.mlp_sib_d(lemb)
        sib_h = self.mlp_sib_h(lemb)
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)

        #rel score calculation
        rel_h = self.mlp_rel_h(lemb)
        rel_d = self.mlp_rel_d(lemb)
        s_rel = self.rel_attn(rel_d,rel_h).permute(0,3,2,1)
        e_rel = s_rel[:,1:,1:,:]
        idx = torch.arange(0,lengths.max(),dtype=torch.long,device=self.device)
        e_rel[:,idx,idx,:] = s_rel[:,0,1:,:]
        logp_rel = e_rel - torch.logsumexp(e_rel,-1).unsqueeze(-1)

        return (s_arc, s_sib, logp_rel)

class DEPSODBERT(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, bert=None, xembed=None):
        
        super(DEPSODBERT,self).__init__()
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        if bert is None: self.INPUT_SIZE = config.wemb_size + config.n_feat_embed
        else: self.INPUT_SIZE = config.wemb_size + config.n_feat_embed + 768
        self.pretrained = xembed
        self.bert = bert

        ##common parts
        
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        
        
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        
        #BILSTM
        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=config.lstm_layer,dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        self.mlp_arc_d = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_s = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_d = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_sib_h = MLP(n_in=config.hidden_size*2,n_out=config.sib_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.sib_attn = Triaffine(n_in=config.sib_mlp_size, bias_x=True, bias_y=True)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)


    def bertEmb(self):
        embBSent = self.bert(self.bsent)
        embBSent = embBSent.hidden_states[-4:]
        embBSent = [temp.unsqueeze(0) for temp in embBSent]
        embBSent = torch.cat(embBSent, 0)
        embBSent = embBSent.mean(0)
        return embBSent

    #The first level of features by a BILSTM
    def BILSTMFeatures(self,embSent,embBSent, embChar,lengths):
        
        if embBSent is not None: emb = torch.cat((embSent,embBSent,embChar),2)
        else: emb = torch.cat((embSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        Lfeatures, _ = self.lstm(pack_emb)
        pad_emb,_ = pad_packed_sequence(Lfeatures,batch_first=True,total_length=lengths.max()+1)
        pad_emb = self.lstm_dropout(pad_emb)
        return pad_emb

    #Linear model features
    def LinearFeatures(self, features, max_length):
        Heads = features
        Modifiers = features
        return Heads, Modifiers

    #energy matrix
    def LinearEnergy(self, head, modifier, batch_size, max_length):
        L1H = self.mlp_arc_h(head)
        L1M = self.mlp_arc_d(modifier)
        E = self.arc_attn(L1M,L1H)
        res = E[:,1:,1:].transpose(-1,-2)
        idx = torch.arange(0,max_length,dtype=torch.long,device=self.device)
        res[:,idx,idx]=E[:,1:,0]
        res=res.reshape(batch_size,max_length,max_length)
        return (E, res)

    #siblings
    def SibEnergy(self, feature, batch_size, max_length):
        sib_s = self.mlp_sib_s(feature)
        sib_d = self.mlp_sib_d(feature)
        sib_h = self.mlp_sib_h(feature)
        s_sib = self.sib_attn(sib_s, sib_d, sib_h).permute(0, 3, 1, 2)
        return s_sib

    #rel matrix
    def LinearRel(self, head, modifier, batch_size, max_length):
        L1H = self.mlp_rel_h(head)
        L1M = self.mlp_rel_d(modifier)
        E = self.rel_attn(L1M,L1H).permute(0,3,2,1)
        res = E[:,1:,1:,:]
        idx = torch.arange(0,max_length,dtype=torch.long,device=self.device)
        res[:,idx,idx,:]=E[:,0,1:,:]
        return (res - torch.logsumexp(res,-1).unsqueeze(-1), res)

    def LinearEnergyMatrix(self, features, batch_size, max_length=None):
        Heads, Modifiers = self.LinearFeatures(features, max_length)
        return self.LinearEnergy(Heads, Modifiers, batch_size, max_length), self.LinearRel(Heads, Modifiers, batch_size, max_length), self.SibEnergy(features, batch_size, max_length)
    
    #for every new sentence, do once initialization
    def forward(self, sent, bsent, char, lengths):
        self.sent = sent.to(self.device)
        self.bsent = bsent.to(self.device)
        mask_sent = self.sent!=0
        ext_mask = self.sent>=self.VOCAB_SIZE
        self.ext_sent = self.sent.masked_fill(ext_mask, 1)
        self.batch_size = self.sent.size(0)
        self.lengths = lengths.to(self.device)
        self.char = char.to(self.device)
        self.max_length = self.lengths.max()
        embSent = self.word_embed(self.ext_sent)
        if self.pretrained is not None: 
            xembSent = self.pretrained(self.sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(self.char[mask_sent])
        embChar = pad_sequence(feat_embed.split((self.lengths+1).tolist()), True) 
        embSent, embChar = self.embed_dropout(embSent,embChar)
        if self.bert is not None: embBSent = self.bertEmb()
        else: embBSent = None
        self.LSTMfeatures = self.BILSTMFeatures(embSent,embBSent,embChar,self.lengths)
        
        self.features = self.LSTMfeatures
        (self.E, self.linearMatrix), (self.logRel, self.Erel), self.E_sib = self.LinearEnergyMatrix(self.features, self.batch_size, self.max_length)
        self.E.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))
        return (self.E, self.E_sib, self.logRel)

#first order model
class DEPLINBERT(nn.Module):

    def __init__(self, device, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, bert=None, xembed=None):

        super(DEPLINBERT,self).__init__()
        #initialization of parameters
        self.device = device
        self.config = config
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CHAR_SIZE = CHAR_SIZE
        self.DEPREL_SIZE = DEPREL_SIZE
        self.bert = bert
        self.INPUT_SIZE = config.wemb_size + config.n_feat_embed + 768
        self.pretrained = xembed

        #create network
        self.word_embed = nn.Embedding(VOCAB_SIZE,config.wemb_size)
        nn.init.zeros_(self.word_embed.weight)
        self.feat_embed = CHAR_LSTM(n_chars=CHAR_SIZE, n_embed=config.n_char_embed, n_out=config.n_feat_embed)
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        self.lstm = BiLSTM(input_size=self.INPUT_SIZE,hidden_size=config.hidden_size,num_layers=config.lstm_layer,dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        self.mlp_arc_d = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_arc_h = MLP(n_in=config.hidden_size*2,n_out=config.arc_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=config.hidden_size*2,n_out=config.rel_mlp_size,dropout=config.mlp_dropout)

        self.arc_attn = Biaffine(n_in=config.arc_mlp_size, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=config.rel_mlp_size, n_out=self.DEPREL_SIZE+1, bias_x=True, bias_y=True)



    def forward(self, sent, bsent, char, lengths):
        sent = sent.to(self.device)
        bsent = bsent.to(self.device)
        char = char.to(self.device)
        lengths = lengths.to(self.device)
        mask_sent = sent!=0
        ext_mask = sent>=self.VOCAB_SIZE
        ext_sent = sent.masked_fill(ext_mask, 1)
        embSent = self.word_embed(ext_sent)
        if self.pretrained is not None:
            xembSent = self.pretrained(sent)
            embSent = embSent + xembSent
        feat_embed = self.feat_embed(char[mask_sent])
        embChar = pad_sequence(feat_embed.split((lengths+1).tolist()), True)
        embSent, embChar = self.embed_dropout(embSent,embChar)

        #bert embedding
        embBSent = self.bert(bsent)
        embBSent = embBSent.hidden_states[-4:]
        embBSent = [temp.unsqueeze(0) for temp in embBSent]
        embBSent = torch.cat(embBSent, 0)
        embBSent = embBSent.mean(0)


        #BiLSTM
        emb = torch.cat((embSent,embBSent,embChar),2)
        pack_emb = pack_padded_sequence(emb,lengths.cpu()+1,batch_first=True,enforce_sorted=False)
        features, _ = self.lstm(pack_emb)
        pad_emb,_ = pad_packed_sequence(features,batch_first=True,total_length=lengths.max()+1)
        lemb = self.lstm_dropout(pad_emb)

        #arc score calculation
        arc_h = self.mlp_arc_h(lemb)
        arc_d = self.mlp_arc_d(lemb)
        s_arc = self.arc_attn(arc_d,arc_h)
        s_arc.masked_fill_(~mask_sent.unsqueeze(1), float('-inf'))

        #rel score calculation
        rel_h = self.mlp_rel_h(lemb)
        rel_d = self.mlp_rel_d(lemb)
        s_rel = self.rel_attn(rel_d,rel_h).permute(0,3,2,1)
        e_rel = s_rel[:,1:,1:,:]
        idx = torch.arange(0,lengths.max(),dtype=torch.long,device=self.device)
        e_rel[:,idx,idx,:] = s_rel[:,0,1:,:]
        logp_rel = e_rel - torch.logsumexp(e_rel,-1).unsqueeze(-1)

        return (s_arc, logp_rel)


