import torch
import torch.nn as nn

import numpy as np

def load(path,unk=None):

    with open(path,'r') as f:
        lines = [line for line in f]
    splits = [line.split() for line in lines]
    tokens, vectors = zip(*[(s[0],list(map(float,s[1:]))) for s in splits])
    glove = {token: vector for (token,vector) in zip(tokens,vectors)}
    return glove

def weights(glove,w2i):

    weights_matrix = np.zeros((len(w2i)+1, 100))
    tokens = glove.keys()
    for token in tokens:
        #if i2w[i+1] in glove:
        weights_matrix[w2i[token]] = glove[token]
    return weights_matrix

def create_emb_layer(glove ,i2w, unk=None, non_trainable=True):

    #glove = load(pth,unk)
    weights_matrix = weights(glove,i2w)
    np.save('xembed.npy',weights_matrix)
    weightss = torch.tensor(weights_matrix,dtype=torch.float)
    #normalization by std
    weightss /= torch.std(weightss)
    emb_layer = nn.Embedding.from_pretrained(weightss)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer

def create_emb_layer_(weights_matrix,non_trainable=True):

    #emb_layer = nn.Embedding(weights_matrix.shape[0], weights_matrix.shape[1])
    #emb_layer.load_state_dict({'weight': weights_matrix})
    weights = torch.tensor(weights_matrix,dtype=torch.float)
    weights /= torch.std(weights)
    emb_layer = nn.Embedding.from_pretrained(weights)
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer


