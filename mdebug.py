import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

import argparse
import random
import numpy as np
import time
import pickle
import os
import copy

#from DataGen import collate_train, collate_eva, TrainingDataset, DevDataset
from ext_emb import load, create_emb_layer, create_emb_layer_
from preprocessing import Preprocessing
from mparser import COEModel
from mmodel import mmodel
from model import model
from config import Config
import utils

import adabound
#import swats


if __name__ == '__main__':
    
    #torch.autograd.set_detect_anomaly(True)
    #torch.set_default_dtype(torch.float64)
    #set parameters for training
    
    random.seed(time.time())
    torch.manual_seed(random.randint(0,999999))
    np.random.seed(random.randint(0,999999))

    
    userParser = argparse.ArgumentParser()
    userParser.add_argument('--num', '-n', type=int, default=3, help="number of experts")
    userParser.add_argument("--gpu", "-g", type=int, default=1, choices=[0,1,2,3,4], help="the number of gpu to use")
    userParser.add_argument("--train", "-t", default='/users/xudong.zhang/data/ud-treebanks-v2.0/UD_Chinese/zh-ud-train.conllu', help="path to training data")
    userParser.add_argument("--dev", "-d", default='/users/xudong.zhang/data/ud-treebanks-v2.0/UD_Chinese/zh-ud-dev.conllu', help="path to dev data")
    userParser.add_argument("--test", '-u', default='/users/xudong.zhang/data/ud-test-v2.0-conll2017/gold/conll17-ud-test-2017-05-09/zh.conllu', help='path to test data')
    userParser.add_argument('--epoch', '-e', type=int, default=30)
    userParser.add_argument('--config', '-c', default='config.ini', help='path to config file')
    userParser.add_argument('--preTrained', '-p', default='./GMMB10/', help='Pretrained Part')
    userParser.add_argument('--save', '-s', default='./modelGMMC4/', help='Folder to save the files')
    userParser.add_argument('--xembed', '-x', type=int, default=1,choices=[0,1],help='Using external embedding or not (0 no, 1 yes)')
    userParser.add_argument('--yembed', '-y', default='../data/cpretrained.txt', help='path to external embedding')
    args = userParser.parse_args()
    tokenizer = BertTokenizer.from_pretrained('../BertZh/')
    #preprocessing and initialization
    glove = None
    if args.xembed: glove = load(args.yembed)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    train_data = Preprocessing(args.train,tokenizer=tokenizer,glove=glove)
    #train_data = Preprocessing(args.train,glove=glove)
    emb_layer=None

    N = args.num
    devices = []
    for i in range(3):
        #if i == N: devices.append(torch.device('cuda:' + str(0)))
        #else: devices.append(torch.device('cuda:' + str(i)))
        #devices.append(torch.device('cuda:' + str(i)))
        if i==0: 
            devices.append(torch.device('cuda:' + str(i)))
            #devices.append(torch.device('cuda:' + str(i)))
        else:
            devices.append(torch.device('cuda:' + str(i)))
            #devices.append(torch.device('cuda:' + str(i)))
    if args.xembed==1:
        emb_layer = train_data.emb_layer
    

    #dev_data = Preprocessing(args.dev,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,deprel=train_data.deprel)
    #test_data = Preprocessing(args.test,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,deprel=train_data.deprel)
    dev_data = Preprocessing(args.dev,tokenizer=tokenizer,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)
    test_data = Preprocessing(args.test,tokenizer=tokenizer,isTrain=False,w2i=train_data.w2i,c2i=train_data.c2i,xpos=train_data.xpos,deprel=train_data.deprel)

    #bert = BertModel.from_pretrained('../BertEn/', output_hidden_states=True)

    #bert = bert.requires_grad_(False)
    bert = None
    VOCAB_SIZE = train_data.VOCAB_SIZE
    CHAR_SIZE = train_data.CHAR_SIZE
    DEPREL_SIZE = len(train_data.deprel)
    config = Config(args.config)
    num_batch = torch.tensor(train_data.chunks).sum().item()
    net = mmodel(N, devices, config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=emb_layer, bert = bert)
    cnet = COEModel(N, devices[0], config, VOCAB_SIZE, CHAR_SIZE, DEPREL_SIZE, xembed=emb_layer).to(devices[0])
    
    
    #load the local model which gives the highest UAS on dev
    checkpoint = torch.load(args.preTrained+'DISLinear0.pth', map_location=device)
    net.models[0].load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    checkpoint = torch.load(args.preTrained+'DISLinear0_.pth', map_location=device)
    net.models[1].load_state_dict(checkpoint['model_state_dict'], strict=True)
    del checkpoint
    #checkpoint = torch.load(args.preTrained+'Sep2_8.pth', map_location=device)
    #net.models[2].load_state_dict(checkpoint['model_state_dict'], strict=True)
    #del checkpoint
    #checkpoint = torch.load(args.preTrained+'Sep3_8.pth', map_location=device)
    #net.models[3].load_state_dict(checkpoint['model_state_dict'], strict=True)
    #del checkpoint
    #checkpoint = torch.load(args.preTrained+'Sep4_8.pth', map_location=device)
    #net.models[4].load_state_dict(checkpoint['model_state_dict'], strict=True)
    #del checkpoint
    #checkpoint = torch.load(args.preTrained+'Sep5_8.pth', map_location=device)
    #net.models[5].load_state_dict(checkpoint['model_state_dict'], strict=True)
    #del checkpoint
    
    #checkpoint = torch.load(args.preTrained+'DIS8.pth', map_location=device)
    #checkpoint = torch.load('./modelLinearCRF/DISLinear268.pth', map_location=device)
    #cnet.load_state_dict(checkpoint['model_state_dict'], strict=True)
    #del checkpoint
    #for i in range(N): lnet[i].load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    
    parameters = []
    for i in range(N):
        parameters += list(filter(lambda p: p.requires_grad,net.models[i].parameters()))
    
    optimizer = optim.Adam(parameters,config.lr/20.0,(config.beta_1, config.beta_2),config.epsilon)
    scheduler = ExponentialLR(optimizer, (config.decay) ** (1 / config.steps))
   
    #coptimizer=None
    #cscheduler=None
    coptimizer = optim.Adam(filter(lambda p: p.requires_grad,cnet.parameters()),config.lr,(config.beta_1, config.beta_2),config.epsilon)
    cscheduler = ExponentialLR(coptimizer, (config.decay) ** (1 / config.steps))
    
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #coptimizer.load_state_dict(checkpoint['coptimizer_state_dict'])
    #cscheduler.load_state_dict(checkpoint['schedulers_state_dict'])
    #del checkpoint
    
    torch.cuda.empty_cache()
    #cnet = None
    Model = model(N, net, cnet, optimizer, scheduler, coptimizer, cscheduler)
    print('Successfully initialize network')    
    
    sentNum = 0.0
    sumLoss = 0.0
    iter_time = 0.0
    old_uas = 0
    old_test_uas = 0
    old_las = 0.0
    t_patience = 0
    data_to_save ={}
    data_to_save['UAS'] = []
    data_to_save['UASD'] = []
    data_to_save['loss'] = []
    data_to_save['sumLoss'] = []
    index_best = 0
    base_iter=10
    
    print('initial state')
    ITER = 0 
    dev_ouas, dev_olas, dev_uas, dev_auas, dev_las, dev_alas = utils.evaluate(device,dev_data,Model,args.save+'dev',old_test_uas,0,train_data.ideprel,N)
    print('oUAS for iterations ' + str(ITER) + '=' + str(dev_ouas),flush=True)
    print('oLAS for iterations ' + str(ITER) + '=' + str(dev_olas),flush=True)
    print('UAS for iterations ' + str(ITER) + '=' + str(dev_uas),flush=True)
    print('aUAS for iterations ' + str(ITER) + '=' + str(dev_auas),flush=True)
    print('LAS for iterations: ' + str(ITER) + '=' + str(dev_las),flush=True)
    print('aLAS for iterations: ' + str(ITER) + '=' + str(dev_alas),flush=True)
   
    test_ouas, test_olas, test_uas, test_auas, test_las, test_alas = utils.evaluate(device,test_data,Model,args.save+'test',old_test_uas,0,train_data.ideprel,N)
    print('Test oUAS for iterations ' + str(ITER) + '=' + str(test_ouas),flush=True)
    print('Test oLAS for iterations ' + str(ITER) + '=' + str(test_olas),flush=True)
    print('Test UAS for iterations ' + str(ITER) + '=' + str(test_uas),flush=True)
    print('Test aUAS for iterations ' + str(ITER) + '=' + str(test_auas),flush=True)
    print('Test LAS for iterations: ' + str(ITER) + '=' + str(test_las),flush=True)
    print('Test aLAS for iterations: ' + str(ITER) + '=' + str(test_alas),flush=True)
    #exit()
    
    #start the loop for training
    for ITER in range(args.epoch):
        
        epoch_loss = 0.0
        epoch_arc_loss = 0.0
        epoch_rel_loss = 0.0
        epoch_num = 0.0
        epoch_numu = 0.0

        range_fn = torch.randperm
        t=1
        for i in range_fn(len(train_data.buckets)).tolist():
            split_sizes = [(len(train_data.buckets[i]) - j - 1) // train_data.chunks[i] + 1
                           for j in range(train_data.chunks[i])]
            #break
            for batch in range_fn(len(train_data.buckets[i])).split(split_sizes):
                indexs = [train_data.buckets[i][j] for j in batch.tolist()]
                start_time = time.time()
                optimizer.zero_grad()
                coptimizer.zero_grad()        
                sent = [train_data.torchSent[index] for index in indexs]
                bsent = [train_data.torchBSent[index] for index in indexs]
                char = [train_data.torchChar[index] for index in indexs]
                deprel = [train_data.torchDepreltag[index] for index in indexs]
                structure = [train_data.torchArbores[index] for index in indexs]
                sib = [train_data.torchSib[index] for index in indexs]
                arc = [train_data.torchArc[index] for index in indexs]
                batch_size = len(sent)
                pad_sent,pad_bsent,pad_char,pad_deprel,pad_structure,pad_sib,pad_arc,lengths, mask_tree = utils.padding_train(sent,bsent,char,deprel,structure,sib,arc,batch_size)
                
                print('size of batch')
                print(lengths.sum())

                loss, loss_arc, loss_rel, loss_acoe, loss_rcoe = Model.train(pad_sent,pad_bsent,pad_char,pad_deprel,pad_structure,pad_sib,pad_arc,lengths, mask_tree)

                epoch_loss += loss
                epoch_arc_loss += loss_arc
                epoch_rel_loss += loss_rel
                epoch_num += 1.0
                sentNum+=1.0
                sumLoss+=loss
                data_to_save['sumLoss'].append(sumLoss/sentNum)
                data_to_save['loss'].append(loss)
                delta_time = time.time()-start_time
                iter_time += delta_time

                print('batch: {}/{} avg.loss: {} loss: {} avg.time: {} time: {}'.format(t, num_batch, sumLoss/sentNum,loss,iter_time / sentNum,delta_time), flush=True,end='\n')
                t+=1
        #epoch_num = 1
        print('finish epoch ' + str(ITER) + ' epoch loss: ' + str(epoch_loss/epoch_num) + ' epoch arc_loss: ' + str(epoch_arc_loss/epoch_num) + ' epoch rel_loss: ' + str(epoch_rel_loss/epoch_num),flush=True)
        #Model.T += 1.0/100
        #if Model.T>1: Model.T=1


        #start to evaluate over the training data and dev data
        dev_ouas, dev_olas, dev_uas, dev_auas, dev_las, dev_alas = utils.evaluate(device,dev_data,Model,args.save+'ptest' + str(ITER) + '.conll',old_test_uas,ITER,train_data.ideprel,N)
        print('oUAS for iterations ' + str(ITER) + '=' + str(dev_ouas),flush=True)
        print('oLAS for iterations ' + str(ITER) + '=' + str(dev_olas),flush=True)
        print('UAS for iterations ' + str(ITER) + '=' + str(dev_uas),flush=True)
        print('aUAS for iterations ' + str(ITER) + '=' + str(dev_auas),flush=True)
        print('LAS for iterations: ' + str(ITER) + '=' + str(dev_las),flush=True)
        print('aLAS for iterations: ' + str(ITER) + '=' + str(dev_alas),flush=True)
        if (dev_las > old_las) and (ITER>=0): 
            if index_best!=0:
                os.remove(args.save+'DIS'+str(index_best)+'.pth')
                for i in range(N):
                    os.remove(args.save+'Sep'+str(i)+'_'+str(index_best)+'.pth')
            index_best = ITER
            old_uas = dev_uas
            old_las = dev_las
            t_patience = 0
            #torch.save({'optimizer_state_dict': Model.optimizer.state_dict(),'scheduler_state_dict': Model.scheduler.state_dict()}, args.save+'DIS'+str(ITER)+'.pth')
            torch.save({'model_state_dict': Model.cnet.state_dict(),'optimizer_state_dict': Model.optimizer.state_dict(),'scheduler_state_dict': Model.scheduler.state_dict(),'coptimizer_state_dict': Model.coptimizer.state_dict(),'cscheduler_state_dict': Model.cscheduler.state_dict()}, args.save+'DIS'+str(ITER)+'.pth')
            for i in range(N):
                torch.save({'model_state_dict': Model.net.models[i].state_dict()}, args.save+'Sep'+str(i)+'_'+str(ITER)+'.pth')
        else: 
            if ITER>=0: t_patience += 1
        
        test_ouas, test_olas, test_uas, test_auas, test_las, test_alas = utils.evaluate(device,test_data,Model,args.save+'ptest' + str(ITER) + '.conll',old_test_uas,ITER,train_data.ideprel,N)
        print('Test oUAS for iterations ' + str(ITER) + '=' + str(test_ouas),flush=True)
        print('Test oLAS for iterations ' + str(ITER) + '=' + str(test_olas),flush=True)
        print('Test UAS for iterations ' + str(ITER) + '=' + str(test_uas),flush=True)
        print('Test aUAS for iterations ' + str(ITER) + '=' + str(test_auas),flush=True)
        print('Test LAS for iterations: ' + str(ITER) + '=' + str(test_las),flush=True)
        print('Test aLAS for iterations: ' + str(ITER) + '=' + str(test_alas),flush=True)
        if t_patience > 20: break
        #exit()
    with open(args.save+'score.pickle', 'wb') as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('All bugs fixed!')


