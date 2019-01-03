#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchtext
import torchtext.data as data
import sys

def random(data, subset, args):
    size = len(data)
    nsel = min(size,args.inc)
    random_perm = torch.randperm(size)
    n = 0
    for i in random_perm:
        if not(i in subset):
            subset.append(int(i))
            n += 1
            if n >= nsel: break
    return subset, n

def entropy(data, model, subset, act_func, args):
    model.eval()
    top_e = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_e, act_func = top_e.cuda(), act_func.cuda()
    ind = torch.zeros(args.inc)
    text_field = data.fields['text']
    
    for i,example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        logit = get_output(example, text_field, model, args)
        if (torch.max(torch.isnan(logit)) == 1): 
            entropy = torch.tensor([-1]) 
            print('NaN returned from get_output, iter {}'.format(i))
        else:
            if args.cuda: logit = logit.cuda()
            logPy = act_func(logit)
            entropy = -(logPy*torch.exp(logPy)).sum()
        if args.cuda: entropy = entropy.cuda()            
        if entropy.double() > torch.min(top_e).double():
            min_e, idx = torch.min(top_e, dim=0)
            top_e[int(idx)] = entropy.double()
            ind[int(idx)] = i
                        
    print('Top entropy: ', top_e)
    total_entropy = top_e.sum()
    for i in ind:
        subset.append(i)
        
    return subset, args.inc, total_entropy
            
def dropout(data, model, subset, act_func, args):
    if args.cuda: model = model.cuda()
    model.train()
    text_field = data.fields['text']
    top_var = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_var = top_var.cuda()
    ind = torch.zeros(args.inc)
    for i, example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        probs = torch.empty((args.num_preds, args.class_num))
        if args.cuda: probs = probs.cuda()
        feature = get_feature(example, text_field, args)
        if torch.max(torch.isnan(feature)) == 1: 
            var = torch.tensor([float('-inf')])
            print('NaN returned from get_feature, iter {}'.format(i))
        else:
            if args.cuda: feature, act_func = feature.cuda(), act_func.cuda()
            
            for j in range(args.num_preds):
                if args.cuda: probs[j,:] = act_func(model(feature)).cuda()
                else: probs[j,:] = act_func(model(feature)) 
            var = torch.abs(probs - probs.mean(dim=0)).sum() # absolute value or squared here?
        if args.cuda: var = var.cuda()            
        if var > torch.min(top_var):
            min_var, idx = torch.min(top_var,dim=0)
            top_var[int(idx)] = var
            ind[int(idx)] = i
    
    print('Top variance: {}'.format(top_var))
    total_var = top_var.sum()
    for i in ind:
        subset.append(i)
        
    model.eval()
        
    return subset, args.inc, total_var


def vote(data, model, subset, args):
    model.train()
    top_ve = float('-inf')*torch.ones(args.inc)
    if args.cuda: top_ve = top_ve.cuda()
    ind = torch.zeros((args.inc))
    text_field = data.fields['text']
    for i, example in enumerate(data):
        if i % int(len(data)/100) == 0: print(i)
        preds = torch.zeros(args.num_preds)
        votes = torch.zeros(args.class_num)
        feature = get_feature(example, text_field, args)
        if args.cuda: preds, votes, feature = preds.cuda(), votes.cuda(), feature.cuda()
        
        for j in range(args.num_preds):
            _, preds[j] = torch.max(model(feature),1)
            
        for j in range(args.class_num):
            votes[j] = (preds == j).sum()/args.num_preds
            
        ventropy = -(votes*torch.log(votes)).sum()
        
        if args.cuda: ventropy = ventropy.cuda()    
        if ventropy > torch.min(top_ve):
            min_ve, idx = torch.min(top_ve,dim=0)
            top_ve[int(idx)] = ventropy
            ind[int(idx)] = i 
            
    print('Top vote entropy: {}'.format(top_ve))
    total_ve = top_ve.sum()
    for i in ind:
        subset.append(i)
        
    model.eval()

    return subset, args.inc, total_ve


def get_iter_size(data_iter,args): # LEGG DENNE FUNKSJONEN I EN ANNEN FIL
    size = 0
    for batch in data_iter:
        size += len(batch)
    return size

def get_feature(data, text_field, args):
    feature = data.text
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    if len(feature[0]) < 1: 
        feature = torch.tensor([float('nan')])
    else: 
        feature = torch.tensor(feature)
        if feature.shape[1] < max(args.kernel_sizes):
            feature = torch.cat((feature, torch.zeros((1, max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
            with torch.no_grad(): feature = autograd.Variable(feature)
            if args.cuda: feature = feature.cuda()
    return feature

def get_output(data, text_field, model, args):
    model.eval()
    feature = data.text
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    if len(feature[0]) < 1: logit = torch.tensor([float('nan')])
    else:
        feature = torch.tensor(feature)
        if feature.shape[1] < max(args.kernel_sizes): 
            feature = torch.cat((feature, torch.zeros((1,max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
        with torch.no_grad(): feature = autograd.Variable(feature)
        if args.cuda:
            feature = feature.cuda()
        
        logit = model(feature)

    return logit
    
def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    #x = autograd.Variable(x, volatile=True)
    with torch.no_grad():
        x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    print(output)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    #return label_feild.vocab.itos[predicted.data[0]+1]
    return predicted.data[0]

def update_datasets(train, test, subset, args):
    fields = train.fields
    test = list(test)
    new_train = list(train)
    new_test = []
    for i in range(len(test)):
        if not (i in subset): new_test.append(test[i])
        else: new_train.append(test[i])
    return data.Dataset(new_train,fields), data.Dataset(new_test,fields)
