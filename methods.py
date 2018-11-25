#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
import torchtext
import torchtext.data as data

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
    top_e = torch.zeros(args.inc)
    ind = torch.zeros(args.inc)
    text_field = data.fields['text']
    for i,example in enumerate(data):
        logit = get_output(example, text_field, model, args)
        logPy = act_func(logit)
        entropy = -(logPy*torch.exp(logPy)).sum()
        if i < args.inc:
            top_e[i] = entropy
            ind[i] = i
        elif entropy > torch.min(top_e):
            idx = torch.argmin(top_e)
            top_e[idx] = entropy
            ind[idx] = i
            
    total_entropy = top_e.sum()
    for i in ind:
        subset.append(i)
        
    return subset, args.inc, total_entropy
            
def dropout(data, model, subset, act_func, args):
    model.train()
    text_field = data.fields['text']
    top_var = torch.zeros(args.inc)
    ind = torch.zeros(args.inc)
    for i, example in enumerate(data):
        probs = torch.zeros((args.num_pres, args.class_num))
        feature, target = get_feature_target(example, text_field, args)
        
        for j in range(args.num_preds):
            probs[j,:] = act_func(model(feature))
            
        var = torch.abs(probs - probs.mean(dim=0)).sum()
        if i < args.inc:
            top_var[i] = var
            ind[i] = i
        elif var > torch.min(top_var):
            idx = torch.argmin(top_var)
            top_var[idx] = var
            ind[idx] = i
            
    total_var = top_var.sum()
    for i in ind:
        subset.append(i)
        
    return subset, args.inc, total_var


def vote_dropout(data, model, subset, args):
    model.train()
    top_ve = torch.zeros((args.inc))
    ind = torcs.zeros((args.inc))
    text_field = data.fields['text']
    for i, example in enumerate(data):
        preds = torch.zeros(args.num_preds)
        votes = torch.zeros(args.class_num)
        feature, target = get_feature_target(example, text_field, args)
        
        for j in range(args.num_preds):
            preds[j] = torch.max(model(feature),1)[1]
            
        for j in range(args.class_num):
            votes[j] = (preds == j).sum()/args.num_preds
            
        ventropy = -(votes*torch.log(votes)).sum()
            
        if i < args.inc:
            top_ve[i] = ventropy
            ind[i] = i
        elif ventropy > torch.min(top_ve):
            idx = torch.argmin(top_ve)
            top_ve[idx] = ventropy
            ind[idx] = ventropy
            
    total_ve = top_ve.sum()
    for i in ind:
        subset.append(i)
            
    return subset, args.inc, total_ve


def get_iter_size(data_iter,args): # LEGG DENNE FUNKSJONEN I EN ANNEN FIL
    size = 0
    for batch in data_iter:
        size += len(batch)
    return size

def get_feature_target(data, text_field, args):
    feature, target= data.text, data.label
    feature = torch.tensor([[text_field.vocab.stoi[x] for x in feature]])
    if feature.shape[1] < max(args.kernel_sizes):
        feature = torch.cat((feature, torch.zeros((1, max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
    with torch.no_grad(): feature = autograd.Variable(feature)
    if args.cuda: feature, target = feature.cuda(), target.cuda()
    return feature, target

def get_output(data, text_field, model, args):
    model.eval()
    feature,target = data.text, data.label
    feature = [[text_field.vocab.stoi[x] for x in feature]]
    feature = torch.tensor(feature)
    if feature.shape[1] < max(args.kernel_sizes): 
        feature = torch.cat((feature, torch.zeros((1,max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
    with torch.no_grad(): feature = autograd.Variable(feature)
    if args.cuda:
        feature, target = feature.cuda(), target.cuda()
        
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
