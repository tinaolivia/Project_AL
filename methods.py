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
    size = len(data)
    logits = get_output(data, model, args)
    logPy = act_func(logits)
    entropy = -(logPy*torch.exp(logPy)).sum(1)
    top_e, ind = entropy.sort(0,True)
    nsel = min(size, args.inc)
    n = 0
    total_entropy = 0
    for i in range(size):
        if not (int(ind[i]) in subset):
            subset.append(int(ind[i]))
            n += 1
            total_entropy += float(top_e[i])
            if n >= nsel: break
        
    return subset, n, total_entropy
            
def dropout(data, model, subset, act_func, args):
    model.train()
    npts = len(data)
    var = torch.zeros(npts)
    text_field = data.fields['text']
    for i,example in enumerate(data):
        probs = torch.zeros(args.num_preds,args.class_num)
        feature, target = get_feature_target(example, text_field, args)
            
        for j in range(args.num_preds):
            #Py = act_func(model(feature))
            #probs[j,:] = Py[:,0] # NB! looking at only negative class
            probs[j,:] = act_func(model(feature))
            
        mean = probs.mean(dim=0)
        var[i] = torch.abs(probs-mean).sum() # sum all? 
        
        if i % 5000 == 0: print('Example [{}]'.format(i))
        
    top_var, ind = var.sort(0,True)
    nsel = min(npts, args.inc)
    n = 0
    total_var = 0
    for i in range(npts):
        if not(int(ind[i]) in subset):
            subset.append(int(ind[i]))
            n += 1
            total_var += float(top_var[i])
            if n >= nsel: break
        
    return subset, n

def vote_dropout(data, model, subset, args):
    model.train()
    npts = len(data)
    votes = torch.zeros((npts, args.class_num))
    text_field = data.fields['text']
    label_field = data.fields['label']
    for i, example in enumerate(data):
        preds = torch.zeros(args.num_preds)
        feature, target = get_feature_target(example, text_field, args)
        
        for j in range(args.num_preds):
            preds[j] = torch.max(model(feature),1)[1]

        for j in range(args.class_num):
            votes[i,j] = (preds == j).sum()

        
        if i % 5000 == 0: print('Example [{}]'.format(i))
    
    Py = votes/args.num_preds
    ventropy = -(Py*torch.log(Py)).sum(1)
    top_ve, ind = ventropy.sort(0,True)
    nsel = min(npts, args.inc)
    n = 0
    total_ve = 0
    for i in range(npts):
        if not(int(ind[i]) in subset):
            subset.append(int(ind[i]))
            n += 1
            total_ve += float(top_ve[i])
            if n >= nsel: break
    
    return subset, n


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

def get_output(data, model, args):
    model.eval()
    npts = len(data)
    logits = torch.zeros((npts,2))
    text_field = data.fields['text']
    for i,example in enumerate(data):
        #print(i)
        feature,target = example.text, example.label
        #feature.data.t_(), target.data.sub_(1)
        feature = [[text_field.vocab.stoi[x] for x in feature]]
        feature = torch.tensor(feature)
        if feature.shape[1] < max(args.kernel_sizes): 
            feature = torch.cat((feature, torch.zeros((1,max(args.kernel_sizes)-feature.shape[1]),dtype=torch.long)), dim=1)
        with torch.no_grad(): feature = autograd.Variable(feature)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
        
        logits[i,:] = model(feature)

    return logits

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
