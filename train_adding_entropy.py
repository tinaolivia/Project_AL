#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:03:32 2018

@author: toofteda
"""

import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            
            if steps % args.test_interval == 0:
                dev_acc = evaluate(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
                


def train_with_al(train_iter, dev_iter, test_iter, model, args):
    
    subset = [] # vet ikke om jeg trenger denne
    test_size = get_iter_size(test_iter,args)
    train_size = get_iter_size(train_iter,args)
    
    for al_iter in range(args.rounds):
    
    # training model according to train function
        if args.snapshot == None: train(train_iter, dev_iter, model, args)
    
    # compute scores from the pool of "unlabeled" data
    # evaluate test accuracy
        model.eval()
        npts = get_iter_size(test_iter,args) 
        j = 0
        for i,batch in enumerate(test_iter):
            r = min(npts,j+args.batch_size)
            feature, target = batch.text, batch.label
            feature.data.t_(), target.data.sub_(1)
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()
                
            logit = model(feature)
            
           #_, preds = tensor.max(logit,1)
            
            scores[j:r,:] = F.log_softmax(logit,dim=1).data # må denne initializeres?
            j += len(batch)
            
        print('Iter {}, train score computed'.format(iter_al))
            
            
    
        if args.method == 'entropy':
            logpy = util.math.logmeanexp(scores,dim=1,keepdim=True) #function to be defined later 
            entropy = -(logpy*torch.exp(logpy)).sum(1) # computing entropy
            top_e, ind = entropy.sort(0,True) # sorting entriopies and indices
            nsel = min(test_size-train_size,args.inc) # NB! Må ha større test sett enn train sett
            n = 0
            total_entropy = 0
            test_size = get_iter_size(test_iter,args)
            for i in range(test_size):
                if not (int(ind[i]) in subset):
                    subset.add(ind[i])
                    n += 1
                    total_entropy += float(top_e[i])
                    if n >= nsel: break
            
            # finne en måte å legge til de nye eksemplene i train_iter, og fjerne dem fra test_iter
            print('\nIter {}, selected {} by entropy uncertainty, entropy {}\n'.format(al_iter, n, total_entropy))
        
        if args.method == 'curiosity': # expected model change (expecdet gradient length)
            logpy = util.math.logmeanexp(scores,dim=1,keepdim=True)
            model_param_mi=(torch.exp(score_train)*(score_train-logpy)).mean(1).sum(1)
            top_mi,ind = model_param_mi.sort(0,True)
            nsel = min(test_size-train_size,args.inc) #NB! Må ha større test set enn train set
            n = 0
            total_gain = 0
            for i in range(test_size):
                if not (int(ind[i]) in subset):
                    subset.add(ind[i])
                    n += 1
                    total_gain += float(top_mi[i])
                    if n >= nsel: break
            print('Iter {}, selected {} using expected gradient length, param information gain {}.'.format(iter_al,n,total_gain))
            
        if args.method == 'dropout':
            
    
                
              

def get_iter_size(data_iter,args): # LEGG DENNE FUNKSJONEN I EN ANNEN FIL
    size = 0
    for batch in data_iter:
        size += len(batch)
    return size

                
                
def entropy():
    return 0


def evaluate(data_iter, model, args):
    model.eval()
    corrects = 0
    avg_loss = 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


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
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    #return label_feild.vocab.itos[predicted.data[0]+1]
    return predicted.data[0]

def dropout_prediction(data_iter,model,args):
    
    model.train()
    
    npts = get_iter_size(data_iter)
    j = 0
    for batch in data_iter:
        r = min(npts, j+len(batch))
        feature,target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()
            
        for i in range(args.num_preds):
            logit = model(feature)
            _, preds[:,i] = torch.max(logit,1)
            
        mean_preds[j:r] = np.abs(preds.mean(1)-0.5)
        j += len(batch)
        
    
        
        
        
    
        
    

def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)