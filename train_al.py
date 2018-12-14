import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import torchtext
import torchtext.data as data
import methods
import csv



def train(train_iter, dev_iter, model, round_, args):
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
                #    if args.save_best:
                #        save(model, args.save_dir, 'best', steps, args)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            #elif steps % args.save_interval == 0:
             #   save(model, 'al', 'snapshot', steps)
             
    save(model, save_dir=args.method, save_prefix='al', steps=steps, round_=round_, args=args, al=True)
                


#NB! for now this is only for binary classification (neg/pos sentiment analysis)
def train_with_al(train_set, val_set, test_set, model, args):
    
    #softmax = nn.Softmax(dim=1)
    log_softmax = nn.LogSoftmax(dim=1)
    if args.cuda: log_softmax = log_softmax.cuda()
    val_iter = data.BucketIterator(test_set, batch_size=args.batch_size, device=-1, repeat=False)
    
    initial_acc = evaluate(val_iter, model, args).cpu()
    with open('accuracies/{}_{}.csv'.format(args.method, args.dataset), mode='w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Train Size', 'Accuracy'])
        csvwriter.writerow([len(train_set) , initial_acc.numpy()])
        
        
    for al_iter in range(args.rounds):
        
        subset = []
        
        print('\nTrain: {}, Validation: {}, Test: {} \n'.format(len(train_set),len(val_set), len(test_set)))
    
    
        if args.method == 'random':
            subset,n = methods.random(test_set, subset, args)
            print('\nIter {}, selected {} samples at random\n'.format(al_iter, n))
            #print('subset after: ', subset)
            
        if args.method == 'entropy':
            subset, n, total_entropy = methods.entropy(test_set, model, subset, log_softmax, args)                
            print('\nIter {}, selected {} by entropy uncertainty, total entropy {}\n'.format(al_iter, n, total_entropy))
            #print('subset after: ', subset)
        
        if args.method == 'dropout':
            subset, n, total_var = methods.dropout(test_set, model, subset, log_softmax, args)
            print('\nIter {}, selected {} samples with dropout, total variability {}\n'.format(al_iter, n, total_var))
            #print('subset after: ', subset)
                    
        # NB! for now, this is made for positive/negative sentiment ananlysis
        if args.method == 'vote dropout':
            subset, n, total_ve = methods.vote_dropout(test_set, model, subset, args)
            print('\nIter {}, selected {} by dropout and vote entropy, total vote entropy {}\n'.format(al_iter, n, total_ve))
            #print('subset after: ', subset)
        
        print('\nUpdating datasets ...')
        train_set, test_set = methods.update_datasets(train_set, test_set, subset, args)

        #print(train_set, test_set)
        
        print('\nTrain: {}, Validation: {}, Test: {} \n'.format(len(train_set),len(val_set), len(test_set)))
        
        train_iter = data.BucketIterator(train_set, batch_size=args.batch_size, device=-1, repeat=False)
        #test_iter = data.BucketIterator(test_set, batch_size=args.batch_size, device=-1, repeat=False)
        
        print('\nLoading initial model for dataset {} ...'.format(args.dataset))
        model.load_state_dict(torch.load(args.snapshot))
        
        
        train(train_iter, val_iter, model, al_iter, args)
        
        print('\n\nLoading model {}, method {}'.format(al_iter, args.method))
        model.load_state_dict(torch.load('{}/al_{}_{}.pt'.format(args.method, args.dataset, al_iter)))
        
        acc = evaluate(val_iter, model, args).cpu()
        with open('accuracies/{}_{}.csv'.format(args.method,args.dataset), mode='a') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow([len(train_set), acc.numpy()])
        

        

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
    print(output)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    #return label_feild.vocab.itos[predicted.data[0]+1]
    return predicted.data[0]


def save(model, save_dir, save_prefix, steps, round_, args, al=False):
    if al: 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_{}_{}.pt'.format(save_prefix, args.dataset, round_)
        torch.save(model.state_dict(), save_path)
    else: 
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_prefix = os.path.join(save_dir, save_prefix)
        save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
        torch.save(model.state_dict(), save_path)
    
def save_file(itemlist,filename, args):
    with open(filename, 'w') as file:
        for item in itemlist:
            file.write('{}\n'.format(item))
    
    
    
    
    
    
    
