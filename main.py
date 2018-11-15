#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import train_al


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=10, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=True, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
# active learning 
parser.add_argument('-method', type=str, default=None,
                    help='active learning query strategy [default: None], alternatives: random,entropy, egl, dropout,vote dropout,concrete dropout')
parser.add_argument('-rounds', type=int, default=1, help='rounds of active querying [default: 10]')
parser.add_argument('-inc', type=int, default=10, help='number of instances added to training data at each round')
parser.add_argument('-num_preds', type=int, default=100, help='number of predictions made when computing dropout uncertainty [default:100]')
args = parser.parse_args()

if args.test: filename  = 'cnn_test.txt'
else: filename = 'cnn.txt'

with open(filename, 'w') as file:
    file.write('Arguments defined. \n')

# load twitter dataset
def twitter_iterator(text_field, label_field, **kargs):
    datafields = [("text", text_field), ("label", label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data', train='train.csv', validation='val.csv',test='test.csv',
                                               format='csv', fields=datafields)
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    train_iter = data.BucketIterator(trn, batch_size=args.batch_size,**kargs)
    val_iter = data.BucketIterator(val, batch_size=args.batch_size,**kargs)
    test_iter = data.BucketIterator(tst, batch_size=args.batch_size,**kargs)
    return train_iter, val_iter, test_iter

def twitter_dataset(text_field,label_field,**kargs):
    datafields = [("text",text_field),("label",label_field)]
    trn, val, tst = data.TabularDataset.splits(path='data',train='train.csv', validation='val.csv', test='test.csv',
                                               format='csv', fields=datafields)
    text_field.build_vocab(trn)
    label_field.build_vocab(trn)
    
    #return list(trn), list(val), list(tst) 
    return trn, val, tst       
    

with open(filename, 'a') as file:
    file.write('\nDataloaders defined. \nLoading data ... \n')


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
if args.snapshot is not None: train_set, val_set, test_set = twitter_dataset(text_field, label_field, device=-1, repeat=False)
else: train_iter, dev_iter, test_iter = twitter_iterator(text_field, label_field, device=-1, repeat=False)

with open(filename, 'a') as file:
    file.write('Data loaded. \n \nUpdating arguments ... \n')


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

with open(filename, 'a') as file:
    file.write('Arguments updated. \n \n Parameters: \n')

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    with open(filename, 'a') as file:
        file.write('\t{}={}\n'.format(attr.upper(),value))


# model
with open(filename, 'a') as file:
    file.write('\nDefining model ... \n') 
cnn = model.CNN_Text(args)
with open(filename, 'a') as file:
    file.write('Model defined. \n \nSnapshot ... \n')
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    with open(filename, 'a') as file:
        file.write('Loading model from {}...\n'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))
    
with open(filename, 'a') as file:
    file.write('Passed snapshot. \n \nDevice ... \n')

if args.cuda:
    with open(filename, 'a') as file:
        file.write('Setting device ... \nDevice: {} \n'.format(args.device))
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()
    with open(filename, 'a') as file:
        file.write('Model set to device. \n')

with open(filename, 'a') as file:
    file.write('Device passed. \n \nTraining and predicting ... \n')
        
    
# start training with something like: if args.al is not None: train.train_with_al etc

# train or predict
if args.predict is not None:
    with open(filename, 'a') as file:
        file.write('Predicting ... \n')
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    with open(filename, 'a') as file:
        file.write('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
        file.write('Done predicting. \n')
elif args.test:
    with open(filename,'a') as file:
        file.write('\nTesting ... \n')
        train.evaluate(test_iter, cnn, args)
        with open(filename,'a') as file:
            file.write('Done testing. \n')
elif args.method is not None:
    train_al.train_with_al(train_set,val_set,test_set,cnn,args)
else:
    print()
    try:
        with open(filename, 'a') as file:
            file.write('\nTraining ... \n')
        train.train(train_iter, dev_iter, cnn, args)
        with open(filename, 'a') as file:
            file.write('Done training. \n')
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
        with open(filename, 'a') as file:
            file.write('\n' + '-' * 89)
            file.write('\nExiting from training early. \n')

