# THIS IS MY MAIN FILE
# Changes to be done:
    # - Change the dataloading files so that they load my datasets (must also change mydatasets.py for this)
    # - Change so that we use the correct model (could test with the cnn first)
    # - Have changed from model.py to models.py -> must change this in the script as well!

import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets
import models # changed from model
import train
import mydatasets


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
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
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# THESE TWO MUST BE CHANGED FOR MY DATASETS
# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
                                (train_data, dev_data), 
                                batch_sizes=(args.batch_size, len(dev_data)),
                                **kargs)
    return train_iter, dev_iter

# -------------------------------------------------------------------------------
# My code - loading twitter data function

def twitter(text_field, label_field, **kargs):
    train_data, val_data = mydatasets.Twitter.splits(text_field, label_field)
    text_field.build_vocab(train_data, val_data)
    label_field.build_vocab(train_data, val_data)
    train_iter, val_iter = data.Iterator.splits((train_data, val_data), batch_sizes=(args.batch_size, len(val_data)), **kargs)
    return train_iter, val_iter

# My code - loading data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=True)
train_iter, val_iter = twitter(text_field, label_field, device=-1, repeat=False)
# -------------------------------------------------------------------------------


# load data
'''print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False) # NB!!! 
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False) # NB!!!'''


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
    
    
# -------------------------------------------------------------------------------
# Models
#model = models.Model(args) # concrete dropout NB!! MUST POSSIBLY CHANGE SOMETHING IN THE MODEL 
model = models.CNN_Text(args) # CNN
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    model.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    model = model.cuda()

# -------------------------------------------------------------------------------

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, model, text_field, label_field, args.cuda) # model must be changed here
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, model, args) # and model must be changed here
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, val_iter, model, args) # and here
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')