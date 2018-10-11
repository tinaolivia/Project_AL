import numpy as np

import torchtext
from torchtext import data
from torchtext import vocab

import torch.optim as optim
from pathlib import Path

from helpers import *
from models import BatchGenerator, ConcatPoolingGRUAdaptive



PATH = Path('full_data/')

# STEP 1: Define how to process the data

# define the columns that we want to process and how to process
txt_field = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

# process labels as labels, process texts as text
train_val_fields = [('labels', label_field), ('text', txt_field)]
trainds, valds = data.TabularDataset.splits(path=PATH, format='csv', train='train.csv',
                                           validation='test.csv', fields=train_val_fields,skip_header=True)


# STEP 2: Create a torchtext dataset - already done? 


# STEP 3: Load pretrained word vectors and bulding vocabulary

# specify the path to the locally saved vectors
vec = vocab.Vectors('glove.twitter.27B.200d.txt',PATH/'glove_embedding/')

# build the vocabulary using train and validation dataset and assign the vectors
txt_field.build_vocab(trainds, valds, max_size=100000, vectors=vec)

#build vocab for labels 
label_field.build_vocab(trainds)


# STEP 4: Loading the data in batches

traindl, valdl = data.BucketIterator.splits(datasets=(trainds,valds), batch_sizes=(3,3), 
                                            sort_key=lambda x: len(x.text), device=-1,
                                           sort_within_batch=True, repeat=False)

batch = next(iter(traindl))
train_batch_it = BatchGenerator(traindl, 'text', 'labels')



#STEP 5: Model and training

vocab_size = len(txt_field.vocab)
embedding_dim = 200
n_hidden = 64
n_out = 2
device = -1 # 0 GPU, -1 CPU

tranidl, valdl = data.BucketIterator.splits(datasets=(trainds,valds), batch_sizes=(512,1024), 
                                           sort_key=lambda x: len(x.text), device=0, 
                                           sort_within_batch=True, repeat=False)


# use the wrapper t0 convert Batch to data
train_batch_it = BatchGenerator(traindl, 'text', 'labels')
val_batch_it = BatchGenerator(valdl, 'text', 'labels')

# model, optimizer and loss - second argument of opt is learning rate?
model = ConcatPoolingGRUAdaptive(vocab_size, embedding_dim, n_hidden, n_out, 
                             trainds.fields['text'].vocab.vectors).to(device)
opt = optim.Adam(filter(lambda p: p.requires_grad, m.paramters()), 1e-3)
loss_func = nn.NLLLoss()


# line from the tutorial (but cant be done like this?)
#fit(model=m, train_dl=train_batch_it, val_dl=val_batch_it, loss_fn=F.nll_loss, opt=opt, epochs=5)

# training loop
epochs = 5

for epoch in range(1, epochs+1):
    running_loss = 0.0
    running_corrects = 0
    model.train() # training mode
    
    for text, label in tqdm.tqdm(train_batch_it):
        
        #Step 1: clear gradients
        opt.zero_grad()
        
        # Step 2: forward pass
        preds = model(text)
        
        # step 3: compute loss
        loss = loss_func(label, preds)
        loss.backward()
        opt.step()
        
        running_loss += loss.data[0]*text.size(0)
    
    epoch_loss = running_loss/len(trainds)
    
    # calculate the validation loss for this epoch
    val_loss = 0.0
    model.eval() # evaluation mode
    
    for text, label in val_batch_it:
        preds = model(text)
        loss = loss_func(label, preds)
        val_loss += loss.data[0]*text.size(0)
        
    val_loss /= len(valds)
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epoch_loss, val_loss))
    
# validating
test_preds = []
for text, label in tqdm.tqdm(val_batch_it):
    preds = model(text)
    preds = preds.data.numpy()
    preds = 1/(1 + np.exp(-preds))
    test_preds.append(preds)
    test_preds = np.hstack(test_preds)
    
np.save('preds', test_preds)







