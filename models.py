import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable # same as Tensor 
import sys


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield(X, y)
            

# GRU (Gated Recurrent Unit) network 
# tutorial https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
class ConcatPoolingGRUAdaptive(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, bidirectional=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bidirectional = bidirectional
        
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec) # load pretrained vectors
        self.emb.weight.requires_grad = False # make embedding non trainabla
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, bidirectional=bidirectional)
        self.drop = nn.Dropout(p=0.5) # added dropout layer
        if bidirectional:
            self.out = nn.Linear(self.n_hidden*2*2, self.n_out)
        else:
            self.out = nn.Linear(self.n_hidden*2, self.n_out)
            
    
    def forward(self, seq, lengths):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        seq = seq.transpose(1,1)
        embs = self.emb(seq)
        embs = embs.transpose(0,1)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.out = self.gru(embs, self.h)
        gru_out, lengths = pad_packed_sequence(gru_out)
        
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0).view(bs,-1))
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0).view(bs,-1))
        outp = self.out(torch.cat([avg_pool, max_pool], dim=1))
        return F.log_softmax(outp, dim=-1)
    
    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros((2, batch_size, self.n_hidden)).to(device)
        else:
            return torch.zeros((1, batch_size, self.n_hidden)).cuda().to(device)
        

# Concrete Dropout 
# code from https://discuss.pytorch.org/t/concrete-dropout-implementation/4396
class ConcreteDropout(nn.Module):
    
    def __init__(self, layer, input_shape, wr=1e-6, dr=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        
        # Post dropout layer
        self.layer = layer
        # Input dim
        self.input_dim = np.prod(input_shape)
        # Regularisation hyper-params
        self.w_reg_param = wr
        self.d_reg_param = dr
        
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)
        
    def sum_of_square(self):
        """
        For paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square
    
    def regularisation(self):
        """
        Returns regularisation term, should be added to the loss
        """
        weights_regularizer = self.w_reg_param * self.sum_of_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.d_reg_param * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer
    
    def forward(self, x):
        """
        Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        
        self.p = nn.functional.sigmoid(self.p_logit)

        unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size()))))
        
        drop_prob = (torch.log(self.p + eps) 
                    - torch.log(1 - self.p + eps)
                    + np.log(unif_noise + eps)
                    - np.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return self.layer(x)
    
class Linear_relu(nn.Module):
    
    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())
        
    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    
    def __init__(self, wr, dr):
        super(Model, self).__init__()
        self.forward_main = nn.Sequential(
                  ConcreteDropout(Linear_relu(1, nb_features), input_shape=1, wr=wr, dr=dr),
                  ConcreteDropout(Linear_relu(nb_features, nb_features), input_shape=nb_features, wr=wr, dr=dr),
                  ConcreteDropout(Linear_relu(nb_features, nb_features), input_shape=nb_features, wr=wr, dr=dr))
        self.forward_mean = ConcreteDropout(Linear_relu(nb_features, D), input_shape=nb_features, wr=wr, dr=dr)
        self.forward_logvar = ConcreteDropout(Linear_relu(nb_features, D), input_shape=nb_features, wr=wr, dr=dr)
        
    def forward(self, x):
        x = self.forward_main(x)
        mean = self.forward_mean(x)
        log_var = self.forward_logvar(x)
        return mean, log_var

    def heteroscedastic_loss(self, true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.sum(precision * (true - mean)**2 + log_var)
    
    def regularisation_loss(self):
        reg_loss = self.forward_main[0].regularisation()+self.forward_main[1].regularisation()+self.forward_main[2].regularisation()
        reg_loss += self.forward_mean.regularisation()
        reg_loss += self.forward_logvar.regularisation()
        return reg_loss


        
        

# CNN classifier
# code from https://github.com/Shawn1993/cnn-text-classification-pytorch
# will be altered for this project
class CNN_Text(nn.Module):
    
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C) # changed from C to Ci, changed back to C

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C) changed from C to Ci, changed back to c
        return logit