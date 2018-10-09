import torch
import torch.nn as nn
import torch.nn.functional as F

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