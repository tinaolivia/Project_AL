# THIS IS MY MYDATASETS FILE

import re
import os # don't think this is necessary
import random 
import tarfile # don't think this is necessary
import urllib # don't think this is necessary
from torchtext import data

from pathlib import Path  # added by me
import csv # added by me
    
# --------------------------------------------------------------------------------
# My code - modifications of TarDataset (MyDataset) and MR (Twitter)

class MyDataset(data.Dataset):
    '''
    PATH: folder where the data is stored
    filename: name of the files, in this case train.csv and test.csv
    dirname: ??
    '''
    
    @classmethod
    def get_file(cls, root):
        path = Path(root)
        path = path/cls.dirname
        return path/''

class Twitter(MyDataset):
    
    PATH = Path('data/')
    filename = 'twitter_data.csv'
    dirname = 'twitter_data'
    
    @staticmethod
    def sort_key(ex):
        return len(ex.text)
    
    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        
        
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()
        
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)] 
        
        if examples is None:
            examples = []
            with open('data/twitter_data.csv') as csvfile:
                f = csv.reader(csvfile)
                examples += [data.Example.fromlist([line[0], line[1]] , fields) for line in f]       
        super(Twitter, self).__init__(examples, fields, **kwargs)
        
        
    @classmethod
    def splits(cls, text_field, label_field, val_ratio=0.9, shuffle=True, root='.', **kwargs):
        path = cls.get_file(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        val_index = -1*int(val_ratio*len(examples))
            
        return (cls(text_field, label_field, examples=examples[:val_index]), 
                cls(text_field, label_field, examples=examples[val_index:]))
    
# --------------------------------------------------------------------------------------