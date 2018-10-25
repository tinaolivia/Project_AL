import numpy as np
import pandas as pd
import csv
import spacy
import re

from sklearn.model_selection import train_test_split

def txt_to_csv(path,label,newpath):
    with open(path) as file:
        text = file.read()
    text = text.split('\n') # dividing into separate tweets
    
    with open(newpath,'w',newline='') as csvfile:
        newfile = csv.writer(csvfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in text:
            line = line.replace(',','')
            newfile.writerow([line,label])
            
            
# modification of the first get_texts from fast ai imdb notebook
def get_data_as_shuffled_arrays(path,LABELS):
    texts, labels = [],[]
    for idx,label in enumerate(LABELS):
        for fname in (path/label).glob('*.txt'):
            with open(fname,'r',encoding='utf-8') as file:
                opened = file.read()
                opened = opened.split('\n')
                for line in opened:
                    line = line.replace(',','')
                    line = line.replace('\n','')
                    texts.append(line)
                    labels.append(idx)
    texts, labels = np.array(texts), np.array(labels)
    new_idx = np.random.permutation(len(texts))
    texts, labels = texts[new_idx], labels[new_idx]
    return texts, labels


# functions fixup, get_texts, get_all copied from fast ai imdb notebook
def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

def get_texts(df, n_lbls=1):
    labels = df.iloc[:,range(n_lbls)].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[n_lbls].astype(str)
    for i in range(n_lbls+1, len(df.columns)): texts += f' {FLD} {i-n_lbls} ' + df[i].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)

def get_all(df, n_lbls):
    tok, labels = [], []
    for i, r in enumerate(df):
        print(i)
        tok_, labels_ = get_texts(r, n_lbls)
        tok += tok_;
        labels += labels_
    return tok, labels

# from medium sentiment analysis torchtext tutorial
nlp = spacy.load('en', disable=['parser','tagger','ner'])
def tokenizer(s):
    return [w.text.lower() for w in nlp(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric characters
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()




def train_val_test_split(X, y, test_size, val_size):
    '''
    Input:
        X: data
        y: labels
        test_size: fraction of the entire dataset to be for testing
        val_size: fraction of the training set to be for validation
    '''
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=test_size)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_size)
    
    return X_trn, X_val, X_tst, y_trn, y_val, y_tst

def array_to_csv(X, y, col_names, path):
    '''
    Input:
        data: data given as an array of 2 arrays
        col_names: names of the columns in the data
        path: path to where the csv is being saved
    '''
        
    df = pd.DataFrame({col_names[0]:X, col_names[1]:y}, columns=col_names)
    df.to_csv(path, header=False, index=False)
     
    

























