import numpy as np
import csv
import spacy
import re

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
def get_texts1(path,LABELS):
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
    return np.array(texts), np.array(labels)


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