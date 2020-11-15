import pandas as pd
import numpy as np
from math import sqrt
from helper_functions import *

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from string import punctuation as punc
from gensim.models import Phrases
from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()

    
class Switch(dict):
    def __getitem__(self, item):
        for key in self.keys():                   # iterate over the intervals
            if item in key:                       # if the argument is part of that interval
                return super().__getitem__(key)   # return its associated value
        raise KeyError(item)                      # if not in any interval, raise KeyError


def switch_value(i):
    switch = Switch({
        "White Wines": 0,
        "Red Wines": 1
    })
    
    return switch[i]


def normalize_text(text):
    norm_text = text.lower()
    #Replace and breaks with regular spaces
    norm_text = norm_text.replace('<br />',' ')
    norm_text = norm_text.replace(', ',' ')
    #Use regex to pad all punctuation with spaces on both sides
    norm_text = re.sub(r"([\.\",\(\)!\?;:])", " \\1 ", norm_text)
    norm_text = norm_text.lower()
    return norm_text


def tokenize_text(text):
    tokens = []
    for sentence in nltk.sent_tokenize(text):
        for word in tokenizer.tokenize(text): #nltk.word_tokenize(sentence):
            if len(word)<2:
                continue
            tokens.append(word.lower())
    return tokens


def process_text(text):
    token_list_orig = tokenize_text(text)
    token_list = []
    for token_orig in token_list_orig:
        token = lemmatizer.lemmatize(normalize_text(token_orig), pos='a') #pos = 'a' --> adjective
        if token.isdigit()==False and token not in token_list:
            token_list.append(token)
    return token_list


def process_all(sample_text):
    df = pd.read_csv('./winemaker_data.csv', encoding='latin-1', index_col='Name')
    df = df.rename(columns={"Varietal_WineType_Name": "label", "Winemakers_Notes":"description"})
    
    df['label'] = df['label'].apply(switch_value)
    df = df[['description', 'label']]
    print(df)
    sample = pd.DataFrame([[sample_text, 'X']], columns=['description', 'label'], index=['SAMPLE WINE'])
    df = df.append(sample, ignore_index=False)
    print(df)
    STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that's", "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'will', 'yet', 'therefore']
            
    df['description']=df['description'].transform(process_text)
    df['description']=df['description'].transform(lambda x: [word for word in x if word not in set(STOPWORDS)])

    bigram = Phrases(df['description'], min_count=3, delimiter=b' ')
    trigram = Phrases(bigram[df['description']], min_count=3, delimiter=b' ')

    for i in range(len(df['description'])):
        description = df['description'].iloc[i]
        bigrams_list = [b for b in bigram[description] if b.count(' ') == 1]
        trigrams_list = [t for t in trigram[bigram[description]] if t.count(' ') == 2]
        
        # Add identified bigrams to the tokenized description
        if len(bigrams_list) != 0:
            #print(bigrams_list)
            for sequence in bigrams_list:
                if sequence not in description:
                    df['description'].iloc[i].append(sequence)

        # Add identified trigrams to the tokenized description
        if len(trigrams_list) !=0:
            #print(trigrams_list)
            for sequence in trigrams_list:
                 if sequence not in description:
                    df['description'].iloc[i].append(sequence)
        
    return df
    

def similar_descriptions(simTRAIN, simTEST, n_neighbours=5):
    desc_sim = {}
    v2 = np.asarray(simTEST.iloc[0, 0:-1][0])
    if 'target_label' in simTRAIN.columns:
        simTRAIN = simTRAIN.drop(columns=['label'])
        
    for i in range(len(simTRAIN)-1):
        v1 = np.asarray(simTRAIN.iloc[i]['description'])
        similarity_score = (np.linalg.norm(v1) * np.linalg.norm(v2)) / np.dot(v1, v2)
        
        idx = simTRAIN.loc[simTRAIN['description'].isin([v1])].index.values[0]
        desc_sim[idx] = similarity_score     
        
    desc_sim_sorted = sorted(desc_sim.items(), key=lambda kv: kv[1], reverse=False)
    return desc_sim_sorted[:n_neighbours]
