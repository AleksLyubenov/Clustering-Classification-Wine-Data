import pandas as pd
import numpy as np

from helper_functions import *
from embedding import *

from gensim.models import Phrases
from gensim.models import word2vec as w2v
import multiprocessing


def get_word_vector(model, word):
    return model.wv[word]


def gensim_w2v_embedding(df, n_features=50, min_word_count=1, context_size=5, downsampling=1e-3, seed=42):
    num_workers = multiprocessing.cpu_count()
    wine2vec = w2v.Word2Vec(sg=1, seed=seed, workers=num_workers, size=n_features, min_count=min_word_count, window=context_size, sample=downsampling)
    wine2vec.build_vocab(df['description'])
    wine2vec.train(df['description'], total_examples=wine2vec.corpus_count, epochs=wine2vec.iter)
    
    df['description'] = df['description'].transform(lambda x: [get_word_vector(model=wine2vec, word=word) for word in x])
    print('Word2Vec Embedded DataFrame UNSUMMED: \n{}\n'.format(df))
    df['description'] = [np.sum(df['description'].iloc[i], axis=0) for i in range(len(df['description']))]
    print('Word2Vec Embedded DataFrame SUMMED: \n{}\n'.format(df))
    
    columns = ['_'+str(i)+'_' for i in range(0,n_features)]
    columns.append('target_label')
    
    indices = df.index.tolist()
    test_indices = indices[-1]
    test_df = df.loc[test_indices]
    test_df = pd.DataFrame([[test_df.description, 'X']], columns=['description', 'label'])
    train_df = df.drop(test_indices)
    train_df.index = df.index[:-1]
    test_df.index = [df.index[-1]]    
    
    Xtrain=np.asarray([train_df.iloc[i][0] for i in range(len(train_df))])
    ytrain=np.asarray(train_df.label).reshape(Xtrain.shape[0],1)
    Xtest=np.asarray([test_df.iloc[i][0] for i in range(len(test_df))])
    ytest=np.asarray(test_df.label).reshape(Xtest.shape[0],1)

    # TRAIN contains all samples from dataset & TEST contains only the sample wine provided
    TRAIN = pd.DataFrame(data=np.concatenate((Xtrain,ytrain), axis=1), columns=columns)
    TEST = pd.DataFrame(data=np.concatenate((Xtest,ytest), axis=1), columns=columns)
    TRAIN.index = df.index[:-1]
    TEST.index = [df.index[-1]]
    print('TRAIN:\n',TRAIN)
    print('TEST:\n',TEST)
    
    #TRAIN and TEST are DataFrame objects that can be used by the Random Forest Classifier
    #train_df and test_df are DataFrame object that will be used by the similarity_score function
    return TRAIN, TEST, train_df, test_df
