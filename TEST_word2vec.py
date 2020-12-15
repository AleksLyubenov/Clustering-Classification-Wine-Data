import pandas as pd
import numpy as np
from word2vec import *

if __name__ == "__main__":

    settings = {
        'window_size': 6,       # context window +- center word
        'n': 50,                # dimensions of word embeddings
        'epochs': 33,           # number of training epochs
        'learning_rate': 0.05   # learning rate
    }
    np.random.seed(0)               

    text = "A wonderful merlot, with a range of flavors: ranging from graphite, herbs and blackberries, to black cherries, plums, and cocoa, often layered with notes of clove, vanilla, and cedar when aged in oak."

    df = process_all(text)
    print('NLP Processed DataFrame: \n{}\n'.format(df))

    corpus = df['description']
    # Initialise object
    w2v = word2vec(settings)
    # Numpy ndarray with one-hot representation for [target_word, context_words]
    training_data = w2v.generate_training_data(settings, corpus)
    # Training
    w2v.train(training_data)


    #w2v = load_obj('trained_w2v_model')

    df['description'] = df['description'].transform(lambda x: [w2v.word_vec(word=w) for w in x])
    print('Word2Vec Embedded DataFrame UNSUMMED: \n{}\n'.format(df))
    df['description'] = [np.sum(df['description'].iloc[i], axis=0) for i in range(len(df['description']))]
    print('Word2Vec Embedded DataFrame SUMMED: \n{}\n'.format(df))

    columns = ['_'+str(i)+'_' for i in range(0,50)]
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

    TRAIN = pd.DataFrame(data=np.concatenate((Xtrain,ytrain), axis=1), columns=columns)
    TEST = pd.DataFrame(data=np.concatenate((Xtest,ytest), axis=1), columns=columns)
    TRAIN.index = df.index[:-1]
    TEST.index = [df.index[-1]]

    print('TRAIN:\n{}\n'.format(TRAIN))
    print('TEST:\n{}\n'.format(TEST))

    save_obj(TRAIN, 'TRAINC')
    save_obj(TEST, 'TESTC')
    save_obj(train_df, 'simTRAINC')
    save_obj(test_df, 'simTESTC')
    save_obj(w2v, 'trained_w2v_modelC')

