import numpy as np
import scipy.sparse as sp
from collections import defaultdict

from embedding import *
from helper_functions import *


class word2vec():

    def __init__(self, settings):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']


    def generate_training_data(self, settings, corpus):
        word_counts = defaultdict(int)
        for description in corpus:
            for word in description:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())
        self.words_list = list(word_counts.keys())
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        for sentence in corpus:
            sent_len = len(sentence)
            for i, word in enumerate(sentence):
                w_target = self.word2onehot(sentence[i])
                
                w_context = []
                # Note: window_size n will have range of 2n+1 values
                for j in range(i - self.window, i + self.window+1):
                    if j != i and j <= sent_len-1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))

                training_data.append([w_target, w_context])
        print('Training data successfully created')
        return np.array(training_data)


    def word2onehot(self, word):
        word_vec = np.zeros(self.v_count)
        word_index = self.word_index[word]
        # Change value from 0 to 1 in the corresponding location based on ID of word
        word_vec[word_index] = 1

        sparse_word_vec = sp.csr_matrix(word_vec)

        return sparse_word_vec


    def train(self, training_data):
        # np.random.uniform(HIGH, LOW, OUTPUT_SHAPE)
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        for i in range(self.epochs):
            self.loss = 0
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward_propagate(w_t.toarray().T.squeeze())
                EI = np.sum([np.subtract(y_pred, word.toarray()) for word in w_c], axis=0)
                # print("Error", EI)

                self.back_propagate(EI, h, w_t.toarray().T.squeeze())
                self.loss += -np.sum([u[word.toarray().tolist()[0].index(1)]
                                     for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('Epoch:', i, "Loss:", self.loss)


    def forward_propagate(self, x):
        # Recall that x the one-hot-encoded target word
        h = np.dot(x, self.w1)
        u = np.dot(h, self.w2)
        y_c = self.softmax(u)
        return y_c, h, u


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def back_propagate(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)


    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w


def naive_w2v_embedding(df, w_size=6, v_dim=50, n_epoch=10, l_rate=0.05):
    settings = {
        'window_size': w_size,    
        'n': v_dim,               
        'epochs': n_epoch,        
        'learning_rate': l_rate   
    }
    np.random.seed(0)               

    text = "A wonderful merlot, with a range of flavors: ranging from graphite, herbs and blackberries, to black cherries, plums, and cocoa, often layered with notes of clove, vanilla, and cedar when aged in oak."

    df = process_all(text)
    print('NLP Processed DataFrame: \n{}\n'.format(df))

    corpus = df['description']
    w2v = word2vec()
    training_data = w2v.generate_training_data(settings, corpus)
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
    
    #TRAIN and TEST are DataFrame objects that can be used by the Random Forest Classifier
    #train_df and test_df are DataFrame object that will be used by the similarity_score function
    return TRAIN, TEST, train_df, test_df


