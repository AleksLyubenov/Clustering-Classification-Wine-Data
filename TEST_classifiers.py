import numpy as np
import pandas as pd

from embedding import *
from sklearn.utils import shuffle

from decision_tree_class import *
from random_forest_class import *
from pruned_decision_tree_class import *
from helper_functions import *
from embedding import *
from gensim_word2vec import *

if __name__ == "__main__":

    #THE SAMPLE WINE THAT IS BEING CLASSIFIED HAS THE FOLLOWING DESCRIPTION:
    text = "A wonderful merlot, with a range of flavors: ranging from graphite, herbs and blackberries, to black cherries, plums, and cocoa, often layered with notes of clove, vanilla, and cedar when aged in oak."

    #df = process_all(text)                         # For Gensim Embeddings
    #TRAIN, TEST, _, _ = gensim_w2v_embedding(df)   # For Gensim Embeddings
    TRAIN = load_obj('TRAIN')                      # For Naive Embeddings
    TEST = load_obj('TEST')                        # For Naive Embeddings
    
    print('TRAIN:\n', TRAIN)
    print('TEST:\n', TEST)
    
    # DECISION TREE & PRUNED DECISION TREE TRAINING AND TESTING:
    decision_tree_obj = decision_tree(df=TRAIN.iloc[0:1800,:], ml_task='classfication', max_depth=10)
    print('REGULAR TREE')
    print(decision_tree_obj.tree)
    print(decision_tree_obj.predict_example(TEST, decision_tree_obj.tree))
    save_obj(decision_tree_obj, 'DECISION_TREE_FULL_TRAIN_NAIVE')
    
    pruned_decision_tree_obj = pruned_decision_tree(decision_tree_obj.tree, df_train=TRAIN.iloc[0:1800,:], df_val=TRAIN.iloc[1800:,:], ml_task='classfication')
    print('PRUNED TREE')
    print(pruned_decision_tree_obj.tree)
    print(pruned_decision_tree_obj.predict_example(TEST, pruned_decision_tree_obj.tree))
    save_obj(pruned_decision_tree_obj, 'PRUNED_DECISION_TREE_FULL_TRAIN_NAIVE')
    '''
    
    # RANDOM FOREST TRAINING AND TESTING: 
    train_rows = len(TRAIN.axes[0])
    train_cols = len(TRAIN.axes[1])
    bootstrap_constant = int(train_rows*0.95)
    subspace_constant = int(train_cols*0.9)
    
    random_forest_obj = random_forest(train_df=TRAIN, n_trees=64, n_bootstrap=bootstrap_constant, n_features=subspace_constant, tree_max_depth=10, ml_task='classification')
    
    save_obj(random_forest_obj, 'RANDOM_FOREST_FULL_TRAIN_MYW2V')
    #random_forest_obj = load_obj('RANDOM_FOREST_FULL_TRAIN_MYW2V')
    
    
    print('PRUNED RANDOM FOREST')
    for tree_obj in random_forest_obj.forest:
        print(tree_obj.tree)
        print('\n')
    
    prediction = random_forest_obj.random_forest_predict(TEST)
    print('PREDICTION: {}'.format(prediction))
    '''

        
    