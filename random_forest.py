import numpy as np
from decision_tree import *
from helper_functions import *
from pruning import *

import multiprocessing 
import time
import os

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error, r2_score

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped
    
    
def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, tree_max_depth, ml_task):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, ml_task, max_depth=tree_max_depth, random_subspace=n_features)
        
        #prune the tree
        #df_prune_train, df_prune_val = train_test_split(train_df, test_size=0.15)
        #pruned_tree = post_pruning(tree, df_prune_train, df_prune_val, ml_task="classification")
        forest.append(tree)
        
    return forest
    
    
def multiprocessor_build(train_df, n_bootstrap, n_features, tree_max_depth, ml_task):  
    #bootstrap the data
    df_bootstrapped = bootstrapping(train_df, n_bootstrap)
    #build regular tree
    tree = decision_tree_algorithm(df_bootstrapped, ml_task, max_depth=tree_max_depth, random_subspace=n_features)
    #prune the tree
    df_prune_train, df_prune_val = train_test_split(train_df, test_size=0.15)
    pruned_tree = post_pruning(tree, df_prune_train, df_prune_val, ml_task="classification")

    return pruned_tree
  
    
def multiprocessor_random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, tree_max_depth, ml_task):
    start_time = time.time()        
    pool = multiprocessing.Pool(n_trees)
    arguments = (train_df, n_bootstrap, n_features, tree_max_depth, ml_task, )
    forest_pruned = [pool.apply(multiprocessor_build, args=arguments) for tree in range(n_trees)]
        
    print('\nTime taken to build and prune forest = {} seconds'.format(time.time() - start_time))
    print('\nForest contains: {} trees'.format(len(forest_pruned)))    
    
    return forest_pruned
    

def random_forest_predict(test_df, forest):
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = make_predictions(test_df, tree=forest[i])
        df_predictions[column_name]=predictions
        
    df_predictions = pd.DataFrame(df_predictions)
    #We use the 0th (first) element because sometimes multiple predictions appear with the same frequency
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions
    

def calculate_forest_accuracy(df, forest, ml_task):
    predictions = random_forest_predict(df, forest)
    predictions_correct = predictions == df.target_label
    accuracy = predictions_correct.mean()    

    if(ml_task == 'classification'):
        precision, recall, fscore, support = score(df['target_label'], predictions, average='weighted')
        print('precision: \t {}'.format(precision))
        print('recall: \t {}'.format(recall))
        print('fscore: \t {}'.format(fscore))
        print('support:\t {}'.format(support))
    # Regression Task
    else:
        r2 = r2_score(df['target_label'], predictions)
        mse = mean_squared_error(df['target_label'], predictions)
        print('R2 Score: \t {}'.format(r2))
        print('Mean Squared Error: \t {}'.format(mse))
        
    return accuracy, predictions