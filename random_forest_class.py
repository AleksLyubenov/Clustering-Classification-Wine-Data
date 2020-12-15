import numpy as np
from multiprocessing import Pool 
import time
import os

from decision_tree_class import *
from pruned_decision_tree_class import *
from helper_functions import *

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import mean_squared_error, r2_score


def bootstrapping(train_df, n_bootstrap):
        bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
        df_bootstrapped = train_df.iloc[bootstrap_indices]
        
        return df_bootstrapped


def build_tree_async(train_df, n_bootstrap, n_features, tree_max_depth, ml_task):
    df_bootstrapped = bootstrapping(train_df, n_bootstrap)
    
    decision_tree_obj = decision_tree(df_bootstrapped, ml_task, max_depth=tree_max_depth, random_subspace=n_features)        
    
    df_prune_train, df_prune_val = train_test_split(train_df, test_size=0.15)
    pruned_decision_tree_obj = pruned_decision_tree(decision_tree_obj.tree, df_prune_train, df_prune_val, ml_task)

    return pruned_decision_tree_obj
        
        

class random_forest:
    
    def __init__(self, train_df, n_trees, n_bootstrap, n_features, tree_max_depth, ml_task):
        self.forest = self.build_forest_async_with_callback(train_df, n_trees, n_bootstrap, n_features, tree_max_depth, ml_task)

        
    def build_forest_async_with_callback(self, train_df, n_trees, n_bootstrap, n_features, tree_max_depth, ml_task):
        global FOREST 
        FOREST = []
        
        def append_tree_obj(tree_obj):
            FOREST.append(tree_obj)
        
        arguments = (train_df, n_bootstrap, n_features, tree_max_depth, ml_task, )
        
        pool = Pool()
        for i in range(n_trees):
            pool.apply_async(build_tree_async, args=arguments, callback=append_tree_obj)
        pool.close()
        pool.join()
        
        print('build_forest_async_with_callback complete', FOREST)
        
        return FOREST    

         
    def random_forest_predict(self, test_df):
        df_predictions = {}
        for i in range(len(self.forest)):
            column_name = "tree_{}".format(i)
            
            predictions = self.forest[i].make_predictions(test_df, tree=self.forest[i].tree)
            df_predictions[column_name]=predictions

        df_predictions = pd.DataFrame(df_predictions)
        print(df_predictions)
        #We use the 0th (first) element because sometimes multiple predictions appear with the same frequency
        random_forest_predictions = df_predictions.mode(axis=1)[0]
        
        return random_forest_predictions


    def calculate_forest_accuracy(self, df, ml_task):
        predictions = self.random_forest_predict(df)
        predictions_correct = predictions == df.target_label
        accuracy = predictions_correct.mean()    
        '''
        if(ml_task == 'classification'):
            precision, recall, fscore, support = score(df['target_label'], predictions, average='weighted')
            print('precision: \t {}'.format(precision))
            print('recall: \t {}'.format(recall))
            print('fscore: \t {}'.format(fscore))
        # Regression Task
        else:
            r2 = r2_score(df['target_label'], predictions)
            mse = mean_squared_error(df['target_label'], predictions)
            print('R2 Score: \t {}'.format(r2))
            print('Mean Squared Error: \t {}'.format(mse))
        ''' 
        return accuracy, predictions
        
        