from helper_functions import *
from decision_tree_class import *
from pruned_decision_tree_class import *
from random_forest_class import *
from word2vec import *

if __name__ == "__main__":
    
    Naive_TRAIN, Naive_TEST = load_obj('TRAINC'), load_obj('TESTC')
    word2vec_model = load_obj('trained_w2v_modelC')
    Gensim_TRAIN, Gensim_TEST = load_obj('Gensim_TRAIN'), load_obj('Gensim_TEST') 
    num_folds=5
    
    # We then train our model(s) on num_folds-1 of the sets and evaluate on the final set (giving every set a chance to be the evaluation set)
    print("TREE(S) CROSS VALIDATION RESULTS")
    for TRAIN in [Gensim_TRAIN, Naive_TRAIN]:
        # We select k random samples from our dataset, and divide them into num_folds disjoint sets of equal length
        indices = TRAIN.index.tolist()
        cv_dataset_indices = random.sample(population=indices, k=500)
        cv_dataset = TRAIN.loc[cv_dataset_indices]
        cv_dataset = np.asarray(cv_dataset)
        cv = cross_validation_fold_split(dataset=cv_dataset, folds = num_folds)
        cv = np.asarray(cv)
        print('Cross Validation Split Shape: \t{}\n'.format(cv.shape))
        
        flag = 0
        total_accuracy_regular = 0
        total_accuracy_pruned = 0
        for i in range(num_folds):
            df_cv_train, df_cv_test = cross_validation_train_test_split(cv_set=cv, df=TRAIN, test_set_index=i)
            
            # Build regular tree and record accuracy
            cv_tree = decision_tree(df=df_cv_train, ml_task='classfication', max_depth=10)
            accuracy_regular = cv_tree.calculate_accuracy(df_cv_test, cv_tree.tree)
            print("REGULAR TREE Accuracy for Test Fold: \t{} \t{}".format(i, accuracy_regular))
            total_accuracy_regular += accuracy_regular
            
            j = random_exclude(excluded=i, range_list=range(num_folds))
            _, df_val = cross_validation_train_test_split(cv_set=cv, df=TRAIN, test_set_index=j)
            
            # Build pruned tree and record accuracy
            pruned_cv_tree = pruned_decision_tree(cv_tree.tree, df_train=df_cv_train, df_val=df_val, ml_task='classfication')
            accuracy_pruned = pruned_cv_tree.calculate_accuracy(df_cv_test, pruned_cv_tree.tree)
            print("PRUNED TREE Accuracy for Test Fold: \t{} \t{}".format(i, accuracy_pruned))
            total_accuracy_pruned += accuracy_pruned
            
        cv_accuracy_pruned = total_accuracy_pruned/num_folds
        cv_accuracy_regular = total_accuracy_regular/num_folds
        if flag == 0:
            print('(GENSIM) REGULAR TREE Cross Validation Accuracy: \t{}'.format(cv_accuracy_regular))
            print('(GENSIM) PRUNED TREE Cross Validation Accuracy: \t{}'.format(cv_accuracy_pruned))
        else:
            print('(NAIVE) REGULAR TREE Cross Validation Accuracy: \t{}'.format(cv_accuracy_regular))
            print('(NAIVE) PRUNED TREE Cross Validation Accuracy: \t{}'.format(cv_accuracy_pruned))
        
        flag = 1
        

        
    print("RANDOM FOREST CROSS VALIDATION RESULTS")
    # Reset the flag
    flag=0
    for TRAIN in [Gensim_TRAIN, Naive_TRAIN]:
        # We select k random samples from our dataset, and divide them into num_folds disjoint sets of equal length
        indices = TRAIN.index.tolist()
        cv_dataset_indices = random.sample(population=indices, k=500)
        cv_dataset = TRAIN.loc[cv_dataset_indices]
        cv_dataset = np.asarray(cv_dataset)
        cv = cross_validation_fold_split(dataset=cv_dataset, folds = num_folds)
        cv = np.asarray(cv)
        print('Cross Validation Split Shape: \t{}\n'.format(cv.shape))

        total_accuracy_forest = 0
        for i in range(num_folds):
            df_cv_train, df_cv_test = cross_validation_train_test_split(cv_set=cv, df=TRAIN, test_set_index=i)
            
            train_rows = len(df_cv_train.axes[0])
            train_cols = len(df_cv_train.axes[1])
            bootstrap_constant = int(train_rows*0.95)
            subspace_constant = int(train_cols*0.9)
            
            # Build forest and record accuracy
            cv_forest = random_forest(train_df=df_cv_train, n_trees=64, n_bootstrap=bootstrap_constant, n_features=subspace_constant, tree_max_depth=10, ml_task='classification')
            
            accuracy_forest, predictions = cv_forest.calculate_forest_accuracy(df=df_cv_test, ml_task="classification")
            print("Accuracy for Test Fold: \t{} \t{}".format(i, accuracy_forest))
            total_accuracy_forest += accuracy_forest

        cv_accuracy_forest = total_accuracy_forest/num_folds
        if flag == 0:
            print('(GENSIM) Cross Validation Accuracy: \t{}'.format(cv_accuracy_forest))
        else:
            print('(NAIVE) Cross Validation Accuracy: \t{}'.format(cv_accuracy_forest))
            
        flag=1


        