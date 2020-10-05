import numpy as np
import pandas as pd

import random
from random import randrange
import pickle 


#Save and load objects from file
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df



# Split a dataset into k folds
def cross_validation_fold_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    
    return dataset_split
    

def cross_validation_train_test_split(cv_set, df, test_set_index):
    df_cv_train = pd.DataFrame(columns = df.columns)
    
    cv_set = np.asarray(cv_set)
    num_folds = cv_set.shape[0] #Test set index will be 0 to num_folds
    count = 0
    
    for i in cv_set:
        if count == test_set_index:
            df_cv_test = pd.DataFrame(i, columns = df.columns)
            count += 1
        else:
            df_cv = pd.DataFrame(i, columns = df.columns)
            df_cv_train = df_cv_train.append(df_cv)
            count += 1
            
    return df_cv_train, df_cv_test


def random_exclude(excluded, range_list):
    number = random.randrange(start=range_list[0], stop=range_list[-1])
    if number == excluded:
        return random_exclude(excluded, range_list)
    else:
        return number  