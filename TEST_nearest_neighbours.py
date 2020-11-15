from random import seed
from random import randrange
from csv import reader

import numpy as np
import pandas as pd

from helper_functions import save_obj, load_obj
from embedding import similar_descriptions
from sklearn.utils import shuffle

if __name__ == "__main__":
    '''
    THE SAMPLE WINE THAT IS BEING CLASSIFIED HAS THE FOLLOWING DESCRIPTION:
    text = "A wonderful merlot, with a range of flavors: ranging from graphite, herbs and blackberries, to black cherries, plums, and cocoa, often layered with notes of clove, vanilla, and cedar when aged in oak."
    '''

    simTRAIN = load_obj('simTRAIN')
    simTEST = load_obj('simTEST')

    similar_wines = similar_descriptions(simTRAIN, simTEST, 10)
    print('TOP 5 Similar Wines:')
    for i in range(len(similar_wines)):
        print('No. {}: {}'.format(i+1, similar_wines[i][0]))
