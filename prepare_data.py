"""
Prepare train and validation dataframes
"""

import pandas as pd 
import os 
from decouple import config as cfg
from numpy import random 
import numpy as np

def main():
    directory = cfg('DATA_DIRECTORY')
    random_state = int(cfg('RANDOM_STATE'))
    random.seed(random_state)
    for which in range(1, 5):
        df = pd.read_csv(f'{directory}/train_{which}.csv')
        n_engines = df['engine_no'].max()
        n_train = int(0.8*n_engines)
        n_test = int(0.*n_engines)
        n_val = n_engines - n_train - n_test

        engine_array = np.array(range(1, n_engines+1))
        id_train = random.choice(engine_array, replace=False, size=(n_train, ))
        id_val_test = np.delete(engine_array, id_train - 1)
        id_val = random.choice(id_val_test, size=(n_val, ), replace=False)
        id_test = np.delete(id_val_test, np.argwhere(np.isin(id_val_test, id_val)))
        df_train = df.where(df['engine_no'].isin(id_train)).dropna()
        df_val = df.where(df['engine_no'].isin(id_val)).dropna()
        df_test = df.where(df['engine_no'].isin(id_test)).dropna()

        df_train.to_csv(f'{directory}/train_engines_{which}.csv')
        df_val.to_csv(f'{directory}/val_engines_{which}.csv')
        df_test.to_csv(f'{directory}/test_engines_{which}.csv')
        

if __name__ == '__main__':
    main()