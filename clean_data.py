"""
Cleans data somewhat, code adapted from
https://www.kaggle.com/code/juhumbertaf/tutorial
"""

import pandas as pd 
import numpy as np 
import os
from decouple import config as cfg

def clean_df(df):
    nan_column = df.columns[df.isna().any()].tolist()
    const_columns = [c for c in df.columns if len(df[c].drop_duplicates()) <= 2]
    features_new = [feature for feature in df.columns if not (feature in nan_column or feature in const_columns)] 
    return df[features_new]

def add_rul(g):
    g['RUL'] = [max(g['time_in_cycles'])] * len(g)
    g['RUL'] = g['RUL'] - g['time_in_cycles']
    del g['engine_no']
    return g.reset_index()

if __name__ == '__main__':
    source_path = cfg('SOURCE_DIRECTORY')
    target_directory = cfg('DATA_DIRECTORY')
    for p in [source_path, target_directory]:
        if not os.path.exists(p):
            os.mkdir(p)
    data_directory = os.path.relpath('predictive-maintenance')

    data_dictionnary = {}

    operational_settings = ['op_setting_{}'.format(i + 1) for i in range (3)]
    sensor_columns = ['sensor_{}'.format(i + 1) for i in range(27)]
    features = operational_settings + sensor_columns
    metadata = ['engine_no', 'time_in_cycles']
    list_columns = metadata + features


    list_file_train = [x for x in sorted(os.listdir(data_directory)) if 'train' in x]

    for file_train in list_file_train:
        data_set_name = file_train.replace('train_', '').replace('.txt', '')
        file_test = 'test_' + data_set_name + '.txt'
        rul_test = 'RUL_' + data_set_name + '.txt'
        
        data_dictionnary[data_set_name] = {
            'df_train': pd.read_csv(os.path.join(data_directory, file_train), sep=' ', header=None, names=list_columns),
            'df_test': pd.read_csv(os.path.join(data_directory, file_test), sep=' ', header=None, names=list_columns),
            'RUL_test' :pd.read_csv(os.path.join(data_directory, rul_test), header=None, names=['RUL']),
    }

    for data_set in data_dictionnary:
        data_dictionnary[data_set]['df_train'] = data_dictionnary[data_set]['df_train']\
                            .groupby('engine_no').apply(add_rul).reset_index()
        del data_dictionnary[data_set]['df_train']['level_1']

    for i in range(1, 5):
        chosen_dataset = f'FD00{i}'
        df = clean_df(data_dictionnary[chosen_dataset]['df_train'].copy())
        df_eval = clean_df(data_dictionnary[chosen_dataset]['df_test'].copy())
        assert all([c1 in df_eval.columns for c1 in df.columns if not (c1 in ['RUL', 'index'])]), f'failed at dataset {i}'
        df.to_csv(f'{target_directory}/train_{i}.csv', index=False)
        df_eval.to_csv(f'{target_directory}/test_{i}.csv', index=False)
