"""
Prepare train and validation dataframes
"""

import pandas as pd 
import os 
from decouple import config as cfg


def main():
    directory = cfg('DATA_DIRECTORY')
    for which in range(1, 5):
        df = pd.read_csv(f'{directory}/train_{which}.csv')
        df_train = df.where(df['engine_no'] < 80).dropna()
        df_val = df.where(df['engine_no'] < 80).dropna()
        df_train.to_csv(f'{directory}/train_engines_{which}.csv')
        df_val.to_csv(f'{directory}/val_engines_{which}.csv')
        

if __name__ == '__main__':
    main()