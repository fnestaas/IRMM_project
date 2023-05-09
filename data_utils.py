import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset

def moving_avg(df: pd.DataFrame, alpha=.25):
    """
    Exponential moving average to smoothen the signal
    """
    tmp = df.set_index('engine_no').sort_index()
    tmp = tmp.ewm(alpha=alpha).mean()

    return tmp.reset_index()

class MyDataset(Dataset):
    """
    Custom dataset
    """
    def __init__(self, data_path, train=True, normalize=True, duration=50, pad=True):
        super().__init__()
        self.path = data_path
        self.train = train
        self.normalize = normalize
        self.pad = pad 
        self.make_dataset(duration)
    
    def make_dataset(self, duration):
        """
        Takes the data path (and other information provided in __init__) to make a dataset. 
        The dataset consists of time blocks of the specified duration, and smoothens the signal
        """
        def pad_df(df, duration):
            # pad the df into chunks of size duration, padding the last df if neccessary 
            l = (len(df) // duration) * duration 
            if l != len(df):
                df = pd.concat([df, pd.DataFrame(np.zeros((duration - len(df) % duration, df.shape[-1])), columns=df.columns)], axis=0)
            return df 
        
        def shift_to_fit(df, duration):
            """
            TODO: wrong if l is divisible by duration
            Instead of padding, repeat parts of the dataframe to make it divisible by duration
            """
            l = len(df)
            mod = l % duration
            n_chunks = len(df) // duration + int(mod != 0)
            if n_chunks > 1:
                skip = (l - duration) / (n_chunks - 1) 
                new_df = pd.concat([df[int(i*skip): int(i*skip) + duration] for i in range(n_chunks)], axis=0)
            else:
                new_df = pad_df(df.copy(), duration)
            return new_df
        
        def prepare_sample(df, i, duration):
            """
            Take one engine i and 
            - smoothen the signal
            - make blocks of duration duration (can be helpful if we want to predict on short time windows)
            - pad if necessary (TODO: we could also implement random selection instead of splitting the data into chunks of size duration)
            """
            x = df.where(df['engine_no'] == i).dropna()
            x = x.drop(columns=['RUL', 'time_in_cycles', 'index', 'Unnamed: 0'], inplace=False) 
            x = moving_avg(x)
            if self.pad:
                x = pad_df(x, duration)
            else:
                x = shift_to_fit(x, duration)
            retval = torch.tensor(x.to_numpy())
            return retval.reshape((-1, duration, x.shape[-1]))
        
        df = pd.read_csv(self.path)
        
        if self.normalize: 
            rul = df['RUL'] # don't normalize the label
            df = (df - df.mean()) / df.std()
            # df = (df - m) / (M - m)
            df['RUL'] = rul

        # we will need the lengths of the time series for each engine to recover the label
        lens = [len(df.where(df['engine_no'] == i).dropna()) for i in pd.unique(df['engine_no'].values)]
        # make per-engine samples
        x_engines = [prepare_sample(df, i, duration) for i in pd.unique(df['engine_no'].values)]
        # make labels
        if self.train:
            class_label = lambda x: int(x > 150) + int(x > 50) # 0 is the "failure" label
            labels = df['RUL'].apply(class_label)
        
        # convert from pandas to pytorch
        self.x = torch.concat(x_engines, axis=0).double()
        self.n_features = self.x.shape[-1] # useful information to store
       
        # recover vector of labels
        labels = labels.to_list()
        cumulative_len = 0
        y0 = []
        if self.pad: # pad the signal
            offset = duration//2 # duration //2 means we take the label in the middle
            for l in lens:
                targets = labels[cumulative_len:cumulative_len + l]  #relevant labels
                y0.append(targets[offset::duration]) # take the offset-th label of each block of size duration
                if (l-1) % duration < offset: 
                    # there will be one label missing, which we set to be 0
                    y0.append([0])
                cumulative_len = cumulative_len + l
        else:
            for l in lens: # the counterpart to shift_to_fit for y
                mod = l % duration
                n_chunks = l //duration + int(mod != 0)
                skip = 0 if l <= duration else (l - duration) / (n_chunks - 1) 
                offset = int(skip // 2)
                targets = labels[cumulative_len:cumulative_len + l]  #relevant labels
                cumulative_len = cumulative_len + l
                if n_chunks == 1 and l-1 < offset: # short frame
                    y0.append([0])
                else:
                    y0.append([targets[offset + int(i*skip)] for i in range(n_chunks)]) # take the offset-th label of each block of size duration

        y = []
        for l in y0:
            y = y + l
        
        # save labels as torch tensor
        self.y = torch.tensor(y).double()
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.x)
    
    @property
    def class_distribution(self):
        return {i: torch.sum(torch.where(self.y == i, 1, 0)) for i in range(self.y.max().int() + 1)}
    

class AdvDataset(Dataset):
    def __init__(self, images, ref_dataset):
        super().__init__()
        self.images = images 
        self.ref = ref_dataset 
    
    def __len__(self):
        return len(self.ref)
    
    def __getitem__(self, index):
        return self.images[index], self.ref[index][1]



