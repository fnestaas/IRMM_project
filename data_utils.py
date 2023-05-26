import pandas as pd 
import numpy as np 
import torch 
from torch.utils.data import Dataset

def moving_avg(df: pd.DataFrame, alpha=.25):
    """
    Exponential moving average to smoothen the signal
    """
    assert len(df['engine_no'].unique()) == 1, 'multiple engines in same smoothing!'
    # tmp = df.set_index('engine_no').sort_index()
    tmp = df.ewm(alpha=alpha).mean()

    return tmp.reset_index(drop=True)

class MyDataset(Dataset):
    """
    Custom dataset
    """
    def __init__(self, data_path, train=True, normalize=True, duration=50, pad=True, thresholds=None, n_classes=3, smoothing=.25, mean=None, std=None, skip_chunk=10, df=None):
        super().__init__()
        self.path = data_path
        self.df = df 
        self.train = train
        self.normalize = normalize
        self.pad = pad 
        self.thresholds = thresholds
        self.n_classes = n_classes
        self.alpha = smoothing
        self.mean = mean if mean is None else pd.Series(mean) # pd.DataFrame({k: [v] for k, v in mean.items()})
        self.std = std if std is None else pd.Series(std) # pd.DataFrame({k: [v] for k, v in std.items()})
        self.skip_chunk = skip_chunk
        self.duration = duration 
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
                df['engine_no'] = df['engine_no'].iloc[0]
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
                new_df = pd.concat([moving_avg(df[int(i*skip): int(i*skip) + duration], alpha=self.alpha) for i in range(n_chunks)], axis=0)
            else:
                new_df = moving_avg(pad_df(df.copy(), duration), alpha=self.alpha)
            return new_df
        
        def prep_test(df, duration, chunk):
            N = (len(df) - duration) // chunk
            if N > 0: new_df = pd.concat([moving_avg(df.iloc[i*chunk: i*chunk + duration], alpha=self.alpha) for i in range(N)], axis=0)
            else: new_df = moving_avg(pad_df(df.copy(), duration), alpha=self.alpha)
            return new_df

        def prepare_sample(df, i, duration):
            """
            Take one engine i and 
            - smoothen the signal
            - make blocks of duration duration (can be helpful if we want to predict on short time windows)
            - pad if necessary (TODO: we could also implement random selection instead of splitting the data into chunks of size duration)
            """
            x = df.where(df['engine_no'] == i).dropna()
            x = x.drop(columns=['RUL', 'time_in_cycles', 'index', 'Unnamed: 0'], inplace=False, errors='ignore') 
            # x = moving_avg(x) # this has to happen per sample...
            if self.pad:
                x = pad_df(x, duration)
            else:
                if self.train: x = shift_to_fit(x, duration)
                else: x = prep_test(x, duration, self.skip_chunk)
            retval = torch.tensor(x.to_numpy())

            # if self.normalize: retval = (retval -retval.mean()) / retval.std()

            return retval.reshape((-1, duration, x.shape[-1]))
            
        
        if self.df is None: df = pd.read_csv(self.path)
        else: df = self.df 
        
        if self.normalize: 
            rul = df['RUL'] # don't normalize the label
            if self.mean is None: self.mean = df.mean()
            if self.std is None: self.std = df.std()
            self.mean = self.mean[df.columns]
            self.std = self.std[df.columns]
            df = (df - self.mean) / self.std
            # df = (df - m) / (M - m)
            df['RUL'] = rul

        # we will need the lengths of the time series for each engine to recover the label
        lens = [len(df.where(df['engine_no'] == i).dropna()) for i in pd.unique(df['engine_no'].values)]
        self.lens = lens
        # make per-engine samples
        x_engines = [prepare_sample(df, i, duration) for i in pd.unique(df['engine_no'].values)]

        # make labels
        if self.thresholds is None:
            # choose thresholds for an even distribution using quantiles
            self.thresholds = [df['RUL'].quantile(i/self.n_classes) for i in range(1, self.n_classes)]
        elif isinstance(self.thresholds, float):
            divisor = self.thresholds
            self.thresholds = [df['RUL'].quantile(i/self.n_classes / divisor) for i in range(1, self.n_classes)] # more classes around the failure state
        else: assert isinstance(self.thresholds, list), f'dataset not implemented for {self.thresholds=}'
        class_label = lambda x: sum([int(x > t)  for t in self.thresholds]) # 0 is the "failure" label # TODO: windowing here
        labels = df['RUL'].apply(class_label)
        
        # convert from pandas to pytorch
        self.x = torch.concat(x_engines, axis=0).double()
        self.n_features = self.x.shape[-1] # useful information to store
       
        # recover vector of labels
        labels = labels.to_list()
        cumulative_len = 0
        y0 = []
        if self.pad: # pad the signal
            # offset = duration//2 # duration //2 means we take the label in the middle
            offset = duration - 1 # take the final label
            for l in lens:
                targets = labels[cumulative_len:cumulative_len + l]  #relevant labels
                y0.append(targets[offset::duration]) # take the offset-th label of each block of size duration
                if (l-1) % duration < offset: 
                    # there will be one label missing, which we set to be 0
                    y0.append([0])
                cumulative_len = cumulative_len + l
        else:
            if self.train:
                for l in lens: # the counterpart to shift_to_fit for y
                    # instead of padding, we split the signal into segments all containing data, but which potentially overlap
                    mod = l % duration
                    n_chunks = l //duration + int(mod != 0)
                    skip = 0 if l <= duration else (l - duration) / (n_chunks - 1) # the distance between the start of each frame
                    # offset = int(skip // 2) # if offset is skip // 2, then we take the label in the middle of the frame. 
                    offset = int(skip - 1) # take the final label of each frame
                    targets = labels[cumulative_len:cumulative_len + l]  #relevant labels
                    cumulative_len = cumulative_len + l
                    if n_chunks == 1 and l-1 < offset: # short frame
                        y0.append([0])
                    else:
                        y0.append([targets[offset + int(i*skip)] for i in range(n_chunks)]) # take the offset-th label of each block of size duration
            else:
                # prepare a test dataset by skipping a fixed length each time
                for l in lens:
                    if l < duration:
                        y0.append([0])
                    else:
                        N = (l - duration) // self.skip_chunk
                        y0.append([labels[cumulative_len + i*self.skip_chunk + duration - 1] for i in range(N)])
                    cumulative_len += l

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
        return {i: torch.sum(torch.where(self.y == i, 1, 0)).item() for i in range(self.y.max().int() + 1)}

    def get_mean(self):
        return self.mean.to_dict()
    
    def get_std(self):
        return self.std.to_dict()
    
    def map_id_to_engine_time(self, id):
        """
        given an id in the dataset, map it to the correct engine and time for that time series 
        """
        if self.train or self.pad: raise NotImplementedError('map_id_to_engine is only available for test data with pad==False')
        cum_idx = 0
        for i, l in enumerate(self.lens):
            skip = 1 if l < self.duration else (l - self.duration) // self.skip_chunk
            cum_idx += skip
            if id < cum_idx:
                engine = i 
                idx = id - (cum_idx - skip)
                return (engine, idx)



class AdvDataset(Dataset):
    def __init__(self, images, ref_dataset):
        super().__init__()
        self.images = images 
        self.ref = ref_dataset 
    
    def __len__(self):
        return len(self.ref)
    
    def __getitem__(self, index):
        return self.images[index], self.ref[index][1]
    
    def get_deviation(self):
        # does not work properly, but the attacks themselves still seem to be fine 
        max_deviations = torch.zeros((len(self.images), ))
        mean_deviations = torch.zeros((len(self.images), ))
        for i, (adv, org) in enumerate(zip(self.images, self.ref)):
            org, j = org # only data
            mean_deviations[i] = (torch.abs(adv - org)).mean()
            max_deviations[i] = (torch.abs(adv - org)).max() # TODO: could it be that these don't actually have the same index?
        # the mean of the means is the true mean since all data have the same shape
        return {'mean': mean_deviations.mean().item(), 'max': max_deviations.max().item()}

    def get_incorrect(self, adv_preds, preds, max_examples=10):
        """
        if pred was correct, find the up to max_examples signals which tripped up the model
        """
        if preds.dim() > 1:
            preds = torch.argmax(preds, axis=1)
        if adv_preds.dim() > 1:
            adv_preds = torch.argmax(adv_preds, axis=1)
        id_correct = [i for i, (p, y) in enumerate(zip(preds, self.ref.y)) if p == y] # correct predictions
        id_different = [i for i, (p, y) in enumerate(zip(adv_preds, preds)) if p != y] # changed by adversary
        # we want the intersection of these two:
        ids = [i for i in range(len(self)) if i in id_correct and i in id_different][:max_examples]
        if len(ids) == 0: return None, None, None, None
        return torch.stack(
                [self.images[i] for i in ids]
            ), torch.stack(
                [self.ref[i][0] for i in ids]
            ), torch.stack(
                [adv_preds[i] for i in ids]
            ), torch.stack(
                [preds[i] for i in ids]
            ), torch.tensor(
                ids
            ) 

    def save_incorrect(self, adv_preds, preds, max_examples=10, dir=None, ids_only=False):
        examples, ref, adv_lbl, lbl, ids = self.get_incorrect(adv_preds, preds, max_examples)
        if examples is None: return False
        dir.mkdir(exist_ok=True, parents=True)
        if not ids_only:
            torch.save(examples, dir / 'examples.pt')
            torch.save(ref, dir / 'ref.pt')
        torch.save(adv_lbl, dir / 'adv_lbl.pt')
        torch.save(lbl, dir / 'lbl.pt')
        torch.save(ids, dir/ 'ids.pt')
        ids_times = torch.stack([torch.tensor(self.ref.map_id_to_engine_time(i)) for i in ids])
        torch.save(ids_times, dir/'ids_times.pt')


