"""
Test the model using consecutive preds. Save the sequence of predictions to model directory
"""

from argparse import ArgumentParser
from pathlib import Path 
from data_utils import MyDataset
from decouple import config as cfg
import json 
from torch.utils.data import DataLoader
import torch
from metrics import validate
from matplotlib import pyplot as plt 
import pandas as pd 

def pred_sequence(df, engine, path, duration, pad, normalize, n_classes, thresholds, mean, std, model, plot=False):
    df = df.where(df['engine_no'] == engine).dropna()

    test_data = MyDataset(data_path=path, duration=duration, pad=pad, normalize=normalize, n_classes=n_classes, thresholds=thresholds, mean=mean, std=std, train=False, df=df)
    test_loader = DataLoader(test_data, batch_size=32)

    preds = []
    ys = test_data.y
    for x, y in test_loader:
        predict = lambda m, z: torch.argmax(m(z), axis=-1).tolist()
        preds.extend(predict(model, x))

    if plot:
        plt.plot(list(range(len(test_data))), ys, 'o')
        plt.plot(list(range(len(test_data))), preds, 'x')
        plt.show()
    return preds, ys

def main(args):
    directory = cfg('DATA_DIRECTORY')
    model_path = Path(args.model_path)
    engine_id = None if args.engine == 'None' else int(args.engine)

    # prepare data
    with open(model_path / 'data_metadata.json') as f:
        data_metadata = json.load(f)
    which = data_metadata['which']
    duration = data_metadata['duration']
    pad = data_metadata['pad']
    normalize = data_metadata['normalize']
    n_classes = data_metadata['n_classes']
    thresholds = data_metadata['thresholds']
    mean = data_metadata['mean']
    std = data_metadata['std']

    path = f'{directory}/test_{which}.csv'
    df = pd.read_csv(path)
    # load model
    model = torch.load(model_path / 'model.pth')
    model.eval()
    engines = df['engine_no'].unique()
    if isinstance(engine_id, int): 
        engine = engines[engine_id]
        pred_sequence(df, engine, path, duration, pad, normalize, n_classes, thresholds, mean, std, model, plot=True)
    else:
        ys = []
        preds = []
        for engine in engines:
            p, y = pred_sequence(df, engine, path, duration, pad, normalize, n_classes, thresholds, mean, std, model)
            preds.append(p)
            ys.append(y.tolist())
        # (model_path / 'preds_ys.json').mkdir(exist_ok=True, parents=True)
        with open(model_path / 'preds_ys.json', "wt") as f:
            json.dump({'preds': preds, 'ys': ys}, f)

        
    



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', help='the full path of the model as a string')
    parser.add_argument('--engine', default='None', help='out of the unique engines, choose this in the list')

    args = parser.parse_args()
    main(args)

