"""
Test the model using the test dataset and not just validation
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

def main(args):
    directory = cfg('DATA_DIRECTORY')
    model_path = Path(args.model_path)

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

    test_data = MyDataset(f'{directory}/test_{which}.csv', duration=duration, pad=pad, normalize=normalize, n_classes=n_classes, thresholds=thresholds, mean=mean, std=std)
    test_loader = DataLoader(test_data, batch_size=32)

    val_data = MyDataset(f'{directory}/val_engines_{which}.csv', duration=duration, pad=pad, normalize=normalize, n_classes=n_classes, thresholds=thresholds, mean=mean, std=std)
    val_loader = DataLoader(val_data, batch_size=32)

    print(test_data.thresholds)
    print(val_data.thresholds)

    # load model
    model = torch.load(model_path / 'model.pth')
    model.eval()

    # validate
    validate(model, test_loader, savemodel=False, data_metadata=data_metadata)
    print('train results')
    validate(model, val_loader, savemodel=False, data_metadata=data_metadata)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', help='the full path of the model as a string')

    args = parser.parse_args()
    main(args)

