import torch 
import json 
from pathlib import Path

class TakeLast(torch.nn.Module):
    """
    This module selects the relevant LSTM state, since by default, we get 
    much more information from the LSTM module than we need
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x[0][:, -1, :]

def save_json(dir, stats, filename='stats.json'):
    with open(str(dir / filename), "w") as fp:
        json.dump(stats, fp)

def save_model(model, dir, stats, data_metadata):
    torch.save(model, str(dir / 'model.pth'))
    save_json(dir, stats, 'stats.json')
    save_json(dir, data_metadata, 'data_metadata.json')

def choose_model(dir, model='max'):
    """
    Root is e.g. a folder containing multiple models with similar parameters
    """
    if model == 'max':
        model = max([str(i) for i in dir.iterdir()])
    elif model =='best':
        model, _ = find_best_model(dir)
    return model 

def find_best_model(root):
    """
    Root is e.g. a folder containing multiple models with similar parameters
    """
    dir = Path(root)
    best = ''
    best_acc = 0
    for subdir in dir.iterdir():
        with open(subdir / 'stats.json') as f:
            acc = json.load(f)['acc']
        if acc > best_acc:
            best_acc = acc 
            best = subdir 
    return (best, best_acc)

