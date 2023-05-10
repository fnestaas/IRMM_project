import torch 
import json 

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