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
    

def save_model(model, dir, stats, data_metadata):
    def as_str(x):
        # x.mkdir(exist_ok=True, parents=True)
        return str(x)
    torch.save(model, as_str(dir / 'model.pth'))
    with open(as_str(dir / 'stats.json'), "w") as fp:
        json.dump(stats, fp)
    with open(as_str(dir / 'data_metadata.json'), "w") as fp:
        json.dump(data_metadata, fp)