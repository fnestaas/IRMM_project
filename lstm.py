"""
Train a model 
"""

from data_utils import MyDataset
import torch
from torch.nn import LSTM 
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import one_hot
from decouple import config as cfg
from datetime import datetime 
from pathlib import Path 
import json 
from model_utils import TakeLast
from metrics import * 

def train(model, train_loader, criterion, optimizer, n_epochs=100, print_every=5, val_loader=None, dir=None, data_metadata=None):
    """
    Training loop and reporting accuracy etc
    Parameters:
        model:
            the model to be trained
        train_loader:
            DataLoader for training data
        criterion:
            criterion (loss function) to optimize during training
        optimizer:
            the optimizer to be used
        n_epochs:
            number of epochs to train for
        print_every:
            number of steps between each validation on the training data; -1 if we want to report only at end of epoch
        val_loader:
            DataLoader for validation data, or None if we do not want to validate.
    """
    best_acc = -1
    for epoch in range(n_epochs):
        running_loss = 0
        model.train()
        for i, (x, y) in enumerate(train_loader):
            y_pred = model(x)
            loss = criterion(y_pred, one_hot(y.to(torch.int64), num_classes=3).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if print_every > 0 and (i+1) % print_every == 0:
                with torch.no_grad():
                    avg_loss = running_loss / (i+1)
                    print(f'{epoch=}, {i=}, {avg_loss=}')
                    report_acc(y_pred, y)
        if print_every == -1:
            with torch.no_grad():
                avg_loss = running_loss / (i+1)
                print(f'{epoch=}, {i=}, {avg_loss=}')
                report_acc(y_pred, y)
        if val_loader is not None:
            print('validating:')
            best_acc = validate(model, val_loader, best=best_acc, dir=dir, data_metadata=data_metadata)
            print('')

    model.eval()

def main():
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir = Path().cwd() / 'models' / now
    dir.mkdir(parents=True, exist_ok=True)

    directory = cfg('DATA_DIRECTORY')
    which = 4 # which dataset to load (1, 2, 3 or 4)
    duration = 100 # duration of time chunks to feed the model
    pad = False 
    normalize = True
    data_metadata = {'which': which, 'duration': duration, 'pad': pad, 'normalize': normalize}
    train_dataset = MyDataset(f'{directory}/train_engines_{which}.csv', duration=duration, pad=pad, normalize=normalize)
    val_dataset = MyDataset(f'{directory}/val_engines_{which}.csv', duration=duration, pad=pad, normalize=normalize)

    # validate class distributions 
    print(f'class distribution on the training set: {train_dataset.class_distribution}')
    print(f'class distribution on the validation set: {val_dataset.class_distribution}')
    print('')

    hs = 128 # LSTM hidden size
    model = LSTM(input_size=train_dataset.n_features, hidden_size=hs, num_layers=2, dropout=.25, batch_first=True)
    model = torch.nn.Sequential(model, TakeLast(), torch.nn.Linear(hs, 3)).double()
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, criterion, optimizer, val_loader=val_loader, dir=dir, data_metadata=data_metadata)



if __name__ == '__main__':
    main()