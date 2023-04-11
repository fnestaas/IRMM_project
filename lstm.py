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

def train(model, train_loader, criterion, optimizer, n_epochs=10, print_every=5, val_loader=None):
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
            validate(model, val_loader)
            print('')

    model.eval()

def report_acc(y_pred, y):
    """
    report accuracy of preds y_pred vs labels y
    """
    acc = pred2lbl(y_pred) == y
    acc = acc.double().mean()
    print(f'accuracy: {acc.item()}')

def pred2lbl(y_pred):
    """
    convert model output (class probabilities) to label (a prediction of the form 0, 1, 2, ...)
    """
    return torch.argmax(y_pred, axis=-1)

def confusion_matrix(y_pred, y, n_lbl=3):
    """
    Computes the confusion matrix
    The element [i, j] is the number of times we predict i for true label j
    E.g. if row i is 0, 0, 0, then we never predict label i
    """
    y_pred = pred2lbl(y_pred)
    res = torch.zeros((n_lbl, n_lbl))
    for i in range(n_lbl):
        for j in range(n_lbl):
            res[i, j] = torch.where(torch.logical_and(y_pred == i, y == j), 1, 0).sum()
    return res

def validate(model, val_loader):
    """
    Run validations of the model vs the val_loader
    """
    model.eval()
    # accumulate predictions and report accuracy
    preds = []
    ys = []
    for x, y in val_loader:
        preds.append(model(x))
        ys.append(y)
    y_pred = torch.concat(preds, axis=0)
    y = torch.concat(ys, axis=0)
    report_acc(y_pred, y)
    # confusion matrix; each column shows the predictions for one label
    print('preds v; ys >')
    print(confusion_matrix(y_pred, y))

class TakeLast(torch.nn.Module):
    """
    This module selects the relevant LSTM state, since by default, we get 
    much more information from the LSTM module than we need
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x):
        return x[0][:, -1, :]

def main():
    directory = cfg('DATA_DIRECTORY')
    which = 1 # which dataset to load
    duration = 150 # duration of time chunks to feed the model
    train_dataset = MyDataset(f'{directory}/train_engines_{which}.csv', duration=duration)
    val_dataset = MyDataset(f'{directory}/val_engines_{which}.csv', duration=duration)
    hs = 128 # LSTM hidden size
    model = LSTM(input_size=train_dataset.n_features, hidden_size=hs, num_layers=4, dropout=.25, batch_first=True)
    model = torch.nn.Sequential(model, TakeLast(), torch.nn.Linear(hs, 3)).double()
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    train(model, train_loader, criterion, optimizer, val_loader=val_loader)



if __name__ == '__main__':
    main()