import torch 
from model_utils import save_model, save_json

def report_acc(y_pred, y):
    """
    report accuracy of preds y_pred vs labels y
    """
    acc = pred2lbl(y_pred) == y
    acc = acc.double().mean()
    print(f'accuracy: {acc.item()}')
    return acc

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

def validate(model, val_loader, savemodel=True,  best=-1, dir=None, data_metadata=None):
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
    acc = report_acc(y_pred, y).item()
    cm = confusion_matrix(y_pred, y, n_lbl=data_metadata['n_classes'])
    # confusion matrix; each column shows the predictions for one label
    print('preds v; ys >')
    print(cm)
    
    tensor_to_list = lambda x: [x[:, i].tolist() for i in range(x.shape[1])]
    cm = tensor_to_list(cm)
    if acc > best:
        best = acc
        if savemodel:
            save_model(model, dir=dir, stats={'acc': acc, 'confusion_matrix': cm}, data_metadata=data_metadata)
    return cm, best 