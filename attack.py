"""
Take an LSTM model and attack it using an attack from torchattacks
To that end, take an input batch of size (N, n_features, n_timepoints), and 
reshape it  to (N, 1, ...). Then it is in an appropriate shape for an attack

It seems that the hyperparameters of the model matters
if the duration is 150, we can't really attack
but if it is 50, then all classes are predicted, and it is easier to attack

There also seems to be a lot of variability; attacking which=3 is easier than which=1

TargetedPGD is more effective than UntargetedPGD
"""

from argparse import ArgumentParser
import torch 
from pathlib import Path 
from data_utils import MyDataset, AdvDataset
from decouple import config as cfg
import json 
from model_utils import TakeLast
from torch import nn
from torch.utils.data import DataLoader 
from metrics import * 
from torch.nn.functional import one_hot

class UntargetedPGD(nn.Module):
    def __init__(self, model, eps=1, alpha=.1, noise_std=.1, steps=100, random_start=True, lower=None, upper=None, n_classes=3):
        super().__init__()
        self.model = model 
        self.eps = eps 
        self.alpha = alpha 
        self.std = noise_std
        self.steps = steps 
        self.random_start = random_start
        self.lower = lower 
        self.upper = upper
        self.n_classes = n_classes

    def forward(self, loader):
        """
        untargeted attack
        """
        def step(input_data, labels):
            input_data.requires_grad = True
            outputs = self.model(input_data)
            loss = criterion(outputs, one_hot(labels.to(torch.int64), num_classes=self.n_classes).float())
            self.model.zero_grad()
            loss.backward()
            data_grad = input_data.grad.data
            pert_data = self.pgd_step(input_data, data_grad)
            return pert_data
        
        criterion = torch.nn.CrossEntropyLoss()
        perturbed = []
        for data, labels in loader:
            if self.random_start:
                data = data + torch.normal(torch.zeros(data.shape), self.std)
            pert_data = step(data, labels)
            for _ in range(self.steps):
                pert_data = step(pert_data, labels)
            pert_data = torch.clamp(pert_data, min=data-self.eps, max=data+self.eps)

            perturbed.append(pert_data)
                
        return torch.concat(perturbed, axis=0)

    def pgd_step(self, imgs, data_grad):
        epsilon = self.alpha
        perturbed_image = imgs + epsilon*data_grad
        # Return the perturbed image
        return perturbed_image.detach()

class TargetedPGD(UntargetedPGD):
    def forward(self, loader):
        def step(input_data, target_labels):
            input_data.requires_grad = True
            outputs = self.model(input_data)
            loss = criterion(outputs, one_hot(target_labels.to(torch.int64), num_classes=3).float())
            self.model.zero_grad()
            loss.backward()
            data_grad = input_data.grad.data
            pert_data = self.pgd_step(input_data, data_grad)
            return pert_data
        
        def select_label(labels):
            """
            take the majority class, except if the target is the majority, in which case
            we take the second majority class
            """
            freqs = torch.bincount(labels.to(torch.int64)).tolist()
            s = [i[0] for i in sorted(enumerate(freqs), key=lambda x: -x[1])]
            majority = s[0]
            second = s[1]
            return torch.where(labels == majority, second, majority).float()
        
        criterion = torch.nn.CrossEntropyLoss()
        perturbed = []
        
        for data, labels in loader:
            if self.random_start:
                data = data + torch.normal(torch.zeros(data.shape), self.std)
            labels = select_label(labels)
            pert_data = step(data, labels)
            for _ in range(self.steps):
                pert_data = step(pert_data, labels)
            pert_data = torch.clamp(pert_data, min=data-self.eps, max=data+self.eps)

            perturbed.append(pert_data)
                
        return torch.concat(perturbed, axis=0)

    def pgd_step(self, imgs, data_grad):
        epsilon = self.alpha
        perturbed_image = imgs - epsilon*data_grad
        # Return the perturbed image
        return perturbed_image.detach()

def main(args):

    model_name = args.model_name # which model to use
    attack_type = args.attack_type
    eps = float(args.perturbation_size)
    dir = Path().cwd() / args.source_directory # directory containing the models
    if model_name == 'max': # take the model that was trained last
        model_name = max([str(i) for i in dir.iterdir()])
    dir = dir / model_name # the directory of the model to use
    filename = dir /'model.pth'
    model = torch.load(filename)
    model.eval()

    # Take the same images as the model was tested on to generate an attack 
    data_directory = cfg('DATA_DIRECTORY')
    data_metadata = dir / 'data_metadata.json'
    with open(data_metadata) as f:
        data_metadata = json.load(f)

    # prepare output directory
    if args.target_directory is None:
        stats_filename = 'adv_stats.json' 
    else:
        target_directory = dir / args.target_directory 
        target_directory.mkdir(exist_ok=True, parents=True)
        stats_filename = args.target_directory + 'adv_stats.json'

    stats_metadata = dir / 'stats.json'
    with open(stats_metadata) as f:
        model_stats = json.load(f)
    cm = model_stats['confusion_matrix']
    acc = model_stats['acc']
    # if prediction is constant, don't bother
    cm_torch = torch.tensor(cm)
    # check that at least two classes are predicted
    n_class = cm_torch.sum(axis=1)
    n_class = torch.where(n_class > 0, 1, 0).sum()
    n_pred = cm_torch.sum(axis=0)
    n_pred = torch.where(n_pred > 0, 1, 0).sum()
    if n_class < 2 or n_pred < 2:
        print('constant prediction or only one class!')
        stats = {'cm': cm, 'acc': acc, 'max_dev': 0, 'mean_dev': 0}
        save_json(dir, stats, stats_filename)
        exit()
    else:
        print(f'valid model: {n_pred=}, {n_class=}')
    
    # setup
    which = data_metadata['which']
    duration = data_metadata['duration']
    pad = data_metadata['pad']
    thresholds = data_metadata['thresholds']
    n_classes = data_metadata['n_classes']
    val_dataset = MyDataset(f'{data_directory}/val_engines_{which}.csv', duration=duration, pad=pad, thresholds=thresholds, n_classes=n_classes)
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # do the attack
    if attack_type == 'targeted':
        attacker = TargetedPGD(model, eps=eps, n_classes=n_classes)
    else:
        attacker = UntargetedPGD(model, eps=eps, n_classes=n_classes)

    adv_imgs = attacker(loader)
    adv_dataset = AdvDataset(adv_imgs, val_dataset)
    adv_loader = DataLoader(adv_dataset)

    # report accuracies and compare adversarial and normal cases
    # perform attack
    print('adversarial:')
    cm, acc = validate(model, adv_loader, savemodel=False)
    # save stuff about adversarial attack
    deviations = adv_dataset.get_deviation()
    stats = {'cm': cm, 'acc': acc, 'max_dev': deviations['max'], 'mean_dev': deviations['mean']}
    save_json(dir, stats, stats_filename)

    print('regular:')
    validate(model, loader, savemodel=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='max')
    parser.add_argument('--attack_type', default='targeted')
    parser.add_argument('--perturbation_size', default=1)
    parser.add_argument('--source_directory', default='models', help='where to find model folders')
    parser.add_argument('--target_directory', default=None, help='where to save the results of the attack')

    args = parser.parse_args()
    main(args)