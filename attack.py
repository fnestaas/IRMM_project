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
    def __init__(self, model, eps=1, alpha=.1, noise_std=.1, steps=100, random_start=True, lower=None, upper=None):
        super().__init__()
        self.model = model 
        self.eps = eps 
        self.alpha = alpha 
        self.std = noise_std
        self.steps = steps 
        self.random_start = random_start
        self.lower = lower 
        self.upper = upper

    def forward(self, loader):
        """
        untargeted attack
        """
        def step(input_data, labels):
            input_data.requires_grad = True
            outputs = self.model(input_data)
            loss = criterion(outputs, one_hot(labels.to(torch.int64), num_classes=3).float())
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

            deviation = torch.abs(pert_data - data)
            print(f'average absolute deviation: {deviation.mean()}')
            print(f'max abs deviation: {deviation.max()}')
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

            deviation = torch.abs(pert_data - data)
            print(f'average absolute deviation: {deviation.mean()}')
            print(f'max abs deviation: {deviation.max()}')
            perturbed.append(pert_data)
                
        return torch.concat(perturbed, axis=0)

    def pgd_step(self, imgs, data_grad):
        epsilon = self.alpha
        perturbed_image = imgs - epsilon*data_grad
        # Return the perturbed image
        return perturbed_image.detach()

def main(args):

    model_name = args.model_name # which model to use
    dir = Path().cwd() / 'models'
    if model_name == 'max': # take the model that was trained last
        model_name = max([str(i) for i in dir.iterdir()])
    dir = dir / model_name
    filename = dir /'model.pth'
    model = torch.load(filename)
    model.eval()

    # Take the same images as the model was tested on to generate an attack 
    data_directory = cfg('DATA_DIRECTORY')
    data_metadata = dir / 'data_metadata.json'
    with open(data_metadata) as f:
        data_metadata = json.load(f)

    # setup
    which = data_metadata['which']
    duration = data_metadata['duration']
    pad = data_metadata['pad']
    val_dataset = MyDataset(f'{data_directory}/val_engines_{which}.csv', duration=duration, pad=pad)
    loader = DataLoader(val_dataset, batch_size=32)

    # do the attack
    attacker = TargetedPGD(model)
    adv_imgs = attacker(loader)
    adv_loader = DataLoader(AdvDataset(adv_imgs, val_dataset))

    # report accuracies and copmare adversarial and normal cases
    print('adversarial:')
    validate(model, adv_loader, save=False)
    print('regular:')
    validate(model, loader, save=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='max')
    args = parser.parse_args()
    main(args)