import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST


def _make_conv_with_undersample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, 2, 2, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Network(pl.LightningModule):

    def __init__(self):
        super().__init__()
        # not the best model...
        self.l1 = _make_conv_with_undersample(1, 32)
        self.l2 = _make_conv_with_undersample(32, 64)
        self.linear = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.linear(out.view(out.size(0), -1))
        return out

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        _, preds = y_hat.max(1)
        correct = torch.sum(y == preds)
        wrong = x.size(0) - correct
        return {'val_loss': F.cross_entropy(y_hat, y), 'correct': correct, 'wrong': wrong}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        correct = torch.stack([x['correct'] for x in outputs]).sum().item()
        wrong = torch.stack([x['wrong'] for x in outputs]).sum().item()
        acc = correct / (correct + wrong)
        return {'avg_val_loss': avg_loss, 'acc': acc}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=0.002)

    @pl.data_loader
    def tng_dataloader(self):
        # REQUIRED
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return DataLoader(MNIST('../mnist_dataset', train=True, download=True, transform=T), batch_size=128, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return DataLoader(MNIST('../mnist_dataset', train=True, download=True, transform=T), batch_size=128)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return DataLoader(MNIST('../mnist_dataset', train=True, download=True, transform=T), batch_size=128)
