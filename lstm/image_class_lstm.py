import math
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.jit as jit
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm

class RNN_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, rnn_model='LSTM'):
        super().__init__()
        assert rnn_model in ['GRU', 'LSTM']
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        self.layer_norm_h = nn.LayerNorm([1, hidden_size])
        self.layer_norm_c = nn.LayerNorm([1, hidden_size])

        self.linear = nn.Linear(hidden_size, 10)

        self.rnn_model = rnn_model
        self.n_layers = 1
        self.n_directions = 1
        self.hidden_size = hidden_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _reset_hidden(self, batch_size):
        if self.rnn_model == 'LSTM':
            self.hidden = (
                torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size,
                            dtype=torch.float, device=self.device),
                torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size,
                            dtype=torch.float, device=self.device),
            )
        elif self.rnn_model == 'GRU':
            self.hidden = (
                torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size,
                            dtype=torch.float, device=self.device)
            )

    def forward(self, imgs):
        batch_size = imgs.size(0)
        self._reset_hidden(batch_size)

        out = imgs.view(batch_size, -1, 1)

        # output is in the format output[batch_size, seq_len, hidden_size], (hidden, cell_state)
        out, self.hidden = self.rnn(out, self.hidden)
        out = out[:, -1, :]

        # convert output to batch_size, hidden_size
        out = out.view(batch_size, -1)

        out = self.linear(out)
        return out


class RNN_Classifier2(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.rnn = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.layer_norm_h = nn.LayerNorm([1, hidden_size])
        self.layer_norm_c = nn.LayerNorm([1, hidden_size])

        self.linear = nn.Linear(hidden_size, 10)

        self.n_layers = 1
        self.n_directions = 1
        self.hidden_size = hidden_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _reset_hidden(self, batch_size):
        self.hidden = (
            torch.zeros(batch_size, self.hidden_size,
                        dtype=torch.float, device=self.device),
            torch.zeros(batch_size, self.hidden_size,
                        dtype=torch.float, device=self.device),
        )

    def forward(self, imgs):
        batch_size = imgs.size(0)
        self._reset_hidden(batch_size)

        imgs = imgs.view(batch_size, -1, 1)
        imgs = imgs.permute(1, 0, 2)

        for i in range(imgs.size(0)):
            self.hidden = self.rnn(imgs[0].view(batch_size, -1), self.hidden)

            # h, c = self.hidden

            # h = self.layer_norm_h(h)
            # c = self.layer_norm_c(c)

            # self.hidden = (h, c)

        out = self.hidden[0]
        out = out.view(batch_size, -1)

        out = self.linear(out)
        return out


class RNN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.rnn = RNN_Classifier2(input_size=1)

    @staticmethod
    def compute_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb):
        imgs, y = batch

        y_hat = self.rnn(imgs)

        loss = self.compute_loss(y_hat, y)

        return {'loss': loss}

    # def on_after_backward(self):
    #     print(self.rnn.rnn.all_weights[0])
    #     exit()
    #     params = self.state_dict()
    #     for name, grads in params.items():
    #         print(name, grads)
    #         exit()
    #     exit()

    def validation_step(self, batch, batch_nb):
        imgs, y = batch

        y_hat = self.rnn(imgs)

        _, pred = y_hat.max(1)
        acc = (pred == y).sum().to(dtype=float) / pred.size(0)

        loss = self.compute_loss(y_hat, y)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):
        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
            val_acc_mean += output['val_acc']

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dic = {'val_loss': val_loss_mean.item(), 'val_acc': val_acc_mean.item()}
        return tqdm_dic

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.rnn.parameters(), lr=0.002)
        return optim

    @pl.data_loader
    def tng_dataloader(self):
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        mnist = MNIST('../mnist_dataset', train=True, download=True, transform=T)
        return DataLoader(mnist, batch_size=512, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        mnist = MNIST('../mnist_dataset', train=False, download=True, transform=T)
        return DataLoader(mnist, batch_size=128, shuffle=False)


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from test_tube import Experiment

    log_dir = 'pt_lightning_logs'

    if log_dir not in os.listdir():
        os.mkdir(log_dir)

    exp = Experiment(
        name=os.path.join(log_dir, 'simple_rnn'),
        save_dir=os.getcwd(),
        autosave=True,
        version='simple_rnn2',
        debug=False,
    )

    model_save_path = os.path.join(exp.name, 'simple_rnn', exp.version)

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=False,
        verbose=False,
    )

    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        gpus=1,
        train_percent_check=1.0,
        test_percent_check=0.01,
    )

    model = RNN()
    trainer.fit(model)
