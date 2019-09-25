import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm


def _make_conv_with_upsample(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 5, 2, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def _make_conv_with_undersample(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),

        nn.Conv2d(out_channels, out_channels, 2, 2, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # input = 1 x 100
        self.projection = nn.Sequential(
            nn.Linear(100, 7 * 7 * 512),
            nn.BatchNorm1d(7 * 7 * 512),
            nn.ReLU()

        )

        self.l1 = nn.Conv2d(512, 256, 5, 1, padding=2)
        self.l1_bn = nn.BatchNorm2d(256)

        self.l2 = nn.ConvTranspose2d(256, 128, 5, 2, padding=2)
        self.l2_bn = nn.BatchNorm2d(128)

        self.l3 = nn.Conv2d(128, 64, 5, 1, padding=2)
        self.l3_bn = nn.BatchNorm2d(64)

        self.l4 = nn.ConvTranspose2d(64, 1, 5, 2, padding=2)
        self.l4_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.projection(x)
        out = out.view(-1, 512, 7, 7)

        out = torch.relu(self.l1_bn(self.l1(out)))
        assert out.shape[2:] == (7, 7)

        out = torch.relu(self.l2_bn(self.l2(out, output_size=(14, 14))))
        assert out.shape[2:] == (14, 14)

        out = torch.relu(self.l3_bn(self.l3(out)))
        assert out.shape[2:] == (14, 14)

        out = torch.tanh(self.l4_bn(self.l4(out, output_size=(28, 28))))
        assert out.shape[2:] == (28, 28)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # not the best model...
        self.l1 = _make_conv_with_undersample(1, 64)
        self.l2 = _make_conv_with_undersample(64, 128)
        self.linear = nn.Linear(128 * 7 * 7, 2)

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.linear(out.view(out.size(0), -1))
        return out


# class Generator(nn.Module):
#     def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
#         super(Generator, self).__init__()
#         self.img_shape = img_shape

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(latent_dim, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, z):
#         img = self.model(z)
#         img = img.view(img.size(0), *self.img_shape)
#         return img


# class Discriminator(nn.Module):
#     def __init__(self, img_shape=(1, 28, 28)):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(int(np.prod(img_shape)), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 2),
#             # nn.Sigmoid(),
#         )

#     def forward(self, img):
#         img_flat = img.view(img.size(0), -1)
#         validity = self.model(img_flat)

#         return validity


def save_imgs(out_generator):
    global exp
    for i in range(10):
        image = out_generator[i]
        generated_digit = transforms.ToPILImage()(image.cpu())
        generated_digit.save(f'digits_conv/{i}.png')


class GAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    @staticmethod
    def compute_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        imgs, _ = batch

        latent_space = 100

        z = torch.randn(imgs.shape[0], latent_space)
        real_labels = torch.zeros(z.size(0), dtype=int)
        fake_labels = torch.ones(z.size(0), dtype=int)
        if self.on_gpu:
            z = z.cuda(imgs.device.index)
            real_labels = real_labels.cuda(imgs.device.index)
            fake_labels = fake_labels.cuda(imgs.device.index)

        out_generator = self.generator(z)

        save_imgs(out_generator[:10])

        # train generator
        if optimizer_idx == 0:
            # (make generator label instances as real)
            generator_loss = self.compute_loss(self.discriminator(out_generator), real_labels)
            loss = generator_loss
        # train discriminator
        else:
            # detect real images
            real_loss = self.compute_loss(self.discriminator(imgs), real_labels)

            # detect fake images
            fake_loss = self.compute_loss(self.discriminator(out_generator.detach()), fake_labels)

            discriminator_loss = (real_loss + fake_loss) / 2
            loss = discriminator_loss

        return {'loss': loss}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optim_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optim_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        return [optim_generator, optim_discriminator], []

    @pl.data_loader
    def tng_dataloader(self):
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        return DataLoader(MNIST('../mnist_dataset', train=True, download=True, transform=T), batch_size=128, shuffle=True)


if __name__ == '__main__':
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from test_tube import Experiment

    log_dir = 'pt_lightning_logs'

    exp = Experiment(
        name=os.path.join(log_dir, 'dcgan'),
        save_dir=os.getcwd(),
        autosave=True,
        version='more_layers',
    )

    model_save_path = os.path.join(exp.name, 'model_weights', exp.version)

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        save_best_only=False,
        verbose=True,
    )

    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        gpus=1,
    )

    model = GAN()
    trainer.fit(model)
