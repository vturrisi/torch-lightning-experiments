import os

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from test_tube import Experiment

from model import Network

log_dir = 'pt_lightning_logs'

if log_dir not in os.listdir():
    os.mkdir(log_dir)

exp = Experiment(
    name=os.path.join(log_dir, 'dcgan'),
    save_dir=os.getcwd(),
    autosave=True,
    version='more_layers',
)

model_save_path = os.path.join(exp.name, 'model_weights', exp.version)

checkpoint = ModelCheckpoint(
    filepath=model_save_path,
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min',
)

trainer = Trainer(
    experiment=exp,
    checkpoint_callback=checkpoint,
    gpus=1,
)

model = Network()

# exp = Experiment(version=checkpoint)
trainer = Trainer(checkpoint_callback=checkpoint, gpus=1)
trainer.fit(model)
