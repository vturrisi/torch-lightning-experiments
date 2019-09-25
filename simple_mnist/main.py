import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from model import Network


checkpoint = ModelCheckpoint(
    filepath='model_ckp',
    save_best_only=True,
    verbose=True,
    monitor='avg_val_loss',
    mode='min',
    prefix=''
)

model = Network()

# exp = Experiment(version=checkpoint)
trainer = Trainer(checkpoint_callback=checkpoint, gpus=1)
trainer.fit(model)
