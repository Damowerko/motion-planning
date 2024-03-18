import pytorch_lightning as pl
import torch


class GraphConvDelayed(pl.LightningModule):
    def __init__(self, gnn, optimizer, scheduler, loss):
        super().__init__()
        self.gnn = gnn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss

    def forward(self, batch):
        return self.gnn(batch)

    def training_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch.y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(self(batch), batch.y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]
