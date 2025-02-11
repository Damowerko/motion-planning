import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter


class TensorboardHistogramLogger(pl.Callback):
    def __init__(self, every_n_steps=1000):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.last_called = 0

    @staticmethod
    def _get_writer(pl_module: pl.LightningModule) -> SummaryWriter:
        writer = pl_module.logger.experiment
        assert isinstance(writer, SummaryWriter), "Expexted logger to be SummaryWriter."
        return writer

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx):
        writer = self._get_writer(pl_module)
        if trainer.global_step - self.last_called >= self.every_n_steps:
            self.last_called = trainer.global_step
            for name, params in pl_module.named_parameters():
                writer.add_histogram("param/" + name, params, trainer.global_step)
                writer.add_histogram("grad/" + name, params.grad, trainer.global_step)

    def on_validation_epoch_start(self, trainer, pl_module):
        writer = self._get_writer(pl_module)
        assert isinstance(writer, SummaryWriter), "Expexted logger to be SummaryWriter."
        for name, param in pl_module.named_parameters():
            writer.add_histogram("param/" + name, param, trainer.current_epoch)
