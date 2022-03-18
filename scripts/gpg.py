import argparse
import os
import re
from glob import glob
from subprocess import call

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from reconstrain.gpg import GPG
from reconstrain.utils import TensorboardHistogramLogger


def make_trainer(params):
    logger = None
    callbacks = [
        EarlyStopping(
            monitor="train/loss",
            patience=params.patience,
        ),
    ]
    if params.log:
        logger = TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        callbacks += [
            ModelCheckpoint(
                monitor="train/loss",
                dirpath="./checkpoints",
                filename="epoch={epoch}-loss={train/loss:0.4f}",
                auto_insert_metric_name=False,
                mode="min",
                save_last=True,
                save_top_k=1,
            ),
            TensorboardHistogramLogger(every_n_steps=1000),
        ]

    print("starting training")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=params.log,
        precision=32,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
        check_val_every_n_epoch=50,
    )
    return trainer


def train(params):
    model = GPG(**vars(params))
    trainer = make_trainer(params)
    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)


def test(params):
    # load from checkpoint with lowest loss
    filenames = glob("./checkpoints/[!last]*.ckpt")
    losses = [
        float(re.search("loss=(-*[0-9]+.[0-9]+)", filename).group(1))
        for filename in filenames
    ]
    model = GPG.load_from_checkpoint(filenames[np.argmin(losses)])
    trainer = make_trainer(params)
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("operation", metavar="OP", type=str, default="train")
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)

    # model arguments
    group = parser.add_argument_group("Model")
    GPG.add_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()

    if params.operation == "train":
        train(params)
    elif params.operation == "test":
        test(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")
