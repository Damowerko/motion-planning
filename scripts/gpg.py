import argparse
import os
import re
from glob import glob
from subprocess import call
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from reconstrain.models import MotionPlanningGPG, MotionPlanningImitation
from reconstrain.utils import TensorboardHistogramLogger


def make_trainer(params):
    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if params.log
        else None
    )
    callbacks = [
        EarlyStopping(
            monitor="val/loss",
            patience=params.patience,
        ),
        (
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=f"./checkpoints/{params.operation}/",
                filename="best",
                auto_insert_metric_name=False,
                mode="min",
                save_last=True,
                save_top_k=1,
            )
            if params.log
            else None
        ),
        (
            TensorboardHistogramLogger(every_n_steps=1000)
            if params.log and params.histograms
            else None
        ),
    ]
    callbacks = [cb for cb in callbacks if cb is not None]

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


def pretrain(params):
    trainer = make_trainer(params)
    model = MotionPlanningImitation(**vars(params))
    ckpt_path = "./checkpoints/pretrain/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if os.path.exists("./checkpoints/pretrain/best.ckpt"):
        model.load_from_checkpoint("./checkpoints/pretrain/best.ckpt")
    model.save_policy("./policy.pt")


def train(params):
    trainer = make_trainer(params)
    model = MotionPlanningGPG(**vars(params))
    if os.path.exists("./policy.pt"):
        print("Loading policy")
        model.load_policy("./policy.pt")

    # check if checkpoint exists
    ckpt_path = "./checkpoints/train/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)


def test(params):
    trainer = make_trainer(params)
    if os.path.exists("./checkpoints/train/best.ckpt"):
        model = MotionPlanningGPG.load_from_checkpoint("./checkpoints/train/best.ckpt")
    else:
        model = MotionPlanningGPG.load_from_checkpoint("./checkpoints/train/last.ckpt")
    trainer.test(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("operation", metavar="OP", type=str, default="train")
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--histograms", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=1000)
    group.add_argument("--gpus", type=int, default=1)

    # model arguments
    operation = sys.argv[1]
    group = parser.add_argument_group("Model")
    if operation == "pretrain":
        MotionPlanningImitation.add_args(group)
    elif operation == "train":
        MotionPlanningGPG.add_args(group)

    params = parser.parse_args()

    if params.operation == "train":
        train(params)
    elif params.operation == "test":
        test(params)
    elif params.operation == "pretrain":
        pretrain(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")
