import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import os
from reconstrain.imitation import ImitationLearning


def train(params):
    model = ImitationLearning(**vars(params))

    logger = (
        TensorBoardLogger(save_dir="./", name="tensorboard", version="")
        if params.log
        else None
    )
    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            dirpath="./checkpoints",
            filename="epoch={epoch}-loss={train/loss:0.4f}",
            auto_insert_metric_name=False,
            mode="min",
            save_last=True,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=params.patience,
        ),
    ]

    print("starting training")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        precision=64,
        gpus=params.gpus,
        max_epochs=params.max_epochs,
        default_root_dir=".",
    )

    # check if checkpoint exists
    ckpt_path = "./checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None

    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument("--log", type=int, default=1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="-1 uses all cpus. 0 uses the main process.",
    )

    # model arguments
    group = parser.add_argument_group("Model")
    ImitationLearning.add_args(group)

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=100)
    group.add_argument("--gpus", type=int, default=1)

    params = parser.parse_args()
    train(params)
