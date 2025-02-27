import argparse
import os
import sys
from functools import partial

import optuna
import torch
from lightning.pytorch.loggers import WandbLogger

from motion_planning.lightning.imitation import MotionPlanningImitation
from motion_planning.utils import get_architecture_cls, get_operation_cls, make_trainer


def parse_args():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        type=str,
        default="imitation",
        choices=[
            "imitation",
            "td3",
        ],
        help="The operation to perform.",
    )
    parser.add_argument(
        "architecture",
        type=str,
        choices=[
            "gnn",
            "transformer",
        ],
    )
    operation = sys.argv[1]
    architecture = sys.argv[2]

    if operation != "imitation":
        raise NotImplementedError(f"Operation {operation} not implemented.")

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    get_operation_cls(operation).add_model_specific_args(group)

    group = parser.add_argument_group("Architecture")
    get_architecture_cls(architecture).add_model_specific_args(group)

    # training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--no_log", action="store_false", dest="log")
    training_group.add_argument("--max_epochs", type=int, default=100)
    # disable early stopping by default (set to 10_000 epochs)
    training_group.add_argument("--patience", type=int, default=10_000)
    training_group.add_argument("--simple_progress", action="store_true")

    # reinforcement learning specific args
    if operation in ("ddpg", "td3", "ppo"):
        training_group.add_argument("--checkpoint", type=str)

    params = parser.parse_args()
    return vars(params)


def study(params: dict):
    study_name = "ump-rotary"
    storage = os.environ["OPTUNA_STORAGE"]
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=10,
        max_resource=params["max_epochs"],
        reduction_factor=4,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True,
        directions=["minimize"],
    )
    study.set_metric_names(["val/actor_loss"])
    study.optimize(
        partial(objective, default_params=params),
        n_trials=1,
    )


def objective(trial: optuna.trial.Trial, default_params: dict):
    # search over embed_dim to ensure that it is not too large
    embed_dim = 16 * trial.suggest_int("embed_dim/16", 1, 16)
    # embed_dim must be divisible by n_heads
    n_heads = 2 ** trial.suggest_int("log2(n_heads)", 0, 4)
    n_channels = embed_dim // n_heads

    params = dict(
        actor_lr=trial.suggest_float("actor_lr", 1e-8, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-16, 1, log=True),
        n_layers=trial.suggest_int("n_layers", 2, 10),
        n_heads=n_heads,
        n_channels=n_channels,
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        encoding_type="rotary",
        encoding_frequencies="linear",
        attention_window=trial.suggest_categorical(
            "attention_window", [0.0, 500.0, 1000.0]
        ),
        connected_mask=True,
        batch_size=128,
        num_workers=32,
        gamma=trial.suggest_float("gamma", 0.9, 0.999, log=True),
    )

    params = {**default_params, **params}
    params["group"] = trial.study.study_name
    trainer = make_trainer(
        params,
        callbacks=[
            optuna.integration.PyTorchLightningPruningCallback(
                trial, monitor="val/actor_loss"
            )
        ],
    )

    architecture = get_architecture_cls(params["architecture"])(**params)
    model = MotionPlanningImitation(architecture, **params)
    trainer.fit(model)

    # finish up
    if isinstance(trainer.logger, WandbLogger):
        trial.set_user_attr("wandb_id", trainer.logger.experiment.id)
    for logger in trainer.loggers:
        logger.finalize("finished")

    print(trainer.callback_metrics)
    return trainer.callback_metrics["val/actor_loss"].item()


if __name__ == "__main__":
    params = parse_args()
    study(params)
