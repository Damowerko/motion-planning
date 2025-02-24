import argparse
import logging
import os
import sys

import torch

from motion_planning.lightning.imitation import MotionPlanningImitation
from motion_planning.utils import (
    get_architecture_cls,
    get_operation_cls,
    load_model,
    make_trainer,
)

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        type=str,
        default="imitation",
        choices=["imitation", "ddpg", "td3"],
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

    # common args
    group = parser.add_argument_group("Simulation")
    group.add_argument("--n_trials", type=int, default=10)

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    get_operation_cls(operation).add_model_specific_args(group)

    group = parser.add_argument_group("Architecture")
    get_architecture_cls(architecture).add_model_specific_args(group)

    # training arguments
    group = parser.add_argument_group("Training")
    group.add_argument("--no_log", action="store_false", dest="log")
    group.add_argument("--test", action="store_true")
    group.add_argument("--max_epochs", type=int, default=100)
    group.add_argument("--patience", type=int, default=10)
    group.add_argument("--notes", type=str, default="")
    group.add_argument("--group", type=str, default=None)
    group.add_argument("--tag", type=str, dest="tags", action="append")
    group.add_argument("--simple_progress", action="store_true")

    # reinforcement learning specific args
    if operation in ("ddpg", "td3"):
        group.add_argument("--checkpoint", type=str)

    params = vars(parser.parse_args())
    if operation in ("ddpg", "td3"):
        reinforce(params)
    elif operation == "imitation":
        imitate(params)


def imitate(params: dict):
    trainer = make_trainer(params)
    architecture = get_architecture_cls(params["architecture"])(**params)
    model = MotionPlanningImitation(architecture, **params)

    ckpt_path = "./imitation/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params["test"]:
        trainer.test(model)


def reinforce(params: dict):
    trainer = make_trainer(params)

    if params["checkpoint"]:
        params["pretrain"] = True
        logger.info("Resuming from pretraining.")
        lightning_module, _ = load_model(params["checkpoint"])
        model = lightning_module.model
    else:
        params["pretrain"] = False
        logger.info("Training from scratch. Pretrained checkpoint was not provided.")
        model = get_architecture_cls(params["architecture"])(**params)

    model = get_operation_cls(params["operation"])(model, **params)

    # check if checkpoint exists
    ckpt_path = "./train/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params["test"]:
        trainer.test(model)


if __name__ == "__main__":
    main()
