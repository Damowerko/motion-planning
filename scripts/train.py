import argparse
import os
import sys
from copy import deepcopy

import torch
from utils import get_architecture_cls, get_operation_cls, load_model, make_trainer

from motion_planning.lightning.imitation import MotionPlanningImitation


def main():
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

    # common args
    group = parser.add_argument_group("Simulation")
    group.add_argument("--n_trials", type=int, default=10)

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    get_operation_cls(operation).add_model_specific_args(group)

    group = parser.add_argument_group("Architecture")
    get_architecture_cls(architecture).add_model_specific_args(group)

    # training arguments
    training_group = parser.add_argument_group("Training")
    training_group.add_argument("--no_log", action="store_false", dest="log")
    training_group.add_argument("--test", action="store_true")
    training_group.add_argument("--max_epochs", type=int, default=100)
    training_group.add_argument("--patience", type=int, default=10)
    training_group.add_argument("--notes", type=str, default="")

    # reinforcement learning specific args
    if operation in ("ddpg", "td3", "ppo"):
        training_group.add_argument("--checkpoint", type=str)

    params = parser.parse_args()
    if params.operation in ("ddpg", "td3", "ppo"):
        reinforce(params)
    elif params.operation == "imitation":
        imitate(params)


def imitate(params):
    trainer = make_trainer(params)
    architecture = get_architecture_cls(params.architecture)(**vars(params))
    model = MotionPlanningImitation(architecture, **vars(params))

    ckpt_path = "./imitation/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


def reinforce(params):
    trainer = make_trainer(params)

    if params.checkpoint:
        params.pretrain = True
        print("Resuming from pretraining.")
        imitation, _ = load_model(params.checkpoint)
        model = get_operation_cls(params.operation)(**vars(params))
        model.model.actor = imitation.model.actor
        model.ac_target = deepcopy(model.model)
    else:
        params.pretrain = False
        print("Did not find a pretrain checkpoint.")
        model = get_operation_cls(params.operation)(**vars(params))

    # check if checkpoint exists
    ckpt_path = "./train/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


if __name__ == "__main__":
    main()
