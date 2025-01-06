import argparse
import itertools
import json
import os
import sys
import typing
from copy import deepcopy
from pathlib import Path

import imageio.v3 as iio
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from scipy.spatial.distance import cdist
from tqdm import tqdm
from wandb.wandb_run import Run

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.lightning import MotionPlanningImitation
from scripts.utils import get_operation_cls, load_model

from .utils import make_trainer


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument(
        "operation",
        type=str,
        default="td3",
        choices=[
            "imitation",
            "ddpg",
            "td3",
            "ppo",
            "test",
            "baseline",
        ],
        help="The operation to perform.",
    )
    operation = sys.argv[1]

    # common args
    group = parser.add_argument_group("Simulation")
    group.add_argument("--n_trials", type=int, default=10)

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    if operation in ("imitation", "ddpg", "td3", "ppo"):
        get_operation_cls(operation).add_model_specific_args(group)

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
    elif operation in (
        "test",
        "baseline",
    ):
        # test specific args
        if operation in ("test",):
            group.add_argument("--checkpoint", type=str, required=True)
        # baseline specific args
        if operation == "baseline":
            group.add_argument(
                "--policy", type=str, default="c", choices=["c", "d0", "d1", "capt"]
            )
        # common args
        group.add_argument("--n_agents", type=int, default=100)
        group.add_argument("--width", type=float, default=10.0)
        group.add_argument("--render", action="store_true")
        group.add_argument("--max_steps", type=int, default=200)
        group.add_argument(
            "--scenario",
            type=str,
            default="uniform",
            choices=["uniform", "gaussian_uniform"],
        )
        group.add_argument("--agent_radius", type=float, default=0.05)
        group.add_argument("--agent_margin", type=float, default=0.05)
        group.add_argument("--collision_coefficient", type=float, default=5.0)
        group.add_argument("--output_images", action="store_true")

    params = parser.parse_args()
    if params.operation in ("ddpg", "td3", "ppo"):
        train(params)
    elif params.operation == "test":
        test(params)
    elif params.operation == "imitation":
        imitation(params)
    elif params.operation == "baseline":
        baseline(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")


if __name__ == "__main__":
    main()
