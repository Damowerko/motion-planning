import argparse
import os
import sys
from typing import Union
import shutil

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from reconstrain.envs.motion_planning import MotionPlanning
from reconstrain.models import (
    MotionPlanningGPG,
    MotionPlanningImitation,
    MotionPlanningPolicy,
)
from reconstrain.utils import TensorboardHistogramLogger
from tqdm import tqdm
import torch


def make_trainer(params):
    logger = (
        TensorBoardLogger(
            save_dir=f"./{params.operation}/", name="tensorboard", version=""
        )
        if params.log
        else None
    )
    callbacks = [
        EarlyStopping(
            monitor="val/metric",
            patience=params.patience,
        ),
        (
            ModelCheckpoint(
                monitor="val/metric",
                dirpath=f"./{params.operation}/checkpoints/",
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
        check_val_every_n_epoch=10,
    )
    return trainer


def find_checkpoint(operation):
    candidates = [
        f"./{operation}/checkpoints/best.ckpt",
        f"./{operation}/checkpoints/last.ckpt",
    ]
    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt


def pretrain(params):
    trainer = make_trainer(params)
    model = MotionPlanningImitation(**vars(params))
    ckpt_path = "./pretrain/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    trainer.test(model)

    model.load_from_checkpoint(find_checkpoint("pretrain"))
    model.save_policy("./pretrain/policy.pt")


def train(params):
    trainer = make_trainer(params)

    if os.path.exists("./checkpoints"):
        os.makedirs("./pretrain")
        shutil.copytree("./checkpoints/pretrain/", "./pretrain/checkpoints/")
        shutil.copy("./policy.pt", "./pretrain/policy.pt")

    pretrain_checkpoint = find_checkpoint("pretrain")
    if pretrain_checkpoint is not None:
        print("Resuming from pretraining.")
        imitation = MotionPlanningImitation.load_from_checkpoint(pretrain_checkpoint)
        merged = {
            **vars(params),
            **{
                "F": imitation.F,
                "K": imitation.K,
                "n_layers": imitation.n_layers,
            },
        }
        model = MotionPlanningGPG(**merged)
        model.policy = imitation.policy
    else:
        print("Did not find a pretrain checkpoint.")
        model = MotionPlanningGPG(**vars(params))

    # check if checkpoint exists
    ckpt_path = "./train/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    trainer.test(model)


def _test(policy: Union[MotionPlanningPolicy, str], n_trials=10, render=False):
    env = MotionPlanning()
    rewards = []

    reference_policy = {
        "c": env.centralized_policy,
        "d0": lambda: env.decentralized_policy(0),
        "d1": lambda: env.decentralized_policy(1),
    }

    for _ in tqdm(range(n_trials)):
        done = False
        trial_rewards = []
        observation = env.reset()
        while not done:
            if render:
                env.render()
            if isinstance(policy, str):
                action = reference_policy[policy]()
            else:
                with torch.no_grad():
                    data = policy.to_graph_data(observation, env.adjacency())
                    action = (
                        policy.choose_action(
                            *policy(data.x, data.edge_index, data.edge_attr),
                            deterministic=True,
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
            observation, reward, done, _ = env.step(action)
            trial_rewards.append(reward)
        trial_reward = np.mean(trial_rewards)
        rewards.append(trial_reward)

    rewards = np.asarray(rewards)
    print(
        f"""
    MEAN: {rewards.mean():.2f}
    STD: {rewards.std():.2f}
    """
    )


def test(params):
    checkpoint_path = find_checkpoint("pretrain")
    if checkpoint_path is not None:
        model = MotionPlanningImitation.load_from_checkpoint(checkpoint_path)
    else:
        checkpoint_path = find_checkpoint("train")
        model = MotionPlanningGPG.load_from_checkpoint(checkpoint_path)
    _test(model, n_trials=params.n_trials, render=params.render)


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
    elif operation in ("test", "baseline"):
        group.add_argument("--render", type=int, default=0)
        group.add_argument("--n_trials", type=int, default=10)
        group.add_argument("--policy", type=str, default="c")

    params = parser.parse_args()

    if params.operation == "train":
        train(params)
    elif params.operation == "test":
        test(params)
    elif params.operation == "pretrain":
        pretrain(params)
    elif params.operation == "baseline":
        _test(params.policy, n_trials=params.n_trials, render=params.render)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")
