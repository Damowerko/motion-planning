import argparse
import json
import os
import sys
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm
from wandb.wandb_run import Run

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.models import (
    MotionPlanningActorCritic,
    MotionPlanningGPG,
    MotionPlanningImitation,
    MotionPlanningTD3,
)


def main():
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()

    # program arguments
    parser.add_argument(
        "operation",
        type=str,
        default="td3",
        choices=["imitation", "gpg", "td3", "test", "baseline"],
        help="The operation to perform.",
    )
    operation = sys.argv[1]

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    if operation in ("imitation", "gpg", "td3"):
        get_model_cls(operation).add_model_specific_args(group)

        # training arguments
        training_group = parser.add_argument_group("Training")
        training_group.add_argument("--no_log", action="store_false", dest="log")
        training_group.add_argument("--test", action="store_true")
        training_group.add_argument("--max_epochs", type=int, default=100)
        training_group.add_argument("--patience", type=int, default=10)
    elif operation in ("test", "baseline"):
        # test specific args
        if operation == "test":
            group.add_argument("--checkpoint", type=str, required=True)
        # baseline specific args
        if operation == "baseline":
            group.add_argument(
                "--policy", type=str, default="c", choices=["c", "d0", "d1"]
            )
        # common args
        group.add_argument("--render", action="store_true")
        group.add_argument("--n_trials", type=int, default=10)
        group.add_argument("--n_agents", type=int, default=100)
        group.add_argument("--max_steps", type=int, default=200)
        group.add_argument(
            "--scenario",
            type=str,
            default="uniform",
            choices=["uniform", "gaussian_uniform"],
        )

    params = parser.parse_args()
    if params.operation in ("gpg", "td3"):
        train(params)
    elif params.operation == "test":
        test(params)
    elif params.operation == "imitation":
        imitation(params)
    elif params.operation == "baseline":
        baseline(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")


def load_model(uri: str) -> tuple[MotionPlanningActorCritic, str]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
        cls: The class of the model to load.
    """
    with TemporaryDirectory() as tmpdir:
        if uri.startswith("wandb://"):
            import wandb

            user, project, run_id = uri[len("wandb://") :].split("/")

            # Download the model from wandb to temporary directory
            api = wandb.Api()
            artifact = api.artifact(
                f"{user}/{project}/model-{run_id}:best", type="model"
            )
            artifact.download(root=tmpdir)
            uri = f"{tmpdir}/model.ckpt"
            # set the name and model_str
            name = run_id
            model_str = api.run(f"{user}/{project}/{run_id}").config["operation"]
        else:
            name = os.path.basename(uri).split(".")[0]
            if "imitation" in uri:
                model_str = "imitation"
            elif "gpg" in uri:
                model_str = "gpg"
            elif "td3" in uri:
                model_str = "td3"
            else:
                raise ValueError(f"Invalid model uri {uri}.")

        cls = get_model_cls(model_str)
        model = cls.load_from_checkpoint(uri)
        return model, name


def make_trainer(params):
    logger = False
    callbacks: list[pl.Callback] = [
        EarlyStopping(
            monitor="val/reward",
            mode="max",
            patience=params.patience,
        ),
    ]

    if params.log:
        logger = WandbLogger(
            project="motion-planning", save_dir="logs", config=params, log_model=True
        )
        logger.log_hyperparams(params)
        run = typing.cast(Run, logger.experiment)
        run.log_code(
            Path(__file__).parent.parent,
            include_fn=lambda path: (
                path.endswith(".py")
                and "logs" not in path
                and ("src" in path or "scripts" in path)
            ),
        )
        callbacks += [
            ModelCheckpoint(
                monitor="val/reward",
                mode="max",
                dirpath=f"logs/{params.operation}/{run.id}/",
                filename="best",
                auto_insert_metric_name=False,
                save_last=True,
                save_top_k=1,
            )
        ]

    print("starting training")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=params.log,
        precision=32,
        max_epochs=params.max_epochs,
        default_root_dir="logs/",
        check_val_every_n_epoch=1,
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


def imitation(params):
    trainer = make_trainer(params)
    model = MotionPlanningImitation(**vars(params))
    ckpt_path = "./imitation/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


def train(params):
    trainer = make_trainer(params)
    imitation_checkpoint = find_checkpoint("imitation")
    if imitation_checkpoint is not None:
        print("Resuming from pretraining.")
        imitation = MotionPlanningImitation.load_from_checkpoint(imitation_checkpoint)
        merged = {
            **vars(params),
            **{
                "F": imitation.F,
                "K": imitation.K,
                "n_layers": imitation.n_layers,
            },
        }
        model = get_model_cls(params.operation)(**merged)
        model.actor = imitation.actor
    else:
        print("Did not find a pretrain checkpoint.")
        model = get_model_cls(params.operation)(**vars(params))

    # check if checkpoint exists
    ckpt_path = "./train/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


def rollout(
    env: MotionPlanning, policy_fn: typing.Callable, params: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform rollouts in the environment using a given policy.

    Args:
        env (MotionPlanning): The environment to perform rollouts in.
        policy_fn (typing.Callable): The policy function to use for selecting actions.
        params (argparse.Namespace): Additional parameters for the rollouts.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the rewards and frames for each rollout.
        - rewards (np.ndarray): An ndarray of shape (n_trials, max_steps) containing the rewards for each step in each rollout.
        - frames (np.ndarray): An array of shape (n_trial, max_steps, H, W) where H and W are the heights and widths of the rendered frames.
    """
    rewards = []
    frames = []
    for _ in tqdm(range(params.n_trials)):
        rewards_trial = []
        frames_trial = []
        observation = env.reset()
        for _ in range(params.max_steps):
            action = policy_fn(observation, env.adjacency())
            observation, reward, _, _ = env.step(action)
            rewards_trial.append(reward)
            frames_trial.append(env.render(mode="rgb_array"))
        rewards.append(rewards_trial)
        frames.append(frames_trial)
    return np.asarray(rewards), np.asarray(frames)


def baseline(params):
    env = MotionPlanning(n_agents=params.n_agents, scenario=params.scenario)
    if params.policy == "c":
        policy_fn = lambda o, g: env.centralized_policy()
    elif params.policy == "d0":
        policy_fn = lambda o, g: env.decentralized_policy(0)
    elif params.policy == "d1":
        policy_fn = lambda o, g: env.decentralized_policy(1)
    else:
        raise ValueError(f"Invalid policy {params.policy}.")

    rewards, frames = rollout(env, policy_fn, params)
    save_results(
        params.policy, Path("figures") / "test_results" / params.policy, rewards, frames
    )


def test(params):
    env = MotionPlanning(n_agents=params.n_agents, scenario=params.scenario)
    model, name = load_model(params.checkpoint)
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, graph):
        data = model.to_data(observation, graph)
        return model.actor.forward(data.state, data)[0].detach().cpu().numpy()

    rewards, frames = rollout(env, policy_fn, params)
    save_results(name, Path("figures") / "test_results" / name, rewards, frames)


def save_results(name: str, path: Path, rewards: np.ndarray, frames: np.ndarray):
    """
    Args:
        path (Path): The path to save the summary to.
        rewards (np.ndarray): An ndarray of shape (n_trials, max_steps).
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
    """
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "rewards.npy", rewards)

    # make a single plot of all reward functions
    plt.figure()
    plt.plot(rewards.T)
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title(f"{name}")
    plt.savefig(path / f"rewards_{name}.png")

    # make a single video of all trials
    iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=30)

    # summary metrics
    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
    }
    json.dump(metrics, open(path / "metrics.json", "w"))

    print(metrics)


def get_model_cls(model_str) -> typing.Type[MotionPlanningActorCritic]:
    if model_str == "imitation":
        return MotionPlanningImitation
    elif model_str == "gpg":
        return MotionPlanningGPG
    elif model_str == "td3":
        return MotionPlanningTD3
    raise ValueError(f"Invalid model {model_str}.")


if __name__ == "__main__":
    main()
