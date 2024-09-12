import argparse
import json
import os
import sys
import typing
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from scipy.spatial.distance import cdist
from tqdm import tqdm
from wandb.wandb_run import Run

from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.models import (
    MotionPlanningActorCritic,
    MotionPlanningDDPG,
    MotionPlanningImitation,
    MotionPlanningPPO,
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
        choices=[
            "imitation",
            "ddpg",
            "td3",
            "ppo",
            "test",
            "baseline",
            "transfer-agents",
            "transfer-area",
            "transfer-density",
            "test-q",
        ],
        help="The operation to perform.",
    )
    operation = sys.argv[1]

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    if operation in ("imitation", "ddpg", "td3", "ppo"):
        get_model_cls(operation).add_model_specific_args(group)

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
        "transfer-agents",
        "transfer-area",
        "transfer-density",
        "test-q",
    ):
        # test specific args
        if operation in (
            "test",
            "transfer-agents",
            "transfer-area",
            "transfer-density",
            "test-q",
        ):
            group.add_argument("--checkpoint", type=str, required=True)
        # baseline specific args
        if operation == "baseline":
            group.add_argument(
                "--policy", type=str, default="c", choices=["c", "d0", "d1", "capt"]
            )
        # transfer specific args
        if operation in (
            "transfer-area",
            "transfer-density",
        ):  # keep the area or density constant
            group.add_argument("--n_agents", nargs="+", type=int, default=100)
        else:
            group.add_argument("--n_agents", type=int, default=100)

        if operation == "transfer-agents":  # keep the number of agents constant
            group.add_argument("--width", nargs="+", type=float, default=10.0)
        else:
            group.add_argument("--width", type=float, default=10.0)
        # common args
        group.add_argument("--render", action="store_true")
        group.add_argument("--n_trials", type=int, default=10)
        group.add_argument("--max_steps", type=int, default=200)
        group.add_argument(
            "--scenario",
            type=str,
            default="uniform",
            choices=["uniform", "gaussian_uniform"],
        )
        group.add_argument(
            "--agent_radius",
            type=float,
            default=0.1,
            help="The radius of the agents. Used to measure collisions",
        )
        group.add_argument(
            "--agent_margin",
            type=float,
            default=0.0,
            help="Additional margin to consider when initializing for collision avoidance.",
        )
        group.add_argument(
            "--collision_coefficient",
            type=float,
            default=5.0,
            help="Scalling cofficient for the reward penalty for collisions.",
        )

    params = parser.parse_args()
    if params.operation in ("ddpg", "td3", "ppo"):
        train(params)
    elif params.operation == "test":
        test(params)
    elif params.operation == "test-q":
        test_q(params)
    elif params.operation in ("transfer-agents", "transfer-area", "transfer-density"):
        transfer(params)
    elif params.operation == "imitation":
        imitation(params)
    elif params.operation == "baseline":
        baseline(params)
    else:
        raise ValueError(f"Invalid operation {params.operation}.")


def load_model(uri: str, best: bool = True) -> tuple[MotionPlanningActorCritic, str]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
        cls: The class of the model to load.
    """
    with TemporaryDirectory() as tmpdir:
        if uri.startswith("wandb://"):
            import wandb

            user, project, run_id = uri[len("wandb://") :].split("/")
            suffix = "best" if best else "latest"

            # Download the model from wandb to temporary directory
            api = wandb.Api()
            artifact = api.artifact(
                f"{user}/{project}/model-{run_id}:{suffix}", type="model"
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
            project="motion-planning",
            save_dir="logs",
            config=params,
            log_model=True,
            notes=params.notes,
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

    if params.checkpoint:
        params.pretrain = True
        print("Resuming from pretraining.")
        imitation, _ = load_model(params.checkpoint)
        model = get_model_cls(params.operation)(**vars(params))
        model.ac.actor = imitation.ac.actor
        model.ac_target = deepcopy(model.ac)
    else:
        params.pretrain = False
        print("Did not find a pretrain checkpoint.")
        model = get_model_cls(params.operation)(**vars(params))

    # check if checkpoint exists
    ckpt_path = "./train/checkpoints/last.ckpt"
    ckpt_path = ckpt_path if os.path.exists(ckpt_path) else None
    trainer.fit(model, ckpt_path=ckpt_path)
    if params.test:
        trainer.test(model)


def rollout(
    env: MotionPlanning,
    policy_fn: typing.Callable,
    params: argparse.Namespace,
    baseline: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Perform rollouts in the environment using a given policy.

    Args:
        env (MotionPlanning): The environment to perform rollouts in.
        policy_fn (typing.Callable): The policy function to use for selecting actions.
        params (argparse.Namespace): Additional parameters for the rollouts.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the rewards and frames for each rollout.
        - rewards (pd.DataFrame): A Pandas dataframe with columns=['trial', 'step', 'reward', 'coverage', 'collisions', 'near_collisions'].
        - frames (np.ndarray): An array of shape (n_trial, max_steps, H, W) where H and W are the heights and widths of the rendered frames.
    """
    data = []
    frames = []
    for trial in tqdm(range(params.n_trials)):
        frames_trial = []
        observation, centralized_state = env.reset()
        curr_collision = np.zeros(env.n_agents)
        for step in range(params.max_steps):
            action = (
                policy_fn(observation, centralized_state, step + 1, env.adjacency())
                if not baseline
                else policy_fn(observation, env.adjacency())
            )
            observation, _, reward, _, _ = env.step(action)
            data.append(
                dict(
                    trial=trial,
                    step=step,
                    reward=reward.mean(),
                    coverage=env.coverage(),
                    collisions=env.n_collisions(r=params.agent_radius),
                    near_collisions=env.n_collisions(
                        r=params.agent_radius + params.agent_margin
                    ),
                )
            )
            frames_trial.append(env.render(mode="rgb_array"))
        frames.append(frames_trial)
    return pd.DataFrame(data), np.asarray(frames)


def baseline(params):
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        scenario=params.scenario,
        agent_radius=params.agent_radius + params.agent_margin,
        collision_coefficient=params.collision_coefficient,
    )
    if params.policy == "c":
        policy_fn = lambda o, g: env.centralized_policy()
    elif params.policy == "d0":
        policy_fn = lambda o, g: env.decentralized_policy(0)
    elif params.policy == "d1":
        policy_fn = lambda o, g: env.decentralized_policy(1)
    elif params.policy == "capt":
        policy_fn = lambda o, g: env.capt_policy()
    else:
        raise ValueError(f"Invalid policy {params.policy}.")

    data, frames = rollout(env, policy_fn, params, baseline=True)
    save_results(
        params.policy, Path("figures") / "test_results" / params.policy, data, frames
    )


def test(params):
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        scenario=params.scenario,
        agent_radius=params.agent_radius + params.agent_margin,
        collision_coefficient=params.collision_coefficient,
    )
    model, name = load_model(params.checkpoint)
    model = model.eval()

    @torch.no_grad()
    def policy_fn(observation, centralized_state, step, graph):
        data = model.to_data(observation, centralized_state, step, graph)
        return model.ac.actor.forward(data.state, data)[0].detach().cpu().numpy()

    data, frames = rollout(env, policy_fn, params)
    save_results(name, Path("figures") / "test_results" / name, data, frames)


def test_q(params):
    env = MotionPlanning(
        n_agents=params.n_agents,
        width=params.width,
        scenario="q-scenario",
        agent_radius=params.agent_radius,
        collision_coefficient=params.collision_coefficient,
    )
    model, name = load_model(params.checkpoint)
    model = model.eval()

    null_action = torch.zeros(env.n_agents, 2).to(device="cuda")

    observation, centralized_state = env.reset()
    data = model.to_data(observation, centralized_state, env.adjacency())
    print(
        model.ac.critic.forward(data.state, null_action, data)
        .mean()
        .detach()
        .cpu()
        .numpy()
    )


def transfer(params):
    model, name = load_model(params.checkpoint)
    model = model.eval()
    iv_type = params.width if params.operation == "transfer-agents" else params.n_agents
    if params.operation == "transfer-area":
        code = "r"
    elif params.operation == "transfer-agents":
        code = "a"
    else:
        code = "d"

    for iv_value in iv_type:
        if params.operation == "transfer-area":
            env = MotionPlanning(
                n_agents=iv_value,
                width=params.width,
                scenario=params.scenario,
                agent_radius=params.agent_radius,
                collision_coefficient=params.collision_coefficient,
            )
        elif params.operation == "transfer-agents":
            env = MotionPlanning(
                n_agents=params.n_agents,
                width=iv_value,
                scenario=params.scenario,
                agent_radius=params.agent_radius,
                collision_coefficient=params.collision_coefficient,
            )
        else:
            env = MotionPlanning(
                n_agents=iv_value,
                width=1.0 * np.sqrt(iv_value),
                scenario=params.scenario,
                agent_radius=params.agent_radius,
                collision_coefficient=params.collision_coefficient,
            )

        @torch.no_grad()
        def policy_fn(observation, centralized_state, step, graph):
            data = model.to_data(observation, centralized_state, step, graph)
            return model.ac.actor.forward(data.state, data)[0].detach().cpu().numpy()

        data, frames = rollout(env, policy_fn, params)
        filename = name + f"-{iv_value}-{code}"
        save_results(name, Path("figures") / "test_results" / filename, data, frames)


def save_results(name: str, path: Path, data: pd.DataFrame, frames: np.ndarray):
    """
    Args:
        path (Path): The path to save the summary to.
        data (pd.DataFrame): Dataframe containing numerical info of performance at each step.
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
    """
    path.mkdir(parents=True, exist_ok=True)

    data.to_parquet(path / "data.parquet")

    print(data)

    # make a single plot of basic metrics
    metric_names = ["reward", "coverage", "collisions", "near_collisions"]
    for metric_name in metric_names:
        sns.relplot(data=data, x="step", y=metric_name, hue="trial", kind="line")
        plt.xlabel("Step")
        plt.ylabel(f"{metric_name.replace('_', ' ').capitalize()}")
        plt.savefig(path / f"{metric_name}_{name}.png")

    # summary metrics
    metrics = {
        "Reward Mean": data["reward"].mean(),
        "Reward Std": data["reward"].std(),
        "Coverage Mean": data["coverage"].mean(),
        "Coverage Std": data["coverage"].std(),
        # Sum over step but mean over trials
        "Collisions Mean": data.groupby("trial")["collisions"].sum().mean(),
        "Near Collisions Mean": data.groupby("trial")["near_collisions"].sum().mean(),
    }
    with open(path / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # make a single video of all trials
    iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=30)


def get_model_cls(model_str) -> typing.Type[MotionPlanningActorCritic]:
    if model_str == "imitation":
        return MotionPlanningImitation
    elif model_str == "ddpg":
        return MotionPlanningDDPG
    elif model_str == "td3":
        return MotionPlanningTD3
    elif model_str == "ppo":
        return MotionPlanningPPO
    raise ValueError(f"Invalid model {model_str}.")


if __name__ == "__main__":
    main()
