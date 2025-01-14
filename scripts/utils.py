import os
import typing
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from motion_planning.architecture import GNNActorCritic, TransformerActorCritic
from motion_planning.architecture.base import ActorCritic
from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.lightning import (
    MotionPlanningActorCritic,
    MotionPlanningImitation,
    MotionPlanningTD3,
)


def get_architecture_cls(model_str) -> typing.Type[ActorCritic]:
    if model_str == "transformer":
        return TransformerActorCritic
    elif model_str == "gnn":
        return GNNActorCritic
    raise ValueError(f"Invalid model {model_str}.")


def get_operation_cls(operation_str) -> typing.Type[MotionPlanningActorCritic]:
    if operation_str == "imitation":
        return MotionPlanningImitation
    elif operation_str == "td3":
        return MotionPlanningTD3
    raise ValueError(f"Invalid operation {operation_str}.")


def make_trainer(
    params: dict,
    callbacks: list[pl.Callback] = [],
    wandb_kwargs: dict = {},
) -> pl.Trainer:
    logger = False
    if params["log"]:
        logger = WandbLogger(
            project="motion-planning",
            save_dir="logs",
            config=params,
            log_model=True,
            notes=params.get("notes", None),
            **wandb_kwargs,
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

        if params["operation"] == "imitation":
            monitor = "val/actor_loss"
            mode = "min"
        else:
            monitor = "val/reward"
            mode = "max"

        callbacks += [
            EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=params["patience"],
            ),
            ModelCheckpoint(
                monitor=monitor,
                mode=mode,
                dirpath=f"logs/{params['operation']}/{run.id}/",
                filename="best",
                auto_insert_metric_name=False,
                save_last=True,
                save_top_k=1,
            ),
        ]

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        devices=1,
        enable_checkpointing=params["log"],
        precision=32,
        max_epochs=params["max_epochs"],
        default_root_dir="logs/",
        check_val_every_n_epoch=1,
    )
    return trainer


def load_model_name(uri: str) -> str:
    if uri.startswith("wandb://"):
        user, project, run_id = uri[len("wandb://") :].split("/")
        return run_id
    return os.path.basename(uri).split(".")[0]


def load_model(uri: str, best: bool = True) -> tuple[MotionPlanningActorCritic, str]:
    """Load a model from a uri.

    Args:
        uri (str): The uri of the model to load. By default this is a path to a file. If you want to use a wandb model, use the format wandb://<user>/<project>/<run_id>.
        cls: The class of the model to load.

    Returns:
        model (MotionPlanningActorCritic): The loaded model.
        name (str): The name of the model.
    """
    name = load_model_name(uri)
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
            params: dict = api.run(f"{user}/{project}/{run_id}").config
            model_str = params["operation"]
        else:
            with open(Path(uri).with_suffix("yaml")) as f:
                params = yaml.safe_load(f)

        try:
            # New checkpoints should include the architecture in the state_dict
            model = get_operation_cls(model_str).load_from_checkpoint(uri)
        except TypeError:
            # If the model was saved without the architecture, we need to load it manually
            architecture = get_architecture_cls(params["architecture"])(**params)
            model = get_operation_cls(model_str).load_from_checkpoint(
                uri, model=architecture
            )
        return model, name


def rollout(
    env: MotionPlanning,
    policy_fn: typing.Callable,
    params: dict,
    baseline: bool = False,
    pbar: bool = True,
    frames: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Perform rollouts in the environment using a given policy.

    Args:
        env (MotionPlanning): The environment to perform rollouts in.
        policy_fn (typing.Callable): The policy function to use for selecting actions.
        params (dict): Additional parameters for the rollouts.
        frames (bool): Whether to return the frames of the rollouts.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the rewards and frames for each rollout.
        - rewards (pd.DataFrame): A Pandas dataframe with columns=['trial', 'step', 'reward', 'coverage', 'collisions', 'near_collisions'].
        - frames (np.ndarray): An array of shape (n_trial, max_steps, H, W) where H and W are the heights and widths of the rendered frames.
    """
    data = []
    frames_all = []
    for trial in tqdm(range(params["n_trials"])) if pbar else range(params["n_trials"]):
        frames_trial = []
        observation, positions, targets = env.reset()
        for step in range(params["max_steps"]):
            action = (
                policy_fn(observation, positions, targets, env.adjacency())
                if not baseline
                else policy_fn(observation, env.adjacency())
            )
            observation, _, _, reward, _, _ = env.step(action)
            data.append(
                dict(
                    trial=trial,
                    step=step,
                    reward=reward.mean(),
                    coverage=env.coverage(),
                    collisions=env.n_collisions(r=params["agent_radius"]),
                    near_collisions=env.n_collisions(
                        r=params["agent_radius"] + params["agent_margin"]
                    ),
                )
            )
            if frames:
                frames_trial.append(env.render(mode="rgb_array"))
        if frames:
            frames_all.append(frames_trial)
    frames_array = np.asarray(frames_all) if frames else None
    return pd.DataFrame(data), frames_array


def find_checkpoint(operation):
    candidates = [
        f"./{operation}/checkpoints/best.ckpt",
        f"./{operation}/checkpoints/last.ckpt",
    ]
    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt
