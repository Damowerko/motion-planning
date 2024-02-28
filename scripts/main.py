import argparse
import os
import sys
import typing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import imageio.v3 as iio
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
        metavar="OP",
        type=str,
        default="td3",
        choices=["imitation", "gpg", "td3", "test", "baseline"],
    )
    operation = sys.argv[1]

    parser.add_argument("--no_log", action="store_false", dest="log")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test", action="store_true")

    # trainer arguments
    group = parser.add_argument_group("Trainer")
    group.add_argument("--max_epochs", type=int, default=100)

    # operation specific arguments arguments
    group = parser.add_argument_group("Operation")
    if operation in ("imitation", "gpg", "td3"):
        get_model_cls(operation).add_model_specific_args(group)
    elif operation in ("test", "baseline"):
        # test specific args
        if operation == "test":
            group.add_argument("--checkpoint", type=str, required=True)
        # baseline specific args
        if operation == "baseline":
            group.add_argument("--policy", type=str, default="c")
        # common args
        group.add_argument("--render", action="store_true")
        group.add_argument("--n_trials", type=int, default=10)
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
        _test(
            params.policy,
            n_trials=params.n_trials,
            render=params.render,
            scenario=params.scenario,
        )
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
                dirpath=f"logs/{params.operation}/checkpoints/{run.id}/",
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


def _test(
    policy: Union[MotionPlanningActorCritic, str],
    max_steps=200,
    n_trials=10,
    render=False,
    n_agents=100,
    scenario="uniform",
):
    if isinstance(policy, str):
        env = MotionPlanning(n_agents=n_agents, scenario=scenario)
    else:
        env = policy.env

    reference_policy = {
        "c": env.centralized_policy,
        "d0": lambda: env.decentralized_policy(0),
        "d1": lambda: env.decentralized_policy(1),
    }
    if isinstance(policy, MotionPlanningActorCritic):
        policy.eval()

    rewards = []
    for trial_idx in tqdm(range(n_trials)):
        trial_rewards = []
        frames = []
        observation = env.reset()
        for _ in range(max_steps):
            if render:
                env.render(mode="human")
            else:
                frames.append(env.render(mode="rgb_array"))

            if isinstance(policy, str):
                action = reference_policy[policy]()
            else:
                model: MotionPlanningActorCritic = policy
                with torch.no_grad():
                    data = model.to_data(observation, env.adjacency())
                    action = (
                        model.actor.forward(data.state, data)[0].detach().cpu().numpy()
                    )

            observation, reward, done, _ = env.step(action)
            trial_rewards.append(reward)
            if done:
                break

        # save as gif
        if not render:
            iio.imwrite(f"figures/test_{trial_idx}.mp4", frames, fps=30)

        trial_reward = np.mean(trial_rewards)
        rewards.append(trial_reward)

    rewards = np.asarray(rewards)
    print(
        f"""
    MEAN: {rewards.mean():.2f}
    STD: {rewards.std():.2f}
    """
    )


def get_model_cls(model_str) -> typing.Type[MotionPlanningActorCritic]:
    if model_str == "imitation":
        return MotionPlanningImitation
    elif model_str == "gpg":
        return MotionPlanningGPG
    elif model_str == "td3":
        return MotionPlanningTD3
    raise ValueError(f"Invalid model {model_str}.")


def test(params):
    model = load_model(params.checkpoint)[0]
    _test(model, n_trials=params.n_trials, render=params.render)


if __name__ == "__main__":
    main()
