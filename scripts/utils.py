import os
import typing
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as pl
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from motion_planning.architecture import GNNActorCritic, TransformerActorCritic
from motion_planning.architecture.base import ActorCritic
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
            run = api.run(f"{user}/{project}/{run_id}")
            # set the name and model_str
            name = run_id
            params: dict = api.run(f"{user}/{project}/{run_id}").config
            model_str = params["operation"]
        else:
            name = os.path.basename(uri).split(".")[0]
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


def find_checkpoint(operation):
    candidates = [
        f"./{operation}/checkpoints/best.ckpt",
        f"./{operation}/checkpoints/last.ckpt",
    ]
    for ckpt in candidates:
        if os.path.exists(ckpt):
            return ckpt
