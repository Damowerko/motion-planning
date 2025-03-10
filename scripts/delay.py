import argparse
import logging
from itertools import product
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate import delay
from motion_planning.utils import load_model

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()

    # common args
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_workers", type=int)
    params = vars(parser.parse_args())

    logger.info(f"Loading model from {params['checkpoint']}")
    model, name = load_model(params["checkpoint"])
    policy = model.model.get_policy_operator().eval()

    df_list = []
    for comm_interval in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        logger.info(f"Testing delay of {comm_interval} seconds.")
        env_params = MotionPlanningEnvParams()
        df, _ = delay(
            env_params=env_params,
            policy=policy,
            max_steps=params["max_steps"],
            num_episodes=params["n_trials"],
            num_workers=params["n_workers"],
            comm_interval=comm_interval,
        )
        df_list.append(df)

    logger.info("Saving results")
    path = Path("data") / "test_results" / name
    path.mkdir(parents=True, exist_ok=True)
    pd.concat(df_list).to_parquet(path / "delay.parquet")

    frames_list = []
    env_params = MotionPlanningEnvParams()
    logger.info(f"Making video with delay 0.1")
    df, frames = delay(
        env_params=env_params,
        policy=policy,
        max_steps=params["max_steps"],
        num_episodes=10,
        comm_interval=0.1,
        render=True,
    )
    frames_list.append(frames)
    logger.info("Saving video")
    frames = np.concatenate(frames_list)
    iio.imwrite(path / f"{name}_delay_100ms.mp4", frames, fps=40)


if __name__ == "__main__":
    main()
