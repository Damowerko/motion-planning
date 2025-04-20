import argparse
import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from motion_planning.utils import compute_width

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate import evaluate_policy
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
    parser.add_argument("--n_agents", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="")
    params = vars(parser.parse_args())

    width = compute_width(params["n_agents"], 100 / 1000**2)

    logger.info(f"Loading model from {params['checkpoint']}")
    model, name = load_model(params["checkpoint"])
    policy = model.model.get_policy_operator().eval()

    logger.info("Making video from training distribution")
    env_params = MotionPlanningEnvParams(n_agents=params["n_agents"], width=width)
    logger.info("Running simulation")
    _, frames = evaluate_policy(
        env_params=env_params,
        policy=policy,
        max_steps=params["max_steps"],
        num_episodes=params["n_trials"],
        num_workers=params["n_workers"],
        render=True,
    )
    assert frames is not None

    for scenario in ["circle", "two_lines", "gaussian_uniform", "icra"]:
        if scenario == "icra" and params["n_agents"] != 100:
            continue
        logger.info(f"Making video for scenario: {scenario}")
        env_params = MotionPlanningEnvParams(
            scenario=scenario, n_agents=params["n_agents"], width=width
        )
        _, _frames = evaluate_policy(
            env_params=env_params,
            policy=policy,
            max_steps=2 * params["max_steps"],
            num_episodes=3,
            num_workers=3,
            render=True,
        )
        assert _frames is not None
        frames = np.concatenate([frames, _frames])
    logger.info("Saving video")
    path = Path("data") / "test_results" / name
    path.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path / f"{name}{params['suffix']}.mp4", frames, fps=40)


if __name__ == "__main__":
    main()
