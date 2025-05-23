from logging import config
from pathlib import Path
from typing import Literal

import pandas as pd
import wandb

api = wandb.Api()

DEFAULT_DATA_PATH = Path("/nfs/general/motion_planning_data/test_results")

BASELINE_POLICIES = {"c_sq": "LSAP"} | {f"d{i}_sq": f"DHBA-{i}" for i in range(10)}


def load_baseline(
    policy: str, test_kind: Literal["basic", "scenarios"], data_path=DEFAULT_DATA_PATH
):
    filename = f"{policy}.parquet" if test_kind == "basic" else f"{test_kind}.parquet"
    df = pd.read_parquet(data_path / policy / filename).assign(
        policy=BASELINE_POLICIES[policy],
        id=BASELINE_POLICIES[policy],
    )
    return df


def load_baselines(data_path=DEFAULT_DATA_PATH):
    df = pd.concat(
        [
            load_baseline(policy, "basic", data_path).assign(
                policy=BASELINE_POLICIES[policy]
            )
            for policy in BASELINE_POLICIES
        ]
    )
    return df


def configs_from_tag(tag):
    runs = api.runs(
        path="damowerko-academic/motion-planning",
        filters={"tags": tag},
    )
    runs = api.runs(
        path="damowerko-academic/motion-planning",
        filters={"tags": tag},
    )
    configs = {run.id: run.config for run in runs}
    return configs


def load_test(
    id: str,
    test_kind: Literal["basic", "scalability", "scenarios", "delay"],
    data_path=DEFAULT_DATA_PATH,
):
    if test_kind == "basic":
        df = pd.read_parquet(data_path / id / f"{id}.parquet")
    elif test_kind == "scalability":
        df = pd.read_parquet(data_path / id / "scalability.parquet")
    elif test_kind == "scenarios":
        df = pd.read_parquet(data_path / id / "scenarios.parquet")
    elif test_kind == "delay":
        df = pd.read_parquet(data_path / id / "delay.parquet")
    return df


def load_delay(
    models_delay: dict[str, str],
    data_path=DEFAULT_DATA_PATH,
):
    return pd.concat(
        [
            aggregate_results(load_test(id, "delay", data_path)).assign(
                id=id, policy=models_delay[id]
            )
            for id in models_delay
        ],
        ignore_index=True,
    )


def load_comparison(
    baselines: list[str],
    models: dict[str, str],
    models_delay: dict[str, str],
    delay_s: float = 0.1,
):
    df_baseline = aggregate_results(load_baselines()).query(f"policy in {baselines}")
    if len(models) > 0:
        df_models = pd.concat(
            [
                aggregate_results(load_test(model, "basic")).assign(policy=name)
                for model, name in models.items()
            ]
        )
    if len(models_delay) > 0:
        df_delay = (
            load_delay(models_delay)
            .query(f"delay_s == {delay_s}")
            .assign(policy=lambda df: df["policy"])
        )

    dfs = [df_baseline]
    if len(models) > 0:
        dfs.append(df_models)
    if len(models_delay) > 0:
        dfs.append(df_delay)
    return pd.concat(dfs, ignore_index=True)


def df_from_tag(
    tag,
    test_kind: Literal["basic", "scalability", "scenarios"],
    data_path=DEFAULT_DATA_PATH,
):
    configs = configs_from_tag(tag)
    df = pd.concat(
        [load_test(id, test_kind, data_path).assign(id=id) for id in configs]
    )
    config_df = pd.DataFrame.from_dict(configs).transpose()[
        [
            "encoding_type",
            "encoding_frequencies",
            "attention_window",
            "connected_mask",
        ]
    ]
    # divide attention window by 2 to get the attention radius
    config_df["attention_window"] /= 2
    df = df.join(config_df, on="id")
    return df


def aggregate_results(df):
    gb_columns = list(
        set(df.columns).intersection(
            [
                "id",
                "time",
                "n_agents",
                "delay_s",
                "scenario",
            ]
        )
    )
    keep_columns = list(set(df.columns) - set(gb_columns) - {"coverage", "collisions"})
    return (
        df.groupby(gb_columns, sort=False)
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_se=("coverage", "sem"),
            collisions_mean=("collisions", "mean"),
            collisions_se=("collisions", "sem"),
            **{c: (c, "first") for c in keep_columns},
        )
        .reset_index()
        .assign(
            coverage_min=lambda df: df["coverage_mean"] - 1.96 * df["coverage_se"],
            coverage_max=lambda df: df["coverage_mean"] + 1.96 * df["coverage_se"],
            collisions_min=lambda df: df["collisions_mean"]
            - 1.96 * df["collisions_se"],
            collisions_max=lambda df: df["collisions_mean"]
            + 1.96 * df["collisions_se"],
        )
    )
