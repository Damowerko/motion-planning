# motion-planning
Constrained reniforcement learning for motion planning.


## Installation
1. Install (poetry)[https://python-poetry.org/docs/#installing-with-the-official-installer]. You can use the official installer or the `pipx` installer, whichever you prefer. 
2. Poetry will create a new virtualenv for this project automatically. If you manage your virtualenvs yourself or with `pyenv` like me create a new virtualenv for this project and set `poetry config virtualenvs.create false --local`.
3. `poetry install` and all pip packages will install.

## Training
To training a model we can use the `train.py` script. It has the following usage. 
```
python scripts/train.py <operation> <architecture> [options]
```
The operation can be either `imitaiton` or `td3`. The architecture can be `transformer` or `gnn`. There are many options that can be passed to the script. They can be listed by passing `--help` to the script. 
```
python scripts/train.py imitation transformer --help
```
For example we can change the batch size and the learning rate of the actor.
```
python scripts/train.py imitation transformer --batch_size 32 --actor_lr 1e-6
```

## Testing a Checkpoint
```
python scripts/test.py <operation> <architecture> --checkpoint <uri>
```
As before the operation can be either `imitaiton` or `td3`. The architecture can be `transformer` or `gnn`. The uri can be either a path to a local file or a uri for wandb. In the latter case the uri should be in the form of `wandb://<project>/<run>/<checkpoint>`.
```
python scripts/test.py imitation transformer --checkpoint wandb://damowerko/motion-planning/a3qcx5i8
```
The results of the testing will be saved into `data/test_results/<checkpoint_name>`. This will include a video of the policy evaluation, several plots, summary metrics, and a parquet file containing detailed metrics.

### Common flags
- `--name` To change the folder/filename of where the results are saved.
- `--n_agents` To change the number of agents in the environment. Environment area is scaled proportionally, by default.
- `--density` Alter the number of agents per unit area. By default the density is 1.0.

## Testing Checkpoint Generalizatioin
Use `scripts/scalability.py` to test a checkpoint while varying the environment parameters such as the number of agents, their density, or radius. The script has the following usage.
```
python scripts/scalability.py --checkpoint <uri>
```