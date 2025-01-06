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