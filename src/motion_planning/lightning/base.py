import logging
from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from tensordict import TensorDictBase
from torch_geometric.data import Batch, Data
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torchcps.utils import add_model_specific_args
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import ParallelEnv
from torchrl.modules import ActorCriticWrapper

from motion_planning.envs import MotionPlanningEnv
from motion_planning.rl import ExperienceSourceDataset

logger = logging.getLogger(__name__)


class MotionPlanningActorCritic(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCriticWrapper,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0001,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        gamma: float = 0.95,
        max_steps: int = 200,
        buffer_size: int = 20_000,
        # MotionPlanning environment parameters
        n_agents: int = 100,
        max_vel: float = 5.0,
        width: float = 1000.0,
        scenario: str = "uniform",
        collision_distance: float = 2.5,
        initial_separation: float = 5.0,
        collision_coefficient: float = 5.0,
        reward_sigma: float = 10.0,
        coverage_reward: str = "dist_sq",
        num_workers: int = 32,
        # Noise parameters
        noise: float = 0.1,
        noise_clip: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "kwargs"])
        self.model = model
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_steps = max_steps
        self.buffer_size = buffer_size
        # MotionPlanning environment parameters
        self.n_agents = n_agents
        self.max_vel = max_vel
        self.width = width
        self.scenario = scenario
        self.collision_distance = collision_distance
        self.initial_separation = initial_separation
        self.collision_coefficient = collision_coefficient
        self.reward_sigma = reward_sigma
        self.coverage_reward = coverage_reward
        self.num_workers = num_workers
        self.expert_policy = None
        # Noise parameters
        self.noise = noise
        self.noise_clip = noise_clip

    def setup(self, stage: str):
        self._init_torchrl()
        self.populate()

    def _init_torchrl(self):
        self.model.to(self.device)
        make_env = partial(
            MotionPlanningEnv,
            n_agents=self.n_agents,
            width=self.width,
            scenario=self.scenario,
            initial_separation=self.initial_separation,
            max_vel=self.max_vel,
            collision_distance=self.collision_distance,
            collision_coefficient=self.collision_coefficient,
            reward_sigma=self.reward_sigma,
            coverage_reward=self.coverage_reward,
            expert_policy=self.expert_policy,
        )
        self.env = ParallelEnv(self.num_workers, make_env, device=self.device)
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.buffer_size, device=self.device),
            batch_size=self.batch_size,
            priority_key="priority",
        )
        self.collector = SyncDataCollector(
            self.env,
            self.rollout_action,
            frames_per_batch=self.num_workers,
            max_frames_per_traj=self.max_steps,
            postproc=lambda td: td.reshape(-1),
            device=self.device,
            trust_policy=True,
        )

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.model.load_state_dict(state_dict, *args, **kwargs)

    def policy(self, mu: torch.Tensor):
        """
        Sample policy from a normal distribution centered at mu and with standard devation `self.noise`.
        Noise will be clipped to `self.noise_clip` and the action will be clipped to the unit ball.
        """
        eps = torch.randn_like(mu) * self.noise
        action = mu + torch.clip(eps, -self.noise_clip, self.noise_clip)
        action = self.clip_action(action)
        return action

    def critic_loss(
        self,
        q: torch.Tensor,
        next_q: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = done.size(0)
        N = q.size(0) // batch_size
        done = done[:, None].expand(batch_size, N).reshape(-1)
        return F.mse_loss(q, reward + torch.logical_not(done) * self.gamma * next_q)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """
        Clip action to [-1,1].
        """
        return action.clip(-1, 1)

    def optimizers(self):
        opts = super().optimizers()
        if not isinstance(opts, list):
            raise ValueError(
                "Expected a list of optimizers: an actor and multiple critics. Double check that `configure_optimizers` returns multiple optimziers."
            )
        return opts

    def configure_optimizers(self):
        return [
            torch.optim.AdamW(
                self.model.get_policy_operator().parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.model.get_value_operator().parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def to_data(
        self,
        state: torch.Tensor,
        positions: torch.Tensor,
        targets: torch.Tensor,
        edge_index: torch.Tensor,
        components: torch.Tensor,
        time: torch.Tensor,
    ) -> Data:
        return to_data(state, positions, targets, edge_index, components, time, self.device, self.dtype)  # type: ignore

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_action(self, td: TensorDictBase) -> TensorDictBase:
        """
        Choose an action to take in a rollout step.

        Returns:
            Modified data with data.action set to the action. Can set other fields as well.
        """
        with torch.no_grad():
            td = self.model.get_policy_operator()(td)
            if self.training:
                td["action"] = self.policy(td["action"])
            else:
                td["action"] = self.clip_action(td["action"])
            return td

    def populate(self):
        # populate replay buffer with initial data
        logger.info("Populating replay buffer with one rollout.")
        for i, data in enumerate(self.collector):
            if i >= self.max_steps:
                break
            self.buffer.extend(data)

    def train_generator(self):
        _training = self.training
        self.train(True)
        self.rollout_start()
        for i, data in enumerate(self.collector):
            if i >= self.max_steps:
                break
            self.buffer.extend(data)
            yield self.buffer.sample()
        self.train(_training)

    def test_generator(self):
        _training = self.training
        self.train(False)
        self.rollout_start()
        self.collector.reset()
        for i, data in enumerate(self.collector):
            if i >= self.max_steps:
                break
            yield data
        self.train(_training)

    def train_dataloader(self):
        return ExperienceSourceDataset(self.train_generator, length=self.max_steps)

    def val_dataloader(self):
        return ExperienceSourceDataset(self.test_generator, length=self.max_steps)
