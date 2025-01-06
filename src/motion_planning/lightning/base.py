from typing import Iterator, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torchcps.utils import add_model_specific_args

from motion_planning.architecture.base import ActorCritic
from motion_planning.envs.motion_planning import MotionPlanning
from motion_planning.rl import ExperienceSourceDataset


class MotionPlanningActorCritic(pl.LightningModule):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCritic,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.0001,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        gamma=0.99,
        max_steps=100,
        n_agents: int = 100,
        width: float = 10.0,
        agent_radius: float = 0.05,
        agent_margin: float = 0.05,
        collision_coefficient: float = 5.0,
        scenario: str = "uniform",
        noise: float = 0.05,
        noise_clip: float = 0.1,
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
        self.agent_radius = agent_radius
        self.agent_margin = agent_margin
        self.collision_coefficient = collision_coefficient
        self.noise = noise
        self.noise_clip = noise_clip

        self.env = MotionPlanning(
            n_agents=n_agents,
            width=width,
            scenario=scenario,
            agent_radius=agent_radius + agent_margin,
            collision_coefficient=collision_coefficient,
        )

    def policy(self, mu: torch.Tensor):
        """
        Sample policy from a normal distribution centered at mu and with standard devation `self.noise`.
        Noise will be clipped to `self.noise_clip` and the action will be clipped to the unit ball.
        """

        sigma = torch.ones_like(mu) * self.noise
        noise = torch.normal(0, sigma).clip(-self.noise_clip, self.noise_clip)
        action = mu + noise
        return self.clip_action(action)

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
        Clip action to the unit ball.
        """
        magnitude = torch.norm(action, dim=-1)
        mask = magnitude > 1.0
        action[mask, :] = action[mask, :] / magnitude[mask][..., None]
        return action

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
                self.model.actor.parameters(),
                lr=self.actor_lr,
                weight_decay=self.weight_decay,
            ),
            torch.optim.AdamW(
                self.model.critic.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        ]

    def to_data(self, state, positions, targets, adjacency) -> Data:
        if isinstance(adjacency, list):
            data = []
            for i, adj in enumerate(adjacency):
                data.append(self.to_data(state[i], positions[i], targets[i], adj))
            return Batch.from_data_list(data)  # type: ignore
        state = torch.from_numpy(state).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        positions = torch.from_numpy(positions).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        targets = torch.from_numpy(targets).to(
            dtype=self.dtype, device=self.device  # type: ignore
        )
        # assert state.shape == (self.env.n_nodes, self.env.observation_ndim)
        edge_index, edge_weight = from_scipy_sparse_matrix(adjacency)
        edge_index = edge_index.to(dtype=torch.long, device=self.device)
        edge_weight = edge_weight.to(dtype=self.dtype, device=self.device)  # type: ignore
        return Data(
            state=state,
            positions=positions,
            targets=targets,
            edge_index=edge_index,
            edge_attr=edge_weight,
            num_nodes=state.shape[0],
        )

    def rollout_start(self):
        """
        Called before rollout starts.
        """
        return None

    def rollout_action(self, data: Data):
        """
        Choose an action to take in a rollout step.

        Returns:
            Modified data with data.action set to the action. Can set other fields as well.
        """
        data.mu = self.model.forward_actor(data)
        data.action = self.policy(data.mu)
        return data

    def rollout_step(self, data: Data):
        """
        Take a step in the environment.

        Returns:
            next_data: The next data.
            reward: The reward.
            done: Whether the episode is done.
            coverage: The coverage.
            n_collisions: The number of collisions.
        """
        data = self.rollout_action(data)
        next_state, next_positions, next_targets, reward, done, _ = self.env.step(
            data.action.detach().cpu().numpy()  # type: ignore
        )
        next_data = self.to_data(
            next_state, next_positions, next_targets, self.env.adjacency()
        )
        coverage = self.env.coverage()
        n_collisions = self.env.n_collisions(r=self.agent_radius)
        return (
            data,
            next_data,
            reward,
            done,
            coverage,
            n_collisions,
        )

    @torch.no_grad()
    def rollout(self, render=False) -> tuple[List[Data], List[np.ndarray]]:
        self.rollout_start()
        episode = []
        observation, positions, targets = self.env.reset()
        data = self.to_data(observation, positions, targets, self.env.adjacency())
        frames = []
        for step in range(self.max_steps):
            if render:
                frames.append(self.env.render(mode="rgb_array"))

            # take step
            (
                data,
                next_data,
                reward,
                done,
                coverage,
                n_collisions,
            ) = self.rollout_step(data)

            # add additional attributes
            data = data.clone()
            data.step = step
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.coverage = torch.tensor([coverage]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.n_collisions = torch.tensor([n_collisions]).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append((data, next_data))
            data = next_data.clone()
            if done:
                break

        return episode, frames

    def batch_generator(self, *args, **kwargs) -> Iterator:
        """
        Generate batches of data.

        Args:
            n_episodes: Number of new episodes to generate. Can be zero.
            render: Whether to render the environment as we generate samples.
            use_buffer: Whether to use a replay buffer to generate samples.
        """
        raise NotImplementedError("Should be overriden by subclasses.")

    def _dataloader(self, **kwargs):
        return DataLoader(
            ExperienceSourceDataset(self.batch_generator, **kwargs),  # type: ignore
            batch_size=self.batch_size,
        )


# class TransformerLayer(gnn.MessagePassing):
#     def __init__(
#         self,
#         n_channels: int = 512,
#         n_heads: int = 8,
#     ):
#         super().__init__(aggr="add")
#         self.head_dim = n_channels // n_heads
#         self.mlp_pos_query = gnn.MLP(
#             in_channels=2,
#             hidden_channels=2 * n_channels,
#             out_channels=n_channels,
#         )
#         self.mlp_pos_value = gnn.MLP(
#             in_channels=2,
#             hidden_channels=2 * n_channels,
#             out_channels=n_channels,
#         )
#         self.mlp = gnn.MLP(
#             in_channels=n_channels,
#             hidden_channels=2 * n_channels,
#             out_channels=n_channels,
#             num_layers=2,
#             norm=None,
#         )

#         self.out
#         self.lin_q = nn.Linear(n_channels, n_channels)
#         self.lin_k = nn.Linear(n_channels, n_channels)
#         self.lin_v = nn.Linear(n_channels, n_channels)

#         self.linear1 = nn.Linear(n_channels, n_channels)
#         self.linear2 = nn.Linear(n_channels, n_channels)

#         self.norm_attention = nn.LayerNorm(n_channels)
#         self.norm_mlp = nn.LayerNorm(n_channels)

#     def forward(
#         self,
#         x: Tensor | PairTensor,
#         pos: Tensor | PairTensor,
#         edge_index: torch.Tensor,
#     ):
#         x_src, x_dst = (x, x) if isinstance(x, Tensor) else x
#         pos = (pos, pos) if isinstance(pos, Tensor) else pos
#         query = self.lin_q(x_dst).view(-1, self.n_heads, self.head_dim)
#         key = self.lin_k(x_src).view(-1, self.n_heads, self.head_dim)
#         value = self.lin_v(x_src).view(-1, self.n_heads, self.head_dim)
#         y = self.propagate(edge_index, query=query, key=key, value=value, pos=pos)
#         y = self.norm_attention(x_dst + y)
#         z = self.linear2(self.linear1(y).relu())
#         z = self.norm_mlp(y + z)
#         return z

#     def message(
#         self,
#         pos_i: Tensor,
#         pos_j: Tensor,
#         query_i: Tensor,
#         key_j: Tensor,
#         value_j: Tensor,
#         index: Tensor,
#         ptr: OptTensor,
#         size_i: Optional[int],
#     ):
#         delta_pos = pos_i - pos_j
#         pos_query = self.embedding_mlp(delta_pos)
#         pos_value = self.embedding_mlp(delta_pos)
#         # compute attention
#         alpha = ((query_i + pos_query) * key_j).sum(dim=-1) / self.head_dim**0.5
#         alpha = softmax(alpha, index, ptr, size_i)
#         out = alpha.view(-1, 1) * (value_j + pos_value)
#         return out
