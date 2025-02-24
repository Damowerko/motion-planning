import pandas as pd
import torch
import torch_scatter
import tqdm
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase

from motion_planning.envs.motion_planning import MotionPlanningEnvParams
from motion_planning.evaluate.common import evaluate_policy


def delay(
    env_params: MotionPlanningEnvParams,
    policy: TensorDictModuleBase,
    max_steps: int,
    num_episodes: int,
    num_workers: int | None = None,
):
    dfs = []
    for i in tqdm.trange(0, 11, desc="Comm interval"):
        comm_interval = i * 0.1
        delayed_policy = DelayedModel(policy, comm_interval=comm_interval)
        df, _ = evaluate_policy(
            env_params, delayed_policy, max_steps, num_episodes, num_workers
        )
        df["delay_s"] = comm_interval
        dfs.append(df)
    return pd.concat(dfs)


class DelayedModel(TensorDictModuleBase):

    def __init__(
        self,
        model: TensorDictModuleBase,
        comm_interval: float = 1.0,
    ):
        """
        Wrap a model to simulate communication delays.

        Args:
            model (torch.nn.Module): The model to wrap.
            comm_interval (float): Interval between communication exchanges.
            batch_size (int): The batch size of the model.
        """
        super().__init__()
        self.model = model
        self.initialized = False
        self.comm_interval = comm_interval
        self.in_keys = list(
            set(model.in_keys + ["observation", "positions", "edge_index", "time"])
        )
        self.out_keys = model.out_keys

    def _init_buffers(
        self, batch_size: int, n_agents: int, n_features: int, device: torch.device
    ):
        self.initialized = True
        self.batch_size = batch_size
        self.n_agents = n_agents
        self.n_features = n_features
        self.device = device
        self.comm_time = 0.0

        # time is an NxN array that stores the most recent time that agent i has received information about agent j
        # negative values indicate that agent i has not received any information about agent j
        self.time_buffer = -torch.ones(
            self.batch_size, self.n_agents, self.n_agents, device=self.device
        )
        # state_buffer is an NxNxF array that stores the most recent state of agent j that agent i has received
        self.observation_buffer = torch.zeros(
            self.batch_size,
            self.n_agents,
            self.n_agents,
            self.n_features,
            device=self.device,
        )
        self.positions_buffer = torch.zeros(
            self.batch_size, self.n_agents, self.n_agents, 2, device=self.device
        )

    def _update_buffer(
        self,
        observation: torch.Tensor,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        time: float,
    ):
        # locally we always have the most recent information
        self._update_self(observation, positions, time)
        # the robot simulation environment may be discretized at a different time step
        while self.comm_time <= time:
            self.comm_time += self.comm_interval
            self._simulate_communication(edge_index)

    def _update_self(
        self, observation: torch.Tensor, positions: torch.Tensor, time: float
    ):
        self.time_buffer.diagonal(dim1=1, dim2=2).fill_(time)
        self.observation_buffer.diagonal(dim1=1, dim2=2).copy_(observation.mT)
        self.positions_buffer.diagonal(dim1=1, dim2=2).copy_(positions.mT)

    def _simulate_communication(
        self,
        edge_index: torch.Tensor,
    ):
        """
        Exchange information between neighboring agents. Updates `self.time_buffer`, `self.observation_buffer` and `self.positions_buffer`.

        Args:
            edge_index (torch.Tensor): (B, 2, E) The edge index of the graph.

        """
        # Vectorized approach using torch_scatter
        _src = edge_index[..., 0, :]  # (B, E)
        _dst = edge_index[..., 1, :]  # (B, E)
        # since the graph is undirected, we need to consider both directions
        # src = torch.cat([_src, _dst], dim=-1)
        # dst = torch.cat([_dst, _src], dim=-1)
        src = _src
        dst = _dst

        # Information flows from src to target, for each edge (src, dst) in edge_index,
        # we find the most recent time the src agent has received information from each agent
        time_src = torch.gather(
            self.time_buffer, 1, src.unsqueeze(-1).expand(-1, -1, self.n_agents)
        )  # (B, E, n_agents)

        # Now for each dst agent, we find the maximum of all the edges that connect to it
        # This gives us the the most recent time that any src agent going to dst
        # has received information from any other agent
        max_time, max_arg = torch_scatter.scatter_max(
            time_src, dst, dim=1
        )  # (B, n_agents, n_agents)

        # Identify which indices should be updated and update all buffers at once
        mask = max_time > self.time_buffer
        b, i, j = torch.nonzero(mask, as_tuple=True)
        k = src[b, max_arg[b, i, j]]

        # Update all buffers in one operation using index_put_
        self.time_buffer.index_put_((b, i, j), max_time[b, k, j])
        self.observation_buffer.index_put_((b, i, j), self.observation_buffer[b, k, j])
        self.positions_buffer.index_put_((b, i, j), self.positions_buffer[b, k, j])

    def get_batch(self):
        """
        Represent delayed information for each agents as a batch. The information from the perspective of the i-th agent is stored in the i-th batch element.

        Args:
            - components (torch.Tensor): (B, N,) The component id of each agent.

        Returns:
            - observation (torch.Tensor): (B, N, N, F) Observations from the perspective of each agent. observation[b, i, j] contains the latest information about agent j from the perspective of agent i.
            - positions (torch.Tensor): (B, N, N, 2) Positions from the perspective of each agent. positions[b, i, j] contains the latest position of agent j from the perspective of agent i.
            - padding_mask (torch.Tensor): (B, N, N) padding_mask[i, j] is True if agent i has received information about agent j.
        """
        return (
            self.observation_buffer.clone(),
            self.positions_buffer.clone(),
            self.time_buffer > -1,
        )

    def forward(
        self,
        td: TensorDictBase,
    ):
        if self.comm_interval == 0:
            return self.model(td)

        # we need to make sure that all the time tensors are equal
        time_flat = td["time"].flatten()
        if (time_flat != time_flat[0]).any():
            raise ValueError("Time must be the same for all batch elements.")
        time = time_flat[0]

        if not self.initialized or time == 0:
            batch_size, n_agents, n_features = td["observation"].shape
            assert td.device is not None
            self._init_buffers(batch_size, n_agents, n_features, td.device)

        self._update_buffer(td["observation"], td["positions"], td["edge_index"], time)
        # create a batch of data objects with the state for each agent
        observations, positions, padding_mask = self.get_batch()
        # the outputs of get_batch have dimensions (B, N, N, *)
        # need to expand the dims of td to match the new dimensions
        td_expanded = td.unsqueeze(1).expand(self.batch_size, self.n_agents)
        td_expanded["observation"] = observations
        td_expanded["positions"] = positions
        td_expanded["padding_mask"] = padding_mask

        td_outputs = self.model(
            td_expanded.reshape(self.batch_size * self.n_agents)
        ).reshape(self.batch_size, self.n_agents)
        for key in self.out_keys:
            td[key] = td_outputs[key].diagonal(dim1=1, dim2=2).mT
        return td
