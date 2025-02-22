from typing import Callable, Optional

import numpy as np
import scipy.sparse
import torch
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from motion_planning.envs.render_matplotlib import MotionPlanningRender


class MotionPlanningEnv(EnvBase):
    scenarios = {
        "uniform",
        "clusters",
        "gaussian_uniform",
        "circle",
        "two_lines",
        "icra",
    }

    def __init__(
        self,
        n_agents: int = 100,
        width: float = 1000.0,
        initial_separation: float = 5.0,
        scenario: str = "uniform",
        max_vel: float = 5.0,
        dt: float = 1.0,
        collision_distance: float = 2.5,
        collision_coefficient: float = 5.0,
        coverage_cutoff: float = 5.0,
        reward_sigma: float = 10.0,
        expert_policy: str | None = None,
    ):
        super().__init__(device="cpu", batch_size=torch.Size([]))
        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario {scenario} is not a valid scenario. Possibilities {self.scenarios}."
            )
        self.n_agents = n_agents
        self.n_targets = n_agents
        self.expert_policy = expert_policy
        self.scenario = scenario
        self.dt = dt
        self.width = width
        self.initial_separation = initial_separation
        self.coverage_cutoff = coverage_cutoff
        self.reward_sigma = reward_sigma
        self.collision_coefficient = collision_coefficient
        # agent properties
        self.max_vel = max_vel
        self.collision_distance = collision_distance
        self.observe_max_agents = 3
        self.observe_max_targets = 3
        # comm graph properties
        self.comm_max_neighbors = 3

        self.action_ndim = 2
        self.state_ndim = 4
        self.observation_ndim = int(
            2 + self.observe_max_targets * 2 + self.observe_max_agents * 2
        )

        self._render: Optional[MotionPlanningRender] = None
        self._make_spec()
        self._set_seed()
        self._reset()

    def _make_spec(self):
        self.action_spec = Bounded(
            -1, 1, torch.Size((self.n_agents, 2)), dtype=torch.float32
        )
        self.observation_spec = Composite(
            observation=Unbounded(
                # -self.width / 2,
                # self.width / 2,
                shape=torch.Size((self.n_agents, self.observation_ndim)),
                dtype=torch.float32,
            ),
            positions=Unbounded(
                # -self.width / 2,
                # self.width / 2,
                torch.Size((self.n_agents, 2)),
                dtype=torch.float32,
            ),
            targets=Unbounded(
                # -self.width / 2,
                # self.width / 2,
                torch.Size((self.n_targets, 2)),
                dtype=torch.float32,
            ),
            edge_index=Bounded(
                0,
                self.n_agents,
                torch.Size((2, 4 * self.n_agents)),
                dtype=torch.long,
            ),
            components=Bounded(
                0, self.n_agents, torch.Size((self.n_agents,)), dtype=torch.long
            ),
            n_collisions=Bounded(0, self.n_agents, torch.Size(()), dtype=torch.long),
            coverage=Bounded(0, 1, torch.Size(()), dtype=torch.float32),
            time=Bounded(0, float("inf"), torch.Size(()), dtype=torch.float32),
            shape=torch.Size(()),
        )
        if self.expert_policy is not None:
            self.observation_spec["expert"] = self.action_spec.clone()
        self.observation_spec
        self.reward_spec = Unbounded(shape=torch.Size((1,)), dtype=torch.float32)
        self.done_spec = Categorical(2, torch.Size((1,)), dtype=torch.bool)

    def _set_seed(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    @property
    def positions(self):
        return self.state[..., 0:2]

    @positions.setter
    def positions(self, value):
        self.state[..., 0:2] = value

    @property
    def velocity(self):
        return self.state[..., 2:4]

    @velocity.setter
    def velocity(self, value):
        self.state[..., 2:4] = value

    def clip_action(self, action):
        """
        Clip action to a unit circle with radius self.max_vel.
        Args:
            action: An array of shape (..., 2) representing the action for each agent.
        """
        action = action.copy()
        magnitude = np.linalg.norm(action, axis=-1)
        to_clip = magnitude > self.max_vel
        action[to_clip] = action[to_clip] / magnitude[to_clip, None] * self.max_vel
        return action

    def baseline_policy(self, name: str) -> NDArray:
        if name in ["c", "c_sq"]:
            return self.centralized_policy(distance_squared=name == "c_sq")
        elif name in ["d1", "d1_sq"]:
            return self.decentralized_policy(distance_squared=name == "d_sq")
        elif name == "d0":
            return self.decentralized_policy(hops=0)
        elif name == "capt":
            return self.capt_policy()
        else:
            raise ValueError(f"Unknown policy {name}.")

    def centralized_policy(self, distance_squared=False):
        cost = self.dist_pt**2 if distance_squared else self.dist_pt
        row_idx, col_idx = linear_sum_assignment(cost)
        action = self.targets[col_idx] - self.positions[row_idx]
        action = self.clip_action(action)
        return action / self.max_vel

    def decentralized_policy(self, hops=0, distance_squared=False):
        observed_targets = self._observed_targets()
        observed_agents = self._observed_agents()
        if hops == 0:
            action = observed_targets[:, 0, :]
        elif hops == 1:
            action = np.zeros(self.action_space.shape)  # type: ignore
            for i in range(self.n_agents):
                agent_positions = np.concatenate(
                    (
                        self.positions[i, None, :],
                        observed_agents[i] + self.positions[i, None, :],
                    ),
                    axis=0,
                )
                target_positions = observed_targets[i] + agent_positions[0]
                distances = cdist(agent_positions, target_positions)
                cost = distances**2 if distance_squared else distances
                row_idx, col_idx = linear_sum_assignment(cost)
                assignment = col_idx[np.nonzero(row_idx == 0)]
                if len(assignment) > 0:
                    action[i] = target_positions[assignment] - agent_positions[0]
        else:
            raise NotImplementedError("Hops > 1 not implemented.")
        action = self.clip_action(action)
        return action / self.max_vel

    def capt_policy(self):
        # compute the linear sum assignment on distance squared
        row_idx, col_idx = linear_sum_assignment(self.dist_pt**2)
        # find the distance for each target
        distances = np.linalg.norm(
            self.targets[col_idx] - self.positions[row_idx], axis=1
        )
        # find the maximum distance between agents and targets
        time_to_target = distances.max() / self.max_vel
        # since we are in discrete time, set dt as the minimum time to target
        time_to_target = max(time_to_target, self.dt)
        # find the velocity to reach the target in time_to_target
        action_raw = (self.targets[col_idx] - self.positions[row_idx]) / time_to_target
        action = self.clip_action(action_raw)
        return action / self.max_vel

    def _observed_agents(self):
        idx = argtopk(-self.dist_pp, self.observe_max_agents + 1, axis=1)
        idx = idx[:, 1:]  # remove self
        observed_positions = self.positions[idx] - self.positions[:, np.newaxis, :]
        return observed_positions

    def _observed_targets(self):
        idx = argtopk(-self.dist_pt, self.observe_max_targets, axis=1)
        observed_positions = self.targets[idx, :] - self.positions[:, np.newaxis, :]
        return observed_positions

    def _compute_distances(self):
        self.dist_pp = cdist(self.positions, self.positions)
        self.dist_pt = cdist(self.positions, self.targets)

    def _compute_graph(self):
        idx = argtopk(-self.dist_pp, self.comm_max_neighbors + 1, axis=1)
        coo = index_to_coo(idx)
        self.edge_index = np.stack((coo.row, coo.col), axis=0)

    def n_collisions(self, threshold: float) -> int:
        x_idx, y_idx = np.triu_indices_from(self.dist_pp, k=1)
        distances = self.dist_pp[x_idx, y_idx]
        return np.sum((distances < threshold).astype(int))

    def _reward(self):
        row_idx, col_idx = linear_sum_assignment(self.dist_pt)
        assert (row_idx == np.arange(self.n_agents)).all()
        # use the distance to the optimal assignment agent as a reward
        distances = self.dist_pt[row_idx, col_idx]
        reward_coverage = np.exp(-((distances / self.reward_sigma) ** 2))
        # count the number of collisions per agent
        n_collisions_per_agent = (
            np.sum(self.dist_pp < self.collision_distance, axis=1) - 1
        )
        penalty_collision = self.collision_coefficient * n_collisions_per_agent
        # the reward for each agent is the coverage reward minus the collision penalty
        reward = reward_coverage - penalty_collision
        return reward.mean()

    def components(self) -> np.ndarray:
        """
        Returns an array representing the connected components of the graph. Each node is assigned a component id. Nodes with the same component id are connected.
        """
        edge_index = self.edge_index
        edge_weight = np.ones(edge_index.shape[1], dtype="d")
        coo = scipy.sparse.coo_matrix(
            (edge_weight, edge_index), shape=(self.n_agents, self.n_agents)
        )
        _, component_ids = scipy.sparse.csgraph.connected_components(
            coo, directed=True, connection="weak", return_labels=True
        )
        return component_ids

    def coverage(self) -> float:
        return np.mean(np.any(self.dist_pt < self.coverage_cutoff, axis=0))

    def _make_output(self) -> TensorDictBase:
        observed_targets = self._observed_targets().reshape(self.n_agents, -1)
        observed_agents = self._observed_agents().reshape(self.n_agents, -1)
        observation = np.concatenate(
            (self.velocity / self.max_vel, observed_targets, observed_agents), axis=1
        )
        output = TensorDict(
            {
                "observation": torch.from_numpy(observation).float(),
                "positions": torch.from_numpy(self.positions).float(),
                "targets": torch.from_numpy(self.targets).float(),
                "edge_index": torch.from_numpy(self.edge_index).long(),
                "components": torch.from_numpy(self.components()).long(),
                "n_collisions": torch.as_tensor(
                    self.n_collisions(self.collision_distance)
                ).long(),
                "coverage": torch.as_tensor(self.coverage()).float(),
                "time": torch.as_tensor(self.time).float(),
            }
        )
        if self.expert_policy is not None:
            output["expert"] = torch.from_numpy(
                self.baseline_policy(self.expert_policy)
            )
        return output

    def _step(self, input: TensorDictBase) -> TensorDictBase:
        """
        Args:
            action (np.ndarray): The action to take. Normalized to a unit ball. Will be multipled by self.max_vel to get the actual velocity command.

        Returns:
            observation (dict): The observation of the environment. Keys are "observation", "positions", and "targets".
            reward (float): The reward of the environment.
            terminated (bool): Whether the episode is terminated.
            truncated (bool): Whether the episode is truncated.
            info (dict): Additional information. Keys are "edge_index", "components", "n_collisions", "coverage", and "time".
        """
        action = input["action"].detach().cpu().numpy()
        action = self.clip_action(action * self.max_vel)
        self.velocity = action
        self.positions += self.velocity * self.dt
        self.time += self.dt
        self._compute_distances()
        self._compute_graph()
        output = self._make_output()
        output["reward"] = (torch.as_tensor(self._reward()).float(),)
        output["done"] = (torch.as_tensor(False),)
        return output

    def _reset(self, *args) -> TensorDictBase:
        self.state = np.zeros((self.n_agents, self.state_ndim))
        if self.scenario == "uniform":
            self.targets = collision_free_sampling(
                self.initial_separation,
                lambda: self.rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.positions = collision_free_sampling(
                self.initial_separation,
                lambda: self.rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_agents, 2)
                ),
            )
        elif self.scenario == "gaussian_uniform":
            # agents are normally distributed around the origin
            # targets are uniformly distributed
            self.targets = collision_free_sampling(
                self.initial_separation,
                lambda: self.rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.positions = collision_free_sampling(
                self.initial_separation,
                lambda: self.rng.normal(size=(self.n_agents, 2)),
            )
        elif self.scenario == "clusters":
            n_targets_per_cluster = self.rng.choice([1, 5, 10])
            n_agents_per_cluster = self.rng.choice([1, 5, 10])
            self.targets = init_clusters(
                self.n_targets,
                n_targets_per_cluster,
                self.width,
                self.initial_separation,
                self.rng,
            )
            self.positions = init_clusters(
                self.n_agents,
                n_agents_per_cluster,
                self.width,
                self.initial_separation,
                self.rng,
            )
        elif self.scenario == "circle":

            def circ_sampler(n):
                radius = self.rng.uniform(
                    3 * self.width / 16, 5 * self.width / 16, (n, 1)
                )
                angle = self.rng.uniform(-np.pi, np.pi, (n, 1))
                return np.concatenate(
                    [radius * np.cos(angle), radius * np.sin(angle)], axis=1
                )

            self.targets = collision_free_sampling(
                self.initial_separation,
                lambda: self.rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.positions = collision_free_sampling(
                self.initial_separation,
                lambda: circ_sampler(self.n_targets),
            )
        elif self.scenario == "two_lines":
            sampler = lambda x: (
                lambda: np.concatenate(
                    [
                        self.rng.uniform(
                            self.width / 4 if x == 0 else -3 * self.width / 8,
                            3 * self.width / 8 if x == 0 else -self.width / 4,
                            (self.n_targets, 1),
                        ),
                        self.rng.uniform(
                            -self.width / 2, self.width / 2, (self.n_targets, 1)
                        ),
                    ],
                    axis=1,
                )
            )
            self.positions = collision_free_sampling(
                self.initial_separation, sampler(1)
            )
            self.targets = collision_free_sampling(self.initial_separation, sampler(0))
        elif self.scenario == "icra":
            self.positions = init_uniform(
                self.n_agents, self.width, self.initial_separation, self.rng
            )
            self.targets = init_icra(self.n_targets, self.width)
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Should be one of {self.scenarios}."
            )

        self.time = 0.0
        self._compute_distances()
        self._compute_graph()
        if self._render is not None:
            self._render.reset()
        return self._make_output()

    def render(self):
        if self._render is None:
            self._render = MotionPlanningRender(self.width, self.state_ndim)

        return self._render.render(
            self.targets,
            self.positions,
            self._reward(),
            self.coverage(),
            self.edge_index,
            self._observed_targets(),
        )


def init_uniform(n_samples, width, initial_separation, rng):
    return collision_free_sampling(
        initial_separation,
        lambda: rng.uniform(-width / 2, width / 2, (n_samples, 2)),
    )


def uniform_circle(n_samples, radius, rng):
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    r = rng.uniform(0, radius, n_samples)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


def init_clusters(n_samples, n_samples_per_cluster, width, initial_separation, rng):
    assert (
        n_samples % n_samples_per_cluster == 0
    ), f"n_samples={n_samples} must be divisible by n_samples_per_cluster={n_samples_per_cluster}"
    n_clusters = n_samples // n_samples_per_cluster
    if n_clusters == n_samples:
        return init_uniform(n_samples, width, initial_separation, rng)
    cluster_radius = 5 * initial_separation * n_samples_per_cluster**0.5
    cluster_centers = collision_free_sampling(
        2 * cluster_radius,
        lambda: rng.uniform(-width / 2, width / 2, (n_clusters, 2)),
    )
    return collision_free_sampling(
        initial_separation,
        lambda: uniform_circle(n_samples, cluster_radius, rng)
        + cluster_centers.repeat(n_samples_per_cluster, axis=0),
    )


def init_icra(n_samples, width):
    if n_samples != 100:
        raise ValueError("ICRA scenario only supports 100 agents.")
    i_points = []
    for row in range(8):
        for col in range(2):
            i_points.append(
                np.array(
                    [
                        row * width / 16 - width / 4,
                        col * width / 16 - width / 2,
                    ]
                )
            )
    i_points = np.array(i_points)

    c_points = []
    for row in range(2):
        for col in range(4):
            c_points.append(
                np.array(
                    [
                        row * width / 16 - width / 4,
                        col * width / 16 - 5 * width / 16,
                    ]
                )
            )
    for row in range(4):
        for col in range(2):
            c_points.append(
                np.array(
                    [
                        row * width / 16 - width / 8,
                        col * width / 16 - 5 * width / 16,
                    ]
                )
            )
    for row in range(2):
        for col in range(4):
            c_points.append(
                np.array(
                    [
                        row * width / 16 + width / 8,
                        col * width / 16 - 5 * width / 16,
                    ]
                )
            )
    c_points = np.array(c_points)

    r_points = []
    for row in range(2):
        for col in range(4):
            r_points.append(
                np.array(
                    [
                        row * width / 16 - width / 4,
                        col * width / 16,
                    ]
                )
            )
    for row in range(2):
        for col in range(2):
            r_points.append(
                np.array(
                    [
                        row * width / 16 - width / 8,
                        col * width / 16,
                    ]
                )
            )
    for row in range(2):
        for col in range(1):
            r_points.append(
                np.array(
                    [
                        row * width / 16 - width / 8,
                        col * width / 16 + 3 * width / 16,
                    ]
                )
            )
    for row in range(2):
        for col in range(4):
            r_points.append(
                np.array(
                    [
                        row * width / 16 - width / 16,
                        col * width / 16,
                    ]
                )
            )
    for col in range(3):
        r_points.append(np.array([width / 16, col * width / 16]))
    for row in range(2):
        for col in range(2):
            r_points.append(
                np.array(
                    [
                        row * width / 16 + width / 8,
                        col * width / 16,
                    ]
                )
            )
    for row in range(2):
        for col in range(1):
            r_points.append(
                np.array(
                    [
                        row * width / 16 + width / 8,
                        col * width / 16 + 3 * width / 16,
                    ]
                )
            )

    a_points = []
    for row in range(2):
        for col in range(4):
            a_points.append(
                np.array(
                    [
                        row * width / 16 - width / 4,
                        col * width / 16 + 5 * width / 16,
                    ]
                )
            )
    for row in range(2):
        a_points.append(np.array([row * width / 16 - width / 8, 5 * width / 16]))
    for row in range(2):
        a_points.append(np.array([row * width / 16 - width / 8, width / 2]))
    for row in range(2):
        for col in range(4):
            a_points.append(
                np.array(
                    [
                        row * width / 16 - width / 16,
                        col * width / 16 + 5 * width / 16,
                    ]
                )
            )
    for row in range(3):
        a_points.append(np.array([row * width / 16 + width / 16, 5 * width / 16]))
    for row in range(3):
        a_points.append(np.array([row * width / 16 + width / 16, width / 2]))

    others = np.array(
        [
            [3 * width / 8, -width / 4],
            [3 * width / 8, 0],
            [3 * width / 8, width / 4],
        ]
    )

    targets = np.concatenate([i_points, c_points, r_points, a_points, others], axis=0)
    return targets


def argtopk(X, K, axis=-1):
    """
    Return the indices of the top K largest elements along an axis in descending order.
    """
    r = range(K)
    idx = np.argpartition(-X, r, axis=axis)
    return idx.take(r, axis=axis)


def index_to_coo(idx, mask=None):
    """
    Create an scipy coo_matrix from an index array.

    Args:
        idx: An array of shape (N, M) that represents a matrix A[i, idx[i, j]] = 1 for i,j.
        mask: A boolean array of shape (N, M) that represents the mask of the matrix.

    Returns:
        A scipy coo_matrix of shape (N, N)
    """
    N = idx.shape[0]
    M = idx.shape[1]
    i = np.repeat(np.arange(N), M)
    j = np.ravel(idx, order="C")
    if mask is not None:
        mask = np.ravel(mask, order="C")
        i = i[mask]
        j = j[mask]
    data = np.ones_like(i)
    return scipy.sparse.coo_matrix((data, (i, j)), shape=(N, N))


def collision_free_sampling(d: float, sampler: Callable[[], np.ndarray]):
    """
    Sample n positions without collisions within a given radius r.

    Args:
        d: The minimum distance between positions.
        sampler: A function that samples positions given the number of positions to sample.
    """
    # set of indices of agents that were changed in this iteration
    positions: np.ndarray = sampler()
    idx: np.ndarray = np.arange(positions.shape[0])
    while True:
        dist = cdist(positions, positions[idx])
        dist[idx, np.arange(len(idx))] = np.inf
        min_dist = dist.min(axis=0)
        if (min_dist >= d).all():
            break
        # find indices of agents that are too close and need to be changed
        # this set will be monotonically decreasing
        idx = idx[min_dist < d]
        positions[idx] = sampler()[idx]
    return positions
