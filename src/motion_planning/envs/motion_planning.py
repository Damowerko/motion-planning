import random
from abc import ABC, abstractmethod
from typing import Callable, Optional

import gym
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
from gym import spaces
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

rng = np.random.default_rng()


def init_uniform(n_samples, width, initial_separation):
    return collision_free_sampling(
        initial_separation,
        lambda: rng.uniform(-width / 2, width / 2, (n_samples, 2)),
    )


def uniform_circle(n_samples, radius):
    theta = rng.uniform(0, 2 * np.pi, n_samples)
    r = rng.uniform(0, radius, n_samples)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)


def init_clusters(n_samples, n_samples_per_cluster, width, initial_separation):
    assert (
        n_samples % n_samples_per_cluster == 0
    ), f"n_samples={n_samples} must be divisible by n_samples_per_cluster={n_samples_per_cluster}"
    n_clusters = n_samples // n_samples_per_cluster
    if n_clusters == n_samples:
        return init_uniform(n_samples, width, initial_separation)
    cluster_radius = 5 * initial_separation * n_samples_per_cluster**0.5
    cluster_centers = collision_free_sampling(
        2 * cluster_radius,
        lambda: rng.uniform(-width / 2, width / 2, (n_clusters, 2)),
    )
    return collision_free_sampling(
        initial_separation,
        lambda: uniform_circle(n_samples, cluster_radius)
        + cluster_centers.repeat(n_samples_per_cluster, axis=0),
    )


class MotionPlanningRender:
    def __init__(self, width, state_ndim, mode="human"):
        self.positions = []
        self.width = width
        self.state_ndim = state_ndim
        self.mode = mode
        if mode == "human":
            plt.ion()
        else:
            plt.ioff()
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.reset()

    def reset(self):
        self.ax.clear()
        self.target_scatter = None
        self.agent_scatter = None

    def render(
        self,
        goal_positions,
        agent_positions,
        reward,
        coverage,
        observed_targets,
        adjacency,
    ):
        """
        Renders the environment with the given parameters.

        Args:
            goal_positions (array-like): The positions of the goal targets.
            agent_positions (array-like): The positions of the agents.
            reward (float): The reward value.
            observed_targets (array-like): The observed targets.
            adjacency (scipy.sparse.csr_matrix): The adjacency matrix.

        Returns:
            matplotlib.figure.Figure: The rendered figure.
        """
        self.reset()
        if not isinstance(self.fig.canvas, FigureCanvasAgg):
            raise ValueError("Only agg matplotlib backend is supported.")

        markersize = 6.0 * (1000 / self.width)

        if self.target_scatter is None:
            self.target_scatter = self.ax.plot(
                *goal_positions, "rx", markersize=markersize
            )[0]

        if self.agent_scatter is None:
            self.agent_scatter = self.ax.plot(
                *agent_positions, "bo", markersize=markersize
            )[0]

        self.ax.set_xlim(-self.width / 2, self.width / 2)
        self.ax.set_ylim(-self.width / 2, self.width / 2)

        G = nx.from_scipy_sparse_array(adjacency)
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.draw_networkx_edges(G, pos=agent_positions.T, ax=self.ax)

        targets = (observed_targets + agent_positions.T[:, np.newaxis, :]).reshape(
            -1, 2
        )
        self.ax.plot(*targets.T, "y^", markersize=markersize)

        reward = reward.mean()
        self.ax.set_title(f"Reward: {reward:.2f}, Coverage: {np.round(coverage*100)}%")

        self.agent_scatter.set_data(*agent_positions)

        if self.mode == "human":
            self.fig.canvas.flush_events()
            self.fig.canvas.draw_idle()
        elif self.mode == "rgb_array":
            self.fig.canvas.draw()
            return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        else:
            raise ValueError(
                "Unknown mode: {self.mode}. Should be one of ['human', 'rgb_array']."
            )


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


class GraphEnv(gym.Env, ABC):
    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def action_ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_ndim(self) -> int:
        pass

    @abstractmethod
    def adjacency(self) -> scipy.sparse.coo_matrix:
        pass


class MotionPlanning(GraphEnv):
    metadata = {"render.modes": ["human"]}
    scenarios = {
        "uniform",
        "gaussian_uniform",
        "clusters",
        "circle",
        "two_lines",
        "icra",
        "q-scenario",
    }

    def __init__(
        self,
        n_agents: int = 100,
        width: float = 1000.0,
        initial_separation: float = 5.0,
        scenario: str = "uniform",
        max_vel: float = 10.0,
        dt: float = 1.0,
        collision_distance: float = 2.5,
        collision_coefficient: float = 5.0,
        coverage_cutoff: float = 5.0,
        reward_sigma: float = 10.0,
    ):
        self.n_agents = n_agents
        self.n_targets = n_agents
        self.n_obstacles = int(n_agents / 10)

        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario {scenario} is not a valid scenario. Possibilities {self.scenarios}."
            )
        self.scenario = scenario

        # Since space is 2D scale is inversely proportional to sqrt of the number of agents

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

        self._n_nodes = self.n_agents

        self._action_ndim = 2
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.n_agents,
                self.action_ndim,
            ),
        )

        self.state_ndim = 4
        self._observation_ndim = int(
            self.state_ndim / 2
            + self.observe_max_targets * 2
            + self.observe_max_agents * 2
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.n_agents,
                self.observation_ndim,
            ),
        )

        self.render_: Optional[MotionPlanningRender] = None
        self.reset()

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def action_ndim(self):
        return self._action_ndim

    @property
    def observation_ndim(self):
        return self._observation_ndim

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

    def adjacency(self):
        return self._adjacency

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

    def centralized_policy(self, distance_squared=False):
        cost = self.dist_pt**2 if distance_squared else self.dist_pt
        row_idx, col_idx = linear_sum_assignment(cost)
        assert (row_idx == np.arange(self.n_agents)).all()
        action = self.targets[col_idx] - self.positions[row_idx]
        action = self.clip_action(action)
        assert action.shape == self.action_space.shape  # type: ignore
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
        assert action.shape == self.action_space.shape  # type: ignore
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

    def _observation(self):
        tgt = self._observed_targets().reshape(self.n_agents, -1)
        agt = self._observed_agents().reshape(self.n_agents, -1)
        obs = np.concatenate((self.velocity / self.max_vel, tgt, agt), axis=1)
        assert obs.shape == self.observation_space.shape  # type: ignore
        return obs

    def _done(self) -> bool:
        too_far_gone = (np.abs(self.positions) > self.width).any(axis=1).all(axis=0)
        return too_far_gone

    def _compute_distances(self):
        self.dist_pp = cdist(self.positions, self.positions)
        self.dist_pt = cdist(self.positions, self.targets)

    def _compute_adjacency(self):
        idx = argtopk(-self.dist_pp, self.comm_max_neighbors + 1, axis=1)
        self._adjacency = index_to_coo(idx)

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
        return reward

    def components(self) -> np.ndarray:
        """
        Returns an array representing the connected components of the graph. Each node is assigned a component id. Nodes with the same component id are connected.
        """
        _, component_ids = scipy.sparse.csgraph.connected_components(
            self.adjacency(), directed=True, connection="weak", return_labels=True
        )
        return component_ids

    def coverage(self) -> float:
        return np.mean(np.any(self.dist_pt < self.coverage_cutoff, axis=0))

    def step(self, action):
        """
        Args:
            action (np.ndarray): The action to take. Normalized to a unit ball. Will be multipled by self.max_vel to get the actual velocity command.

        Returns:
            observation (np.ndarray): The observation of the environment.
            agent_positions (np.ndarray): The position of the agents.
            target_positions (np.ndarray): The position of the targets.
            reward (float): The reward of the environment.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        assert action.shape == self.action_space.shape  # type: ignore
        action = self.clip_action(action * self.max_vel)
        self.velocity = action
        self.positions += self.velocity * self.dt
        self.t += self.dt
        self._compute_distances()
        self._compute_adjacency()
        return (
            self._observation(),
            self.positions,
            self.targets,
            self._reward(),
            self._done(),
            {},
        )

    def reset(self):
        self.state = np.zeros((self.n_agents, self.state_ndim))
        if self.scenario == "uniform":
            self.targets = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.positions = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_agents, 2)
                ),
            )
        elif self.scenario == "gaussian_uniform":
            # agents are normally distributed around the origin
            # targets are uniformly distributed
            self.targets = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.positions = collision_free_sampling(
                self.initial_separation,
                lambda: rng.normal(size=(self.n_agents, 2)),
            )
        elif self.scenario == "clusters":
            n_targets_per_cluster = random.choice([1, 10, 25])
            n_agents_per_cluster = random.choice([1, 10, 25])
            self.targets = init_clusters(
                self.n_targets,
                n_targets_per_cluster,
                self.width,
                self.initial_separation,
            )
            self.positions = init_clusters(
                self.n_agents,
                n_agents_per_cluster,
                self.width,
                self.initial_separation,
            )
        elif self.scenario == "circle":

            def circ_sampler(n):
                radius = rng.uniform(3 * self.width / 16, 5 * self.width / 16, (n, 1))
                angle = rng.uniform(-np.pi, np.pi, (n, 1))
                return np.concatenate(
                    [radius * np.cos(angle), radius * np.sin(angle)], axis=1
                )

            self.target_positions = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )
            self.position = collision_free_sampling(
                self.initial_separation,
                lambda: circ_sampler(self.n_targets),
            )
        elif self.scenario == "two_lines":
            sampler = lambda x: (
                lambda: np.concatenate(
                    [
                        rng.uniform(
                            self.width / 4 if x == 0 else -3 * self.width / 8,
                            3 * self.width / 8 if x == 0 else -self.width / 4,
                            (self.n_targets, 1),
                        ),
                        rng.uniform(
                            -self.width / 2, self.width / 2, (self.n_targets, 1)
                        ),
                    ],
                    axis=1,
                )
            )
            self.position = collision_free_sampling(self.initial_separation, sampler(1))
            self.target_positions = collision_free_sampling(
                self.initial_separation, sampler(0)
            )
        elif self.scenario == "icra":
            self.position = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_agents, 2)
                ),
            )

            i_points = []
            for row in range(8):
                for col in range(2):
                    i_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 - self.width / 2,
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
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 - 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(4):
                for col in range(2):
                    c_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16 - 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(4):
                    c_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16 - 5 * self.width / 16,
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
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(2):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(1):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16 + 3 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(4):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 16,
                                col * self.width / 16,
                            ]
                        )
                    )
            for col in range(3):
                r_points.append(np.array([self.width / 16, col * self.width / 16]))
            for row in range(2):
                for col in range(2):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(1):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16 + 3 * self.width / 16,
                            ]
                        )
                    )

            a_points = []
            for row in range(2):
                for col in range(4):
                    a_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 + 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                a_points.append(
                    np.array(
                        [row * self.width / 16 - self.width / 8, 5 * self.width / 16]
                    )
                )
            for row in range(2):
                a_points.append(
                    np.array([row * self.width / 16 - self.width / 8, self.width / 2])
                )
            for row in range(2):
                for col in range(4):
                    a_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 16,
                                col * self.width / 16 + 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(3):
                a_points.append(
                    np.array(
                        [row * self.width / 16 + self.width / 16, 5 * self.width / 16]
                    )
                )
            for row in range(3):
                a_points.append(
                    np.array([row * self.width / 16 + self.width / 16, self.width / 2])
                )

            others = np.array(
                [
                    [3 * self.width / 8, -self.width / 4],
                    [3 * self.width / 8, 0],
                    [3 * self.width / 8, self.width / 4],
                ]
            )

            self.target_positions = np.concatenate(
                [i_points, c_points, r_points, a_points, others], axis=0
            )

        elif self.scenario == "icra":
            self.position = collision_free_sampling(
                self.initial_separation,
                lambda: rng.uniform(
                    -self.width / 2, self.width / 2, (self.n_targets, 2)
                ),
            )

            i_points = []
            for row in range(8):
                for col in range(2):
                    i_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 - self.width / 2,
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
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 - 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(4):
                for col in range(2):
                    c_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16 - 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(4):
                    c_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16 - 5 * self.width / 16,
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
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(2):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(1):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 8,
                                col * self.width / 16 + 3 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(4):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 16,
                                col * self.width / 16,
                            ]
                        )
                    )
            for col in range(3):
                r_points.append(np.array([self.width / 16, col * self.width / 16]))
            for row in range(2):
                for col in range(2):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                for col in range(1):
                    r_points.append(
                        np.array(
                            [
                                row * self.width / 16 + self.width / 8,
                                col * self.width / 16 + 3 * self.width / 16,
                            ]
                        )
                    )

            a_points = []
            for row in range(2):
                for col in range(4):
                    a_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 4,
                                col * self.width / 16 + 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(2):
                a_points.append(
                    np.array(
                        [row * self.width / 16 - self.width / 8, 5 * self.width / 16]
                    )
                )
            for row in range(2):
                a_points.append(
                    np.array([row * self.width / 16 - self.width / 8, self.width / 2])
                )
            for row in range(2):
                for col in range(4):
                    a_points.append(
                        np.array(
                            [
                                row * self.width / 16 - self.width / 16,
                                col * self.width / 16 + 5 * self.width / 16,
                            ]
                        )
                    )
            for row in range(3):
                a_points.append(
                    np.array(
                        [row * self.width / 16 + self.width / 16, 5 * self.width / 16]
                    )
                )
            for row in range(3):
                a_points.append(
                    np.array([row * self.width / 16 + self.width / 16, self.width / 2])
                )

            others = np.array(
                [
                    [3 * self.width / 8, -self.width / 4],
                    [3 * self.width / 8, 0],
                    [3 * self.width / 8, self.width / 4],
                ]
            )

            self.target_positions = np.concatenate(
                [i_points, c_points, r_points, a_points, others], axis=0
            )
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Should be one of {self.scenarios}."
            )

        self.t = 0
        self._compute_distances()
        self._compute_adjacency()
        if self.render_ is not None:
            self.render_.reset()
        return self._observation(), self.positions, self.targets

    def render(self, mode="human"):
        if self.render_ is None or self.render_.mode != mode:
            self.render_ = MotionPlanningRender(self.width, self.state_ndim, mode=mode)

        return self.render_.render(
            self.targets.T,
            self.positions.T,
            self._reward(),
            self.coverage(),
            self._observed_targets(),
            self.adjacency(),
        )

    def close(self):
        pass


if __name__ == "__main__":
    env = MotionPlanning(
        scenario="uniform", n_agents=100, width=10, collision_distance=0.1
    )
    n_steps = 200
    for i in range(100):
        env.reset()
        for i in range(n_steps):
            action = env.centralized_policy()
            _, _, _, _, done, _ = env.step(action)
            adj = env.adjacency()
            env.render()
            if done:
                break
