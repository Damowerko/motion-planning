from time import sleep
from typing import Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from gym import spaces
from scipy.spatial import KDTree, distance_matrix
from scipy.optimize import linear_sum_assignment
import networkx as nx
from abc import ABC, abstractmethod

rng = np.random.default_rng()


class MotionPlanningRender:
    def __init__(self, width, state_ndim):
        self.positions = []
        self.width = width
        self.state_ndim = state_ndim
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.reset()
        # plt.show(block=False)

    def reset(self):
        self.ax.clear()
        self.target_scatter = None
        self.agent_scatter = None

    def render(
        self, goal_positions, agent_positions, reward, observed_targets, adjacency
    ):
        self.reset()
        if self.target_scatter is None:
            self.target_scatter = self.ax.plot(*goal_positions, "rx")[0]
        if self.agent_scatter is None:
            self.agent_scatter = self.ax.plot(*agent_positions, "bo")[0]

        G = nx.from_scipy_sparse_array(adjacency)
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.draw_networkx_edges(G, pos=agent_positions.T, ax=self.ax)
        targets = observed_targets.reshape(-1, 2)
        self.ax.plot(*targets.T, "y^")
        self.ax.set_title(f"Reward: {reward:.2f}")
        self.agent_scatter.set_data(*agent_positions)
        self.ax.set_xlim(-self.width, self.width)
        self.ax.set_ylim(-self.width, self.width)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()


class GraphEnv(gym.Env, ABC):
    @property
    @abstractmethod
    def n_nodes(self):
        pass

    @property
    @abstractmethod
    def action_ndim(self):
        pass

    @property
    @abstractmethod
    def observation_ndim(self):
        pass

    @abstractmethod
    def adjacency(self):
        pass


class MotionPlanning(GraphEnv):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.n_targets = 20
        self.n_agents = 20
        self.dt = 0.1
        self.max_steps = 200
        self.width = 1.0
        self.reward_cutoff = 0.2
        self.reward_sigma = 0.1

        # agent properties
        self.start_radius = 0.1
        self.max_accel = 0.5
        self.agent_radius = 0.01
        self.n_observed_agents = 3
        self.n_observed_targets = 3

        # comm graph properties
        self.adj_type = "knn"  # distance or knn
        self.n_neighbors = 5
        self.comm_radius = 0.5

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
        self._observation_ndim = (
            self.state_ndim + self.n_observed_targets * 2 + self.n_observed_agents * 2
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
    def position(self):
        return self.state[..., 0:2]

    @position.setter
    def position(self, value):
        self.state[..., 0:2] = value

    @property
    def velocity(self):
        return self.state[..., 2:4]

    @velocity.setter
    def velocity(self, value):
        self.state[..., 2:4] = value

    def adjacency(self):
        position_tree = KDTree(self.position)
        if self.adj_type == "distance":
            adj = position_tree.sparse_distance_matrix(position_tree, self.comm_radius)
            # adj.data = np.exp(-(adj.data ** 2))
            adj[adj > 0] = 1
            adj[np.arange(self.n_agents), np.arange(self.n_agents)] = 1
        elif self.adj_type == "knn":
            _, idx = position_tree.query(self.position, k=self.n_neighbors + 1)
            adj = scipy.sparse.dok_array((self.n_agents, self.n_agents))
            for i in range(self.n_agents):
                for j in idx[i]:
                    adj[i, j] = 1
                    adj[j, i] = 1
        else:
            raise ValueError(f"Unknown adjacency type: {self.adj_type}")
        return scipy.sparse.coo_matrix(adj)

    def reference_policy(self):
        distance = distance_matrix(self.position, self.target_positions)
        row_idx, col_idx = linear_sum_assignment(distance)
        assert (row_idx == np.arange(self.n_agents)).all()
        action = self.target_positions[col_idx] - self.position[row_idx]

        action = np.clip(action, -self.max_accel, self.max_accel)
        return action.reshape(self.action_space.shape)

    def decentralized_policy(self, hops=0):
        observed_targets = self._observed_targets().reshape(
            self.n_agents, self.n_observed_targets, 2
        )
        observed_agents = self._observed_agents().reshape(
            self.n_agents, self.n_observed_agents, 2
        )
        if hops == 0:
            action = observed_targets[:, 0, :] - self.position
        else:
            action = np.zeros(self.action_space.shape)
            for i in range(self.n_agents):
                agent_positions = np.concatenate(
                    (self.position[i, None, :], observed_agents[i]), axis=0
                )
                target_positions = observed_targets[i]
                distances = distance_matrix(agent_positions, target_positions)
                row_idx, col_idx = linear_sum_assignment(distances)
                assignment = col_idx[np.nonzero(row_idx == 0)]
                if len(assignment) > 0:
                    action[i] = target_positions[assignment] - agent_positions[0]

        action = np.clip(action, -self.max_accel, self.max_accel)
        return action.reshape(self.action_space.shape)

    def _observed_agents(self):
        position_tree = KDTree(self.position)
        _, idx = position_tree.query(
            self.position, k=np.arange(2, self.n_observed_agents + 2)
        )
        closest_agents = self.position[idx, :].reshape(self.n_agents, -1)
        return closest_agents

    def _observed_targets(self):
        _, idx = self.target_tree.query(self.position, k=self.n_observed_targets)
        closest_tagets = self.target_positions[idx, :].reshape(self.n_agents, -1)
        return closest_tagets

    def _observation(self):
        return np.concatenate(
            (self.state, self._observed_targets(), self._observed_agents()), axis=1
        ).reshape(self.observation_space.shape)

    def _done(self):
        timeout = self.t >= self.max_steps * self.dt
        too_far_gone = (np.abs(self.position) > self.width * 2).any(axis=1).all(axis=0)
        return timeout or too_far_gone

    def _reward(self):
        position_tree = KDTree(self.position)
        d, _ = position_tree.query(
            self.target_positions, k=1, distance_upper_bound=self.reward_cutoff
        )
        reward = np.exp(-(d ** 2) / (2 * self.reward_sigma ** 2))
        return np.sum(reward)

    def step(self, action):
        action = action.reshape(self.n_agents, self.action_ndim)
        action = np.clip(action, -1, 1) * self.max_accel
        self.velocity = action
        self.position += self.velocity * self.dt
        self.t += self.dt
        return self._observation(), self._reward(), self._done(), {}

    def reset(self):
        self.target_positions = rng.uniform(
            -self.width, self.width, (self.n_targets, 2)
        )
        self.target_tree = KDTree(self.target_positions)

        self.state = np.zeros((self.n_agents, self.state_ndim))
        self.position = rng.uniform(-self.width, self.width, (self.n_agents, 2))

        self.t = 0
        if self.render_ is not None:
            self.render_.reset()
        return self._observation()

    def render(self, mode="human"):
        assert mode == "human"
        if self.render_ is None:
            self.render_ = MotionPlanningRender(self.width, self.state_ndim)
        self.render_.render(
            self.target_positions.T,
            self.position.T,
            self._reward(),
            self._observed_targets(),
            self.adjacency(),
        )

    def close(self):
        pass
