from abc import ABC, abstractmethod
from typing import Optional

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
        self, goal_positions, agent_positions, reward, coverage, observed_targets, adjacency
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
        
        markersize = 75 / self.width

        if self.target_scatter is None:
            self.target_scatter = self.ax.plot(*goal_positions, "rx", markersize=markersize)[0]

        if self.agent_scatter is None:
            self.agent_scatter = self.ax.plot(*agent_positions, "bo", markersize=markersize)[0]

        self.ax.set_xlim(-self.width / 2, self.width / 2)
        self.ax.set_ylim(-self.width / 2, self.width / 2)

        G = nx.from_scipy_sparse_array(adjacency)
        G.remove_edges_from(nx.selfloop_edges(G))
        nx.draw_networkx_edges(G, pos=agent_positions.T, ax=self.ax)

        targets = (observed_targets + agent_positions.T[:,None,:]).reshape(-1, 2)
        self.ax.plot(*targets.T, "y^", markersize=markersize)

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


def idist(X_A, X_B):
    """
    Return the distance matrix given two complex coordinate matrices
    """
    X_A = np.concatenate((X_A.real, X_A.imag), axis=1)
    X_B = np.concatenate((X_B.real, X_B.imag), axis=1)
    return cdist(X_A, X_B)


def argtopk(X, K, axis=-1):
    """
    Return the indices of the top K largest elements along an axis in descending order.
    """
    r = range(K)
    idx = np.argpartition(-X, r, axis=axis)
    return idx.take(r, axis=axis)


def index_to_coo(idx):
    """
    Create an scipy coo_matrix from an index array.

    Args:
        idx: An array of shape (N, M) that represents a matrix A[i, idx[i, j]] = 1 for i,j.

    Returns:
        A scipy coo_matrix of shape (N, N)
    """
    N = idx.shape[0]
    M = idx.shape[1]
    i = np.repeat(np.arange(N), M)
    j = np.ravel(idx, order="C")
    data = np.ones_like(i)
    return scipy.sparse.coo_matrix((data, (i, j)), shape=(N, N))


def imag_to_coo(position: np.ndarray, n_neighbors: int):
    """
    Create a displacement matrix from a complex coordinate matrix of agents.

    Args:
        position: Complex coordinate matrix of agents
        n_neighbors: Number of neighbors for each agent

    Returns:
        A scipy coo_matrix of shape (N, N) where each element is the displacement of the agent with its neighbor if two agents are neighbors and zero otherwise
    """

    position = position.squeeze()
    N = position.size
    A = position[:,None] - position[None,:]
    vals = np.partition(np.abs(A), n_neighbors+1, axis=1)[:,n_neighbors+1]
    mask = np.abs(A) < vals[:,None]
    return scipy.sparse.coo_matrix(mask * A)


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
    scenarios = {"uniform", "gaussian_uniform"}

    def __init__(self, n_agents=100, width=10, scenario="uniform"):
        self.n_agents = n_agents
        self.n_targets = n_agents

        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario {scenario} is not a valid scenario. Possibilities {self.scenarios}."
            )
        self.scenario = scenario

        # Since space is 2D scale is inversely proportional to sqrt of the number of agents

        self.dt = 0.1
        # self.width = 1.0 * np.sqrt(self.n_agents)
        self.width = width
        self.reward_cutoff = 0.2
        self.reward_sigma = 0.1

        # agent properties
        self.max_accel = 0.5
        self.agent_radius = 0.1
        self.n_observed_agents = 3
        self.n_observed_targets = 3

        # comm graph properties
        self.n_neighbors = 3

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
            self.state_ndim / 2 + self.n_observed_targets * 2 + self.n_observed_agents * 2
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
        dist = cdist(self.position, self.position)
        idx = argtopk(-dist, self.n_neighbors + 1, axis=1)
        return index_to_coo(idx)

    def clip_action(self, action):
        """
        Clip action to a unit circle with radius self.max_accel.
        Args:
            action: An array of shape (..., 2) representing the action for each agent.
        """
        action = action.copy()
        magnitude = np.linalg.norm(action, axis=-1)
        to_clip = magnitude > self.max_accel
        action[to_clip] = action[to_clip] / magnitude[to_clip, None] * self.max_accel
        return action

    def centralized_policy(self):
        distance = cdist(self.position, self.target_positions)
        row_idx, col_idx = linear_sum_assignment(distance)
        assert (row_idx == np.arange(self.n_agents)).all()
        action = self.target_positions[col_idx] - self.position[row_idx]
        action = self.clip_action(action)
        assert action.shape == self.action_space.shape  # type: ignore
        return action

    def decentralized_policy(self, hops=0):
        observed_targets = self._observed_targets()
        observed_agents = self._observed_agents()
        if hops == 0:
            action = observed_targets[:, 0, :] - self.position
        elif hops == 1:
            action = np.zeros(self.action_space.shape)  # type: ignore
            for i in range(self.n_agents):
                agent_positions = np.concatenate(
                    (self.position[i, None, :], observed_agents[i]), axis=0
                )
                target_positions = observed_targets[i]
                distances = cdist(agent_positions, target_positions)
                row_idx, col_idx = linear_sum_assignment(distances)
                assignment = col_idx[np.nonzero(row_idx == 0)]
                if len(assignment) > 0:
                    action[i] = target_positions[assignment] - agent_positions[0]
        else:
            raise NotImplementedError("Hops > 1 not implemented.")
        action = self.clip_action(action)
        assert action.shape == self.action_space.shape  # type: ignore
        return action

    def _observed_agents(self):
        dist = cdist(self.position, self.position)
        idx = argtopk(-dist, self.n_observed_agents + 1, axis=1)
        idx = idx[:, 1:]  # remove self
        return self.position[idx] - self.position[:,None,:]

    def _observed_targets(self):
        dist = cdist(self.position, self.target_positions)
        idx = argtopk(-dist, self.n_observed_targets, axis=1)
        return self.target_positions[idx, :] - self.position[:,None,:]

    def _observation(self):
        tgt = self._observed_targets().reshape(self.n_agents, -1)
        agt = self._observed_agents().reshape(self.n_agents, -1)
        obs = np.concatenate((self.state[:,2:], tgt, agt), axis=1)
        assert obs.shape == self.observation_space.shape  # type: ignore
        return obs

    def _done(self) -> bool:
        too_far_gone = (np.abs(self.position) > self.width).any(axis=1).all(axis=0)
        return too_far_gone

    def _reward(self):
        dist = cdist(self.target_positions, self.position)
        idx = argtopk(-dist, 1, axis=1).squeeze()
        d = dist[np.arange(len(idx)), idx]
        reward = np.exp(-((d / self.reward_sigma) ** 2))
        reward[d > self.reward_cutoff] = 0
        return reward.mean()
    
    def coverage(self):
        dist = cdist(self.target_positions, self.position)
        return np.mean(np.any(dist < 0.1*np.ones_like(dist), axis=1))

    def step(self, action):
        assert action.shape == self.action_space.shape  # type: ignore
        action = self.clip_action(action)
        self.velocity = action
        self.position += self.velocity * self.dt
        self.t += self.dt
        return self._observation(), self._reward(), self._done(), {}

    def reset(self):
        self.state = np.zeros((self.n_agents, self.state_ndim))
        if self.scenario == "uniform":
            self.target_positions = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_targets, 2)
            )
            self.position = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_agents, 2)
            )
        elif self.scenario == "gaussian_uniform":
            # agents are normally distributed around the origin
            # targets are uniformly distributed
            self.target_positions = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_targets, 2)
            )
            self.position = rng.normal(size=(self.n_agents, 2))
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Should be one of {self.scenarios}."
            )

        self.t = 0
        if self.render_ is not None:
            self.render_.reset()
        return self._observation()

    def render(self, mode="human"):
        if self.render_ is None or self.render_.mode != mode:
            self.render_ = MotionPlanningRender(self.width, self.state_ndim, mode=mode)

        return self.render_.render(
            self.target_positions.T,
            self.position.T,
            self._reward(),
            self.coverage(),
            self._observed_targets(),
            self.adjacency(),
        )

    def close(self):
        pass


class ComplexMotionPlanning(GraphEnv):
    metadata = {"render.modes": ["human"]}
    scenarios = {"uniform", "gaussian_uniform"}

    def __init__(self, n_agents=100, width=10, scenario="uniform"):
        self.n_agents = n_agents
        self.n_targets = n_agents

        if scenario not in self.scenarios:
            raise ValueError(
                f"Scenario {scenario} is not a valid scenario. Possibilities {self.scenarios}."
            )
        self.scenario = scenario

        # Since space is 2D scale is inversely proportional to sqrt of the number of agents

        self.dt = 0.1
        # self.width = 1.0 * np.sqrt(self.n_agents)
        self.width = width
        self.reward_cutoff = 0.2
        self.reward_sigma = 0.1

        # agent properties
        self.max_accel = 0.5
        self.agent_radius = 0.1
        self.n_observed_agents = 3
        self.n_observed_targets = 3

        # comm graph properties
        self.n_neighbors = 3

        self._n_nodes = self.n_agents

        self.invariant_act_ndim = 1
        self.equivariant_act_ndim = 1
        self._action_ndim = self.invariant_act_ndim + self.equivariant_act_ndim
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.n_agents,
                self.action_ndim,
            ),
        )

        self.state_ndim = 2
        self.invariant_obs_ndim = int(self.state_ndim / 2)
        self.equivariant_obs_ndim = self.n_observed_targets + self.n_observed_agents
        self._observation_ndim = self.invariant_obs_ndim + self.equivariant_obs_ndim
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
        return np.expand_dims(self.state[..., 0], axis=1)

    @position.setter
    def position(self, value):
        self.state[..., 0] = np.squeeze(value)

    @property
    def velocity(self):
        return np.expand_dims(self.state[..., 1], axis=1)

    @velocity.setter
    def velocity(self, value):
        self.state[..., 1] = np.squeeze(value)

    def adjacency(self):
        return imag_to_coo(self.position, self.n_neighbors)

    def clip_action(self, action):
        """
        Clip action to a unit circle with radius self.max_accel.
        Args:
            action: An array of shape (..., 1) representing the action for each agent.
        """
        action = action.copy()
        magnitude = np.abs(action)
        to_clip = magnitude > self.max_accel
        action[to_clip] = action[to_clip] / magnitude[to_clip] * self.max_accel
        return action

    def centralized_policy(self):
        # Create both equivariant and invariant actions (the model will only do anything to the invariant actions, but we still need equivariant actions to train on)
        distance = idist(self.position, self.target_positions)
        row_idx, col_idx = linear_sum_assignment(distance)
        assert (row_idx == np.arange(self.n_agents)).all()
        equivariant_action = self.target_positions[col_idx] - self.position[row_idx]
        equivariant_action = self.clip_action(equivariant_action)
        invariant_action = np.abs(equivariant_action)
        assert np.concatenate([equivariant_action, invariant_action], axis=1).shape == self.action_space.shape  # type: ignore
        return equivariant_action, invariant_action

    def decentralized_policy(self, hops=0):
        observed_targets = self._observed_targets()
        observed_agents = self._observed_agents()
        if hops == 0:
            action = observed_targets[:, 0, :] - self.position
        elif hops == 1:
            action = np.zeros(self.action_space.shape)  # type: ignore
            for i in range(self.n_agents):
                agent_positions = np.concatenate(
                    (self.position[i, None, :], observed_agents[i]), axis=0
                )
                target_positions = observed_targets[i]
                distances = idist(agent_positions, target_positions)
                row_idx, col_idx = linear_sum_assignment(distances)
                assignment = col_idx[np.nonzero(row_idx == 0)]
                if len(assignment) > 0:
                    action[i] = target_positions[assignment] - agent_positions[0]
        else:
            raise NotImplementedError("Hops > 1 not implemented.")
        action = self.clip_action(action)
        assert action.shape == self.action_space.shape  # type: ignore
        return action

    def _observed_agents(self):
        dist = idist(self.position, self.position)
        idx = argtopk(-dist, self.n_observed_agents + 1, axis=1)
        idx = idx[:, 1:]  # remove self
        return self.position[idx] - self.position[:,None,:]

    def _observed_targets(self):
        dist = idist(self.position, self.target_positions)
        idx = argtopk(-dist, self.n_observed_targets, axis=1)
        return self.target_positions[idx, :] - self.position[:,None,:]

    def _observation(self):
        # Create both invariant and equivariant observations
        tgt = self._observed_targets().reshape(self.n_agents, -1)
        agt = self._observed_agents().reshape(self.n_agents, -1)
        equivariant_obs = np.concatenate((tgt, agt), axis=1)
        invariant_obs = np.abs(self.velocity)
        assert np.concatenate([equivariant_obs, invariant_obs], axis=1).shape == self.observation_space.shape  # type: ignore
        return equivariant_obs, invariant_obs

    def _done(self) -> bool:
        too_far_gone = (np.abs(self.position) > self.width * np.sqrt(2)).all()
        return too_far_gone

    def _reward(self):
        dist = idist(self.target_positions, self.position)
        idx = argtopk(-dist, 1, axis=1).squeeze()
        d = dist[np.arange(len(idx)), idx]
        reward = np.exp(-((d / self.reward_sigma) ** 2))
        reward[d > self.reward_cutoff] = 0
        return reward.mean()
    
    def coverage(self):
        dist = idist(self.target_positions, self.position)
        return np.mean(np.any(dist < 0.1*np.ones_like(dist), axis=1))

    def step(self, action):
        if isinstance(action, tuple):
            action = np.concatenate([action[0], action[1]], axis=1)
        assert action.shape == self.action_space.shape  # type: ignore
        action = np.split(action, [self.equivariant_act_ndim], axis=1)[0]
        action = self.clip_action(action)
        self.velocity = action
        self.position += self.velocity * self.dt
        self.t += self.dt
        return self._observation(), self._reward(), self._done(), {}

    def reset(self):
        self.state = np.zeros((self.n_agents, self.state_ndim), dtype=np.complex128)
        if self.scenario == "uniform":
            target_positions = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_targets, 2)
            )
            position = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_agents, 2)
            )
        elif self.scenario == "gaussian_uniform":
            # agents are normally distributed around the origin
            # targets are uniformly distributed
            target_positions = rng.uniform(
                -self.width / 2, self.width / 2, (self.n_targets, 2)
            )
            position = rng.normal(size=(self.n_agents, 2))
        else:
            raise ValueError(
                f"Unknown scenario: {self.scenario}. Should be one of {self.scenarios}."
            )
        
        self.target_positions = np.expand_dims(target_positions[:,0] + 1j*target_positions[:,1], axis=1)
        self.position = np.expand_dims(position[:,0] + 1j*position[:,1], axis=1)

        self.t = 0
        if self.render_ is not None:
            self.render_.reset()
        return self._observation()

    def render(self, mode="human"):
        if self.render_ is None or self.render_.mode != mode:
            self.render_ = MotionPlanningRender(self.width, self.state_ndim, mode=mode)

        target_positions = np.concatenate((self.target_positions.real, self.target_positions.imag), axis=1)
        position = np.concatenate((self.position.real, self.position.imag), axis=1)

        return self.render_.render(
            target_positions.T,
            position.T,
            self._reward(),
            self.coverage(),
            self._observed_targets(),
            self.adjacency(),
        )

    def close(self):
        pass

if __name__ == "__main__":
    env = ComplexMotionPlanning(scenario="gaussian_uniform")
    n_steps = 200
    env.reset()
    for i in range(n_steps):
        action = env.centralized_policy()
        _, reward, done, _ = env.step(action)
        adj = env.adjacency()
        env.render()
        if done:
            break
