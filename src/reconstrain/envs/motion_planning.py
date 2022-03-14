from time import sleep
from typing import Optional, Union

import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from gym import spaces
from scipy.spatial import KDTree
import networkx as nx

rng = np.random.default_rng()


class MotionPlanningRender:
    def __init__(self, width):
        self.positions = []
        self.width = width
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.reset()
        # plt.show(block=False)


    def reset(self):
        self.ax.clear()
        self.target_scatter = None
        self.agent_scatter = None

    def render(self, goal_positions, agent_positions, reward, observation):
        self.reset()
        if self.target_scatter is None:
            self.target_scatter = self.ax.plot(
                *goal_positions, "rx"
            )[0]
        if self.agent_scatter is None:
            self.agent_scatter = self.ax.plot(
                *agent_positions, "bo"
            )[0]

        observation, adjacency = observation
        G = nx.from_scipy_sparse_array(adjacency)
        nx.draw_networkx_edges(G, pos=agent_positions.T, ax=self.ax)
        targets = observation[:, 6:].reshape(-1, 3, 2).reshape(-1, 2)
        self.ax.plot(*targets.T, "y^")

        self.ax.set_title(f"Reward: {reward:.2f}")
        self.agent_scatter.set_data(*agent_positions)
        self.ax.set_xlim(-self.width, self.width)
        self.ax.set_ylim(-self.width, self.width)
        self.fig.canvas.flush_events()
        self.fig.canvas.draw_idle()


class MotionPlanning(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.n_targets = 10
        self.n_agents = 10
        self.dt = 0.1
        self.max_steps = 1000
        self.width = 100
        self.max_accel = 1.0
        self.agent_radius = 1
        self.comm_radius = 10
        self.n_neighbors = 3
        self.n_observed_targets = 3
        

        self.action_ndim = 2
        self.action_space = spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(self.n_agents, self.action_ndim)
        )

        self.state_ndim = 6
        self.observation_ndim = self.state_ndim + self.n_observed_targets * 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, self.observation_ndim))

        self.render_: Optional[MotionPlanningRender] = None
        self.reset()

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

    @property
    def acceleration(self):
        return self.state[..., 4:6]

    @acceleration.setter
    def acceleration(self, value):
        self.state[..., 4:6] = value

    def adjacency(self):
        kdtree = KDTree(self.position)
        # distance_matrix = kdtree.sparse_distance_matrix(
        #     kdtree, self.comm_radius, output_type="coo_matrix"
        # )
        # distance_matrix.data = np.exp(-(distance_matrix.data ** 2))
        _, idx = kdtree.query(self.position, k=self.n_neighbors)
        adj = scipy.sparse.dok_array((self.n_agents, self.n_agents))
        for i in range(self.n_agents):
            adj[i, i] = 1
            for j in idx[i]:
                adj[i, j] = 1
        return adj

    def observe(self):
        _, idx = self.target_tree.query(self.position, k=self.n_observed_targets)
        closest_tagets = self.target_positions[idx, :].reshape(self.n_agents, -1)
        observation = np.concatenate((self.state, closest_tagets), axis=1)
        return observation, self.adjacency()

    def step(self, action):
        self.acceleration = np.reshape(action, (self.n_agents, 2))
        if np.linalg.norm(self.acceleration) > self.max_accel:
            self.acceleration *= self.max_accel / np.linalg.norm(self.acceleration)
        self.velocity += self.acceleration * self.dt
        self.position += self.velocity * self.dt + self.acceleration * self.dt ** 2 / 2
        self.t += self.dt
        observation = self.observe()

        d, _ = self.target_tree.query(
            self.position, k=1, distance_upper_bound=self.agent_radius
        )
        self.reward = np.sum(d < self.agent_radius)

        timeout = self.t >= self.max_steps * self.dt
        too_far_gone = (np.abs(self.position) > self.width * 2).any(axis=1).all(axis=0)
        done = timeout or too_far_gone

        info = {}
        
        return observation, self.reward, done, info

    def reset(self):
        self.target_positions = rng.uniform(-self.width, self.width, size=(self.n_targets, 2))
        self.target_tree = KDTree(self.target_positions)

        self.state = np.zeros((self.n_agents, self.state_ndim))
        self.position = rng.normal(scale=10, size=(self.n_agents, 2))

        self.t = 0
        if self.render_ is not None:
            self.render_.reset()
        return self.observe()

    def render(self, mode="human"):
        assert mode == "human"
        if self.render_ is None:
            self.render_ = MotionPlanningRender(self.width)
        self.render_.render(self.target_positions.T, self.position.T, self.reward, self.observe())

    def close(self):
        pass
