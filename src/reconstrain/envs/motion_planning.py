import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial import KDTree

rng = np.random.default_rng()


class UnlabeledMotionPlanning(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.n_goals = 10
        self.n_agents = 10
        self.dt = 0.1
        self.max_steps = 1000
        self.width = 100
        self.max_accel = 1.0
        self.agent_radius = 0.1
        self.comm_radius = 1.0

        self.state_ndim = 6
        self.action_space = gym.spaces.Box(
            low=-self.max_accel, high=self.max_accel, shape=(self.n_agents, 2)
        )
        self.observation_space = gym.spaces.Box(shape=(self.n_agents, self.state_ndim))

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
        distance_matrix = kdtree.sparse_distance_matrix(
            kdtree, self.comm_radius, output_type="coo_matrix"
        )
        return np.exp(-(distance_matrix ** 2))

    def step(self, action):
        acceleration = np.reshape(action, (2, self.n_agents))
        self.acceleration = acceleration
        self.velocity += acceleration * self.dt
        
        self.position += self.velocity * self.dt + acceleration * self.dt ** 2 / 2
        self.t += self.dt
        observation = (self.state, self.adjacency())

        d, _ = self.goal_tree.query(self.position, k=1, distance_upper_bound=self.agent_radius)
        reward = np.sum(d < self.agent_radius)
        done = self.t >= self.max_steps * self.dt
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.goal_positions = rng.uniform(0, self.width, size=(self.n_goals, 2))
        self.goal_tree = KDTree(self.goal_positions)
        self.state = np.zeros((self.state_ndim, self.n_agents))
        self.position = rng.normal(size=(self.n_agents, 2))
        self.t = 0

    def render(self, mode="human"):
        raise NotImplementedError()

    def close(self):
        pass
