import numpy as np
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import IterableDataset, DataLoader
from typing import Callable, Iterator
import pytorch_lightning as pl

from learner.state_with_delay import MultiAgentStateWithDelay
from learner.replay_buffer import ReplayBuffer
from learner.replay_buffer import Transition
from learner.actor import Actor
import gym
import gym_flock


class ExperienceSourceDataset(IterableDataset):
    """Basic experience source dataset.
    Takes a generate_batch function that returns an iterator. The logic for the experience source and how the batch is
    generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable) -> None:
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


# TODO: how to deal with bounded/unbounded action spaces?? Should I always assume bounded actions?
class ImitationLearning(pl.LightningModule):
    def __init__(
        self,
        env_name: str,
        n_states: int,
        n_actions: int,
        k: int,
        n_agents: int,
        hidden_size=32,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 10000,
        updates_per_step: int = 1,
        n_train_episodes: int = 1000,
        n_test_episodes: int = 100,
    ):
        """
        Initialize the DDPG networks.
        :param device: CUDA device for torch
        :param args: experiment arguments
        """
        super().__init__()

        self.save_hyperparameters()

        self.env = gym.make(env_name)
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.k = k

        hidden_layers = [hidden_size, hidden_size]
        ind_agg = 0  # int(len(hidden_layers) / 2)  # aggregate halfway

        # Define Networks
        self.actor = Actor(n_states, n_actions, hidden_layers, k, ind_agg)
        self.state = MultiAgentStateWithDelay(
            self.device, n_states, n_agents, ind_agg, self.env.reset(), prev_state=None
        )

    def configure_optimizers(self):
        return Adam(self.actor.parameters(), lr=self.hparams.lr)

    def select_action(self, state):
        """
        Evaluate the Actor network over the given state, and with injection of noise.
        :param state: The current state.
        :param graph_shift_op: History of graph shift operators
        :param action_noise: The action noise
        :return:
        """
        self.actor.eval()  # Switch the actor network to Evaluation Mode.
        mu = self.actor(state.delay_state, state.delay_gso)  # .to(self.device)

        # mu is (B, 1, nA, N), need (N, nA)
        mu = mu.permute(0, 1, 3, 2)
        mu = mu.view((self.n_agents, self.n_actions))

        self.actor.train()  # Switch back to Train mode.
        mu = mu.data
        return mu

        # return mu.clamp(-1, 1)  # TODO clamp action to what space?

    def training_step(self, batch, batch_idx):
        """
        Take a gradient step given a batch of sampled transitions.
        :param batch: The batch of training samples.
        :return: The loss function in the network.
        """
        delay_gso_batch = Variable(
            torch.cat(tuple([s.delay_gso for s in batch.state]))
        ).to(self.device)
        delay_state_batch = Variable(
            torch.cat(tuple([s.delay_state for s in batch.state]))
        ).to(self.device)
        actor_batch = self.actor(delay_state_batch, delay_gso_batch)
        optimal_action_batch = Variable(torch.cat(batch.action)).to(self.device)
        loss = F.mse_loss(actor_batch, optimal_action_batch)
        self.log("loss", loss)
        return loss

    def generate_episode(self):
        state = MultiAgentStateWithDelay(
            self.device,
            self.n_states,
            self.n_agents,
            self.k,
            self.env.reset(),
            prev_state=None,
        )
        done = False
        while not done:
            optimal_action = self.env.controller()
            next_state, reward, done, _ = self.step(optimal_action)
            next_state = MultiAgentStateWithDelay(
                self.device,
                self.n_states,
                self.n_agents,
                self.k,
                next_state,
                prev_state=state,
            )
            # action = torch.Tensor(action)
            notdone = torch.Tensor([not done]).to(self.device)
            reward = torch.Tensor([reward]).to(self.device)

            # action is (N, nA), need (B, 1, nA, N)
            action = torch.Tensor(optimal_action).to(self.device)
            action = action.transpose(1, 0)
            action = action.reshape((1, 1, self.n_actions, self.n_agents))
            self.memory.insert(Transition(state, action, notdone, next_state, reward))
            state = next_state
        return self.memory

    def train_batch(self):
        memory = self.generate_episode(self.env)
        for _ in range(self.hparams.updates_per_step):
            transitions = memory.sample(self.hparams.batch_size)
            yield Transition(*zip(*transitions))

    def train_dataloader(self) -> DataLoader:
        self.memory = ReplayBuffer(max_size=self.hparams.buffer_size)
        dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)

    def test_step(self, batch, batch_idx):
        test_rewards = []
        for _ in range(n_test_episodes):
            ep_reward = 0
            state = MultiAgentStateWithDelay(device, args, env.reset(), prev_state=None)
            done = False
            while not done:
                action = learner.select_action(state)
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                next_state = MultiAgentStateWithDelay(
                    device, args, next_state, prev_state=state
                )
                ep_reward += reward
                state = next_state
            test_rewards.append(ep_reward)

    def save_model(self, env_name, suffix="", actor_path=None):
        """
        Save the Actor Model after training is completed.
        :param env_name: The environment name.
        :param suffix: The optional suffix.
        :param actor_path: The path to save the actor.
        :return: None
        """
        if not os.path.exists("models/"):
            os.makedirs("models/")

        if actor_path is None:
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        print("Saving model to {}".format(actor_path))
        torch.save(self.actor.state_dict(), actor_path)

    def load_model(self, actor_path):
        """
        Load Actor Model from given paths.
        :param actor_path: The actor path.
        :return: None
        """
        print("Loading model from {}".format(actor_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path).to(self.device))


def train_cloning(env, args, device):
    debug = args.getboolean("debug")

    learner = ImitationLearning(device, args)

    rewards = []
    total_numsteps = 0
    updates = 0

    stats = {"mean": -1.0 * np.Inf, "std": 0}

    for i in range(n_train_episodes):
        if i % test_interval == 0:
            pass

    env.close()
    return stats
