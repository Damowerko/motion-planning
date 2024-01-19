import gym
from stable_baselines3.common.env_util import make_vec_env

from .base import *


class MotionPlanningGPG(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        entropy_weight: float = 0.001,
        n_envs: int = 1,
        render: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.entropy_weight = entropy_weight
        self.n_envs = n_envs
        self.render = render > 0

        env = gym.make("motion-planning-v0")
        self.env = env
        self.vec_env = make_vec_env("motion-planning-v0", n_envs=n_envs)

        # disable automatic optimization since we only take one step every epoch
        self.automatic_optimization = False

    def forward(self, *args, **kwargs):
        return self.actor(*args, **kwargs)

    def training_step(self, data, batch_idx):
        policy_loss = -(data.R[:, None, None] * log_prob).mean()
        entropy_loss = -self.entropy_weight * entropy.mean()
        loss = policy_loss + entropy_loss
        self.manual_backward(loss)
        self.log_dict(
            {
                "train/loss": loss,
                "train/policy_loss": policy_loss,
                "train/entropy_loss": entropy_loss,
                "train/reward": reward.mean(),
                "train/sigma": sigma.mean(),
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return policy_loss

    def training_epoch_end(self, outputs):
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()

    def validation_step(self, batch, batch_idx):
        log_prob, entropy, reward, R, sigma = batch
        self.log("val/reward", reward.mean(), prog_bar=True)
        self.log("val/metric", -reward.mean())

    def test_step(self, batch, batch_idx):
        log_prob, entropy, reward, R, sigma = batch
        self.log("test/reward", reward.mean())

    def batch_generator(self, render=False):
        episode = self.rollout(render)

        # compute the discoutned cost to go
        rewards = torch.stack([data.reward for data in episode], axis=0)
        R = torch.zeros_like(rewards)
        R[-1] = 0
        for i in range(1, len(rewards)):
            R[-i - 1] = rewards[-i] + self.gamma * R[-i]
        # baseline
        R = R - R.mean()
        for i in range(len(episode)):
            episode[i].R = R[i]

        yield Batch.from_data_list(episode)

    def train_dataloader(self):
        return self._dataloader(render=False)

    def val_dataloader(self):
        return self._dataloader(render=self.render)

    def test_dataloader(self):
        return self._dataloader(render=self.render)
