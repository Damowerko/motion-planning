import random
from collections import deque
from reconstrain.models.base import *
from reconstrain.rl import ReplayBuffer


@auto_args
class MotionPlanningImitation(MotionPlanningActorCritic):
    def __init__(
        self,
        buffer_size: int = 10000,
        target_policy: str = "c",
        render: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.target_policy = target_policy
        self.render = render > 0
        self.buffer = ReplayBuffer[BaseData](buffer_size)

    def training_step(self, data, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            mu, _ = self.actor.forward(data.state, data.edge_index, data.edge_attr)
            loss = F.mse_loss(mu.view(data.expert.shape), data.expert)
            self.log("train/actor_loss", loss, prog_bar=True)
            return loss
        elif optimizer_idx == 1:
            q = self.critic.forward(
                data.state, data.action, data.edge_index, data.edge_attr
            )
            with torch.no_grad():
                muprime, _ = self.actor.forward(
                    data.next_state, data.edge_index, data.edge_attr
                )
                qprime = self.critic(
                    data.next_state, muprime, data.edge_index, data.edge_attr
                )
            loss = self.critic_loss(q, qprime, data.reward, data.done)
            self.log("train/critic_loss", loss, prog_bar=True)
            return loss

    def validation_step(self, data, batch_idx):
        yhat, _ = self.actor(data.state, data.edge_index, data.edge_attr)
        loss = F.mse_loss(yhat.view(data.expert.shape), data.expert)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/reward", data.reward.mean(), prog_bar=True)
        self.log("val/metric", -data.reward.mean())
        return loss

    def test_step(self, data, batch_idx):
        yhat, _ = self.actor(data.state, data.edge_index, data.edge_attr)
        loss = F.mse_loss(yhat.view(data.expert.shape), data.expert)
        self.log("test/loss", loss)
        self.log("test/reward", data.reward.mean())
        return loss

    def before_rollout_step(self, data: BaseData):
        if self.target_policy == "c":
            expert = self.env.centralized_policy()
        elif self.target_policy == "d0":
            expert = self.env.decentralized_policy(0)
        elif self.target_policy == "d1":
            expert = self.env.decentralized_policy(1)
        else:
            raise ValueError(f"Unknown target policy {self.target_policy}")
        data.expert = torch.as_tensor(expert, dtype=self.dtype, device=self.device)  # type: ignore
        return data

    def batch_generator(self, n_episodes=1, render=False, use_buffer=True):
        data = []
        for _ in range(n_episodes):
            data.extend(self.rollout(render=render))
        # no point in not using the validation data in training
        self.buffer.extend(data)
        if use_buffer:
            data = self.buffer.collect(shuffle=True)
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(n_episodes=1, render=False, use_buffer=True)

    def val_dataloader(self):
        return self._dataloader(n_episodes=10, render=self.render, use_buffer=False)

    def test_dataloader(self):
        return self._dataloader(n_episodes=100, render=self.render, use_buffer=False)
