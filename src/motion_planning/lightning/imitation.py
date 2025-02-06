import random

from motion_planning.lightning.base import *
from motion_planning.rl import ReplayBuffer


class MotionPlanningImitation(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        model: ActorCritic,
        buffer_size: int = 10_000,
        target_policy: str = "c",
        expert_probability: float = 0.5,
        expert_probability_decay: float = 0.99,
        render: bool = False,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            target_policy_distance_squared: whether to use the squared distance assignment in the target policy
            expert_probability: probability of sampling from the expert
            render: whether to render the environment
        """
        super().__init__(model, **kwargs)
        self.save_hyperparameters(ignore=["model"])
        self.target_policy = target_policy
        self.render = render > 0
        self.buffer = ReplayBuffer[tuple[Data, Data]](buffer_size)
        self.expert_probability = expert_probability
        self.expert_probability_decay = expert_probability_decay
        self.automatic_optimization = False

    def training_step(self, data_pair, *args):
        data, next_data = data_pair
        opt_actor, opt_critic = self.optimizers()

        assert data.expert is not None

        # actor step
        mu = self.model.forward_actor(data)
        loss_actor = F.mse_loss(mu, data.expert)
        self.log(
            "train/actor_loss", loss_actor, prog_bar=True, batch_size=data.batch_size
        )
        opt_actor.zero_grad()
        self.manual_backward(loss_actor)
        opt_actor.step()

        # critic step
        # q = self.model.forward_critic(data.action, data)
        # with torch.no_grad():
        #     # we get the expected action, since we want the critic to predict the expected reward
        #     next_action = self.model.forward_actor(next_data)
        #     next_q = self.model.forward_critic(next_action, next_data)
        # loss_critic = self.critic_loss(q, next_q, data.reward, data.done)
        # self.log(
        #     "train/critic_loss", loss_critic, prog_bar=True, batch_size=data.batch_size
        # )
        # opt_critic.zero_grad()
        # self.manual_backward(loss_critic)
        # opt_critic.step()

    def validation_step(self, data_pair, *args):
        data, next_data = data_pair
        mu = self.model.forward_actor(data)
        loss = F.mse_loss(mu, data.expert)
        self.log("val/actor_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )
        self.log("val/coverage", data.coverage.mean(), batch_size=data.batch_size)
        self.log(
            "val/n_collisions", data.n_collisions.mean(), batch_size=data.batch_size
        )

        return loss

    def test_step(self, data_pair, *args):
        data, next_data = data_pair
        mu = self.model.forward_actor(data)
        loss = F.mse_loss(mu, data.expert)
        self.log("test/actor_loss", loss, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "test/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss

    def rollout_start(self):
        # decide if expert will be used for rollout
        if self.training:
            expert_probability = (
                self.expert_probability
                * self.expert_probability_decay**self.current_epoch
            )
            self.use_expert = random.random() < expert_probability
        else:
            self.use_expert = False

    def rollout_action(self, data: Data):
        if self.target_policy in ["c", "c_sq"]:
            expert = self.env.centralized_policy(self.target_policy == "c_sq")
        elif self.target_policy == "d0":
            expert = self.env.decentralized_policy(0)
        elif self.target_policy in ["d1", "d1_sq"]:
            expert = self.env.decentralized_policy(1, self.target_policy == "d1_sq")
        else:
            raise ValueError(f"Unknown target policy {self.target_policy}")

        data.expert = torch.as_tensor(expert, dtype=self.dtype, device=self.device)  # type: ignore

        if self.use_expert:
            # use expert policy
            data.action = data.expert
        elif self.training:
            # stochastic policy from actor for trai
            mu = self.model.forward_actor(data)
            data.action = self.policy(mu)
        else:
            # use greedy policy
            data.action = self.model.forward_actor(data)
        return data

    def batch_generator(
        self, n_episodes=1, render=False, use_buffer=True, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode, frames = self.rollout(render=render)
            data.extend(episode)
        if use_buffer:
            self.buffer.extend(data)
            data = self.buffer.collect(shuffle=True)
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, render=False, use_buffer=True, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, render=self.render, use_buffer=False, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=100, render=self.render, use_buffer=False, training=False
        )
