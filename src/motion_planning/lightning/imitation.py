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
        target_policy: str = "c",
        expert_probability: float = 0.5,
        expert_probability_decay: float = 0.99,
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
        self.expert_probability = expert_probability
        self.expert_probability_decay = expert_probability_decay
        self.automatic_optimization = False

    def training_step(self, data_pair, *args):
        data, next_data = data_pair
        opt_actor, opt_critic = self.optimizers()

        assert data.expert is not None

        # actor step
        mu = self.model.forward_actor(self.model.actor, data)
        loss_actor = F.mse_loss(mu, data.expert)
        self.log(
            "train/actor_loss", loss_actor, prog_bar=True, batch_size=data.batch_size
        )
        opt_actor.zero_grad()
        self.manual_backward(loss_actor)
        opt_actor.step()

    def validation_step(self, data_pair, *args):
        data, next_data = data_pair
        mu = self.model.forward_actor(self.model.actor, data)
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
        mu = self.model.forward_actor(self.model.actor, data)
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
        expert = self.env.baseline_policy(self.target_policy)
        data.expert = torch.as_tensor(expert, dtype=self.dtype, device=self.device)  # type: ignore
        if self.use_expert:
            # use expert policy
            data.action = data.expert
        elif self.training:
            # stochastic policy from actor for trai
            mu = self.model.forward_actor(self.model.actor, data)
            data.action = self.policy(mu)
        else:
            # use greedy policy
            data.action = self.clip_action(
                self.model.forward_actor(self.model.actor, data)
            )
        return data
