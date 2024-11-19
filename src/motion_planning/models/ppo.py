from copy import deepcopy
from pathlib import Path
import imageio.v3 as iio
import matplotlib.pyplot as plt
import wandb

from motion_planning.models.base import *
from motion_planning.rl import ReplayBuffer


def save_results(name: str, path: Path, frames: np.ndarray):
    """
    Args:
        path (Path): The path to save the summary to.
        rewards (np.ndarray): An ndarray of shape (n_trials, max_steps).
        frames (np.ndarray): An array of shape (n_trial, max_steps, H, W).
    """
    path.mkdir(parents=True, exist_ok=True)

    # make a single video of all trials
    iio.imwrite(path / f"{name}.mp4", np.concatenate(frames, axis=0), fps=30)


class MotionPlanningPPO(MotionPlanningActorCritic):
    @classmethod
    def add_model_specific_args(cls, group):
        return add_model_specific_args(cls, group)

    def __init__(
        self,
        ratio_clip: float = 0.2,
        early_stopping: int = 15,
        policy_delay: int = 2,
        pretrain: bool = False,
        render: bool = False,
        **kwargs,
    ):
        """
        Args:
            buffer_size: size of the replay buffer
            target_policy: the target policy to use for the expert
            expert_probability: probability of sampling from the expert
            render: whether to render the environment
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.render = render > 0
        self.ratio_clip = ratio_clip
        self.early_stopping = early_stopping
        self.policy_delay = policy_delay
        self.delay_count = 0
        self.automatic_optimization = False

        if pretrain:
            self.frozen_epochs = 100
            self.warmup_epochs = 50
        else:
            self.frozen_epochs = 0
            self.warmup_epochs = 0

    def configure_optimizers(self):
        return (
            torch.optim.AdamW(
                self.ac.actor.parameters(), lr=self.actor_lr, weight_decay=self.weight_decay
            ),
            torch.optim.AdamW(
                self.ac.value.parameters(),
                lr=self.critic_lr,
                weight_decay=self.weight_decay,
            ),
        )
    
    def update_optimizers(self):
        opt_actor, opt_value = self.optimizers()
        if self.current_epoch < self.frozen_epochs:
            for g in opt_actor.param_groups:
                g['lr'] = 0
            for g in opt_value.param_groups:
                g['lr'] = 2 * self.critic_lr
        elif self.current_epoch < self.frozen_epochs + self.warmup_epochs:
            warmed_rate = (self.current_epoch - self.frozen_epochs) / self.warmup_epochs
            for g in opt_actor.param_groups:
                g['lr'] = warmed_rate * self.actor_lr
            for g in opt_value.param_groups:
                g['lr'] = (2 - warmed_rate) * self.critic_lr
        else:
            for g in opt_actor.param_groups:
                g['lr'] = self.actor_lr
            for g in opt_value.param_groups:
                g['lr'] = self.critic_lr
        
    
    def value_loss(self, state, rtg, data):
        return F.mse_loss(self.ac.value(state, data), rtg)
    
    def actor_loss(self, state, centralized_state, action, advantage, log_prob, data):
        action_mu, action_sigma = self.ac.actor(state, data)
        _, logp = self.ac.policy(action_mu, action_sigma, action)
        ratio = torch.exp(logp - log_prob)
        clip_adv = torch.clip(ratio, 1 - self.ratio_clip, 1 + self.ratio_clip) * advantage[:,None]
        loss = -(torch.min(ratio * advantage[:,None], clip_adv)).mean()

        return loss

    def training_step(self, data, batch_idx):
        self.update_optimizers()
        opt_actor, opt_value = self.optimizers()

        if self.current_epoch >= self.frozen_epochs:
            # Update the actor function
            loss_pi = self.actor_loss(data.state, data.centralized_state, data.action, data.adv, data.log_prob, data)
            self.log("train/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
            opt_actor.zero_grad()
            self.manual_backward(loss_pi)
            torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1e-2)
            for name, param in self.ac.actor.named_parameters():
                self.log(f"train/actor_gradients/{name}", param.grad.mean(), batch_size=data.batch_size)
            opt_actor.step()

        # Update the value function
        loss_val = self.value_loss(data.state, data.rtg, data)
        self.log("train/value_loss", loss_val, prog_bar=True, batch_size=data.batch_size)
        opt_value.zero_grad()
        self.manual_backward(loss_val)
        torch.nn.utils.clip_grad_norm_(self.ac.value.parameters(), 1e-2)
        for name, param in self.ac.value.named_parameters():
            self.log(f"train/value_gradients/{name}", param.grad.mean(), batch_size=data.batch_size)
        opt_value.step()

    def validation_step(self, data, batch_idx):
        loss_pi = self.actor_loss(data.state, data.centralized_state, data.action, data.adv, data.log_prob, data)
        loss_val = self.value_loss(data.state, data.rtg, data)

        self.log("val/actor_loss", loss_pi, prog_bar=True, batch_size=data.batch_size)
        self.log("val/value_loss", loss_val, prog_bar=True, batch_size=data.batch_size)
        self.log(
            "val/reward", data.reward.mean(), prog_bar=True, batch_size=data.batch_size
        )

        return loss_val, loss_pi

    def test_step(self, data, batch_idx):
        loss_pi = self.actor_loss(data.state, data.centralized_state, data.action, data.adv, data.log_prob, data)
        loss_val = self.value_loss(data.state, data.rtg, data)

        self.log("test/actor_loss", loss_pi, batch_size=data.batch_size)
        self.log("test/value_loss", loss_val, batch_size=data.batch_size)
        self.log(
            "test/reward", data.reward.mean(), batch_size=data.batch_size
        )

        return loss_val, loss_pi

    def rollout_step(
        self,
        data: BaseData,
    ):
        # if self.training:
        #     # use actor policy
        #     eps = torch.randn_like(data.mu) * self.noise
        #     data.action = data.mu + torch.clip(eps, -self.noise_clip, self.noise_clip)
        #     data.action = self.clip_action(data.action)
        # else:
        #     # use greedy policy
        #     data.action = data.mu

        data.action = data.mu
        next_state, centralized_state, reward, done, _ = self.env.step(data.action.detach().cpu().numpy())
        return data, next_state, centralized_state, reward, done
    
    @torch.no_grad()
    def rollout(self, render=False) -> List[BaseData]:
        self.rollout_start()
        episode = []
        observation, centralized_state = self.env.reset()
        data = self.to_data(observation, centralized_state, self.env.adjacency())
        frames = []
        for _ in range(self.max_steps):
            if render:
                frames.append(self.env.render(mode="rgb_array"))

            # sample action
            data.mu, data.log_prob, data.value = self.ac.step(data.state, data)

            # take step
            data, next_state, centralized_state, reward, done = self.rollout_step(data)

            # add additional attributes
            next_data = self.to_data(next_state, centralized_state, self.env.adjacency())
            data.reward = torch.as_tensor(reward).to(device=self.device, dtype=self.dtype)  # type: ignore
            data.next_state = next_data.state
            data.next_centralized_state = next_data.centralized_state
            data.done = torch.tensor(done, dtype=torch.bool, device=self.device)  # type: ignore

            episode.append(data)
            data = next_data
            # if done or (len(episode) >= self.early_stopping and episode[-1].reward <= episode[-self.early_stopping].reward):
            if done:
                break
        
        for i in reversed(range(len(episode))):
            data = episode[i]
            if i == len(episode) - 1:
                data.rtg = data.reward.repeat(self.env.n_agents)
                data.adv = data.reward - data.value
            else:
                data.rtg = data.reward + self.gamma * episode[i+1].rtg
                data.adv = data.reward + self.gamma * episode[i+1].value - data.value + (self.gamma * self.lam) * episode[i+1].adv

        return episode, frames

    def batch_generator(
        self, n_episodes=1, render=False, training=True
    ):
        # set model to appropriate mode
        self.train(training)

        data = []
        for _ in range(n_episodes):
            episode, frames = self.rollout(render=render)
            data.extend(episode)
            # if render and self.current_epoch % 10 == 0:
            #     save_results(f"{self.current_epoch}", Path("figures") / "test_results", frames)
            #     wandb.save(f"figures/test_results/{self.current_epoch}.mp4")
        return iter(data)

    def train_dataloader(self):
        return self._dataloader(
            n_episodes=10, render=False, training=True
        )

    def val_dataloader(self):
        return self._dataloader(
            n_episodes=1, render=True, training=False
        )

    def test_dataloader(self):
        return self._dataloader(
            n_episodes=100, render=self.render, training=False
        )
