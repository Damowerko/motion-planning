from reconstrain.envs.motion_planning import MotionPlanning
import numpy as np

env = MotionPlanning()

n_trials = 10

rewards = []
for i in range(n_trials):
    env.reset()
    done = False
    trial_rewards = []
    while not done:
        env.render()
        action = env.decentralized_policy(hops=0)
        obs, reward, done, info = env.step(action)
        trial_rewards.append(reward)
    trial_reward = np.mean(trial_rewards)
    print(f"Trial reward: {trial_reward:0.2f}")
    rewards.append(trial_reward)

rewards = np.asarray(rewards)
print(
    f"""
MEAN: {rewards.mean():.2f}
STD: {rewards.std():.2f}
"""
)
