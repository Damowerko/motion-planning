import gym
import reconstrain
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

if __name__ == "__main__":
    env = make_vec_env("motion-planning-v0", n_envs=32, vec_env_cls=SubprocVecEnv)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=32000)

    env = gym.make("motion-planning-v0")
    obs = env.reset()
    for _ in range(10000):
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if done:
            break
        env.render()
