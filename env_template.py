import gymnasium as gym
from gym import spaces


class MyEnv(gym.Env):
    def __init__(self):
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.action_space = spaces.Discrete(2)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        # Update the environment state based on the action
        # ...

        # Return the next observation, reward, done flag, and info
        observation = self.observation_space.sample()
        reward = 1.0
        done = False
        info = {}
        return observation, reward, done, info


# Interacting with the environment
#observation = env.reset()
#while True:
#    action = env.action_space.sample()
#    observation, reward, done, info = env.step(action)
#    print(f"Observation: {observation}, Reward: {reward}, Done: {done}")
#    if done:
#        observation = env.reset()
