import gym
import numpy as np
import torch
from image_utils import random_augment_color, apply_augment_fns


class ToTorchTensors(gym.ObservationWrapper):
    def __init__(self, env=None, device='cpu'):
        super(ToTorchTensors, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)
        self.device = device

    def observation(self, observation):
        tensor = torch.from_numpy(np.rollaxis(observation, 2)).to(self.device)
        tensor = tensor.float() / 127.5 - 1.0
        return tensor


class ToTorchTensorsWithAug(gym.ObservationWrapper):
    def __init__(self, env=None, device='cpu'):
        super(ToTorchTensorsWithAug, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)
        self.device = device
        self.aug_fns = random_augment_color(np.zeros(obs_shape), return_fn=True)

    def observation(self, observation):
        cache_stats = observation[0:10, :, :].copy()
        observation = apply_augment_fns(observation, self.aug_fns, out_type=np.uint8)
        observation[0:10, :, :] = cache_stats
        tensor = torch.from_numpy(np.rollaxis(observation, 2)).to(self.device)
        tensor = tensor.float() / 127.5 - 1.0
        return tensor

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.aug_fns = random_augment_color(observation, return_fn=True)
        return self.observation(observation)


class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
