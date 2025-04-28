# procgen_env.py
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
import gym
import cv2
from PIL import Image
import procgen

class ProcGenEnv(VecEnvWrapper):
    def __init__(self, task, num_envs=1, size=(64, 64), resize='pillow', **kwargs):
        """
        ProcGen environment wrapper that resizes observations to the specified size.

        Args:
            task (str): The name of the ProcGen task, e.g., 'coinrun'.
            num_envs (int): Number of parallel environments.
            size (tuple): Desired image observation size, e.g., (64, 64).
            resize (str): Method for resizing the images ('opencv' or 'pillow').
            kwargs: Additional keyword arguments for ProcGen.
        """
        self.size = size
        self.resize = resize
        self.env = procgen.ProcgenEnv(num_envs=num_envs, env_name=task, **kwargs)
        super().__init__(self.env)
    
    def reset(self):
        obs = self.env.reset()
        return self._resize_obs(obs)

    def step_async(self, actions):
        self.env.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.env.step_wait()
        return self._resize_obs(obs), rewards, dones, infos

    def _resize_obs(self, obs):
        resized_obs = []
        for img in obs:
            if self.resize == 'opencv':
                img_resized = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            elif self.resize == 'pillow':
                img_resized = Image.fromarray(img)
                img_resized = img_resized.resize(self.size, Image.BILINEAR)
                img_resized = np.array(img_resized)
            else:
                raise ValueError("Resize method must be 'opencv' or 'pillow'")
            resized_obs.append(img_resized)
        return np.array(resized_obs)
    
    @property
    def observation_space(self):
        obs_space = self.env.observation_space
        obs_space.spaces['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(*self.size, 3), dtype=np.uint8
        )
        return obs_space

    @property
    def action_space(self):
        return self.env.action_space