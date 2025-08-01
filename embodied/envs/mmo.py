import os, json, random
import numpy as np
import elements
import embodied
import cv2

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from social_dilemmas.envs.env_creator import get_env_creator


class SocialDilemma(embodied.Env):
    """
    Wrapper for sequential social dilemma games (harvest / cleanup / switch)
    to match the Dreamer-style Overcooked wrapper interface.

    - step(): controls all agents via a vector of discrete actions.
    - Observations: per-agent local views (flattened) as state_i + a global RGB image.
    - Reward: sum (or mean) of per-agent rewards.
    """

    def __init__(self,
                 task='harvest',
                 num_agents=2,
                 use_collective_reward=False,
                 inequity_averse_reward=False,
                 alpha=0.0,
                 beta=0.0,
                 aggregate_reward='sum',      # 'sum' or 'mean'
                 image_size=(64, 64),
                 seed=None):

        super().__init__()
        assert task in ('harvest', 'cleanup', 'switch'), task
        assert num_agents >= 1

        env_fn = get_env_creator(
            env=task,
            num_agents=num_agents,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
        self._env = env_fn(None)
        if seed is not None:
            self._env.seed(seed)

        # Initial reset so that .agents exists
        self._last_obs = self._env.reset()

        self.num_agents = num_agents
        self.env_name = task
        self.aggregate_reward = aggregate_reward
        self._image_size = image_size

        self._action_size = self._env.action_space.n
        try:
            agent0 = self._env.agents['agent-0']
            self._act_names = [agent0.action_map(i) for i in range(self._action_size)]
        except Exception:
            self._act_names = [f'act{i}' for i in range(self._action_size)]

        # Episode bookkeeping
        self._episode = 0
        self._reward_cum = 0.0
        self._done = True
        self._length = 0

    # ---------- Spaces ----------
    @property
    def act_names(self):
        return self._act_names

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (self.num_agents,), 0, self._action_size),
            'reset': elements.Space(bool),
        }

    @property
    def obs_space(self):
        # Use last observation to infer shapes.
        local = self._last_obs['agent-0']['curr_obs']
        flat_size = int(np.prod(local.shape))
        spaces = {
            'image': elements.Space(np.uint8, self._global_image().shape),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            'log/reward': elements.Space(np.float32),
        }
        for i in range(self.num_agents):
            spaces[f'state_{i}'] = elements.Space(np.float32, (flat_size,))
        return spaces

    # ---------- Core ----------
    def step(self, action):
        if action['reset'] or self._done:
            return self._reset()

        joint = {f'agent-{i}': int(a) for i, a in enumerate(action['action'])}
        obs, rew, dones, _ = self._env.step(joint)

        self._done = bool(dones.get('__all__', False))
        if self.aggregate_reward == 'sum':
            r = float(sum(rew.values()))
        elif self.aggregate_reward == 'mean':
            r = float(np.mean(list(rew.values())))
        else:
            raise ValueError(self.aggregate_reward)
        self._reward_cum += r
        self._length += 1

        self._last_obs = obs
        image = self._global_image()
        flat_locals = [
            obs[f'agent-{i}']['curr_obs'].astype(np.float32).flatten() / 255.0
            for i in range(self.num_agents)
        ]
        return self._make_obs(flat_locals, image, r, is_last=self._done)

    def _reset(self):
        obs = self._env.reset()
        self._last_obs = obs
        self._episode += 1
        self._reward_cum = 0.0
        self._length = 0
        self._done = False
        image = self._global_image()
        flat_locals = [
            obs[f'agent-{i}']['curr_obs'].astype(np.float32).flatten() / 255.0
            for i in range(self.num_agents)
        ]
        return self._obs(flat_locals, image, 0.0, is_first=True)

    # ---------- Helpers ----------
    def _global_image(self):
        world = self._env.world_map_color
        view = self._env.view_len
        trimmed = world[view:-view, view:-view]  # remove padding
        if self._image_size:
            trimmed = cv2.resize(trimmed, self._image_size, interpolation=cv2.INTER_NEAREST)
        return trimmed.astype(np.uint8)

    def _obs(self, flat_locals, image, reward, is_first=False, is_last=False):
        obs = {
            'image': image,
            'reward': np.float32(reward),
            'is_first': is_first,
            'is_last': is_last,
            'is_terminal': False,
            'log/reward': np.float32(self._reward_cum),
        }
        for i, arr in enumerate(flat_locals):
            obs[f'state_{i}'] = arr.astype(np.float32)
        if self.vlm is not None:
            obs['instructions_ids'] = np.zeros(32)
            obs['action_ids'] = np.ones(2) * -100
        return obs
