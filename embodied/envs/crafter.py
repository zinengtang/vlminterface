import json

import random
import crafter
import elements
import embodied
import numpy as np
from PIL import Image
import torch

class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), logs=False, logdir=None, seed=None, vlm=None, embedder=None, use_action=False):
    assert task in ('reward', 'noreward')
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    self._logs = logs
    self._logdir = logdir and elements.Path(logdir)
    self._logdir and self._logdir.mkdir()
    self._episode = 0
    self._length = None
    self._reward = None
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

    self.vlm = vlm
    self.embedder = embedder
    self.action_cache = []
    self.max_actions = 30
    self.use_action = use_action
    self.last_instruction_step = 0
    self.instr_interval = 30
    self.min_instr_interval = 5
    self.dropout_rate = 0.15
    self._step = 0

  @property
  def act_names(self):
    return self._env.action_names
  
  @property
  def obs_space(self):
    spaces = {
        'image': elements.Space(np.uint8, self._env.observation_space.shape),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/reward': elements.Space(np.float32),
    }
    if self.vlm is not None:
      # spaces['instructions'] = elements.Space(np.float32, 384)
      spaces['instructions_ids'] = elements.Space(np.uint8, 32)
      spaces['action_ids'] = elements.Space(np.int32)
    if self._logs:
      spaces.update({
          f'log/achievement_{k}': elements.Space(np.int32)
          for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._episode += 1
      self._length = 0
      self._reward = 0
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    self._reward += reward
    self._length += 1
    if self._done and self._logdir:
      self._write_stats(self._length, self._reward, info)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    
    
    # self._last_instr_ids = np.ones(32, dtype=np.uint8)
    # if self.vlm is not None and random.random() > self.dropout_rate:
    #   if self._step % self.min_instr_interval == 0 and False:
    #     proactive = self.sample_with_vlm(
    #       [pil_frame], 
    #       self.action_cache, 
    #       check_proactive=True
    #     )
    #   else:
    #     proactive = False
    #   if is_first or (self._step % self.instr_interval == 0) or proactive:
    #     pil_frame = Image.fromarray(image)
    #     self._last_instr_embed, self._last_instr_ids = self.sample_with_vlm([pil_frame], self.action_cache)
        
    #     self.action_cache = []

    # instr_dict = dict(
    #   intr_ids=self._last_instr_ids
    # )

    obs = dict(
        image=image,
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        **{'log/reward': np.float32(info['reward'] if info else 0.0)},
    )
    # if self.vlm is not None:
    #   # obs['instructions'] = self._last_instr_embed
    #   obs['instructions_ids'] = self._last_instr_ids

    if self._logs:
      log_achievements = {
          f'log/achievement_{k}': info['achievements'][k] if info else 0
          for k in self._achievements}
      obs.update({k: np.int32(v) for k, v in log_achievements.items()})
    self._step += 1
    return obs

  def _write_stats(self, length, reward, info):
    stats = {
        'episode': self._episode,
        'length': length,
        'reward': round(reward, 1),
        **{f'achievement_{k}': v for k, v in info['achievements'].items()},
    }
    filename = self._logdir / 'stats.jsonl'
    lines = filename.read() if filename.exists() else ''
    lines += json.dumps(stats) + '\n'
    filename.write(lines, mode='w')
    print(f'Wrote stats: {filename}')
