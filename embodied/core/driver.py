import time

import random

import cloudpickle
import elements
import numpy as np
import portal
from PIL import Image

import torch

class Driver:

  def __init__(self, make_env_fns, vlm, embedder, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      self.stop = context.Event()
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          portal.Process(self._env_server, self.stop, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
      self.pipes[0].send(('act_names',))
      self.act_names = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
      self.act_names = self.envs[0].act_names

    # self.kwargs["act_names"] = self.act_names
    self.callbacks = []
    self.acts = None
    self.carry = None

    # self.instr_ids = np.ones([self.length, 32], dtype=np.uint8)
    # self.action_ids = np.ones([self.length, 2], dtype=np.int32) * -100

    # self.vlm = vlm
    # self.embedder = embedder
    # self.instr_interval = 30
    # self.max_instr_interval = 30
    # self.last_step = 0
    # self.action_cache = []
    # self.dropout_rate = 0.2

    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.kill() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def sample_with_vlm(self, frames, actions=None, check_proactive=False):
    """
    Given a list of PIL frames and corresponding low-level action strings,
    returns a JAX array of shape (batch, hidden_dim) as the frozen text embedding.
    """
    if check_proactive:
      with torch.no_grad():
        proactive_signal = self.vlm(frames, check_proactive=True)
        return proactive_signal
    else:
      with torch.no_grad():
        captions = self.vlm(frames, actions)
        print(captions)
        # 3. encode (frozen)
        embeds, caption_ids = self.embedder.encode(captions, 32)
        if embeds is not None:
          embeds = embeds.detach().cpu().float().numpy()
        caption_ids = caption_ids.detach().cpu().numpy().astype(np.uint8)
      return embeds, caption_ids

  def _step(self, policy, step, episode):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]

    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    assert all(len(x) == self.length for x in obs.values()), obs
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      # for func in self.callbacks:
      #   print(f"{func.__module__}.{func.__qualname__}")
      trn = elements.tree.map(lambda x: x[i], trans)
      # one-liner, using a ternary expression
      [
          fn(trn, self.act_names, i, **self.kwargs)                           # value to append
          if f"{fn.__module__}.{fn.__qualname__}" == "embodied.core.replay.Replay.add"                              # condition
          else fn(trn, i, **self.kwargs)                      # alternative
          for fn in self.callbacks                            # iteration
      ]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(stop, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while not stop.is_set():
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        elif msg == 'act_names':
          assert len(args) == 0
          pipe.send(('result', env.act_names))
        else:
          raise ValueError(f'Invalid message {msg}')
    except ConnectionResetError:
      print('Connection to driver lost')
    except Exception as e:
      pipe.send(('error', e))
      raise
    finally:
      try:
        env.close()
      except Exception:
        pass
      pipe.close()
