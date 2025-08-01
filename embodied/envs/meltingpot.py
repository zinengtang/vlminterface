import json, random, os
from typing import Any, Dict, List, Optional, Tuple

import elements
import embodied
import numpy as np
import cv2

# Offscreen, like your Overcooked wrapper
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# ---- Pick 2p substrates so we can keep action shape = (2,) ----
# MELTINGPOT_TASKS_2P = [
#     # Repeated / matrix games (2 players)
#     # "prisoners_dilemma_in_the_matrix__arena",
#     # "stag_hunt_in_the_matrix__arena",
#     # "chicken_in_the_matrix__arena",
#     # "bach_or_stravinsky_in_the_matrix__arena",
#     # "pure_coordination_in_the_matrix__arena",
#     # "rationalizable_coordination_in_the_matrix__arena",
#     "prisoners_dilemma_in_the_matrix__arena"
# ]

class MeltingPot(embodied.Env):
    """
    Melting Pot wrapper that mirrors your Overcooked env:
      - obs has: state_0, state_1, image, reward, is_first, is_last, is_terminal, log/reward
      - act_space: {'action': int32 tensor of shape (2,), 'reset': bool}
      - step() expects 2 discrete actions (we choose 2p substrates)
      - logs to stats.jsonl when an episode ends
    """
    EXTRA_KEYS = (
        "COLLECTIVE_REWARD",
        "INTERACTION_INVENTORIES",
        "READY_TO_SHOOT",
        "INVENTORY",
        'MY_OFFER', 
        'STAMINA', 
        'HUNGER', 
        'OFFERS',
    )
    def _sanitize_key(self, k: str) -> str:
        return k.lower().replace(".", "_").replace("/", "_").replace(" ", "_")
    
    def _dtype_for(self, arr: np.ndarray):
        dt = getattr(arr, "dtype", None)
        if dt is None:
            return np.float32
        if dt == np.bool_ or dt == bool:
            return bool
        if np.issubdtype(dt, np.floating):
            return np.float32
        if np.issubdtype(dt, np.integer):
            # keep uint8 as uint8 (e.g., masks); otherwise int32
            return np.float32
        return np.float32

    # Will be set after env build, but we expose the property for parity
    @property
    def act_names(self):
        # In Melting Pot, discrete action ids are scenario-specific, so we surface ids.
        return list(range(self._n_actions))

    @property
    def obs_space(self):
        # Build dummy obs by looking at current env/timestep
        image = self._render_image()
        # Keep small placeholders for state_0/state_1 so your nets can ignore/use as needed
        # feat_shape = (1,)
        spaces = {
            # "state_0": elements.Space(np.float32, feat_shape),
            # "state_1": elements.Space(np.float32, feat_shape),
            "image": elements.Space(np.uint8, image.shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/reward": elements.Space(np.float32),
        }
        for name, (dtype, shape, _, _) in getattr(self, "_extra_specs", {}).items():
           if dtype is np.int32:
               spaces[name] = elements.Space(np.int32, shape, low=0)
           else:
               spaces[name] = elements.Space(dtype, shape)
        if self.vlm is not None:
            spaces["instructions_ids"] = elements.Space(np.uint8, 32)
            spaces["action_ids"] = elements.Space(np.int32, 2)
        return spaces

    @property
    def act_space(self):
        # Keep shape=(2,) to match your Overcooked training loop
        return {
            "action": elements.Space(np.int32, (2,), 0, self._n_actions),
            "reset": elements.Space(bool),
        }

    # ---- Lifecycle ----
    def __init__(self,
                 task: Optional[str] = None,
                 horizon: int = 400,
                 logs: bool = False,
                 logdir: Optional[str] = None,
                 seed: Optional[int] = None,
                 vlm=None,
                 embedder=None):
        super().__init__()
        self._task = task
        self._horizon = horizon
        self._logs = logs
        self._logdir = logdir and elements.Path(logdir)
        self._logdir and self._logdir.mkdir()

        self.vlm = vlm
        self.embedder = embedder

        # Internal episode bookkeeping
        self._episode = 0
        self._length = 0
        self._reward_sum = 0.0
        self._done = True
        self._last_ts = None

        # Build the first substrate
        self._build_env()

    # ---- Core API ----
    def step(self, action: Dict[str, Any]):
        if action["reset"] or self._done:
            return self._reset()

        # Expect two ints for two players (we build 2p substrates below)
        a0, a1 = int(action["action"][0]), int(action["action"][1])
        joint = [a0, a1]  # 2 players
        ts = self._env.step(joint)
        self._last_ts = ts

        # Reward is per-player; sum to a scalar like you do in Overcooked
        # (change to np.mean if you'd prefer average-per-agent)
        r = float(np.sum(ts.reward))
        self._reward_sum += r
        self._length += 1
        self._done = ts.last()

        if (self._done or (self._horizon and self._length >= self._horizon)) and self._logdir:
            self._write_stats(self._length, self._reward_sum)

        # Observation
        return self._obs(r,
                         is_first=False,
                         is_last=self._done,
                         is_terminal=False)

    # ---- Helpers ----
    def _reset(self):
        # Optionally rotate tasks
        # if self._task is None:
        #     self._task = random.choice(MELTINGPOT_TASKS_2P)
        self._build_env()  # (re)build for this task
        ts = self._env.reset()
        self._last_ts = ts

        self._episode += 1
        self._length = 0
        self._reward_sum = 0.0
        self._done = False

        return self._obs(0.0, is_first=True, is_last=False, is_terminal=False)

    def _flat_shape(self, v0: np.ndarray):
        # v0 is single-agent array; we stack over players on axis 0.
        if v0.ndim <= 1:
            return (self._n_players,) + tuple(v0.shape)
        else:
            return (self._n_players, int(np.prod(v0.shape)))

    def _build_extra_specs(self):
        """Infer shapes/dtypes for EXTRA_KEYS from the current timestep."""
        self._extra_specs = {}  # name -> (dtype, shape, original_key)
        ts_obs = self._last_ts.observation  # list per agent
        if not isinstance(ts_obs, (list, tuple)) or len(ts_obs) == 0:
            return
        for k in self.EXTRA_KEYS:
            if k not in ts_obs[0]:
                continue
            v0 = np.asarray(ts_obs[0][k])
            dtype = self._dtype_for(v0)
            flatten = (v0.ndim >= 2)                      # flatten multi-d arrays
            shape = self._flat_shape(v0) if flatten else (self._n_players,) + tuple(v0.shape)
            name = f"{self._sanitize_key(k)}"
            self._extra_specs[name] = (dtype, shape, k, flatten)

    def _build_env(self):
        # Build Melting Pot substrate
        import meltingpot
        from meltingpot import substrate  # official API
        # Get default roles and keep two players (row/column roles are preserved for matrix games)
        # cfg = substrate.get_config(self._task or random.choice(MELTINGPOT_TASKS_2P))
        # roles = list(cfg.default_player_roles)[:2]
        # self._env = substrate.build(cfg, roles=roles)  # roles=... is the documented pattern

        cfg = substrate.get_config(self._task)
        roles = list(cfg.default_player_roles)[:2]
        self._n_players = len(roles)
        self._env = substrate.build_from_config(cfg, roles=roles)

        # Infer action space size from player 0 spec
        spec = self._env.action_spec()
        # action_spec may be a list/tuple of per-agent specs
        spec0 = spec[0] if isinstance(spec, (list, tuple)) else spec
        self._n_actions = int(spec0.num_values)

        # Prime a timestep so obs_space can see image shape
        if self._last_ts is None or self._done:
            self._last_ts = self._env.reset()

        self._build_extra_specs()

    def _render_image(self):
        # Prefer WORLD.RGB; fallback to player_0's local RGB
        # _last_ts.observation is a per-agent list of dicts
        obs = self._last_ts.observation
        img = None
        try:
            img = obs[0].get("WORLD.RGB", None)
        except Exception:
            img = None
        if img is None:
            img = obs[0].get("RGB", None)
        if img is None:
            # Fallback: blank frame
            img = np.zeros((88, 88, 3), np.uint8)
        # Normalize to 128x128 like your Overcooked renderer
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_NEAREST)
        return img.astype(np.uint8)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        # Simple 1D placeholders for state_0 / state_1 (you can swap for richer features)
        # state0 = np.zeros((1,), np.float32)
        # state1 = np.zeros((1,), np.float32)
        image = self._render_image()
        out = {
            # "state_0": state0,
            # "state_1": state1,
            "image": np.array(image, dtype=np.uint8),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "log/reward": np.float32(self._reward_sum),
        }
        # Pack the extra MP observations
        ts_obs = self._last_ts.observation
        for name, (dtype, shape, orig, flatten) in self._extra_specs.items():
            try:
                stacked = np.stack([np.asarray(ts_obs[i][orig]) for i in range(self._n_players)], axis=0)
                if flatten and stacked.ndim >= 3:
                    stacked = stacked.reshape(self._n_players, -1)
                if dtype is bool:
                    stacked = stacked.astype(np.bool_, copy=False)
                else:
                    stacked = stacked.astype(dtype, copy=False)
                if dtype is np.int32:
                   # Map any negative sentinel(s) to 0.
                   # (Faster than np.maximum for large arrays.)
                   neg_mask = stacked < 0
                   if np.any(neg_mask):
                       stacked[neg_mask] = 0
                out[name] = stacked
            except Exception:
                # Key missing or shape mismatch: fill zeros of expected shape
                out[name] = np.zeros(shape, dtype=dtype if dtype is not bool else np.bool_)

        if self.vlm is not None:
            out["instructions_ids"] = np.zeros(32, dtype=np.uint8)
            out["action_ids"] = np.full((2,), -100, dtype=np.int32)
        return out

    def _write_stats(self, length: int, reward: float):
        if not self._logdir:
            return
        stats = {"episode": self._episode, "length": length, "reward": round(reward, 1)}
        path = self._logdir / "stats.jsonl"
        lines = path.read() if path.exists() else ""
        lines += json.dumps(stats) + "\n"
        path.write(lines, mode="w")
        print(f"[MeltingPot] Wrote stats to {path}")
