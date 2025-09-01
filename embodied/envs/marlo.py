import json
import random
from typing import Tuple, List, Dict, Any, Optional

import elements
import embodied
import numpy as np
import cv2

# === Optional but recommended: set headless rendering for any viewers you might add ===
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# --- MARLÖ / Malmo imports (fail gracefully if not available) ---
try:
    import marlo
except Exception as e:
    marlo = None
    _MARLO_IMPORT_ERROR = e
else:
    _MARLO_IMPORT_ERROR = None


# ======= MARLÖ multi-agent tasks (2 agents) ====================================
# These mission templates ship with MarLo and define 2 AgentSections.
# You can add/remove any of the *Train* envs; they’re multi-agent by template.
MARLO_TASKS_2P = [
    "MarLo-MobchaseTrain1-v0",
    "MarLo-MobchaseTrain2-v0",
    "MarLo-MobchaseTrain3-v0",
    "MarLo-MobchaseTrain4-v0",
    "MarLo-MobchaseTrain5-v0",
    "MarLo-BuildbattleTrain1-v0",
    "MarLo-BuildbattleTrain2-v0",
    "MarLo-BuildbattleTrain3-v0",
    "MarLo-BuildbattleTrain4-v0",
    "MarLo-BuildbattleTrain5-v0",
    "MarLo-TreasurehuntTrain1-v0",
    "MarLo-TreasurehuntTrain2-v0",
    "MarLo-TreasurehuntTrain3-v0",
    "MarLo-TreasurehuntTrain4-v0",
    "MarLo-TreasurehuntTrain5-v0",
]


def _resize_rgb(img: np.ndarray, size=(64, 64)) -> np.ndarray:
    """Malmo returns HxWxC RGB frames; standardize to 64x64 like your Overcooked image."""
    if img is None or img.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    return cv2.resize(img, size)  # default interpolation is fine for tiny UI frames


def _tile_images(imgs: List[np.ndarray], out_size=(64, 64)) -> np.ndarray:
    """Tile per-agent frames side-by-side, then downscale to a single frame."""
    if not imgs:
        return np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    # Normalize shapes
    h = max(i.shape[0] for i in imgs if i is not None) if imgs[0] is not None else 240
    w = max(i.shape[1] for i in imgs if i is not None) if imgs[0] is not None else 320
    norm = []
    for im in imgs:
        if im is None or im.size == 0:
            im = np.zeros((h, w, 3), dtype=np.uint8)
        elif (im.shape[0], im.shape[1]) != (h, w):
            im = cv2.resize(im, (w, h))
        norm.append(im)
    strip = np.concatenate(norm, axis=1)  # horizontal concat
    return _resize_rgb(strip, out_size)


class MarLoEnv(embodied.Env):
    """
    Drop-in wrapper modeled after your Overcooked class, but for MARLÖ (Minecraft).

    Key points:
    - Multi-agent (default 2 agents) via MARLÖ join tokens.
    - Obs dict matches your template keys:
        {"state_0","state_1","image","reward","is_first","is_last","is_terminal","log/reward",
         optional: instructions_ids, action_ids, action_text_ids}
    - Action interface: integer actions per-agent (shape (num_agents,)).
      We pass them directly to each MarLo env. MarLo internally wraps or mods indices
      to its own discrete command list (move/turn/use/attack/look).
    - Tokenized action text (optional) mirrors your logic, using action name strings
      from MarLo’s env to generate short textual descriptions and pack to ids.
    """

    # We won’t hard-code ACTIONS because MarLo envs build their own discrete actions.
    # Instead we expose a vector of ints per agent.
    def __init__(
        self,
        task: Optional[str] = None,
        horizon: int = 1000,
        logs: bool = False,
        logdir: Optional[str] = None,
        seed: Optional[int] = None,
        vlm=None,
        embedder=None,
        tokenizer=None,
        action_text_len: int = 64,
        num_agents: int = 2,
        client_pool: Optional[List[Tuple[str, int]]] = None,
        video_resolution: Tuple[int, int] = (320, 240),
        observe_full_inventory: bool = False,
        observe_recent_commands: bool = False,
        continuous_to_discrete: bool = True,
        allow_discrete_movement: bool = True,
        allow_continuous_movement: bool = False,
        # If you want survival-open-world feel, you can feed RawXMLEnv with Survival missions,
        # but here we use the shipped, multi-agent templates.
    ):
        super().__init__()
        if marlo is None:
            raise ImportError(
                f"Could not import marlo: {_MARLO_IMPORT_ERROR}\n"
                "Install Malmo and MarLo first (e.g., `pip install malmo` then `pip install marlo`), "
                "and make sure Minecraft clients are running on the specified ports."
            )

        self._logs = logs
        self._logdir = logdir and elements.Path(logdir)
        self._logdir and self._logdir.mkdir()
        self._episode = 0
        self._length = None
        self._reward = None
        self._done = True
        self._step = 0

        self.vlm = vlm
        self.embedder = embedder

        # Text tokenizer (same pattern as your Overcooked wrapper)
        self.tokenizer = tokenizer
        self.action_text_len = int(action_text_len)
        self.total_action_text_len = num_agents * self.action_text_len
        if self.tokenizer is not None:
            # choose a sensible pad id
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
            self._pad_id = int(pad_id)
        else:
            self._pad_id = 0

        self._num_agents = int(num_agents)
        self._horizon = int(horizon)

        # Pick task (multi-agent templates by default)
        self._task_key = task or random.choice(MARLO_TASKS_2P)

        # Default client pool for N agents if not provided
        if client_pool is None:
            # Expect Minecraft clients already launched on these ports
            base = 10000
            client_pool = [( "127.0.0.1", base + i ) for i in range(self._num_agents)]
        self._client_pool = client_pool

        # Build MARLÖ game params (mirrors marlo.init docs)
        self._marlo_params = dict(
            client_pool=self._client_pool,
            videoResolution=list(video_resolution),
            observeFullInventory=bool(observe_full_inventory),
            observeRecentCommands=bool(observe_recent_commands),
            continuous_to_discrete=bool(continuous_to_discrete),
            allowDiscreteMovement=bool(allow_discrete_movement),
            allowContinuousMovement=bool(allow_continuous_movement),
            gameMode="survival",  # survival feels: open-worldish policies even in templates
            suppress_info=True,
        )

        self._envs = []
        self._action_names: List[List[str]] = []   # per-agent list of command strings
        self._act_sizes: List[int] = []            # per-agent Discrete sizes
        self._reset_build_env()

    # -------- Public API mirrors your Overcooked wrapper --------
    @property
    def act_names(self) -> List[str]:
        # Expose a merged, de-duplicated set of *this mission’s* action strings
        if not self._action_names:
            return []
        merged = []
        for names in self._action_names:
            for n in names:
                if n not in merged:
                    merged.append(n)
        return merged

    @property
    def obs_space(self):
        # We don’t have mdp.featurize_state like Overcooked; provide stable placeholders.
        # If you want richer stats, toggle observeFullInventory/observeRecentCommands and
        # parse `info` to build vectors here.
        dummy_state0 = np.zeros((32,), dtype=np.float32)
        dummy_state1 = np.zeros((32,), dtype=np.float32)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)

        spaces = {
            "state_0": elements.Space(np.float32, dummy_state0.shape),
            "state_1": elements.Space(np.float32, dummy_state1.shape),
            "image": elements.Space(np.uint8, dummy_image.shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/reward": elements.Space(np.float32),
        }
        if self.vlm is not None:
            spaces['instructions_ids'] = elements.Space(np.uint8, 32)
            spaces['action_ids'] = elements.Space(np.int32, self._num_agents)
        if self.tokenizer is not None:
            spaces['action_text_ids'] = elements.Space(np.int32, self.total_action_text_len)
        return spaces

    @property
    def act_space(self):
        # Accept an int per agent. We expose max width; marlo will internally mod indices.
        max_n = max(self._act_sizes) if self._act_sizes else 12
        return {
            "action": elements.Space(np.int32, (self._num_agents,), 0, max_n),
            "reset": elements.Space(bool),
        }

    # --------- Core loop ----------
    def step(self, action):
        if action["reset"] or self._done:
            return self._reset()

        joint = np.asarray(action["action"]).tolist()
        if len(joint) != self._num_agents:
            # pad/trim to num_agents to be robust
            joint = (joint + [0] * self._num_agents)[:self._num_agents]

        per_images = []
        per_rewards = []
        per_dones = []

        # Step each agent’s env
        for i, (env_i, a_i) in enumerate(zip(self._envs, joint)):
            try:
                img, rew, done, info = env_i.step(int(a_i))
            except Exception:
                # If a client died mid-episode, mark done and zero obs
                img, rew, done, info = (None, 0.0, True, {})
            per_images.append(img)
            per_rewards.append(float(rew))
            per_dones.append(bool(done))

        # Aggregate (cooperative default)
        reward = float(np.sum(per_rewards))
        self._reward += reward
        self._length += 1
        self._done = bool(any(per_dones) or (self._length >= self._horizon))

        # Image: tile per-agent frames horizontally → 64x64
        image = _tile_images(per_images, out_size=(64, 64))

        # Build (optional) short action texts → ids
        action_text_ids = None
        if self.tokenizer is not None:
            texts = []
            for i, a_i in enumerate(joint):
                name = self._action_names[i][int(a_i) % len(self._action_names[i])] \
                        if i < len(self._action_names) and self._action_names[i] else ""
                # Convert like "move 1" → "move forward", "turn -1" → "turn left", etc.
                name = name.strip()
                if name.startswith("move "):
                    name = "move forward" if name.endswith("1") else "move backward"
                elif name.startswith("turn "):
                    name = "turn right" if name.endswith("1") else "turn left"
                elif name.startswith("look "):
                    name = "look up" if name.endswith("1") else "look down"
                elif name.startswith("strafe "):
                    name = "strafe right" if name.endswith("1") else "strafe left"
                elif name in ("attack 1", "attack"):
                    name = "attack"
                elif name in ("use 1", "use"):
                    name = "use"
                elif name in ("jump 1", "jump"):
                    name = "jump"
                texts.append(name)
            # pad to num_agents
            while len(texts) < self._num_agents:
                texts.append("")
            action_text_ids = self._encode_action_texts(texts[:self._num_agents])

        if self._done and self._logdir:
            self._write_stats(self._length, self._reward)

        # (Placeholders for state_0/1; customize if you parse info)
        feat0 = np.zeros((32,), dtype=np.float32)
        feat1 = np.zeros((32,), dtype=np.float32)

        return self._obs(
            feat0, feat1, image, reward,
            is_last=self._done, is_terminal=False,
            action_text_ids=action_text_ids,
        )

    def _reset(self):
        # (Re)connect to a fresh mission each episode for stability.
        self._reset_build_env()

        self._episode += 1
        self._length = 0
        self._reward = 0.0
        self._done = False
        self._step = 0

        # reset each agent env to get first frame
        per_images = []
        for env_i in self._envs:
            try:
                img = env_i.reset()
            except Exception:
                img = None
            per_images.append(img)
        image = _tile_images(per_images, out_size=(64, 64))

        action_text_ids = None
        if self.tokenizer is not None:
            action_text_ids = self._encode_action_texts([""] * self._num_agents)

        feat0 = np.zeros((32,), dtype=np.float32)
        feat1 = np.zeros((32,), dtype=np.float32)

        return self._obs(feat0, feat1, image, 0.0, is_first=True, action_text_ids=action_text_ids)

    # --------- Internals ----------
    def _obs(self, feat0, feat1, image, reward,
             is_first=False, is_last=False, is_terminal=False,
             action_text_ids=None):
        obs = {
            "state_0": np.array(feat0, dtype=np.float32),
            "state_1": np.array(feat1, dtype=np.float32),
            "image": np.array(image, dtype=np.uint8),
            "reward": np.float32(reward),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
            "log/reward": np.float32(self._reward),
        }
        if self.vlm is not None:
            obs['instructions_ids'] = np.zeros(32, dtype=np.uint8)
            obs['action_ids'] = np.ones(self._num_agents, dtype=np.int32) * -100
        if self.tokenizer is not None:
            if action_text_ids is None:
                action_text_ids = np.full(self.total_action_text_len, self._pad_id, dtype=np.int32)
            obs['action_text_ids'] = action_text_ids.flatten()
        self._step += 1
        return obs

    def _encode_action_texts(self, texts: List[str]) -> np.ndarray:
        """Tokenize N texts, pad/truncate to action_text_len, then concat → (N*L,)."""
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.action_text_len,
            padding="max_length",
        )
        ids = np.array(batch["input_ids"], dtype=np.int32)  # (N, L)
        if self._pad_id != 0:
            ids = np.where(ids == 0, self._pad_id, ids)
        return ids.flatten()

    def _reset_build_env(self):
        """(Re)create the MARLÖ envs and join both agents into the same mission."""
        # Close any previous envs
        for e in self._envs:
            try:
                e.close()
            except Exception:
                pass
        self._envs = []
        self._action_names = []
        self._act_sizes = []

        # Choose task (keep previously chosen unless user changes)
        env_key = self._task_key

        # Request a mission for all agents, then init one env per join_token
        join_tokens = marlo.make(env_key, params=dict(self._marlo_params))
        if len(join_tokens) < self._num_agents:
            # Some envs are single-agent; force a multi-agent template if necessary.
            self._task_key = random.choice(MARLO_TASKS_2P)
            env_key = self._task_key
            join_tokens = marlo.make(env_key, params=dict(self._marlo_params))

        # Initialize agent envs
        for i in range(self._num_agents):
            env_i = marlo.init(join_tokens[i], params=dict(self._marlo_params))
            self._envs.append(env_i)
            # Introspect action names / counts (from Marlo builder)
            names = getattr(env_i, "action_names", None)
            if isinstance(names, list) and names:
                # Single agent → [list]; Multi → [list] or nested; normalize to per-agent list
                if isinstance(names[0], list):
                    # already nested by agents; pick the appropriate head
                    self._action_names.append(names[0])
                else:
                    self._action_names.append(names)
            else:
                self._action_names.append([])

            n = getattr(env_i.action_space, "n", None)
            self._act_sizes.append(int(n) if n is not None else 12)

    # ---------- logging ----------
    def _write_stats(self, length, reward):
        stats = {
            "episode": self._episode,
            "length": length,
            "reward": round(float(reward), 1),
        }
        if not self._logdir:
            return
        filepath = self._logdir / "stats.jsonl"
        lines = filepath.read() if filepath.exists() else ""
        lines += json.dumps(stats) + "\n"
        filepath.write(lines, mode="w")
        print(f"[MarLoEnv] Wrote stats to {filepath}")
