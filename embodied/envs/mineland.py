import json
import random
from typing import Tuple, List, Dict, Any, Optional

import elements
import embodied
import numpy as np
import cv2
import os, sys
MINELAND_HOME = os.environ.get("MINELAND_HOME", "/opt/MineLand")
pkg_root = os.path.join(MINELAND_HOME, "mineland")
if os.path.isdir(pkg_root):
    # Only insert if the real source tree is there and not already on path.
    if MINELAND_HOME not in sys.path:
        sys.path.insert(0, MINELAND_HOME)
else:
    raise RuntimeError(f"MINELAND_HOME does not contain 'mineland' package: {pkg_root}")

# MineLand API
import mineland
from mineland.sim import Action as MLAction
from mineland.sim import LowLevelAction as MLLowLevelAction
# at top
# import os, time, fcntl, subprocess, tempfile

# def _prepare_sim_once():
#     lock_path = "/tmp/mineland_server_setup.lock"
#     with open(lock_path, "w") as f:
#         fcntl.flock(f, fcntl.LOCK_EX)
#         stamp = "/tmp/mineland_server_ready"
#         if not os.path.exists(stamp):
#             try:
#                 subprocess.run(
#                     ["python", f"{os.environ.get('MINELAND_HOME','/opt/MineLand')}/scripts/validate_install_simulator.py"],
#                     check=True
#                 )
#             finally:
#                 open(stamp, "a").close()
#         fcntl.flock(f, fcntl.LOCK_UN)

# # inside your MineLand env wrapper, before calling mineland.make(...)
# _prepare_sim_once()

class MineLand(embodied.Env):
    """
    A light wrapper that adapts MineLand to the `embodied.Env` interface,
    mirroring the structure of your Overcooked wrapper.

    Key features:
    - Supports multi-agent (N agents) with a single discrete action per agent.
    - Two action backends:
        * High-level (default): actions are Mineflayer JS snippets via `mineland.Action`.
        * Low-level: 8‑dim discrete control via `mineland.LowLevelAction` (optional).
    - Returns a unified dict obs with fields similar to your Overcooked env:
        {state_0, state_1, image, reward, is_first, is_last, is_terminal, log/reward,
         (optional) instructions_ids, action_ids, action_text_ids}
    - Includes optional text tokenization for action strings (like your Overcooked wrapper).

    Notes:
    - We *do not* spin up the MineLand server until the first reset(), so construction is cheap.
    - `image` is built from the agents' egocentric RGB (concatenated) and resized to `image_size`.
    - Reward defaults to 0 each step; if the chosen MineLand *task* returns a TaskInfo with a
      `score` or `local_score`, we add that as shaped reward.
    - Done/terminal flags are forwarded from the MineLand task (if any). If you instantiate the
      raw simulator (playground), `done` will always be False unless the task logic says otherwise.
    """

    # Discrete actions interpreted per agent. Expand freely.
    # For high-level mode we map these to Mineflayer code strings.
    ACTIONS_HL = [
        "noop",                  # 0: Resume previous code (no new code)
        "chat_hi",              # 1: bot.chat("hi")
        "chat_status",          # 2: bot.chat status (health/hunger)
    ]

    # For low-level mode we offer two simple choices: no-op and random.
    ACTIONS_LL = [
        "noop",                 # 0: LowLevelAction.no_op()
        "random",               # 1: LowLevelAction.random_op()
    ]

    # Optional pretty names (mirrors your Overcooked wrapper API pattern)
    _ACT_DICT_HL = {0: "noop", 1: "chat_hi", 2: "chat_status"}
    _ACT_DICT_LL = {0: "noop", 1: "random"}

    def __init__(
        self,
        task_id: str = "survival_0.01_days",
        agents_count: int = 2,
        # MineLand sim/task kwargs (tweak as needed):
        ticks_per_step: int = 5,
        enable_auto_pause: bool = False,
        enable_low_level_action: bool = False,
        image_size: Tuple[int, int] = (64, 128),  # (H, W) of returned `image`
        headless: bool = True,
        # Tokenization (optional):
        tokenizer=None,
        action_text_len: int = 64,
        # Logging:
        logs: bool = False,
        logdir: Optional[str] = None,
        # Pass-through kwargs are forwarded to mineland.make(...)
        vlm=None,
        embedder=None,
        **make_kwargs,
    ):
        super().__init__()
        self.task_id = task_id
        self.agents_count = int(agents_count)
        self.ticks_per_step = int(ticks_per_step)
        self.enable_auto_pause = bool(enable_auto_pause)
        self.enable_low_level_action = bool(enable_low_level_action)
        self.image_h, self.image_w = map(int, image_size)
        self.headless = bool(headless)
        self.logs = logs
        self._logdir = logdir and elements.Path(logdir)
        self._logdir and self._logdir.mkdir()

        # Tokenizer setup (mirrors your Overcooked behavior)
        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
            self._pad_id = int(pad_id)
        else:
            self._pad_id = 0
        self.action_text_len = int(action_text_len)
        self.total_action_text_len = self.agents_count * self.action_text_len

        # Underlying MineLand env is created on first reset()
        self._env = None
        self._episode = 0
        self._length = None
        self._reward = None
        self._done = True

        # Keep a rolling cache of last observation for feature extraction
        self._last_obs = None  # type: Optional[List[Any]]

        # Stash kwargs for make()
        self._make_kwargs = dict(
            task_id=self.task_id,
            agents_count=self.agents_count,
            ticks_per_step=self.ticks_per_step,
            enable_auto_pause=self.enable_auto_pause,
            enable_low_level_action=self.enable_low_level_action,
            headless=self.headless,
            image_size=(self.image_h, self.image_w),
            **make_kwargs,
        )

    # ---- Introspection ----
    @property
    def act_names(self) -> List[str]:
        return [self._ACT_DICT_LL[i] if self.enable_low_level_action else self._ACT_DICT_HL[i]
                for i in range(self.num_actions)]

    @property
    def num_actions(self) -> int:
        return len(self.ACTIONS_LL) if self.enable_low_level_action else len(self.ACTIONS_HL)

    # ---- Spaces ----
    @property
    def obs_space(self) -> Dict[str, elements.Space]:
        # Fixed shapes for stability; contents are populated at runtime.
        spaces = {
            "state_0": elements.Space(np.float32, (32,)),
            "state_1": elements.Space(np.float32, (32,)),
            "image": elements.Space(np.uint8, (self.image_h, self.image_w, 3)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/reward": elements.Space(np.float32),
        }
        # Optional fields for text conditioning (kept for drop-in parity with your Overcooked env)
        spaces["instructions_ids"] = elements.Space(np.uint8, 32)
        spaces["action_ids"] = elements.Space(np.int32, (self.agents_count,))
        if self.tokenizer is not None:
            spaces["action_text_ids"] = elements.Space(np.int32, (self.total_action_text_len,))
        return spaces

    @property
    def act_space(self) -> Dict[str, elements.Space]:
        # One discrete action per agent + reset flag
        return {
            "action": elements.Space(np.int32, (self.agents_count,), 0, self.num_actions),
            "reset": elements.Space(bool),
        }

    # ---- Core loop ----
    def step(self, act: Dict[str, Any]):
        if act["reset"] or self._done:
            return self._reset()

        # Build MineLand actions (HL/LL) per agent
        if self.enable_low_level_action:
            actions = self._build_low_level_actions(act["action"])  # List[MLLowLevelAction]
        else:
            actions, action_texts = self._build_high_level_actions(act["action"])  # List[MLAction], List[str]

        # Step MineLand
        obs_list, code_info, events, done, task_info = self._env.step(actions)

        # Reward: prefer task_info.score/local_score if provided
        shaped = 0.0
        if task_info is not None:
            # TaskInfo may carry both global and per-agent scores; prefer global `score`
            shaped = float(getattr(task_info, "score", 0.0) or getattr(task_info, "local_score", 0.0))
        reward = float(shaped)
        self._reward += reward
        self._length += 1
        self._done = bool(done)

        if self._done and self._logdir:
            self._write_stats(self._length, self._reward)

        # Build features & image
        feat0 = self._featurize_agent(obs_list, 0)
        feat1 = self._featurize_agent(obs_list, 1 if self.agents_count > 1 else 0)
        image = self._compose_image(obs_list)

        # Optional action text tokens
        action_text_ids = None
        if (not self.enable_low_level_action) and (self.tokenizer is not None):
            # Ensure we have exactly N action strings
            if "action_texts" not in locals():
                action_texts = ["" for _ in range(self.agents_count)]
            action_text_ids = self._encode_action_texts(action_texts)

        return self._obs(
            feat0, feat1, image, reward,
            is_last=self._done, is_terminal=False,
            action_text_ids=action_text_ids,
        )

    def _reset(self):
        # Lazy-create MineLand env (task wrapper recommended via mineland.make)
        if self._env is None:
            self._env = mineland.make(**self._make_kwargs)
        obs_list = self._env.reset()
        self._episode += 1
        self._length = 0
        self._reward = 0.0
        self._done = False

        feat0 = self._featurize_agent(obs_list, 0)
        feat1 = self._featurize_agent(obs_list, 1 if self.agents_count > 1 else 0)
        image = self._compose_image(obs_list)

        action_text_ids = None
        if (not self.enable_low_level_action) and (self.tokenizer is not None):
            action_text_ids = self._encode_action_texts(["" for _ in range(self.agents_count)])

        return self._obs(feat0, feat1, image, 0.0, is_first=True, action_text_ids=action_text_ids)

    # ---- Builders ----
    def _build_high_level_actions(self, action_indices: np.ndarray) -> Tuple[List[MLAction], List[str]]:
        actions: List[MLAction] = []
        texts: List[str] = []
        for i in range(self.agents_count):
            idx = int(action_indices[i])
            name = self.ACTIONS_HL[idx]
            if name == "noop":
                actions.append(MLAction(MLAction.RESUME, ""))
                texts.append("idle")
            elif name == "chat_hi":
                code = f"bot.chat('hi from agent {i}')"
                actions.append(MLAction(MLAction.NEW, code))
                texts.append("chat hi")
            elif name == "chat_status":
                # Print a minimal status; Mineflayer exposes bot.health, bot.food, etc.
                code = (
                    "bot.chat(`hp=${bot.health.toFixed(0)} food=${bot.food.toFixed(0)} sat=${bot.foodSaturation.toFixed(0)}`)"
                )
                actions.append(MLAction(MLAction.NEW, code))
                texts.append("chat status")
            else:
                actions.append(MLAction(MLAction.RESUME, ""))
                texts.append("")
        return actions, texts

    def _build_low_level_actions(self, action_indices: np.ndarray) -> List[MLLowLevelAction]:
        # Very simple: index 0 = noop, index 1 = random
        if int(np.max(action_indices)) == 0:
            return MLLowLevelAction.no_op(self.agents_count)
        # If any agent selects random, just randomize all for now (simple baseline)
        return MLLowLevelAction.random_op(self.agents_count)

    # ---- Representation ----
    def _compose_image(self, obs_list: List[Any]) -> np.ndarray:
        """Concatenate agents' egocentric RGB and resize to (H, W)."""
        frames = []
        for i in range(self.agents_count):
            rgb = getattr(obs_list[i], "rgb", None) if (obs_list and len(obs_list) > i) else None
            if rgb is None:
                frames.append(np.zeros((self.image_h, self.image_w // self.agents_count, 3), dtype=np.uint8))
            else:
                # Ensure HWC uint8
                arr = np.asarray(rgb, dtype=np.uint8)
                # Keep aspect, then fit into slot width
                target_w = max(1, self.image_w // self.agents_count)
                resized = cv2.resize(arr, (target_w, self.image_h))
                frames.append(resized)
        cat = np.concatenate(frames, axis=1)
        # If width mismatched due to int division, pad right
        if cat.shape[1] < self.image_w:
            pad = np.zeros((self.image_h, self.image_w - cat.shape[1], 3), dtype=np.uint8)
            cat = np.concatenate([cat, pad], axis=1)
        return cat

    def _featurize_agent(self, obs_list: List[Any], i: int) -> np.ndarray:
        """Build a small numeric feature vector, robust to missing keys."""
        vec = []
        o = obs_list[i] if (obs_list and len(obs_list) > i and obs_list[i] is not None) else None
        # Basic world/time
        for key in ("age", "time", "day"):
            vec.append(float(getattr(o, key, 0.0)))
        # Agent health/food status
        for key in ("health", "hunger", "foodSaturation", "oxygen"):
            vec.append(float(getattr(o, key, 0.0)))
        # Position (x,y,z) if available
        pos = getattr(o, "pos", None)
        if isinstance(pos, (tuple, list)) and len(pos) >= 3:
            vec.extend([float(pos[0]), float(pos[1]), float(pos[2])])
        else:
            # Some builds may have o.position instead
            p2 = getattr(o, "position", None)
            if isinstance(p2, dict):
                vec.extend([float(p2.get(k, 0.0)) for k in ("x", "y", "z")])
            else:
                vec.extend([0.0, 0.0, 0.0])
        # Orientation
        for key in ("yaw", "pitch"):
            vec.append(float(getattr(o, key, 0.0)))
        # Inventory sketch (counts for a few common items if present)
        inv = getattr(o, "inventory", None)
        if isinstance(inv, dict):
            for k in ("oak_log", "spruce_log", "oak_planks", "cobblestone", "stick", "coal", "iron_ore", "iron_ingot", "bread"):
                vec.append(float(inv.get(k, 0)))
        else:
            vec.extend([0.0] * 9)
        # Pad / trim to 32 dims
        if len(vec) < 32:
            vec += [0.0] * (32 - len(vec))
        return np.array(vec[:32], dtype=np.float32)

    # ---- Tokenization helpers ----
    def _encode_action_texts(self, texts: List[str]) -> np.ndarray:
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
        return ids.flatten()  # (N*L,)

    # ---- Output packer ----
    def _obs(self,
             feat0: np.ndarray,
             feat1: np.ndarray,
             image: np.ndarray,
             reward: float,
             is_first: bool = False,
             is_last: bool = False,
             is_terminal: bool = False,
             action_text_ids: Optional[np.ndarray] = None):
        obs = {
            "state_0": np.array(feat0, dtype=np.float32),
            "state_1": np.array(feat1, dtype=np.float32),
            "image": np.array(image, dtype=np.uint8),
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/reward": np.float32(self._reward),
            # Keep parity with your Overcooked wrapper for VLM hooks
            "instructions_ids": np.zeros(32, dtype=np.uint8),
            "action_ids": np.ones(2, dtype=np.int32) * -100,
        }
        if self.tokenizer is not None:
            if action_text_ids is None:
                action_text_ids = np.full(self.total_action_text_len, self._pad_id, dtype=np.int32)
            obs["action_text_ids"] = action_text_ids
        return obs

    # ---- Logging ----
    def _write_stats(self, length: int, reward: float):
        stats = {"episode": self._episode, "length": int(length), "reward": float(round(reward, 1))}
        filepath = self._logdir / "stats.jsonl"
        lines = filepath.read() if filepath.exists() else ""
        lines += json.dumps(stats) + "\n"
        filepath.write(lines, mode="w")
        print(f"[MineLandEnv] Wrote stats to {filepath}")
