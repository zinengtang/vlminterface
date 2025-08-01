import logging
import threading

import random
import elements
import embodied
import numpy as np
from PIL import Image

np.float = float
np.int = int
np.bool = bool

import time
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP

from skimage.metrics import structural_similarity as ssim

import torch

import lpips, torch
from PIL import Image
import torchvision.transforms as T


BLUEPRINT_OBJECTS = [
#     "stone",
    "cobblestone",
    "brick_block",
#     "quartz_stairs",
#     "sandstone",
# #     # "sandstone_stairs",
# #     # "end_bricks",
# #     # "obsidian",
#     "planks",
#     "log",
#     # "oak_stairs",
# #     # "birch_stairs",
# #     # "fence",
# #     # "dirt", 
#     "gravel",
# #     # "glass",
    "glowstone",
#     # "torch",
#     # "bookshelf",
#     # "flower_pot",
# ]
# + [
    "quartz_block",        # white
    # "magma",               # orange
    "nether_wart_block",   # magenta
    "packed_ice",          # light_blue
    "gold_block",          # yellow
    "emerald_block",         # lime
    # "red_mushroom_block",  # pink-ish
    "prismarine",          # cyan
    "purpur_block",        # purple
    "lapis_block",         # blue
    "brown_mushroom_block",# brown
    "emerald_block",       # green
    "redstone_block",      # red
    "coal_block",          # black
]
BLUEPRINT_OBJECTS = list(set(BLUEPRINT_OBJECTS))
# Copyright (c) 2020 All Rights Reserved
# Author: William H. Guss, Brandon Houghton

import logging
class Wood(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    self.rewards = [
        CollectReward('log', repeated=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs, _ = self.env.step(action)
    reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    obs['reward'] = np.float32(reward)
    return obs


def shaped_reward(h_prev, h_curr, t, max_h, γ = 0.99, k_time = 0.1, B = 5.0):
    # potential‐based term
    F = γ*h_curr - h_prev
    # time penalty baked in
    R = F - k_time
    # bonus for clearance
    if h_curr > max_h:
        R += B
        max_h = h_curr
    return R

class Climb(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    self._previous = None
    self._health_reward = HealthReward()
    self.previous_height = np.array([-1e10] * 1000)
    self.starting_location = 0
    super().__init__(env)
    

  def step(self, action):
    obs, _ = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self.starting_location = height
      self._previous = height
    reward = shaped_reward(self._previous, height, self.starting_location) + self._health_reward(obs)
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs



class MineRLBasaltFindCave(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 3600)
    env = MinecraftBase(actions, name='MineRLBasaltFindCave-v0' *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    self._previous = None
    self.rewards = [
        HealthReward(),
    ]
    super().__init__(env)
    

  def step(self, action):
    obs, _ = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    reward = (height - self._previous) + height / 10.0
    # self.rewards[0](obs) + 
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs


class MineRLBasaltCreateVillageAnimalPen(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 3600)
    kwargs
    env = MinecraftBase(actions, name='MineRLBasaltCreateVillageAnimalPen-v0' *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    self._previous = None
    self.rewards = [
        HealthReward(),
    ]
    super().__init__(env)
    

  def step(self, action):
    obs, _ = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    reward = (height - self._previous) + height / 10.0
    # self.rewards[0](obs) + 
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs

class MineRLBasaltBuildVillageHouse(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 3600)
    kwargs
    env = MinecraftBase(actions, name='MineRLBasaltBuildVillageHouse-v0' *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    self._previous = None
    self.rewards = [
        HealthReward(),
    ]
    super().__init__(env)
    

  def step(self, action):
    obs, _ = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    reward = (height - self._previous) + height / 10.0
    # self.rewards[0](obs) + 
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs


class MineRLBasaltMakeWaterfall(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = BASIC_ACTIONS
    length = kwargs.pop('length', 3600)
    kwargs
    env = MinecraftBase(actions, name='MineRLBasaltMakeWaterfall-v0' *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    self._previous = None
    self.rewards = [
        HealthReward(),
    ]
    super().__init__(env)
    

  def step(self, action):
    obs, _ = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    reward = (height - self._previous) + height / 10.0
    # self.rewards[0](obs) + 
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs


# -------------------------------------------------------
# Safe access helpers – put them near the top of the file
# -------------------------------------------------------
def _unwrap_to_malmo(env):
    """
    Drill through any Gym / Embodied wrappers and return the bare Malmo env
    (the object that actually owns .agent_host).
    """
    core = env
    while hasattr(core, "env"):
        core = core.env
    return core

# ------------------------------------------------------------------ #
#  helpers – put these near the top of the file (outside the class)
# ------------------------------------------------------------------ #
def _send_and_wait(agent_host, cmd, n_ticks=1):
    agent_host.sendCommand(cmd)
    for _ in range(n_ticks):
        agent_host.sendCommand("move 0")      # advance the mission clock

def _tp(ah, x, y, z, yaw=None, pitch=None):
    """Teleport the **active** agent via chat + adjust view."""
    _send_and_wait(ah, f'chat /tp {int(x)} {int(y)} {int(z)}')
    if yaw is not None:
        _send_and_wait(ah, f'setYaw {yaw}')
    if pitch is not None:
        _send_and_wait(ah, f'setPitch {pitch}')

def _extract_pose(obs):
    """Return (x, y, z, yaw, pitch) as float32 from a MineRL observation dict."""
    return (
        np.float32(obs["location_stats/xpos"]),
        np.float32(obs["location_stats/ypos"]),
        np.float32(obs["location_stats/zpos"]),
        np.float32(obs["location_stats/yaw"]),
        np.float32(obs["location_stats/pitch"]),
    )

# lpips_metric = lpips.LPIPS(net='alex').eval().to('cuda:2')   #   1 = completely different
# to_tensor = T.Compose([T.Resize(256), T.ToTensor()])

# def lpips_similarity(img1_pil, img2_pil):
#     t1 = (img1_pil/255.0)[None]*2-1   # scale to [-1,1]
#     t2 = (img2_pil/255.0)[None]*2-1
#     t1 = torch.tensor(t1).permute(0, 3, 1, 2).to(torch.float32)
#     t2 = torch.tensor(t2).permute(0, 3, 1, 2).to(torch.float32)
#     # print(t1.shape, t1.max(), t1.min(), t2.shape, t2.max(), t2.min())
#     d  = lpips_metric(t1.to('cuda:2'), t2.to('cuda:2'))
#     # torch.save(t1.detach().cpu(), '/root/logdir/cam.pt')
#     # torch.save(t2.detach().cpu(), '/root/logdir/blueprint.pt')
#     return 1.0 - d.item()            # convert to “higher = more similar”


import collections
import numpy as np

class MovementReward:
    """
    Dense reward / penalty for movement.

    Parameters
    ----------
    every : int
        Window size (in env steps) over which to check movement.
    min_dist : float
        Minimum Euclidean distance (blocks) that counts as “moved”.
    reward_move : float
        Reward added when the agent has moved ≥ min_dist in the window.
    penalty_stuck : float
        Penalty added when the agent stayed within min_dist for the whole window.
    """
    def __init__(self, every=5, min_dist=0.5, reward_move=0.1, penalty_stuck=-0.1):
        self.every = every
        self.min_dist = min_dist
        self.reward_move = reward_move
        self.penalty_stuck = penalty_stuck
        self._pos_hist = collections.deque(maxlen=every)

    def reset(self):
        self._pos_hist.clear()

    def __call__(self, obs):
        # 1. record current position
        self._pos_hist.append(np.array(obs['player_pos'], dtype=np.float32))

        # 2. haven’t filled the window yet → no reward/penalty
        if len(self._pos_hist) < self.every:
            return 0.0

        # 3. distance travelled in the last `every` steps
        dist = np.linalg.norm(self._pos_hist[-1] - self._pos_hist[0])

        # 4. assign reward / penalty
        return self.reward_move if dist >= self.min_dist else self.penalty_stuck
from collections import deque   # (if not already imported above)

import torch
import torch.nn.functional as F

# 3. Modified environment wrapper
class Blueprints(embodied.Wrapper):
    def __init__(self, *args, **kwargs):
        BUILD_ACTIONS = {f'place_{item}': dict(place=item) for item in BLUEPRINT_OBJECTS}
        # ── movement ────────────────────────────────────────────────────────────────
        MOVE_ACTIONS = {
            "noop":         {},
            "sneak_on":     {"sneak": 1},       # precision edge-building
            "sneak_off":    {"sneak": 0},
            "sprint_on":    {"sprint": 1},      # faster traversal
            "sprint_off":   {"sprint": 0},
            'attack': dict(attack=1),
            'turn_up': dict(camera=(-15, 0)),
            'turn_down': dict(camera=(15, 0)),
            'turn_left': dict(camera=(0, -15)),
            'turn_right': dict(camera=(0, 15)),
            'forward': dict(forward=1),
            'back': dict(back=1),
            'left': dict(left=1),
            'right': dict(right=1),
            'jump': dict(jump=1, forward=1),
        }
        # ── full action dictionary ─────────────────────────────────────────────────
        ACTIONS = {
            **MOVE_ACTIONS,
            **BUILD_ACTIONS,
        }
        length = kwargs.pop('length', 36000)
        env = MinecraftBase(ACTIONS, *args, **kwargs)
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

        self.image_size = env._size

        if True:
          from transformers import CLIPProcessor, CLIPModel
          import torch
          self.device = 'cuda:1'
          # 1. Load CLIP model and processor
          clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval().to(torch.bfloat16)
          clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

          
          self.clip_model = clip_model
          self.clip_processor = clip_processor

        # self.max_reward = -1
        # self.rewards = [
        #     HealthReward(),
        # ]
        self.move_reward = MovementReward(every=10, min_dist=0.5, reward_move=0.05, penalty_stuck=0.0)

        self.reward_every = kwargs.pop('reward_every', 5)
        self.spawn_pos   = None   # filled in on reset
        self._anchor_yaw   = None   # filled on first _capture_from_vantage()
        self._anchor_pitch = None
        self._yaw_halfspan   = 45    # ±35°  →  70° total horizontal FOV
        self._pitch_halfspan = 25    # ±25°

        self._inv_log_keys = []
        import glob
        self.all_blueprints = list(glob.iglob('blueprints/minecraft/*'))
        self.all_blueprints_real = list(glob.iglob('blueprints/images/*'))

        self._stuck_window   = kwargs.pop('stuck_window', 50)    # steps in one window
        self._stuck_epsilon  = kwargs.pop('stuck_epsilon', 0.75) # min displacement to count as movement
        self._stuck_patience = kwargs.pop('stuck_patience', 16)   # how many *consecutive* bad windows trigger reset
        self._pos_hist = deque(maxlen=self._stuck_window)
        self._stuck_windows = 0

        self.alpha = kwargs.pop('clip_alpha', 0.2)  # weight for global vs patch

        # In __init__
        self.best_sim = -1.0
        self.prev_sim = None
        self.imp_scale = 10.0      # improvement bonus weight
        self.dense_scale = 1.0     # small dense shaping weight

        # When you handle reset (both external and forced)
        self.best_sim = -1.0
        self.prev_sim = None
        self.move_reward.reset()


    def _reset_stuck_tracking(self):
        self._pos_hist.clear()
        self._stuck_windows = 0
      
    def _update_and_check_stuck(self, pos_xyz):
        self._pos_hist.append(np.asarray(pos_xyz, dtype=np.float32))
        # Need a full window before evaluating
        if len(self._pos_hist) < self._stuck_window:
            return False
        disp = np.linalg.norm(self._pos_hist[-1] - self._pos_hist[0])
        if disp < self._stuck_epsilon:
            self._stuck_windows += 1
        else:
            self._stuck_windows = 0
        return self._stuck_windows > self._stuck_patience
    
    def load_blueprint(self,):
      if random.random() < 0.5:
        self.blueprint_image = Image.open(random.choice(self.all_blueprints)).convert("RGB").resize(self.image_size)
      else:
        self.blueprint_image = Image.open(random.choice(self.all_blueprints_real)).convert("RGB").resize(self.image_size)
      self.blueprint_image = np.array(self.blueprint_image)

      if self.clip_model is not None:
        img = Image.open(random.choice(self.all_blueprints)).convert("RGB")
        self.blueprint_image = np.array(img.resize(self.image_size))  # for observation panel

        # Precompute CLIP features on device
        with torch.no_grad():
            self.blueprint_global  = self._clip_encode_global(img)  # [1, D] float32 on device
            self.blueprint_patches = self._clip_patch_tokens(img)

    def _capture_from_vantage(self, location_stats_all):
      first_init = False
      
      if self.spawn_pos is None:
        first_init = True
        self.spawn_pos = location_stats_all
      sx, sy, sz, syaw, spitch = self.spawn_pos
      if self._anchor_yaw is None:
        self._anchor_yaw   = syaw
        self._anchor_pitch = spitch

      flight_height = 3
      self.env._gymenv.set_next_chat_message(f"/setblock {sx} {sy+flight_height} {sz} barrier")
      for _ in range(4):
        _ = self.env.step({'action': np.array(0, dtype=np.int32), 'reset': np.array(False)})
      self.env._gymenv.set_next_chat_message(f"/tp @a {sx} {sy+flight_height+2} {sz} {syaw} {spitch-5}")
      for _ in range(4):
        obs = self.env.step({'action': np.array(0, dtype=np.int32), 'reset': np.array(False)})
      init_pov = obs['image']
      sx, sy, sz, syaw, spitch = location_stats_all
      self.env._env._env.set_next_chat_message(f"/tp @a {sx} {sy} {sz} {syaw} {spitch}")
      for _ in range(4):
        _ = self.env.step({'action': np.array(0, dtype=np.int32), 'reset': np.array(False)})
      if first_init:
        self.spawn_pos = location_stats_all
        fwd_idx = self.env.act_names.index('forward') 
        for _ in range(random.randint(18,24)):
            self.env.step({'action': np.array(fwd_idx), 'reset': np.array(False)})
      return init_pov

    @torch.no_grad()
    def _clip_encode_global(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Returns L2-normalized global CLIP image embedding [1, D] (float32, on self.device).
        """
        clip_inputs = self.clip_processor(images=img_pil, return_tensors="pt").to(self.device)
        # keep compute stable in f32 for similarity math; convert just before model if you prefer bf16
        pixel_values = clip_inputs["pixel_values"]
        # If you really want bf16: pixel_values = pixel_values.to(torch.bfloat16)
        feats = self.clip_model.get_image_features(pixel_values=pixel_values)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.float()  # [1, D]

    @torch.no_grad()
    def _clip_patch_tokens(self, img_pil: Image.Image) -> torch.Tensor:
        """
        Returns L2-normalized patch tokens (orderless) with CLS removed: [P, D]
        Uses HF CLIP vision tower to grab last hidden states.
        """
        clip_inputs = self.clip_processor(images=img_pil, return_tensors="pt").to(self.device)
        pixel_values = clip_inputs["pixel_values"]
        # Forward the vision tower to get token sequence
        vision_out = self.clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=False)
        tokens = vision_out.last_hidden_state  # [B, 1+P, D_model]
        # Apply post layernorm if present (stabilizes scales)
        post_ln = getattr(self.clip_model.vision_model, "post_layernorm", None)
        if post_ln is not None:
            tokens = post_ln(tokens)
        # Drop CLS
        tokens = tokens[:, 1:, :]            # [B, P, D]
        tokens = tokens.reshape(-1, tokens.shape[-1])  # [P, D]
        tokens = F.normalize(tokens, dim=-1)
        return tokens.float()

    @torch.no_grad()
    def _chamfer_similarity(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Cosine-based symmetric Chamfer similarity on L2-normalized token sets.
        A: [M, D], B: [N, D] on same device. Returns scalar tensor.
        """
        # cosine distance = 1 - cosine sim
        d = 1.0 - (A @ B.T)         # [M, N]
        a2b = d.min(dim=1).values.mean()
        b2a = d.min(dim=0).values.mean()
        chamfer = 0.5 * (a2b + b2a)
        return 1.0 - chamfer        # higher is better

    @torch.no_grad()
    def _combined_clip_similarity(self, frame_rgb: np.ndarray) -> tuple[float, float, float]:
        """
        Compute (global_sim, patch_sim, combined_sim) between current frame and blueprint.
        Returns floats.
        """
        img = Image.fromarray(frame_rgb).convert("RGB")
        # global
        g_frame = self._clip_encode_global(img)            # [1, D]
        g_bp    = self.blueprint_global                    # [1, D]
        g_sim = F.cosine_similarity(g_frame, g_bp, dim=-1).item()

        # patch
        p_frame = self._clip_patch_tokens(img)             # [Pf, D]
        p_bp    = self.blueprint_patches                   # [Pb, D]
        p_sim = self._chamfer_similarity(p_frame, p_bp).item()

        # blend
        combined = self.alpha * g_sim + (1.0 - self.alpha) * p_sim
        return g_sim, p_sim, combined


    def _compute_clip_reward(self, frame_rgb):
        """
        Kept the old method name for minimal changes elsewhere.
        Now returns the *combined* score; also stores per-component sims for logging.
        """
        g, p, c = self._combined_clip_similarity(frame_rgb)
        # Optional: keep for debugging/telemetry
        self._last_clip_global = g
        self._last_clip_patch  = p
        return c

    def _dump_gif(self):
      """Write out and clear the frame buffer."""
      if not self._frames:
          return
      # unique name: <wrapper>_<idx>_<epoch>.gif  (easy to grep later)
      fname = self._gif_dir / f"{self.__class__.__name__.lower()}_" \
                              f"{self._gif_index:04d}_{int(time.time())}.gif"
      self._frames[0].save(
          fname,
          save_all=True,
          append_images=self._frames[1:],
          duration=100,          # ≈10 fps; tweak as you like
          loop=0
      )
      print(f"[GIF] wrote {fname}")
      self._frames.clear()
      self._gif_index += 1
    
    def _hard_reset(self):
        # One place to do all resets + reinitialization.
        forced = {'action': np.array(0, np.int32), 'reset': np.array(True)}
        obs = self.env.step(forced)

        # Reinit your own state
        self.spawn_pos = None
        self.move_reward.reset()
        self.load_blueprint()
        self._reset_stuck_tracking()
        self.prev_sim = None
        self.best_sim = -1.0
        return obs
    
    def step(self, action):
      # print('action', action)
      step_idx = self.env._step
      if action['reset']:
          self.spawn_pos = None
          self.move_reward.reset()
          self.load_blueprint()
          self._reset_stuck_tracking()   # <<< reset tracking on external reset
          self.env._gymenv.set_next_chat_message(f"/viewdistance 32")
          self.env._gymenv.set_next_chat_message(f"/simulationdistance 32")
          self.prev_sim = None
          self.best_sim = -1.0
          
      if self._need_reset:
          self._need_reset = False
          obs = self._hard_reset()
      else:
          obs = self.env.step(action)

      if not obs["is_first"]:
          pos = obs['player_pos']
          if self._update_and_check_stuck(pos):
              # Force reset
              obs['is_last'] = True
              obs['reward'] = np.float32(obs['reward'] - 0.5)
              self._need_reset = True
      else:
          # First frame of an episode: clear tracking and seed with current position
          self._reset_stuck_tracking()
          self._pos_hist.append(np.asarray(obs['player_pos'], dtype=np.float32))

      if obs["is_first"]:
        for _ in range(30):
          self.env.step({'action': np.array(0, dtype=np.int32), 'reset': np.array(False)})

      # print(step_idx)
      if (step_idx % self.reward_every) == 0:
        self.last_frame = self._capture_from_vantage(obs['location_stats_all'])
        self.last_frame = np.reshape(self.last_frame, [self.image_size[0], self.image_size[1], 3])
      del obs['location_stats_all']
      sim = self._compute_clip_reward(self.last_frame)
      if self.prev_sim is None:          # first tick of an episode
          self.prev_sim = sim
          self.best_sim = sim
      dense = (sim - self.prev_sim) * self.dense_scale          # can be +/- small
      imp   = max(0.0, sim - self.best_sim) * self.imp_scale    # only when you set a new record
      self.prev_sim = sim
      self.best_sim = max(self.best_sim, sim)
      obs['reward'] = np.float32(dense + imp)

      obs['image'] = np.reshape(obs['image'], [self.image_size[0], self.image_size[1], 3])
      obs['image'] = np.concatenate([obs['image'], self.last_frame])
      obs['image'] = np.concatenate([obs['image'], self.blueprint_image])
      return obs


class Diamond(embodied.Wrapper):

  def __init__(self, *args, **kwargs):
    actions = {
        **BASIC_ACTIONS,
        'craft_planks': dict(craft='planks'),
        'craft_stick': dict(craft='stick'),
        'craft_crafting_table': dict(craft='crafting_table'),
        'place_crafting_table': dict(place='crafting_table'),
        'craft_wooden_pickaxe': dict(nearbyCraft='wooden_pickaxe'),
        'craft_stone_pickaxe': dict(nearbyCraft='stone_pickaxe'),
        'craft_iron_pickaxe': dict(nearbyCraft='iron_pickaxe'),
        'equip_stone_pickaxe': dict(equip='stone_pickaxe'),
        'equip_wooden_pickaxe': dict(equip='wooden_pickaxe'),
        'equip_iron_pickaxe': dict(equip='iron_pickaxe'),
        'craft_furnace': dict(nearbyCraft='furnace'),
        'place_furnace': dict(place='furnace'),
        'smelt_iron_ingot': dict(nearbySmelt='iron_ingot'),
    }
    self.rewards = [
        CollectReward('log', once=1),
        CollectReward('planks', once=1),
        CollectReward('stick', once=1),
        CollectReward('crafting_table', once=1),
        CollectReward('wooden_pickaxe', once=1),
        CollectReward('cobblestone', once=1),
        CollectReward('stone_pickaxe', once=1),
        CollectReward('iron_ore', once=1),
        CollectReward('furnace', once=1),
        CollectReward('iron_ingot', once=1),
        CollectReward('iron_pickaxe', once=1),
        CollectReward('diamond', once=1),
        HealthReward(),
    ]
    length = kwargs.pop('length', 36000)
    env = MinecraftBase(actions, *args, **kwargs)
    env = embodied.wrappers.TimeLimit(env, length)
    super().__init__(env)

  def step(self, action):
    obs, _ = self.env.step(action)
    reward = sum([fn(obs, self.env.inventory) for fn in self.rewards])
    obs['reward'] = np.float32(reward)
    return obs


BASIC_ACTIONS = {
    'noop': dict(),
    'attack': dict(attack=1),
    'turn_up': dict(camera=(-15, 0)),
    'turn_down': dict(camera=(15, 0)),
    'turn_left': dict(camera=(0, -15)),
    'turn_right': dict(camera=(0, 15)),
    'forward': dict(forward=1),
    'back': dict(back=1),
    'left': dict(left=1),
    'right': dict(right=1),
    'jump': dict(jump=1, forward=1),
    'place_dirt': dict(place='dirt'),
}


class CollectReward:

  def __init__(self, item, once=0, repeated=0):
    self.item = item
    self.once = once
    self.repeated = repeated
    self.previous = 0
    self.maximum = 0

  def __call__(self, obs, inventory):
    current = inventory[self.item]
    if obs['is_first']:
      self.previous = current
      self.maximum = current
      return 0
    reward = self.repeated * max(0, current - self.previous)
    if self.maximum == 0 and current > 0:
      reward += self.once
    self.previous = current
    self.maximum = max(self.maximum, current)
    return reward


class HealthReward:

  def __init__(self, scale=0.01):
    self.scale = scale
    self.previous = None

  def __call__(self, obs, inventory=None):
    health = obs['health']
    if obs['is_first']:
      self.previous = health
      return 0
    reward = self.scale * (health - self.previous)
    self.previous = health
    return np.float32(reward)

# import torch
# WOOL_COLORS = [
#         "white", "orange", "magenta", "light_blue",
#         "yellow", "lime", "pink", "gray",
#         "light_gray", "cyan", "purple", "blue",
#         "brown", "green", "red", "black",
#     ]

# WOOL_IDS = [f"{c}_wool" for c in WOOL_COLORS]
class MinecraftBase(embodied.Env):

  LOCK = threading.Lock()
  NOOP = dict(
      camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
      jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
      place='none', equip='none')

  def __init__(
      self, actions,
      repeat=1,
      size=(64, 64),
      break_speed=100.0,
      gamma=10.0,
      sticky_attack=30,
      sticky_jump=10,
      pitch_limit=(-60, 60),
      log_inv_keys=('wool', 'log', 'cobblestone', 'iron_ingot', 'diamond'),
      logs=False,
      name=None,
      vlm=None,
      embedder=None,
      use_action=False,
  ):
    if logs:
      logging.basicConfig(level=logging.DEBUG)
    self._repeat = repeat
    self._size = size
    if break_speed != 1.0:
      sticky_attack = 0

    
    # Make env
    with self.LOCK:
      if name is not None:
        import gym
        self._gymenv = gym.make(name)
      else:
        self._gymenv = MineRLEnv(size, break_speed).make()
    from . import from_gym
    self._env = from_gym.FromGym(self._gymenv)
    self._inventory = {}

    # Observations
    self._inv_keys = [
        k for k in self._env.obs_space if k.startswith('inventory/')
        if k != 'inventory/log2']
    # print(self._env.obs_space)
        # Dump all observation-space keys to ./obs_space_keys.txt
    print(self._env.obs_space.keys())
    # with open("obs_space_keys.txt", "w", encoding="utf-8") as f:
    #     for key in self._env.obs_space:
    #         f.write(f"{key}\n")        # one key per line


    # raise
    self._inv_log_keys = [f'inventory/{k}' for k in log_inv_keys]
    # missing_from_log = [k for k in self._inv_keys if k not in self._inv_log_keys]
    # # …and, if you also need the other direction:
    # missing_from_keys = [k for k in self._inv_log_keys if k not in self._inv_keys]
    
    assert all(k in self._inv_keys for k in self._inv_log_keys), (
        self._inv_keys, self._inv_log_keys)
    self._step = 0
    self._max_inventory = None
    self._equip_enum = self._gymenv.observation_space[
        'equipped_items']['mainhand']['type'].values.tolist()
    
    self.vlm = vlm
    self.embedder = embedder
    self._obs_space = self.obs_space

    # Actions
    actions = self._insert_defaults(actions)
    self._action_names = tuple(actions.keys())
    self._action_values = tuple(actions.values())
    # message = f'Minecraft action space ({len(self._action_values)}):'
    # print(message, ', '.join(self._action_names))
    self._sticky_attack_length = sticky_attack
    self._sticky_attack_counter = 0
    self._sticky_jump_length = sticky_jump
    self._sticky_jump_counter = 0
    self._pitch_limit = pitch_limit
    self._pitch = 0

    self.action_cache = []
    self.max_actions = 30
    self.use_action = use_action
    self.last_instructiwon_step = 0
    self.instr_interval = 30
    self.min_instr_interval = 5
    self.dropout_rate = 0.15
      
  @property
  def act_names(self):
    return list(self._action_names)
  
  @property
  def obs_space(self):
    space = {
        # 'image': elements.Space(np.uint8, self._size + (3,)),
        'image': elements.Space(np.uint8, (self._size[0]*3, self._size[0]) + (3,)),
        # 'inventory': elements.Space(np.float32, len(self._inv_keys), 0),
        # 'inventory_max': elements.Space(np.float32, len(self._inv_keys), 0),
        # 'equipped': elements.Space(np.float32, len(self._equip_enum), 0, 1),
        'reward': elements.Space(np.float32),
        # 'health': elements.Space(np.float32),
        # 'hunger': elements.Space(np.float32),
        # 'breath': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        # **{f'log/{k}': elements.Space(np.int64) for k in self._inv_log_keys},
        'player_pos': elements.Space(np.float32, 3),
    }
    if self.vlm is not None:
      # spaces['instructions'] = elements.Space(np.float32, 384)
      space['instructions_ids'] = elements.Space(np.uint8, 32)
      space['action_ids'] = elements.Space(np.int32)
    return space
  
  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, len(self._action_values)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    # action = action.copy()
    # self.action_cache.append(action)
    # if len(self.action_cache) > self.max_actions:
    #   self.action_cache = self.action_cache[-self.max_action:]

    index = action.pop('action')
    action.update(self._action_values[index])
    action = self._action(action)
    if action['reset']:
      obs = self._reset()
    else:
      # following = self.NOOP.copy()
      # for key in ('attack', 'forward', 'back', 'left', 'right'):
      #   following[key] = action[key]
      for act in [action]:
      # + ([following] * (self._repeat - 1)):
        obs = self._env.step(act)
        if self._env.info and 'error' in self._env.info:
          obs = self._reset()
          break
    obs = self._obs(obs)
    self._step += 1
    assert 'pov' not in obs, list(obs.keys())
    return obs

  @property
  def inventory(self):
    return self._inventory

  def _reset(self):
    with self.LOCK:
      obs = self._env.step({'reset': True})
    self._step = 0
    self._max_inventory = None
    self._sticky_attack_counter = 0
    self._sticky_jump_counter = 0
    self._pitch = 0
    self._inventory = {}
    return obs

  def _obs(self, obs):
    # for key in obs:
    #   if 'wool' in key:
    #     print(key, obs[key])
    # print(self._inv_keys)
    obs['inventory/log'] += obs.pop('inventory/log2')
    self._inventory = {
        k.split('/', 1)[1]: obs[k] for k in self._inv_keys
        if k != 'inventory/air'}
    inventory = np.array([obs[k] for k in self._inv_keys], np.float32)
    if self._max_inventory is None:
      self._max_inventory = inventory
    else:
      self._max_inventory = np.maximum(self._max_inventory, inventory)
    index = self._equip_enum.index(obs['equipped_items/mainhand/type'])
    equipped = np.zeros(len(self._equip_enum), np.float32)
    equipped[index] = 1.0
    player_x = obs['location_stats/xpos']
    player_y = obs['location_stats/ypos']
    player_z = obs['location_stats/zpos']
    # for key in obs:
    #   # if 'wool' in key:
        # print(key)
    obs = {
        'image': obs['pov'],
        # 'image': 
        # elements.Space(np.uint8, (128, 64) + (3,)),
        # 'inventory': inventory,
        # 'inventory_max': self._max_inventory.copy(),
        # 'equipped': equipped,
        # 'health': np.float32(obs['life_stats/life'] / 20),
        # 'hunger': np.float32(obs['life_stats/food'] / 20),
        # 'breath': np.float32(obs['life_stats/air'] / 300),
        'reward': np.float32(0.0),
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
        # **{f'log/{k}': np.int64(obs[k]) for k in self._inv_log_keys},
        'player_pos': np.array([player_x, player_y, player_z], np.float32),
        'location_stats_all': [
          np.float32(obs["location_stats/xpos"]),
          np.float32(obs["location_stats/ypos"]),
          np.float32(obs["location_stats/zpos"]),
          np.float32(obs["location_stats/yaw"]),
          np.float32(obs["location_stats/pitch"]),
        ]
    }
    if self.vlm is not None:
      # spaces['instructions'] = elements.Space(np.float32, 384)
      obs['instructions_ids'] = np.zeros(32)
      obs['action_ids'] = np.array(-100)
    # for key, value in obs.items():
    #   if key == "location_stats_all":
    #     continue
    #   space = self._obs_space[key]
    #   if not isinstance(value, np.ndarray):
    #     value = np.array(value)
    #   assert value in space, (key, value, value.dtype, value.shape, space)

    return obs

  def _action(self, action):
    if self._sticky_attack_length:
      if action['attack']:
        self._sticky_attack_counter = self._sticky_attack_length
      if self._sticky_attack_counter > 0:
        action['attack'] = 1
        action['jump'] = 0
        self._sticky_attack_counter -= 1
    if self._sticky_jump_length:
      if action['jump']:
        self._sticky_jump_counter = self._sticky_jump_length
      if self._sticky_jump_counter > 0:
        action['jump'] = 1
        action['forward'] = 1
        self._sticky_jump_counter -= 1
    if self._pitch_limit and action['camera'][0]:
      lo, hi = self._pitch_limit
      if not (lo <= self._pitch + action['camera'][0] <= hi):
        action['camera'] = (0, action['camera'][1])
      self._pitch += action['camera'][0]
    return action

  def _insert_defaults(self, actions):
    actions = {name: action.copy() for name, action in actions.items()}
    for key, default in self.NOOP.items():
      for action in actions.values():
        if key not in action:
          action[key] = default
    return actions

# from minerl.herobraine.hero.handlers.agent.start import CreativeInventoryAgentStart
class MineRLEnv(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=50):
    self.resolution = resolution
    self.break_speed = break_speed
    super().__init__(name='FlatMineRLEnv-v0')
    # super().__init__(name='MineRLEnv-v1')

  # def create_agent_start(self):
  #   return [BreakSpeedMultiplier(self.break_speed)]

  def create_agent_handlers(self):
    return []
  
  def create_server_world_generators(self):
    # superflat preset: three layers of dirt on bedrock
    return [
        handlers.FlatWorldGenerator(
            generatorString="1;7,2x3,2;1",
            force_reset=True,
        )
    ]

  # def create_server_world_generators(self):
  #   return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=False, start_time=0),
        handlers.SpawningInitialCondition(allow_spawning=False),
        # handlers.GameMode('creative'),
    ]

  def create_observables(self):
    return [
        handlers.POVObservation(self.resolution),
        handlers.FlatInventoryObservation(mc.ALL_ITEMS),
        handlers.EquippedItemObservation(
            mc.ALL_ITEMS, _default='air', _other='other'),
        handlers.ObservationFromCurrentLocation(),
        handlers.ObservationFromLifeStats(),
    ]
  
  def create_agent_start(self):
    # WOOL_ITEMS = [
    #     "white_wool", "orange_wool", "magenta_wool", "light_blue_wool",
    #     "yellow_wool", "lime_wool", "pink_wool", "gray_wool",
    #     "light_gray_wool", "cyan_wool", "purple_wool", "blue_wool",
    #     "brown_wool", "green_wool", "red_wool", "black_wool",
    # # ]
    # WOOL_COLORS = [
    #     "white", "orange", "magenta", "light_blue",
    #     "yellow", "lime", "pink", "gray",
    #     "light_gray", "cyan", "purple", "blue",
    #     "brown", "green", "red", "black",
    # ]


    return [
        
        # CreativeInventoryAgentStart()
        handlers.SimpleInventoryAgentStart(
            [dict(type=obj, quantity=32) for obj in BLUEPRINT_OBJECTS]
        ),
        # handlers.AgentStartPlacement(x=0.5, y=4, z=0.5, yaw=0, pitch=0),
        # handlers.SimpleInventoryAgentStart(
        #     [dict(type=obj, quantity=64) for obj in objects]
        # ),
        BreakSpeedMultiplier(1000000),
    ]
  
  def create_actionables(self):
    kw = dict(_other='none', _default='none')
    return [
        handlers.KeybasedCommandAction('forward', INVERSE_KEYMAP['forward']),
        handlers.KeybasedCommandAction('back', INVERSE_KEYMAP['back']),
        handlers.KeybasedCommandAction('left', INVERSE_KEYMAP['left']),
        handlers.KeybasedCommandAction('right', INVERSE_KEYMAP['right']),
        handlers.KeybasedCommandAction('jump', INVERSE_KEYMAP['jump']),
        handlers.KeybasedCommandAction('sneak', INVERSE_KEYMAP['sneak']),
        handlers.KeybasedCommandAction('attack', INVERSE_KEYMAP['attack']),
        handlers.CameraAction(),
        handlers.PlaceBlock(['none'] + mc.ALL_ITEMS, **kw),
        handlers.EquipAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftNearbyAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.SmeltItemNearby(['none'] + mc.ALL_ITEMS, **kw),
        handlers.ChatAction()
    ]

  def is_from_folder(self, folder):
    return folder == 'none'

  def get_docstring(self):
    return ''

  def determine_success_from_rewards(self, rewards):
    return True

  def create_rewardables(self):
    return []

  def create_server_decorators(self):
    return []
  #       return [
  #           ServerPropertiesDecorator({
  #               "view-distance": "32",        # default 10, max 32
  #               "simulation-distance": "32",  # optional, keeps redstone/mob AI active
  #           })
  #       ]

  def create_mission_handlers(self):
    return []

  def create_monitors(self):
    return []

# from minerl.herobraine.hero.handler import Handler
# class ServerPropertiesDecorator(Handler):
#     """
#     Injects <ServerSection><ServerProperties>…</ServerProperties></ServerSection>
#     so you can tweak *view‑distance*, *simulation‑distance*, etc.
#     """

#     def __init__(self, props: dict[str, str]):
#         super().__init__()
#         self.props = props

#     # ---------- XML that Malmo understands ----------
#     def xml_template(self) -> str:
#         body = "".join(f"<{k}>{v}</{k}>" for k, v in self.props.items())
#         return f"<ServerSection><ServerProperties>{body}</ServerProperties></ServerSection>"

#     # ---------- required stub ----------
#     def to_string(self) -> str:
#         # must be unique within the spec
#         return "server_props"
    
class BreakSpeedMultiplier(handler.Handler):

  def __init__(self, multiplier=1.0):
    self.multiplier = multiplier

  def to_string(self):
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'
