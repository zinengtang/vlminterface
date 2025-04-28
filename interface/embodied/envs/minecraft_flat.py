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

from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


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
    obs = self.env.step(action)
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
    obs = self.env.step(action)
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
    obs = self.env.step(action)
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
    obs = self.env.step(action)
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
    obs = self.env.step(action)
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
    obs = self.env.step(action)
    x, y, z = obs['player_pos']
    height = np.float32(y)
    if obs['is_first']:
      self._previous = height
    reward = (height - self._previous) + height / 10.0
    # self.rewards[0](obs) + 
    obs['reward'] = np.float32(reward)
    self._previous = height
    return obs


# 3. Modified environment wrapper
class Blueprints(embodied.Wrapper):
    def __init__(self, *args, **kwargs):
        actions = {
          **BASIC_ACTIONS,
          'place_planks': dict(place='planks'),
          'place_cobblestone': dict(place='cobblestone'),
          'place_log': dict(place='log'),
      }
        length = kwargs.pop('length', 36000)
        env = MinecraftBase(actions, *args, **kwargs)
        env = embodied.wrappers.TimeLimit(env, length)
        super().__init__(env)

        from transformers import CLIPProcessor, CLIPModel
        import torch
        from PIL import Image

        self.device = 'cuda:0'
        # 1. Load CLIP model and processor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval().to(torch.bfloat16)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # 2. Precompute the blueprint embedding once
        blueprint_image = Image.open("blueprints/real/eiffel_tower.jpg").convert("RGB")
        inputs = clip_processor(images=blueprint_image, return_tensors="pt").to(self.device).to(torch.bfloat16)
        with torch.no_grad():
            blueprint_emb = clip_model.get_image_features(**inputs)
            blueprint_emb = (blueprint_emb / blueprint_emb.norm(p=2, dim=-1, keepdim=True)).detach().cpu()

        # Attach CLIP components
        self.blueprint_emb = blueprint_emb  # precomputed (1, dim)
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.cur_reward = -1e10
        self.rewards = [
            HealthReward(),
        ]

        self.starting_inventory = {
            "acacia_log": 64,
            "black_dye": 64,
            "blue_dye": 64,
            "brown_dye": 64,
            "cactus": 64,
            "cobblestone": 64,
            "cobweb": 64,
            "dirt": 64,
            "flower_pot": 64,
            "glass_pane": 64,
            "grass_block": 64,
            "green_dye": 64,
            "jungle_log": 64,
            "lantern": 64,
            "oak_log": 64,
            "packed_ice": 64,
            "poppy": 64,
            "red_dye": 64,
            "sand": 64,
            "sandstone": 64,
            "smooth_sandstone": 64,
            "snow_block": 64,
            "spruce_log": 64,
            "stone_axe": 1,
            "stone_pickaxe": 1,
            "terracotta": 64,
            "torch": 64,
            "white_dye": 64,
            "white_wool": 64,
            "yellow_dye": 64
        }
        
    
    def reset(self):
        # First, reset the inner env and get initial obs
        with self.LOCK:
          obs = self._env.step({'reset': True})
        # Override the inventory before the first action
        # Update the inner env inventory dict
        inv = self._inventory
        for item, count in self.starting_items.items():
            inv[item] = count
        # Recompute inventory vector and max_inventory
        inv_keys = self.env._inv_keys  # e.g. ['inventory/log', ...]
        inventory = np.array([inv[k.split('/',1)[1]] for k in inv_keys], np.float32)
        self.env._max_inventory = inventory.copy()
        # Update obs fields
        obs['inventory'] = inventory
        obs['inventory_max'] = inventory.copy()
        for item, count in self.starting_items.items():
            obs[f'log/{item}'] = np.int64(count)
        return obs
    
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
    
    def step(self, action):
    
        cur_step = self.env._step
        obs = self.env.step(action)

        if cur_step % 10 == 0 or self.cur_reward is None:
          # Convert current frame to PIL image
          frame = Image.fromarray(obs['image']).convert("RGB")
          # Preprocess for CLIP
          inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device).to(torch.bfloat16)
          with torch.no_grad():
              img_emb = self.clip_model.get_image_features(**inputs)
              img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
              img_emb = img_emb.detach().cpu()
              # Cosine similarity between frame and blueprint
              sim = torch.cosine_similarity(img_emb, self.blueprint_emb, dim=-1).item()
          self.cur_reward = max(np.float32(sim), self.cur_reward)
        # Use similarity as reward (optionally scale or combine)
        obs['reward'] = self.cur_reward
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
    obs = self.env.step(action)
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

import torch
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
      log_inv_keys=('log', 'cobblestone', 'iron_ingot', 'diamond'),
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
    self._inv_log_keys = [f'inventory/{k}' for k in log_inv_keys]
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
    message = f'Minecraft action space ({len(self._action_values)}):'
    print(message, ', '.join(self._action_names))
    self._sticky_attack_length = sticky_attack
    self._sticky_attack_counter = 0
    self._sticky_jump_length = sticky_jump
    self._sticky_jump_counter = 0
    self._pitch_limit = pitch_limit
    self._pitch = 0

    self.action_cache = []
    self.max_actions = 30
    self.use_action = use_action
    self.last_instruction_step = 0
    self.instr_interval = 30
    self.min_instr_interval = 5
    self.dropout_rate = 0.15
    

  def sample_with_vlm(self, frames, actions, check_proactive=False):
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
          # 3. encode (frozen)
          embeds, caption_ids = self.embedder.encode(captions, 32)
          embeds = embeds.detach().cpu().float().numpy()[0]
          caption_ids = caption_ids.detach().cpu().numpy()[0].astype(np.uint8)
        return embeds, caption_ids
  
  @property
  def obs_space(self):
    space = {
        'image': elements.Space(np.uint8, self._size + (3,)),
        'inventory': elements.Space(np.float32, len(self._inv_keys), 0),
        'inventory_max': elements.Space(np.float32, len(self._inv_keys), 0),
        'equipped': elements.Space(np.float32, len(self._equip_enum), 0, 1),
        'reward': elements.Space(np.float32),
        'health': elements.Space(np.float32),
        'hunger': elements.Space(np.float32),
        'breath': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        **{f'log/{k}': elements.Space(np.int64) for k in self._inv_log_keys},
        'player_pos': elements.Space(np.float32, 3),
    }
    if self.vlm is not None:
      space['instructions'] = elements.Space(np.float32, 384)
      space['instructions_ids'] = elements.Space(np.uint8, 32)
    return space
  
  @property
  def act_space(self):
    return {
        'action': elements.Space(np.int32, (), 0, len(self._action_values)),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    action = action.copy()
    self.action_cache.append(action)
    if len(self.action_cache) > self.max_actions:
      self.action_cache = self.action_cache[-self.max_action:]

    index = action.pop('action')
    action.update(self._action_values[index])
    action = self._action(action)
    if action['reset']:
      obs = self._reset()
    else:
      following = self.NOOP.copy()
      for key in ('attack', 'forward', 'back', 'left', 'right'):
        following[key] = action[key]
      for act in [action] + ([following] * (self._repeat - 1)):
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

    if self.vlm is not None and random.random() > self.dropout_rate:
      if self._step % self.min_instr_interval == 0:
        proactive = self.sample_with_vlm(
          [pil_frame], 
          self.action_cache, 
          check_proactive=True
        )
      else:
        proactive = False
      if obs['is_first'] or (self._step % self.instr_interval == 0) or proactive:
        pil_frame = Image.fromarray(obs['pov'])
        self._last_instr_embed, self._last_instr_ids = self.sample_with_vlm([pil_frame], self.action_cache)
    
    obs = {
        'image': obs['pov'],
        'inventory': inventory,
        'inventory_max': self._max_inventory.copy(),
        'equipped': equipped,
        'health': np.float32(obs['life_stats/life'] / 20),
        'hunger': np.float32(obs['life_stats/food'] / 20),
        'breath': np.float32(obs['life_stats/air'] / 300),
        'reward': np.float32(0.0),
        'is_first': obs['is_first'],
        'is_last': obs['is_last'],
        'is_terminal': obs['is_terminal'],
        **{f'log/{k}': np.int64(obs[k]) for k in self._inv_log_keys},
        'player_pos': np.array([player_x, player_y, player_z], np.float32),
    }
    if self.vlm is not None:
      obs['instructions'] = self._last_instr_embed
      obs['instructions_ids'] = self._last_instr_ids
    for key, value in obs.items():
      space = self._obs_space[key]
      if not isinstance(value, np.ndarray):
        value = np.array(value)
      assert value in space, (key, value, value.dtype, value.shape, space)
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


class MineRLEnv(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=50):
    self.resolution = resolution
    self.break_speed = break_speed
    super().__init__(name='MineRLEnv-v1')

  def create_agent_start(self):
    return [BreakSpeedMultiplier(self.break_speed)]

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True, start_time=0),
        handlers.SpawningInitialCondition(allow_spawning=True),
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

  def create_mission_handlers(self):
    return []

  def create_monitors(self):
    return []


class BreakSpeedMultiplier(handler.Handler):

  def __init__(self, multiplier=1.0):
    self.multiplier = multiplier

  def to_string(self):
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'
