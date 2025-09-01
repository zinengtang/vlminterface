"""
TeamCraftEmbodied: Refactored observation processing with structured keys
"""

import atexit
import json
import os
import random
import socket
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from teamcraft import teamCraft
from teamcraft.utils import NpEncoder, filter_voxel
from teamcraft.minecraft import MCServerManager
from datetime import datetime, timezone
import gymnasium as gym
import random
import math
import json
import os

import numpy as np

import elements
import embodied

# Optional HF tokenizer
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# Optional fcntl lock (Linux/macOS). If unavailable, lock is a no-op.
try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None


# ========================= User-tunable defaults ==============================

TEAMCRAFT_SERVER_HOST = "http://localhost"
TEAMCRAFT_SERVER_PORT: Optional[int] = None
REQUEST_TIMEOUT_SEC   = 600

CONTROLLED_BOTS = ["bot1", "bot2"]  # Note: actual bot names from metadata
IMAGE_SIZE      = (64, 64)
MOVE_SECONDS    = 0.7
DEFAULT_TOKENIZER_NAME = "nreimers/MiniLM-L6-H384-uncased"

DEFAULT_MC_PORT: Optional[int] = None
DEFAULT_TASK_NAME = "smelt"
DEFAULT_OUTPUT    = "./outputs"

# Allocation ranges
MC_PORT_RANGE = (25565, 25999)
HTTP_PORT_RANGE = (3000, 3999)


# ========================= Wrapper implementation =============================

class TeamCraft(embodied.Env):
    """Embodied wrapper with MC auto-launch and structured observations."""

    ACTIONS = ["stay", "north", "south", "west", "east", "interact"]
    _ACT_DICT = {i: a for i, a in enumerate(ACTIONS)}

    def __init__(
        self,
        task,
        # HTTP bridge
        server_host: str = TEAMCRAFT_SERVER_HOST,
        server_port: Optional[int] = None,
        request_timeout: int = REQUEST_TIMEOUT_SEC,
        # MC server integration
        mc_port: Optional[int] = None,
        task_name: str = DEFAULT_TASK_NAME,
        output_folder: str = DEFAULT_OUTPUT,
        auto_launch_mc: bool = True,
        # Observations & macros
        logs: bool = False,
        logdir: Optional[str] = None,
        tokenizer: Optional[str] = DEFAULT_TOKENIZER_NAME,
        action_text_len: int = 64,
        move_seconds: float = MOVE_SECONDS,
        image_size: Tuple[int, int] = IMAGE_SIZE,
        vlm=None,
        embedder=None,
        world_path: Optional[str] = '/tmp/mc_world',
    ) -> None:
        super().__init__()

        # ---- config ---------------------------------------------------------
        self.cwd = '/TeamCraft/teamcraft/tasks/task_smelt'
        self.output_folder = str(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        # Unique port assignment
        self.mc_port = int(mc_port) if mc_port is not None else self._alloc_port(*MC_PORT_RANGE, purpose="mc")
        self.server_port = int(server_port) if server_port is not None else self._alloc_port(*HTTP_PORT_RANGE, purpose="http")

        self.server_host = server_host
        self.request_timeout = int(request_timeout)
        self.task_name = str(task_name)

        self._logs = logs
        self._logdir = elements.Path(logdir) if logdir else None
        self._logdir and self._logdir.mkdir()

        self._episode = 0
        self._length = 0
        self._reward = 0.0
        self._done = True
        self._step = 0

        self.seed = random.randint(0, 249)
        self.json_file_location = '/tmp'
        self.metadata = {}
        self.env = None
        self.mc_server_thread = None

        self.move_seconds = float(move_seconds)
        self.image_size = tuple(image_size)

        # Tokenizer
        self.action_text_len = int(action_text_len)
        self.total_action_text_len = 2 * self.action_text_len
        self.tokenizer = None
        self._pad_id = 0

        # World path
        self.world_path_cfg = world_path
        self.mc_world = self._resolve_world_path(self.world_path_cfg, self.task_name)
        
        # Create world dir if needed
        if not os.path.isdir(self.mc_world):
            try:
                os.makedirs(self.mc_world, exist_ok=True)
                open(os.path.join(self.mc_world, '.placeholder'), 'a').close()
                print(f"[TeamCraft] Created empty seed world at {self.mc_world}")
            except Exception as e:
                print(f"[TeamCraft] Failed to create seed world at {self.mc_world}: {e}")
        
        # Paths & logs
        self.mineflayer_log = os.path.join(self.output_folder, "logs/")
        os.makedirs(self.mineflayer_log, exist_ok=True)

        self.mc_log = os.path.join(self.output_folder, "logs/minecraft/")
        os.makedirs(self.mc_log, exist_ok=True)

        # MC server lifecycle
        self.mc_server: Optional[MCServerManager] = None
        if auto_launch_mc:
            self._start_mc_server()
        atexit.register(self._graceful_shutdown)

        print(f"[TeamCraft] Using mc_port={self.mc_port}, http_port={self.server_port}")

        if tokenizer and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is None:
                    pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
                self._pad_id = int(pad_id)
            except Exception:
                self.tokenizer = None
                self._pad_id = 0

        self._reset()

    @property
    def act_names(self):
        return list(self._ACT_DICT.values())
    
    @property
    def act_space(self) -> Dict[str, elements.Space]:
        return {
            "action": elements.Space(np.int32, (2,), 0, len(self.ACTIONS) - 1),
            "reset":  elements.Space(bool),  # Keep as bool - DreamerV3 expects this
        }

    @property
    def obs_space(self) -> Dict[str, elements.Space]:
        """Define structured observation space with separate keys for different components."""
        spaces = {
            # Per-bot observations (for bot1 and bot2)
            "bot1_health": elements.Space(np.float32, ()),
            "bot1_food": elements.Space(np.float32, ()),
            "bot1_oxygen": elements.Space(np.float32, ()),
            "bot1_position": elements.Space(np.float32, (3,)),
            "bot1_velocity": elements.Space(np.float32, (3,)),
            "bot1_yaw": elements.Space(np.float32, ()),
            "bot1_pitch": elements.Space(np.float32, ()),
            # "bot1_on_ground": elements.Space(np.uint8, ()),  # uint8 for MC state
            # "bot1_in_water": elements.Space(np.uint8, ()),   # uint8 for MC state
            "bot1_inventory": elements.Space(np.float32, (32,)),
            
            "bot2_health": elements.Space(np.float32, ()),
            "bot2_food": elements.Space(np.float32, ()),
            "bot2_oxygen": elements.Space(np.float32, ()),
            "bot2_position": elements.Space(np.float32, (3,)),
            "bot2_velocity": elements.Space(np.float32, (3,)),
            "bot2_yaw": elements.Space(np.float32, ()),
            "bot2_pitch": elements.Space(np.float32, ()),
            # "bot2_on_ground": elements.Space(np.uint8, ()),  # uint8 for MC state
            # "bot2_in_water": elements.Space(np.uint8, ()),   # uint8 for MC state
            "bot2_inventory": elements.Space(np.float32, (32,)),
            
            # Shared environment observations
            "voxels": elements.Space(np.float32, (64,)),
            "image": elements.Space(np.uint8, (self.image_size[1], self.image_size[0], 3)),
            
            # Task-related
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),      # Keep as bool for DreamerV3
            "is_last": elements.Space(bool, ()),       # Keep as bool for DreamerV3
            "is_terminal": elements.Space(bool, ()),   # Keep as bool for DreamerV3
            "log/reward": elements.Space(np.float32, ()),
            
            # Instructions
            "instructions_ids": elements.Space(np.int32, (32,), 0, 100_000),
        }
        # If you keep it only for logging/analysis (not for encoding), don't make it discrete uint8.
        spaces["action_ids"] = elements.Space(np.int32, (2,))

        # if self.tokenizer is not None:
        #     spaces["action_text_ids"] = elements.Space(np.int32, (self.total_action_text_len,))
        
        return spaces

    def _obs(self, metadata: Dict[str, Any], action: Optional[Dict[str, Any]] = None, 
             is_first: bool = False) -> Dict[str, Any]:
        """Process metadata dict into structured observation."""
        obs = {}
        
        # Process bot1 observations
        bot1_data = metadata.get('state', {}).get('bot1', {})
        obs.update(self._process_bot_obs(bot1_data, 'bot1'))
        
        # Process bot2 observations
        bot2_data = metadata.get('state', {}).get('bot2', {})
        obs.update(self._process_bot_obs(bot2_data, 'bot2'))
        
        # Process voxels (use any bot's voxel data, they should be similar)
        voxels = bot1_data.get('voxels', [])
        obs['voxels'] = self._encode_voxels(voxels)
        
        # Process image
        if hasattr(self, 'image') and self.image is not None:
            obs['image'] = self._fuse_images(self.image)
        else:
            obs['image'] = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # Task-related observations (keep as bool for DreamerV3 internal usage)
        obs['reward'] = np.float32(metadata.get('reward', 0.0))
        obs['is_first'] = is_first  # Keep as bool
        obs['is_last'] = metadata.get('done', False)  # Keep as bool
        obs['is_terminal'] = metadata.get('done', False)  # Keep as bool
        obs['log/reward'] = np.float32(metadata.get('reward', 0.0))
        
        # Instructions (placeholder)
        obs['instructions_ids'] = np.zeros(32, dtype=np.int32)
        obs['action_ids'] = np.ones(2, dtype=np.int32) * -100
        
        # Action text encoding if tokenizer available
        # if self.tokenizer is not None and action is not None:
        #     joint_action = []
        #     if isinstance(action, dict) and 'action' in action:
        #         act_array = action['action']
        #         if hasattr(act_array, '__iter__'):
        #             joint_action = list(act_array)[:2]
            
        #     if not joint_action:
        #         joint_action = [0, 0]
                
        #     action_texts = self._action_texts(joint_action, metadata)
        #     obs['action_text_ids'] = self._encode_action_texts(action_texts)
        # elif self.tokenizer is not None:
        #     obs['action_text_ids'] = np.full(self.total_action_text_len, self._pad_id, dtype=np.int32)
        
        return obs

    def _process_bot_obs(self, bot_data: Dict[str, Any], bot_name: str) -> Dict[str, Any]:
        """Process individual bot observations."""
        obs = {}
        
        # Get status data
        status = bot_data.get('status', {})
        
        # Health, food, oxygen (normalized to [0, 1])
        obs[f'{bot_name}_health'] = np.float32(status.get('health', 20) / 20.0)
        obs[f'{bot_name}_food'] = np.float32(status.get('food', 20) / 20.0)
        obs[f'{bot_name}_oxygen'] = np.float32(status.get('oxygen', 20) / 20.0)
        
        # Position
        pos = status.get('position', {})
        obs[f'{bot_name}_position'] = np.array([
            pos.get('x', 0.0),
            pos.get('y', 0.0),
            pos.get('z', 0.0)
        ], dtype=np.float32)
        
        # Velocity
        vel = status.get('velocity', {})
        obs[f'{bot_name}_velocity'] = np.array([
            vel.get('x', 0.0),
            vel.get('y', 0.0),
            vel.get('z', 0.0)
        ], dtype=np.float32)
        
        # Orientation
        obs[f'{bot_name}_yaw'] = np.float32(status.get('yaw', 0.0))
        obs[f'{bot_name}_pitch'] = np.float32(status.get('pitch', 0.0))
        
        # # Boolean states (converted to uint8: 0 or 1)
        # obs[f'{bot_name}_on_ground'] = np.uint8(1 if status.get('onGround', True) else 0)
        # obs[f'{bot_name}_in_water'] = np.uint8(1 if status.get('isInWater', False) else 0)
        
        # Inventory encoding
        inventory = bot_data.get('inventory', {})
        obs[f'{bot_name}_inventory'] = self._encode_inventory(inventory)
        
        return obs

    def _encode_inventory(self, inventory: Dict[str, int]) -> np.ndarray:
        """Encode inventory into fixed-size vector."""
        # Define common inventory items
        vocab = [
            "oak_log", "spruce_log", "planks", "cobblestone", "wheat", "seeds",
            "coal", "iron_ingot", "raw_iron", "furnace", "crafting_table",
            "beef", "porkchop", "mutton", "chicken", "rabbit", "potato",
            "kelp", "cod", "salmon", "baked_potato", "dried_kelp",
            "cooked_beef", "cooked_porkchop", "cooked_mutton", "cooked_chicken",
            "cooked_rabbit", "cooked_cod", "cooked_salmon",
            "iron_pickaxe", "iron_axe", "iron_shovel", "iron_sword"
        ]
        
        vec = np.zeros(32, dtype=np.float32)
        for i, item in enumerate(vocab[:32]):
            vec[i] = float(inventory.get(item, 0))
        return vec

    def _encode_voxels(self, voxels: List) -> np.ndarray:
        """Encode voxel information into fixed-size vector."""
        vec = np.zeros(64, dtype=np.float32)
        
        # Map block types to indices
        block_types = {
            'air': 0, 'cobblestone': 1, 'spruce_log': 2, 'oak_log': 3,
            'furnace': 4, 'crafting_table': 5, 'chest': 6, 'planks': 7
        }
        
        for i, voxel in enumerate(voxels[:16]):  # Take first 16 voxels
            if isinstance(voxel, list) and len(voxel) >= 2:
                block_type = voxel[0]
                block_pos = voxel[1]
                
                # Encode block type
                if i * 4 < 64:
                    vec[i * 4] = block_types.get(block_type, -1)
                    
                # Encode relative position (if available)
                if isinstance(block_pos, dict) and i * 4 + 3 < 64:
                    vec[i * 4 + 1] = float(block_pos.get('x', 0) % 100)  # Relative x
                    vec[i * 4 + 2] = float(block_pos.get('y', 0) % 100)  # Relative y
                    vec[i * 4 + 3] = float(block_pos.get('z', 0) % 100)  # Relative z
        
        return vec

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Step the environment and return structured observation."""
        # print(action)
        # Check for reset action
        if isinstance(action, dict) and action.get('reset', False):
            return self._reset()
        
        # Extract action array from the action dict
        action_array = None
        if isinstance(action, dict) and 'action' in action:
            action_array = action['action']
            if hasattr(action_array, '__iter__'):
                action_array = list(action_array)[:2]  # Take first 2 elements
        
        # Convert action indices to JavaScript commands if action provided
        js_code = None
        if action_array is not None and len(action_array) >= 2:
            # Build JavaScript commands for each bot
            js_commands = []
            for i, (bot_name, act_id) in enumerate(zip(['bot1', 'bot2'], action_array)):
                js_cmd = self._macro_to_js_call(bot_name, int(act_id))
                if js_cmd:
                    js_commands.append(js_cmd)
            
            if js_commands:
                js_code = " ".join(js_commands)
        
        # Execute action in environment
        if js_code is not None:
            # print(f"Executing JS code: {js_code}")
            self.observation = self.env.step(code=js_code)
            self.actions = action
            # Look back at middle every timestep for smelt task
            code = ""
            for bot_name in self.bot_list:
                code += f"await {bot_name}.lookAt(new Vec3({self.center_position[0]},{self.center_position[1]},{self.center_position[2]}));"
            self.env.step_manuual(code=code)
        else:
            self.observation = self.env.step("")
            self.actions = action
            
        # print(self.observation)
        # Extract key information
        onChat = [self.observation[a]["onChat"] for a in self.bot_list]
        self.inventory = {a: self.observation[a]["inventory"] for a in self.bot_list}
        _inventory_list = [self.inventory[a] for a in self.bot_list]
        self.voxels = [item for item in self.observation['bot1']['voxels'] if isinstance(item, list)]
        self.reward = self.reward_function(_inventory_list, self.done_input)
        self.done = self.reward == 1
        self.state = filter_voxel(self.observation, self.place_of_interest)
        self.image = self.env.render()
        
        # Create metadata
        metadata = {
            'time': datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S%f')[:-3],
            'action': action,
            'state': self.state,
            'done': self.done,
            'reward': self.reward,
            'inventory': self.inventory
        }
        # print(metadata)
        self.time_step += 1
        
        # Process into structured observation
        obs = self._obs(metadata, action, is_first=False)
        
        return obs

    def _reset(self) -> Dict[str, Any]:
        """Reset environment and return initial observation."""
        random.seed(self.seed)
        
        # Load config
        self.config_path = os.path.join(self.cwd, f'./configure/103.json')
        with open(self.config_path, 'r') as file:
            json_data = json.load(file)
        
        self.init_command = json_data['command']
        self.bot_count = len(json_data["bot_list"])
        self.done_input = json_data['done_input']
        self.action_length = len(json_data['actions'])
        self.bot_list = json_data["bot_list"]
        self.center_position = json_data["center_position"]
        self.obs_command = json_data["obs_command"]
        
        [a1, b1, c1] = self.center_position
        self.place_of_interest = [[x, y, z] for x in range(a1-3, a1+3) 
                                  for y in range(b1, b1+4) 
                                  for z in range(c1-3, c1+3)]
        
        print(f"Agents count: {self.bot_count}")
        print(f"Bot list: {self.bot_list}")
        
        # Initialize environment
        self.env = teamCraft(
            agent_count=2,
            mc_port=self.mc_port,
            server_port=self.server_port,
            env_wait_ticks=20,
            log_path=str(self._logdir or self.mineflayer_log),
        )
        self.env.start()
        
        # Reset internal state
        self.time_step = 0
        self.actions = []
        self.reward = 0
        self.done = False
        self.task_image = {}
        self.observation = None
        self.metadata = {}
        self.inventory = None
        self.state = None
        self.image = None
        self.voxels = None
        
        # World setup
        self.observation = self.env.step_manuual(code=self.init_command)
        print('World has been set up')
        
        # Start recording
        code = "await bot3.chat('startRecoding -1 "+ " "+"\');"
        self.env.step_manuual(code=code)
        
        # Look at middle
        code = ""
        for bot_name in self.bot_list:
            code += f"await {bot_name}.lookAt(new Vec3({self.center_position[0]},{self.center_position[1]},{self.center_position[2]}));"
        self.env.step_manuual(code=code)
        print("All agents now looked at center position")
        
        # 3-Views observation
        self.metadata['obs'] = {}
        obs_step = 0
        for obs_action in self.obs_command:
            self.metadata['obs'][obs_step] = {}
            self.metadata['obs'][obs_step]['time'] = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S%f')[:-3]
            self.metadata['obs'][obs_step]['action'] = obs_action[0]
            self.task_image[obs_step] = self.env.render()
            self.env.step_manuual(code=obs_action[0])
            obs_step += 1
        
        print("Orthographic projections observation finished")
        
        # Get first observation
        first_obs = self.step(None)
        first_obs["is_first"] = True  # Keep as bool
        first_obs["is_last"] = False  # Keep as bool
        first_obs["is_terminal"] = False  # Keep as bool
        
        print("Reset finished")
        return first_obs
    
    def reward_function(self, bag_item, done_input):
        total = 0
        for item in bag_item:
            total += item.get(done_input[0], 0)
        return total / done_input[1]

    def _macro_to_js_call(self, bot_name: str, act_id: int) -> str:
        """Convert action ID to JavaScript command for a specific bot."""
        if act_id == 0:   # STAY
            return ""
        elif act_id == 1:   # NORTH (negative Z)
            return f"await exploreUntil({bot_name}, new Vec3(0, 0, -1), {self.move_seconds});"
        elif act_id == 2:   # SOUTH
            return f"await exploreUntil({bot_name}, new Vec3(0, 0, 1), {self.move_seconds});"
        elif act_id == 3:   # WEST (negative X)
            return f"await exploreUntil({bot_name}, new Vec3(-1, 0, 0), {self.move_seconds});"
        elif act_id == 4:   # EAST (positive X)
            return f"await exploreUntil({bot_name}, new Vec3(1, 0, 0), {self.move_seconds});"
        elif act_id == 5:   # INTERACT
            return self._interact_macro(bot_name)
        return ""

    def _interact_macro(self, bot_name: str) -> str:
        """Generate interact command for a bot (task-specific)."""
        # This should be customized per task
        # For now, return empty string or a generic interact command
        return f"await {bot_name}.interact();"

    def _action_texts(self, joint_action: List[int], info: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        for a in joint_action[:2]:
            name = self._ACT_DICT.get(a, "stay")
            if name in ("north", "south", "west", "east"):
                texts.append(f"move {name}")
            elif name == "interact":
                texts.append("interact")
            else:
                texts.append("stay")
        while len(texts) < 2:
            texts.append("")
        return texts[:2]

    def _encode_action_texts(self, texts: List[str]) -> np.ndarray:
        if not self.tokenizer:
            return np.full(self.total_action_text_len, self._pad_id, dtype=np.int32)
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.action_text_len,
            padding="max_length",
        )
        ids = np.array(batch["input_ids"], dtype=np.int32)
        if self._pad_id != 0:
            ids = np.where(ids == 0, self._pad_id, ids)
        return ids.flatten()

    def _fuse_images(self, obs_raw: Any) -> np.ndarray:
        imgs: List[np.ndarray] = []
        if isinstance(obs_raw, dict):
            for k in sorted(obs_raw.keys()):
                arr = obs_raw[k]
                if isinstance(arr, np.ndarray):
                    imgs.append(arr)
        if not imgs:
            imgs = [np.zeros((128, 128, 3), dtype=np.uint8)]
        imgs = [self._ensure_uint8_rgb(im) for im in imgs[:4]]
        mosaic = self._mosaic(imgs)
        mosaic = self._resize(mosaic, self.image_size)
        return mosaic

    def _ensure_uint8_rgb(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _mosaic(self, images: List[np.ndarray]) -> np.ndarray:
        if len(images) == 1:
            return images[0]
        if len(images) == 2:
            h = max(images[0].shape[0], images[1].shape[0])
            w = images[0].shape[1] + images[1].shape[1]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            canvas[:images[0].shape[0], :images[0].shape[1]] = images[0]
            canvas[:images[1].shape[0], images[0].shape[1]:] = images[1]
            return canvas
        h = max(im.shape[0] for im in images[:2])
        w = max(im.shape[1] for im in images[:2])
        canvas = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        for i, im in enumerate(images[:4]):
            r = i // 2
            c = i % 2
            canvas[r*h:r*h+im.shape[0], c*w:c*w+im.shape[1]] = im
        return canvas

    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(image, size)
        except Exception:
            h, w = image.shape[:2]
            th, tw = size[1], size[0]
            y_idx = (np.linspace(0, h-1, th)).astype(int)
            x_idx = (np.linspace(0, w-1, tw)).astype(int)
            return image[y_idx][:, x_idx]

    def _alloc_port(self, start: int, end: int, purpose: str, tries: int = 64) -> int:
        """Randomly pick a free TCP port in [start, end], with a cross-process lock."""
        assert start <= end, "invalid port range"

        lock_dir = os.environ.get("TEAMCRAFT_PORTLOCK_DIR", "/tmp")
        os.makedirs(lock_dir, exist_ok=True)
        lock_path = os.path.join(lock_dir, f"teamcraft_portlock_{purpose}.lock")

        with self._file_lock(lock_path):
            candidates = list(range(start, end + 1))
            random.shuffle(candidates)
            for _ in range(tries):
                for port in candidates:
                    if not self._is_port_open("127.0.0.1", port) and self._try_bind(port):
                        return port
                random.shuffle(candidates)
        raise RuntimeError(f"no free port found in range [{start}, {end}] for {purpose}")

    def _try_bind(self, port: int) -> bool:
        """Best-effort reservation: bind+listen then immediately release."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            s.listen(1)
            return True
        except OSError:
            return False
        finally:
            try:
                s.close()
            except Exception:
                pass

    @contextmanager
    def _file_lock(self, path: str):
        """Advisory file lock; no-op fallback if fcntl not available."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = open(path, "a+")
        try:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_UN)
            finally:
                f.close()

    def _start_mc_server(self) -> None:
        """Launch Java MC server with MCServerManager and wait for port."""
        if self._is_port_open("127.0.0.1", self.mc_port):
            print(f"[TeamCraft] MC server already listening on {self.mc_port}")
            return
        print(f"[TeamCraft] Starting MC server on port {self.mc_port} (world={self.mc_world})")
        self.mc_server = MCServerManager(self.mc_port, self.mc_world, self.mc_log)
        try:
            self.mc_server.start()
        except Exception as e:
            print(f"[TeamCraft] Failed to start MC server: {e}")
            raise
        self._wait_for_port("127.0.0.1", self.mc_port, timeout=120.0)

    def _wait_for_port(self, host: str, port: int, timeout: float = 60.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._is_port_open(host, port):
                print(f"[TeamCraft] MC server ready on {host}:{port}")
                return
            time.sleep(0.5)
        raise RuntimeError(f"Minecraft server on {host}:{port} did not become ready within {timeout}s")

    @staticmethod
    def _is_port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except Exception:
            return False

    def _resolve_world_path(self, world_path_cfg: Optional[str], task_name: str) -> str:
        if world_path_cfg and os.path.isdir(world_path_cfg):
            print(f"[TeamCraft] Using provided world path: {world_path_cfg}")
            return world_path_cfg

        candidate_here = os.path.join(self.cwd, f"world_{task_name}")
        if os.path.isdir(candidate_here):
            print(f"[TeamCraft] Using local world path: {candidate_here}")
            return candidate_here

        try:
            import teamcraft as _tc
            tc_root = os.path.dirname(_tc.__file__)
            cands = [
                os.path.join(tc_root, "minecraft", f"world_{task_name}"),
                os.path.join(tc_root, "minecraft", "worlds", f"world_{task_name}"),
            ]
            for c in cands:
                if os.path.isdir(c):
                    print(f"[TeamCraft] Using package world path: {c}")
                    return c
        except Exception:
            pass

        candidate_cwd = os.path.join(os.getcwd(), "worlds", f"world_{task_name}")
        if os.path.isdir(candidate_cwd):
            print(f"[TeamCraft] Using cwd world path: {candidate_cwd}")
            return candidate_cwd

        fallback = candidate_here
        print(f"[TeamCraft] No existing world found; will seed new world at: {fallback}")
        return fallback

    def _graceful_shutdown(self) -> None:
        json_file = str(self.seed) + '.json'
        with open(self.json_file_location + json_file, 'w') as json_file:
            json.dump(self.metadata, json_file, indent=4, cls=NpEncoder)
        if self.env is not None:
            self.env.close()
            self.env = None
        if self.mc_server_thread is not None:
            self.mc_server_thread.stop()
            self.mc_server_thread = None

    def _write_stats(self, length, reward):
        if not self._logdir:
            return
        stats = {
            "episode": self._episode,
            "length": int(length),
            "reward": float(round(reward, 3)),
        }
        filepath = self._logdir / "teamcraft_stats.jsonl"
        lines = filepath.read() if filepath.exists() else ""
        lines += json.dumps(stats) + "\n"
        filepath.write(lines, mode="w")
        print(f"[TeamCraft] Wrote stats to {filepath}")