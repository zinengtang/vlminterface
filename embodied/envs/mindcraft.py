import os
import json
import random
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np
import elements
import embodied
import socketio


# ----------------------------- Tasks & IDs ------------------------------------

TASKS_LIST = [
    "collect_wood",
    "build_shelter",
    "mine_iron",
    "craft_tools",
    "hunt_animals",
    "farm_crops",
    "explore_cave",
    "build_nether_portal",
    "defeat_mobs",
    "collect_diamonds",
]

ITEM_IDS = {
    "air": 0,
    "stone": 1,
    "grass_block": 2,
    "dirt": 3,
    "cobblestone": 4,
    "oak_log": 5,
    "oak_planks": 6,
    "crafting_table": 7,
    "furnace": 8,
    "chest": 9,
    "iron_ore": 10,
    "iron_ingot": 11,
    "diamond": 12,
    "coal": 13,
    "stick": 14,
    "wooden_pickaxe": 15,
    "stone_pickaxe": 16,
    "iron_pickaxe": 17,
    "diamond_pickaxe": 18,
    "wooden_sword": 19,
    "stone_sword": 20,
    "iron_sword": 21,
    "diamond_sword": 22,
    "bread": 23,
    "cooked_beef": 24,
    "apple": 25,
    "wheat": 26,  # for farm_crops task
    # extend as needed...
}

# Treat blocks as a superset of item names for this adapter.
BLOCK_IDS = dict(ITEM_IDS)  # extend with block-only names if needed

MOB_IDS = {
    "player": 0,
    "zombie": 1,
    "skeleton": 2,
    "creeper": 3,
    "spider": 4,
    "enderman": 5,
    "cow": 6,
    "pig": 7,
    "sheep": 8,
    "chicken": 9,
    "villager": 10,
}

MODE_IDS = {
    "assistant": 0,
    "survival": 1,
    "creative": 2,
    "god_mode": 3,
}

BIOME_IDS = {
    "plains": 0,
    "forest": 1,
    "desert": 2,
    "mountains": 3,
    "swamp": 4,
    "jungle": 5,
    "taiga": 6,
    "ocean": 7,
    "cave": 8,
    "nether": 9,
}


# ------------------------------ State -----------------------------------------

@dataclass
class MinecraftState:
    """Container for Minecraft game state (mocked or filled via server)."""
    position: np.ndarray        # (3,) float32
    health: float
    hunger: float
    inventory: Dict[str, int]   # item_name -> count
    nearby_blocks: np.ndarray   # (G,G,G) int32
    nearby_entities: List[Dict] # list of dicts with x,y,z,type_id,health,distance
    equipped_item: str
    time_of_day: float
    biome: str
    experience: int
    game_mode: str


# ---------------------- Mindcraft Action Registry -----------------------------

# This registry mirrors the macro-commands defined in the Mindcraft repo
# (src/agent/commands/actions.js). Each entry corresponds to an `!Command`.
# We group parameters into three generic lanes for RL convenience:
#   - p_int[4]: generic integers (counts, distances, x/y/z, closeness, seconds, slot, radius, depth, ...)
#   - p_ids[3]: generic enums/IDs:
#       * p_ids[0] -> item/block/recipe name id (ITEM_IDS or BLOCK_IDS)
#       * p_ids[1] -> entity/mob id (MOB_IDS)
#       * p_ids[2] -> mode id (MODE_IDS) or spare
#   - p_text_ids[64]: optional text tokens (player names, labels, freeform goals)
#
# The formatting in `_format_command()` maps these lanes to the concrete `!command(...)` strings.

ACTIONS_REGISTRY: List[Dict[str, Any]] = [
    {"name": "newAction"},
    {"name": "stop"},
    {"name": "stfu"},
    {"name": "restart"},
    {"name": "clearChat"},
    {"name": "goToPlayer"},          # text: player name (or nearest if empty)
    {"name": "followPlayer"},        # text: player name
    {"name": "goToCoordinates"},     # p_int: x,y,z,closeness
    {"name": "searchForBlock"},      # p_ids[0]: block, p_int[0]: radius
    {"name": "searchForEntity"},     # p_ids[1]: entity, p_int[0]: radius
    {"name": "moveAway"},            # p_int[0]: distance
    {"name": "rememberHere"},        # text: place name
    {"name": "goToRememberedPlace"}, # text: place name
    {"name": "givePlayer"},          # p_ids[0]: item, p_int[0]: count, text: player (optional)
    {"name": "consume"},             # p_ids[0]: item
    {"name": "equip"},               # p_int[0]: slot (>=0) OR p_ids[0]: item
    {"name": "putInChest"},          # p_ids[0]: item, p_int[0]: count
    {"name": "takeFromChest"},       # p_ids[0]: item, p_int[0]: count
    {"name": "viewChest"},
    {"name": "discard"},             # p_ids[0]: item, p_int[0]: count
    {"name": "collectBlocks"},       # p_ids[0]: block, p_int[0]: count
    {"name": "craftRecipe"},         # p_ids[0]: recipe/item, p_int[0]: count
    {"name": "smeltItem"},           # p_ids[0]: item
    {"name": "clearFurnace"},
    {"name": "placeHere"},           # p_ids[0]: block
    {"name": "attack"},
    {"name": "attackPlayer"},        # text: player name
    {"name": "goToBed"},
    {"name": "activate"},
    {"name": "stay"},                # p_int[0]: seconds
    {"name": "setMode"},             # p_ids[2]: mode
    {"name": "goal"},                # text: freeform goal
    {"name": "endGoal"},
    {"name": "startConversation"},   # text: player name
    {"name": "endConversation"},
    {"name": "lookAtPlayer"},        # text: player name
    {"name": "lookAtPosition"},      # p_int: x,y,z
    {"name": "digDown"},             # p_int[0]: depth
]

ACTION_INDEX = {a["name"]: i for i, a in enumerate(ACTIONS_REGISTRY)}


# ---------------------------- Environment -------------------------------------

class MindCraft(embodied.Env):
    """
    Gym-like / Dreamer-style Minecraft environment wrapper with Mindcraft
    macro action space integration.

    Observations follow the Dreamer/embodied pattern:
      dict with keys including 'reward', 'is_first', 'is_last', 'is_terminal'

    Actions (dict):
      {
        'action': int in [0, len(ACTIONS_REGISTRY)-1],
        'p_int': int32[4],
        'p_ids': int32[3],     # [item/block/recipe, entity, mode]
        'p_text_ids': int32[64]  (optional but supported even without tokenizer),
        'reset': bool
      }
    """

    # Reward shaping defaults
    REWARD_PARAMS = {
        "block_break": 0.1,
        "block_place": 0.1,
        "item_craft": 0.5,
        "item_pickup": 0.2,
        "mob_kill": 1.0,
        "damage_taken": -0.5,   # per lost health point
        "death": -10.0,
        "task_complete": 20.0,
        "exploration": 0.01,
        "tool_upgrade": 2.0,
        "food_eaten": 0.3,
    }

    def __init__(
        self,
        task: Optional[str] = None,
        horizon: int = 1000,
        host: str = "localhost",
        mindserver_port: int = 48080,
        mc_port: int = 25565,
        username: str = "Bot",
        version: str = "1.20.1",
        reward_shaping: Optional[Dict[str, float]] = None,
        logs: bool = False,
        logdir: Optional[str] = None,
        seed: Optional[int] = None,
        vlm: Any = None,
        embedder: Any = None,
        tokenizer: Any = None,
        action_text_len: int = 64,
        view_distance: int = 5,
        mindcraft_profile: str = "./andy.json",
        node_cwd: str = "./mindcraft",
        node_entry: str = "main.js",
        node_args: Optional[List[str]] = None,
        connect_timeout_s: float = 5.0,
    ):
        super().__init__()

        # --- configuration
        self.task = task or random.choice(TASKS_LIST)
        self.horizon = int(horizon)
        self.host = host
        self.mindserver_port = int(mindserver_port)
        self.mc_port = int(mc_port)
        self.username = username
        self.version = version
        self.mindcraft_profile = mindcraft_profile
        self.node_cwd = node_cwd
        self.node_entry = node_entry
        self.node_args = node_args or []
        self.connect_timeout_s = float(connect_timeout_s)

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.reward_params = reward_shaping or dict(self.REWARD_PARAMS)

        self._logs = logs
        self._logdir = elements.Path(logdir) if logdir else None
        if self._logdir:
            self._logdir.mkdir(parents=True, exist_ok=True)

        self.vlm = vlm
        self.embedder = embedder
        self.view_distance = int(view_distance)

        # --- tokenizer (optional)
        self.action_text_len = int(action_text_len)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "bert-base-uncased", local_files_only=True
                )
            except Exception:
                self.tokenizer = None  # proceed without text fields

        self._pad_id = getattr(self.tokenizer, "pad_token_id", 0) if self.tokenizer else 0

        # --- episode tracking
        self._episode = 0
        self._step_count = 0
        self._length = 0
        self._total_reward = 0.0
        self._done = True

        # --- runtime state
        self._state: Optional[MinecraftState] = None
        self._prev_state: Optional[MinecraftState] = None
        self._bot_process: Optional[subprocess.Popen] = None
        self._sio: Optional[socketio.Client] = None

        # --- task trackers
        self._task_goals = self._get_task_goals(self.task)
        self._visited_positions = set()
        self._blocks_broken: Dict[str, int] = {}
        self._items_crafted: Dict[str, int] = {}
        self._mobs_killed: Dict[str, int] = {}

    # ----------------------------- Spaces -------------------------------------

    @property
    def obs_space(self):
        g = 2 * self.view_distance + 1
        spaces = {
            # Player state
            "position":        elements.Space(np.float32, (3,)),
            "velocity":        elements.Space(np.float32, (3,)),
            "pitch_yaw":       elements.Space(np.float32, (2,)),
            "health":          elements.Space(np.float32, ()),
            "hunger":          elements.Space(np.float32, ()),
            # was int -> make float so it's not one-hot
            "experience":      elements.Space(np.float32, ()),

            # Inventory & equipment
            # counts → floats (no one-hot)
            "inventory":       elements.Space(np.float32, (36,)),
            # keep as small categorical with bounds if you want one-hot (OK)
            "equipped_item_id":elements.Space(np.int32, (), 0, len(ITEM_IDS) - 1),

            # Environment
            # big grid → floats (avoid one-hot explosion)
            "nearby_blocks": elements.Space(np.float32, (g * g * g,)),

            "visible_entities":elements.Space(np.float32, (20, 6)),
            "time_of_day":     elements.Space(np.float32, ()),
            # small categorical with bounds (OK to one-hot)
            "biome_id":        elements.Space(np.int32, (), 0, len(BIOME_IDS) - 1),
            "light_level":     elements.Space(np.int32, (), 0, 15),

            # Task
            "task_progress":   elements.Space(np.float32, (10,)),

            # Meta (Dreamer-style)
            "reward":          elements.Space(np.float32, ()),
            "is_first":        elements.Space(np.bool_,   ()),
            "is_last":         elements.Space(np.bool_,   ()),
            "is_terminal":     elements.Space(np.bool_,   ()),

            # Image
            "image":           elements.Space(np.uint8,   (64, 64, 3)),
        }

        # If you want to keep these, make them floats so they won't be one-hot.
        # (Also, ensure the SHAPE is a tuple, not a bare int.)
        spaces["instructions_ids"] = elements.Space(np.uint8, (2, 32))
        spaces["action_ids"]       = elements.Space(np.int32, (2,))

        # Avoid text token arrays unless you have a specialized encoder.
        if False and self.tokenizer is not None:
            spaces["action_text_ids"] = elements.Space(np.int32, (self.action_text_len,))
            spaces["chat_message_ids"] = elements.Space(np.int32, (128,))
        return spaces


    @property
    def act_space(self):
        # Broad but safe-ish bounds for generic ints.
        INT_MIN, INT_MAX = -32768, 32767
        # IDs allow a generous range; actual mapping will clamp/lookup.
        ID_MAX = 4096
        spaces = {
            "action":      elements.Space(np.int32,  (), 0, len(ACTIONS_REGISTRY) - 1),
            "p_int":       elements.Space(np.int32,  (4,), INT_MIN, INT_MAX),
            "p_ids":       elements.Space(np.int32,  (3,), 0, ID_MAX),
            "reset":       elements.Space(np.bool_,  ()),
        }
        # Always expose text lane for stability, even without tokenizer
        spaces["p_text_ids"] = elements.Space(np.int32, (64,))
        return spaces

    # --------------------------- Core API -------------------------------------

    def step(self, action: Dict[str, Any]):
        """Perform one environment step. Returns a full observation dict."""
        if action.get("reset", False) or self._done:
            return self._reset()

        # Keep reference to previous state for reward computation
        self._prev_state = self._state

        # Parse action payload
        a_idx = int(action["action"])
        p_int = np.array(action.get("p_int", np.zeros(4, np.int32)), dtype=np.int32)
        p_ids = np.array(action.get("p_ids", np.zeros(3, np.int32)), dtype=np.int32)
        p_txt = np.array(action.get("p_text_ids", np.zeros(64, np.int32)), dtype=np.int32)

        # Execute macro-command
        self._execute_action(a_idx, p_int, p_ids, p_txt)

        # Query new state
        self._state = self._get_game_state()

        # Compute reward BEFORE marking visited (so exploration can pay out)
        reward = self._calculate_reward(ACTIONS_REGISTRY[a_idx]["name"])

        # Mark visited position now
        if self._state is not None:
            pos_tuple = tuple(self._state.position.astype(int))
            self._visited_positions.add(pos_tuple)

        # Update time/accounting
        self._length += 1
        self._step_count += 1

        # Termination conditions
        time_up = self._step_count >= self.horizon
        death = (self._state.health <= 0.0) if self._state is not None else False
        self._done = bool(self._done or time_up or death)
        self._total_reward += float(reward)

        # Build observation
        obs = self._build_observation(reward, is_first=False)

        # Keep text fields stable if tokenizer is active
        if self.tokenizer is not None:
            # Attach simple NL description of the macro
            desc = self._describe_macro(a_idx, p_int, p_ids, p_txt)
            obs["action_text_ids"] = self._encode_text(desc)
            if "chat_message_ids" not in obs:
                obs["chat_message_ids"] = np.zeros((128,), dtype=np.int32)

        # Log end of episode
        if self._done and self._logdir:
            self._write_stats()

        return obs

    def _reset(self):
        """Reset the episode and underlying runtime."""
        # Ensure runtime is up
        if self._bot_process is None or self._sio is None or not self._sio.connected:
            self._start_mindcraft_bot()
        else:
            # If your server supports an explicit reset event, emit it here:
            # self._sio.emit("env/reset", {"agent": self.username, "task": self.task})
            pass

        # Episode counters
        self._episode += 1
        self._length = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False

        # Clear trackers
        self._visited_positions.clear()
        self._blocks_broken.clear()
        self._items_crafted.clear()
        self._mobs_killed.clear()

        # Fetch initial state
        self._state = self._get_game_state()

        # Build initial obs
        obs = self._build_observation(reward=0.0, is_first=True)

        if self.tokenizer is not None:
            obs["action_text_ids"] = self._encode_text("")
            obs["chat_message_ids"] = self._encode_text(f"Task: {self.task}")

        return obs

    def close(self):
        """Tear down sockets and subprocess."""
        try:
            if self._sio is not None and self._sio.connected:
                # self._sio.emit("env/close", {"agent": self.username})
                self._sio.disconnect()
        except Exception:
            pass

        if self._bot_process:
            try:
                self._bot_process.terminate()
                self._bot_process.wait(timeout=5)
            except Exception:
                try:
                    self._bot_process.kill()
                except Exception:
                    pass
            self._bot_process = None

    # ---------------------------- Helpers -------------------------------------

    def _get_task_goals(self, task: str) -> Dict[str, Any]:
        """Define task-specific goals and success criteria."""
        task_configs = {
            "collect_wood": {
                "target_items": {"oak_log": 10},
                "time_limit": 300,
                "success_reward": 50.0,
            },
            "build_shelter": {
                "required_blocks": 20,
                "has_roof": True,
                "time_limit": 600,
                "success_reward": 100.0,
            },
            "mine_iron": {
                "target_items": {"iron_ingot": 5},
                "time_limit": 900,
                "success_reward": 75.0,
            },
            "craft_tools": {
                "target_items": {"iron_pickaxe": 1, "iron_sword": 1},
                "time_limit": 1200,
                "success_reward": 80.0,
            },
            "hunt_animals": {
                "target_food": 20,
                "time_limit": 400,
                "success_reward": 40.0,
            },
            "farm_crops": {
                "target_items": {"wheat": 20},
                "time_limit": 1500,
                "success_reward": 60.0,
            },
            "explore_cave": {
                "min_depth": -20,
                "unique_blocks": 10,
                "time_limit": 600,
                "success_reward": 70.0,
            },
            "defeat_mobs": {
                "target_kills": 10,
                "time_limit": 500,
                "success_reward": 90.0,
            },
            "collect_diamonds": {
                "target_items": {"diamond": 3},
                "time_limit": 2000,
                "success_reward": 200.0,
            },
        }
        return task_configs.get(task, task_configs["collect_wood"])

    # --- Communication with MindServer/Agent (Socket.IO) ----------------------

    def _ensure_connected(self):
        if self._sio is None:
            self._connect_sio()

    def _connect_sio(self):
        base = self.host
        if not (base.startswith("http://") or base.startswith("https://")):
            base = f"http://{base}"
        url = f"{base}:{self.mindserver_port}"

        self._sio = socketio.Client()

        try:
            self._sio.connect(url, transports=["websocket"], wait=True, wait_timeout=self.connect_timeout_s)
        except Exception as e:
            print(f"[MinecraftEnv] Socket.IO connect failed: {e}")

    def _start_mindcraft_bot(self):
        """Launch Node server (mindserver) and connect to it."""
        # Start server process if not already running
        if self._bot_process is None:
            try:
                cmd = ["node", self.node_entry]
                if "--profiles" not in self.node_args:
                    # Provide profile by default if not already in node_args
                    cmd += ["--profiles", self.mindcraft_profile]
                if "--port" not in self.node_args:
                    cmd += ["--port", str(self.mindserver_port)]
                cmd += list(self.node_args)

                self._bot_process = subprocess.Popen(
                    cmd,
                    cwd=self.node_cwd if self.node_cwd else None,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                time.sleep(2.0)  # brief warm-up
            except Exception as e:
                print(f"[MinecraftEnv] Failed to start MindServer: {e}")
                self._bot_process = None

        # Connect socket
        self._connect_sio()

    def _emit_command(self, command: str):
        """Emit the final command string to the Mindcraft agent via Socket.IO."""
        if self._sio is None or not self._sio.connected:
            self._ensure_connected()
        try:
            if self._sio and self._sio.connected:
                # Adjust event name & payload to match your server:
                self._sio.emit("agent_command", {"agent": self.username, "command": command})
        except Exception as e:
            print(f"[MinecraftEnv] Failed to send command '{command}': {e}")

    # --- Action execution ------------------------------------------------------

    def _execute_action(self, action_idx: int, p_int: np.ndarray, p_ids: np.ndarray, p_txt_ids: np.ndarray) -> bool:
        """Translate registry action into a server/bot command."""
        try:
            cmd = self._format_command(action_idx, p_int, p_ids, p_txt_ids)
            if cmd:
                self._emit_command(cmd)
            return True
        except Exception as e:
            print(f"[MinecraftEnv] Action execution failed: {e}")
            return False

    def _format_command(self, action_idx: int, p_int: np.ndarray, p_ids: np.ndarray, p_txt_ids: np.ndarray) -> str:
        """Map (action, params) to a Mindcraft `!command(...)` string."""
        name = ACTIONS_REGISTRY[action_idx]["name"]

        # Helpers
        def _item_name():
            return self._id_to_name(p_ids[0], ITEM_IDS)

        def _block_name():
            return self._id_to_name(p_ids[0], BLOCK_IDS)

        def _entity_name():
            return self._id_to_name(p_ids[1], MOB_IDS)

        def _mode_name():
            return self._id_to_name(p_ids[2], MODE_IDS)

        def _text(default: str = ""):
            return self._decode_text_ids(p_txt_ids) or default

        # Switch
        if name in ("newAction", "stop", "stfu", "restart", "clearChat", "viewChest",
                    "clearFurnace", "attack", "goToBed", "activate", "endGoal", "endConversation"):
            return f"!{name}()"

        if name in ("goToPlayer", "followPlayer", "attackPlayer", "startConversation",
                    "lookAtPlayer",):
            player = _text("nearest")
            return f"!{name}('{player}')"

        if name == "goToCoordinates":
            x, y, z, closeness = [int(p_int[i]) for i in range(4)]
            return f"!goToCoordinates({x}, {y}, {z}, {max(0, closeness)})"

        if name == "searchForBlock":
            block = _block_name()
            radius = max(1, int(p_int[0]))
            return f"!searchForBlock('{block}', {radius})"

        if name == "searchForEntity":
            entity = _entity_name()
            radius = max(1, int(p_int[0]))
            return f"!searchForEntity('{entity}', {radius})"

        if name == "moveAway":
            distance = max(1, int(p_int[0]))
            return f"!moveAway({distance})"

        if name in ("rememberHere", "goToRememberedPlace", "goal"):
            label = _text("here")
            return f"!{name}('{label}')"

        if name == "givePlayer":
            item = _item_name()
            count = max(1, int(p_int[0]))
            player = _text("nearest")
            return f"!givePlayer('{player}', '{item}', {count})"

        if name == "consume":
            item = _item_name()
            return f"!consume('{item}')"

        if name == "equip":
            slot = int(p_int[0])
            if slot >= 0:
                return f"!equip({slot})"
            item = _item_name()
            return f"!equip('{item}')"

        if name in ("putInChest", "takeFromChest", "discard", "collectBlocks"):
            item_or_block = _item_name() if name != "collectBlocks" else _block_name()
            count = max(1, int(p_int[0]))
            return f"!{name}('{item_or_block}', {count})"

        if name == "craftRecipe":
            recipe = _item_name()
            count = max(1, int(p_int[0]))
            return f"!craftRecipe('{recipe}', {count})"

        if name == "smeltItem":
            item = _item_name()
            return f"!smeltItem('{item}')"

        if name == "placeHere":
            block = _block_name()
            return f"!placeHere('{block}')"

        if name == "stay":
            seconds = max(0, int(p_int[0]))
            return f"!stay({seconds})"

        if name == "setMode":
            mode = _mode_name() or "assistant"
            return f"!setMode('{mode}')"

        if name == "lookAtPosition":
            x, y, z = [int(p_int[i]) for i in range(3)]
            return f"!lookAtPosition({x}, {y}, {z})"

        if name == "digDown":
            depth = max(1, int(p_int[0]))
            return f"!digDown({depth})"

        # Fallback
        return f"!{name}()"

    # --- State & Observations --------------------------------------------------

    def _get_game_state(self) -> MinecraftState:
        """
        Query the current game state from the Mindcraft bot/server.
        This version returns a mocked state; replace with real queries.
        """
        g = 2 * self.view_distance + 1
        state = MinecraftState(
            position=np.random.randn(3).astype(np.float32) * 10.0,
            health=float(20.0),
            hunger=float(20.0),
            inventory={"oak_log": random.randint(0, 10)},
            nearby_blocks=np.random.randint(0, 10, size=(g, g, g), dtype=np.int32),
            nearby_entities=[],
            equipped_item="",
            time_of_day=float(random.random() * 24000.0),
            biome="plains",
            experience=int(0),
            game_mode="survival",
        )
        return state

    def _calculate_reward(self, action_name: str) -> float:
        """Reward from state deltas and task progress."""
        if self._state is None or self._prev_state is None:
            return 0.0
        reward = 0.0

        # Health penalty
        health_diff = float(self._state.health) - float(self._prev_state.health)
        if health_diff < 0.0:
            reward += self.reward_params["damage_taken"] * (-health_diff)

        # Death penalty
        if float(self._state.health) <= 0.0:
            reward += self.reward_params["death"]

        # Inventory pickups
        for item, count in self._state.inventory.items():
            prev = self._prev_state.inventory.get(item, 0)
            if count > prev:
                reward += self.reward_params["item_pickup"] * float(count - prev)

        # Exploration bonus (position not yet marked visited at this point)
        pos_tuple = tuple(self._state.position.astype(int))
        if pos_tuple not in self._visited_positions:
            reward += self.reward_params["exploration"]

        # Task-specific shaping
        reward += self._check_task_progress()

        return float(reward)

    def _check_task_progress(self) -> float:
        """Set self._done on success and return success reward."""
        if self._state is None:
            return 0.0
        r = 0.0

        if self.task == "collect_wood":
            have = self._state.inventory.get("oak_log", 0)
            target = self._task_goals["target_items"]["oak_log"]
            if have >= target:
                r += float(self._task_goals["success_reward"])
                self._done = True

        elif self.task == "mine_iron":
            have = self._state.inventory.get("iron_ingot", 0)
            target = self._task_goals["target_items"]["iron_ingot"]
            if have >= target:
                r += float(self._task_goals["success_reward"])
                self._done = True

        elif self.task == "farm_crops":
            have = self._state.inventory.get("wheat", 0)
            target = self._task_goals["target_items"]["wheat"]
            if have >= target:
                r += float(self._task_goals["success_reward"])
                self._done = True

        elif self.task == "collect_diamonds":
            have = self._state.inventory.get("diamond", 0)
            target = self._task_goals["target_items"]["diamond"]
            if have >= target:
                r += float(self._task_goals["success_reward"])
                self._done = True

        # Add additional tasks as needed...
        return float(r)

    def _calculate_task_progress_vector(self) -> np.ndarray:
        """Task progress indicators normalized to [0,1]."""
        p = np.zeros(10, dtype=np.float32)
        if self._state is None:
            return p

        if self.task == "collect_wood":
            target = float(self._task_goals["target_items"]["oak_log"])
            have = float(self._state.inventory.get("oak_log", 0))
            p[0] = np.float32(min(1.0, max(0.0, have / max(1.0, target))))

        elif self.task == "mine_iron":
            target = float(self._task_goals["target_items"]["iron_ingot"])
            have = float(self._state.inventory.get("iron_ingot", 0))
            p[0] = np.float32(min(1.0, max(0.0, have / max(1.0, target))))
            depth = -float(self._state.position[1]) / 50.0
            p[1] = np.float32(min(1.0, max(0.0, depth)))

        elif self.task == "farm_crops":
            target = float(self._task_goals["target_items"]["wheat"])
            have = float(self._state.inventory.get("wheat", 0))
            p[0] = np.float32(min(1.0, max(0.0, have / max(1.0, target))))

        elif self.task == "collect_diamonds":
            target = float(self._task_goals["target_items"]["diamond"])
            have = float(self._state.inventory.get("diamond", 0))
            p[0] = np.float32(min(1.0, max(0.0, have / max(1.0, target))))

        return p

    def _build_observation(self, reward: float, is_first: bool = False) -> Dict[str, Any]:
        """Assemble observation dict with stable shapes/dtypes."""
        if self._state is None:
            return self._get_zero_observation(reward, is_first)

        inventory_array = np.zeros((36,), dtype=np.float32)
        for i, (_, count) in enumerate(self._state.inventory.items()):
            if i < 36:
                inventory_array[i] = float(count)

        # Entities: up to 20 with 6 features
        ent = np.zeros((20, 6), dtype=np.float32)
        for i, e in enumerate(self._state.nearby_entities[:20]):
            ent[i] = np.array([
                float(e.get("x", 0.0)),
                float(e.get("y", 0.0)),
                float(e.get("z", 0.0)),
                float(e.get("type_id", 0)),
                float(e.get("health", 0.0)),
                float(e.get("distance", 0.0)),
            ], dtype=np.float32)

        task_progress = self._calculate_task_progress_vector()

        obs = {
            "position":         self._state.position.astype(np.float32),
            "velocity":         np.zeros((3,), dtype=np.float32),
            "pitch_yaw":        np.zeros((2,), dtype=np.float32),
            "health":           np.array(self._state.health, dtype=np.float32),
            "hunger":           np.array(self._state.hunger, dtype=np.float32),
            "experience": np.array(float(self._state.experience), dtype=np.float32),
            "inventory":        inventory_array,
            "equipped_item_id": np.array(ITEM_IDS.get(self._state.equipped_item, 0), dtype=np.int32),
            "nearby_blocks": self._state.nearby_blocks.reshape(-1).astype(np.float32),
            "visible_entities": ent,
            "time_of_day":      np.array(self._state.time_of_day, dtype=np.float32),
            "biome_id": np.array(BIOME_IDS.get(self._state.biome, 0), dtype=np.int32),
            "light_level": np.array(15, dtype=np.int32),
            "task_progress":    task_progress,
            "reward":           np.array(reward, dtype=np.float32),
            "is_first":         np.array(is_first, dtype=np.bool_),
            "is_last":          np.array(self._done, dtype=np.bool_),
            "is_terminal":      np.array(self._state.health <= 0.0, dtype=np.bool_),
            "image":            np.zeros((64, 64, 3), dtype=np.uint8),
        }
        obs['instructions_ids'] = np.zeros((2, 32), dtype=np.uint8)
        obs['action_ids'] = np.ones(2, dtype=np.int32) * 0

        # Optional text fields
        if self.tokenizer is not None:
            if "action_text_ids" not in obs:
                obs["action_text_ids"] = np.zeros((self.action_text_len,), dtype=np.int32)
            if "chat_message_ids" not in obs:
                obs["chat_message_ids"] = np.zeros((128,), dtype=np.int32)

        return obs

    def _get_zero_observation(self, reward: float, is_first: bool) -> Dict[str, Any]:
        """Zero-filled observation with correct dtypes/shapes."""
        obs = {}
        for key, space in self.obs_space.items():
            if space.dtype == np.bool_:
                obs[key] = np.array(False, dtype=np.bool_)
            else:
                obs[key] = np.zeros(space.shape, dtype=space.dtype)

        obs["reward"] = np.array(reward, dtype=np.float32)
        obs["is_first"] = np.array(is_first, dtype=np.bool_)
        obs["is_last"] = np.array(False, dtype=np.bool_)
        obs["is_terminal"] = np.array(False, dtype=np.bool_)
        return obs

    # --- Text helpers ----------------------------------------------------------

    def _describe_macro(self, a_idx: int, p_int: np.ndarray, p_ids: np.ndarray, p_txt_ids: np.ndarray) -> str:
        name = ACTIONS_REGISTRY[a_idx]["name"]
        if name in ("goToPlayer", "followPlayer", "attackPlayer", "startConversation", "lookAtPlayer"):
            player = self._decode_text_ids(p_txt_ids) or "nearest"
            return f"{name} {player}"
        if name == "goToCoordinates":
            return f"{name} to ({int(p_int[0])},{int(p_int[1])},{int(p_int[2])})"
        if name in ("searchForBlock", "placeHere", "collectBlocks"):
            return f"{name} {self._id_to_name(p_ids[0], BLOCK_IDS)}"
        if name in ("searchForEntity",):
            return f"{name} {self._id_to_name(p_ids[1], MOB_IDS)}"
        if name in ("givePlayer", "consume", "putInChest", "takeFromChest", "discard", "craftRecipe", "smeltItem"):
            return f"{name} {self._id_to_name(p_ids[0], ITEM_IDS)}"
        if name == "setMode":
            return f"{name} {self._id_to_name(p_ids[2], MODE_IDS)}"
        if name == "lookAtPosition":
            return f"{name} ({int(p_int[0])},{int(p_int[1])},{int(p_int[2])})"
        if name == "digDown":
            return f"{name} {int(p_int[0])}"
        return name

    def _encode_text(self, text: str) -> np.ndarray:
        if self.tokenizer is None:
            # best-effort: zeros
            return np.zeros((self.action_text_len,), dtype=np.int32)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.action_text_len,
            padding="max_length",
            return_tensors="np",
        )
        return tokens["input_ids"].reshape(-1).astype(np.int32)

    def _decode_text_ids(self, ids: np.ndarray) -> str:
        if self.tokenizer is None:
            return ""
        try:
            # Some tokenizers expose batch_decode
            return self.tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip()
        except Exception:
            return ""

    # --- Misc ------------------------------------------------------------------

    @staticmethod
    def _id_to_name(idx: int, table: Dict[str, int]) -> str:
        for name, i in table.items():
            if i == int(idx):
                return name
        return "unknown"

    def _write_stats(self):
        stats = {
            "episode": int(self._episode),
            "task": self.task,
            "length": int(self._length),
            "reward": float(round(self._total_reward, 2)),
            "blocks_broken": dict(self._blocks_broken),
            "items_crafted": dict(self._items_crafted),
            "mobs_killed": dict(self._mobs_killed),
            "positions_explored": int(len(self._visited_positions)),
        }
        if self._logdir is not None:
            filepath = self._logdir / "stats.jsonl"
            with open(str(filepath), "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
        print(
            f"[MinecraftEnv] Episode {self._episode}: "
            f"Task={self.task}, Length={self._length}, Reward={self._total_reward:.2f}"
        )
