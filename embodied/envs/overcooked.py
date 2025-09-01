import json
import random
from typing import Tuple, List, Dict, Any

import elements
import embodied
import numpy as np
import pygame, cv2

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.mdp.actions import Action, Direction

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # before importing pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
# Optional: self._visualizer.configure(tile_size=60, is_rendering_hud=False, ...)

from overcooked_ai_py.mdp.actions import Action, Direction

# add at top with your other imports
# add with your imports
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator

def _mdp_schedule_fn(_info):
    import random
    w = random.randint(5, 8)   # inner width
    h = random.randint(4, 6)   # inner height
    return dict(
        inner_shape=(w, h),
        prop_empty=random.uniform(0.85, 0.95),
        prop_feats=random.uniform(0.08, 0.18),
        start_all_orders=[{"ingredients": ["onion","onion","onion"]}],
        recipe_values=[20],
        recipe_times=[20],
        display=False,
        # or set generate_all_orders=True to auto-make recipes
    )

def _make_random_mdp():
    # IMPORTANT: give a non-None outer_shape when using a schedule
    # Pick something comfortably larger than the max inner shape.
    OUTER_SHAPE = (12, 10)  # (width, height)
    mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(
        mdp_params=None,
        outer_shape=OUTER_SHAPE,
        mdp_params_schedule_fn=_mdp_schedule_fn,
    )
    return mdp_fn({})  # returns an OvercookedGridworld


# === NEW: simple enums for compact IDs ========================================
INTERACT_IDS = {
    "NONE": 0,
    "ONION_DISPENSER": 1,
    "TOMATO_DISPENSER": 2,
    "DISH_DISPENSER": 3,
    "POT": 4,
    "COUNTER": 5,
    "SERVE": 6,
    "TRASH": 7,
    "ONION": 11,
    "TOMATO": 12,
    "DISH": 13,
    "SOUP": 14,
    "PARTNER": 20,
}

# Terrain character -> interact tag
# (These are the standard Overcooked-AI layout chars.)
TERRAIN_TO_TAG = {
    "O": "ONION_DISPENSER",
    "T": "TOMATO_DISPENSER",
    "D": "DISH_DISPENSER",
    "P": "POT",
    "S": "SERVE",
    "X": "COUNTER",  # counters/walls
    "#": "COUNTER",
    "C": "COUNTER",
}

DIR_TO_VEC = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0,  1),
    Direction.WEST:  (-1, 0),
    Direction.EAST:  (1,  0),
}
# SHAPING = {
#     "PLACEMENT_IN_POT_REW": 3.0,    # + when you place final ingredient
#     "SOUP_PICKUP_REWARD": 5.0,      # + when agent picks up a finished soup
#     "DELIVERY_REWARD": 20.0,        # final sparse reward
# }
tasks_list = [
    "counter_circuit",
    # "bonus_order_test",
    "bottleneck",
    # "centre_objects",
    # "centre_pots",
    "coordination_ring",
    # "corridor",
    # "counter_circuit_o_1order",
    "cramped_corridor", 
    "cramped_room",
    # "cramped_room_o_3orders",
    # "cramped_room_single",
    # "cramped_room_tomato",
    # "five_by_five",
    "forced_coordination",
    "forced_coordination_tomato",
    # "inverse_marshmallow_experiment",
    "large_room",
    # "long_cook_time",
    # "m_shaped_s",
    # "marshmallow_experiment",
    # "marshmallow_experiment_coordination",
    # "mdp_test",
    # "multiplayer_schelling",
    # "old_dynamics_cook_test",
    # "old_dynamics_put_test",
    # "pipeline",
    # "random0",
    # "random3",
    # "scenario1_s",
    # "scenario2",
    # "scenario2_s",
    # "scenario3",
    # "scenario4",
    # "schelling",
    # "schelling_s",
    # "simple_o",
    # "simple_o_t",
    # "simple_tomato",
    # "small_corridor",
    # "soup_coordination",
    # "unident",
    # "you_shall_not_pass"
]

BASE_REW_SHAPING_PARAMS = {
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 1,
    "POT_DISTANCE_REW": 1,
    "SOUP_DISTANCE_REW": 1,
    "useful_onion_pickup": 0.5,
    "useful_tomato_pickup": 0.5,
    # "useful_dish_drop": 0.5,          # handoff near pots/partner
    # "soup_drop": -5.0,
    # "useless_dish_drop": -0.5,
    # "optimal_onion_potting": 1.0,
    # "optimal_tomato_potting": 1.0,
    "viable_onion_potting": 0.3,
    "viable_tomato_potting": 0.3,
    # "useless_onion_potting": -0.5,
    # "useless_tomato_potting": -0.5,
    # "catastrophic_onion_potting": -2.0,
    # "catastrophic_tomato_potting": -2.0,
}


class Overcooked(embodied.Env):
    ACTIONS = [
        Action.STAY,          # 0
        Direction.NORTH,      # 1
        Direction.SOUTH,      # 2
        Direction.WEST,       # 3
        Direction.EAST,       # 4
        Action.INTERACT,      # 5
    ]

    # Human-readable names for static terrain chars used by Overcooked layouts.
    _TERRAIN_NAME = {
        ' ': 'floor',
        'X': 'counter',
        'P': 'pot',
        'D': 'dish',
        'O': 'onion',
        'T': 'tomato',
        'S': 'serve',
        '#': 'wall',
    }

    _ACT_DICT = {
        Action.STAY: 'stay', 
        Direction.NORTH: 'north', 
        Direction.SOUTH: 'south', 
        Direction.WEST: 'west', 
        Direction.EAST: 'east', 
        Action.INTERACT: 'interact', 
    }

    def init_env(self):
        task = random.choice(tasks_list)
        # mdp = _make_random_mdp()
        mdp = OvercookedGridworld.from_layout_name(task)
        self._mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=True
        )
        self._env = OvercookedEnv.from_mdp(mdp, info_level=0, horizon=333)

    def __init__(
        self,
        task="asymmetric_advantages",
        horizon=400,
        reward_shaping=None,
        logs=False,
        logdir=None,
        seed=None,
        vlm=None,
        embedder=None,
        tokenizer = None,      # <-- NEW
        action_text_len: int = 64,            # <-- NEW (change if you want 32)
    ):
        super().__init__()
        self.init_env()

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
        self._visualizer = StateVisualizer()
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased") 

        self.action_text_len = int(action_text_len)
        self.total_actino_text_len = 2 * self.action_text_len
        if self.tokenizer is not None:
            # pick a sane pad id even if tokenizer has none
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is None:
                # try eos; otherwise fallback 0
                pad_id = getattr(self.tokenizer, "eos_token_id", 0) or 0
            self._pad_id = int(pad_id)
        else:
            self._pad_id = 0

    @property
    def act_names(self):
        return list(self._ACT_DICT.values())

    @property
    def obs_space(self):
        dummy_feat0, dummy_feat1 = self._env.mdp.featurize_state(self._env.state, self._mlam)
        dummy_image = self._render_image()
        spaces = {
            "state_0": elements.Space(np.float32, dummy_feat0.shape),
            "state_1": elements.Space(np.float32, dummy_feat1.shape),
            "image": elements.Space(np.uint8, dummy_image.shape),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool),
            "is_last": elements.Space(bool),
            "is_terminal": elements.Space(bool),
            "log/reward": elements.Space(np.float32),
        }
        if self.vlm is not None:
            spaces['instructions_ids'] = elements.Space(np.uint8, (2, 32), 0, 100_000)
            spaces['action_ids'] = elements.Space(np.int32, 2)
        # NEW: always expose action_text_ids if tokenizer provided
        if self.tokenizer is not None:
            spaces['action_text_ids'] = elements.Space(np.int32, self.total_actino_text_len)
        return spaces

    @property
    def act_space(self):
        return {
            "action": elements.Space(np.int32, (2,), 0, len(self.ACTIONS)),
            "reset": elements.Space(bool),
        }

    def step(self, action):
        if action["reset"] or self._done:
            return self._reset()

        # Snapshot pre-step player states to determine what is *in front of* each agent.
        prev_players = list(self._env.state.players)
        joint_action = tuple(self.ACTIONS[int(a)] for a in action["action"])
        state, sparse_reward, self._done, info = self._env.step(joint_action)
        shaped = info.get("shaped_r_by_agent", [])
        shaped_reward = float(np.sum(shaped))
        reward = sparse_reward + shaped_reward
        self._reward += reward
        self._length += 1

        if self._done and self._logdir:
            self._write_stats(self._length, self._reward)

        feat0, feat1 = self._env.mdp.featurize_state(state, self._mlam)
        image = self._render_image()

        # --- NEW: build the short action text for each agent and tokenize ---
        action_text_ids = None
        if self.tokenizer is not None:
            texts = []
            for i, a in enumerate(joint_action):
                # print(joint_action)
                if prev_players[i].held_object is not None:
                    hold_obj = str(prev_players[i].held_object.name)
                if a == Action.INTERACT:
                    terrain = self._terrain_ahead(prev_players[i])
                    if not terrain in ['floor', 'counter', 'wall']:
                        texts.append(f"interact {terrain}")
                    else:
                        texts.append(f"")
                else:
                    move_act = self._ACT_DICT[a]
                    terrain = self._terrain_around(prev_players[i])
                    if prev_players[i].held_object is not None:
                        move_act = move_act + f" hold {hold_obj}" 
                    if terrain is not None:
                        move_act = move_act + f" at {terrain}"
                    texts.append(move_act)  # movement or stay → empty string
            # print(texts)
            if len(texts) < 2:
                print(joint_action)
            if 1 <= len(texts) < 2:
                texts.append("")
            if len(texts) < 1:
                texts.append("")
                texts.append("")
            action_text_ids = self._encode_action_texts(texts)
        
        # print(texts)
        return self._obs(
            feat0, feat1, image, reward,
            is_last=self._done, is_terminal=False,
            action_text_ids=action_text_ids,   # <-- NEW
        )

    def _render_image(self):
        rewards_dict = {}
        surf = self._visualizer.render_state(
            state=self._env.state,
            grid=self._env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(self._env.state, **rewards_dict),
        )
        arr = pygame.surfarray.array3d(surf)
        arr = np.flip(np.rot90(arr, 3), 1)
        arr = cv2.resize(arr, (64, 64))
        return arr.astype(np.uint8)

    def _reset(self):
        self.init_env()
        self._env.reset()
        self._episode += 1
        self._length = 0
        self._reward = 0.0
        self._done = False
        self._step = 0
        feat0, feat1 = self._env.mdp.featurize_state(self._env.state, self._mlam)
        image = self._render_image()

        # empty tokens on reset
        action_text_ids = None
        if self.tokenizer is not None:
            action_text_ids = self._encode_action_texts(["", ""])

        return self._obs(feat0, feat1, image, 0.0, is_first=True, action_text_ids=action_text_ids)

    def _obs(self, feat0, feat1, image, reward,
             is_first=False, is_last=False, is_terminal=False,
             action_text_ids = None):
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
            obs['instructions_ids'] = np.zeros((2, 32), dtype=np.uint8)
            obs['action_ids'] = np.ones(2, dtype=np.int32) * 0

        if self.tokenizer is not None:
            if action_text_ids is None:
                action_text_ids = np.full(self.total_actino_text_len, self._pad_id, dtype=np.int32)
            obs['action_text_ids'] = action_text_ids.flatten()

        self._step += 1
        return obs
    
    def _terrain_around(self, player) -> str:
        """Return a readable terrain name that's directly in front of `player`."""
        terrains = []
        for dx, dy in [(1,0), (-1,0), (0,-1), (0,1)]:
            x, y = player.position
            # dx, dy = player.orientation
            tx, ty = x + dx, y + dy

            grid = self._env.mdp.terrain_mtx
            if ty < 0 or ty >= len(grid) or tx < 0 or tx >= len(grid[0]):
                continue
            
            ch = grid[ty][tx]
            terrain = self._TERRAIN_NAME.get(ch, str(ch))
            # print(self._TERRAIN_NAME.get(ch, str(ch)), ch, x, y, player.orientation)
            if terrain not in ['floor', 'counter', 'wall']:
                terrains.append(terrain)
        if len(terrains) > 0:
            return random.choice(terrains)
        else:
            return None

    def _terrain_ahead(self, player) -> str:
        """Return a readable terrain name that's directly in front of `player`."""
        x, y = player.position
        dx, dy = player.orientation
        tx, ty = x + dx, y + dy

        grid = self._env.mdp.terrain_mtx
        if ty < 0 or ty >= len(grid) or tx < 0 or tx >= len(grid[0]):
            return "out_of_bounds"

        ch = grid[ty][tx]
        # print(self._TERRAIN_NAME.get(ch, str(ch)), ch, x, y, player.orientation)
        return self._TERRAIN_NAME.get(ch, str(ch))

    def _encode_action_texts(self, texts: List[str]) -> np.ndarray:

        """Tokenize 2 texts, pad/truncate to action_text_len, then concat → (2*L,)."""
        batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=self.action_text_len,
            padding="max_length",
        )
        ids = np.array(batch["input_ids"], dtype=np.int32)  # (2, L)
        if self._pad_id != 0:
            ids = np.where(ids == 0, self._pad_id, ids)
        return ids.flatten()  # (2*L,)
    
    # ---------- logging ----------
    def _write_stats(self, length, reward):
        stats = {
            "episode": self._episode,
            "length": length,
            "reward": round(reward, 1),
        }
        filepath = self._logdir / "stats.jsonl"
        lines = filepath.read() if filepath.exists() else ""
        lines += json.dumps(stats) + "\n"
        filepath.write(lines, mode="w")
        print(f"[Overcooked] Wrote stats to {filepath}")
