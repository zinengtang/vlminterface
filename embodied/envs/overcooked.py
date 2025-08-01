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

# SHAPING = {
#     "PLACEMENT_IN_POT_REW": 3.0,    # + when you place final ingredient
#     "SOUP_PICKUP_REWARD": 5.0,      # + when agent picks up a finished soup
#     "DELIVERY_REWARD": 20.0,        # final sparse reward
# }
tasks_list = [
    "asymmetric_advantages_tomato",
    # "bonus_order_test",
    "bottleneck",
    # "centre_objects",
    # "centre_pots",
    # "coordination_ring",
    "corridor",
    # "counter_circuit",
    # "counter_circuit_o_1order",
    # "cramped_corridor",
    "cramped_room",
    # "cramped_room_o_3orders",
    # "cramped_room_single",
    # "cramped_room_tomato",
    # "five_by_five",
    # "forced_coordination",
    # "forced_coordination_tomato",
    # "inverse_marshmallow_experiment",
    # "large_room",
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
}

class Overcooked(embodied.Env):
    ACTIONS = [
        Action.STAY,         # act0
        Direction.NORTH,     # act1 (up)
        Direction.SOUTH,     # act2 (down)
        Direction.WEST,      # act3 (left)
        Direction.EAST,      # act4 (right)
        Action.INTERACT,     # act5
    ]

    def init_env(self):
        task = random.choice(tasks_list)
        mdp = OvercookedGridworld.from_layout_name(task)
        self._mlam = MediumLevelActionManager.from_pickle_or_compute(
            mdp, NO_COUNTERS_PARAMS, force_compute=True
        )
        self._env = OvercookedEnv.from_mdp(mdp, 
            info_level=0, horizon=333)

    def __init__(self, task="asymmetric_advantages", horizon=400, reward_shaping=None, logs=False, logdir=None, seed=None, vlm=None, embedder=None,):
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

    @property
    def act_names(self):
        return self.ACTIONS

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
            # spaces['instructions'] = elements.Space(np.float32, 384)
            spaces['instructions_ids'] = elements.Space(np.uint8, 32)
            spaces['action_ids'] = elements.Space(np.int32, 2)
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
        return self._obs(feat0, feat1, image, reward, is_last=self._done, is_terminal=False)

        
    def _render_image(self):
        rewards_dict = {}  # e.g. pull from self._env.game_stats if you want HUD numbers
        surf = self._visualizer.render_state(
            state=self._env.state,
            grid=self._env.mdp.terrain_mtx,
            hud_data=StateVisualizer.default_hud_data(self._env.state, **rewards_dict),
        )
        arr = pygame.surfarray.array3d(surf)
        arr = np.flip(np.rot90(arr, 3), 1)        # match orientation used in repo
        # Resize if you want a fixed resolution; otherwise skip this:
        arr = cv2.resize(arr, (128, 128))
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
        return self._obs(feat0, feat1, image, 0.0, is_first=True)

    def _obs(self, feat0, feat1, image, reward, is_first=False, is_last=False, is_terminal=False):
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
            # spaces['instructions'] = elements.Space(np.float32, 384)
            obs['instructions_ids'] = np.zeros(32)
            obs['action_ids'] = np.ones(2) * -100
        self._step += 1
        return obs

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
