import os
import glob
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

import elements
import embodied

import virtualhome
from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz
# Visibility helper used in the notebook demos (partial observation)
try:
    from virtualhome.simulation.evolving_graph import utils as eg_utils
except Exception:
    eg_utils = None


import os
import random
import socket
from typing import Optional, Tuple, Union
def find_free_port():
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))             # Bind to any available port
        return str(s.getsockname()[1])
    
class VirtualHome(embodied.Env):
    """
    VirtualHome wrapper that mirrors the usage pattern in the official notebook.

    - Launches Unity via UnityCommunication (auto/manual like the nb).
    - Step executes one primitive (optionally with a target object) via render_script(skip_animation=True).
    - Observations: image (H,W,3 uint8), visible_ids (padded), reward, flags, log/reward.
    - Action: int32[2] = [primitive_id, target_index]; target_index is over current visible_ids (-1 => no target).
    """

    # Minimal primitive set (expand as needed; names are the notebook's lowercase verbs)
    PRIMITIVES: List[str] = [
        "noop",        # 0
        "turnleft",    # 1
        "turnright",   # 2
        "walkforward", # 3
        "walktowards", # 4 (needs object)
        "open",        # 5 (needs object)
        "close",       # 6 (needs object)
        "grab",        # 7 (needs object)
        "drop",        # 8 (needs object)
        "putin",       # 9 (needs object)
        "putback",     # 10 (needs object)
        "switchon",    # 11 (needs object)
        "switchoff",   # 12 (needs object)
    ]
    ARGLESS = {"noop", "turnleft", "turnright", "walkforward"}

    def __init__(
        self,
        task,
        # Notebook-like launch options
        x_display: Optional[str] = "0",          # notebook sets x_display="0"
        # RL settings
        scene_id: Optional[int] = 0,
        image_size: Tuple[int, int] = (128, 128),
        max_steps: int = 300,
        seed: Optional[int] = None,
        reward_fn=None,                          # callable(prev_graph, graph, info) -> float
        # Image modality (same terms as notebook)
        modality: str = "normal",                # 'normal' | 'seg_class' | 'seg_inst' | 'depth' | 'surf_normals'
        camera_index: Optional[int] = None,      # if None: auto-pick a character camera
        # Housekeeping
        logs: bool = False,
        logdir: Optional[str] = None,
        vlm=None,
        embedder=None,
    ):
        super().__init__()
        self._rng = random.Random(seed or 0)
        self._np = np.random.RandomState(seed or 0)

        file_name = "/app/linux_exec.v2.3.0.x86_64"
        self._comm = UnityCommunication(file_name=file_name, port=find_free_port(), x_display=x_display)

        # === Config & bookkeeping ===
        self._scene_id = scene_id
        self._image_size = image_size
        self._max_steps = max_steps
        self._reward_fn = reward_fn or (lambda prev_g, g, info: 0.0)
        self._modality = modality
        self._camera_index = camera_index  # if None, we’ll auto-pick later

        self._episode = 0
        self._t = 0
        self._done = True
        self._prev_graph = None
        self._visible_ids: List[int] = []
        self._max_objects = 65
        self._ret_cum = 0.0

        self._logs = logs
        self._logdir = elements.Path(logdir) if logdir else None
        if self._logdir:
            self._logdir.mkdir()

        self.vlm = vlm
        self.embedder = embedder

    # ---------------- Embodied interface ----------------

    @property
    def act_space(self):
        max_bound = max(len(self.PRIMITIVES), self._max_objects)
        return {
            "action": elements.Space(np.int32, (2,), 0, max_bound),
            "reset": elements.Space(bool),
        }

    @property
    def obs_space(self):
        H, W = self._image_size
        spaces = {
            "image":        elements.Space(np.uint8, (H, W, 3)),
            "visible_ids":  elements.Space(np.int32, (self._max_objects,), low=0, high=1023),
            "reward":       elements.Space(np.float32),
            "is_first":     elements.Space(bool),
            "is_last":      elements.Space(bool),
            "is_terminal":  elements.Space(bool),
            "log/reward":   elements.Space(np.float32),
        }
        if self.vlm is not None:
            # spaces['instructions'] = elements.Space(np.float32, 384)
            spaces['instructions_ids'] = elements.Space(np.uint8, 32)
            spaces['action_ids'] = elements.Space(np.int32, 2)
        return spaces

    @property
    def act_names(self):
        return self.PRIMITIVES

    def step(self, action: Dict[str, Any]):
        if action["reset"] or self._done:
            return self._reset()

        prim_id, tgt_idx = int(action["action"][0]), int(action["action"][1])
        prim_id = np.clip(prim_id, 0, len(self.PRIMITIVES) - 1).item()
        primitive = self.PRIMITIVES[prim_id]
        needs_arg = primitive not in self.ARGLESS and primitive != "noop"

        # Resolve target object from current visible list
        obj_id = None
        if needs_arg:
            if 0 <= tgt_idx < len(self._visible_ids):
                obj_id = self._visible_ids[tgt_idx]
            else:
                primitive = "noop"  # avoid invalid script

        # Compose a one-line script like the notebook (lowercase verbs)
        if primitive == "noop":
            script_line = None
        elif primitive in self.ARGLESS:
            script_line = f"<char0> [{primitive}]"
        else:
            # Single-arg verbs
            script_line = f"<char0> [{primitive}] <obj> ({obj_id})"

        # Execute fast (skip_animation=True, image_synthesis=[])
        if script_line is not None:
            self._comm.render_script(
                [script_line],
                recording=False,
                skip_animation=True,
                image_synthesis=[],
            )

        # Update graph and image (same APIs the notebook calls)
        graph = self._environment_graph()
        image = self._grab_image()

        # Visible set like the demo (requires evolving_graph.utils)
        self._visible_ids = self._compute_visible_ids(graph)

        r = float(self._reward_fn(self._prev_graph, graph, info={}))
        self._prev_graph = graph
        self._ret_cum += r
        self._t += 1
        self._done = (self._t >= self._max_steps)
        # print(self._visible_ids)
        return self._obs(image, r, is_last=self._done, is_terminal=False)

    def _reset(self):
        self._episode += 1
        self._t = 0
        self._ret_cum = 0.0
        self._done = False

        # Notebook uses: comm.reset(scene_id)
        self._comm.reset(self._scene_id if self._scene_id is not None else 0)

        # Add a default character like in the notebook sections that interact
        # (If you prefer to add explicitly outside, remove this.)
        room = random.choice(['kitchen', 'bedroom', 'livingroom', 'bathroom'])
        try:
            self._comm.add_character('Chars/Female2', initial_room=room)
        except Exception:
            # Fallback in case resource name differs across builds
            self._comm.add_character()

        graph = self._environment_graph()
        self._prev_graph = graph
        self._visible_ids = self._compute_visible_ids(graph)
        image = self._grab_image()
        # print(self._visible_ids)

        return self._obs(image, 0.0, is_first=True)

    def close(self):
        try:
            self._comm.close()
        except Exception:
            pass

    # ---------------- Helpers that mirror the notebook ----------------

    def _environment_graph(self) -> Dict[str, Any]:
        ok, g = self._comm.environment_graph()
        if not ok:
            # Rare, but keep the loop robust
            return self._prev_graph if self._prev_graph is not None else {"nodes": [], "edges": []}
        return g

    def _grab_image(self) -> np.ndarray:
        # If you didn’t set camera_index, mimic the notebook behavior:
        # pick a character-attached camera (last 6 cameras belong to the last character).
        if self._camera_index is None:
            ok, n = self._comm.camera_count()
            if ok and n >= 1:
                # Heuristic similar to the demo: last char camera (back view-ish)
                cam = max(0, n - 1)
            else:
                cam = 0
        else:
            cam = int(self._camera_index)

        ok, ims = self._comm.camera_image(
            [cam],
            mode=self._modality,
            image_width=self._image_size[1],
            image_height=self._image_size[0],
        )
        if not ok or not ims:
            img = np.zeros((*self._image_size, 3), np.uint8)
        else:
            img = ims[0]
            if img.shape[:2] != self._image_size:
                img = cv2.resize(img, (self._image_size[1], self._image_size[0]), interpolation=cv2.INTER_AREA)
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _compute_visible_ids(self, graph: Dict[str, Any]) -> List[int]:
        # Use the same helper as in the notebook: get_visible_nodes(graph, agent_id=1)
        if eg_utils is None or not graph:
            # Fallback: expose all ids
            return [n["id"] for n in graph.get("nodes", [])][: self._max_objects]
        try:
            partial = eg_utils.get_visible_nodes(graph, agent_id=1)  # graphs are 1-based for characters
            return [n["id"] for n in partial["nodes"]]
        except Exception:
            return [n["id"] for n in graph.get("nodes", [])][: self._max_objects]

    def _obs(self, image: np.ndarray, reward: float, is_first=False, is_last=False, is_terminal=False):
        padded = np.full((self._max_objects,), 64, dtype=np.int32)
        if self._visible_ids:
            n = min(self._max_objects, len(self._visible_ids))
            padded[:n] = np.array(self._visible_ids[:n], dtype=np.int32)
        
        obs = {
            "image": image,
            "visible_ids": padded,
            "reward": np.float32(reward),
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "log/reward": np.float32(self._ret_cum),
        }
        if self.vlm is not None:
            # spaces['instructions'] = elements.Space(np.float32, 384)
            obs['instructions_ids'] = np.zeros(32)
            obs['action_ids'] = np.ones(2) * -100
        return obs

# --- small helper ---
def sys_platform():
    import sys
    return sys.platform  # 'linux', 'darwin', 'win32'
