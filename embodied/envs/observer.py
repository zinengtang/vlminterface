# observer_camera.py — add‑on to your existing MineRL stack
"""
Plug‑and‑play **camera‑only agent** that lets you grab a real RGB frame from
any (x, y, z, yaw, pitch) in the world, *without* teleporting the player.  It
works by:

1.   Injecting a second `<AgentSection>` called **Observer** into the mission
     XML before the mission starts.  That section only contains a
     `<VideoProducer>` handler – no movement handlers – so it never acts in the
     world but still renders pixels.
2.   Spinning up a dedicated `malmo.AgentHost` for that agent and joining the
     same mission as your regular MineRL agent.
3.   Every time you call `capture()`, the helper teleports the Observer via a
     chat command, steps the world a couple of ticks, and returns the RGB
     frame.

The helper is completely self‑contained: import it *after* you created your
MineRL env and call `ObserverCamera(env)` once.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from minerl.herobraine.hero.mc import INVERSE_KEYMAP  # type: ignore

try:
    import MalmoPython as malmo
except ImportError as e:  # pragma: no cover  — makes mypy happy during docs build
    raise ImportError("MalmoPython is required for ObserverCamera") from e

###############################################################################
#  helper that finds the low‑level Malmo object — copy‑paste from your stack  #
###############################################################################

def _unwrap_to_malmo(env):
    """Drill through Gym / Embodied wrappers and return the bare Malmo core."""
    core = env
    while hasattr(core, "env"):
        core = core.env
    # MineRL's Gym env exposes the Malmo objects as .agent_host / .mission_spec
    if not (hasattr(core, "agent_host") and hasattr(core, "mission_spec")):
        raise AttributeError("Could not find Malmo objects on env; make sure you "
                             "pass the *raw* MineRL environment, not a vectorised "
                             "or already‑running wrapper, to ObserverCamera.")
    return core

###############################################################################
#  main class                                                                 #
###############################################################################

class ObserverCamera:
    """Attach to an existing MineRL env and provide `capture()` images."""

    def __init__(self, env, *, width: int = 64, height: int = 64, fov: int = 60):
        self._core = _unwrap_to_malmo(env)
        self._player_host: malmo.AgentHost = self._core.agent_host  # type: ignore
        self._player_mission: malmo.MissionSpec = self._core.mission_spec  # type: ignore

        self._width, self._height, self._fov = width, height, fov

        # ------------------------------------------------------------------
        # 1. patch the XML *before* the mission starts (only once!)
        # ------------------------------------------------------------------
        if not self._player_mission.getMissionXML().count("<AgentSection>") == 2:
            self._inject_observer_section()

        # ------------------------------------------------------------------
        # 2. spin up the secondary AgentHost and join the (already prepared)
        #    mission.  We keep it around for the lifetime of the wrapper.
        # ------------------------------------------------------------------
        self._obs_host = malmo.AgentHost()
        self._obs_host.setObservationsPolicy(malmo.AgentHost.OBSERVATIONS_POLICY_LATEST)
        self._obs_host.setVideoPolicy(malmo.AgentHost.VIDEO_POLICY_LATEST)

        # The Minecraft server is already up because MineRL called
        #   .startMission() for the player. We just have to connect.
        self._obs_host.startMission(self._player_mission, self._core.mission_record,  # type: ignore
                                    1,  # role=1   (player is 0)
                                    "Observer")

        # wait until we actually join — otherwise first capture() could race
        while True:
            world_state = self._obs_host.getWorldState()
            if world_state.is_mission_running:
                break

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def capture(self, x: float, y: float, z: float,
                yaw: float, pitch: float,
                n_ticks: int = 2) -> np.ndarray:
        """Teleport camera and return a `(H, W, 3)` uint8 RGB numpy array."""
        cmd = f"/tp Observer {x:.3f} {y:.3f} {z:.3f} {yaw:.1f} {pitch:.1f}"
        self._player_host.sendCommand(f"chat {cmd}")
        # advance the world → make sure the teleport is applied + a frame rendered
        for _ in range(n_ticks):
            self._player_host.sendCommand("move 0")

        # fetch the *latest* video frame from the observer
        world_state = self._obs_host.getWorldState()
        if not world_state.video_frames:
            raise RuntimeError("No video frame received from Observer agent.")
        frame = world_state.video_frames[-1]
        # Malmo gives raw bytes; reshape into H×W×4 BGRA then drop alpha & swap
        img = np.frombuffer(frame.pixels, dtype=np.uint8)
        img = img.reshape((frame.height, frame.width, 4))[..., :3]
        img = img[..., ::-1]  # BGR → RGB
        return img.copy()     # contiguous, safe to keep after next tick

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _inject_observer_section(self) -> None:
        """Add a second AgentSection with a VideoProducer to the mission XML."""
        xml = self._player_mission.getMissionXML()
        root = ET.fromstring(xml)
        ms_node = root if root.tag == "Mission" else root.find("Mission")
        if ms_node is None:
            raise ValueError("Malformed mission XML — no <Mission> root node")

        # --- build observer section ------------------------------------------------
        agent_sec = ET.SubElement(ms_node, "AgentSection")
        ET.SubElement(agent_sec, "Name").text = "Observer"

        # <AgentStart>
        agent_start = ET.SubElement(agent_sec, "AgentStart")
        placement = ET.SubElement(agent_start, "Placement")
        placement.set("x", "0.5")
        placement.set("y", "80")
        placement.set("z", "0.5")
        placement.set("yaw", "0")
        placement.set("pitch", "0")

        # <AgentHandlers>
        handlers = ET.SubElement(agent_sec, "AgentHandlers")
        vp = ET.SubElement(handlers, "VideoProducer")
        vp.set("want_depth", "false")
        ET.SubElement(vp, "Width").text = str(self._width)
        ET.SubElement(vp, "Height").text = str(self._height)
        ET.SubElement(vp, "FOV").text = str(self._fov)

        # ------------------------------------------------------------------
        #  Write back the modified XML to the *same* MissionSpec so MineRL
        #  still thinks it's the mission it created.
        # ------------------------------------------------------------------
        new_xml = ET.tostring(root, encoding="unicode")
        self._player_mission.setMissionXML(new_xml)

###############################################################################
#  convenience shim so you can just do:                                      #
#      cam = attach_observer(env)                                            #
#      img = cam.capture(123, 64, 456, 90, 0)                                #
###############################################################################

def attach_observer(env, **kw) -> "ObserverCamera":
    """Return a *singleton* ObserverCamera for the given env; cache per env."""
    if not hasattr(env, "_observer_camera"):
        env._observer_camera = ObserverCamera(env, **kw)  # type: ignore[attr-defined]
    return env._observer_camera
