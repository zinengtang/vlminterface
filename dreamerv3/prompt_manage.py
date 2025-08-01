"""
Structured high‑level prompt templates for instructing an LLM game‑agent.  
Each template encourages the agent to:
  • Formulate a short PLAN.
  • Emit medium‑level, imperative ACTIONS (5‑10 words each).  
  • Maintain and update a distilled MEMORY_BANK summarizing past context.  
The low‑level controller outside the LLM consumes these ACTIONS and handles concrete controls.  
"""

from dataclasses import dataclass
from textwrap import dedent
from typing import Dict

@dataclass(frozen=True)
class PromptTemplate:
    """Container for a system + user prompt."""
    system: str
    user: str

# ---------------------------------------------------------------------------
# Generic template — extend or reuse when adding new games
# ---------------------------------------------------------------------------
GENERIC_PROMPT = PromptTemplate(
    system=dedent(
        """
        You are a high‑level strategist for the game <GAME>.  
        • Think step‑by‑step but do NOT reveal chain‑of‑thought.  
        • Output strictly in the sections requested.  
        • Use the game‑specific knowledge below to ground your reasoning.  
        • When you learn something useful, condense it (<50 words) into MEMORY_UPDATE.  
        • Never issue low‑level keystrokes; only medium‑level, multi‑word ACTIONS.  
        • Keep each ACTION imperative and between 5 and 10 words.  
        • After finishing all goals, emit the single action: “END TASK”.  
        """
    ),
    user=dedent(
        """
        <CONTEXT>
        Current Goal: {goal}
        Observation: {observation}
        Memory Bank:
        {memory_bank}
        </CONTEXT>

        === PLAN ===
        (bullet list, 3‑7 items)

        === ACTIONS ===
        1. <action>
        2. <action>
        ...

        === MEMORY_UPDATE ===
        <distilled_new_memory>
        """
    ),
)

# ---------------------------------------------------------------------------
# Game‑specific templates
# ---------------------------------------------------------------------------

def mc_tasks() -> str:
    """Details used in Minecraft templates."""
    return dedent(
        """
        Minecraft specifics:
        • Resources: wood, stone, iron, diamond, food, wool (white, red, blue, green, yellow, etc.).
        • Common verbs: gather <resource>, craft <item>, explore <direction>, build <structure>, fight <enemy>.
        • Coordinates are (x, y, z); y is height.  
        • Tool progression: wood → stone → iron → diamond.
        • Avoid nighttime exposure without shelter or armor.
        """
    )

def crafter_tasks() -> str:
    return dedent(
        """
        Crafter (Janner et al., 2021) specifics:
        • Objective: maximize cumulative reward through crafting.
        • Reward tiers: craft stone_pickaxe → furnace → iron_pickaxe → house, etc.
        • Environment is 64×64; agent starts at (0,0).
        • Key resources: wood, cobblestone, iron_ore, coal.
        • Survival aspects are disabled; focus on efficiency.
        """
    )

def montezuma_tasks() -> str:
    return dedent(
        """
        Atari Montezuma’s Revenge specifics:
        • Goal: navigate rooms, collect keys, avoid enemies, reach high scores.
        • Avatar can: move <direction>, jump <direction>, climb ladder, collect key, open door.
        • One life only; falling or touching enemies loses a life.
        • Room coordinates are implicit; use relative navigation (e.g., go right ladder‑up).
        """
    )

def pico_park_tasks() -> str:
    return dedent(
        """
        Pico Park 2 specifics (single‑agent emulation of coop):
        • Solve puzzle levels by carrying keys, pushing blocks, toggling switches.
        • Actions should refer to abstract teammates (e.g., “signal teammate jump”, “stack on partner”).
        • Objective: get all players to goal door simultaneously with key.
        • Use concise actions: coordinate jump‑stack, push block right, grab key.
        """
    )

def blueprint_build_tasks() -> str:
    return dedent(
        """
        Custom Minecraft Blueprint Build task:
        • Blueprint provided as 2D layers, each cell maps to a wool color.
        • Steps: gather dyes, craft wools, scaffold, place blocks coordinate‑wise.
        • Accuracy critical; verify each layer before proceeding.
        • End when structure matches blueprint checksum.
        """
    )

GAME_PROMPTS: Dict[str, PromptTemplate] = {
    "minecraft": PromptTemplate(
        system=GENERIC_PROMPT.system.replace("<GAME>", "Minecraft") + mc_tasks(),
        user=GENERIC_PROMPT.user,
    ),
    "crafter": PromptTemplate(
        system=GENERIC_PROMPT.system.replace("<GAME>", "Crafter") + crafter_tasks(),
        user=GENERIC_PROMPT.user,
    ),
    "montezuma": PromptTemplate(
        system=GENERIC_PROMPT.system.replace("<GAME>", "Montezuma’s Revenge") + montezuma_tasks(),
        user=GENERIC_PROMPT.user,
    ),
    "pico_park_2": PromptTemplate(
        system=GENERIC_PROMPT.system.replace("<GAME>", "Pico Park 2") + pico_park_tasks(),
        user=GENERIC_PROMPT.user,
    ),
    "mc_blueprint_build": PromptTemplate(
        system=GENERIC_PROMPT.system.replace("<GAME>", "Minecraft Blueprint Build") + blueprint_build_tasks(),
        user=GENERIC_PROMPT.user,
    ),
}

# ---------------------------------------------------------------------------
# Helper: render a ready‑to‑send prompt
# ---------------------------------------------------------------------------

def render_prompt(game: str, goal: str, observation: str, memory_bank: str) -> str:
    """Return the concatenated SYSTEM + USER prompt with formatted slots."""
    if game not in GAME_PROMPTS:
        raise KeyError(f"Unknown game {game!r}. Available: {list(GAME_PROMPTS)}")
    tmpl = GAME_PROMPTS[game]
    user_part = tmpl.user.format(goal=goal, observation=observation, memory_bank=memory_bank)
    return tmpl.system + "\n" + user_part

# Example usage --------------------------------------------------------------
if __name__ == "__main__":
    example = render_prompt(
        game="minecraft",
        goal="Secure food for first night",
        observation="Spawned in plains near oak trees and sheep",
        memory_bank="(empty)",
    )
    print(example)
