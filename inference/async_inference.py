#!/usr/bin/env python3
"""
async_inference.py
==================

Asynchronous planner–controller inference for the DreamerV3-based controller
and a vision-language planner (VLM). The controller runs every env step at high
frequency; the planner runs in the background and injects short instructions
when the controller's STOP head indicates completion (or when no instruction is
active).

Usage (single GPU, 1 env):
--------------------------
python async_inference.py \
  --task overcooked_l2_simple \
  --from_checkpoint /path/to/checkpoint \
  --model_type qwenvl \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 64 \
  --temperature 0.4

Notes
-----
- This script uses `dreamerv3.main.make_agent` and `dreamerv3.main.make_env` to
  construct the controller and environment from repo configs.yaml.
- It wraps the env with `web.utils.ManualInstrWrapper` and uses `dreamerv3.vlm_utils.VLMWrapper`
  for planning.
- The controller's STOP probability (p_stop) comes from Agent.policy(..., return_stop_token=True).
- If you want to use the environment's built-in instruction fields (e.g., Overcooked),
  this wrapper will simply inject/override the instruction token ids.
"""

from __future__ import annotations

import argparse
import asyncio as aio
import collections
import functools as ft
import os
import time
from typing import Deque, Optional

import numpy as np
from PIL import Image

# Local project imports (ensure repo root on path)
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import elements

from dreamerv3 import main as drv3_main
from .vlm_infer import VLMInference
from dreamerv3.sentence_embedding import SentenceEmbedder
from web.utils import ManualInstrWrapper, build_config


def _pil_from_obs_image(img: np.ndarray) -> Image.Image:
    """Convert an HxWxC uint8 image array to a PIL Image."""
    if img is None:
        return None
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 2:
        return Image.fromarray(img, mode="L")
    elif img.shape[-1] == 1:
        return Image.fromarray(img[..., 0], mode="L")
    else:
        return Image.fromarray(img)


async def planner_task(
    vlm: VLMInference,
    frame_buf: Deque[Image.Image],
    instr_setter,
    need_instr_evt: aio.Event,
    stop_evt: aio.Event,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    proactive_check: bool = False,
) -> None:
    """
    Background planner: waits for `need_instr_evt`, then generates an instruction
    from recent frames and calls `instr_setter(text)` to inject it into the env.
    """
    try:
        while not stop_evt.is_set():
            # Wait until controller asks for a new instruction
            await need_instr_evt.wait()
            need_instr_evt.clear()
            if stop_evt.is_set():
                break

            # Snapshot a few most recent frames (latest first)
            frames = list(frame_buf)[-3:] or list(frame_buf)
            if not frames:
                # No frames yet; try again soon.
                await aio.sleep(0.01)
                continue

            # The VLM wrapper expects PIL images list; action_lists can be None.
            try:
                if proactive_check:
                    # Optional precheck: ask VLM if it wants a new instruction.
                    if not vlm.should_emit(frames=[frames[-1]]):
                        # Defer; controller will ping us again soon.
                        continue

                texts = vlm(frames=frames, action_lists=None,
                           max_new_tokens=max_new_tokens,
                           temperature=temperature)
                text = texts[0] if texts else ""
                if not text:
                    # Degenerate; skip to next loop
                    continue
                instr_setter(text)
            except Exception as e:
                print(f"[planner] VLM generation error: {e}")
                # Backoff slightly to avoid hot loop on errors
                await aio.sleep(0.05)

    finally:
        pass


def _stack_time_batch(obs: dict) -> dict:
    """Add (B=1,T=1) dims for policy()."""
    add = lambda x: x[None, None, ...] if isinstance(x, np.ndarray) else np.array([[x]])
    return {k: add(v) for k, v in obs.items()}


async def controller_task(
    agent,
    env,
    frame_buf: Deque[Image.Image],
    need_instr_evt: aio.Event,
    stop_evt: aio.Event,
    max_steps: Optional[int] = None,
    stop_threshold: float = 0.65,
    log_interval_sec: float = 5.0,
) -> None:
    """
    Controller loop: steps the env, calls policy(return_stop_token=True), and
    notifies the planner when a new instruction is needed.
    """
    step = 0
    episode = 0
    last_log = time.time()

    # Initialize recurrent carry for batch_size=1
    carry = agent.init_policy(1)

    # Reset environment
    obs = env.reset()
    last_frame = _pil_from_obs_image(obs.get("image"))
    if last_frame is not None:
        frame_buf.append(last_frame)

    # Ask for an instruction at episode start
    need_instr_evt.set()

    try:
        while (max_steps is None) or (step < max_steps):
            # Shape obs to (B=1,T=1,...)
            obs_bt = _stack_time_batch(obs)

            # Evaluate policy and STOP head
            carry, acts, outs, p_stop = agent.policy(carry, obs_bt, mode="eval", return_stop_token=True)
            # p_stop may be a ndarray with shape (B,T) or (B,); extract scalar
            if hasattr(p_stop, "shape"):
                try:
                    p_stop_scalar = float(np.asarray(p_stop).reshape(-1)[-1])
                except Exception:
                    p_stop_scalar = float(np.asarray(p_stop).mean())
            else:
                p_stop_scalar = float(p_stop)

            # Clear instruction if STOP fired; planner will refill
            env.clear_instruction_if_stopped(p_stop_scalar)
            if not env.has_active_instruction():
                need_instr_evt.set()

            # Step environment with current action (unbatched dict)
            act_unbatched = {k: (np.asarray(v)[0, 0] if isinstance(v, np.ndarray) and v.ndim >= 2 else v)
                             for k, v in acts.items()}
            obs = env.step(act_unbatched)

            # Cache last frame for planner context
            last_frame = _pil_from_obs_image(obs.get("image"))
            if last_frame is not None:
                frame_buf.append(last_frame)
                if len(frame_buf) > 8:
                    frame_buf.popleft()

            step += 1

            # Handle episode end and reset
            if obs.get("is_last", False):
                episode += 1
                obs = env.reset()
                env.clear_instruction_if_stopped(1.0)  # force clear at reset
                need_instr_evt.set()
                last_frame = _pil_from_obs_image(obs.get("image"))
                if last_frame is not None:
                    frame_buf.clear()
                    frame_buf.append(last_frame)

            # Periodic logging
            if (time.time() - last_log) >= log_interval_sec:
                last_log = time.time()
                rew = float(obs.get("log/reward", obs.get("reward", 0.0)))
                print(f"[controller] step={step} ep={episode} p_stop={p_stop_scalar:.3f} reward={rew:.2f} "
                      f"has_instr={env.has_active_instruction()}")

            # Friendly yield to event loop
            await aio.sleep(0)

    finally:
        stop_evt.set()
        try:
            env.close()
        except Exception:
            pass


def load_agent_from_ckpt(agent, ckpt_path: str):
    """Load parameters into agent from elements.Checkpoint file or directory."""
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(ckpt_path, keys=["agent"])
    print(f"[async_inference] Loaded checkpoint from: {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async planner–controller inference.")
    p.add_argument("--task", type=str, default="overcooked_l2_simple",
                   help="Task string, e.g., 'overcooked_l2_simple', 'crafter_norepeat'.")
    p.add_argument("--from_checkpoint", type=str, required=True,
                   help="Path to Dreamer checkpoint (dir or .npz/jsonl bundle)")
    p.add_argument("--logdir", type=str, default="~/logdir/async_infer_{time}",
                   help="Where to write minimal logs (created).")
    p.add_argument("--stop_threshold", type=float, default=0.65,
                   help="STOP head probability threshold for clearing instruction.")
    p.add_argument("--max_steps", type=int, default=0, help="Stop after N steps; 0 = unlimited.")

    # VLM options
    p.add_argument("--model_type", type=str, default="qwenvl", choices=["qwenvl", "phi3"],
                   help="Planner VLM family.")
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="HF model id for the VLM.")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Planner token budget for each instruction.")
    p.add_argument("--temperature", type=float, default=0.2, help="Planner sampling temperature.")
    p.add_argument("--proactive_check", action="store_true",
                   help="Ask VLM if new instruction is needed before generating.")

    return p.parse_args()


def main():
    args = parse_args()

    # Build config (leverages repo configs.yaml and sets agent.use_vlm True)
    cfg = build_config(
        logdir=args.logdir,
        task=args.task,
        extra_argv=[
            "--script", "eval_only",             # enable policy only
            "--jax.enable_policy", "True",
            "--run.from_checkpoint", args.from_checkpoint,
        ],
    )

    # Create agent and load checkpoint
    agent = drv3_main.make_agent(cfg, text_encoder=(None, None))
    load_agent_from_ckpt(agent, args.from_checkpoint)

    # Create env and wrap with instruction injector
    base_env = drv3_main.make_env(cfg, index=0)
    wrapper = ManualInstrWrapper(base_env, text_encoder=SentenceEmbedder(), stop_threshold=args.stop_threshold)

    # Planner (VLM)
    vlm = VLMInference(model_type=args.model_type, model_id=args.model_id, device=0, prompt_domain=args.task.split('_',1)[0] if '_' in args.task else 'generic')

    # Shared buffers and events
    frame_buf: Deque[Image.Image] = collections.deque(maxlen=8)
    need_instr_evt = aio.Event()
    stop_evt = aio.Event()

    # Simple setter that both sets wrapper instruction and prints it
    def set_instr(text: str):
        print(f"[planner] new instruction: {text}")
        wrapper.set_instruction(text)

    # Start loops
    max_steps = None if args.max_steps <= 0 else int(args.max_steps)
    tasks = [
        aio.create_task(controller_task(agent, wrapper, frame_buf, need_instr_evt, stop_evt,
                                        max_steps=max_steps, stop_threshold=args.stop_threshold)),
        aio.create_task(planner_task(vlm, frame_buf, set_instr, need_instr_evt, stop_evt,
                                     max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                     proactive_check=args.proactive_check)),
    ]

    # Wait for controller to finish then cancel planner
    async def _run():
        try:
            await tasks[0]  # controller
        finally:
            stop_evt.set()
            for t in tasks[1:]:
                t.cancel()
            await aio.gather(*tasks, return_exceptions=True)

    aio.run(_run())


if __name__ == "__main__":
    main()
