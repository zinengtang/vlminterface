#!/usr/bin/env python
"""
Stabilized version: Single step count and single run number per suite for consistent benchmarking.
Each suite uses the same steps and runs parameters for all configurations to ensure stability.
Extended with multi-agent testing for 2, 4, and 8 agents with decentralized and centralized communication.
FIXED: Separated VLM token count from instruction cadence. Cadence is fixed at 32, VLM tokens are variable.
"""

import os
import re
import time
import json
import argparse
import threading
import numpy as np
from functools import partial
from datetime import datetime
from typing import Dict, List, Any, Optional

import jax
from dreamerv3 import main as drv3_main
from dreamerv3.encoder import load_flax_text_encoder
from web.utils import build_config, ManualInstrWrapper

# ----------------- helpers -----------------
_addb = lambda t: jax.tree.map(lambda x: x[None], t)
_rmb  = lambda t: jax.tree.map(lambda x: np.asarray(x[0]), t)

def _slug(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s)).strip('_')

# ----------------- scale & VLM configs -----------------
SCALE_CONFIGS = {
    "small": {"name": "size50m","params": "~50M","config_overrides": {
        "agent.dyn.rssm.deter": 4096,"agent.dyn.rssm.hidden": 512,"agent.dyn.rssm.classes": 32,
        "agent.enc.simple.depth": 32,"agent.dec.simple.depth": 32,
        "agent.policy.units": 512,"agent.value.units": 512}},
    "medium": {"name": "size200m","params": "~200M","config_overrides": {
        "agent.dyn.rssm.deter": 8192,"agent.dyn.rssm.hidden": 1024,"agent.dyn.rssm.classes": 64,
        "agent.enc.simple.depth": 64,"agent.dec.simple.depth": 64,
        "agent.policy.units": 1024,"agent.value.units": 1024}},
    "large": {"name": "size400m","params": "~400M","config_overrides": {
        "agent.dyn.rssm.deter": 12288,"agent.dyn.rssm.hidden": 1536,"agent.dyn.rssm.classes": 96,
        "agent.enc.simple.depth": 96,"agent.dec.simple.depth": 96,
        "agent.policy.units": 1536,"agent.value.units": 1536}},
}

VLM_SCALE_CONFIGS = {
    "none": {"name": "no_vlm", "params": "0", "model_type": None, "model_id": None},
    "qwen-2b": {"name": "qwen2_5_vl_2b", "params": "~2B", "model_type": "qwenvl", "model_id": "Qwen/Qwen2.5-VL-2B-Instruct"},
    "qwen-3b": {"name": "qwen2_5_vl_3b", "params": "~3B", "model_type": "qwenvl", "model_id": "Qwen/Qwen2.5-VL-3B-Instruct"},
    "qwen-7b": {"name": "qwen2_5_vl_7b", "params": "~7B", "model_type": "qwenvl", "model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    "qwen-32b": {"name": "qwen2_5_vl_32b", "params": "~32B", "model_type": "qwenvl", "model_id": "Qwen/Qwen2.5-VL-32B-Instruct"},
    "qwen-72b": {"name": "qwen2_5_vl_72b", "params": "~72B", "model_type": "qwenvl", "model_id": "Qwen/Qwen2.5-VL-72B-Instruct"},
}
DEFAULT_BOUNDING = {"small": "qwen-2b", "medium": "qwen-7b", "large": "qwen-72b"}
ALT_BOUNDING_3_7_32 = {"small": "qwen-3b", "medium": "qwen-7b", "large": "qwen-32b"}

# Fixed cadence for instruction calls
FIXED_INSTRUCTION_CADENCE = 32

# ----------------- config apply -----------------
def _apply_scale_config(cfg, scale: str = "medium"):
    sc = SCALE_CONFIGS[scale]["config_overrides"]
    updates = {}
    for key, value in sc.items():
        parts = key.split('.'); cur = updates
        for p in parts[:-1]:
            if p not in cur: cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
    return cfg.update(**updates)

def _apply_vlm_config(cfg, vlm_scale: str = "none", vlm_tokens: int = 64):
    v = VLM_SCALE_CONFIGS[vlm_scale]; use_vlm = v["model_type"] is not None
    vlm_payload = None if not use_vlm else {
        "model_type": v["model_type"], "model_id": v["model_id"],
        "dtype": "bfloat16", "max_new_tokens": vlm_tokens, "temperature": 0.8}
    cfg = cfg.update(agent={"use_vlm": use_vlm})
    try: cfg = cfg.update(agent={"vlm": vlm_payload if use_vlm else None})
    except Exception: pass
    if use_vlm:
        os.environ["DREAMER_VLM_MODEL_TYPE"] = v["model_type"] or ""
        os.environ["DREAMER_VLM_MODEL_ID"] = v["model_id"] or ""
        os.environ["DREAMER_VLM_DTYPE"] = "bfloat16"
        os.environ["DREAMER_VLM_MAX_NEW_TOKENS"] = str(vlm_tokens)
        os.environ["DREAMER_VLM_TEMP"] = "0.8"
    else:
        for k in ("DREAMER_VLM_MODEL_TYPE","DREAMER_VLM_MODEL_ID","DREAMER_VLM_DTYPE","DREAMER_VLM_MAX_NEW_TOKENS","DREAMER_VLM_TEMP"):
            os.environ.pop(k, None)
    return cfg

def _apply_eval_settings(cfg, instruction_cadence=None, planning_mode=None,
                         num_agents=None, comm_mode=None, comm_frequency=None, comm_leader=None):
    """
    Apply evaluation settings to configuration.
    
    instruction_cadence: How often to call VLM (fixed at 32)
    comm_mode options:
    - 'peer_to_peer': decentralized, all agents communicate with each other
    - 'hub_spoke': centralized, one leader agent communicates with all others
    - 'none': no communication
    """
    ev = {}
    if instruction_cadence is not None: ev["instruction_cadence"] = int(instruction_cadence)
    if planning_mode is not None:
        ev["planning_mode"] = str(planning_mode); ev["async_vlm"] = bool(planning_mode == "online")
    if num_agents is not None:
        ev["multi_agent"] = ev.get("multi_agent", {}); ev["multi_agent"].update({"enabled": True, "num_agents": int(num_agents)})
        if comm_mode is not None or comm_frequency is not None or comm_leader is not None:
            ev["multi_agent"]["comm"] = ev["multi_agent"].get("comm", {})
            if comm_mode is not None: ev["multi_agent"]["comm"]["mode"] = str(comm_mode)
            if comm_frequency is not None: ev["multi_agent"]["comm"]["frequency"] = int(comm_frequency)
            if comm_leader is not None: ev["multi_agent"]["comm"]["leader"] = str(comm_leader)
    try: cfg = cfg.update(eval=ev)
    except Exception:
        if "instruction_cadence" in ev: os.environ["DREAMER_INSTR_CADENCE"] = str(ev["instruction_cadence"])
        if "planning_mode" in ev:
            os.environ["DREAMER_PLANNING_MODE"] = str(ev["planning_mode"])
            os.environ["DREAMER_ASYNC_VLM"] = "1" if ev.get("async_vlm") else "0"
        if "multi_agent" in ev:
            os.environ["DREAMER_MA_ENABLED"] = "1"
            os.environ["DREAMER_MA_NUM_AGENTS"] = str(ev["multi_agent"].get("num_agents", 1))
            comm = ev["multi_agent"].get("comm", {})
            if comm:
                os.environ["DREAMER_MA_COMM_MODE"] = str(comm.get("mode", "none"))
                os.environ["DREAMER_MA_COMM_FREQ"] = str(comm.get("frequency", 0))
                if "leader" in comm: os.environ["DREAMER_MA_COMM_LEADER"] = str(comm["leader"])
    return cfg

def _vlm_key_for_scale(scale: str, mapping: Dict[str, str]) -> str:
    vk = mapping.get(scale)
    if vk is None or vk not in VLM_SCALE_CONFIGS: raise ValueError(f"No VLM bound for scale '{scale}'")
    return vk

# ----------------- encoder -----------------
_ENCODER = None
def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = load_flax_text_encoder('nreimers/MiniLM-L6-H384-uncased')
    return _ENCODER

# ----------------- VLM stub -----------------
class VLMStub:
    def __init__(self, ttft_ms=1.0, tokens=64, tps=100.0, dump_first_n=3):
        self.ttft_ms=float(ttft_ms); self.tokens=int(tokens); self.tps=float(tps)
        self.dump_first_n=int(dump_first_n); self.calls=0; self.latencies_ms=[]; self.samples=[]
    def generate(self, prompt: str, image: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.ttft_ms > 0: time.sleep(self.ttft_ms / 1000.0)
        if self.tokens > 0 and self.tps > 0: time.sleep(self.tokens / self.tps)
        plan = f"[plan #{self.calls}] step-left; pickup-onion; go-stove"
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.latencies_ms.append(dt_ms)
        if self.calls < self.dump_first_n: self.samples.append({"prompt": prompt, "plan": plan, "latency_ms": dt_ms})
        self.calls += 1
        return {"text": plan, "latency_ms": dt_ms}

# ----------------- env/agent -----------------
def make(cfg, shared_encoder: bool = True):
    make_env = partial(drv3_main.make_env, cfg); make_agent = partial(drv3_main.make_agent, cfg)
    if shared_encoder: lang_model, lang_params = get_encoder()
    else: lang_model, lang_params = load_flax_text_encoder('nreimers/MiniLM-L6-H384-uncased')
    env = ManualInstrWrapper(make_env(0)); agent = make_agent(text_encoder=(lang_model, lang_params))
    carry = agent.init_policy(1); return env, agent, carry

def step_once(env, agent, carry, obs):
    obs_no_logs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    carry, actsb, _ = agent.policy(carry, _addb(obs_no_logs), mode='eval')
    jax.block_until_ready(actsb)
    acts = _rmb(actsb); acts['reset'] = bool(obs.get('is_last', False))
    if 'action' in acts and hasattr(acts['action'], 'astype'): acts['action'] = acts['action'].astype(np.int32)
    nxt = env.env_step(acts); return carry, nxt

# ----------------- benchmark -----------------
def run_benchmark(env, agent, carry, steps: int, *, use_vlm: bool, instruction_cadence: Optional[int],
                  planning_mode: Optional[str], stub: Optional[VLMStub]=None, warmup_steps: int = 10) -> Dict[str, Any]:
    """
    Run a single benchmark iteration with specified steps.
    Added warmup_steps parameter for stabilization.
    instruction_cadence: How often to call VLM (fixed at 32 by default)
    """
    obs = env.env_step({"action": np.array([0, 0], np.int32), "reset": True})
    
    # Warmup phase for stabilization
    for _ in range(warmup_steps): 
        carry, obs = step_once(env, agent, carry, obs)
    
    # Actual benchmark
    dt0 = time.perf_counter(); threads = []
    for t in range(steps):
        if use_vlm and stub is not None and instruction_cadence and instruction_cadence > 0 and (t % instruction_cadence == 0):
            prompt = f"[obs@t={t}] goal: serve onion soup; status: {obs.get('status','N/A')}"
            if planning_mode == "online":
                import threading as _th
                th = _th.Thread(target=stub.generate, args=(prompt, None), daemon=True); th.start(); threads.append(th)
            else:
                _ = stub.generate(prompt, None)
        carry, obs = step_once(env, agent, carry, obs)
    for th in threads: th.join()
    dt = time.perf_counter() - dt0; throughput = steps / dt if dt > 0 else 0.0
    info = {"total_time": float(dt), "throughput": float(throughput)}
    if stub is not None and use_vlm:
        lat = np.array(stub.latencies_ms) if stub.latencies_ms else np.array([])
        info["vlm_stub"] = {
            "calls": int(stub.calls),
            "latency_ms_mean": float(lat.mean()) if lat.size else 0.0,
            "latency_ms_p95": float(np.percentile(lat, 95)) if lat.size else 0.0,
            "samples": stub.samples,
        }
    return info

# ----------------- profile with stabilization -----------------
def profile_configuration_stable(task: str, logdir: str, steps: int, scale: str, vlm_key: str, runs: int,
                                 instruction_cadence: int = FIXED_INSTRUCTION_CADENCE, planning_mode: str = None, 
                                 vlm_tokens: int = 64, num_agents: int = None,
                                 comm_mode: str = None, comm_frequency: int = None, comm_leader: str = None, tag: str = "",
                                 warmup_steps: int = 20, *, stub_vlm: bool=False, stub_ttft_ms: float = 30.0, 
                                 stub_tokens: int = None, stub_tps: float = 40.0) -> Dict[str, Any]:
    """
    Stable profiling with consistent steps and runs across all configurations.
    Added warmup_steps for better stabilization.
    instruction_cadence: How often to call VLM (fixed at 32)
    vlm_tokens: How many tokens VLM generates (variable: 8, 32, 128)
    """
    if stub_tokens is None:
        stub_tokens = vlm_tokens  # Use vlm_tokens for stub if not specified
    
    scale_info = SCALE_CONFIGS[scale]; vlm_info = VLM_SCALE_CONFIGS[vlm_key]; use_vlm = (vlm_key != "none")
    results = {
        "scale": scale, "scale_name": scale_info["name"], "ctrl_params": scale_info["params"],
        "vlm_key": vlm_key, "vlm_name": vlm_info["name"], "vlm_params": vlm_info["params"],
        "eval": {"instruction_cadence": instruction_cadence, "planning_mode": planning_mode, "vlm_tokens": vlm_tokens,
                 "num_agents": num_agents, "comm_mode": comm_mode, "comm_frequency": comm_frequency, "comm_leader": comm_leader},
        "runs": [], "tag": tag or "", 
        "benchmark_params": {"steps": steps, "runs": runs, "warmup_steps": warmup_steps}
    }
    print(f" Controller: {scale_info['name']} ({scale_info['params']}) | VLM: {vlm_info['name']} ({vlm_info['params']}) | "
          f"Eval: cadence={instruction_cadence}, vlm_tokens={vlm_tokens}, planning={planning_mode}, agents={num_agents}, comm={comm_mode} | "
          f"Steps={steps}, Runs={runs}, Warmup={warmup_steps}")
    
    for run_idx in range(runs):
        print(f"  Run {run_idx + 1}/{runs}...", end=" ", flush=True)
        try:
            cfg = build_config(logdir=logdir, task=task)
            cfg = _apply_scale_config(cfg, scale)
            cfg = _apply_vlm_config(cfg, vlm_key, vlm_tokens)
            cfg = _apply_eval_settings(cfg, instruction_cadence, planning_mode, num_agents, comm_mode, comm_frequency, comm_leader)
            env, agent, carry = make(cfg, shared_encoder=True)
            stub = VLMStub(ttft_ms=stub_ttft_ms, tokens=stub_tokens, tps=stub_tps)
            info = run_benchmark(env, agent, carry, steps, use_vlm=use_vlm, instruction_cadence=instruction_cadence, 
                               planning_mode=planning_mode, stub=stub, warmup_steps=warmup_steps)
            run_result = {"run_idx": run_idx, "total_time": float(info["total_time"]), 
                         "throughput": float(info["throughput"]), "steps": int(steps), "status": "success"}
            if "vlm_stub" in info:
                run_result["vlm_stub"] = info["vlm_stub"]
            print(f"✓ {run_result['throughput']:.2f} steps/s")
        except Exception as e:
            print(f"✗ Error: {e}"); run_result = {"run_idx": run_idx, "error": str(e), "status": "failed"}
        results["runs"].append(run_result)
    
    # Calculate statistics
    succ = [r for r in results["runs"] if r.get("status") == "success"]
    if succ:
        thr = [r["throughput"] for r in succ]
        results["stats"] = {"mean_throughput": float(np.mean(thr)), "std_throughput": float(np.std(thr)),
                            "min_throughput": float(np.min(thr)), "max_throughput": float(np.max(thr)),
                            "median_throughput": float(np.median(thr)),
                            "successful_runs": len(succ), "failed_runs": len(results["runs"]) - len(succ)}
        # Add coefficient of variation for stability assessment
        if results["stats"]["mean_throughput"] > 0:
            results["stats"]["cv"] = float(np.std(thr) / np.mean(thr))
    else:
        results["stats"] = {"mean_throughput": 0.0, "std_throughput": 0.0, "min_throughput": 0.0, 
                            "max_throughput": 0.0, "median_throughput": 0.0,
                            "successful_runs": 0, "failed_runs": len(results["runs"])}
    return results

# ----------------- stabilized suites -----------------
def suite_scales_vlm_toggle_stable(task, logdir, steps, runs, scales, binding_map, warmup_steps, **stubkw):
    """Suite with consistent steps and runs across all configurations."""
    suite = {"name": "scales_vlm_toggle", "suite_params": {"steps": steps, "runs": runs, "warmup_steps": warmup_steps}, "variants": []}
    for scale in scales:
        vlm_key = _vlm_key_for_scale(scale, binding_map)
        suite["variants"].append(profile_configuration_stable(task, logdir, steps, scale, vlm_key, runs,
                               instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode="online", 
                               vlm_tokens=64, tag="with_vlm", warmup_steps=warmup_steps, **stubkw))
        suite["variants"].append(profile_configuration_stable(task, logdir, steps, scale, "none", runs,
                               instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode="online", 
                               vlm_tokens=64, tag="no_vlm", warmup_steps=warmup_steps, **stubkw))
    return suite

def suite_vlm_token_numbers_all_scales_stable(task, logdir, steps, runs, vlm_token_numbers, scales, binding_map, warmup_steps, **stubkw):
    """Suite testing different VLM token counts across all scales with fixed cadence."""
    suite = {"name": "vlm_token_numbers_all_scales", "suite_params": {"steps": steps, "runs": runs, "warmup_steps": warmup_steps}, "variants": []}
    
    for scale in scales:
        vlm_key = _vlm_key_for_scale(scale, binding_map)
        for vlm_tokens in vlm_token_numbers:
            suite["variants"].append(profile_configuration_stable(task, logdir, steps, scale, vlm_key, runs,
                                   instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode="online", 
                                   vlm_tokens=vlm_tokens, tag=f"{scale}_vlm{vlm_tokens}tok_with_vlm",
                                   warmup_steps=warmup_steps, stub_tokens=vlm_tokens, **stubkw))
    return suite

def suite_planning_modes_all_scales_stable(task, logdir, steps, runs, scales, binding_map, warmup_steps, **stubkw):
    """Suite with consistent steps and runs across planning mode and scale configurations."""
    suite = {"name": "planning_modes_all_scales", "suite_params": {"steps": steps, "runs": runs, "warmup_steps": warmup_steps}, "variants": []}
    
    for scale in scales:
        vlm_key = _vlm_key_for_scale(scale, binding_map)
        for mode in ("online", "offline"):
            suite["variants"].append(profile_configuration_stable(task, logdir, steps, scale, vlm_key, runs,
                                   instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode=mode, 
                                   vlm_tokens=64, tag=f"{scale}_{mode}_planning_with_vlm",
                                   warmup_steps=warmup_steps, **stubkw))
    return suite

def suite_multi_agent_stable(task, logdir, steps, runs, binding_map, agent_counts, warmup_steps, **stubkw):
    """
    Multi-agent suite testing communication patterns (decentralized vs centralized).
    
    - Decentralized: All agents communicate with each other (peer-to-peer)
    - Centralized: One leader agent communicates with all others (hub-and-spoke)
    
    Only uses 'large' scale for multi-agent testing to ensure sufficient capacity.
    """
    suite = {"name": "multi_agent_communication_patterns", 
             "suite_params": {"steps": steps, "runs": runs, "warmup_steps": warmup_steps, "agent_counts": agent_counts}, 
             "variants": []}
    
    scale = "large"  # Only use large scale for multi-agent testing
    vlm_key = _vlm_key_for_scale(scale, binding_map)
    
    for num_agents in agent_counts:
        # Decentralized: all agents communicate with each other (peer-to-peer)
        suite["variants"].append(profile_configuration_stable(
            task, logdir, steps, scale, vlm_key, runs,
            instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode="online", vlm_tokens=64,
            num_agents=num_agents, comm_mode="peer_to_peer", comm_frequency=1, comm_leader=None, 
            tag=f"ma{num_agents}_decentralized", warmup_steps=warmup_steps, **stubkw))
        
        # Centralized: one leader agent communicates with all others (hub-and-spoke)
        suite["variants"].append(profile_configuration_stable(
            task, logdir, steps, scale, vlm_key, runs,
            instruction_cadence=FIXED_INSTRUCTION_CADENCE, planning_mode="online", vlm_tokens=64,
            num_agents=num_agents, comm_mode="hub_spoke", comm_frequency=1, comm_leader="agent_0", 
            tag=f"ma{num_agents}_centralized", warmup_steps=warmup_steps, **stubkw))
    
    return suite

# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser(description="Stabilized profiling for DreamerV3 with consistent parameters per suite.")
    parser.add_argument("--task", type=str, default="overcooked")
    parser.add_argument("--logdir", type=str, default="/tmp/profile")
    
    # Single step count and run number for all suites
    parser.add_argument("--steps", type=int, default=200, help="Number of steps per benchmark run (applies to all suites)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration (applies to all suites)")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps before benchmark (for stabilization)")
    
    parser.add_argument("--output", type=str, default="profile_stabilized.json", help="Combined JSON file.")
    parser.add_argument("--outdir", type=str, default=None, help="Directory for per-suite JSON files.")
    parser.add_argument("--scales", nargs="+", default=["small", "medium", "large"], choices=list(SCALE_CONFIGS.keys()))
    parser.add_argument("--vlm-token-numbers", nargs="+", type=int, default=[8, 32, 128], 
                       help="List of VLM token numbers to test (how many tokens VLM generates)")
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[2, 4, 8], 
                       help="List of agent counts for multi-agent suite (e.g., 2 4 8) - uses large scale only")
    parser.add_argument("--binding", choices=["qwen_2_7_72", "qwen_3_7_32"], default="qwen_2_7_72")
    parser.add_argument("--suite", choices=["base","scales_vlm_toggle","vlm_token_numbers_all_scales","planning_modes_all_scales","multi_agent","all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Quick test mode (100 steps, 2 runs, 10 warmup)")
    parser.add_argument("--stub-vlm", action="store_true", help="Force VLM stub calls")
    parser.add_argument("--vlm-ttft-ms", type=float, default=30.0)
    parser.add_argument("--vlm-tps", type=float, default=40.0)
    args = parser.parse_args()

    if args.quick:
        args.steps = 100; args.runs = 2; args.warmup_steps = 10
        print("Quick test mode enabled (100 steps, 2 runs, 10 warmup)")

    binding_map = DEFAULT_BOUNDING if args.binding == "qwen_2_7_72" else ALT_BOUNDING_3_7_32
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = args.outdir or os.path.dirname(args.output) or "."
    os.makedirs(outdir, exist_ok=True)

    print("=" * 80)
    print("DreamerV3 Stabilized Profiling (consistent steps/runs per suite)")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Steps per run (ALL suites): {args.steps}")
    print(f"Runs per config (ALL suites): {args.runs}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Scales: {args.scales}")
    print(f"VLM token numbers (sweep): {args.vlm_token_numbers}")
    print(f"Fixed instruction cadence: {FIXED_INSTRUCTION_CADENCE}")
    print(f"Agent counts (MA with large scale): {args.agent_counts}")
    print(f"Binding: {args.binding}")
    print(f"Suite: {args.suite}")
    print(f"Stub VLM: {args.stub_vlm} | TTFT(ms)={args.vlm_ttft_ms}, tps={args.vlm_tps}")
    print(f"Output dir: {outdir}")
    print()

    print("Loading text encoder..."); _ = get_encoder(); print("Devices:", jax.devices()); print()

    all_results = {
        "metadata": {
            "task": args.task, 
            "global_steps": args.steps, 
            "global_runs": args.runs,
            "global_warmup_steps": args.warmup_steps,
            "fixed_instruction_cadence": FIXED_INSTRUCTION_CADENCE,
            "timestamp": datetime.now().isoformat(), 
            "jax_devices": str(jax.devices()),
            "binding": args.binding, 
            "stub_vlm": args.stub_vlm,
            "stub_params": {"ttft_ms": args.vlm_ttft_ms, "tps": args.vlm_tps},
        },
        "suites": [],
    }

    def add_and_print_suite(suite_obj):
        all_results["suites"].append(suite_obj)
        print("\n" + "=" * 80)
        print(f"SUITE: {suite_obj['name']} | Steps={suite_obj['suite_params']['steps']} | "
            f"Runs={suite_obj['suite_params']['runs']} | Warmup={suite_obj['suite_params']['warmup_steps']}")
        if 'agent_counts' in suite_obj['suite_params']:
            print(f"Agent counts tested: {suite_obj['suite_params']['agent_counts']}")
        print("=" * 80)

        header = (f"{'Tag':<28} {'Scale':<8} {'Ctrl-Params':<10} {'VLM':<18} "
                f"{'VLM-Params':<10} {'Cadence':<8} {'VLM-Tokens':<10} {'Plan':<8} {'Agents':<6} "
                f"{'Comm Mode':<14} {'Throughput (mean±std)':<24} {'CV':<8}")
        print(header)
        print("-" * len(header))

        def _s(x, default="-"):
            # stringify with default for None
            return default if x is None else str(x)

        for v in suite_obj.get("variants", []):
            stats = v.get('stats', {}) or {}
            thr = (_s(f"{stats.get('mean_throughput', 0):.2f}") + "±" +
                _s(f"{stats.get('std_throughput', 0):.2f}")) if stats.get("successful_runs", 0) > 0 else "Failed"
            cv = f"{stats.get('cv', 0):.3f}" if stats.get('cv') is not None else "N/A"

            comm_mode_disp = (v.get('eval', {}).get('comm_mode', 'none')) or 'none'
            if comm_mode_disp == 'peer_to_peer':
                comm_mode_disp = 'decentralized'
            elif comm_mode_disp == 'hub_spoke':
                comm_mode_disp = 'centralized'

            # Safely fetch & stringify everything before alignment
            tag         = _s(v.get('tag', ''))
            scale       = _s(v.get('scale'))
            ctrlparams  = _s(v.get('ctrl_params'))
            vlm_disp    = _s(v.get('vlm_name'))
            vlm_params  = _s(v.get('vlm_params'))
            cadence     = _s(v.get('eval', {}).get('instruction_cadence'))
            vlm_tokens  = _s(v.get('eval', {}).get('vlm_tokens'))
            plan        = _s(v.get('eval', {}).get('planning_mode'))
            agents      = _s(v.get('eval', {}).get('num_agents'))
            comm_disp   = _s(comm_mode_disp)
            thr_disp    = _s(thr)
            cv_disp     = _s(cv)

            print(f"{tag:<28} {scale:<8} {ctrlparams:<10} {vlm_disp:<18} "
                f"{vlm_params:<10} {cadence:<8} {vlm_tokens:<10} {plan:<8} {agents:<6} "
                f"{comm_disp:<14} {thr_disp:<24} {cv_disp:<8}")

        # Save suite to file
        sfname = f"suite_{_slug(args.task)}_{suite_obj['name']}_{timestamp}.json"
        spath = os.path.join(outdir, sfname)
        with open(spath, 'w') as f:
            json.dump(suite_obj, f, indent=2)
        print(f"Saved suite to {spath}")


    stubkw = dict(stub_vlm=args.stub_vlm, stub_ttft_ms=args.vlm_ttft_ms, stub_tps=args.vlm_tps)

    # Run suites with consistent parameters
    if args.suite == "base":
        large_vlm = _vlm_key_for_scale("large", binding_map)
        base_suite = {"name": "base_single_config", "suite_params": {"steps": args.steps, "runs": args.runs, "warmup_steps": args.warmup_steps}, "variants": []}
        base_suite["variants"].append(profile_configuration_stable(args.task, args.logdir, args.steps, "large", large_vlm, 
                                                                  args.runs, instruction_cadence=FIXED_INSTRUCTION_CADENCE, 
                                                                  vlm_tokens=64, warmup_steps=args.warmup_steps, **stubkw))
        add_and_print_suite(base_suite)

    if args.suite in ("scales_vlm_toggle", "all"):
        s = suite_scales_vlm_toggle_stable(args.task, args.logdir, args.steps, args.runs, args.scales, 
                                          binding_map, args.warmup_steps, **stubkw)
        add_and_print_suite(s)

    if args.suite in ("vlm_token_numbers_all_scales", "all"):
        s = suite_vlm_token_numbers_all_scales_stable(args.task, args.logdir, args.steps, args.runs, args.vlm_token_numbers, 
                                                     args.scales, binding_map, args.warmup_steps, **stubkw)
        add_and_print_suite(s)

    if args.suite in ("planning_modes_all_scales", "all"):
        s = suite_planning_modes_all_scales_stable(args.task, args.logdir, args.steps, args.runs, 
                                                  args.scales, binding_map, args.warmup_steps, **stubkw)
        add_and_print_suite(s)

    if args.suite in ("multi_agent", "all"):
        s = suite_multi_agent_stable(args.task, args.logdir, args.steps, args.runs, 
                                    binding_map, args.agent_counts, args.warmup_steps, **stubkw)
        add_and_print_suite(s)

    # Save combined results
    with open(args.output, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {args.output}")
    
    # Print stability summary
    print("\n" + "=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)
    for suite in all_results["suites"]:
        print(f"\nSuite: {suite['name']}")
        if 'agent_counts' in suite.get('suite_params', {}):
            print(f"  Agent counts tested: {suite['suite_params']['agent_counts']}")
        cvs = [v['stats'].get('cv', 0) for v in suite['variants'] if v.get('stats', {}).get('cv') is not None]
        if cvs:
            print(f"  Average CV: {np.mean(cvs):.3f} (lower is more stable)")
            print(f"  Max CV: {np.max(cvs):.3f}")
            print(f"  Min CV: {np.min(cvs):.3f}")
            
            # Per agent count statistics if this is the multi-agent suite
            if suite['name'] == "multi_agent_communication_patterns" and 'agent_counts' in suite.get('suite_params', {}):
                for agent_count in suite['suite_params']['agent_counts']:
                    agent_cvs = [v['stats'].get('cv', 0) for v in suite['variants'] 
                                if v.get('stats', {}).get('cv') is not None 
                                and f"ma{agent_count}_" in v.get('tag', '')]
                    if agent_cvs:
                        print(f"    {agent_count} agents - Avg CV: {np.mean(agent_cvs):.3f}, "
                              f"Max: {np.max(agent_cvs):.3f}, Min: {np.min(agent_cvs):.3f}")


if __name__ == "__main__":
    main()