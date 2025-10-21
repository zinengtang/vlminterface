# DECOUPLING PLANNING AND CONTROL FORINSTRUCTABLE AGENTS
## Overview

Our approach combines a low-level agent with a high-level vision-language planner in an asynchronous hierarchical architecture. The controller executes primitive actions at every environment step while maintaining a world model for planning, while the VLM planner observes recent visual frames and generates natural language instructions to guide task completion. A learned completion signal allows the controller to autonomously request new instructions when subtasks finish, creating a feedback loop where the planner adapts its instruction granularity to the controller's capabilities. Unlike prior work that runs planners at fixed intervals or requires explicit task decomposition, our design enables efficient execution of both short and long-horizon tasks by allowing the controller to regulate planning frequency based on task progress.

---

## Main Script

- `async_inference.py` — main async inference loop (controller + planner)
- `dreamerv3/main.py` — main training loop, etc.  

---

## Quick Start

You can run with **Docker** (recommended for consistent GPU/JAX stacks) or **Conda + pip**.

### Prerequisites

- **GPU**
- **TPU** 
- **CPU**

### Option A: Docker (Recommended)

Build & run:
```bash
docker build -t async-vlm .
# Allow GPU access:
docker run --gpus all -it --rm \
  -v $PWD:/app \
  -e CUDA_VISIBLE_DEVICES=0 \
  async-vlm
```

### Option B: Conda + pip (Need to install headless packages)

Create environment:
```bash
conda create -y -n async-vlm python=3.11
conda activate async-vlm
pip install --upgrade pip
```

Install dependencies:
```bash
pip install -e .
```

### Running
If you are using docker, any script will start with
```
docker run \
 --network=host \ \\give access to network
 -it --rm \
 --gpus '"device=3,4"' \
 -p 8888:8888 \ \\(optional, forward port on some environments API)
 --name vlminterface_run5 \
 -v ~/logdir/docker:/root/logdir \ \\map your logs/checkpoints path to the path on the right
 -v ~/.cache:/root/.cache \ \\map your cache path to the path on the right
 -v "$PWD":/app   -w /app \ \\the scripts you use will be your local scripts, skipping the need to overwrite to container
 vlminterface:latest
```

Training Loop:
```bash
bash scripts/[env_you_want_to_train_on].sh
```

Inference Loop:
```bash
python async_inference.py \
  --task overcooked_l2_simple \
  --from_checkpoint /path/to/checkpoint \
  --model_type qwenvl \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 64 \
  --temperature 0.4 \
  --stop_threshold 0.65
```

## Command Line Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| --task | str | overcooked_l2_simple | Env/task name; used by build_config |
| --from_checkpoint | str | required | Dreamer checkpoint dir or bundle file |
| --logdir | str | ~/logdir/async_infer_{time} | Minimal logs/checkpoints location |
| --stop_threshold | float | 0.65 | Threshold to clear active instruction via wrapper |
| --max_steps | int | 0 | Stop after N steps; 0 = unlimited |
| --model_type | str | qwenvl | Planner family: qwenvl, phi3 |
| --model_id | str | Qwen/Qwen2.5-VL-7B-Instruct | HF model id for the VLM |
| --max_new_tokens | int | 64 | Token budget per instruction |
| --temperature | float | 0.2 | Sampling temperature for VLM |
| --proactive_check | flag | off | Ask VLM if new instruction is needed before generating |

## Code Structure
Instructable-Agents-main/

├── dreamerv3/                # DreamerV3-based controller + encoders + configs

├── embodied/                 # Minimal RL framework (envs, JAX utils, wrappers)

├── inference/                # Asynchronous planner–controller inference scripts

├── prompts/                  # Long-horizon prompt snippets for VLM planning

├── scripts/                  # One-liners for common tasks (envs, profiling)

├── app.py                    # Flask demo server (controller + manual instr)

├── baselines.yaml            # Optional preset baselines

├── Dockerfile                # Reproducible JAX/Flax CUDA container
├── entrypoint.sh             # Container entrypoint
├── requirements.txt          # Python deps (JAX/Flax, envs, transformers, etc.)
└── setup.py                  # Editable install

## Troublesshooting & Tips (High-Level)

#### Checkpoints
- Pass the path with --from_checkpoint for loading from trained checkpoint.
- Loading is done via elements.Checkpoint and keys=["agent"].
#### Environment Assumptions
- The env observation dict should include (optionally) an image key (HxWxC, uint8 or [0,1] float).
  This is used only for planner context frames.\
#### Tips & Troubleshooting
- **CUDA/JAX:** If JAX can't find CUDA, ensure your driver is new enough and you installed the matching jax[cuda12_pip] wheel.
- **VLM auth:** Some HF models require login or acceptance; use huggingface-cli login.
- **Throughput:** For slow VLM planners, lower --max_new_tokens, increase --temperature, or reduce frame history.
- **No frames:** If your env doesn't provide image, planner will skip until frames are seen (you can modify to use no frames).


## License

This code is provided for research purposes. Check the licenses of any third-party models (e.g., Qwen) before use.
