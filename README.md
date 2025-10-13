# Async Planner–Controller Inference (DreamerV3 + VLM)

Asynchronous inference loop that pairs a **DreamerV3-based controller** with a **vision-language planner (VLM)**.  
The controller runs every env step; the planner runs in the background and injects short instructions when the controller’s **STOP** head indicates completion (or when no instruction is active).

- Controller/Env built via `dreamerv3.main.make_agent` / `dreamerv3.main.make_env`
- Planner via `VLMInference` (e.g., Qwen2.5-VL)
- Instruction injection via `web.utils.ManualInstrWrapper`
- STOP head trained with CE at action-sequence end; STOP decisions are also fed back to the RSSM as a shifted `stop_prev` channel (internal)

---

## Contents

- `async_inference.py` — main async inference loop (controller + planner)
- `dreamerv3/main` — main training loop, etc.  

---

## Quick Start

You can run with **Docker** (recommended for consistent GPU/JAX stacks) or **Conda + pip**.

### Prerequisites

- **GPU (recommended):** NVIDIA GPU + recent driver
- **CPU-only:** supported but slower for Dreamer
- **Hugging Face:** optional login if your chosen VLM requires auth  
  ```bash
  huggingface-cli login   # if needed for gated models
Option A: Docker (GPU)
Create a Dockerfile (or copy into your repo):

dockerfile
Copy code
# Dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Use Python 3 as 'python'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Workdir
WORKDIR /app
COPY . /app

# Python deps (minimal; adjust if you keep a requirements.txt)
# JAX CUDA wheels (CUDA 12) are hosted on Google Storage.
RUN pip install --upgrade pip && \
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install flax==0.8.* numpy==1.* pillow==10.* einops==0.7.* \
               transformers==4.* accelerate==0.* \
               dm-env==1.* gymnasium==0.* \
               tqdm pyyaml && \
    pip install -e .

# Optional: cache HF models inside container
ENV HF_HOME=/root/.cache/huggingface

CMD ["/bin/bash"]
Build & run:

docker build -t async-vlm .
# Allow GPU access:
docker run --gpus all -it --rm \
  -v $PWD:/app \
  -e CUDA_VISIBLE_DEVICES=0 \
  async-vlm
Inside the container, run async inference:


python async_inference.py \
  --task overcooked_l2_simple \
  --from_checkpoint /app/checkpoints/agent_ckpt \
  --model_type qwenvl \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 64 \
  --temperature 0.4 \
  --stop_threshold 0.65
Option B: Conda + pip (GPU or CPU)
Create environment:

conda create -y -n async-vlm python=3.11
conda activate async-vlm
pip install --upgrade pip
Install dependencies:

GPU (CUDA 12):

bash
Copy code
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
CPU-only:


pip install jax  # CPU wheel
Core libraries:

pip install flax==0.8.* numpy==1.* pillow==10.* einops==0.7.* \
            transformers==4.* accelerate==0.* \
            dm-env==1.* gymnasium==0.* \
            tqdm pyyaml
This repo (editable mode recommended during dev):


pip install -e .
Run:

CUDA_VISIBLE_DEVICES=0 \
python async_inference.py \
  --task overcooked_l2_simple \
  --from_checkpoint /path/to/checkpoint \
  --model_type qwenvl \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 64 \
  --temperature 0.4 \
  --stop_threshold 0.65
Command Line Arguments
Flag	Type	Default	Description
--task	str	overcooked_l2_simple	Env/task name; used by build_config
--from_checkpoint	str	required	Dreamer checkpoint dir or bundle file
--logdir	str	~/logdir/async_infer_{time}	Minimal logs/checkpoints location
--stop_threshold	float	0.65	Threshold to clear active instruction via wrapper
--max_steps	int	0	Stop after N steps; 0 = unlimited
--model_type	str	qwenvl	Planner family: qwenvl, phi3
--model_id	str	Qwen/Qwen2.5-VL-7B-Instruct	HF model id for the VLM
--max_new_tokens	int	64	Token budget per instruction
--temperature	float	0.2	Sampling temperature for VLM
--proactive_check	flag	off	Ask VLM if new instruction is needed before generating

How It Works (High-Level)
Controller loop (agent.policy(..., return_stop_token=True)):

Produces action and p_stop each step.

When p_stop ≥ stop_threshold, the wrapper clears the current instruction.

If no active instruction, the controller signals the planner task.

Planner loop (VLMInference):

Waits for a signal, reads the latest frames, generates a short instruction, and injects it via ManualInstrWrapper.

STOP head design:

Trained via cross-entropy only at the end of action sequences.

No mixing/gating of instruction embeddings; the policy/value/dream directly consume the instruction embedding (or a learned null placeholder when inactive).

The STOP predictions are fed back as a shifted binary channel (stop_prev) into the RSSM’s action embedding path to inform future dynamics.

Checkpoints
Place your trained Dreamer agent checkpoint where accessible (dir or .npz/.jsonl bundle).

Pass the path with --from_checkpoint.

Loading is done via elements.Checkpoint and keys=["agent"].

Environment Assumptions
The env observation dict should include (optionally) an image key (HxWxC, uint8 or [0,1] float).
This is used only for planner context frames.

ManualInstrWrapper injects/replaces instruction tokens in the env’s fields.

If your env returns Gym-style tuples (obs, reward, done, info), the script handles it.

Tips & Troubleshooting
Event loop errors: We create asyncio tasks inside the running loop; the script should not raise “no running event loop”.

Relative import: vlm_infer is imported with a fallback so you can run python async_inference.py directly.

CUDA/JAX: If JAX can’t find CUDA, ensure your driver is new enough and you installed the matching jax[cuda12_pip] wheel.

VLM auth: Some HF models require login or acceptance; use huggingface-cli login.

Throughput: For slow planners, lower --max_new_tokens, increase --temperature, or reduce frame history.

No frames: If your env doesn’t provide image, planner will skip until frames are seen (you can modify to use other obs).

Minimal Requirements (summary)
If you prefer a requirements.txt, start with:

makefile
Copy code
flax==0.8.*
numpy==1.*
pillow==10.*
einops==0.7.*
transformers==4.*
accelerate==0.*
dm-env==1.*
gymnasium==0.*
tqdm
pyyaml
# JAX wheels vary by platform — install separately:
# CPU: pip install jax
# GPU (CUDA 12): pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
Example

# GPU 0, Overcooked L2, Qwen2.5-VL, 64 token cap
CUDA_VISIBLE_DEVICES=0 \
python async_inference.py \
  --task overcooked_l2_simple \
  --from_checkpoint ./checkpoints/overcooked_agent \
  --model_type qwenvl \
  --model_id Qwen/Qwen2.5-VL-7B-Instruct \
  --max_new_tokens 64 \
  --temperature 0.4 \
  --stop_threshold 0.65 \
  --max_steps 10000
You’ll see controller logs like:

[controller] step=120  ep=0  p_stop=0.71  reward=0.50  has_instr=False
[planner] new instruction: chop onion and place on plate near stove
License
This code is provided for research purposes. Check the licenses of any third-party models (e.g., Qwen) before use.