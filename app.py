import os
os.environ["XLA_FLAGS"]="--xla_gpu_enable_cudnn_fmha=false --xla_gpu_deterministic_ops=true"
os.environ["TF_CUDNN_USE_AUTOTUNE"]="0"
os.environ["TF_CUDNN_DETERMINISTIC"]="1"
os.environ["XLA_FLAGS"]="--xla_gpu_autotune_level=0"

# flask_app.py
import threading, time, io, os, pathlib
import random
import elements
from functools import partial
from PIL import Image
import numpy as np
import jax
from flask import Flask, Response, request, redirect, url_for, render_template_string

from dreamerv3 import main as drv3_main
from web.utils import build_config, ManualInstrWrapper, SentenceEmbedder

# ── helpers to match Dreamer’s batching -------------------------------
_add_batch    = lambda t: jax.tree.map(lambda x: x[None], t)
_remove_batch = lambda t: jax.tree.map(lambda x: np.asarray(x[0]), t)

ckpt_path = "/root/logdir/dreamer/overcooked_oneagent_v1_multienv_multiplayer"

# ── build config, env, agent exactly as before ------------------------
cfg            = build_config(logdir=ckpt_path, task="overcooked")

make_env       = partial(drv3_main.make_env, cfg)
make_agent     = partial(drv3_main.make_agent, cfg)

from dreamerv3.encoder import load_flax_text_encoder
from transformers import AutoTokenizer
lang_model, lang_params = load_flax_text_encoder('nreimers/MiniLM-L6-H384-uncased')  # or your Flax choice
tokenizer = AutoTokenizer.from_pretrained("nreimers/MiniLM-L6-H384-uncased") 

env            = ManualInstrWrapper(make_env(0))
agent          = make_agent(text_encoder=(lang_model, lang_params))                    # <-- if you really want the agent
  
cp = elements.Checkpoint(f'{ckpt_path}/ckpt')
cp.agent = agent
cp.load_or_save()
carry          = agent.init_policy(1)

# ── global simulation state ------------------------------------------
paused         = True
fps            = 1
user_text      = "Grab tomato"
obs            = env.env_step({"action": np.array([0, 0], np.int32), "reset": True})

step_id  = 0               # NEW: tracks current environment step
instr_log = []             # NEW: list of {'step': int, 'text': str, 'ts': str}

def step_loop():
    global carry, obs, paused, fps, step_id, instr_log
    while True:
        if not paused:
            clean = {k: v for k, v in obs.items() if not k.startswith("log/")}
            carry, act, _, p_stop = agent.policy(carry, _add_batch(clean), mode="eval", return_stop_token=True)
            
            # Convert p_stop from JAX array to Python float
            p_stop_value = float(p_stop[0]) if hasattr(p_stop, 'shape') else float(p_stop)
            print(f"p_stop: {p_stop_value:.3f}")
            
            # Check if instruction should be cleared based on p_stop
            env.clear_instruction_if_stopped(p_stop_value)
            
            action_arr = _remove_batch(act)["action"].astype(np.int32)
            obs = env.step({"action": action_arr, "reset": False})

            # decode the action text for both agents
            action_text = tokenizer.batch_decode(
                obs['action_text_ids'].reshape(2, -1),
                skip_special_tokens=True
            )

            step_id += 1

            # Log the action and p_stop value
            log_entry = {
                "step": step_id,
                "text": f"A0: {action_text[0]}   A1: {action_text[1]}   (p_stop: {p_stop_value:.2f})",
                "ts": time.strftime("%H:%M:%S"),
            }
            
            # Add instruction status if there's an active instruction
            if env.has_active_instruction():
                log_entry["text"] += " [INSTR ACTIVE]"
            
            instr_log.append(log_entry)
            instr_log = instr_log[-20:]  # keep last 20 entries

            if obs["is_last"]:
                paused = True
        time.sleep(1.0 / max(1, fps))


threading.Thread(target=step_loop, daemon=True).start()

# ── helpers -----------------------------------------------------------
def encode_frame(img_np: np.ndarray) -> bytes:
    """Converts HWC uint8 array (RGB) -> PNG bytes."""
    img = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ── Flask views -------------------------------------------------------
app = Flask(__name__)

BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <title>Overcooked Live Feed</title>
  <!-- replace only the <style> block or copy-paste over the old rules -->
<style>
  :root { color-scheme: dark; }
  body        { margin:0; padding:0; background:#000; color:#fff;
                font-family:system-ui,sans-serif; font-size:1.15rem; }
  .wrap       { max-width:1600px; margin:0 auto; padding:2rem 1rem; }
  h2          { margin:.2rem 0 1.2rem 0; font-size:2.3rem; font-weight:600;
                text-align:center; }

  /* ── GRID with two columns: video | log  ─────────────────────────── */
  /* ── one-column row, centred ───────────────────────────────────────── */
.row        { position:relative; width:fit-content; margin:0 auto; }

#gameframe  { width:48vw; max-width:720px; border:4px solid #fff;
              border-radius:8px; display:block; }   /* block keeps margin calc */

/* ── log floats to the right of the frame ─────────────────────────── */
.logcard    { position:absolute; top:0; left:100%;           /* stick to right edge */
              margin-left:2rem; width:22vw; max-width:360px; }

  .logcard h3 { margin:.2rem 0 .6rem 0; font-size:1.4rem; }
  #logpane    { list-style:none; padding-left:0; max-height:60vh;
                overflow-y:auto; font-size:.95rem; }
  #logpane li { margin:.1rem 0; }

  /* ── controls row (unchanged) ────────────────────────────────────── */
  .controls   { margin-top:2rem; display:flex; flex-wrap:wrap; gap:1rem;
                justify-content:center; align-items:center; }
  button      { padding:.5rem 1.1rem; font-size:1rem; background:#111;
                color:#fff; border:2px solid #fff; border-radius:6px;
                cursor:pointer; }
  button:hover{ background:#222; }
  input[type=range]{ width:240px; }
  input[type=text] { min-width:280px; background:#111; color:#fff;
                     border:2px solid #fff; border-radius:4px;
                     padding:.4rem .6rem; font-size:1rem; }
  label       { margin-right:.5rem; }
</style>

</head>
<body>
<div class="wrap">
  <h2>Overcooked&nbsp;Live&nbsp;Feed</h2>

  <div class="row">
    <!-- video -->
    <img id="gameframe" src="{{ url_for('frame') }}">

    <!-- log & step -->
    <div class="logcard">
      <h3>Step: <span id="stepnum">0</span></h3>
      <h4>Instruction&nbsp;Log</h4>
      <ul id="logpane"></ul>
    </div>
  </div>

  <!-- controls -->
  <div class="controls">
    <form method="post" action="{{ url_for('toggle') }}">
      <button type="submit">{{ '▶️ Start' if paused else '⏸️ Pause' }}</button>
    </form>

    <form method="post" action="{{ url_for('reset_all') }}">
      <button type="submit">↺ Reset</button>
    </form>

    <label>Speed: <span id="fpsval">{{ fps }}</span>&nbsp;step/s</label>
    <input type="range" min="1" max="20" value="{{ fps }}" id="fps">

    <input type="text" id="instr" size="30" value="{{ user_text }}">
    <button id="send">↩︎ Send</button>
  </div>
</div>

<script>
  const img      = document.getElementById('gameframe');
  const fpsSlide = document.getElementById('fps');
  const fpsVal   = document.getElementById('fpsval');
  const sendBtn  = document.getElementById('send');
  const instrBox = document.getElementById('instr');

  /* refresh frame */
  setInterval(()=>{ img.src="{{ url_for('frame') }}?t="+Date.now(); }, 200);

  /* speed slider */
  fpsSlide.oninput  = ()=> fpsVal.textContent = fpsSlide.value;
  fpsSlide.onchange = ()=> fetch("{{ url_for('set_fps') }}", {
      method:"POST", headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body:"fps="+fpsSlide.value });

  /* send instruction */
  sendBtn.onclick = ()=> fetch("{{ url_for('set_instr') }}", {
      method:"POST", headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body:"text="+encodeURIComponent(instrBox.value) });

  /* poll status */
  function refreshStatus(){
    fetch("{{ url_for('status') }}").then(r=>r.json()).then(d=>{
      document.getElementById('stepnum').textContent = d.step;
      const pane=document.getElementById('logpane'); pane.innerHTML='';
      d.log.slice().reverse().forEach(it=>{
        const li=document.createElement('li');
        li.textContent=`[${it.ts}] step ${it.step}: ${it.text}`;
        pane.appendChild(li);
      });
    });
  }
  setInterval(refreshStatus,500); refreshStatus();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(
        BASE_HTML,
        paused=paused,
        fps=fps,
        user_text=user_text,
    )

@app.route("/toggle", methods=["POST"])
def toggle():
    # cp = elements.Checkpoint(f'{ckpt_path}/ckpt')
    # cp.agent = agent
    # cp.load_or_save()
    global paused, obs, carry
    paused = not paused
    if not paused and (obs["is_last"] or obs["is_terminal"]):
        obs   = env.step({"action": np.array([0, 0], np.int32), "reset": True})
        # action_text = tokenizer.batch_decode(obs['action_text_ids'].reshape([2, 16]), skip_special_tokens=True)
        carry = agent.init_policy(1)
    return redirect(url_for("index"))

@app.route("/set_fps", methods=["POST"])
def set_fps():
    global fps
    fps = int(request.form["fps"])
    return ("", 204)

@app.route("/set_instr", methods=["POST"])
def set_instr():
    global user_text, instr_log
    user_text = request.form["text"]
    env.set_instruction(user_text)
    instr_log.append({
        "step": step_id,
        "text": f"[NEW INSTRUCTION] {user_text}",
        "ts": time.strftime("%H:%M:%S"),
    })
    # keep only last 30 entries
    instr_log = instr_log[-30:]
    return ("", 204)

# Optional: Add a status endpoint to check instruction state
@app.route("/instr_status")
def instr_status():
    return Response(
        json.dumps({
            "has_instruction": env.has_active_instruction(),
            "current_text": user_text if env.has_active_instruction() else None
        }),
        mimetype="application/json",
    )

import json
@app.route("/status")
def status():
    return Response(
        json.dumps({"step": step_id, "log": instr_log}),
        mimetype="application/json",
    )

@app.route("/reset_all", methods=["POST"])
def reset_all():
    global obs, carry, step_id, instr_log, paused, tokenizer
    obs      = env.step({"action": np.array([0,0],np.int32), "reset": True})
    # action_text = tokenizer.batch_decode(obs['action_text_ids'].reshape([2, 16]), skip_special_tokens=True)
    carry    = agent.init_policy(1)         # uncomment if using agent
    step_id  = 0
    instr_log= []
    paused   = True
    # cp = elements.Checkpoint(f'{ckpt_path}/ckpt')
    # cp.agent = agent
    # cp.load_or_save()
    return redirect(url_for("index"))


@app.route("/frame")
def frame():
    # Return current RGB frame as PNG over HTTP
    return Response(encode_frame(obs["image"]), mimetype="image/png")

# ── run ----------------------------------------------------------------
if __name__ == "__main__":
    # Using eventlet for simple concurrency (install with `pip install eventlet`)
    # import eventlet
    # eventlet.monkey_patch()     # let standard libs work cooperatively
    app.run(host="0.0.0.0", port=8601, debug=False, )
