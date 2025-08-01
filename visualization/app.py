"""
app.py – Dreamer replay viewer
------------------------------
Run the Flask web app:
    $ python app.py

The app expects a single **captions.json** (same directory) with
keys = full .npz paths and values = captions, e.g.
{
  "/home/.../replay/xyz.npz": "Move east across the forest."
}
"""

import base64, glob, io, json, os
from pathlib import Path
from typing import Dict, List

import imageio
import numpy as np
from flask import Flask, Response, render_template_string, request
from PIL import Image

# ---------------------- Configuration ----------------------
BASE_PATH        = Path("/home/terran/logdir/docker/dreamer")
GROUP_SIZE       = 9                      # 3×3 grid
DEFAULT_GAME     = "minecraft_diamond_base"
CAPTIONS_PATH    = Path("captions.json")  # single captions file

# ---------------------- Load captions ----------------------
try:
    CAPTIONS: Dict[str, str] = json.loads(CAPTIONS_PATH.read_text())
except FileNotFoundError:
    CAPTIONS = {}

# ----------------------- Flask setup -----------------------
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>VLM Interface Replays Gallery</title>
    <style>
      body   { font-family: sans-serif; margin: 20px; }
      form   { margin-bottom: 20px; display: flex; gap: 20px; align-items: center; }
      select { font-size: 1.4em; padding: 6px; }
      .grid  { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }

      .video-item      { position: relative; }
      .video-item img  { display: block; width: 100%; height: auto; }
      .caption         { position: absolute; left: 0; right: 0; bottom: 0; padding: 2px 4px;
                         background: rgba(0,0,0,0.6); color: #fff; font-size: 1.1em;
                         text-align: center; box-sizing: border-box; }
    </style>
  </head>
  <body>
    <h1>VLM Interface Replays Gallery</h1>
    <form method=\"get\">
      <label>Game:
        <select name=\"game\" onchange=\"this.form.submit()\">
          {% for g in games %}
          <option value=\"{{g}}\" {% if g==selected_game %}selected{% endif %}>{{g}}</option>
          {% endfor %}
        </select>
      </label>
      <label>Group:
        <select name=\"group\" onchange=\"this.form.submit()\">
          {% for i in range(groups|length) %}
          <option value=\"{{i}}\" {% if i==selected_group %}selected{% endif %}>Group {{i+1}}</option>
          {% endfor %}
        </select>
      </label>
    </form>

    <div class=\"grid\">
      {% for fn in display_files %}
      <div class=\"video-item\">
        <img src=\"{{ url_for('video', game=selected_game, filename=fn) }}\" alt=\"gif\" />
        <div class=\"caption\">{{ captions.get(fn, '') }}</div>
      </div>
      {% endfor %}
    </div>
  </body>
</html>
"""

# -------------------- Helper utilities --------------------

def list_games() -> List[str]:
    return sorted([p.name for p in BASE_PATH.iterdir() if p.is_dir()])

def list_replays(game: str) -> List[str]:
    pattern = BASE_PATH / game / "replay" / "*.npz"
    return sorted([Path(p).name for p in glob.glob(str(pattern))])

# ---------------------- Flask views -----------------------
@app.route("/")
def index():
    games = list_games()
    selected_game = request.args.get("game", DEFAULT_GAME)
    if selected_game not in games and games:
        selected_game = games[0]

    files = list_replays(selected_game)
    groups = [files[i:i + GROUP_SIZE] for i in range(0, len(files), GROUP_SIZE)]
    group_idx = int(request.args.get("group", len(groups) - 1)) if groups else 0
    group_idx = max(0, min(group_idx, len(groups) - 1))

    # Build mapping {filename -> caption} for this game/group
    caption_map = {}
    for fn in groups[group_idx] if groups else []:
        full_path = str(BASE_PATH / selected_game / "replay" / fn)
        caption_map[fn] = CAPTIONS.get(full_path, "")

    return render_template_string(
        INDEX_HTML,
        games=games,
        selected_game=selected_game,
        groups=groups,
        selected_group=group_idx,
        display_files=groups[group_idx] if groups else [],
        captions=caption_map,
    )

@app.route("/video/<game>/<filename>")
def video(game: str, filename: str):
    path = BASE_PATH / game / "replay" / filename
    if not path.exists():
        return "Not found", 404

    sample = np.load(path)
    frames = sample["image"][-30:]
    if not np.issubdtype(frames.dtype, np.uint8):
        frames = (frames * 255).astype(np.uint8)

    buf = io.BytesIO()
    imageio.mimsave(buf, frames, fps=10, format="gif", loop=0)
    buf.seek(0)
    return Response(buf.read(), mimetype="image/gif")

# ----------------------- Entrypoint -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=50001, debug=True)
