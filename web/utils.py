# Modified ManualInstrWrapper
import embodied
import numpy as np
from dreamerv3.sentence_embedding import SentenceEmbedder

class ManualInstrWrapper(embodied.wrappers.Wrapper):
    """
    Inject a natural-language instruction into obs['instructions_ids'].
    Clears instruction when p_stop exceeds threshold.
    """
    def __init__(self, env, text_encoder=None, stop_threshold=0.65):
        super().__init__(env)
        self._encoder = text_encoder or SentenceEmbedder()
        self._current_instr = None          # holds current instruction text
        self._current_vec = None            # holds current encoded instruction
        self._stop_threshold = stop_threshold
        self._default_vec = None           # empty/default instruction encoding
        
        # Initialize default (empty) instruction
        _, caption_ids = self._encoder.encode(["", ""], 32)
        self._default_vec = caption_ids.cpu().numpy()

    # -------- public API --------
    def set_instruction(self, text: str):
        """Set a new instruction."""
        with np.errstate(all='ignore'):
            _, caption_ids = self._encoder.encode([text, ""], 32)
            self._current_instr = text
            self._current_vec = caption_ids.cpu().numpy()
    
    def clear_instruction_if_stopped(self, p_stop):
        """Clear instruction if p_stop exceeds threshold."""
        if self._current_vec is not None and p_stop > self._stop_threshold:
            print(f"Clearing instruction (p_stop={p_stop:.3f} > {self._stop_threshold})")
            self._current_instr = None
            self._current_vec = None
    
    def has_active_instruction(self):
        """Check if there's an active instruction."""
        return self._current_vec is not None

    # -------- overrides --------
    def env_reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._inject(obs)

    def env_step(self, action):
        obs = self.env.step(action)
        return self._inject(obs)

    # -------- helpers --------
    def _inject(self, obs):
        # Always inject either current instruction or default
        if self._current_vec is not None:
            obs['instructions_ids'] = self._current_vec
        else:
            obs['instructions_ids'] = self._default_vec
        return obs

# web/runtime_config.py   (drop next to app.py)
import os, sys, pathlib, ruamel.yaml as yaml, elements

def build_config(
    *,                                  # keyword-only
    logdir="logs/overcooked_exp",
    task="overcooked",
    extra_argv=None,                    # e.g. ['--agent.use_vlm', 'False']
):
    """
    Replicates dreamerv3/main.py flag handling but lets you pass your own argv
    from a notebook, Streamlit, Gradio, etc.
    """
    folder = pathlib.Path(__file__).parent.parent / "dreamerv3"
    sys.path.insert(0, str(folder.parent))      # make local imports happy

    # ---------- load raw YAML bundles ---------------------------------
    cfg_text = (folder / "configs.yaml").read_text()
    cfgs = yaml.YAML(typ="safe").load(cfg_text)

    # ---------- synthetic command-line --------------------------------
    argv = [
        "--configs", f"defaults,{task}",
        "--logdir", str(pathlib.Path(logdir).absolute()),
        "--script", "eval_only",
        "--run.eval_envs", "1",
        # "--run.debug", "True",
        # "--jax.debug", "True"
    ]
    if extra_argv:
        argv.extend(extra_argv)

    # ---------- original snippet logic (slightly factored) ------------
    parsed, other = elements.Flags(configs=["defaults"]).parse_known(argv)
    config = elements.Config(cfgs["defaults"])
    for name in parsed.configs:
        config = config.update(cfgs[name])
    config = elements.Flags(config).parse(other)

    # if you want to *force* VLM on by default:
    config = config.update(agent=dict(use_vlm=True))

    # stamp a fresh logdir unless you passed an absolute path
    config = config.update(
        logdir=config.logdir.format(timestamp=elements.timestamp())
    )

    return config
