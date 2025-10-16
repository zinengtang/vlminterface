from transformers import (
  AutoConfig, BertConfig, RobertaConfig, XLMRobertaConfig, T5Config,
  FlaxBertModel, FlaxRobertaModel, FlaxXLMRobertaModel, FlaxT5EncoderModel, AutoTokenizer
)
import jax.numpy as jnp
import jax
import numpy as np

from transformers import FlaxCLIPVisionModel, FlaxDinov2Model
from flax.core import FrozenDict

def load_flax_text_encoder(name: str, dtype=jnp.bfloat16):
  cfg = AutoConfig.from_pretrained(name)
  if   isinstance(cfg, BertConfig):        cls = FlaxBertModel
  elif isinstance(cfg, RobertaConfig):     cls = FlaxRobertaModel
  elif isinstance(cfg, XLMRobertaConfig):  cls = FlaxXLMRobertaModel
  elif isinstance(cfg, T5Config):          cls = FlaxT5EncoderModel
  else:
    raise ValueError("Use a Flax BERT/Roberta/XLM-R/T5 model.")

  # 1) Load on CPU so params aren't device arrays.
  cpu = jax.devices('cpu')[0]
  with jax.default_device(cpu):
    model = cls.from_pretrained(name, dtype=dtype)

  # 2) Convert param pytree to NumPy (host) to avoid device→host during trace.
  params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), model.params)
  return model, params_np


# clip_model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", dtype=jnp.float32)
# dino_model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base", dtype=jnp.float32)

def load_flax_vision_encoders(name: str, dtype=jnp.bfloat16):
  cpu = jax.devices('cpu')[0]
  with jax.default_device(cpu):
    clip_model = FlaxCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", dtype=dtype)
    dino_model = FlaxDinov2Model.from_pretrained("facebook/dinov2-base", dtype=dtype)

  clip_vars = clip_model.params  # FrozenDict
  dino_vars = dino_model.params  # FrozenDict

  # 2) Convert param pytree to NumPy (host) to avoid device→host during trace.
  clip_params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), clip_model.params)
  dino_params_np = jax.tree_util.tree_map(lambda x: np.asarray(x), dino_model.params)
  return clip_model, clip_params_np, dino_model, dino_params_np


# vision_fusion_encoder.py
from typing import Any, Optional, Literal
import jax.numpy as jnp
import jax
from flax import linen as nn

# CLIP & DINOv2 preproc constants
_CLIP_MEAN = jnp.array([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD  = jnp.array([0.26862954, 0.26130258, 0.27577711])
_IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = jnp.array([0.229, 0.224, 0.225])

def _resize_bilinear(x, size: int):
    # x: [B, H, W, C] in [0,1]; use jax.image.resize (no antialias)
    import jax.image as jimage
    return jimage.resize(x, (x.shape[0], size, size, x.shape[-1]), method="linear")

def _norm(x, mean, std):
    return (x - mean) / std

class DualBackboneVisionEncoder(nn.Module):
    """Fuse CLIP ViT-B/32 and DINOv2-base into a single feature vector."""
    clip_model: Any           # FlaxCLIPVisionModel (module object)
    dino_model: Any           # FlaxDinov2Model (module object)
    clip_params: Any          # FrozenDict
    dino_params: Any          # FrozenDict
    enc_dim: int = 1024       # per-branch projection dim
    out_dim: int = 1024       # fused output dim to RSSM
    fusion: Literal["concat_mlp","gated_add","film"] = "gated_add"
    dtype: Any = jnp.float32
    image_size: int = 224
    train_backbones: bool = False  # keep False to freeze

    @nn.compact
    def __call__(self, images: jnp.ndarray, *, train: bool = False):
        # images: [B,H,W,3] in [0,1]
        x = images
        x = _resize_bilinear(x, self.image_size)

        # Branch-specific normalization
        # CLIP branch
        x_clip = _norm(x, _CLIP_MEAN, _CLIP_STD)
        clip_out = self.clip_model.apply(
            {'params': self.clip_params},
            pixel_values=jnp.transpose(x_clip, (0, 3, 1, 2)), # HF Flax expects [B,C,H,W]
            train=self.train_backbones and train,
            mutable=None
        )
        # Many HF Flax vision models expose 'pooler_output'; fallback to CLS if missing.
        clip_pool = getattr(clip_out, "pooler_output", None)
        if clip_pool is None:
            clip_pool = clip_out.last_hidden_state[:, 0]  # [B, 768]
        clip_pool = clip_pool.astype(self.dtype)

        # DINOv2 branch
        x_dino = _norm(x, _IMAGENET_MEAN, _IMAGENET_STD)
        dino_out = self.dino_model.apply(
            {'params': self.dino_params},
            pixel_values=jnp.transpose(x_dino, (0, 3, 1, 2)),
            train=self.train_backbones and train,
            mutable=None
        )
        dino_pool = getattr(dino_out, "pooler_output", None)
        if dino_pool is None:
            dino_pool = dino_out.last_hidden_state[:, 0]  # [B, 768]
        dino_pool = dino_pool.astype(self.dtype)

        # Project to common dim
        clip_proj = nn.Dense(self.enc_dim, use_bias=False, name="clip_proj")(nn.LayerNorm()(clip_pool))
        dino_proj = nn.Dense(self.enc_dim, use_bias=False, name="dino_proj")(nn.LayerNorm()(dino_pool))

        if self.fusion == "concat_mlp":
            h = jnp.concatenate([clip_proj, dino_proj], axis=-1)  # [B, 2*enc_dim]
            h = nn.LayerNorm()(h)
            h = nn.relu(nn.Dense(self.out_dim, name="fuse_mlp_1")(h))
            h = nn.Dense(self.out_dim, name="fuse_mlp_2")(h)
        elif self.fusion == "gated_add":
            # Learn a gate from concatenated features, then mix.
            g = nn.sigmoid(nn.Dense(self.enc_dim, name="gate")(jnp.concatenate([clip_proj, dino_proj], -1)))
            mix = g * clip_proj + (1.0 - g) * dino_proj          # [B, enc_dim]
            h = nn.relu(nn.Dense(self.out_dim, name="fuse_out")(nn.LayerNorm()(mix)))
        else:  # "film"
            # Use DINO as content, CLIP to generate FiLM params
            gamma_beta = nn.Dense(2 * self.enc_dim, name="film_params")(clip_proj)
            gamma, beta = jnp.split(gamma_beta, 2, axis=-1)
            dino_norm = nn.LayerNorm()(dino_proj)
            film = gamma * dino_norm + beta
            h = nn.relu(nn.Dense(self.out_dim, name="fuse_out")(film))

        return h  # [B, out_dim]
