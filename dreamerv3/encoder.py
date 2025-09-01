from transformers import (
  AutoConfig, BertConfig, RobertaConfig, XLMRobertaConfig, T5Config,
  FlaxBertModel, FlaxRobertaModel, FlaxXLMRobertaModel, FlaxT5EncoderModel, AutoTokenizer
)
import jax.numpy as jnp
import jax
import numpy as np

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