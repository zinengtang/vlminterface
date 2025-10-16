# ===== File: vlm_utils.py =====
"""
Unified wrappers around several vision‑language models (VLMs) for Dreamer‑style
agents, with explicit support for Qwen/Qwen2.5‑VL 3B/7B/32B.

Quick usage
-----------
>>> from vlm_utils import VLMWrapper
>>> vlm = VLMWrapper(model_type="qwenvl", model_id="Qwen/Qwen2.5-VL-7B-Instruct", device=0)
>>> cmds = vlm(frames=[pil_image], action_lists=[["north", "north", "interact pot"]])

Notes
-----
- For Qwen2.5‑VL, we rely on `transformers.AutoProcessor` and
  `transformers.AutoModelForCausalLM` with `trust_remote_code=True`.
- We use `qwen_vl_utils.process_vision_info` when available for preprocessing;
  otherwise we pass raw PIL images into the processor.
- Return value is a list of strings, one per (frame, action_list) pair.
- We extract commands wrapped in ##...## when present; otherwise we return the
  raw generated text.
"""

from __future__ import annotations

import os
import re
import json
from typing import List, Optional

import torch
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# Optional helper from Qwen repo for vision inputs
try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover
    process_vision_info = None

# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------
BASE_PROMPT = (
    "You are an expert game analyst. Given a single game frame and the list "
    "of low‑level key presses executed over the next N frames, output ONE "
    "short, high‑level command that best summarises what the agent is trying "
    "to do. Reply with an imperative verb phrase, no punctuation."
)

GENERAL_PROMPT = (
    "You are helpful AI assistant. You are controlling the agent playing a game. "
    "Analyze a series of movement actions to summarize as a medium to high level imperative. "
    "Keep your thought process succinct. And always wrap the final high level action command with ##...## "
    "Diversify your language style. Be creative. Make your instruction randomly from low level to high level. "
    "Don't output instruction that has anything other than the action description. "
    "Your instruction should cover all actions either in high level or low level and juxtapose all instructions with 'and'."
)

OVERCOOKED_PROMPT = (
    "You are helpful AI assistant. In overcooked game, analyze a series of movement actions of an agent "
    "to summarize as a medium to high level imperative. Always wrap the final command with ##...##. "
    "Keep the output succinct (≤16 words). Don't add facts not present in actions."
)

PROACTIVE_PROMPT = (
    "You are an expert game analyst. Given a single frame and key presses for the next N frames, output YES or NO "
    "if the agent needs new actions now."
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
import base64
import io

def pil_to_data_uri(img: Image.Image, format: str = 'JPEG') -> str:
    """Convert a PIL Image to a Base64 data URI (Qwen chat expects data URIs)."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    b64 = base64.b64encode(buffered.getvalue()).decode('ascii')
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# VLM Wrapper
# ---------------------------------------------------------------------------
class VLMWrapper:
    """Load a chosen VLM once and provide a common __call__ interface.

    Supported model_type values:
      - "qwenvl"  -> Qwen/Qwen2.5-VL (3B/7B/32B) via model_id
      - "phi3"    -> (text-only fallback, optional)

    For other families you can extend similarly (llava, gemma, etc.).
    """

    SUPPORTED = {"qwenvl", "phi3"}

    def __init__(
        self,
        model_type: str = "phi3",
        model_id: Optional[str] = None,
        device: int | str = 0,
        dtype: torch.dtype = torch.bfloat16,
        gemma_token: Optional[str] = None,  # reserved for future
        action_map_path: Optional[str] = None,
    ) -> None:
        model_type = (model_type or "").lower()
        if model_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported model_type {model_type}. Choose from {self.SUPPORTED}.")

        self.model_type = model_type
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.dtype = dtype
        self.tokenizer = None
        self.processor = None

        if action_map_path and os.path.exists(action_map_path):
            try:
                self.action_maps = json.load(open(action_map_path, "r"))
            except Exception:
                self.action_maps = None
        else:
            self.action_maps = None

        if self.model_type == "qwenvl":
            from transformers import BitsAndBytesConfig
            self.quantize = "8bit"
            bnb_config = None
            if self.quantize in {"8bit", "4bit"}:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit = (self.quantize == "8bit"),
                    load_in_4bit = (self.quantize == "4bit"),
                    llm_int8_threshold=6.0,           # good default
                    llm_int8_has_fp16_weight=False,   # typical setting
                    # You can enable CPU offload if VRAM is tight:
                    # llm_int8_enable_fp32_cpu_offload=True,
                )

            # Default to 7B if not specified
            self.model_id = model_id or "Qwen/Qwen2.5-VL-7B-Instruct"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=None if bnb_config else self.dtype,  # dtype handled by bnb if quantized
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,                  # << enable 8-bit/4-bit
            ).to(self.device).eval()
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            # Qwen chat typically doesn't use a standalone tokenizer, processor wraps it
        elif self.model_type == "phi3":  # optional text-only fallback
            self.model_id = model_id or "microsoft/Phi-3-medium-4k-instruct"
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    trust_remote_code=False,
                )
                .to(self.device)
                .eval()
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        else:
            raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        frames: List[Image.Image],
        action_lists: Optional[List[List[str]]] = None,
        max_new_tokens: int = 64,
        system_prompt: Optional[str] = None,
        check_proactive: bool = False,
        temperature: float = 0.0,
    ) -> List[str]:
        """Generate a high‑level command for each (frame, action_list) pair."""
        if not frames:
            return []

        # Choose prompt
        if check_proactive:
            sys_prompt = PROACTIVE_PROMPT
        else:
            sys_prompt = OVERCOOKED_PROMPT if action_lists is not None else GENERAL_PROMPT

        if self.model_type == "qwenvl":
            # For Qwen2.5-VL we build chat messages with data URI images.
            # We currently support 1 image per call (extend as needed).
            img_base64 = pil_to_data_uri(frames[0])
            if action_lists is not None:
                prompt_string = f"{sys_prompt} Action list: {action_lists}"
            else:
                prompt_string = sys_prompt

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_base64},
                        {"type": "text", "text": prompt_string},
                    ],
                }
            ]

            # Prepare inputs using processor. Use qwen helper if available to normalize vision inputs.
            if process_vision_info is not None:
                vision_inputs, _ = process_vision_info(messages)
            else:
                vision_inputs = frames  # raw PIL images as best-effort fallback

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=vision_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self.device, self.dtype)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0.0,
                temperature=float(temperature) if temperature > 0.0 else None,
            )

            # Decoder: for Qwen2.5 the processor handles special tokens & offset
            gen_text = self.processor.batch_decode(
                output_ids[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if check_proactive:
                return ["YES" if ("YES" in gen_text) else "NO"]

            # Extract ##...## if present
            m = re.findall(r"##(.*?)##", gen_text)
            return [m[0] if m else gen_text.strip()]

        elif self.model_type == "phi3":
            # Text-only fallback; ignores image for now.
            if action_lists is not None:
                prompt_string = f"{sys_prompt} Action list: {action_lists}"
            else:
                prompt_string = sys_prompt

            # Minimal chat template
            input_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt_string},
                ],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                do_sample=temperature > 0.0,
                temperature=float(temperature) if temperature > 0.0 else None,
            )
            text = self.tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            if check_proactive:
                return ["YES" if ("YES" in text) else "NO"]
            m = re.findall(r"##(.*?)##", text)
            return [m[0] if m else text.strip()]

        else:  # pragma: no cover
            raise AssertionError("unreachable")
