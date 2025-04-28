"""
vlm_utils.py
-------------
Unified wrappers around several vision‑language models (VLMs) for
Dreamer‑style agents.

Supported models
----------------
* Qwen/Qwen2.5‑VL‑3B‑Instruct
* llava-hf/llava‑1.5‑7b‑hf
* google/gemma‑3‑12b‑it (requires HF token)

Each wrapper normalises the call signature to

>>> wrapper = VLMWrapper(model_type="qwen", device=0)
>>> cmd_strs = wrapper(frames, action_lists)  # List[str]

where
frames:        list[PIL.Image]        – single RGB frames
action_lists:  list[list[str]]        – low‑level key strings per frame

The system prompt used to coax a high‑level command is shared across models
but can be overridden per call.
"""

import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    LlavaForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)

# Qwen helper util (shipped with the official repo)
try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except ImportError:
    process_vision_info = None  # Qwen will handle raw PIL images instead

# ---------------------------------------------------------------------------
# Action LUT – map discrete env action ids to human‑readable strings.
# Extend or modify to match the task.
# ---------------------------------------------------------------------------
ACTION_LUT = {
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}

# ---------------------------------------------------------------------------
# Shared prompt
# ---------------------------------------------------------------------------
BASE_PROMPT = (
    "You are an expert game analyst. Given a single game frame and the list "
    "of low‑level key presses executed over the next N frames, output ONE "
    "short, high‑level command that best summarises what the agent is trying "
    "to do. Reply with an imperative verb phrase, no punctuation."
)

CLIMB_PROMPT = """
    You are an expert game analyst. In Minecraft survival mode, you need to reach a latitude as high as possible.
    Based on what you see, output ONE short, medium-level command that best describes what the agent
    needs to do in order to reach a high latitude. Reply with an imperative phrase, no punctuation.
    If the target distance is far, try to break it down into smallers steps and reply step 1 like "go to the forest".
"""

import base64
import io
from PIL import Image

def pil_to_data_uri(img: Image.Image, format: str = 'JPEG') -> str:
    """
    Convert a PIL Image to a Base64 data URI.
    Args:
      img: PIL.Image.Image instance.
      format: image format, e.g. 'JPEG' or 'PNG'.
    Returns:
      A string like 'data:image/jpeg;base64,/9j/4AAQ...'
    """
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    b64 = base64.b64encode(buffered.getvalue()).decode('ascii')
    mime = f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"

# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------
class VLMWrapper:
    """Load a chosen VLM once and provide a common __call__ interface."""

    SUPPORTED = {"qwen", "llava", "gemma"}

    def __init__(
        self,
        model_type: str = "qwen",
        device: int | str = 0,
        dtype: torch.dtype = torch.bfloat16,
        gemma_token: Optional[str] = None,
    ) -> None:
        model_type = model_type.lower()
        if model_type not in self.SUPPORTED:
            raise ValueError(f"Unsupported model_type {model_type}. Choose from {self.SUPPORTED}.")

        self.model_type = model_type
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.dtype = dtype

        if model_type == "qwen":
            self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            self.model = (
                Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                )
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        elif model_type == "llava":
            self.model_id = "llava-hf/llava-1.5-7b-hf"
            self.model = (
                LlavaForConditionalGeneration.from_pretrained(
                    self.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
                )
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        elif model_type == "gemma":
            self.model_id = "google/gemma-3-12b-it"
            token = gemma_token or os.getenv("GEMMA_TOKEN")
            if token is None:
                raise ValueError("Gemma model requires a HF token. Pass gemma_token or set GEMMA_TOKEN env var.")
            self.model = (
                Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_id, token=token, torch_dtype=dtype, low_cpu_mem_usage=True
                )
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id, token=token)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        frames: List[Image.Image],
        action_lists: List[List[str]] = None,
        max_new_tokens: int = 32,
        system_prompt: str | None = None,
    ) -> List[str]:
        """Generate a high‑level command for each (frame, action_list) pair."""
        sys_prompt = system_prompt or CLIMB_PROMPT
        img_base64 = pil_to_data_uri(frames[0])
        # Build per‑item conversations expected by each model
        if self.model_type == "qwen":
            convs = []
            # for acts in action_lists:
            convs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_base64},
                        {"type": "text", "text": f"{sys_prompt}"},
                    ],
                }
            )
            text_batch = self.processor.apply_chat_template(
                convs, tokenize=False, add_generation_prompt=True
            )
            # Qwen specific image preprocessing
            image_inputs, _ = process_vision_info(convs) if process_vision_info else (frames, None)
            inputs = self.processor(
                text=text_batch,
                images=image_inputs,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(self.device, self.dtype)
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            outs = self.processor.batch_decode(
                gen_ids[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            return [o.strip() for o in outs]

        elif self.model_type == "llava":
            convs = []
            for acts in action_lists:
                convs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{sys_prompt} Keys pressed: {', '.join(acts)}"},
                            {"type": "image"},
                        ],
                    }
                )
            prompts = [self.processor.apply_chat_template(c, add_generation_prompt=True) for c in convs]
            # Process each frame separately due to Llava API (no batch image support)
            results = []
            for frame, prompt in zip(frames, prompts):
                inputs = self.processor(images=frame, text=prompt, return_tensors="pt").to(self.device, self.dtype)
                gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                res = self.processor.decode(gen[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
                results.append(res.strip())
            return results

        elif self.model_type == "gemma":
            convs = []
            for acts in action_lists:
                convs.append(
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": f"{sys_prompt}"},
                        ],
                    }
                )
                convs.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": "<image>"},
                            {"type": "text", "text": f"Keys pressed: {', '.join(acts)}"},
                        ],
                    }
                )
            # Gemma currently supports *one* sample at a time – loop instead of batching
            outs: List[str] = []
            for frame, conv in zip(frames, convs[::2]):  # take paired convs
                inputs = self.processor.apply_chat_template(
                    [conv, convs[convs.index(conv)+1]],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.device, self.dtype)
                cut = inputs["input_ids"].shape[-1]
                gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                text = self.processor.decode(gen[0][cut:], skip_special_tokens=True)
                outs.append(text.strip())
            return outs

        else:
            raise AssertionError("unreachable")
