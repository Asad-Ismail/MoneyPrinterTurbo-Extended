"""
AI video generation using Wan2.1/2.2 text-to-video models via HuggingFace diffusers.
Generates short video clips from text prompts as an alternative to stock footage.
"""

import os
import re
import gc
import time
from typing import List, Optional

from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

try:
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Global pipeline cache
_pipeline = None
_pipeline_model_id = None

# Model presets: (model_id, default_height, default_width, num_frames, fps)
MODELS = {
    "wan2.1-1.3b": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "heights": {"9:16": 480, "16:9": 480, "1:1": 480},
        "widths": {"9:16": 272, "16:9": 832, "1:1": 480},
        "num_frames": 81,
        "fps": 15,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "dtype": "bfloat16",
        "vram_gb": 8,
    },
    "wan2.2-5b": {
        "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "heights": {"9:16": 704, "16:9": 704, "1:1": 704},
        "widths": {"9:16": 400, "16:9": 1280, "1:1": 704},
        "num_frames": 121,
        "fps": 24,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "dtype": "bfloat16",
        "vram_gb": 24,
    },
}

DEFAULT_MODEL = "wan2.1-1.3b"

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text overlay, "
    "low resolution, pixelated, overexposed, underexposed"
)


def is_available() -> bool:
    if not DIFFUSERS_AVAILABLE:
        return False
    return torch.cuda.is_available()


def get_model_config(model_name: str = "") -> dict:
    if not model_name:
        model_name = config.app.get("video_gen_model", DEFAULT_MODEL)
    return MODELS.get(model_name, MODELS[DEFAULT_MODEL])


def _load_pipeline(model_name: str = ""):
    global _pipeline, _pipeline_model_id

    model_config = get_model_config(model_name)
    model_id = model_config["model_id"]

    if _pipeline is not None and _pipeline_model_id == model_id:
        return _pipeline

    # Free existing pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Loading video generation model: {model_id}")
    dtype = getattr(torch, model_config["dtype"])

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=dtype)

    # Use CPU offload if VRAM is tight
    try:
        free_vram_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        if free_vram_gb < model_config["vram_gb"]:
            logger.info(f"Free VRAM ({free_vram_gb:.1f}GB) < recommended ({model_config['vram_gb']}GB), using CPU offload")
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    except Exception:
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_tiling()

    _pipeline = pipe
    _pipeline_model_id = model_id
    logger.info(f"Video generation model loaded: {model_id}")
    return _pipeline


def _get_resolution(model_config: dict, aspect: VideoAspect) -> tuple:
    aspect_key = aspect.value if hasattr(aspect, 'value') else str(aspect)
    h = model_config["heights"].get(aspect_key, 480)
    w = model_config["widths"].get(aspect_key, 832)
    return h, w


def _enhance_prompt(prompt: str) -> str:
    """Add cinematic quality hints to the prompt for better generation."""
    prompt = prompt.strip()
    if not prompt:
        return prompt
    # Don't add hints if the prompt already has quality keywords
    quality_words = ["cinematic", "4k", "high quality", "professional", "detailed"]
    if any(w in prompt.lower() for w in quality_words):
        return prompt
    return f"{prompt}, cinematic lighting, high quality, smooth motion"


def generate_clip(
    prompt: str,
    output_path: str,
    video_aspect: VideoAspect = VideoAspect.portrait,
    model_name: str = "",
) -> Optional[str]:
    """
    Generate a single video clip from a text prompt.
    Returns the output file path on success, None on failure.
    """
    if not is_available():
        logger.error("AI video generation not available (need diffusers + CUDA GPU)")
        return None

    model_config = get_model_config(model_name)
    height, width = _get_resolution(model_config, video_aspect)
    enhanced_prompt = _enhance_prompt(prompt)

    logger.info(f"Generating clip: '{prompt[:80]}...' ({width}x{height}, "
                f"{model_config['num_frames']} frames)")

    try:
        pipe = _load_pipeline(model_name)

        start_time = time.time()
        output = pipe(
            prompt=enhanced_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=height,
            width=width,
            num_frames=model_config["num_frames"],
            guidance_scale=model_config["guidance_scale"],
            num_inference_steps=model_config["num_inference_steps"],
        )

        export_to_video(output.frames[0], output_path, fps=model_config["fps"])
        elapsed = time.time() - start_time
        logger.info(f"Clip generated in {elapsed:.0f}s: {output_path}")

        # Cleanup intermediate tensors
        del output
        gc.collect()
        torch.cuda.empty_cache()

        return output_path

    except Exception as e:
        logger.error(f"Failed to generate clip: {e}")
        return None


def _split_script_to_prompts(script: str, video_terms: list = None) -> List[str]:
    """
    Split a video script into per-clip prompts.
    Each sentence or short paragraph becomes a visual prompt.
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?。！？])\s*', script.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    if not sentences:
        # Fallback: use video_terms as prompts
        if video_terms:
            return video_terms
        return [script[:200]]

    # Merge very short sentences
    prompts = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < 120:
            current = f"{current} {s}".strip()
        else:
            if current:
                prompts.append(current)
            current = s
    if current:
        prompts.append(current)

    return prompts if prompts else sentences


def generate_videos(
    task_id: str,
    video_script: str,
    video_terms: List[str],
    video_aspect: VideoAspect = VideoAspect.portrait,
    audio_duration: float = 0.0,
    max_clip_duration: int = 5,
) -> List[str]:
    """
    Generate AI video clips for a full script. Returns list of file paths.
    This is the main entry point, matching the interface of material.download_videos.
    """
    if not is_available():
        logger.error("AI video generation not available")
        return []

    model_name = config.app.get("video_gen_model", DEFAULT_MODEL)
    model_config = get_model_config(model_name)
    clip_duration = model_config["num_frames"] / model_config["fps"]

    # Figure out how many clips we need
    if audio_duration > 0:
        num_clips_needed = max(1, int(audio_duration / clip_duration) + 1)
    else:
        num_clips_needed = 3

    # Generate prompts from script
    prompts = _split_script_to_prompts(video_script, video_terms)

    # Extend prompts if we need more clips than we have prompts
    while len(prompts) < num_clips_needed:
        prompts.extend(prompts[:num_clips_needed - len(prompts)])
    prompts = prompts[:num_clips_needed]

    # Generate clips
    save_dir = utils.task_dir(task_id)
    os.makedirs(save_dir, exist_ok=True)

    generated_paths = []
    for i, prompt in enumerate(prompts):
        output_path = os.path.join(save_dir, f"ai-clip-{i:03d}.mp4")
        logger.info(f"Generating clip {i+1}/{len(prompts)}")

        result = generate_clip(
            prompt=prompt,
            output_path=output_path,
            video_aspect=video_aspect,
            model_name=model_name,
        )

        if result and os.path.exists(result):
            generated_paths.append(result)
        else:
            logger.warning(f"Clip {i+1} failed, skipping")

    logger.info(f"Generated {len(generated_paths)}/{len(prompts)} video clips")
    return generated_paths
