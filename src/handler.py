import os
import base64
from io import BytesIO

import torch
from diffusers import FluxPipeline
import runpod

# Load once per worker (cold-start cost amortised across all jobs in the worker lifetime).
# CPU offload keeps VRAM low enough for 24GB consumer cards while remaining fast on
# 40GB+ A100/H100 workers. xformers is best-effort — falls back to sdpa if unavailable.
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception:
    pass  # ponytail: xformers not installed / incompatible, diffusers will fall back


def handler(job):
    job_input = job.get("input") or {}

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "A 'prompt' is required in the input."}

    seed = int(job_input.get("seed", 0))
    steps = int(job_input.get("num_inference_steps", 28))
    guidance_scale = float(job_input.get("guidance_scale", 3.5))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))

    # Clamp to multiples of 8 — FLUX rejects awkward strides.
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            max_sequence_length=512,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"image_base64": image_base64, "seed": seed, "size": [width, height]}


runpod.serverless.start({"handler": handler})