import os
import base64
from io import BytesIO

import torch
import runpod
from diffusers import FluxPipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MODEL_ID = os.getenv("MODEL_ID", "black-forest-labs/FLUX.1-dev")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

print(f"Loading {MODEL_ID}...")
print(f"Device: {DEVICE}")

# -----------------------------------------------------------------------------
# Load model ONCE (cold start only)
# -----------------------------------------------------------------------------

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    use_safetensors=True,
)

if DEVICE == "cuda":
    pipe.enable_model_cpu_offload()

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled.")
    except Exception:
        print("xFormers not available, using SDPA.")

print("Model loaded successfully.")

# -----------------------------------------------------------------------------
# Handler
# -----------------------------------------------------------------------------

def handler(job):
    job_input = job["input"]

    prompt = job_input.get("prompt")

    if not prompt:
        return {
            "error": "Missing required field: prompt"
        }

    seed = int(job_input.get("seed", 0))
    width = int(job_input.get("width", 1024))
    height = int(job_input.get("height", 1024))
    steps = int(job_input.get("num_inference_steps", 28))
    guidance = float(job_input.get("guidance_scale", 3.5))

    # FLUX requires multiples of 8
    width = max(64, (width // 8) * 8)
    height = max(64, (height // 8) * 8)

    runpod.serverless.progress_update(job, "Generating image...")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.inference_mode():

        if DEVICE == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                image = pipe(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    max_sequence_length=512,
                    generator=generator,
                ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance,
                num_inference_steps=steps,
                max_sequence_length=512,
                generator=generator,
            ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "image_base64": image_b64,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance,
    }


# -----------------------------------------------------------------------------
# Start worker
# -----------------------------------------------------------------------------

runpod.serverless.start({"handler": handler})  # Required

