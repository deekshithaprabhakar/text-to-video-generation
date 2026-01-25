!pip -q install -U "diffusers>=0.25.0" "transformers>=4.35.0" accelerate safetensors \ imageio imageio-ffmpeg opencv-python gradio

import os, json, time, random
import torch
import imageio
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"  
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_pipe():
    """
    Loads the ModelScope Text-to-Video pipeline.
    Uses fp16 + CPU offload to reduce VRAM usage.
    """
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Helps on smaller GPUs (T4) by offloading parts to CPU
    pipe.enable_model_cpu_offload()

    # Memory-efficient attention if available (Torch 2.x)
    # (Safe: if not available, it won't break)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe

pipe = load_pipe()

def generate_video(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 24,
    fps: int = 8,
    num_inference_steps: int = 30,
    guidance_scale: float = 9.0,
    seed: int | None = None,
):
    """
    Generates a video (list of frames) from text and saves MP4 + metadata JSON.
    Returns: (mp4_path, metadata_path)
    """
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = result.frames[0]
    dt = time.time() - t0

    run_id = f"t2v_{int(time.time())}_{seed}"
    mp4_path = os.path.join(OUT_DIR, f"{run_id}.mp4")
    meta_path = os.path.join(OUT_DIR, f"{run_id}.json")

    # Save mp4
    imageio.mimsave(mp4_path, frames, fps=fps)

    # Save metadata
    meta = {
        "model_id": MODEL_ID,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "seconds_runtime": round(dt, 2),
        "output_mp4": mp4_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return mp4_path, meta_path

def generate_video(
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 24,
    fps: int = 8,
    num_inference_steps: int = 30,
    guidance_scale: float = 9.0,
    seed: int | None = None,
):
    """
    Generates a video (list of frames) from text and saves MP4 + metadata JSON.
    Returns: (mp4_path, metadata_path)
    """
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    frames = result.frames[0]
    dt = time.time() - t0

    run_id = f"t2v_{int(time.time())}_{seed}"
    mp4_path = os.path.join(OUT_DIR, f"{run_id}.mp4")
    meta_path = os.path.join(OUT_DIR, f"{run_id}.json")

    # Save mp4
    imageio.mimsave(mp4_path, frames, fps=fps)

    # Save metadata
    meta = {
        "model_id": MODEL_ID,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "seconds_runtime": round(dt, 2),
        "output_mp4": mp4_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return mp4_path, meta_path

prompt = "A cinematic shot of a corgi wearing sunglasses riding a skateboard at sunset, shallow depth of field"
neg = "low quality, blurry, distorted"

mp4, meta = generate_video(
    prompt=prompt,
    negative_prompt=neg,
    num_frames=24,
    fps=8,
    num_inference_steps=30,
    guidance_scale=9.0,
    seed=1234,
)

print("Saved video:", mp4)
print("Saved metadata:", meta)

from IPython.display import Video, display
display(Video(mp4, embed=True))

def gradio_generate(prompt, negative_prompt, num_frames, fps, steps, guidance, seed):
    mp4, meta = generate_video(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=int(num_frames),
        fps=int(fps),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        seed=int(seed),
    )
    return mp4, meta

demo = gr.Interface(
    fn=gradio_generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=3, value="A futuristic neon city street in the rain, cinematic, 4k"),
        gr.Textbox(label="Negative Prompt", lines=2, value="blurry, low quality, distorted, artifacts"),
        gr.Slider(8, 60, value=24, step=1, label="Num Frames"),
        gr.Slider(4, 24, value=8, step=1, label="FPS"),
        gr.Slider(10, 60, value=30, step=1, label="Inference Steps"),
        gr.Slider(1.0, 15.0, value=9.0, step=0.5, label="Guidance Scale"),
        gr.Number(label="Seed (set -1 for random)", value=1234),
    ],
    outputs=[
        gr.Video(label="Generated Video (MP4)"),
        gr.File(label="Metadata JSON"),
    ],
    title="Text-to-Video (ModelScope 1.7B)",
    description="Generate short videos from text prompts. If you hit OOM, reduce frames/steps.",
)

PORT = int(os.environ.get("PORT", 7861))
demo.launch(server_name="0.0.0.0", server_port=PORT)
