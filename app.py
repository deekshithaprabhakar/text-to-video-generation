import os, json, time, random
import torch
import imageio
import gradio as gr
from diffusers import DiffusionPipeline

MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def load_pipe():
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Reduce VRAM usage (important for Colab T4)
    pipe.enable_model_cpu_offload()

    # Safe memory tweaks
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    return pipe

pipe = load_pipe()

def generate_video(
    prompt: str,
    negative_prompt: str,
    num_frames: int,
    fps: int,
    steps: int,
    guidance: float,
    seed: int,
):
    if seed is None or seed < 0:
        seed = random.randint(0, 2**31 - 1)

    generator = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu"
    ).manual_seed(int(seed))

    t0 = time.time()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        num_frames=int(num_frames),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )
    frames = result.frames[0]  # list[PIL.Image]
    runtime_s = time.time() - t0

    run_id = f"t2v_{int(time.time())}_{seed}"
    mp4_path = os.path.join(OUT_DIR, f"{run_id}.mp4")
    meta_path = os.path.join(OUT_DIR, f"{run_id}.json")

    imageio.mimsave(mp4_path, frames, fps=int(fps))

    meta = {
        "model_id": MODEL_ID,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_frames": int(num_frames),
        "fps": int(fps),
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance),
        "seed": int(seed),
        "seconds_runtime": round(runtime_s, 2),
        "output_mp4": mp4_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return mp4_path, meta_path, f"Done ✅ Runtime: {round(runtime_s, 2)}s"

with gr.Blocks() as demo:
    gr.Markdown("# Text-to-Video (ModelScope 1.7B) — Gradio Demo")
    gr.Markdown("Generate short videos from text prompts using a pretrained video diffusion model. "
                "If you hit OOM, reduce frames/steps.")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                lines=3,
                value="A slow cinematic camera pan through a neon cyberpunk alley at night, rain falling, reflections on wet pavement, fog drifting",
            )
            negative = gr.Textbox(
                label="Negative Prompt",
                lines=2,
                value="blurry, low quality, distorted, artifacts, flicker, jitter",
            )
            num_frames = gr.Slider(8, 60, value=16, step=1, label="Num Frames")
            fps = gr.Slider(4, 24, value=8, step=1, label="FPS")
            steps = gr.Slider(10, 60, value=20, step=1, label="Inference Steps")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance Scale")
            seed = gr.Number(label="Seed (set -1 for random)", value=1234)

            btn = gr.Button("Generate")

        with gr.Column():
            out_video = gr.Video(label="Generated Video (MP4)")
            out_meta = gr.File(label="Metadata JSON")
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(
        fn=generate_video,
        inputs=[prompt, negative, num_frames, fps, steps, guidance, seed],
        outputs=[out_video, out_meta, status],
    )

if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=PORT)
