# Text-to-Video Generation using Diffusion Models

This project implements an end-to-end Text-to-Video generation system using a pretrained video diffusion model from ModelScope, built with Hugging Face Diffusers and an interactive **Gradio web interface**.

Users can enter a natural language prompt and generate short **MP4 videos**, with all generation parameters saved as metadata for reproducibility.

---

## 🚀 Features

- Text → Video generation using a pretrained **ModelScope 1.7B video diffusion model**
- Interactive **Gradio-based web UI**
- Adjustable generation parameters:
  - Number of frames
  - Diffusion steps
  - Guidance scale
  - FPS
  - Random seed
- Automatic export of:
  - MP4 video output
  - JSON metadata (prompt + parameters)
- GPU-optimized inference using FP16

---

## 🧠 Model Overview

- **Base Model:** ModelScope Text-to-Video (1.7B parameters)
- **Model Type:** Video Diffusion Model
- **Framework:** Hugging Face Diffusers
- **Precision:** FP16
- **Execution:** GPU-based inference

> ⚠️ Video diffusion models are computationally expensive and require GPU acceleration for practical runtimes.

---

## 📂 Project Structure

```text
text-to-video-generation/
├── app.py              # Gradio application
├── requirements.txt    # Python dependencies
├── README.md
├── assets/             # Sample outputs & screenshots
│   ├── Cyberpunk_Rainy_Alley.mp4
│   ├── Cyberpunk_Rainy_Alley.json
│   ├── Neon_City.mp4
│   ├── Shooting_Stars.mp4
│   └── UI_Screenshot.png
└── outputs/            # Generated at runtime

```


## ▶️ Running the App
### Option 1: Google Colab (Recommended for quick demo)

1. Open a Google Colab notebook
2. Set:
   Runtime → Change runtime type → GPU
3. Install dependencies:
   pip install -r requirements.txt
4. Run the application:
   python app.py
5. Open the generated https://*.gradio.live URL

### Option 2: Local Machine (NVIDIA GPU required)

- Requirements:
  - NVIDIA GPU with CUDA support
  - Python 3.10+
  - CUDA-enabled PyTorch:
     pip install -r requirements.txt
     python app.py
  - Access the app at:
     [https://c749c09e309f9e4cab.gradio.live](https://798108d7dc3e09563b.gradio.live)

## 🎛 Recommended Generation Settings

- Parameter	Recommended Value:
  - Frames	16–24
  - Steps	20–30
  - Guidance Scale	7–9
  - FPS	6–8
     
- Negative Prompt (Recommended)
  - blurry, low quality, distorted, artifacts, flicker, jitter

## 🖼 Sample Outputs

- Generated examples are available in the assets/ folder:
  - Cyberpunk_Rainy_Alley.mp4
  - Neon_City.mp4
  - Shooting_Stars.mp4

- Each video includes a corresponding .json file containing:
  - prompt
  - seed
  - steps
  - frames
  - guidance scale
  - runtime

## ⚠️ Limitations

- Video generation is slower than image diffusion
- High VRAM usage limits concurrency
- Model weights are large and downloaded on first run
- CPU-only execution is not practical

## 🔮 Future Improvements

- LoRA fine-tuning for custom styles or domains
- Progress indicators per frame
- Authentication and rate limiting
- Video upscaling and post-processing
- Deployment on Hugging Face Spaces

## 📌 Key Takeaways

- This project demonstrates:
  - Practical use of video diffusion models
  - GPU-optimized machine learning inference
  - Full-stack ML workflow (model → inference → UI)
  - Reproducibility through metadata logging

## 📜 License & Attribution

- This project uses pretrained weights released by ModelScope.
- Please refer to the original model license for usage restrictions.
