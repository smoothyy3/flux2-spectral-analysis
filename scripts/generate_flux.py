"""
Generate 200 FLUX face portraits for spectral analysis.

20 diverse prompts x 10 seeds each = 200 images.
All outputs are 1024x1024 PNGs saved to data/generated/.
"""

import torch
from pathlib import Path
from diffusers import Flux2KleinPipeline
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated" / "klein_4b_distilled"
RESOLUTION = 1024
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 1.0

# Framing suffix applied to every prompt for consistent FFHQ-like composition
FRAMING = (
    "close-up portrait photograph, head and shoulders, face centered, "
    "sharp focus, 1024x1024, photorealistic"
)

# 20 diverse face prompts covering age, gender, ethnicity, lighting, expression
PROMPTS = [
    # Varied angles and gazes
    "young woman looking slightly to the left, three-quarter view, candid outdoor photo",
    "man in his 30s looking down with a slight smile, natural light, shallow depth of field",
    "elderly woman photographed from below, looking into the distance, overcast sky",
    "teenage boy looking directly at camera, slightly tilted head, indoor ambient light",
    "woman in her 40s in profile view, wind in her hair, golden hour backlight",
    
    # Candid expressions and situations  
    "man laughing with mouth open, caught mid-conversation, blurred restaurant background",
    "young woman squinting in bright sunlight, messy hair, beach setting",
    "middle-aged man with tired expression, fluorescent office lighting, close crop",
    "woman surprised expression, slightly blurry, indoor party lighting",
    "elderly man mid-speech, animated expression, outdoor crowd in background",
    
    # Accessories and occlusion (common in FFHQ)
    "young man wearing sunglasses, urban street background, harsh midday shadows",
    "woman in her 30s wearing a winter hat, rosy cheeks, cold weather outdoor photo",
    "man with baseball cap, slight stubble, casual selfie angle, natural daylight",
    "young woman with large earrings and makeup, nighttime photo, warm artificial light",
    "elderly man wearing reading glasses on nose tip, looking over them, indoor light",
    
    # Diverse settings and quality (FFHQ has varied photo quality)
    "child around 10 years old, gap-toothed smile, school photo with plain background",
    "woman in her 50s, dark skin, photographed through window glass, soft reflections",
    "young man with beard, selfie in bathroom mirror, mixed lighting",
    "middle-aged woman at desk, webcam-like angle from slightly above, screen glow on face",
    "man in his 60s at outdoor cafe, dappled tree shade on face, background bokeh",
]

# 10 seeds per prompt — spread out for variety
SEEDS = [100, 247, 389, 512, 671, 803, 945, 1087, 1234, 1500]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assert len(PROMPTS) == 20, f"Expected 20 prompts, got {len(PROMPTS)}"
    assert len(SEEDS) == 10, f"Expected 10 seeds, got {len(SEEDS)}"

    print(f"Will generate {len(PROMPTS) * len(SEEDS)} images")
    print(f"Output directory: {OUTPUT_DIR}")

    # Load FLUX pipeline
    print("Loading FLUX model...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

    img_idx = 0
    generator = torch.Generator(device="cuda")
    for prompt_idx, prompt in enumerate(PROMPTS):
        full_prompt = f"{prompt}, {FRAMING}"
        print(f"\nPrompt {prompt_idx + 1}/20: {prompt[:60]}...")

        for seed_idx, seed in enumerate(tqdm(SEEDS, desc=f"  Seeds", leave=False)):
            generator.manual_seed(seed)
            image = pipe(
                prompt=full_prompt,
                height=RESOLUTION,
                width=RESOLUTION,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            ).images[0]

            filename = f"gen_{img_idx:04d}_p{prompt_idx:02d}_s{seed}.png"
            image.save(OUTPUT_DIR / filename, format="PNG")
            img_idx += 1

    print(f"\nDone! Generated {img_idx} images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
