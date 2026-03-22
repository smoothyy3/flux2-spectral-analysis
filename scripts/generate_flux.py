"""
Generate 200 FLUX face portraits for spectral analysis.

20 diverse prompts x 10 seeds each = 200 images.
All outputs are 1024x1024 PNGs saved to data/generated/.
"""

import torch
from pathlib import Path
from diffusers import FluxPipeline
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "generated"
RESOLUTION = 1024
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 3.5

# Framing suffix applied to every prompt for consistent FFHQ-like composition
FRAMING = (
    "close-up portrait photograph, head and shoulders, face centered, "
    "sharp focus, 1024x1024, photorealistic"
)

# 20 diverse face prompts covering age, gender, ethnicity, lighting, expression
PROMPTS = [
    "young woman in her 20s with long dark hair, neutral expression, studio lighting",
    "elderly man in his 70s with white beard and wrinkles, warm smile, soft natural light",
    "teenage boy around 16 with freckles and messy brown hair, slight grin, overcast daylight",
    "middle-aged woman in her 40s with short blonde hair, confident expression, ring light",
    "young man with glasses and neat short hair, serious expression, office lighting",
    "elderly woman in her 80s with silver hair pulled back, gentle smile, window light",
    "man in his 30s with thick dark beard and brown eyes, neutral expression, golden hour light",
    "young woman with dark skin and natural afro hair, joyful smile, bright studio lighting",
    "East Asian man in his 50s with salt-and-pepper hair, calm expression, diffused light",
    "South Asian woman in her 30s with long black hair, subtle smile, natural daylight",
    "young man with light skin and red hair, freckled face, neutral expression, cloudy sky background",
    "middle-aged man in his 50s with bald head and goatee, stern expression, dramatic side lighting",
    "woman in her 20s with hijab, warm brown eyes, peaceful expression, soft studio light",
    "Latino man in his 40s with mustache, friendly smile, outdoor natural light",
    "teenage girl around 15 with braided hair and braces, cheerful grin, school portrait lighting",
    "man in his 60s with deep wrinkles and kind eyes, weathered face, harsh sunlight",
    "young woman with very short pixie cut and pale skin, intense gaze, moody low-key lighting",
    "middle-aged woman with curly gray hair and reading glasses, thoughtful expression, warm interior light",
    "young man in his 20s with dark skin and short fade haircut, confident look, neon-tinted light",
    "woman in her 70s with East Asian features, laugh lines, gentle expression, overcast soft light",
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
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B",
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    img_idx = 0
    for prompt_idx, prompt in enumerate(PROMPTS):
        full_prompt = f"{prompt}, {FRAMING}"
        print(f"\nPrompt {prompt_idx + 1}/20: {prompt[:60]}...")

        for seed_idx, seed in enumerate(tqdm(SEEDS, desc=f"  Seeds", leave=False)):
            generator = torch.Generator(device="cpu").manual_seed(seed)

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
