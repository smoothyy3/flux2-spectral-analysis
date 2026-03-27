"""
Generate ablation grid images for Klein 4B Base guidance × steps experiment.

Condition A (steps=50, guidance=4.0) already exists at data/generated/klein_4b_base/.
This script generates conditions B–F using the same Klein 4B Base model.

Grid:
  B  steps=50  guidance=1.0  → klein_base_g1_s50/   (isolates guidance)
  C  steps=4   guidance=4.0  → klein_base_g4_s4/    (isolates step count)
  D  steps=4   guidance=1.0  → klein_base_g1_s4/    (matches distilled settings)
  E  steps=10  guidance=1.0  → klein_base_g1_s10/   (step-count curve)
  F  steps=20  guidance=1.0  → klein_base_g1_s20/   (step-count curve)

100 images per condition: 20 prompts × 5 seeds (seeds 0–4).

Usage:
    python scripts/generate_grid.py --condition B
    python scripts/generate_grid.py --condition all
    python scripts/generate_grid.py --condition all --no-cpu-offload  # Colab A100
    python scripts/generate_grid.py --condition B,E,F --output-base /content/drive/MyDrive/flux_generated
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------
GRID: dict[str, dict] = {
    "B": {"steps": 50, "guidance": 1.0, "dir": "klein_base_g1_s50"},
    "C": {"steps": 4,  "guidance": 4.0, "dir": "klein_base_g4_s4"},
    "D": {"steps": 4,  "guidance": 1.0, "dir": "klein_base_g1_s4"},
    "E": {"steps": 10, "guidance": 1.0, "dir": "klein_base_g1_s10"},
    "F": {"steps": 20, "guidance": 1.0, "dir": "klein_base_g1_s20"},
}

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
RESOLUTION = 1024

FRAMING = "portrait photograph, face visible, sharp focus, 1024x1024, photorealistic"

PROMPTS = [
    "young woman looking slightly to the left, three-quarter view, candid outdoor photo",
    "man in his 30s looking down with a slight smile, natural light, shallow depth of field",
    "elderly woman photographed from below, looking into the distance, overcast sky",
    "teenage boy looking directly at camera, slightly tilted head, indoor ambient light",
    "woman in her 40s in profile view, wind in her hair, golden hour backlight",
    "man laughing with mouth open, caught mid-conversation, blurred restaurant background",
    "young woman squinting in bright sunlight, messy hair, beach setting",
    "middle-aged man with tired expression, fluorescent office lighting, close crop",
    "woman surprised expression, slightly blurry, indoor party lighting",
    "elderly man mid-speech, animated expression, outdoor crowd in background",
    "young man wearing sunglasses, urban street background, harsh midday shadows",
    "woman in her 30s wearing a winter hat, rosy cheeks, cold weather outdoor photo",
    "man with baseball cap, slight stubble, casual selfie angle, natural daylight",
    "young woman with large earrings and makeup, nighttime photo, warm artificial light",
    "elderly man wearing reading glasses on nose tip, looking over them, indoor light",
    "child around 10 years old, gap-toothed smile, school photo with plain background",
    "woman in her 50s, dark skin, photographed through window glass, soft reflections",
    "young man with beard, selfie in bathroom mirror, mixed lighting",
    "middle-aged woman at desk, webcam-like angle from slightly above, screen glow on face",
    "man in his 60s at outdoor cafe, dappled tree shade on face, background bokeh",
]

SEEDS = list(range(5))  # 0, 1, 2, 3, 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ablation grid images.")
    parser.add_argument(
        "--condition",
        default="all",
        help='Condition(s) to generate: B, C, D, E, F, "all", or comma-separated e.g. B,E,F',
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Base directory for output. Defaults to data/generated/ inside the repo.",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Load full model onto GPU (use on Colab A100 or large-VRAM GPUs).",
    )
    return parser.parse_args()


def resolve_conditions(condition_arg: str) -> list[str]:
    """Parse --condition argument to a list of condition IDs."""
    if condition_arg.strip().lower() == "all":
        return list(GRID.keys())
    ids = [c.strip().upper() for c in condition_arg.split(",")]
    unknown = [c for c in ids if c not in GRID]
    if unknown:
        raise ValueError(f"Unknown condition(s): {unknown}. Valid: {list(GRID.keys())}")
    return ids


def load_pipeline(cpu_offload: bool) -> Flux2KleinPipeline:
    """Load Klein 4B Base pipeline onto GPU."""
    assert torch.cuda.is_available(), "CUDA GPU required."
    props = torch.cuda.get_device_properties(0)
    print(f"GPU  : {props.name}")
    print(f"VRAM : {props.total_memory / 1e9:.1f} GB")
    print(f"Loading {MODEL_ID} ...")

    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
        print("CPU offload: enabled")
    else:
        pipe = pipe.to("cuda")
        print("CPU offload: disabled (full GPU)")
    return pipe


def generate_condition(
    pipe: Flux2KleinPipeline,
    condition_id: str,
    output_base: Path,
) -> None:
    """Generate all 100 images for one grid condition."""
    cond = GRID[condition_id]
    steps: int = cond["steps"]
    guidance: float = cond["guidance"]
    out_dir = output_base / cond["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS) * len(SEEDS)
    existing = len(list(out_dir.glob("*.png")))
    print(f"\nCondition {condition_id}: steps={steps} guidance={guidance}")
    print(f"  Output : {out_dir}")
    print(f"  Target : {total} images | Already done: {existing}")

    generated = 0
    skipped = 0
    generator = torch.Generator(device="cuda")

    with tqdm(total=total, desc=f"  Cond {condition_id}", unit="img") as pbar:
        for prompt_idx, prompt in enumerate(PROMPTS):
            full_prompt = f"{prompt}, {FRAMING}"
            for seed in SEEDS:
                filename = f"{prompt_idx:02d}_s{seed}.png"
                out_path = out_dir / filename

                if out_path.exists():
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix(skip=skipped, gen=generated)
                    continue

                t0 = time.perf_counter()
                generator.manual_seed(seed)
                image = pipe(
                    prompt=full_prompt,
                    height=RESOLUTION,
                    width=RESOLUTION,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                ).images[0]
                elapsed = time.perf_counter() - t0

                image.save(out_path, format="PNG")
                generated += 1
                pbar.update(1)
                pbar.set_postfix(skip=skipped, gen=generated, t=f"{elapsed:.1f}s")

    print(f"  Done — generated {generated}, skipped {skipped}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    conditions = resolve_conditions(args.condition)

    repo_root = Path(__file__).parent.parent
    output_base = Path(args.output_base) if args.output_base else repo_root / "data" / "generated"

    print(f"Conditions : {conditions}")
    print(f"Output base: {output_base}")
    print(f"Images/cond: {len(PROMPTS) * len(SEEDS)}")

    pipe = load_pipeline(cpu_offload=not args.no_cpu_offload)

    for cond_id in conditions:
        generate_condition(pipe, cond_id, output_base)

    print(f"\nAll done. Generated conditions: {conditions}")


if __name__ == "__main__":
    main()
