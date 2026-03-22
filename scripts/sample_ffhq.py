"""
Sample 200 random images from FFHQ dataset via HuggingFace.

Uses numpy random seed 42 for reproducibility. Downloads images and saves
them as 1024x1024 PNGs to data/real/.
"""

import numpy as np
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

SEED = 42
N_SAMPLES = 200
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "real"
RESOLUTION = 1024


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FFHQ dataset from HuggingFace...")
    dataset = load_dataset("merkol/ffhq", split="train")

    n_total = len(dataset)
    print(f"Dataset has {n_total} images.")

    # Reproducible random sample
    rng = np.random.default_rng(SEED)
    indices = rng.choice(n_total, size=N_SAMPLES, replace=False)
    indices.sort()

    print(f"Sampling {N_SAMPLES} images with seed {SEED}...")
    print(f"Selected indices: {indices[:10]}... (showing first 10)")

    for i, idx in enumerate(tqdm(indices, desc="Saving FFHQ samples")):
        item = dataset[int(idx)]
        img = item["image"]

        # Ensure 1024x1024 PNG
        if img.size != (RESOLUTION, RESOLUTION):
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        img.save(OUTPUT_DIR / f"real_{i:04d}.png", format="PNG")

    print(f"Saved {N_SAMPLES} images to {OUTPUT_DIR}")

    # Save the indices for reproducibility
    np.save(OUTPUT_DIR.parent / "ffhq_sample_indices.npy", indices)
    print("Saved sample indices to data/ffhq_sample_indices.npy")


if __name__ == "__main__":
    main()
