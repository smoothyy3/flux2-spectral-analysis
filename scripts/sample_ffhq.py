"""
Sample 200 random images from FFHQ dataset via HuggingFace.

Uses seed 42 for reproducibility. Streams images without downloading the
full dataset and saves them as 1024x1024 PNGs to data/real/.
"""

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

    print("Loading FFHQ dataset from HuggingFace (streaming)...")
    dataset = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)

    # Shuffle with seed and take N_SAMPLES — no full download needed
    dataset = dataset.shuffle(seed=SEED, buffer_size=10_000).take(N_SAMPLES)

    print(f"Sampling {N_SAMPLES} images with seed {SEED}...")

    for i, item in enumerate(tqdm(dataset, total=N_SAMPLES, desc="Saving FFHQ samples")):
        img = item["image"]

        # Ensure 1024x1024 PNG
        if img.size != (RESOLUTION, RESOLUTION):
            img = img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

        img.save(OUTPUT_DIR / f"real_{i:04d}.png", format="PNG")

    print(f"Saved {N_SAMPLES} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
