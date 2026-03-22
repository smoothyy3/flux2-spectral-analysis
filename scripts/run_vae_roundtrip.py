"""
VAE round-trip experiment for spectral analysis.

Encodes 50 real FFHQ images through the FLUX.2 VAE encoder, decodes them
back, and compares the radial power spectra of the originals vs. the
reconstructions.  This isolates the VAE decoder from the flow transformer
and prompt conditioning entirely.

Usage
-----
    python scripts/run_vae_roundtrip.py

Outputs
-------
    data/generated/vae_roundtrip/          50 reconstructed PNG images
    results/figures/vae_roundtrip_comparison.png   overlaid mean spectra
    results/figures/vae_roundtrip_difference.png   signed difference plot

Interpretation
--------------
    ~0 significant freq. bins  → VAE preserves spectrum; artifact is in
                                 the flow transformer / training distribution
    Many sig. bins, +diff      → VAE over-produces HF; decoder is the cause
    Many sig. bins, -diff      → VAE smooths/blurs; decoder suppresses HF
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Ensure src/ is importable when run as a top-level script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.spectral.azimuthal import azimuthal_average
from src.spectral.fft import compute_power_spectrum_2d
from src.spectral.statistics import per_frequency_ttest, population_stats
from src.visualization.difference import plot_spectral_difference
from src.visualization.spectra import plot_mean_spectra

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_IMAGES: int = 50
REAL_DIR: Path = ROOT / "data" / "real"
OUTPUT_DIR: Path = ROOT / "data" / "generated" / "vae_roundtrip"
FIGURES_DIR: Path = ROOT / "results" / "figures"
RESOLUTION: int = 1024
MODEL_ID: str = "black-forest-labs/FLUX.2-klein-4B"

# Floor applied before log10 — must match the value used in spectra.py and
# difference.py so that summary statistics are directly comparable.
_LOG_FLOOR: float = 1.0


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------


def load_vae() -> torch.nn.Module:
    """Load only the FLUX.2 VAE component, freeing everything else.

    Tries the lightweight subfolder route first.  Falls back to loading the
    full ``Flux2KleinPipeline`` and extracting the VAE, then explicitly
    deletes all other components and empties the CUDA cache.

    Returns
    -------
    torch.nn.Module
        The AutoencoderKL in eval mode on CUDA, dtype bfloat16.
    """
    # --- Attempt 1: load just the VAE via subfolder -------------------------
    try:
        from diffusers import AutoencoderKL

        print("Loading VAE via subfolder route ...")
        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        )
        vae = vae.to("cuda")
        vae.eval()
        print("  VAE loaded (subfolder route).")
        return vae
    except Exception as exc:
        print(f"  Subfolder route failed ({exc}).")
        print("  Falling back to full pipeline ...")

    # --- Attempt 2: load full pipeline, extract VAE, discard the rest -------
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    vae = pipe.vae.to("cuda")
    vae.eval()

    _OTHER_COMPONENTS = (
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "scheduler",
        "feature_extractor",
        "image_encoder",
    )
    for attr in _OTHER_COMPONENTS:
        if hasattr(pipe, attr):
            delattr(pipe, attr)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print("  VAE loaded (full-pipeline route; other components freed).")
    return vae


# ---------------------------------------------------------------------------
# Tensor ↔ PIL conversion
# ---------------------------------------------------------------------------


def pil_to_bfloat16_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a normalised bfloat16 CUDA tensor.

    Converts to float32 in [0, 1], normalises to [-1, 1], then casts to
    bfloat16 and moves to CUDA.

    Parameters
    ----------
    img : PIL.Image.Image
        RGB image at the target resolution.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 3, H, W)``, dtype bfloat16, device ``cuda``.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # [0, 1]
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.to(device="cuda", dtype=torch.bfloat16)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a decoded VAE output tensor back to a PIL RGB image.

    Clamps to [-1, 1], rescales to [0, 255], and converts to uint8.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape ``(1, 3, H, W)`` or ``(3, H, W)``, values nominally in [-1, 1].

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """
    t = tensor.squeeze(0).float().clamp(-1.0, 1.0)
    arr = ((t + 1.0) / 2.0 * 255.0).byte().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------


def image_to_radial_spectrum(img: Image.Image) -> np.ndarray:
    """Compute the 1-D radially averaged power spectrum of a PIL image.

    Converts to grayscale (luminance), normalises to [0, 1], computes the
    2-D power spectrum via FFT, and returns the azimuthal average.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (any mode, any size).

    Returns
    -------
    np.ndarray
        1-D radially averaged power spectrum, dtype float64, length
        ``min(H, W) // 2 + 1``.
    """
    gray = np.array(img.convert("L"), dtype=np.float64) / 255.0
    power_2d = compute_power_spectrum_2d(gray)
    return azimuthal_average(power_2d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Collect real image paths -------------------------------------------
    real_paths = sorted(
        p
        for p in REAL_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )[:N_IMAGES]

    if not real_paths:
        raise FileNotFoundError(f"No images found in {REAL_DIR}")
    if len(real_paths) < N_IMAGES:
        print(
            f"Warning: only {len(real_paths)} images available "
            f"(requested {N_IMAGES})."
        )

    print("VAE Round-Trip Experiment")
    print(f"  Images : {len(real_paths)}")
    print(f"  Source : {REAL_DIR}")
    print(f"  Output : {OUTPUT_DIR}")
    print(f"  Figures: {FIGURES_DIR}")

    # ---- Load VAE -----------------------------------------------------------
    vae = load_vae()

    # ---- Encode → decode loop ----------------------------------------------
    orig_spectra: list[np.ndarray] = []
    recon_spectra: list[np.ndarray] = []

    print(f"\nProcessing {len(real_paths)} images through VAE round-trip ...")
    with torch.inference_mode():
        for i, img_path in enumerate(
            tqdm(real_paths, desc="VAE round-trip", unit="img")
        ):
            # Load and ensure correct resolution
            orig_img = Image.open(img_path).convert("RGB")
            if orig_img.size != (RESOLUTION, RESOLUTION):
                orig_img = orig_img.resize((RESOLUTION, RESOLUTION), Image.LANCZOS)

            # Encode → sample from latent distribution → decode
            x = pil_to_bfloat16_tensor(orig_img)
            latent = vae.encode(x).latent_dist.sample()
            recon_tensor = vae.decode(latent).sample
            recon_img = tensor_to_pil(recon_tensor)

            # Persist reconstruction
            recon_img.save(OUTPUT_DIR / f"vae_recon_{i:04d}.png", format="PNG")

            # Accumulate spectra
            orig_spectra.append(image_to_radial_spectrum(orig_img))
            recon_spectra.append(image_to_radial_spectrum(recon_img))

    print(f"  Saved {len(real_paths)} reconstructions to {OUTPUT_DIR}")

    # ---- Population statistics ----------------------------------------------
    orig_arr = np.stack(orig_spectra, axis=0)   # (N, R)
    recon_arr = np.stack(recon_spectra, axis=0)  # (N, R)

    orig_mean, orig_std = population_stats(orig_arr)
    recon_mean, recon_std = population_stats(recon_arr)

    # Independent-samples t-test per frequency bin.
    # (Paired would be tighter, but per_frequency_ttest already exists and
    # is sufficient for detecting systematic VAE-introduced differences.)
    _, p_values = per_frequency_ttest(orig_arr, recon_arr)

    # ---- Figures ------------------------------------------------------------
    print("\nGenerating figures ...")

    overlay_path = FIGURES_DIR / "vae_roundtrip_comparison.png"
    plot_mean_spectra(
        group_spectra={
            "real (original)": (orig_mean, orig_std),
            "vae_roundtrip": (recon_mean, recon_std),
        },
        output_path=overlay_path,
        title="VAE Round-Trip: Original vs. Reconstruction — Mean Radial Power Spectra",
    )
    print(f"  Saved: {overlay_path.name}")

    diff_path = FIGURES_DIR / "vae_roundtrip_difference.png"
    plot_spectral_difference(
        real_mean=orig_mean,
        gen_mean=recon_mean,
        p_values=p_values,
        output_path=diff_path,
        group_name="vae_roundtrip",
    )
    print(f"  Saved: {diff_path.name}")

    # ---- Console summary ----------------------------------------------------
    log_orig = np.log10(np.maximum(orig_mean, _LOG_FLOOR))
    log_recon = np.log10(np.maximum(recon_mean, _LOG_FLOOR))
    log_diff = log_recon - log_orig  # positive → VAE over-produces energy

    n_sig = int((p_values < 0.05).sum())
    mean_abs_diff = float(np.mean(np.abs(log_diff[1:])))  # skip DC bin (index 0)
    max_over_bin = int(log_diff.argmax())
    max_under_bin = int(log_diff.argmin())

    print(f"\n{'=' * 60}")
    print("VAE Round-Trip Spectral Summary")
    print(f"{'=' * 60}")
    print(f"  Images processed         : {len(real_paths)}")
    print(f"  Frequency bins           : {len(orig_mean)}")
    print(f"  Significant bins (p<0.05): {n_sig} / {len(p_values)}")
    print(f"  Mean |Δlog₁₀(power)|     : {mean_abs_diff:.4f}  (excl. DC)")
    print(
        f"  Max over-production      : {log_diff[max_over_bin]:+.4f} "
        f"at bin {max_over_bin}"
    )
    print(
        f"  Max under-production     : {log_diff[max_under_bin]:+.4f} "
        f"at bin {max_under_bin}"
    )
    print(f"{'=' * 60}")
    print("\nInterpretation guide:")
    print(
        "  ~0 significant bins  → VAE preserves spectrum; "
        "artifact is in the flow transformer / training distribution"
    )
    print(
        "  Many sig. bins, +diff → VAE over-produces HF; "
        "decoder is the cause"
    )
    print(
        "  Many sig. bins, −diff → VAE smooths / blurs; "
        "decoder suppresses HF"
    )


if __name__ == "__main__":
    main()
