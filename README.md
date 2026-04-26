# Spectral Forensic Characterization of FLUX.2 Image Generation Models

This project applies azimuthal spectral averaging (Keuper et al., CVPR 2020) to FLUX.2 image generation models, comparing the radial power spectra of generated face images against 200 real FFHQ photographs. The artifact is **opposite in direction to GANs**: FLUX produces a shallower log-log slope than real images (excess high-frequency power), whereas GAN-era models under-produced high frequencies. The LADD-trained distilled model sits closest to real (Δslope +0.072), while the MSE base model deviates more at any tested inference setting (+0.172 to +0.801). The fingerprint is highly consistent per-image despite being small in absolute magnitude — logistic regression AUC reaches 0.995 on clean same-domain data.

![2D Power Spectrum Comparison](results/controls/figures/2d_spectrum_comparison.png)
*2D log-power spectra (|F(u,v)|², DC-centered) for real FFHQ images and three FLUX.2 model variants. The isotropic falloff confirms no directional grid artifacts.*

---

## Motivation

Keuper et al. (CVPR 2020) showed that GAN-based generators fail to reproduce the spectral distributions of real images. The artifact was characteristic: massive under-production of high-frequency content (Δslope ≈ −1 to −3), attributable to transposed convolution checkerboard patterns. Modern rectified flow transformers (FLUX.2) use a fundamentally different architecture: patch-based vision transformers with rotary positional embeddings and a flow matching training objective. This project asks whether these models exhibit analogous spectral artifacts, and if so, how they differ from the GAN case.

---

## Methodology

**Spectral pipeline (replicating Keuper et al.):**
1. Convert images to grayscale (luminance channel)
2. Compute 2D DFT via `np.fft.fft2`, shift DC to center with `np.fft.fftshift`
3. Apply `log10(|F(u,v)|² + ε)` per pixel before any averaging — this is the Keuper order; averaging log-power per radial bin rather than taking the log of the averaged linear power
4. Azimuthal average: for each integer radius r, mean of all log-power values at distance r from center. Produces a 1D spectrum of length 513 for 1024×1024 images
5. Population statistics: mean and standard deviation across N images per group
6. Spectral slope: log-log linear fit restricted to bins 10–400 (excludes DC and noisy near-Nyquist edge)

**Reference dataset:** 200 randomly sampled FFHQ images at 1024×1024

**Models tested:**
- Klein 4B Distilled — LADD adversarial distillation, 4 steps, guidance=1.0 (n=200)
- Klein 4B Base — MSE flow matching, 50 steps, guidance=4.0 (n=41)
- FLUX.2 Max — BFL Playground, inference parameters unknown (n=20)

**Control experiments:**
- Degradation controls: Gaussian blur (σ=1.5), 2× downscale/upscale (512→1024), JPEG Q85 applied to real images
- VAE round-trip: 20 real images encoded and decoded through the FLUX VAE to isolate the decoder's contribution

**Ablation:** Guidance × steps grid using Klein Base at varying (steps, guidance) combinations to disentangle inference parameters from training objective effects.

---

## Results

### Spectral Fingerprint

FLUX-generated images exhibit a spectral fingerprint **opposite in direction** to the GAN artifact documented by Keuper et al. GANs produce Δslope of −1 to −3 (massive HF under-production). FLUX produces positive Δslopes of +0.07 to +0.33 — slopes shallower than real, meaning **excess high-frequency power**. The effect magnitude is roughly two orders of magnitude smaller than GAN-era artifacts.

The signed-difference plots show a consistent W-shaped pattern: mild over-production at low bins (~0–80), significant under-production in the mid-band (~100–230), and monotonically growing over-production from bin ~240 to Nyquist (peak ≈ +0.3–0.6 log₁₀ power). Band energies confirm: high-band mean log-power gen vs. real is 2.007 vs. 1.598 (distilled) and 2.081 vs. 1.598 (FLUX.2 Max). The fingerprint is isotropic; no directional grid artifacts are present.

| Model | n | Δslope | KL | Wass | LR AUC (OOF) |
|---|---|---|---|---|---|
| Klein 4B Distilled (LADD, g=1, s=4) | 200 | **+0.072** | 1.52e-4 | 4.23e-3 | 0.995 |
| Klein 4B Base (MSE, g=4, s=50) | 41 | **+0.172** | 4.42e-4 | 5.11e-3 | 0.984 |
| FLUX.2 Max | 20 | **+0.333** | 3.38e-4 | 2.57e-3 | — |

*Real image slope: −2.978. All Δslopes are gen_slope − real_slope.*

Despite KL/Wasserstein distances of 10⁻⁴–10⁻³, LR AUC saturates at 0.98–1.00, meaning the fingerprint is **per-image consistent**: a small linear classifier on the raw 1D radial spectrum separates real from FLUX with near-perfect reliability within this evaluation set.

### VAE Isolation

The VAE round-trip experiment (n=20) produces **0/513 statistically significant frequency bins** (p < 0.05, paired t-test). Mean absolute log-power deviation: 0.0154. A small slope drift (+0.015) is only detectable past bin ~450 and is two orders of magnitude smaller than the full-generation artifact. The VAE decoder introduces no detectable spectral shift at this sample size; the fingerprint originates in the **flow-matching denoising process**.

![VAE Round-trip Spectral Difference](results/vae_roundtrip/figures/difference.png)
*Encoding and decoding 20 real images through the FLUX VAE produces no statistically significant spectral deviation. The difference signal is indistinguishable from noise across all 513 frequency bins.*

### Degradation Controls

| Condition | gen/deg slope | Δ slope vs. real |
|---|---|---|
| JPEG Q85 | −3.01 | −0.03 |
| Downscale 512→1024 | −4.01 | −1.03 |
| Gaussian blur σ=1.5 | −5.14 | −2.16 |
| Klein Distilled | −2.907 | **+0.072** |
| Klein Base g=4.0 s=50 | −2.807 | **+0.172** |
| FLUX.2 Max | −2.646 | **+0.333** |

*Real image slope: −2.978.*

All three degradations steepen the slope (remove high-frequency content). FLUX generation shallows the slope (adds high-frequency content relative to real). These are opposite effects: the FLUX fingerprint cannot be explained by any classical degradation artifact. Notably, JPEG Q85 alone shifts the slope by −0.03, which is comparable in magnitude to the distilled model's Δslope of +0.072 — the two effects partially cancel, making the fingerprint fragile to a single re-encode.

### Guidance × Steps Ablation

| Condition | Training | Steps | Guidance | n | Δ Slope |
|---|---|---|---|---|---|
| Distilled | LADD | 4 | 1.0 | 200 | +0.072 |
| Base A | MSE | 50 | 4.0 | 41 | +0.172 |
| Base B | MSE | 50 | 1.0 | 19 | +0.352 |
| Base D | MSE | 4 | 1.0 | 56 | +0.801 |

**More inference steps reduce the artifact.** At g=1.0, increasing steps from 4 to 50 reduces Δslope from +0.801 to +0.352 (D→B). Under-converged sampling injects excess HF noise.

**Higher guidance also reduces the artifact.** At s=50, raising guidance from 1.0 to 4.0 reduces Δslope from +0.352 to +0.172 (B→A). Higher CFG likely flattens high-frequency stochastic content. Confound: g=4.0 also introduces visible oversharpening.

**LADD training is the strongest candidate for spectral realism.** At matched inference settings (g=1.0, s=4), distilled Δ=+0.072 vs. base Δ=+0.801 — an **11× reduction**. However, this comparison is confounded: the base model produces visibly degraded images at 4 steps (it was trained for 50+ steps with MSE), so training objective and image quality degradation cannot be cleanly separated with the available models.

![Grid Spectral Difference Panel](results/grid_ablation/figures/grid_spectral_difference_panel.png)
*Signed spectral differences across inference conditions. Base at 50 steps: smooth W-shape. Base at 4 steps: severe deviation, large HF excess. Distilled LADD at 4 steps: near-flat, Δ≈0.*

### FLUX.2 Max

The fingerprint persists in BFL's flagship model (Δslope +0.333, n=20), making it the largest deviation of the three models tested. The shape differs from the base model — Max shows deeper mid-frequency under-production as its dominant feature. Small sample size limits interpretive weight.

---

## Limitations

- **Undisclosed training data.** FLUX training data composition is unknown. Spectral differences may partly reflect training distribution mismatch with FFHQ rather than architectural properties.
- **Cross-distribution comparison.** Unlike Keuper et al., where GANs were trained on the reference dataset, FLUX was not (or may not have been) trained on FFHQ.
- **Small sample sizes** for base model conditions (n=19–56) and FLUX.2 Max (n=20). Only the distilled measurement (n=200) is statistically well-resourced.
- **Prompt conditioning confound.** Text-guided generation may introduce spectral biases absent in unconditional generation.
- **JPEG robustness.** JPEG Q85 shifts the slope by −0.03, comparable in magnitude to the distilled model's Δslope of +0.072. A single re-encode through the standard web image pipeline could plausibly suppress or reverse the fingerprint for the closest-to-real models.
- **LADD isolation confounded by image quality.** The base model at 4 steps produces degraded output. The training objective effect and image quality degradation are inseparable with available models.
- **VAE experiment underpowered.** n=20 with a paired test rules out large effects; a subtle systematic shift below the detection threshold cannot be excluded.
- **No architectural decomposition.** The VAE is excluded as a source, but the flow-matching denoiser is not further decomposed. Which component — transformer blocks, RoPE embeddings, noise schedule, or training objective — produces the HF excess is unknown.

---

## Directions Not Pursued

- **Stable Diffusion 3 comparison:** Would test cross-architecture generality (SD3 shares the MM-DiT family and uses LADD). Deferred due to generation cost — better suited for thesis-scale work.
- **Social media robustness:** The fingerprint concentrates in high-frequency bins that JPEG and downscaling destroy first. Given that JPEG Q85 already shifts slope by −0.03 (comparable to the distilled model's Δ), robustness is expected to be poor.
- **Directional anisotropy analysis:** RoPE and patch tokenization may create directional biases invisible to azimuthal averaging. Infrastructure exists (`src/controls/directional.py`) but this is a separate investigation.
- **Extended classifier development:** The 0.995 AUC on controlled data is an expected baseline. Improving it does not advance understanding of the phenomenon.

---

## Repository Structure

```
flux/
├── configs/experiment.yaml
├── data/
│   ├── real/                          # 200 FFHQ images (1024×1024 PNG)
│   └── generated/
│       ├── klein_distilled/           # 200 images (LADD, s=4, g=1.0)
│       ├── klein_base_g4_s50/         # ~41 images (MSE, s=50, g=4.0)
│       ├── klein_base_g1_s50/         # ~19 images (MSE, s=50, g=1.0)
│       ├── klein_base_g1_s4/          # ~56 images (MSE, s=4, g=1.0)
│       ├── flux2_max/                 # ~20 images (BFL Playground)
│       └── vae_roundtrip/             # 20 VAE encode-decode reconstructions
├── src/
│   ├── spectral/                      # FFT, azimuthal averaging, metrics, stats
│   ├── detection/                     # Feature extraction, classifiers with CV
│   ├── controls/                      # Degradation methods, directional analysis
│   └── visualization/                 # Plotting functions
├── scripts/
│   ├── run_analysis.py                # Per-model spectral analysis
│   ├── run_controls.py                # Degradation controls
│   ├── run_detection.py               # Detection experiment
│   ├── run_grid_analysis.py           # Guidance × steps ablation
│   └── run_vae_roundtrip.py           # VAE isolation experiment
└── results/
    ├── klein_distilled/               # figures/ + metrics.json
    ├── klein_base_g4_s50/
    ├── flux2_max/
    ├── vae_roundtrip/
    ├── controls/
    └── grid_ablation/                 # Grid comparison plots + JSON
```

---

## How to Reproduce

```bash
# Install
pip install numpy scipy scikit-learn matplotlib pillow pyyaml tqdm
pip install torch diffusers transformers accelerate  # for generation only

# Reference dataset (downloads from HuggingFace)
python scripts/sample_ffhq.py

# Per-model analysis
python scripts/run_analysis.py --model klein_distilled

# Controls
python scripts/run_controls.py

# VAE round-trip (requires GPU)
python scripts/run_vae_roundtrip.py

# Guidance × steps ablation
python scripts/run_grid_analysis.py

# Detection
python scripts/run_detection.py --model klein_distilled
```

---

## References

- Durall R., Keuper M., Keuper J. — *Watch your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions* (CVPR 2020)
- Karras T. et al. — *A Style-Based Generator Architecture for Generative Adversarial Networks* (CVPR 2019)
- Lipman Y. et al. — *Flow Matching for Generative Modeling* (ICLR 2023)
- Sauer A. et al. — *Adversarial Diffusion Distillation* (ECCV 2024)
- Black Forest Labs — FLUX.2 Klein model family (2024–2025)

---
