import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # critical for 8GB VRAM

image = pipe(
    prompt="closeup headshot photograph of a middle aged man, natural lighting, monotone background",
    height=1024, width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]
image.save("test_output.png")