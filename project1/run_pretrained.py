from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

for prompt in [
    "fruit of the loom",
    "fruit of the loom logo",
    "fruit of the loom with cornucopia",
    "fruit of the loom with cornucopia logo",
]:

    images = pipe(prompt=prompt).images[0]
    images.save(f"{prompt.replace(' ', '_')}.png")