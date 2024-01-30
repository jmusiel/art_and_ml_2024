from diffusers import DiffusionPipeline
import torch
import os

pipe = DiffusionPipeline.from_pretrained("/home/jovyan/working/art_ml_project/project1/save_model/steps100", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
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
    images.save(os.path.join("outputs",f"steps100_{prompt.replace(' ', '_')}.png"))