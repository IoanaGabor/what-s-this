from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)


image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=load_image("image.png"),
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

torch.save(image_embeds, "image_embeds.ipadpt")


image_embeds = torch.load("image_embeds.ipadpt")
generator = torch.Generator(device="cuda").manual_seed(0)
images = pipeline(
    prompt="",
    ip_adapter_image_embeds=image_embeds,
    negative_prompt="deformed, ugly, wrong proportion, low res",
    num_inference_steps=100,
    generator=generator,
).images


# Save images to disk
output_dir = "generated_images"  # Change this to your desired directory
import os

os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

for idx, img in enumerate(images):
    img_path = os.path.join(output_dir, f"generated_image_{idx}.png")
    img.save(img_path)
