from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline
from diffusers.utils import load_image
from transformers import CLIPVisionModelWithProjection
import torch

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-base-patch32",
    torch_dtype=torch.float16,
)

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("XLabs-AI/flux-ip-adapter", subfolder="", weight_name="ip_adapter.safetensors", use_safetensors=True, 
 low_cpu_mem_usage=False, image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14")
#pipe = pipeline(task="image-feature-extraction", model_name="", device="cuda", pool=True)
pipeline.set_ip_adapter_scale(1)


image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=load_image("image.png"),
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)
print(image_embeds[0].shape)
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=load_image("image.png"),
    ip_adapter_image_embeds=image_embeds,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)
print(len(image_embeds))
print(image_embeds)

torch.save(image_embeds, "image_embeds.ipadpt")


image_embeds = torch.load("image_embeds.ipadpt")
generator = torch.Generator(device="cuda").manual_seed(0)
# images = pipeline(
#     prompt="",
#     ip_adapter_image_embeds=image_embeds,
#     negative_prompt="deformed, ugly, wrong proportion, low res",
#     num_inference_steps=100,
#     generator=generator,
# ).images


images = pipeline(
    prompt_embeds=image_embeds[0],
    negative_prompt_embeds=image_embeds[1],
    num_inference_steps=100  # Adjust for quality
).images



# Save images to disk
output_dir = "generated_images"  # Change this to your desired directory
import os

os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

for idx, img in enumerate(images):
    img_path = os.path.join(output_dir, f"generated_image_{idx}.png")
    img.save(img_path)


