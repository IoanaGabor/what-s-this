import torch
import argparse
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from train_transformer import MindFormer
import os

def load_model(model_path, input_size=15724):
    model = MindFormer(input_size=input_size)
    checkpoint = torch.load(model_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    model.to("cuda").eval()
    return model

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor.to("cuda"))
    return output.view(1, 2, 1, 1280)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindFormer Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--sub", type=int, choices=[1, 2, 5, 7], required=True, help="Subject Number")
    parser.add_argument("--output_path", type=str, default="image_embeds.ipadpt", help="Path to save image embeddings")
    args = parser.parse_args()

    model = load_model(args.model_path)
    
    input_data = torch.tensor(torch.load(args.input_path), dtype=torch.float32).to("cuda")
    input_data = input_data.unsqueeze(0)  # Ensure batch dimension
    
    image_embeds = predict(model, input_data)
    torch.save(image_embeds, args.output_path)
    
    pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    pipeline.set_ip_adapter_scale(0.6)
    
    image_embeds = torch.load(args.output_path)
    generator = torch.Generator(device="cuda").manual_seed(0)
    images = pipeline(
        prompt="",
        ip_adapter_image_embeds=image_embeds,
        negative_prompt="deformed, ugly, wrong proportion, low res",
        num_inference_steps=100,
        generator=generator,
    ).images
    
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    for idx, img in enumerate(images):
        img.save(f"{output_dir}/subj{args.sub}/{idx}.png")

    print(f"Generated images saved in {output_dir}")
