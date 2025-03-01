import torch
import argparse
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
from train_transformer import MindFormer, ExperimentModel
import numpy as np
import os


from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "ip-adapter/models/image_encoder"
ip_ckpt = "ip-adapter/models/ip-adapter-plus_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

print("Loading Stable Diffusion pipeline with IP-Adapter...")

torch.cuda.empty_cache()
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None,
    device="cuda"
)

generator = torch.Generator(device="cuda").manual_seed(0)
ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)



def load_model(model_path, input_size=15724):
    model = MindFormer(input_size=15724)      
    lightning_model = ExperimentModel(model)
    checkpoint = torch.load(model_path, map_location="cuda")
    lightning_model.load_state_dict(checkpoint["state_dict"])
    lightning_model.to("cuda").eval()
    return model

def predict(model, input_tensor):
    outputs = []
    with torch.no_grad():
        output = model(input_tensor[0].to("cuda"))
        return output.view(len(input_tensor[0]), 32, 768)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindFormer Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--sub", type=int, choices=[1, 2, 5, 7], required=True, help="Subject Number")
    args = parser.parse_args()

    #model = load_model(args.model_path)
    train_fmri = np.load(f'data/processed_data/subj{args.sub:02d}/nsd_train_fmriavg_nsdgeneral_sub{args.sub}.npy') / 300
    test_fmri = np.load(f'data/processed_data/subj{args.sub:02d}/nsd_test_fmriavg_nsdgeneral_sub{args.sub}.npy') / 300
    norm_mean_train, norm_std_train = np.mean(train_fmri, axis=0), np.std(train_fmri, axis=0, ddof=1)
    train_fmri = (train_fmri - norm_mean_train) / norm_std_train
    test_fmri = (test_fmri - norm_mean_train) / norm_std_train
    input_data = torch.tensor(test_fmri, dtype=torch.float32).to("cuda")[0:4]
    input_data = input_data.unsqueeze(0)  # Ensure batch dimension
    del train_fmri
    del norm_mean_train
    torch.cuda.empty_cache()

    model = load_model(args.model_path)

    print(input_data.shape)
    image_embeds = predict(model, input_data)
    torch.save(image_embeds, "temp.ipadapt")
    
    image_embeds = torch.load("temp.ipadapt")
#    test_outfile = f"data/extracted_features/subj01/image_features_test.npz"
#    image_embeds = torch.load(test_outfile)
#    image_embeds = [tensor for tensor in image_embeds]
#    pipeline = AutoPipelineForText2Image.from_pretrained("sd-legacy/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
#    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sd15.bin")
#    pipeline.set_ip_adapter_scale(1)
    
    #image_embeds = torch.load("temp.ipadapt")
    generator = torch.Generator(device="cuda").manual_seed(0)
    for i in range(len(image_embeds)):
#        images = [tensor for tensor in image_embeds[i]]
#        print(images[0].shape)

        #print(len(image_embeds[i][0]))
        #print(torch.stack([torch.stack([image_embeds[i][0:16]])]).shape)
        images = pipe(
            prompt_embeds=torch.stack([image_embeds[i][0:16]]),
            negative_prompt_embeds=torch.stack([image_embeds[i][16:32]]),
            num_inference_steps=100  # Adjust for quality
        ).images

#        images = pipeline(
#            prompt="",
#            ip_adapter_image_embeds=image_embeds[i],
#            negative_prompt="deformed, ugly, wrong proportion, low res",
#            num_inference_steps=100,
#             generator=generator,
#        ).images
    
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(f"{output_dir}/subj{args.sub:02d}/{i}.png")

    print(f"Generated images saved in {output_dir}")
