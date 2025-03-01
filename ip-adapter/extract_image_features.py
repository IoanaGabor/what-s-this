import torch
import os
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus


base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder"
ip_ckpt = "models/ip-adapter-plus_sd15.bin"
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

parser = argparse.ArgumentParser(description='Feature Extraction from Images')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
args = parser.parse_args()
sub = args.sub
batch_size = args.bs

print("Loading Stable Diffusion pipeline with IP-Adapter...")

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

ip_model = IPAdapterPlus(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self,idx):
        return self.im[idx]

    def __len__(self):
        return  len(self.im)


image_path = '../data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = '../data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

trainloader = DataLoader(train_images,batch_size,shuffle=False)
testloader = DataLoader(test_images,batch_size,shuffle=False)
generator = torch.Generator(device="cuda").manual_seed(0)



def extract_features(dataloader):
    features = []
    cnt = 0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader)):
            for b in batch:
                img = Image.fromarray(b.numpy())
                img.save(f"{cnt}-original.png")
                image_embeds=ip_model.get_image_embeds(pil_image=img)
                features.append(image_embeds)
            
                images = pipe(
                    prompt_embeds=image_embeds[0], 
                    negative_prompt_embeds=image_embeds[1],
                    num_inference_steps=100  # Adjust for quality
                ).images[0]

                images.save(f"{cnt}-recreated.png")
                cnt += 1

    return features

print("Extracting features...")
train_features = extract_features(trainloader)
test_features = extract_features(testloader)

os.makedirs(f"../data/extracted_features/subj{sub:02d}", exist_ok=True)
test_outfile = f"../data/extracted_features/subj{sub:02d}/image_features_test.npz"
train_outfile = f"../data/extracted_features/subj{sub:02d}/image_features_train.npz"
torch.save(test_features,test_outfile)
torch.save(train_features,train_outfile)
#np.savez(outfile, train_features=train_features, test_features=test_features)

print(f"Feature extraction complete. Saved")

