import torch
import os
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T


parser = argparse.ArgumentParser(description='Feature Extraction from Images')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1, type=int)
parser.add_argument("-bs", "--bs", help="Batch Size", default=30, type=int)
args = parser.parse_args()
sub = args.sub
batch_size = args.bs

print("Loading Stable Diffusion pipeline with IP-Adapter...")
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)


class batch_generator_external_images(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)

    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)


image_path = '../data/processed_data/subj{:02d}/nsd_train_stim_sub{}.npy'.format(sub,sub)
train_images = batch_generator_external_images(data_path = image_path)

image_path = '../data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)

trainloader = DataLoader(train_images,batch_size,shuffle=False)
testloader = DataLoader(test_images,batch_size,shuffle=False)

def extract_features(dataloader):
    features = []
    for batch in dataloader:
        with torch.no_grad():
            image_embeds = pipeline.prepare_ip_adapter_image_embeds(
                ip_adapter_image=batch,
                ip_adapter_image_embeds=None,
                device="cuda",
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            features.append(image_embeds.cpu().numpy())
    return np.concatenate(features)

print("Extracting features...")
train_features = extract_features(trainloader)
test_features = extract_features(testloader)

os.makedirs(f"../data/extracted_features/subj{sub:02d}", exist_ok=True)
outfile = f"../data/extracted_features/subj{sub:02d}/image_features.npz"
np.savez(outfile, train_features=train_features, test_features=test_features)

print(f"Feature extraction complete. Saved to {outfile}")
