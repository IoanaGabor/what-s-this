import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
import torch.optim as optim

# Define MindFormer Model
class MindFormer(nn.Module):
    def __init__(self, input_size, embed_dim=1280, num_patches=2, num_heads=8, num_layers=2):
        super(MindFormer, self).__init__()
        self.embed_layer = nn.Linear(input_size, num_patches * embed_dim)
        self.subject_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        x = self.embed_layer(x).view(x.shape[0], 2, 1280)
        x = torch.cat([self.subject_token.repeat(x.shape[0], 1, 1), x], dim=1)
        x = x + self.position_embeddings
        x = self.transformer(x)
        return self.output_layer(x[:, 1:])

# PyTorch Lightning Module
class ExperimentModel(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, alpha=0.1):
        super(ExperimentModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def contrastive_loss(self, pred, target):
        pred = pred.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        pred = pred.view(pred.shape[0], -1)   # Shape: [4, 2560]
        target = target.view(target.shape[0], -1)  # Shape: [4, 2560]
        numerator = torch.exp(pred * target)
        denominator = torch.sum(torch.exp(pred.unsqueeze(1) * target.unsqueeze(0)), dim=1)
        log_probs = torch.log(numerator / denominator)
        loss = -log_probs.mean()
        return loss

    def mindformer_loss(self, pred, target):
        #print(f"Pred shape: {pred.shape}, Target shape: {target.shape}")

        return self.l1_loss(pred, target) + self.alpha * self.contrastive_loss(pred, target)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.mindformer_loss(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.mindformer_loss(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-4,  # Learning rate set to 3 × 10⁻⁴
            betas=(0.9, 0.999)  # AdamW moment parameters
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"  # Update the learning rate per epoch
                }
        }

# Data Loading
class fMRIDataset(torch.utils.data.Dataset):
    def __init__(self, fmri_data, latents):
        self.fmri_data = fmri_data
        self.latents = latents
    
    def __len__(self):
        return len(self.fmri_data)
    
    def __getitem__(self, idx):
        return self.fmri_data[idx], self.latents[idx]

def load_data(sub, batch_size=4):
    train_fmri = np.load(f'data/processed_data/subj{sub:02d}/nsd_train_fmriavg_nsdgeneral_sub{sub}.npy')
    test_fmri = np.load(f'data/processed_data/subj{sub:02d}/nsd_test_fmriavg_nsdgeneral_sub{sub}.npy')
    test_outfile = f"data/extracted_features/subj{sub:02d}/image_features_test.npz"
    train_outfile = f"data/extracted_features/subj{sub:02d}/image_features_train.npz"
    train_latents = torch.load(train_outfile)
    test_latents = torch.load(test_outfile)
    train_fmri, test_fmri = torch.tensor(train_fmri, dtype=torch.float32), torch.tensor(test_fmri, dtype=torch.float32)
    train_latents = [torch.stack(sublist) for sublist in train_latents]
    train_latents = torch.stack(train_latents)

    test_latents = [torch.stack(sublist) for sublist in test_latents]
    test_latents  = torch.stack(test_latents)


    train_latents, test_latents = torch.tensor(train_latents, dtype=torch.float32), torch.tensor(test_latents, dtype=torch.float32)
    train_dataset = fMRIDataset(train_fmri, train_latents)
    test_dataset = fMRIDataset(test_fmri, test_latents)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader



# Training Function
def train(args):
    pl.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MindFormer(input_size=15724).to(device)  
    lightning_model = ExperimentModel(model)
    
    logger = WandbLogger(project='mindformer_experiment', name=f"exp_{args.seed}")
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(dirpath="./checkpoints/", filename="model_{epoch:02d}_{val_loss:.5f}", save_top_k=3, monitor="val_loss", mode="min"),
        EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
    ]
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0],
        max_epochs=50,
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        val_check_interval=0.25,
    )
    
    train_loader, val_loader = load_data(args.sub, batch_size=4)
    trainer.fit(lightning_model, train_loader, val_loader)

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindFormer Training')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sub", type=int, choices=[1, 2, 5, 7], required=True, help="Subject Number")
    args = parser.parse_args()
    train(args)
