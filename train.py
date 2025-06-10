import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from read_dataset import BananaDataset
from network import DDPM, Estimator
from torchvision import transforms
from tqdm import tqdm
import random
import torchvision
from torchvision.transforms import InterpolationMode

def train_3d_diffusion():
    # Params
    n_epoch = 500
    batch_size = 2048
    n_T = 500
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_feat = 128
    lrate = 4e-4
    save_model_dir = "../models/wc"
    best_val_loss = float("inf")

    # Pre-processing
    transform_RGB = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    transform_Gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    transform_depth = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Init dataset
    dataset = BananaDataset(
        "../dataset/dataset.h5",
        "../dataset/metadata.json",
        transform_RGB=transform_RGB,
        transform_Gray=transform_Gray,
        transform_depth=transform_depth
    )

    # Dataset split
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)

    train_loader = DataLoader(Subset(dataset, indices[:train_size]), batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(dataset, indices[train_size:train_size+val_size]), batch_size=batch_size, shuffle=False, num_workers=2)

    # Init model
    ddpm = DDPM(nn_model=MaskedDenseFusion(n_feat=n_feat, out_dim=3),
                betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)
    optimizer = optim.Adam(ddpm.parameters(), lr=lrate)

    print("train_model_alpha_wc")

    # Train and Validation loops
    for ep in range(n_epoch):
        print(f"\nEpoch {ep+1}/{n_epoch}")
        ddpm.train()
        optimizer.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(train_loader, desc="Training")
        train_loss_ema = None

        for batch in pbar:
            optimizer.zero_grad()

            loss = ddpm(
                batch["banana_position"].to(device),
                batch["floor_plan"].to(device),
                batch["depth"].to(device),
                batch["camera_position"].to(device),
                batch["camera_rotation"].to(device),
                batch["image"].to(device)
            )

            loss.backward()
            optimizer.step()

            if train_loss_ema is None:
                train_loss_ema = loss.item()
            else:
                train_loss_ema = 0.95 * train_loss_ema + 0.05 * loss.item()

            pbar.set_description(f"Train loss EMA: {train_loss_ema:.4f}")

        # Val
        ddpm.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for batch in val_pbar:
                val_loss = ddpm(
                    batch["banana_position"].to(device),
                    batch["floor_plan"].to(device),
                    batch["depth"].to(device),
                    batch["camera_position"].to(device),
                    batch["camera_rotation"].to(device),
                    batch["image"].to(device)
                )
                val_loss_total += val_loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save the best one
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ddpm.state_dict(), f"{save_model_dir}/best_model.pth")
            print(f"Best model updated at epoch {ep+1} with loss {best_val_loss:.4f}")

    torch.save(ddpm.state_dict(), f"{save_model_dir}/model_final.pth")
    print("Training completed!")

if __name__ == "__main__":
    train_3d_diffusion()