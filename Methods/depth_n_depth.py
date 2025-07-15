import re
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import subprocess
import matplotlib.pyplot as plt


# --- CONFIG ---
WIDTH, HEIGHT = 640, 576
DEPTH_FOLDER = "depth"
BATCH_SIZE = 4
NUM_EPOCHS = 10

# Base output folder "5/"
BASE_OUTPUT_FOLDER = "5"
FRAME_OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "encoded_frames")
RECONSTRUCTED_OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "reconstructed_depths")
PLOTS_OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, "plots")
VIDEO_OUTPUT_FILE = os.path.join(BASE_OUTPUT_FOLDER, "depth_compressed.mp4")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create all output dirs
os.makedirs(FRAME_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RECONSTRUCTED_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PLOTS_OUTPUT_FOLDER, exist_ok=True)

# --- Dataset to load depth from rawDepth%d.txt files ---
class DepthFolderDataset(Dataset):
    def __init__(self, folder, dmin=None, dmax=None):
        self.files = sorted(glob.glob(f"{folder}/rawDepth*.txt"),
                            key=lambda x: int(re.findall(r'\d+', x)[-1]))
        self.dmin = dmin
        self.dmax = dmax

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                flat = list(data.values())[0]
            else:
                flat = data
        depth_np = np.array(flat, dtype=np.float32).reshape(HEIGHT, WIDTH)
        if self.dmin is not None and self.dmax is not None:
            depth_norm = (depth_np - self.dmin) / (self.dmax - self.dmin + 1e-6)
        else:
            depth_norm = depth_np  # raw depth if no min/max given
        return torch.from_numpy(depth_norm).unsqueeze(0).float(), torch.from_numpy(depth_np).unsqueeze(0).float()

# --- Simple Conv Block ---
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

def pool(x):
    return nn.MaxPool2d(2)(x)

def upsample(x):
    return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(x)

# --- Bit-plane inspired layer ---
class BitPlaneLayer(nn.Module):
    def __init__(self, bits=4, in_channels=128):
        super().__init__()
        self.bits = bits
        # Reduce to 1 channel before bitplane
        self.reduce = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # Reduce channels to 1
        x_reduced = self.reduce(x)
        # Now x_reduced shape: [batch, 1, H, W], values assumed normalized [0,1]
        out = torch.cat([torch.clamp((x_reduced * (2 ** b)).floor() / (2 ** b), 0, 1) for b in range(self.bits)], dim=1)
        return out


# --- UNet with bitplane ---
class UNetBitplane(nn.Module):
    def __init__(self, channels_in=1):
        super().__init__()
        self.enc1 = conv_block(channels_in, 32)      # enc1 outputs 32 channels
        self.enc2 = conv_block(32, 64)                # enc2 outputs 64 channels
        self.enc3 = conv_block(64, 128)               # enc3 outputs 128 channels
        self.mid = conv_block(128, 128)               # mid outputs 128 channels
        self.bp = BitPlaneLayer(bits=4, in_channels=128)
        self.dec3 = conv_block(128 + 128 + 4, 64)    # expects 128+128+4=260 channels

        self.dec2 = conv_block(64 + 64, 32)
        self.dec1 = conv_block(32 + 32, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(pool(e1))
        e3 = self.enc3(pool(e2))
        m = self.mid(pool(e3))
        bp = self.bp(m)
        bp_up = upsample(bp)
        m_up = upsample(m)
        #print(f"m_up shape: {m_up.shape}")
        #print(f"e3 shape: {e3.shape}")
        #print(f"bp_up shape: {bp_up.shape}")
        concat = torch.cat([m_up, e3, bp_up], dim=1)
        #print(f"Concat shape before dec3: {concat.shape}")
        d3 = self.dec3(concat)
        d2 = self.dec2(torch.cat([upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([upsample(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


# --- Training pipeline ---
def train():
    print(f"Training on device: {DEVICE}")

    all_depths = []
    files = sorted(glob.glob(f"{DEPTH_FOLDER}/rawDepth*.txt"), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                flat = list(data.values())[0]
            else:
                flat = data
        depth_np = np.array(flat, dtype=np.float32).reshape(HEIGHT, WIDTH)
        all_depths.append(depth_np)
    dmin, dmax = np.min(all_depths), np.max(all_depths)
    print(f"Depth min: {dmin}, max: {dmax}")

    dataset = DepthFolderDataset(DEPTH_FOLDER, dmin, dmax)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNetBitplane().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        print(f"Dataset size: {len(dataset)}")
        for inputs, targets in dataloader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Loss: {epoch_loss / len(dataloader):.6f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'dmin': dmin,
        'dmax': dmax
    }, os.path.join(BASE_OUTPUT_FOLDER, 'n_depth_model.pth'))
    print("Training complete and model saved.")

# --- Inference & encode PNG frames ---
def encode_depth_to_rgb(depth_norm):
    depth = depth_norm.squeeze(0).cpu().numpy()

    depth_16bit = (depth * 65535).astype(np.uint16)
    R = (depth_16bit >> 8).astype(np.uint8)
    G = (depth_16bit & 0xFF).astype(np.uint8)
    B = np.zeros_like(R, dtype=np.uint8)

    rgb = np.stack([R, G, B], axis=2)
    return rgb

def inference():
    checkpoint = torch.load(os.path.join(BASE_OUTPUT_FOLDER, 'n_depth_model.pth'), map_location=DEVICE)
    model = UNetBitplane().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dmin = checkpoint['dmin']
    dmax = checkpoint['dmax']

    dataset = DepthFolderDataset(DEPTH_FOLDER, dmin, dmax)
    os.makedirs(FRAME_OUTPUT_FOLDER, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            norm_depth, _ = dataset[i]
            norm_depth = norm_depth.to(DEVICE).unsqueeze(0)
            output = model(norm_depth).squeeze(0)[0]
            output = output.clamp(0,1)

            rgb_img = encode_depth_to_rgb(output)

            filename = os.path.join(FRAME_OUTPUT_FOLDER, f"frame_{i:06d}.png")
            cv2.imwrite(filename, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

    print(f"Inference PNG frames saved to {FRAME_OUTPUT_FOLDER}")

    subprocess.run([
        'ffmpeg', '-y', '-framerate', '5',
        '-i', os.path.join(FRAME_OUTPUT_FOLDER, 'frame_%06d.png'),
        '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
        VIDEO_OUTPUT_FILE
    ], check=True)
    print(f"Video saved as {VIDEO_OUTPUT_FILE}")

# --- Decode and evaluate ---
def decode_rgb_to_depth(rgb_img, dmin, dmax):
    R = rgb_img[:, :, 0].astype(np.uint16)
    G = rgb_img[:, :, 1].astype(np.uint16)

    depth_16bit = (R << 8) + G
    depth_norm = depth_16bit.astype(np.float32) / 65535.0
    depth_mm = depth_norm * (dmax - dmin) + dmin
    return torch.from_numpy(depth_mm)

def decode_and_eval():
    checkpoint = torch.load(os.path.join(BASE_OUTPUT_FOLDER, 'n_depth_model.pth'), map_location=DEVICE)
    model = UNetBitplane().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dmin = checkpoint['dmin']
    dmax = checkpoint['dmax']

    cap = cv2.VideoCapture(VIDEO_OUTPUT_FILE)
    dataset = DepthFolderDataset(DEPTH_FOLDER, dmin, dmax)

    maes = []
    rmss = []

    os.makedirs(RECONSTRUCTED_OUTPUT_FOLDER, exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded_depth = decode_rgb_to_depth(frame_rgb, dmin, dmax)
            original_depth = dataset[i][1].squeeze(0)

            np.save(os.path.join(RECONSTRUCTED_OUTPUT_FOLDER, f"decoded_depth_{i:06d}.npy"), decoded_depth.numpy())

            abs_err = (decoded_depth - original_depth).abs()
            maes.append(abs_err.mean().item())
            rmss.append((abs_err ** 2).mean().sqrt().item())

            if i < 1:
                plt.figure(figsize=(12,4))
                plt.subplot(1,3,1)
                plt.title("Original depth (mm)")
                plt.imshow(original_depth.numpy(), cmap='viridis')
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.title("Decoded depth (mm)")
                plt.imshow(decoded_depth.numpy(), cmap='viridis')
                plt.colorbar()
                plt.subplot(1,3,3)
                plt.title("Abs Error (mm)")
                plt.imshow(abs_err.numpy(), cmap='hot')
                plt.colorbar()
                plt.tight_layout()

                # Save plot instead of showing
                plot_path = os.path.join(PLOTS_OUTPUT_FOLDER, f"eval_plot_{i:06d}.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved evaluation plot to {plot_path}")

    cap.release()
    print(f"Final Average MAE: {np.mean(maes):.2f} mm")
    print(f"Final Average RMS: {np.mean(rmss):.2f} mm")


if __name__ == "__main__":
    print("Starting training...")
    train()
    print("Running inference...")
    inference()
    print("Evaluating reconstruction...")
    decode_and_eval()