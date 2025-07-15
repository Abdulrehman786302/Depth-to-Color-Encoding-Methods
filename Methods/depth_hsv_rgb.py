import numpy as np
import cv2
import json
import subprocess
import os
import matplotlib.pyplot as plt
import open3d as o3d
import re

# --- CONFIG ---
width, height = 640, 576
base_folder = "1"
depth_folder = "depth"
frame_folder = os.path.join(base_folder, "hsv_rgb_frames")
video_file = os.path.join(base_folder, "depth_encoded_hsv.mp4")
ply_folder = os.path.join(base_folder, "ply_output")
plot_path = os.path.join(base_folder, "preview_frame_0.png")
os.makedirs(frame_folder, exist_ok=True)
os.makedirs(ply_folder, exist_ok=True)
os.makedirs(base_folder, exist_ok=True)

depth_files = sorted(
    [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".txt")],
    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
)

# === HSV Multi-Ring Depth Encoding ===
def encode_depth_multi_hue_cycles(depth, dmin, dmax, max_hue_range=1530, max_cycles=8):
    scaled = depth - dmin
    total_range = dmax - dmin
    num_cycles = min(max_cycles, int(np.ceil(total_range / max_hue_range)))
    
    cycle_idx = np.floor(scaled / max_hue_range).astype(np.int32)
    cycle_idx = np.clip(cycle_idx, 0, num_cycles-1)
    hue_in_cycle = np.mod(scaled, max_hue_range)
    
    H = (hue_in_cycle / max_hue_range) * 179
    S = 64 + (cycle_idx / (num_cycles - 1)) * (255 - 64) if num_cycles > 1 else np.ones_like(H) * 255
    V = np.ones_like(H) * 255
    
    hsv = np.stack([H, S, V], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def decode_depth_multi_hue_cycles(rgb, dmin, dmax, max_hue_range=1530, max_cycles=8):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S = hsv[..., 0], hsv[..., 1]
    
    cycle_idx = ((S - 64) / (255 - 64)) * (max_cycles - 1) if max_cycles > 1 else np.zeros_like(S)
    cycle_idx = np.round(np.clip(cycle_idx, 0, max_cycles - 1))
    
    hue_in_cycle = (H / 179) * max_hue_range
    depth = cycle_idx * max_hue_range + hue_in_cycle + dmin
    return np.clip(depth, dmin, dmax)

# === Load Depth Files ===
def load_depth_txt(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        flat = list(data.values())[0]
    else:
        flat = data
    return np.array(flat, dtype=np.float32).reshape((height, width))

def depth_to_pointcloud(depth, color):
    fx = fy = 525
    cx, cy = width // 2, height // 2
    indices = np.indices((height, width)).transpose(1, 2, 0)
    z = depth / 1000.0
    x = (indices[..., 1] - cx) * z / fx
    y = (indices[..., 0] - cy) * z / fy
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = color.reshape(-1, 3) / 255.0
    mask = z.reshape(-1) > 0.1
    return points[mask], colors[mask]

all_depths = [load_depth_txt(f) for f in depth_files]
num_frames = len(all_depths)
dmin, dmax = min(d.min() for d in all_depths), max(d.max() for d in all_depths)

# === Encode Frames ===
for i, depth in enumerate(all_depths):
    rgb = encode_depth_multi_hue_cycles(depth, dmin, dmax)
    path = os.path.join(frame_folder, f"frame_{i:06d}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# === FFmpeg Compression ===
subprocess.run([
    "ffmpeg", "-y", "-framerate", "1",
    "-i", os.path.join(frame_folder, "frame_%06d.png"),
    "-c:v", "libx264", "-crf", "23", "-preset", "slow",
    "-pix_fmt", "yuv420p", video_file
], check=True)

# === Decode & Save PLYs ===
cap = cv2.VideoCapture(video_file)
maes, rmss = [], []

for i in range(num_frames):
    ret, frame_bgr = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    decoded = decode_depth_multi_hue_cycles(rgb, dmin, dmax)
    original = all_depths[i]

    abs_error = np.abs(decoded - original)
    maes.append(np.mean(abs_error))
    rmss.append(np.sqrt(np.mean(abs_error**2)))

    p1, c1 = depth_to_pointcloud(original, encode_depth_multi_hue_cycles(original, dmin, dmax))
    pcd_orig = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p1))
    pcd_orig.colors = o3d.utility.Vector3dVector(c1)
    o3d.io.write_point_cloud(os.path.join(ply_folder, f"frame_{i:06d}_original.ply"), pcd_orig)

    p2, c2 = depth_to_pointcloud(decoded, rgb)
    pcd_dec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p2))
    pcd_dec.colors = o3d.utility.Vector3dVector(c2)
    o3d.io.write_point_cloud(os.path.join(ply_folder, f"frame_{i:06d}_decoded.ply"), pcd_dec)

    # --- Plot First Frame Only ---
    if i == 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Depth")
        axes[1].imshow(rgb)
        axes[1].set_title("Encoded RGB (HSV)")
        axes[2].imshow(decoded, cmap='gray')
        axes[2].set_title("Decoded Depth")
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

cap.release()

# === Report Metrics ===
print(f"\nFinal Average MAE: {np.mean(maes):.2f} mm")
print(f"Final Average RMS: {np.mean(rmss):.2f} mm")
print(f"PLY files saved in: {ply_folder}")
print(f"Preview plot saved to: {plot_path}")
