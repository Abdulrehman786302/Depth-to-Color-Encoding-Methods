import numpy as np
import cv2
import json
import subprocess
import os
import matplotlib.pyplot as plt
import open3d as o3d
import re
from skimage import color

# --- CONFIG ---
width, height = 640, 576
depth_folder = "depth"
folder = "2"
os.makedirs(folder, exist_ok=True)
video_file = "depth_encoded_luv.mp4"

# === Utility Functions ===
def encode_depth_to_cieluv(depth, dmin, dmax):
    norm = (depth - dmin) / (dmax - dmin + 1e-6)
    L = norm * 100
    angle = norm * 2 * np.pi
    u = 50 * np.cos(angle)
    v = 50 * np.sin(angle)
    luv = np.stack([L, u, v], axis=-1)
    rgb = color.luv2rgb(luv)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb

def decode_depth_from_cieluv(rgb_img, dmin, dmax):
    rgb = np.clip(rgb_img.astype(np.float32) / 255, 0, 1)
    luv = color.rgb2luv(rgb)
    L = luv[..., 0]
    norm = L / 100.0
    depth = norm * (dmax - dmin) + dmin
    return np.clip(depth, dmin, dmax)

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

# === Load all depth files ===
depth_files = sorted(
    [os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".txt")],
    key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
)
all_depths = [load_depth_txt(f) for f in depth_files]
num_frames = len(all_depths)
dmin, dmax = min(d.min() for d in all_depths), max(d.max() for d in all_depths)

# === Encode and save images ===
for i, depth in enumerate(all_depths):
    rgb = encode_depth_to_cieluv(depth, dmin, dmax)
    path = os.path.join(folder, f"frame_{i:06d}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# === Encode video with FFmpeg ===
subprocess.run([
    "ffmpeg", "-y", "-framerate", "1",
    "-i", os.path.join(folder, "frame_%06d.png"),
    "-c:v", "libx264", "-crf", "23", "-preset", "slow",
    "-pix_fmt", "yuv420p", os.path.join(folder, video_file)
], check=True)

# === Decode video and evaluate ===
cap = cv2.VideoCapture(video_file)
maes, rmss = [], []

for i in range(num_frames):
    ret, frame_bgr = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    decoded = decode_depth_from_cieluv(rgb, dmin, dmax)
    original = all_depths[i]

    abs_error = np.abs(decoded - original)
    maes.append(np.mean(abs_error))
    rmss.append(np.sqrt(np.mean(abs_error**2)))

    # Save original PLY
    p1, c1 = depth_to_pointcloud(original, encode_depth_to_cieluv(original, dmin, dmax))
    pcd_orig = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p1))
    pcd_orig.colors = o3d.utility.Vector3dVector(c1)
    ply_orig_path = os.path.join(folder, f"frame_{i:06d}_original.ply")
    o3d.io.write_point_cloud(ply_orig_path, pcd_orig)

    # Save decoded PLY
    p2, c2 = depth_to_pointcloud(decoded, rgb)
    pcd_dec = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p2))
    pcd_dec.colors = o3d.utility.Vector3dVector(c2)
    ply_dec_path = os.path.join(folder, f"frame_{i:06d}_decoded.ply")
    o3d.io.write_point_cloud(ply_dec_path, pcd_dec)

    # === Save visualization plot ===
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title("Original Depth")
    axs[0].axis('off')

    axs[1].imshow(rgb)
    axs[1].set_title("Encoded RGB (CIELUV)")
    axs[1].axis('off')

    axs[2].imshow(decoded, cmap='gray')
    axs[2].set_title("Decoded Depth")
    axs[2].axis('off')

    plt.tight_layout()
    viz_path = os.path.join(folder, f"viz_frame_{i:06d}.png")
    plt.savefig(viz_path)
    plt.close()
    print(f"Saved: {viz_path}, {os.path.basename(ply_orig_path)}, {os.path.basename(ply_dec_path)}")

cap.release()

# === Final Report ===
print(f"\nFinal Average MAE: {np.mean(maes):.2f} mm")
print(f"Final Average RMS: {np.mean(rmss):.2f} mm")
