import os
import re
import glob
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing
import subprocess

# === CONFIG ===
width, height = 640, 576
roi_depth_percentile = 2
max_wraps = 50  # wrap candidates for phase unwrapping

DEPTH_FOLDER = "depth"
METHOD_OUTPUT_FOLDER = "4"
os.makedirs(METHOD_OUTPUT_FOLDER, exist_ok=True)

ENCODED_PNG = os.path.join(METHOD_OUTPUT_FOLDER, "encoded_depth.png")
ENCODED_PLY = os.path.join(METHOD_OUTPUT_FOLDER, "encoded_cloud.ply")
ENCODED_VIDEO = os.path.join(METHOD_OUTPUT_FOLDER, "encoded_video.mp4")
VIDEO_FRAME_PNG = os.path.join(METHOD_OUTPUT_FOLDER, "frame_from_video.png")
DECODED_PLY = os.path.join(METHOD_OUTPUT_FOLDER, "decoded_cloud.ply")
PLOT_FILE = os.path.join(METHOD_OUTPUT_FOLDER, "depth_error_plot.png")

# --- Helpers ---

def get_dynamic_depth_range(depth_map, lower_pct=1, upper_pct=99):
    valid_depths = depth_map[(depth_map > 0) & np.isfinite(depth_map)]
    if len(valid_depths) == 0:
        return 0.0, 1.0
    dmin = np.percentile(valid_depths, lower_pct)
    dmax = np.percentile(valid_depths, upper_pct)
    return dmin, dmax

def choose_wavelengths(num=3, min_wl=0.1, max_wl=0.4):
    return np.linspace(min_wl, max_wl, num=num)

def multiwavelength_encode(depth_norm, wavelengths):
    encoded_channels = []
    for w in wavelengths:
        phase = 2 * np.pi * depth_norm / w
        sin_comp = ((np.sin(phase) + 1) / 2 * 255).astype(np.uint8)
        cos_comp = ((np.cos(phase) + 1) / 2 * 255).astype(np.uint8)
        encoded_channels.append(sin_comp)
        encoded_channels.append(cos_comp)
    return np.stack(encoded_channels, axis=-1)

def encode_depth_feature_driven(depth_norm, roi_mask, wavelengths_roi, wavelengths_bg):
    sincos_roi = multiwavelength_encode(depth_norm, wavelengths_roi)
    sincos_bg  = multiwavelength_encode(depth_norm, wavelengths_bg)
    encoded = np.where(roi_mask[..., None], sincos_roi, sincos_bg)
    return encoded

def multiwavelength_phase_unwrap_decode(sincos_stack, wavelengths, max_wraps=50):
    orig_shape = None
    if sincos_stack.ndim == 3:
        h, w, c = sincos_stack.shape
        sincos_stack = sincos_stack.reshape(-1, c)
        orig_shape = (h, w)
    elif sincos_stack.ndim != 2:
        raise ValueError("Unexpected sincos_stack shape")

    N, C = sincos_stack.shape
    assert C % 2 == 0

    num_wavelengths = C // 2

    sin_vals = sincos_stack[:, 0::2].astype(np.float32) / 255.0 * 2 - 1
    cos_vals = sincos_stack[:, 1::2].astype(np.float32) / 255.0 * 2 - 1

    wrapped_phases = np.arctan2(sin_vals, cos_vals)

    zs_candidates = np.zeros((N, num_wavelengths, max_wraps), dtype=np.float32)
    for i, wv in enumerate(wavelengths):
        for k in range(max_wraps):
            candidate = (wrapped_phases[:, i] + 2 * np.pi * k) * (wv / (2 * np.pi))
            zs_candidates[:, i, k] = candidate

    best_depth = np.zeros(N, dtype=np.float32)
    min_std = np.full(N, np.inf, dtype=np.float32)

    for k in range(max_wraps):
        depths_k = zs_candidates[:, :, k]
        std_k = np.std(depths_k, axis=1)
        mean_k = np.mean(depths_k, axis=1)
        better = std_k < min_std
        best_depth[better] = mean_k[better]
        min_std[better] = std_k[better]

    best_depth = np.clip(best_depth, 0, 1.0)

    if orig_shape is not None:
        best_depth = best_depth.reshape(orig_shape)

    print(f"[decode] best_depth min={best_depth.min():.4f}, max={best_depth.max():.4f}")

    return best_depth

def decode_depth_feature_driven(encoded_sincos, roi_mask, wavelengths_roi, wavelengths_bg):
    decoded = np.zeros((height * width,), dtype=np.float32)

    print(f"[decode] roi_mask pixels: {np.sum(roi_mask)} / {roi_mask.size}")

    roi_pixels = encoded_sincos[roi_mask]
    bg_pixels = encoded_sincos[~roi_mask]

    decoded_roi = multiwavelength_phase_unwrap_decode(roi_pixels, wavelengths_roi, max_wraps=max_wraps)
    decoded_bg  = multiwavelength_phase_unwrap_decode(bg_pixels, wavelengths_bg, max_wraps=max_wraps)

    print(f"[decode] decoded_roi min/max: {decoded_roi.min():.4f}/{decoded_roi.max():.4f}")
    print(f"[decode] decoded_bg min/max: {decoded_bg.min():.4f}/{decoded_bg.max():.4f}")

    decoded[roi_mask.flatten()] = decoded_roi
    decoded[~roi_mask.flatten()] = decoded_bg

    return decoded.reshape(height, width)

def save_ply_open3d(filename, depth_mm, color_img=None):
    h, w = depth_mm.shape
    points = []

    for v in range(h):
        for u in range(w):
            z = depth_mm[v, u] / 1000.0
            if z == 0 or not np.isfinite(z):
                continue
            x = float(u) / 1000.0
            y = float(v) / 1000.0

            if color_img is not None:
                b, g, r = color_img[v, u]
                r = int(np.clip(r, 0, 255))
                g = int(np.clip(g, 0, 255))
                b = int(np.clip(b, 0, 255))
                points.append(f"{x} {y} {z} {r} {g} {b}")
            else:
                points.append(f"{x} {y} {z}")

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if color_img is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pt in points:
            f.write(pt + "\n")

    print(f"Saved simple PLY '{filename}' with {len(points)} points.")

def load_depth_from_rawtxt(folder):
    files = sorted(glob.glob(f"{folder}/rawDepth*.txt"),
                   key=lambda x: int(re.findall(r'\d+', x)[-1]))
    all_depths = []
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
            if isinstance(data, dict):
                flat = list(data.values())[0]
            else:
                flat = data
            depth_np = np.array(flat, dtype=np.float32).reshape(height, width)
            all_depths.append(depth_np)
    return all_depths, files
def main():
    # Load all depth maps from rawDepth*.txt files
    all_depths, files = load_depth_from_rawtxt(DEPTH_FOLDER)
    if not all_depths:
        print("No depth files found in folder:", DEPTH_FOLDER)
        return

    # For demo, process only the first depth file (or extend to batch loop as needed)
    depth_map = all_depths[0]

    # Calculate dynamic range same as your UNet training step
    dmin, dmax = np.min(all_depths), np.max(all_depths)
    depth_range = dmax - dmin
    depth_norm = np.clip((depth_map - dmin) / depth_range, 0, 1)

    print(f"Dynamic depth range: min={dmin:.1f} mm, max={dmax:.1f} mm, range={depth_range:.1f} mm")

    # ROI mask using same percentile method as before
    face_thresh = np.percentile(depth_map[depth_map > 0], roi_depth_percentile)
    roi_mask = binary_closing(binary_opening(depth_map < face_thresh, np.ones((5,5))), np.ones((15,15)))

    # Wavelength selection (same as before)
    wavelengths_roi = choose_wavelengths(num=3, min_wl=0.15, max_wl=0.4)
    wavelengths_bg  = choose_wavelengths(num=3, min_wl=0.1,  max_wl=0.35)

    print(f"Wavelengths ROI: {wavelengths_roi}")
    print(f"Wavelengths BG : {wavelengths_bg}")

    # Encode
    encoded_sincos = encode_depth_feature_driven(depth_norm, roi_mask, wavelengths_roi, wavelengths_bg)

    # Save encoded sin channels only as PNG for compression visualization
    rgb_encoded_vis = encoded_sincos[..., 0:6:2]  # sin channels only
    cv2.imwrite(ENCODED_PNG, cv2.cvtColor(rgb_encoded_vis, cv2.COLOR_RGB2BGR))
    print(f"Saved encoded visualization PNG to {ENCODED_PNG}")

    # Save encoded PLY point cloud
    save_ply_open3d(ENCODED_PLY, depth_map, rgb_encoded_vis)

    # Create compressed video from PNG
    subprocess.run([
        "ffmpeg", "-y", "-framerate", "1", "-i", ENCODED_PNG,
        "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p", ENCODED_VIDEO
    ], check=True)
    print(f"Saved encoded video to {ENCODED_VIDEO}")

    # Extract frame back from video as PNG
    subprocess.run([
        "ffmpeg", "-y", "-i", ENCODED_VIDEO, "-frames:v", "1", "-update", "1", VIDEO_FRAME_PNG
    ], check=True)

    rgb_jpeg = cv2.cvtColor(cv2.imread(VIDEO_FRAME_PNG), cv2.COLOR_BGR2RGB)

    # Reconstruct sincos with zeros for cosine channels (middle gray 127)
    encoded_sincos_decoded = np.zeros_like(encoded_sincos)
    encoded_sincos_decoded[..., 0:6:2] = rgb_jpeg
    encoded_sincos_decoded[..., 1:6:2] = 127

    # Decode
    depth_decoded = decode_depth_feature_driven(encoded_sincos_decoded, roi_mask, wavelengths_roi, wavelengths_bg)
    depth_decoded_mm = depth_decoded * depth_range + dmin

    save_ply_open3d(DECODED_PLY, depth_decoded_mm, rgb_jpeg)

    # Error metrics
    rms_roi = np.sqrt(np.mean((depth_map[roi_mask] - depth_decoded_mm[roi_mask])**2))
    rms_bg  = np.sqrt(np.mean((depth_map[~roi_mask] - depth_decoded_mm[~roi_mask])**2))
    mae_roi = np.mean(np.abs(depth_map[roi_mask] - depth_decoded_mm[roi_mask]))
    mae_bg = np.mean(np.abs(depth_map[~roi_mask] - depth_decoded_mm[~roi_mask]))

    print(f"RMS Error in ROI       : {rms_roi:.2f} mm")
    print(f"MAE Error in ROI       : {mae_roi:.2f} mm")
    print(f"RMS Error in Background: {rms_bg:.2f} mm")
    print(f"MAE Error in Background: {mae_bg:.2f} mm")

    # Save plots instead of showing
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plt.title("Original Depth")
    plt.imshow(depth_map, cmap='jet')
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title("Decoded Depth (mm)")
    plt.imshow(depth_decoded_mm, cmap='jet')
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("Error Map (mm)")
    plt.imshow(np.abs(depth_map - depth_decoded_mm), cmap='hot')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"Saved plot to {PLOT_FILE}")

if __name__ == "__main__":
    main()
