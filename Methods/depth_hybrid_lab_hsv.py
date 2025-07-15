import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import subprocess
import time
import glob
import json
import re
width, height = 640, 576
OUTPUT_FOLDER = "6"
DEPTH_FOLDER = "depth"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)



def encode_depth_hybrid_lab_hsv(z_image):
    z_min, z_max = z_image.min(), z_image.max()
    norm = (z_image - z_min) / (z_max - z_min + 1e-6)

    L = np.clip(norm * 70 + 20, 0, 100)

    hue = (norm * 179).astype(np.uint8)
    saturation = np.full_like(hue, 255, dtype=np.uint8)
    value = np.full_like(hue, 255, dtype=np.uint8)
    hsv = cv2.merge([hue, saturation, value])
    hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    a = 40 * hsv_bgr[..., 0]
    b = 40 * hsv_bgr[..., 1]

    lab_float = np.stack([L, a, b], axis=-1).astype(np.float32)
    lab_8u = np.zeros_like(lab_float, dtype=np.uint8)
    lab_8u[..., 0] = np.clip(lab_float[..., 0] * 255 / 100, 0, 255).astype(np.uint8)
    lab_8u[..., 1] = np.clip(lab_float[..., 1] + 128, 0, 255).astype(np.uint8)
    lab_8u[..., 2] = np.clip(lab_float[..., 2] + 128, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab_8u, cv2.COLOR_LAB2BGR)

def decode_depth_hybrid_lab_hsv(bgr_img, original_min, original_max):
    lab_8u = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab_8u[..., 0].astype(np.float32) * 100 / 255
    a = lab_8u[..., 1].astype(np.float32) - 128
    b = lab_8u[..., 2].astype(np.float32) - 128

    norm_L = np.clip((L - 20) / 70, 0, 1)

    hsv_bgr_0 = a / 40.0
    hsv_bgr_1 = b / 40.0
    hsv_bgr_0 = np.clip(hsv_bgr_0, 0, 1)
    hsv_bgr_1 = np.clip(hsv_bgr_1, 0, 1)
    hsv_bgr_2 = np.ones_like(hsv_bgr_0)

    hsv_bgr = np.stack([hsv_bgr_0, hsv_bgr_1, hsv_bgr_2], axis=-1)
    hsv_bgr_uint8 = (hsv_bgr * 255).astype(np.uint8)
    hsv_img = cv2.cvtColor(hsv_bgr_uint8, cv2.COLOR_BGR2HSV)
    hue = hsv_img[..., 0].astype(np.float32) / 179.0

    residual = (hue - 0.5) * (1.0 / 179.0)
    norm_depth = np.clip(norm_L + residual, 0, 1)
    recon_depth = norm_depth * (original_max - original_min) + original_min
    return recon_depth.astype(np.float32)

def write_video_opencv(image, filename, fps=1):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open.")
    writer.write(image)
    writer.release()

def write_png_and_ffmpeg(image, folder, video_filename):
    os.makedirs(folder, exist_ok=True)
    png_path = os.path.join(folder, "frame_0001.png")
    cv2.imwrite(png_path, image)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", "1",
        "-i", f"{folder}/frame_%04d.png",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        video_filename
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def read_video_frame(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Failed to read frame from {filename}")
    cap.release()
    return frame if ret else None

def load_zvalues(filename):
    with open(filename, "r") as f:
        content = f.read()
    z_values = [int(x.strip()) for x in content.strip().split(',') if x.strip().isdigit()]
    expected_size = width * height
    if len(z_values) < expected_size:
        z_values = np.pad(z_values, (0, expected_size - len(z_values)), mode='constant')
    elif len(z_values) > expected_size:
        z_values = z_values[:expected_size]
    return np.array(z_values, dtype=np.uint32)

def compute_error_metrics(original, reconstructed):
    error_img = np.abs(original.astype(np.int32) - reconstructed.astype(np.int32))
    mean_error = error_img.mean()
    return error_img, mean_error

def save_depth_as_ply(depth_image, filename):
    h, w = depth_image.shape
    points = []
    for v in range(h):
        for u in range(w):
            z = depth_image[v, u]
            if z == 0:
                continue
            x = u
            y = v
            points.append((x, y, z))

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('end_header\n')
        for p in points:
            f.write(f'{p[0]} {p[1]} {p[2]}\n')
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
    all_depths, files = load_depth_from_rawtxt(DEPTH_FOLDER)
    print(f"Loaded {len(all_depths)} depth files.")

    for i, (z_image_smoothed, fname) in enumerate(zip(all_depths, files)):
        print(f"\nProcessing {os.path.basename(fname)} ({i + 1}/{len(all_depths)})")

        encoded_hybrid = encode_depth_hybrid_lab_hsv(z_image_smoothed)

        # --- OpenCV ---
        t0 = time.time()
        opencv_video = os.path.join(OUTPUT_FOLDER, f"depth_hybrid_opencv_{i:03}.mp4")
        write_video_opencv(encoded_hybrid, opencv_video)
        t1 = time.time()
        decoded_opencv = read_video_frame(opencv_video)
        t2 = time.time()
        recon_opencv = decode_depth_hybrid_lab_hsv(decoded_opencv, z_image_smoothed.min(), z_image_smoothed.max())
        error_opencv, mae_opencv = compute_error_metrics(z_image_smoothed, recon_opencv)
        size_opencv_kb = os.path.getsize(opencv_video) / 1024

        # --- FFmpeg ---
        temp_folder = os.path.join(OUTPUT_FOLDER, f"temp_hybrid_frames_{i:03}")
        t3 = time.time()
        ffmpeg_video = os.path.join(OUTPUT_FOLDER, f"depth_hybrid_ffmpeg_{i:03}.mp4")
        write_png_and_ffmpeg(encoded_hybrid, temp_folder, ffmpeg_video)
        t4 = time.time()
        decoded_ffmpeg = read_video_frame(ffmpeg_video)
        t5 = time.time()
        recon_ffmpeg = decode_depth_hybrid_lab_hsv(decoded_ffmpeg, z_image_smoothed.min(), z_image_smoothed.max())
        error_ffmpeg, mae_ffmpeg = compute_error_metrics(z_image_smoothed, recon_ffmpeg)
        size_ffmpeg_kb = os.path.getsize(ffmpeg_video) / 1024

        # --- Save PLY ---
        ply_path_opencv = os.path.join(OUTPUT_FOLDER, f"decoded_opencv_{i:03}.ply")
        ply_path_ffmpeg = os.path.join(OUTPUT_FOLDER, f"decoded_ffmpeg_{i:03}.ply")
        save_depth_as_ply(recon_opencv, ply_path_opencv)
        save_depth_as_ply(recon_ffmpeg, ply_path_ffmpeg)

        # --- Print Summary ---
        print("\n--- Benchmark Summary ---")
        print(f"OpenCV   | MAE={mae_opencv:.2f} | Enc={1000*(t1 - t0):.2f}ms | Dec={1000*(t2 - t1):.2f}ms | Size={size_opencv_kb:.2f} KB")
        print(f"FFmpeg   | MAE={mae_ffmpeg:.2f} | Enc={1000*(t4 - t3):.2f}ms | Dec={1000*(t5 - t4):.2f}ms | Size={size_ffmpeg_kb:.2f} KB")

        # --- Plot ---
        plt.figure(figsize=(22, 10))
        zoom_slice = np.s_[100:150, 100:150]
        titles = [
            ("Original Depth", z_image_smoothed, 'gray'),
            ("Hybrid Encoded (OpenCV)", decoded_opencv, None),
            ("Hybrid Decoded (OpenCV)", recon_opencv, 'gray'),
            (f"OpenCV Error Map\nMAE={mae_opencv:.2f}", error_opencv, 'hot'),
            ("Hybrid Encoded (FFmpeg)", decoded_ffmpeg, None),
            ("Hybrid Decoded (FFmpeg)", recon_ffmpeg, 'gray'),
            (f"FFmpeg Error Map\nMAE={mae_ffmpeg:.2f}", error_ffmpeg, 'hot'),
            ("OpenCV-FFmpeg Diff", cv2.absdiff(decoded_opencv, decoded_ffmpeg), None),
        ]

        for j, (title, img, cmap) in enumerate(titles, 1):
            plt.subplot(2, 5, j)
            if cmap:
                im = plt.imshow(img, cmap=cmap)
                if 'Error' in title:
                    plt.colorbar(im, fraction=0.046, pad=0.04)
            else:
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(title)
            plt.axis('off')

        # Zoomed views
        plt.subplot(2, 5, 9)
        plt.imshow(decoded_opencv[zoom_slice])
        plt.title("Zoom OpenCV")
        plt.axis('off')

        plt.subplot(2, 5, 10)
        plt.imshow(decoded_ffmpeg[zoom_slice])
        plt.title("Zoom FFmpeg")
        plt.axis('off')

        plt.tight_layout()
        plot_file = os.path.join(OUTPUT_FOLDER, f"depth_hybrid_plot_{i:03}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()
