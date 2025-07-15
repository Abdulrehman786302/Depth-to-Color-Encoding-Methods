import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage import gaussian_filter

# --- CONFIG ---
width, height = 640, 576
base_folder = "3"
video_filename = os.path.join(base_folder, "depth_lab_encoded.mp4")
plot_filename = os.path.join(base_folder, "depth_lab_plot.png")
depth_folder = "depth"

os.makedirs(base_folder, exist_ok=True)

def encode_depth_to_full_lab(z_image):
    norm = (z_image - z_image.min()) / (z_image.max() - z_image.min() + 1e-6)
    L = np.clip(norm * 70 + 20, 0, 100)
    a = 40 * np.sin(norm * 2 * np.pi)
    b = 40 * np.cos(norm * 2 * np.pi)
    lab_float = np.stack([L, a, b], axis=-1).astype(np.float32)
    lab_8u = np.zeros_like(lab_float, dtype=np.uint8)
    lab_8u[..., 0] = np.clip(lab_float[..., 0] * 255 / 100, 0, 255).astype(np.uint8)
    lab_8u[..., 1] = np.clip(lab_float[..., 1] + 128, 0, 255).astype(np.uint8)
    lab_8u[..., 2] = np.clip(lab_float[..., 2] + 128, 0, 255).astype(np.uint8)
    bgr_img = cv2.cvtColor(lab_8u, cv2.COLOR_LAB2BGR)
    return bgr_img

def decode_full_lab_to_depth(bgr_img, original_min, original_max):
    lab_8u = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    L = lab_8u[..., 0].astype(np.float32) * 100 / 255
    norm_L = (L - 20) / 70
    norm_L = np.clip(norm_L, 0, 1)
    recon_depth = norm_L * (original_max - original_min) + original_min
    return recon_depth.astype(np.float32)

def write_video(image, filename, fps=1):
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height), isColor=True)
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open.")
    writer.write(image)
    writer.release()

def read_video_frame(filename):
    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Failed to read video frame from {filename}")
    return frame
def save_depth_as_ply(depth, filename):
    """
    Save depth image as a simple PLY point cloud.
    Each pixel becomes a vertex with (x, y, z=depth).
    """
    height, width = depth.shape
    points = []

    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if z == 0:  # skip zero depth
                continue
            points.append([u, v, z])

    points = np.array(points)

    with open(filename, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write points
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

    print(f"PLY saved: {filename}")
def save_color_image_as_ply(color_img, filename):
    """
    Save a color image as a PLY point cloud:
    - Each pixel is a vertex with (x, y, z=0)
    - RGB color from the image
    """
    height, width, _ = color_img.shape
    points = []

    for v in range(height):
        for u in range(width):
            b, g, r = color_img[v, u]
            points.append([u, v, 0, r, g, b])  # zero z because no depth here

    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n")

    print(f"Encoded color PLY saved: {filename}")

def load_zvalues(path):
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        flat = list(data.values())[0]
    else:
        flat = data
    return np.array(flat, dtype=np.float32).reshape((height, width))

def main():
    z_image = load_zvalues(os.path.join(depth_folder, "rawDepth0.txt"))
    z_image_smoothed = gaussian_filter(z_image, sigma=1)

    # Encode to LAB-BGR and write to video
    encoded_bgr = encode_depth_to_full_lab(z_image_smoothed)
    write_video(encoded_bgr, video_filename)
    encoded_color_ply = os.path.join(base_folder, "encoded_lab_color.ply")
    save_color_image_as_ply(encoded_bgr, encoded_color_ply)

    # Decode back from video
    decoded_bgr = read_video_frame(video_filename)
    recon_depth = decode_full_lab_to_depth(decoded_bgr, z_image_smoothed.min(), z_image_smoothed.max())

    # Compute error
    error_img = np.abs(z_image_smoothed - recon_depth)
    mean_error = error_img.mean()
    print(f"Mean Absolute Error: {mean_error:.2f}")
    ply_filename = os.path.join(base_folder, "decoded_depth.ply")
    save_depth_as_ply(recon_depth, ply_filename)
    # Plot and save
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(z_image_smoothed, cmap='gray')
    plt.title("Original Depth")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(encoded_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Encoded LAB-BGR Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(recon_depth, cmap='gray')
    plt.title("Decoded Depth Image")
    plt.axis('off')


    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close()

    print(f"Output video saved to: {video_filename}")
    print(f"Plot image saved to: {plot_filename}")

if __name__ == "__main__":
    main()
