# Depth-to-Color Encoding Methods

This project implements multiple methods to encode depth maps (z-values) into RGB images using different color spaces and techniques. These encodings enable efficient compression of depth data with common video codecs.


## Depth-to-Color Encoding under YUV420p Compression

### Why Encode Depth into RGB?

We map depth values into RGB color images to leverage standard video compression (e.g., H.264).  
This allows efficient depth video storage and transmission using widely supported codecs.

---

### What is YUV420p Chroma Subsampling?

YUV420p is a common pixel format used in video codecs like H.264. It reduces color precision to save bandwidth.

- **Y (luma/brightness):** stored for **every pixel**
- **U and V (chroma/color):** shared across **2×2 blocks**  
  → For every 4 pixels:  
    • 4 Y values  
    • 1 U and 1 V value  
  → Chroma resolution is reduced **4:1**

This can significantly affect any data encoded into color channels.

---

### HSV-RGB Encoding for Depth

**How it works:**

- Depth is mapped to **Hue** (0°–360° on the color wheel)
- Saturation and Value are fixed or modulated
- RGB is computed from HSV and saved as an image

**Problem under YUV420p:**

- Hue is stored in chroma channels (U/V), which are downsampled
- Result: **Hue distortion, color bleeding, and depth reconstruction errors**
- Saturation/value may also degrade if encoded in chroma

**Conclusion:**  
🔴 HSV encoding is **fragile** to compression — especially around sharp hue transitions.

---

### LUV-RGB Encoding for Depth

**How it works:**

- Depth is primarily encoded in **L\*** (lightness)
- u\*, v\* (chrominance) encode additional info or orientation
- RGB is derived from LUV using perceptually uniform transformation

**Advantages under YUV420p:**

- **L\*** maps to **luma (Y)** — preserved at full resolution ✅
- u\*, v\* are downsampled — but depth-critical info stays in L\*
- LUV is more resistant to compression due to uniformity

**Conclusion:**  
🟢 LUV encoding **preserves depth better** under YUV420p compression than HSV.



### Summary

| Encoding | Robustness to YUV420p | Depth Channel Placement | Notes |
|----------|------------------------|--------------------------|-------|
| **HSV**  | ❌ Low                | Hue → Chroma (U/V)       | Easily distorted by chroma subsampling |
| **LUV**  | ✅ High               | Lightness → Luma (Y)     | Depth survives compression much better |

---
## Files & Methods Overview

| Script Name           | Encoding Method    | Description                                  |
|-----------------------|--------------------|----------------------------------------------|
| `depth_hsv_rgb.py`    | HSV-RGB            | Encode depth using HSV color mapping.        |
| `depth_luv_rgb.py`    | CIELUV-RGB         | Encode depth in perceptually uniform LUV.   |
| `depth_lab_rgb.py`    | CIELAB-RGB         | Encode depth in perceptually uniform LAB.   |
| `depth_mvd_roi.py`    | MVD + ROI          | Motion vector displacement + ROI encoding.  |
| `depth_n_depth.py`    | N-Depth            | Nonlinear depth quantization method.         |
| `read_depth_methods.py` | Launcher          | Runs all above methods on `zvalues.txt`.     |
| `zvalues.txt`         | Input depth file   | Plain text file with depth values.            |

---

## Prerequisites

Install required Python packages:

```bash
pip install numpy opencv-python
```

## Run for Method 1 (depth_hsv_rgb.py)

```bash
python read_depth_methods.py 
```
and enter 1. When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](1/preview_frame_0.png)

- Two PLY files saved in the `ply_output` folder:
- [frame_000000_decoded.ply](ply_output/frame_000000_decoded.ply)
- [frame_000000_original.ply](ply_output/frame_000000_original.ply)


> To view these `.ply` files, please open them with a 3D viewer such as [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.danielgm.net/cc/).


## Run for Method 2 (depth_luv_rgb.py)

```bash
python read_depth_methods.py 
```
and enter 2. When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](2/viz_frame_000000.png)

- Two PLY files saved in the `ply_output` folder:
- [frame_000000_decoded.ply](frame_000000_decoded.ply)
- [frame_000000_original.ply](frame_000000_original.ply)


## Run for Method 3 (depth_lab_rgb.py)

```bash
python read_depth_methods.py 
```
and enter 3. When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](3/depth_lab_plot.png)

- Two PLY files saved in the `ply_output` folder:
- [frame_000000_decoded.ply](decoded_depth.ply)
- [frame_000000_original.ply](encoded_lab_color.ply)

## Run for Method 4 (depth_mvd_roi.py)

```bash
python read_depth_methods.py 
```
and enter 4. When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](4/depth_error_plot.png)

- Two PLY files saved in the `ply_output` folder:
- [frame_000000_decoded.ply](decoded_cloud.ply)
- [frame_000000_original.ply](encoded_cloud.ply)

## Run for Method 5 (depth_n_depth.py)

```bash
python read_depth_methods.py 
```
- When prompted, enter 5 to select method 5. It will then ask for:
:::Number of epochs (default is 10)
:::Batch size (default is 4) 

- When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](5/plots/eval_plot_000000.png)


### Reference

We use th paper to build this method:
**N-DEPTH: Neural Depth Encoding for Compression-Resilient 3D Video Streaming**  
Stephen Siemonsma and Tyler Bell  
*Department of Electrical and Computer Engineering, University of Iowa*  
Published in: *Electronics 2024, 13(13), 2557*    
📄 [Read the Full Paper](https://doi.org/10.3390/electronics13132557)




## Run for Method 6 (depth_hybrid_lab_hsv.py)

```bash
python read_depth_methods.py 
```
and enter 6. When you run the first method, the following output files are generated:

- A preview image:

![Preview Frame 0](6/depth_hybrid_plot_000.png)

- Two PLY files saved in the `ply_output` folder: 
- [frame_000000_decoded.ply](decoded_ffmpeg_000.ply) – decoded using FFmpeg
- [frame_000000_original.ply](decoded_opencv_000.ply)– decoded using OpenCV