# Depth-to-Color Encoding Methods

This project implements multiple methods to encode depth maps (z-values) into RGB images using different color spaces and techniques. These encodings enable efficient compression of depth data with common video codecs.

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

## Output for Method 1 (depth_hsv_rgb.py)

When you run the first method, the following output files are generated:

- A preview image:
  - `1_output/preview_frame_0.png`

- Two PLY files saved in the `ply_output` folder:
  - `frame_000000_decoded.ply`
  - `frame_000000_original.ply`

These files represent the encoded depth preview and the point clouds before and after decoding.
