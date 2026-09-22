# Video decode RGB sample

This sample demonstrates decoding and optionally color-converting using the FFMpeg demuxer, the rocDecode API, and custom HIP kernels.

The FFmpeg demuxer is used to get individual frames that are then decoded using rocDecode API and optionally color-converted using custom HIP kernels on AMD hardware.

Decoded YUV output is converted to either RGB or BGR formats (24bit, 32bit, 48bit, 64bit) in separate threads so that both the VCN hardware and the compute engine can run in parallel.

HIP kernels are used for color conversion. After the frame has been decoded, the `ColorSpaceConversionThread` is notified and is used for post-processing.

## Prerequisites:

* Install [rocDecode](https://rocm.docs.amd.com/projects/rocDecode/en/latest/install/rocDecode-build-and-install.html)

* [FFMPEG](https://ffmpeg.org/about.html)

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```

## Build

```shell
mkdir video_decode_rgb_sample && cd video_decode_rgb_sample
cmake ../
make -j
```

## Run

```shell
./videodecodergb -i <input video file>  [options]
```

### Options

| Flag | Argument | Description | Default |
|------|----------|-------------|---------|
| `-i` | `<path>` | Input video file path (required) | |
| `-o` | `<path>` | Output file path for decoded frames | (no output) |
| `-d` | `<id>` | GPU device ID | `0` |
| `-of` | `<format>` | Output format: `native`, `bgr`, `bgr48`, `rgb`, `rgb48`, `bgra`, `bgra64`, `rgba`, `rgba64` | `native` |
| `-cs` | `<standard>` | Color standard: `bt709`, `fcc`, `bt470`, `bt601`, `smpte240m`, `bt2020`, `bt2020c` | `bt709` |
| `-resize` | `<W>x<H>` | Resize output (width and height must be even values) | (no resize) |
| `-crop` | `<l,t,r,b>` | Crop rectangle (width and height must be even values) | (no crop) |
| `-disp_delay` | `<n>` | Number of frames to delay before display | `1` |
| `-f` | `<n>` | Number of frames to decode (`0` = entire stream) | `0` |

### Color conversion

The YUV to RGB conversion is performed on the GPU using HIP kernels defined in `utils/colorspace_kernels.cpp`. The conversion applies the standard-specific luma weights and studio-range scaling:

- **Luma (Y)** range: 16-235 (8-bit) or 64-940 (10-bit), scaled by `max / (white - black)`
- **Chroma (Cb/Cr)** range: 16-240 (8-bit) or 64-960 (10-bit), scaled by `max / (white_c - black)`

The following color standards are supported, each defining different luma weight coefficients (Kr, Kb):

| Standard | Kr | Kb |
|----------|------|------|
| BT.709 | 0.2126 | 0.0722 |
| FCC | 0.30 | 0.11 |
| BT.470 | 0.2990 | 0.1140 |
| BT.601 | 0.2990 | 0.1140 |
| SMPTE 240M | 0.212 | 0.087 |
| BT.2020 | 0.2627 | 0.0593 |

#### YUV to RGB formula (studio range)

Given a YUV pixel with studio-range values, the conversion to full-range RGB is:

```
Kg = 1 - Kr - Kb

fy = Y  - low       (low = 16 for 8-bit, 64<<6 for 10-bit P016)
fu = Cb - mid       (mid = 128 for 8-bit, 1<<15 for 10-bit P016)
fv = Cr - mid

Base matrix (unscaled):

       | 1    0                             (1-Kr)/0.5                       |
base = | 1   -Kb*(1-Kb) / (0.5*(1-Kb-Kr))  -Kr*(1-Kr) / (0.5*(1-Kb-Kr))      |
       | 1    (1-Kb)/0.5                    0                                |

Column-wise scaling (luma and chroma scaled independently):

  Sy = max / (white - black)        luma scale
  Sc = max / (white_c - black)      chroma scale

  mat[i][0] = base[i][0] * Sy       for i = 0..2
  mat[i][1] = base[i][1] * Sc       for i = 0..2
  mat[i][2] = base[i][2] * Sc       for i = 0..2

Conversion:

  R = mat[0][0]*fy + mat[0][1]*fu + mat[0][2]*fv
  G = mat[1][0]*fy + mat[1][1]*fu + mat[1][2]*fv
  B = mat[2][0]*fy + mat[2][1]*fu + mat[2][2]*fv

  R, G, B = clamp(R, G, B, 0, maxf)

where maxf is the full storage range for the YUV unit type
(255 for 8-bit, 65535 for 16-bit P016).
```

### Validation with FFmpeg

A test script is provided to validate the GPU color conversion against FFmpeg's CPU-based conversion:

```shell
python3 test/testScripts/run_rocDecode_colorConvert.py \
    --rocdecode_exe ./videodecodergb \
    --input_file input.mp4 \
    --num_frames 3 \
    --tolerance 2
```

The script decodes frames with both `videodecodergb` and `ffmpeg` using each color standard, then compares the raw RGB output pixel-by-pixel. A per-channel tolerance (default ±2) accounts for floating-point rounding differences between GPU and CPU implementations.

The script automatically filters standards by bit depth:
- **8-bit streams**: tests bt709, fcc, bt470, bt601, smpte240m (skips bt2020/bt2020c)
- **10-bit streams**: tests bt2020, bt2020c (skips 8-bit standards)

See `test/testScripts/run_rocDecode_colorConvert.py --help` for all options.