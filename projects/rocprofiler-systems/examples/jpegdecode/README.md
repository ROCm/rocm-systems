# JPEG Decode

## Overview

This example benchmarks JPEG image decoding performance using the AMD rocJPEG library for GPU-accelerated JPEG decompression. It batch-decodes JPEG images on a selected GPU device, measuring throughput in images per second and megapixels per second. The program detects JPEG properties (chroma subsampling, component channels) and supports optional ROI (region of interest) decoding. This is useful for profiling GPU media pipeline performance and VCN (Video Core Next) hardware decoder utilization.

## Source Files

- `jpegdecodeperf.cpp` - Main benchmark with `DecodeImages()` function for batch processing, JPEG property detection, throughput measurement, and GPU device selection.
- `rocjpeg_samples_utils.h` - Utility functions for rocJPEG sample programs.

## Prerequisites

- CMake 3.21+
- HIP runtime
- rocJPEG library
- rocprofiler-register
- Sample JPEG images (copied from ROCm installation at build time)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/jpegdecode -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examples/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target jpegdecode
```

## Running

```bash
# Decode sample images on default GPU
./jpegdecode -i /path/to/jpeg/images/ -b 4

# Specify GPU device
./jpegdecode -i /path/to/jpeg/images/ -b 8 -d 1
```

**Key parameters:**

| Parameter | Description |
|-----------|-------------|
| `-i` | Input directory containing JPEG files |
| `-b` | Batch size (images per decode batch) |
| `-d` | GPU device ID |
| `-o` | Output directory for decoded images (optional) |

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./jpegdecode -i /path/to/images/ -b 4
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API and GPU operations |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./jpegdecode -i /path/to/images/ -b 4
```
