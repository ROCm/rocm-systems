# Video Decode

## Overview

This example benchmarks video decoding performance using the AMD ROCDecode library with VCN (Video Core Next) hardware acceleration. It uses a thread pool to decode video files in parallel batches, leveraging FFmpeg for demuxing and the ROCDecode API for GPU-accelerated frame decoding. The benchmark measures decoding throughput and is useful for profiling hardware video decoder utilization, GPU memory management during media workloads, and multi-threaded decode pipeline performance.

## Source Files

- `videodecodebatch.cpp` - Thread pool implementation with job queue for parallel video decoding, worker thread management, and performance measurement.
- `roc_video_dec.cpp` / `roc_video_dec.h` - ROCDecode video decoder wrapper with GPU memory management.
- `video_demuxer.h` - FFmpeg-based video demuxing utilities.
- `common.h` - Shared definitions and helpers.

## Prerequisites

- CMake 3.21+
- HIP runtime
- ROCDecode library
- FFmpeg libraries (libavcodec, libavformat, libavutil)
- Sample video files (copied from ROCm installation at build time)

## Building

**Standalone build:**

```bash
cmake -B <build_dir> -S <project_root>/examples/videodecode -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir>
```

**As part of the examples suite:**

```bash
cmake -B <build_dir> -S <project_root>/examplse/ -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build <build_dir> --target videodecode
```

## Running

```bash
# Decode a video file
./videodecode -i /path/to/video.mp4

# Specify GPU and thread count
./videodecode -i /path/to/video.mp4 -d 0
```

## Profiling with rocprofiler-systems

```bash
rocprof-sys-run -- ./videodecode -i /path/to/video.mp4
```

### Recommended Configuration

| Variable | Value | Purpose |
|----------|-------|---------|
| `ROCPROFSYS_ROCM_DOMAINS` | `hip_runtime_api,kernel_dispatch,memory_copy` | Trace HIP API and GPU operations |
| `ROCPROFSYS_TRACE` | `true` | Generate Perfetto trace for timeline analysis |
| `ROCPROFSYS_PROFILE` | `true` | Generate call-stack profile |

```bash
rocprof-sys-run \
    -e ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy \
    -e ROCPROFSYS_TRACE=true \
    -- ./videodecode -i /path/to/video.mp4
```
