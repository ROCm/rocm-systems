# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rocDecode is a high-performance video decode SDK for AMD GPUs using the ROCm platform. It provides GPU-accelerated video decoding for H.265 (HEVC), H.264 (AVC), AV1, and VP9 codecs through VA-API integration.

**Requirements:**
- ROCm via TheRock (7.12+)
- gfx908 or higher GPU
- FFmpeg dev libraries (libavcodec-dev, libavformat-dev, libavutil-dev) for samples and tests
- C++17 compiler (AMD Clang++)

## Build Commands

### Standard Build
```bash
mkdir build && cd build
cmake ..
make -j8
sudo make install
```

### Build Configuration
- Default build type: Release
- Default install path: `/opt/rocm` (or `$ROCM_PATH`)
- Compiler: AMD Clang++ at `${ROCM_PATH}/lib/llvm/bin/amdclang++`

### Build Options
- `ROCDECODE_ENABLE_ROCPROFILER_REGISTER`: Enable rocprofiler-register support (default: ON)
- `ROCDECODE_ENABLE_HOST_DECODER`: Enable FFmpeg-based host decoder support (default: ON)
- `ENABLE_EXTENDED_TESTS`: Enable extended FFmpeg-based tests (default: OFF)

Example with custom options:
```bash
cmake -DROCDECODE_ENABLE_HOST_DECODER=OFF ..
```

## Testing

### Run All Tests
```bash
cd build
make test
```

For verbose output:
```bash
make test ARGS="-VV"
```

### Test Requirements
- FFmpeg dev libraries must be installed
- Extended tests require `ENABLE_EXTENDED_TESTS=ON` at configure time
- Core tests work without FFmpeg (videoDecodeRaw tests)

### Manual Test Verification
After installation, verify using sample application:
```bash
mkdir rocdecode-sample && cd rocdecode-sample
cmake /opt/rocm/share/rocdecode/samples/videoDecode/
make -j8
./videodecode -i /opt/rocm/share/rocdecode/video/AMD_driving_virtual_20-H265.mp4
```

Or using CTest:
```bash
mkdir rocdecode-test && cd rocdecode-test
cmake /opt/rocm/share/rocdecode/test/
ctest -VV
```

## Architecture

### Core Components

**rocdecode.so** - Main library with three subsystems:

1. **Parser** (`src/parser/`):
   - Codec-specific parsers: AVC, HEVC, AV1, VP9
   - Parses bitstreams and extracts frame parameters
   - Each codec has dedicated parser implementation (e.g., `avc_parser.cpp`, `hevc_parser.cpp`)
   - Unified interface through `RocVideoParser` class

2. **Decoder** (`src/rocdecode/`):
   - GPU-accelerated decoding via VA-API (`src/rocdecode/vaapi/`)
   - Interfaces with AMD VCN hardware decoder
   - `RocDecoder` class manages decode sessions
   - Uses libva and libdrm_amdgpu for hardware access

3. **Bitstream Reader** (`src/bit_stream_reader/`):
   - Elementary stream reader for raw bitstream formats
   - Handles .264, .265, .ivf files

**rocdecode-host.so** (optional) - CPU-based decoding fallback using FFmpeg avcodec (requires `ROCDECODE_ENABLE_HOST_DECODER=ON`)

### Public API Headers

Located in `api/rocdecode/`:
- `rocdecode.h` - Main decoder API
- `rocparser.h` - Video parser API
- `roc_bitstream_reader.h` - Bitstream reading utilities
- `rocdecode_host.h` - Host decoder API (optional)

### Utilities

`utils/` directory contains helper utilities:
- `video_demuxer.h` - FFmpeg-based demuxing wrapper
- `colorspace_kernels.cpp/h` - HIP kernels for YUV-to-RGB conversion
- `resize_kernels.cpp/h` - HIP kernels for frame resizing
- `video_post_process.h` - Post-processing pipeline utilities
- `rocvideodecode/` - High-level wrapper classes

### Dependencies

Required:
- HIP runtime (`hip::host`)
- Libva >= 1.22 (VA-API)
- libdrm_amdgpu (AMD DRM driver)
- Threads (pthread)

Optional:
- rocprofiler-register (for profiling integration)
- FFmpeg (for samples, tests, and host decoder)

### TheRock Integration

rocDecode detects TheRock installations by checking for `${ROCM_PATH}/lib/rocm_sysdeps/lib`. When present:
- RPATH is set to include `$ORIGIN/rocm_sysdeps/lib`
- Dependencies are resolved from TheRock's vendored libraries

## Sample Applications

Located in `samples/`, each demonstrates different use cases:

- **videoDecode**: Basic single-stream decoding with FFmpeg demuxer
- **videoDecodeRaw**: Decode raw elementary streams (.264, .265, .ivf)
- **videoDecodeBatch**: Multi-threaded batch decoding (max 64 threads)
- **videoDecodeMultiFiles**: Decode multiple files with reconfigure (same codec, different resolutions)
- **videoDecodePerf**: Performance testing with parallel decode threads
- **videoDecodeRGB**: Decoding + YUV-to-RGB conversion using HIP kernels
- **videoDecodeMem**: Chunk-by-chunk streaming decode
- **rocdecDecode**: Decode from individual frame files

All samples use common utilities from `utils/` and `samples/common.h`.

## Development Workflow

### TheRock Build Environment
When built as part of TheRock, all core dependencies (HIP, Clang++, Libva, Libdrm) are provided automatically. Standalone builds require manual ROCm installation.

### Codec Support
When adding codec support, implement:
1. Parser in `src/parser/` (inherit from base parser)
2. VA-API integration in `src/rocdecode/vaapi/`
3. Bitstream format support in `src/bit_stream_reader/` if needed
4. Test cases in `test/CMakeLists.txt`

### Samples
Samples are installed to `${ROCM_PATH}/share/rocdecode/samples` and can be built standalone against installed rocDecode. Each sample has its own CMakeLists.txt and can be used as external project templates.
