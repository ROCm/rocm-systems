# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rocDecode is AMD's high-performance video decode SDK. It provides a C/C++ API to access hardware-accelerated video decoding (VCN engines) on AMD GPUs via VA-API, with HIP for GPU interoperability. Supported codecs: H.265/HEVC (8/10 bit), H.264/AVC (8 bit), AV1 (8/10/12 bit), VP9 (8-bit and 10-bit). Requires AMD GPU gfx908+.

- rocDecode is located at projects/rocdecode
- rocDecode CI is located at .github/workflows/media-libs-ci.yml
- rocDecode python test scripts are in projects/rocdecode/test and samples are located in projects/rocdecode/samples

## Build Commands

```bash
# Standard build (uses amdclang++ from ROCM_PATH, defaults to /opt/rocm)
mkdir build && cd build
cmake ..
make -j$(nproc)

# Release build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Install (required before running tests)
sudo make install

# Run all tests (requires install first)
make test ARGS="-VV"
# or: ctest --extra-verbose --output-on-failure

# Run a single test by name
ctest -R video_decodeRaw-HEVC -VV

# Run extended tests (requires FFmpeg)
cmake .. -DENABLE_EXTENDED_TESTS=ON
make test ARGS="-VV"
```

Tests use a build-and-test pattern: CTest builds each sample in a temp directory and runs it against installed test data at `${ROCM_PATH}/share/rocdecode/video/`. The library must be installed before tests will work.

## Key CMake Options

- `ROCM_PATH` - ROCm installation path (default: `/opt/rocm` or 
- `ROCDECODE_ENABLE_ROCPROFILER_REGISTER` - Enable profiling support (default: ON)
- `ROCDECODE_ENABLE_HOST_DECODER` - Build FFmpeg-based software decoder `librocdecode-host.so` (default: ON; only built when FFmpeg is found)
- `ENABLE_EXTENDED_TESTS` - Enable FFmpeg-dependent tests (default: OFF)

## Required Dependencies

HIP, libva (>= 1.22), libdrm_amdgpu, pthreads. Optional: FFmpeg (>= 4.0.4) for samples, extended tests, and rocdecode-host. Custom Find modules are in `cmake/`.

## Architecture

### Decoding Pipeline

Input (video file) -> Demuxing (FFmpeg or bitstream reader) -> Parsing (codec-specific) -> Decoding (VA-API GPU or FFmpeg host) -> Output (YUV in HIP device/host memory) -> Optional post-processing (HIP color conversion/resize kernels)

### Two Shared Libraries

1. **`librocdecode.so`** - Core GPU-accelerated decoder using VA-API
2. **`librocdecode-host.so`** - Optional FFmpeg-based software decoder (built from `src/rocdecode-host/`)

### Source Layout

- **`api/`** - Public C API headers (`rocdecode.h`, `rocparser.h`, `roc_bitstream_reader.h`, `rocdecode_host.h`)
- **`src/rocdecode/`** - Core decoder: `rocdecode_api.cpp` (C entry points) -> `RocDecoder` -> `VaapiVideoDecoder` (VA-API backend in `vaapi/`)
- **`src/parser/`** - Codec parsers: `rocparser_api.cpp` (C entry points) -> `RocVideoParser` base -> `AvcVideoParser`, `HevcVideoParser`, `Av1VideoParser`, `Vp9VideoParser`
- **`src/bit_stream_reader/`** - Elementary stream file reader (no FFmpeg dependency)
- **`src/rocdecode-host/`** - Host decoder: separate CMake target, wraps FFmpeg avcodec
- **`src/amd_detail/`** - rocprofiler API tracing/dispatch
- **`utils/`** - High-level utility classes shared by samples: `RocVideoDecoder` (ties parser+decoder together), `VideoDemuxer` (FFmpeg wrapper), HIP kernels for color conversion and resize
- **`samples/`** - 10 sample applications demonstrating different decode scenarios
- **`test/`** - CTest definitions; `rocDecodeNegativeApiTests/` for negative API tests; `testScripts/` for Python-based conformance and sample test runners

### API Conventions

- C API with `extern "C"` linkage; functions prefixed `rocDec` (e.g., `rocDecCreateDecoder`)
- Opaque handles (`typedef void *rocDecDecoderHandle`)
- Error codes via `rocDecStatus` enum
- Internal C++ uses PascalCase classes, trailing underscore for members (`va_display_`), snake_case locals
- Callback-driven parser: register `pfn_sequence_callback`, `pfn_decode_picture`, `pfn_display_picture` handlers

### Logging

5-level logging controlled by `ROCDEC_LOG_LEVEL` env var (0=Critical, 1=Error, 2=Warning, 3=Info, 4=Debug). Singleton logger in `src/commons.h`.

## Coding Style

- C++17, compiled with `-Wall`
- `#pragma once` for header guards
- MIT license header block required on all C/C++ and CMake files
- Compiler: `amdclang++` from ROCm toolchain
- Release: `-O3 -DNDEBUG -fPIC`; Debug: `-O0 -gdwarf-4` (Valgrind compatible)
