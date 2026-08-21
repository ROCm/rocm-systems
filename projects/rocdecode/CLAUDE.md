# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rocDecode is AMD's high-performance video decode SDK. It provides a C/C++ API to access hardware-accelerated video decoding (VCN engines) on AMD GPUs via VA-API, with HIP for GPU interoperability. Supported codecs: H.265/HEVC (8/10 bit), H.264/AVC (8 bit), AV1 (8/10 bit), VP9 (8/10 bit). Requires AMD GPU gfx908+.

- rocDecode is located at `projects/rocdecode` within the `rocm-systems` repo
- CI is at `.github/workflows/media-libs-ci.yml` (builds via TheRock super-project)
- Python test scripts are in `test/testScripts/`; samples are in `samples/`

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

# Build all samples without running tests (quick compilation check)
./build_samples.sh
```

Tests use a **build-and-test pattern**: CTest configures and builds each sample from its installed location (`${ROCM_PATH}/share/rocdecode/samples/<name>`) into a temp directory, then runs the built binary against test media at `${ROCM_PATH}/share/rocdecode/video/`. The library **must be installed** before tests will work.

## Key CMake Options

- `ROCM_PATH` - ROCm installation path (default: `/opt/rocm` or `$ROCM_PATH` env var)
- `ROCDECODE_ENABLE_ROCPROFILER_REGISTER` - Enable profiling support (default: ON)
- `ROCDECODE_ENABLE_HOST_DECODER` - Build FFmpeg-based software decoder `librocdecode-host.so` (default: ON; only built when FFmpeg is found)
- `ENABLE_EXTENDED_TESTS` - Enable FFmpeg-dependent tests (default: OFF)

## Required Dependencies

HIP, libva, libdrm_amdgpu, pthreads. Optional: FFmpeg (>= 4.0.4) for samples, extended tests, and rocdecode-host. Custom Find modules are in `cmake/` (FindFFmpeg, FindLibva, FindLibdrm_amdgpu — all support TheRock sysdeps paths).

## Test Inventory

**Always-on tests (no FFmpeg needed):**
- `video_decodeRaw-{HEVC,AVC,AV1,VP9}` — raw elementary bitstream decode
- `rocdec_Decode-HEVC` — low-level C API decode (rocdecDecode sample)
- `rocDecode_Negative_API_Tests` — invalid parameter tests for all public API functions

**Extended tests (require FFmpeg + `ENABLE_EXTENDED_TESTS=ON`):**
- `video_decode-{HEVC,AVC,AV1,VP9}` — container-based decode via FFmpeg demuxer
- `video_decodePerf-HEVC` — multi-thread performance decode
- `video_decodeBatch` — batch directory decode
- `video_decodeRGB-HEVC` — decode + YUV-to-RGB color conversion
- `video_decodeMem-HEVC` — memory chunk-based decode
- `video_decodeRGB-Resize` — decode + RGB + resize
- `video_decode-Host-Backend` — FFmpeg software decode (not built under TheRock)

**ASAN support:** The test CMakeLists auto-detects AddressSanitizer and sets `LD_PRELOAD` for `libclang_rt.asan` on each test.

## Architecture

### Decoding Pipeline

```
Input (video file)
  → Demuxing (FFmpeg VideoDemuxer or bitstream reader)
  → Parsing (codec-specific: AVC/HEVC/AV1/VP9)
  → Decoding (VA-API GPU or FFmpeg host)
  → Output (YUV in HIP device/host memory)
  → Optional post-processing (HIP color conversion/resize kernels)
```

### Two Shared Libraries

1. **`librocdecode.so`** — Core GPU-accelerated decoder using VA-API
2. **`librocdecode-host.so`** — Optional FFmpeg-based software decoder (built from `src/rocdecode-host/`)

### Source Layout

- **`api/rocdecode/`** — Public C API headers: `rocdecode.h` (decoder), `rocparser.h` (parser), `roc_bitstream_reader.h` (elementary stream reader)
- **`api/rocdecode/rocdecode_host.h`** — Host decoder API (mirrors GPU API with `*Host` suffix functions)
- **`api/amd_detail/`** — rocprofiler API tracing header
- **`src/rocdecode/`** — Core decoder: `rocdecode_api.cpp` (C entry points) → `RocDecoder` → `VaapiVideoDecoder` (VA-API backend in `vaapi/`)
- **`src/parser/`** — Codec parsers: `rocparser_api.cpp` (C entry points) → `RocVideoParser` base class → `AvcVideoParser`, `HevcVideoParser`, `Av1VideoParser`, `Vp9VideoParser`. Each codec has a `*_defines.h` with codec-specific constants/structs.
- **`src/bit_stream_reader/`** — Elementary stream file reader (H.264 .264, HEVC .265, AV1/VP9 .ivf; no FFmpeg dependency)
- **`src/rocdecode-host/`** — Host decoder: separate CMake target, wraps FFmpeg avcodec
- **`src/amd_detail/`** — rocprofiler API tracing/dispatch
- **`src/commons.h`** — Singleton logger, exception class, RAII function-scope logging macros
- **`utils/`** — High-level utility classes shared by samples (see below)
- **`samples/`** — 10 sample applications demonstrating different decode scenarios
- **`test/`** — CTest definitions, negative API tests, Python conformance/benchmark scripts

### Key Internal Classes

- **`VaContext`** (`src/rocdecode/vaapi/`) — Meyer's singleton managing GPU-to-VA context mapping. Handles DRM render node discovery, GPU UUID mapping, compute partition awareness (SPX/DPX/TPX/QPX/CPX), and VA profile probing. Thread-safe via mutex.
- **`RocDecoder`** (`src/rocdecode/`) — Wraps `VaapiVideoDecoder` + HIP interop (`HipInteropDeviceMem`). Orchestrates decode frame, get frame, reconfigure.
- **`RocVideoParser`** (`src/parser/`) — Abstract base with NAL extraction, EBSP-to-RBSP conversion, Exp-Golomb utilities, SEI parsing, callback dispatch.

### Utils Layer

- **`RocVideoDecoder`** (`utils/rocvideodecode/roc_video_dec.h`) — High-level C++ wrapper tying parser + decoder together. Four output memory modes: `DEV_INTERNAL` (zero-copy interop), `DEV_COPIED`, `HOST_COPIED`, `NOT_MAPPED` (perf test). Static callback trampolines dispatch to member functions. Thread-safe frame queue.
- **`FFMpegVideoDecoder`** (`utils/ffmpegvideodecode/`) — Derived from `RocVideoDecoder`, wraps the host-based FFmpeg decoder.
- **`VideoDemuxer`** (`utils/video_demuxer.h`) — Header-only FFmpeg-based demuxer. Handles H.264/HEVC bitstream filtering, seek, and custom `StreamProvider` for memory-based input.
- **`VideoPostProcess`** (`utils/video_post_process.h`) — Orchestrates YUV-to-RGB color conversion, dispatching to appropriate HIP kernel by surface format and output format.
- **HIP Kernels** (`utils/colorspace_kernels.*`, `utils/resize_kernels.*`) — GPU kernels for color space conversion (10 standards incl. BT.709/601/2020) and bilinear resize.
- **`MD5Generator`** (`utils/md5.h`) — FFmpeg-based MD5 digest of decoded frames for conformance testing.

### API Conventions

- C API with `extern "C"` linkage; functions prefixed `rocDec` (e.g., `rocDecCreateDecoder`)
- Opaque handles (`typedef void *rocDecDecoderHandle`, `typedef void *RocdecVideoParser`)
- Error codes via `rocDecStatus` enum (`ROCDEC_SUCCESS`, `ROCDEC_INVALID_PARAMETER`, `ROCDEC_NOT_SUPPORTED`, etc.)
- Internal C++ uses PascalCase classes, trailing underscore for members (`va_display_`), snake_case locals
- Callback-driven parser: register `pfn_sequence_callback`, `pfn_decode_picture`, `pfn_display_picture`, `pfn_get_sei_message` handlers

### Logging

5-level logging controlled by `ROCDEC_LOG_LEVEL` env var (0=Critical, 1=Error, 2=Warning, 3=Info, 4=Debug). Singleton logger in `src/commons.h`. RAII macros `FunctionEntryLog` / `FunctionEntryLogWithArgs` auto-log function entry/exit with timing.

## Coding Style

- C++17, compiled with `-Wall`
- `#pragma once` for header guards
- MIT license header block required on all C/C++ and CMake files
- Compiler: `amdclang++` from ROCm toolchain
- Release: `-O3 -DNDEBUG -fPIC`; Debug: `-O0 -gdwarf-4` (Valgrind compatible)
- No `.clang-format` or linter config — no auto-formatting enforced

## Validation after code changes

When source files under `api/`, `src/`, `utils/`, or `samples/` have been modified,
offer to run the `/validate` skill (full build + CTest + conformance sweep) so the change
is verified.

This is a soft prompt, not an enforced gate. To keep it from becoming noisy, follow these
rules:

- **Ask once, at the end.** Wait until a logical batch of edits is complete and you are
  wrapping up the task; then ask a single time whether to run `/validate`. Do not ask after
  every individual edit.
- **Only when it matters.** Ask only if files under `api/`/`src/`/`utils/`/`samples/`
  actually changed this session. Documentation-only, test-script-only, or `.claude/`-only
  changes do not need validation.
- **Don't repeat.** If `/validate` has already been run, or the developer has declined the
  offer this session, don't ask again unless further source changes are made afterward.
- **Let the developer decide.** If they decline, proceed without running it.

## Performance regression checking

The `/perf-check` skill measures decode FPS with the `videoDecodePerf` sample and compares
it against a GPU-specific baseline to catch performance regressions. Unlike `/validate`,
this is **on-demand** (run it when a change could affect decode throughput) — not a soft
prompt after every change — because it needs perf streams and an internal baseline that not
every environment has.

- It detects the local GPU (`amd-smi`/`rocm-smi`/KFD) and maps it to the matching baseline
  column (`MI250X`, `MI300X`, `MI300A`, `MI350`, `MI355`, `Navi31`, `Navi48`).
- Streams and the baseline (`rocDecode_perf_baseline.html`, downloaded from the internal
  SharePoint site) are located via `ROCDECODE_PERF_DIR` (default `$HOME/rocDecodePerformance`).
  A stream is a regression if its Avg FPS is >5% below baseline (`ROCDECODE_PERF_TOLERANCE`).
- `/perf-check` runs the full sweep; `/perf-check quick` measures one stream per leaf
  subfolder capped at ≤4K for a faster sanity check. Each run reports its own elapsed time.

See `test/README.md` for details.
