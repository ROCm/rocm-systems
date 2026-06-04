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

---

## Windows PAL Backend

The Windows backend (`src/rocdecode/pal/`) uses AMD PAL (Platform Abstraction Library) to drive the VCN hardware decode engine directly, mirroring the Linux VA-API backend in structure and semantics.

### Build (Windows)

PAL is pulled from the drivers tree. `UVD_INCLUDE_DIR` points to the directory containing `drv_uvd_if.h` (the UVD firmware codec structures).

```cmd
mkdir build && cd build
cmake -DPAL_ROOT=c:\github\drivers-amd-main\drivers\build\native\Debug\x64\pal\package ^
      -DUVD_INCLUDE_DIR=c:\github\drivers-amd-main\drivers\uvdfwlib\uvdfw_inc\ ^
      -DCMAKE_BUILD_TYPE=Debug ..
cmake --build . --config Debug
cmake --install . --config Debug
```

### Run the rocdecDecode sample

```cmd
cd samples\rocdecdecode\
mkdir build && cd build
cmake --build . --config Debug
.\Debug\rocdecdecode.exe -i c:\opt\rocm\share\rocdecode\frames
```

### Architecture

`RocDecoder` (`src/rocdecode/roc_decoder.cpp`) is the platform switchboard — it owns either `VaapiVideoDecoder` (Linux) or `PalVideoDecoder` (Windows) and forwards all calls via `#ifdef ROCDECODE_BUILD_WINDOWS`. The shared decode pipeline is:

```
rocDecParseVideoData()
  └─ HevcParser / AvcParser  (src/parser/)
       └─ SendPicForDecode()
            └─ rocDecDecodeFrame()
                 └─ RocDecoder::DecodeFrame()
                      └─ PalVideoDecoder::DecodeFrame()
                               └─ DecodeHEVC() / DecodeH264()
```

**Key files:**
- `src/rocdecode/pal/pal_videodecoder.h` — `PalObject<T>` RAII wrapper, `PalDpbSlot`, `PalDpb`, `PalVideoDecoder` class declaration
- `src/rocdecode/pal/pal_videodecoder.cpp` — all PAL backend logic

### PAL Object Lifecycle

PAL uses placement-new — callers own the backing memory. `PalObject<T>` handles this with `Destroy()` + `free()` in its destructor:

```cpp
size_t sz = device->GetFenceSize(&res);
void* mem = malloc(sz);
Pal::IFence* f = nullptr;
device->CreateFence(fci, mem, &f);
// PalObject<Pal::IFence> wraps f + mem and cleans up automatically
```

**Resources owned by `PalVideoDecoder`:**
- `video_queue_` — PAL `IQueue` targeting the VCN decode engine
- `cmd_allocator_` / `cmd_buffer_` — shared across frames (one frame recorded at a time)
- `video_decoder_` — PAL `IVideoDecoder` session object
- `decoder_heap_` — VCN-internal scratch `IGpuMemory`
- `bitstream_` — CPU-writable / GPU-readable upload buffer for compressed data
- `dpb_` — `PalDpb` managing `max_dpb_slots_` `PalDpbSlot` entries

**Per `PalDpbSlot`:**
- `image` / `image_memory` — decoded NV12/P010 surface (`IImage` + `IGpuMemory`)
- `fence` — per-slot `IFence`, signaled when a decode targeting this slot completes

### Submission Model (mirrors VA-API)

`video_queue_->Submit()` is **fire-and-forget** — the GPU queue serializes work internally, exactly like `vaEndPicture()` on Linux. The next frame is submitted without waiting for the previous one.

Per-frame flow in `DecodeHEVC()`:
1. Wait on `dpb_[last_submitted_slot_idx_].fence` **before** `cmd_buffer_->Reset()` — the shared `cmd_allocator_` cannot be recycled while the GPU is still reading its previous commands
2. Record: `Reset` → `Begin` → `CmdBindVideoDecoder` → `CmdBeginVideoDecode` → `CmdDecodeVideoFrame` → `End`
3. `ResetFences(output_slot->fence)` — reset this slot's fence (safe: the wait in step 1 ensures it is no longer in-flight)
4. `Submit(output_slot->fence)` — attach this slot's fence; it signals when this decode finishes
5. Record `last_submitted_slot_idx_ = curr_pic_idx`

**Why per-slot fences, not a single global fence:** `GetDecodeStatus(pic_idx)` must report readiness for a specific picture independently — directly paralleling `vaQuerySurfaceStatus(surface)` on Linux. A single global fence reset before every submit would be reset while still in-flight from the previous frame, corrupting its state and causing `Submit` to fail.

### HIP Interop (`RocDecoder::GetVideoFrame`)

Decoded frames are exposed to applications as HIP device pointers. The interop is performed once per slot and cached in `hip_interop_[pic_idx]`:

| Platform | Export | HIP handle type |
|---|---|---|
| Linux | `vaExportSurfaceHandle` → DRM prime fd | `hipExternalMemoryHandleTypeOpaqueFd` |
| Windows | PAL `ExportExternalHandle` → KMT opaque handle | `hipExternalMemoryHandleTypeOpaqueWin32Kmt` |

On Windows, `GetVideoFrame` calls `GetDecodeStatus` to confirm the frame is ready before exporting. PAL DPB surface memory must be allocated with `flags.interprocess=1` and `flags.shareable=1` to enable KMT handle export.

### Adding a New Codec (Windows)

1. Add `DecodeXxx(RocdecPicParams*)` to `PalVideoDecoder`
2. Map `RocdecXxxPicParams` fields to the PAL/UVD codec struct from `drv_uvd_if.h`
3. Set `decode_info.decodeType` to the appropriate `Pal::VideoDecodeType` enum value
4. Wire into `PalVideoDecoder::DecodeFrame()` dispatch and `Initialize()` codec mapping
5. Add the matching implementation to `src/rocdecode/vaapi/` for Linux parity
