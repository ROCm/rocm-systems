/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include <cstddef>
#include <cstdint>

// Forward declarations to avoid pulling in HIP headers
struct dim3;
typedef struct ihipModule_t* hipModule_t;

namespace hrr {

// ============================================================================
// HIP Record & Replay (HRR) - In-tree recording hooks
//
// When HIP_RECORD=1 is set, these functions capture HIP API calls to a trace
// archive (.hrr directory) containing:
//   - events.bin:    Binary event stream with 32-byte headers
//   - blobs/:        Content-addressed buffer store (XXH3-128 hashed)
//   - code_objects/: Captured GPU code object ELFs (.hsaco)
//   - manifest.json: Device info, ROCm version, capture config
//
// Recording modes (HIP_RECORD_MODE):
//   timeline: API call sequence only, no buffer data
//   inputs:   Snapshot input buffers before each kernel (default)
//   full:     Snapshot inputs + outputs (sync after every kernel)
// ============================================================================

// --- Event type codes (match events.bin format) ---
enum EventType : uint16_t {
  EVENT_MALLOC        = 0x0001,
  EVENT_FREE          = 0x0002,
  EVENT_MEMCPY        = 0x0003,
  EVENT_MEMSET        = 0x0004,
  EVENT_MODULE_LOAD   = 0x0010,
  EVENT_MODULE_UNLOAD = 0x0011,
  EVENT_KERNEL_LAUNCH = 0x0020,
  EVENT_STREAM_CREATE = 0x0030,
  EVENT_STREAM_DESTROY= 0x0031,
  EVENT_STREAM_SYNC   = 0x0032,
  EVENT_EVENT_CREATE  = 0x0040,
  EVENT_EVENT_RECORD  = 0x0041,
  EVENT_EVENT_SYNC    = 0x0042,
  EVENT_DEVICE_SYNC   = 0x0050,
  EVENT_MARKER        = 0x00FF,
};

// --- Recording mode ---
enum class RecordMode {
  Timeline,  // API calls only, no buffer data
  Inputs,    // Snapshot input buffers before kernel launch (default)
  Full,      // Snapshot inputs + outputs
};

// --- Event header (32 bytes, little-endian, written to events.bin) ---
struct EventHeader {
  uint32_t magic;           // 0x52524845 ("HRRE")
  uint16_t version;         // 1
  uint16_t event_type;      // EventType enum
  uint64_t sequence_id;     // monotonic counter
  uint64_t timestamp_ns;    // CLOCK_MONOTONIC / QueryPerformanceCounter
  uint32_t stream_id;       // HIP stream handle (cast to u32)
  uint16_t device_id;
  uint16_t payload_length;  // bytes following this header
};
static_assert(sizeof(EventHeader) == 32, "EventHeader must be 32 bytes");

constexpr uint32_t HRR_MAGIC   = 0x52524845; // "HRRE"
constexpr uint16_t HRR_VERSION = 1;

// --- Lifecycle ---
// Called once from HIP runtime init when HIP_RECORD=1
void init();
// Called at process exit to flush and close the trace
void shutdown();

// --- Query ---
// Fast inline check -- returns true only when recording is active.
// All record_* functions are no-ops when this returns false.
bool enabled();

// Current recording mode
RecordMode mode();

// --- Recording hooks (called from HIP API implementations) ---

// Memory allocation: ptr is the returned device pointer, size in bytes
void record_malloc(const void* ptr, size_t size, unsigned int flags);
void record_free(const void* ptr);

// Memory transfer: captures source data for H2D, records direction
void record_memcpy(void* dst, const void* src, size_t size_bytes,
                   unsigned int kind, const void* stream);

// Memory set
void record_memset(void* dst, int value, size_t size_bytes,
                   const void* stream);

// Module / code object loading
void record_module_load(hipModule_t module, const void* image,
                        size_t image_size);
void record_module_unload(hipModule_t module);

// Kernel launch: captures kernel function, grid/block dims, args.
// hipFunction_t is passed opaquely to avoid pulling in HIP headers.
// Internally cast to DeviceFunc* to access KernelParameterDescriptor
// for pointer/scalar arg identification and buffer snapshots.
void record_kernel_launch(const char* kernel_name,
                          void* func_handle,  // hipFunction_t
                          uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                          uint32_t block_x, uint32_t block_y, uint32_t block_z,
                          uint32_t shared_mem,
                          const void* stream,
                          void** kernel_args);

// Synchronization
void record_device_sync();
void record_stream_sync(const void* stream);

}  // namespace hrr
