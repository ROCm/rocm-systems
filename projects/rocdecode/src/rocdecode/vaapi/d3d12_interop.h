/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

// This entire file is Windows-only. On Linux it compiles to nothing, and it is
// only included from the #ifdef _WIN32 block of vaapi_videodecoder.h.
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_2.h>
#include <vector>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include "../../commons.h"
#include "../../../api/rocdecode/rocdecode.h"

// Checks an HRESULT-returning D3D12/Win32 call; on failure logs the call text and the
// HRESULT (hex) with file:line, then returns ROCDEC_RUNTIME_ERROR. Mirrors CHECK_VAAPI.
#define CHECK_D3D12(call) {\
    HRESULT hres = call;\
    if (FAILED(hres)) {\
        std::ostringstream hres_oss_; hres_oss_ << std::hex << static_cast<uint32_t>(hres);\
        CriticalLog(g_rocdec_logger, ROCDEC_STR("D3D12 failure: ") + #call + " failed with 'HRESULT: 0x" + hres_oss_.str() + "' at " + __FILE__ + ":" + ROCDEC_TOSTR(__LINE__));\
        return ROCDEC_RUNTIME_ERROR;\
    }\
}

// Owns all Direct3D12 interop state for the Windows (vaon12) decode path:
// the D3D12 device, the shared decode textures handed to VA-API as external
// surfaces, and the linear staging buffers + copy infrastructure used to turn
// each tiled decode texture into a linear buffer that HIP can import.
//
// The owning VaapiVideoDecoder holds one of these (Windows-only) and drives it:
//   CreateSharedResources()        -> before vaCreateSurfaces()
//   GetSharedResources()           -> fed into vaCreateSurfaces()
//   CreateStagingInfrastructure()  -> after vaCreateSurfaces()
//   CopyToStagingBuffer()          -> per frame
//   ExportStagingInterop()         -> once per surface slot (feeds the HIP import)
class D3D12Interop {
public:
    // Surface layout info (computed from decoder config, matches GetSurfaceStrideInternal).
    struct SurfaceLayout {
        uint32_t pitch;             // Row pitch in bytes (luma and chroma share this for NV12/P016)
        uint32_t vstride;           // Aligned height
        uint32_t num_planes;        // Total planes (luma + chroma): 1 for mono, 2 for NV12/P016, 3 for planar YUV
        uint32_t plane_offset[3];   // Byte offset of each plane
        uint32_t plane_pitch[3];    // Byte pitch of each plane
        uint32_t plane_height[3];   // Row count of each plane
        uint64_t total_size;        // Total buffer size in bytes
    };

    // Everything the caller needs to set up the HIP import for one surface slot.
    // Produced once per slot by ExportStagingInterop().
    struct StagingInteropInfo {
        HANDLE   shared_handle;     // NT handle from CreateSharedHandle (caller owns; CloseHandle when done)
        uint64_t size;              // Staging buffer size in bytes
        uint32_t pitches[3];        // Byte pitch of each plane
        uint32_t offsets[3];        // Byte offset of each plane
        uint32_t num_planes;        // Number of valid planes
    };

    D3D12Interop() = default;
    ~D3D12Interop();

    // AMD D3D12 video decode requires the decode (DPB) height to be a multiple of 32
    // (HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED); vaon12 rounds DecodeHeight up to this and the
    // decode engine writes that many rows. The shared decode texture AND the height handed to
    // vaCreateSurfaces must both use this aligned value: the vaon12 VA frontend takes the
    // VA-requested height as the expected resource height, so an over-tall texture whose height
    // doesn't match the requested surface height is rejected ("resource allocation failed").
    // Keep this the single source of truth so the two sites can never diverge. The coded height
    // (height_) is still used for the linear staging/output layout, which must match
    // GetSurfaceStrideInternal (align 16) rather than this decode alignment.
    static uint32_t AlignedDecodeSurfaceHeight(uint32_t height) { return (height + 31u) & ~31u; }

    // Companion to AlignedDecodeSurfaceHeight for WIDTH. The decode engine reconstructs whole
    // CTBs/superblocks and writes a CTB-aligned width per row into the decode target (which, in the
    // array-of-textures path, is this shared surface itself). A coded width that is not CTB/SB-aligned
    // (e.g. HEVC PICSIZE 528 -> 9 CTB cols x 64 = 576) under-sizes the surface -> horizontal overrun ->
    // GPU page fault. The API exposes no width-alignment flag, so pad to 128 (>= 64-CTB and 128-SB
    // overhang). Must match the width vaon12 uses for the decode heap / recon resolution and the width
    // handed to vaCreateSurfaces. The coded width_ is still used for the linear staging/output layout.
    static uint32_t AlignedDecodeSurfaceWidth(uint32_t width) { return (width + 127u) & ~127u; }

    D3D12Interop(const D3D12Interop&) = delete;
    D3D12Interop& operator=(const D3D12Interop&) = delete;

    // Phase 1 (before vaCreateSurfaces): set up the D3D12 device and the shared decode textures.
    // Caches format/width/height for later layout math. Creates a LUID-matched D3D12 device.
    rocDecStatus CreateSharedResources(rocDecVideoSurfaceFormat format, uint32_t width,
                                       uint32_t height, uint32_t num_surfaces,
                                       int va_fourcc, const LUID &adapter_luid);

    // Hand the shared-resource array to vaCreateSurfaces (VASurfaceAttribExternalBufferDescriptor).
    ID3D12Resource* const* GetSharedResources() const { return d3d12_shared_resources_.data(); }

    // Phase 2 (after vaCreateSurfaces): create linear staging buffers and the copy
    // queue/allocator/list/fence used by CopyToStagingBuffer.
    rocDecStatus CreateStagingInfrastructure(uint32_t num_surfaces);

    // Per-frame: copy the tiled decode texture into its linear staging buffer.
    rocDecStatus CopyToStagingBuffer(int pic_idx);

    // Once per slot: export the shared NT handle and report the staging buffer size and
    // plane layout, all of which the caller needs to build the HIP external-memory import.
    rocDecStatus ExportStagingInterop(int pic_idx, StagingInteropInfo &out);

private:
    // Internal helpers backing ExportStagingInterop / the copy path.
    SurfaceLayout GetSurfaceLayout() const;
    void GetD3D12ResourceLayout(int pic_idx, uint32_t pitches[3], uint32_t offsets[3], uint32_t &num_planes);
    rocDecStatus ExportStagingBufferHandle(int pic_idx, HANDLE &nt_handle);
    uint64_t GetStagingBufferSize(int pic_idx);

    // Cached decoder config for layout math (matches GetSurfaceStrideInternal).
    rocDecVideoSurfaceFormat output_format_ = rocDecVideoSurfaceFormat_NV12;
    uint32_t width_ = 0;
    uint32_t height_ = 0;

    ID3D12Device* d3d12_device_ = nullptr;                  // D3D12 device for creating shared resources
    std::vector<ID3D12Resource*> d3d12_shared_resources_;  // Shared D3D12 textures used as VA surfaces
    // D3D12 copy infrastructure: tiled texture -> linear staging buffer
    ID3D12CommandQueue* d3d12_copy_queue_ = nullptr;
    ID3D12CommandAllocator* d3d12_cmd_allocator_ = nullptr;
    ID3D12GraphicsCommandList* d3d12_cmd_list_ = nullptr;
    ID3D12Fence* d3d12_fence_ = nullptr;
    HANDLE d3d12_fence_event_ = nullptr;
    uint64_t d3d12_fence_value_ = 0;
    std::vector<ID3D12Resource*> d3d12_staging_buffers_;   // Linear staging buffers (shared, for HIP import)
};

#endif // _WIN32
