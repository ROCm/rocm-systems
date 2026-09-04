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

// Windows-only translation unit. On Linux this compiles to nothing.
#ifdef _WIN32

#include <cmath>
#include <sstream>
#include <iomanip>
#include <va/va.h>  // for VA_FOURCC_* constants used in the fourcc->DXGI mapping
#include "d3d12_interop.h"

namespace {
// Format an HRESULT/flag value as a lowercase hex string (no "0x" prefix) for log messages.
std::string ToHex(uint32_t value) {
    std::ostringstream oss;
    oss << std::hex << value;
    return oss.str();
}
}  // namespace

D3D12Interop::~D3D12Interop() {
    // Release D3D12 resources (VA surfaces that referenced them are destroyed by the owner first).
    for (auto* res : d3d12_staging_buffers_) {
        if (res) res->Release();
    }
    d3d12_staging_buffers_.clear();
    for (auto* res : d3d12_shared_resources_) {
        if (res) res->Release();
    }
    d3d12_shared_resources_.clear();
    if (d3d12_cmd_list_) { d3d12_cmd_list_->Release(); d3d12_cmd_list_ = nullptr; }
    if (d3d12_cmd_allocator_) { d3d12_cmd_allocator_->Release(); d3d12_cmd_allocator_ = nullptr; }
    if (d3d12_copy_queue_) { d3d12_copy_queue_->Release(); d3d12_copy_queue_ = nullptr; }
    if (d3d12_fence_) { d3d12_fence_->Release(); d3d12_fence_ = nullptr; }
    if (d3d12_fence_event_) { CloseHandle(d3d12_fence_event_); d3d12_fence_event_ = nullptr; }
    if (d3d12_device_) {
        d3d12_device_->Release();
        d3d12_device_ = nullptr;
    }
}

SurfaceLayout D3D12Interop::GetSurfaceLayout() const {
    SurfaceLayout layout = {};
    rocDecVideoSurfaceFormat fmt = output_format_;
    uint32_t width = width_;
    uint32_t height = height_;

    // Pitch and vstride: must match GetSurfaceStrideInternal in roc_video_dec.cpp.
    switch (fmt) {
        case rocDecVideoSurfaceFormat_P016:
        case rocDecVideoSurfaceFormat_YUV444_16Bit:
        case rocDecVideoSurfaceFormat_YUV420_16Bit:
        case rocDecVideoSurfaceFormat_YUV422_16Bit:
            layout.pitch = ((width + 127) & ~127u) * 2;
            break;
        case rocDecVideoSurfaceFormat_NV12:
        case rocDecVideoSurfaceFormat_YUV444:
        case rocDecVideoSurfaceFormat_YUV420:
        case rocDecVideoSurfaceFormat_YUV422:
        default:
            layout.pitch = (width + 255) & ~255u;
            break;
    }
    layout.vstride = (height + 15) & ~15u;

    // Chroma height factor and plane count: must match GetChromaHeightFactor/GetChromaPlaneCount.
    float chroma_height_factor = 0.5f;
    int num_chroma_planes = 1;
    switch (fmt) {
        case rocDecVideoSurfaceFormat_NV12:
        case rocDecVideoSurfaceFormat_P016:
            chroma_height_factor = 0.5f;
            num_chroma_planes = 1; // interleaved UV
            break;
        case rocDecVideoSurfaceFormat_YUV420:
        case rocDecVideoSurfaceFormat_YUV420_16Bit:
            chroma_height_factor = 0.5f;
            num_chroma_planes = 2; // separate U, V
            break;
        case rocDecVideoSurfaceFormat_YUV422:
        case rocDecVideoSurfaceFormat_YUV422_16Bit:
            chroma_height_factor = 1.0f;
            num_chroma_planes = 2;
            break;
        case rocDecVideoSurfaceFormat_YUV444:
        case rocDecVideoSurfaceFormat_YUV444_16Bit:
            chroma_height_factor = 1.0f;
            num_chroma_planes = 2;
            break;
        default:
            chroma_height_factor = 0.5f;
            num_chroma_planes = 1;
            break;
    }

    uint32_t chroma_vstride = static_cast<uint32_t>(std::ceil(layout.vstride * chroma_height_factor));

    // Build plane layout.
    layout.num_planes = 1 + num_chroma_planes;
    layout.plane_pitch[0] = layout.pitch;
    layout.plane_offset[0] = 0;
    layout.plane_height[0] = layout.vstride;

    uint32_t offset = layout.pitch * layout.vstride;
    for (int i = 0; i < num_chroma_planes; i++) {
        layout.plane_pitch[1 + i] = layout.pitch;
        layout.plane_offset[1 + i] = offset;
        layout.plane_height[1 + i] = chroma_vstride;
        offset += layout.pitch * chroma_vstride;
    }

    layout.total_size = offset;
    return layout;
}

void D3D12Interop::GetD3D12ResourceLayout(int pic_idx, uint32_t pitches[3], uint32_t offsets[3], uint32_t &num_planes) {
    SurfaceLayout layout = GetSurfaceLayout();
    num_planes = layout.num_planes;
    for (uint32_t i = 0; i < num_planes && i < 3; i++) {
        pitches[i] = layout.plane_pitch[i];
        offsets[i] = layout.plane_offset[i];
    }
    InfoLog(g_rocdec_logger, "Surface layout for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
            ": format=" + ROCDEC_TOSTR(output_format_) +
            " planes=" + ROCDEC_TOSTR(num_planes) +
            " pitch=" + ROCDEC_TOSTR(layout.pitch) +
            " total=" + ROCDEC_TOSTR(layout.total_size));
}

rocDecStatus D3D12Interop::CreateSharedResources(rocDecVideoSurfaceFormat format, uint32_t width,
                                                 uint32_t height, uint32_t num_surfaces,
                                                 int va_fourcc, const LUID &adapter_luid) {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    output_format_ = format;
    width_ = width;
    height_ = height;

    // On Windows, create D3D12 resources with D3D12_HEAP_FLAG_SHARED so that
    // CreateSharedHandle will succeed for HIP interop later.
    // Map the VA fourcc to a DXGI format for D3D12 resource creation.
    DXGI_FORMAT dxgi_format;
    switch (va_fourcc) {
        case VA_FOURCC_NV12: dxgi_format = DXGI_FORMAT_NV12; break;
        case VA_FOURCC_P010: dxgi_format = DXGI_FORMAT_P010; break;
        case VA_FOURCC_P012: dxgi_format = DXGI_FORMAT_P016; break; // D3D12 uses P016 for 12-bit
        default:
            CriticalLog(g_rocdec_logger, "Unsupported fourcc for D3D12 shared surface: 0x" +
                        ToHex(static_cast<uint32_t>(va_fourcc)));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_SUPPORTED;
    }

    if (d3d12_device_ == nullptr) {
        IDXGIFactory2* factory = nullptr;
        CHECK_D3D12(CreateDXGIFactory1(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&factory)));
        IDXGIAdapter1* adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            HRESULT desc_hr = adapter->GetDesc1(&desc);
            if (SUCCEEDED(desc_hr) &&
                desc.AdapterLuid.HighPart == adapter_luid.HighPart &&
                desc.AdapterLuid.LowPart == adapter_luid.LowPart) {
                break;
            }
            if (FAILED(desc_hr)) {
                InfoLog(g_rocdec_logger, "IDXGIAdapter1::GetDesc1 failed for adapter " + ROCDEC_TOSTR(i) +
                        " (HRESULT=0x" + ToHex(desc_hr) + "), skipping");
            }
            adapter->Release();
            adapter = nullptr;
        }
        factory->Release();
        if (adapter == nullptr) {
            CriticalLog(g_rocdec_logger, "Failed to find DXGI adapter matching LUID");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_DEVICE_INVALID;
        }
        HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                               __uuidof(ID3D12Device), reinterpret_cast<void**>(&d3d12_device_));
        adapter->Release();
        if (FAILED(hr) || d3d12_device_ == nullptr) {
            CriticalLog(g_rocdec_logger, "D3D12CreateDevice failed");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    // Release any previously created shared resources (reconfigure path), then reset every
    // slot to nullptr. Use assign (not resize) so pre-existing slots are cleared too --
    // otherwise the just-released pointers would dangle and the slot-0 probe below, or the
    // destructor, would act on freed resources.
    for (auto* res : d3d12_shared_resources_) {
        if (res) res->Release();
    }
    d3d12_shared_resources_.assign(num_surfaces, nullptr);

    D3D12_RESOURCE_DESC res_desc = {};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    res_desc.Width = width_;
    res_desc.Height = height_;
    res_desc.DepthOrArraySize = 1;
    res_desc.MipLevels = 1;
    res_desc.Format = dxgi_format;
    res_desc.SampleDesc.Count = 1;
    // Force linear (row-major) layout so HIP can read the texture as a flat buffer.
    // D3D12_TEXTURE_LAYOUT_ROW_MAJOR requires ALLOW_CROSS_ADAPTER on UMA adapters (APUs).
    // If this fails (e.g. on discrete GPUs), fall back to LAYOUT_UNKNOWN.
    res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;

    D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;

    // Probe ROW_MAJOR (linear) layout on slot 0; fall back to driver-default (tiled) layout.
    // On success, slot 0 is already created, so the main loop below skips it. On failure the
    // out-param is left null, res_desc is switched to the tiled layout, and the loop creates
    // all slots. Either way GetVideoFrame routes through the staging copy, so the decode
    // surface layout only affects whether that copy is redundant (see staging section below).
    {
        HRESULT hr = d3d12_device_->CreateCommittedResource(
            &heap_props, heap_flags,
            &res_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d3d12_shared_resources_[0]));
        if (FAILED(hr)) {
            InfoLog(g_rocdec_logger, "ROW_MAJOR layout not supported (HRESULT=0x" +
                    ToHex(hr) +
                    "), falling back to LAYOUT_UNKNOWN (tiled)");
            res_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            heap_flags = D3D12_HEAP_FLAG_SHARED;
        } else {
            InfoLog(g_rocdec_logger, "Using D3D12_TEXTURE_LAYOUT_ROW_MAJOR (linear) for decode surfaces");
        }
    }

    // Skip slot 0 if the probe above already created it.
    for (uint32_t i = (d3d12_shared_resources_[0] != nullptr ? 1 : 0); i < num_surfaces; i++) {
        HRESULT hr = d3d12_device_->CreateCommittedResource(
            &heap_props, heap_flags,
            &res_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d3d12_shared_resources_[i]));
        if (FAILED(hr) || d3d12_shared_resources_[i] == nullptr) {
            CriticalLog(g_rocdec_logger, "Failed to create shared D3D12 resource " + ROCDEC_TOSTR(i) +
                        ", HRESULT=0x" + ToHex(hr));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus D3D12Interop::CreateStagingInfrastructure(uint32_t num_surfaces) {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");

    // Log the actual D3D12 resource properties to confirm tiling mode.
    if (!d3d12_shared_resources_.empty() && d3d12_shared_resources_[0] != nullptr) {
        D3D12_RESOURCE_DESC desc = d3d12_shared_resources_[0]->GetDesc();
        const char* layout_str = "UNKNOWN";
        switch (desc.Layout) {
            case D3D12_TEXTURE_LAYOUT_UNKNOWN:                layout_str = "UNKNOWN (driver-managed tiled)"; break;
            case D3D12_TEXTURE_LAYOUT_ROW_MAJOR:              layout_str = "ROW_MAJOR (linear)"; break;
            case D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE: layout_str = "64KB_UNDEFINED_SWIZZLE"; break;
            case D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE:  layout_str = "64KB_STANDARD_SWIZZLE"; break;
        }
        D3D12_RESOURCE_ALLOCATION_INFO alloc_info = d3d12_device_->GetResourceAllocationInfo(0, 1, &desc);
        InfoLog(g_rocdec_logger, "D3D12 decode surface[0]: Dimension=" + ROCDEC_TOSTR(desc.Dimension) +
                " Format=" + ROCDEC_TOSTR(desc.Format) +
                " " + ROCDEC_TOSTR(desc.Width) + "x" + ROCDEC_TOSTR(desc.Height) +
                " Layout=" + ROCDEC_STR(layout_str) +
                " Flags=0x" + ToHex(static_cast<uint32_t>(desc.Flags)) +
                " AllocSize=" + ROCDEC_TOSTR(alloc_info.SizeInBytes) +
                " Alignment=" + ROCDEC_TOSTR(alloc_info.Alignment));
    }

    D3D12_HEAP_PROPERTIES heap_props = {};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;

    // Create linear staging buffers for the tiled->linear copy, plus D3D12 copy infrastructure.
    // These are created unconditionally: GetVideoFrame always routes through CopyToStagingBuffer,
    // so the staging path must exist even when the decode surfaces happened to be created with
    // ROW_MAJOR layout. In that (currently AMD-unreachable) case the copy is linear->linear, i.e.
    // redundant but correct. Revisit with a direct-import fast path if ROW_MAJOR becomes usable.
    {
        // Size the staging buffer from GetSurfaceLayout (matches sample layer expectations).
        SurfaceLayout layout = GetSurfaceLayout();
        UINT64 staging_size = layout.total_size;

        // assign (not resize) so pre-existing slots are cleared too -- see note above.
        for (auto* buf : d3d12_staging_buffers_) { if (buf) buf->Release(); }
        d3d12_staging_buffers_.assign(num_surfaces, nullptr);

        D3D12_RESOURCE_DESC buf_desc = {};
        buf_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buf_desc.Width = staging_size;
        buf_desc.Height = 1;
        buf_desc.DepthOrArraySize = 1;
        buf_desc.MipLevels = 1;
        buf_desc.Format = DXGI_FORMAT_UNKNOWN;
        buf_desc.SampleDesc.Count = 1;
        buf_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        for (uint32_t i = 0; i < num_surfaces; i++) {
            HRESULT hr = d3d12_device_->CreateCommittedResource(
                &heap_props, D3D12_HEAP_FLAG_SHARED,
                &buf_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d3d12_staging_buffers_[i]));
            if (FAILED(hr)) {
                CriticalLog(g_rocdec_logger, "Failed to create staging buffer " + ROCDEC_TOSTR(i) +
                            ", HRESULT=0x" + ToHex(hr));
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_RUNTIME_ERROR;
            }
        }

        // Log staging buffer properties for confirmation.
        {
            D3D12_RESOURCE_DESC sdesc = d3d12_staging_buffers_[0]->GetDesc();
            const char* slayout = (sdesc.Layout == D3D12_TEXTURE_LAYOUT_ROW_MAJOR) ? "ROW_MAJOR (linear)" : "OTHER";
            InfoLog(g_rocdec_logger, "D3D12 staging buffer[0]: Dimension=" + ROCDEC_TOSTR(sdesc.Dimension) +
                    " Size=" + ROCDEC_TOSTR(sdesc.Width) +
                    " Layout=" + ROCDEC_STR(slayout));
        }

        // Create copy command queue, allocator, list, and fence.
        if (d3d12_copy_queue_ == nullptr) {
            D3D12_COMMAND_QUEUE_DESC qd = {};
            qd.Type = D3D12_COMMAND_LIST_TYPE_COPY;
            CHECK_D3D12(d3d12_device_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&d3d12_copy_queue_)));
            CHECK_D3D12(d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&d3d12_cmd_allocator_)));
            CHECK_D3D12(d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, d3d12_cmd_allocator_, nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&d3d12_cmd_list_)));
            CHECK_D3D12(d3d12_cmd_list_->Close());
            CHECK_D3D12(d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&d3d12_fence_)));
            d3d12_fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (d3d12_fence_event_ == nullptr) {
                CriticalLog(g_rocdec_logger, "CreateEvent for D3D12 fence failed, GetLastError=" + ROCDEC_TOSTR(GetLastError()));
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_RUNTIME_ERROR;
            }
            d3d12_fence_value_ = 0;
        }
        InfoLog(g_rocdec_logger, "Created D3D12 staging buffers (" + ROCDEC_TOSTR(staging_size) + " bytes each) for tiled->linear copy");
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus D3D12Interop::CopyToStagingBuffer(int pic_idx) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx < 0 || static_cast<size_t>(pic_idx) >= d3d12_shared_resources_.size() || d3d12_shared_resources_[pic_idx] == nullptr ||
        static_cast<size_t>(pic_idx) >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr ||
        d3d12_device_ == nullptr || d3d12_copy_queue_ == nullptr || d3d12_cmd_allocator_ == nullptr || d3d12_cmd_list_ == nullptr ||
        d3d12_fence_ == nullptr || d3d12_fence_event_ == nullptr) {
        CriticalLog(g_rocdec_logger, "D3D12 staging infrastructure not available for pic_idx=" + ROCDEC_TOSTR(pic_idx));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }

    // Get D3D12's footprints for the source texture (provides Width/Height/Format per subresource).
    D3D12_RESOURCE_DESC tex_desc = d3d12_shared_resources_[pic_idx]->GetDesc();
    SurfaceLayout layout = GetSurfaceLayout();

    // Build destination footprints using our surface layout (consistent with GetD3D12ResourceLayout).
    // We override Offset and RowPitch to match the expected linear layout, but keep Width/Height/Format
    // from D3D12's GetCopyableFootprints so the copy source is read correctly.
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT src_footprints[3] = {};
    // NV12/P010: 2 subresources. Planar YUV: may need more, but D3D12 NV12 is always 2.
    UINT num_subresources = (tex_desc.Format == DXGI_FORMAT_NV12 || tex_desc.Format == DXGI_FORMAT_P010 ||
                             tex_desc.Format == DXGI_FORMAT_P016) ? 2 : 1;
    // Only the footprints are used below; pass nullptr for the optional row-count/size/total out-params.
    d3d12_device_->GetCopyableFootprints(&tex_desc, 0, num_subresources, 0,
                                         src_footprints, nullptr, nullptr, nullptr);

    // Record copy commands: texture (tiled) -> buffer (linear) for each subresource.
    // This is a COPY-type queue/command list: D3D12 does not track resource states on
    // copy queues (all resources are treated as COMMON), so no ResourceBarrier transitions
    // to COPY_SOURCE/COPY_DEST are needed -- and issuing them here would be invalid.
    CHECK_D3D12(d3d12_cmd_allocator_->Reset());
    CHECK_D3D12(d3d12_cmd_list_->Reset(d3d12_cmd_allocator_, nullptr));

    for (UINT sub = 0; sub < num_subresources; sub++) {
        D3D12_TEXTURE_COPY_LOCATION src = {};
        src.pResource = d3d12_shared_resources_[pic_idx];
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = sub;

        D3D12_TEXTURE_COPY_LOCATION dst = {};
        dst.pResource = d3d12_staging_buffers_[pic_idx];
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        // Use D3D12's footprint for format/width/height, but override offset and pitch
        // to match our surface layout so the staging buffer is consistent with what
        // GetD3D12ResourceLayout reports to the caller.
        dst.PlacedFootprint = src_footprints[sub];
        dst.PlacedFootprint.Offset = layout.plane_offset[sub];
        dst.PlacedFootprint.Footprint.RowPitch = layout.plane_pitch[sub];

        d3d12_cmd_list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }

    CHECK_D3D12(d3d12_cmd_list_->Close());
    ID3D12CommandList* lists[] = { d3d12_cmd_list_ };
    d3d12_copy_queue_->ExecuteCommandLists(1, lists);

    // Wait for copy to complete.
    d3d12_fence_value_++;
    CHECK_D3D12(d3d12_copy_queue_->Signal(d3d12_fence_, d3d12_fence_value_));
    uint64_t completed = d3d12_fence_->GetCompletedValue();
    // On device removal (TDR/hang), GetCompletedValue() returns UINT64_MAX, which would
    // otherwise satisfy the completion check and let the copy be treated as done -- returning
    // silently corrupt frame data. Detect that explicitly and fail instead. (Full TDR recovery
    // and a finite-timeout wait are deferred; see follow-up.)
    if (completed == UINT64_MAX) {
        CriticalLog(g_rocdec_logger, "D3D12 device removed during staging copy, reason HRESULT=0x" +
                    ToHex(d3d12_device_->GetDeviceRemovedReason()));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    if (completed < d3d12_fence_value_) {
        // If SetEventOnCompletion fails, the event is never signaled and the wait below
        // would block forever -- fail here instead of hanging.
        CHECK_D3D12(d3d12_fence_->SetEventOnCompletion(d3d12_fence_value_, d3d12_fence_event_));
        DWORD wait = WaitForSingleObject(d3d12_fence_event_, INFINITE);
        if (wait != WAIT_OBJECT_0) {
            CriticalLog(g_rocdec_logger, "WaitForSingleObject for D3D12 fence failed, result=" + ROCDEC_TOSTR(wait) +
                        " GetLastError=" + ROCDEC_TOSTR(GetLastError()));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus D3D12Interop::ExportStagingBufferHandle(int pic_idx, HANDLE &nt_handle) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx < 0 || static_cast<size_t>(pic_idx) >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_D3D12(d3d12_device_->CreateSharedHandle(d3d12_staging_buffers_[pic_idx], nullptr, GENERIC_ALL, nullptr, &nt_handle));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

uint64_t D3D12Interop::GetStagingBufferSize(int pic_idx) {
    if (pic_idx < 0 || static_cast<size_t>(pic_idx) >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr) {
        return 0;
    }
    D3D12_RESOURCE_DESC desc = d3d12_staging_buffers_[pic_idx]->GetDesc();
    return desc.Width; // For buffers, Width is the size in bytes.
}

rocDecStatus D3D12Interop::ExportStagingInterop(int pic_idx, StagingInteropInfo &out) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    out = {};
    rocDecStatus status = ExportStagingBufferHandle(pic_idx, out.shared_handle);
    if (status != ROCDEC_SUCCESS) {
        FunctionExitLog(g_rocdec_logger);
        return status;
    }
    out.size = GetStagingBufferSize(pic_idx);
    GetD3D12ResourceLayout(pic_idx, out.pitches, out.offsets, out.num_planes);
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

#endif // _WIN32
