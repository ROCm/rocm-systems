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

#include "../commons.h"
#include "roc_decoder.h"

RocDecoder::RocDecoder(RocDecoderCreateInfo& decoder_create_info): va_video_decoder_{decoder_create_info}, decoder_create_info_{decoder_create_info} {
}

 RocDecoder::~RocDecoder() {
    // clean up the VA-API/HIP interop memories
    for(auto i = 0; i < hip_interop_.size(); i++) {
        if (hip_interop_[i].hip_mapped_device_mem != nullptr) {
            hipError_t hip_status = hipFree(hip_interop_[i].hip_mapped_device_mem);
            if (hip_status != hipSuccess) {
                CriticalLog(g_rocdec_logger, "hipFree failed for picture idx = " + ROCDEC_TOSTR(i));
            }
        }
        if (hip_interop_[i].hip_ext_mem != nullptr) {
            hipError_t hip_status = hipDestroyExternalMemory(hip_interop_[i].hip_ext_mem);
            if (hip_status != hipSuccess) {
                CriticalLog(g_rocdec_logger, "hipDestroyExternalMemory failed for picture idx = " + ROCDEC_TOSTR(i));
            }
        }
#ifdef _WIN32
        if (hip_interop_[i].nt_handle != nullptr) {
            CloseHandle(hip_interop_[i].nt_handle);
        }
#endif
    }
}

 rocDecStatus RocDecoder::InitializeDecoder() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    if (decoder_create_info_.num_decode_surfaces < 1) {
        CriticalLog(g_rocdec_logger, "Invalid number of decode surfaces.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    hip_interop_.resize(decoder_create_info_.num_decode_surfaces);
    for (auto i = 0; i < hip_interop_.size(); i++) {
        memset((void *)&hip_interop_[i], 0, sizeof(hip_interop_[i]));
    }
    rocdec_status = va_video_decoder_.InitializeDecoder();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to initialize the VAAPI Video decoder.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
 }

rocDecStatus RocDecoder::DecodeFrame(RocdecPicParams *pic_params) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(pic_params));
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = va_video_decoder_.SubmitDecode(pic_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Decode submission is not successful.");
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx) + ", " + RocDecFmtPtr(decode_status));
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = va_video_decoder_.GetDecodeStatus(pic_idx, decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to query the decode status.");
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(reconfig_params));
    if (reconfig_params == nullptr || reconfig_params->width == 0 || reconfig_params->height == 0 ||
        reconfig_params->num_decode_surfaces < 1 || reconfig_params->bit_depth_minus_8 > 2) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status;
    for (int pic_idx = 0; pic_idx < hip_interop_.size(); pic_idx++) {
        rocdec_status = FreeVideoFrame(pic_idx);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Releasing the video frame for picture idx = " + ROCDEC_TOSTR(pic_idx) + " failed during reconfiguration.");
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }
    }
    if (hip_interop_.size() != reconfig_params->num_decode_surfaces) {
        hip_interop_.resize(reconfig_params->num_decode_surfaces);
    }
    rocdec_status = va_video_decoder_.ReconfigureDecoder(reconfig_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Reconfiguration of the decoder failed.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::GetVideoFrame(int pic_idx, void *dev_mem_ptr[3], uint32_t horizontal_pitch[3], RocdecProcParams *vid_postproc_params) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx) + ", " + RocDecFmtPtr(dev_mem_ptr) + ", " +
                             RocDecFmtPtr(horizontal_pitch) + ", " + RocDecFmtPtr(vid_postproc_params));
    if (pic_idx >= hip_interop_.size() || dev_mem_ptr == nullptr || vid_postproc_params == nullptr) {
        CriticalLog(g_rocdec_logger, "Invalid parameters.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

    // wait on current surface to make sure that it is ready for the HIP interop
    rocdec_status = va_video_decoder_.SyncSurface(pic_idx);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to export surface for picture idx = " + ROCDEC_TOSTR(pic_idx));
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }

#ifdef _WIN32
    // Windows interop paths (selected once per surface slot on first frame):
    //   1. Direct: NTHANDLE from tiled D3D12 texture → HIP (no GPU copy, but texture must be linear)
    //   2. Staging: GPU CopyTextureRegion tiled→linear buffer → HIP (one GPU copy per frame)
    //   3. CPU fallback: vaGetImage → hipMemcpy (slowest, always works)
    // Default is staging (path 2) since D3D12 NV12 textures are tiled on AMD.
    // Set ROCDEC_DIRECT_INTEROP=1 to try the direct path (for testing/future linear textures).
    static std::vector<int> interop_mode; // 0=unset, 1=direct, 2=staging
    if (interop_mode.size() <= pic_idx) interop_mode.resize(pic_idx + 1, 0);

    static bool try_direct = (std::getenv("ROCDEC_DIRECT_INTEROP") != nullptr &&
                              std::string(std::getenv("ROCDEC_DIRECT_INTEROP")) == "1");

    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        HANDLE nt_handle = nullptr;
        uint64_t resource_size = 0;
        bool import_ok = false;

        // --- Path 1: Direct NTHANDLE import (only if ROCDEC_DIRECT_INTEROP=1) ---
        if (try_direct) {
        rocdec_status = va_video_decoder_.ExportSurfaceNTHandle(pic_idx, nt_handle);
        if (rocdec_status == ROCDEC_SUCCESS) {
            resource_size = va_video_decoder_.GetD3D12ResourceAllocationSize(pic_idx);
            InfoLog(g_rocdec_logger, "Direct NTHANDLE export succeeded for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
                    ", resource_size=" + ROCDEC_TOSTR(resource_size));

            hipExternalMemoryHandleDesc external_mem_handle_desc = {};
            external_mem_handle_desc.type = hipExternalMemoryHandleTypeD3D12Resource;
            external_mem_handle_desc.handle.win32.handle = nt_handle;
            external_mem_handle_desc.size = resource_size;

            hipError_t hip_status = hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc);
            if (hip_status == hipSuccess) {
                hipExternalMemoryBufferDesc buf_desc = {};
                buf_desc.size = resource_size;
                hip_status = hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem,
                                                              hip_interop_[pic_idx].hip_ext_mem, &buf_desc);
                if (hip_status == hipSuccess) {
                    uint32_t d3d_pitches[3] = {}, d3d_offsets[3] = {}, d3d_num_planes = 0;
                    va_video_decoder_.GetD3D12ResourceLayout(pic_idx, d3d_pitches, d3d_offsets, d3d_num_planes);

                    hip_interop_[pic_idx].width = decoder_create_info_.width;
                    hip_interop_[pic_idx].height = decoder_create_info_.height;
                    hip_interop_[pic_idx].num_layers = d3d_num_planes;
                    for (uint32_t i = 0; i < d3d_num_planes && i < 3; i++) {
                        hip_interop_[pic_idx].pitch[i] = d3d_pitches[i];
                        hip_interop_[pic_idx].offset[i] = d3d_offsets[i];
                    }
                    hip_interop_[pic_idx].nt_handle = nt_handle;
                    interop_mode[pic_idx] = 1; // direct
                    import_ok = true;
                    InfoLog(g_rocdec_logger, "Direct D3D12↔HIP interop active for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
                            " (no staging copy)");
                } else {
                    hipDestroyExternalMemory(hip_interop_[pic_idx].hip_ext_mem);
                    hip_interop_[pic_idx].hip_ext_mem = nullptr;
                    hip_interop_[pic_idx].hip_mapped_device_mem = nullptr;
                }
            } else {
                hip_interop_[pic_idx].hip_ext_mem = nullptr;
            }

            if (!import_ok) {
                CloseHandle(nt_handle);
                nt_handle = nullptr;
                DebugLog(g_rocdec_logger, "Direct NTHANDLE import failed, trying staging buffer path");
            }
        }
        } // if (try_direct)

        // --- Path 2: Staging buffer path (tiled → linear GPU copy, default) ---
        if (!import_ok && va_video_decoder_.HasStagingBuffers()) {
            rocdec_status = va_video_decoder_.CopyToStagingBuffer(pic_idx);
            if (rocdec_status != ROCDEC_SUCCESS) {
                ErrorLog(g_rocdec_logger, "CopyToStagingBuffer failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
                FunctionExitLog(g_rocdec_logger);
                return rocdec_status;
            }

            rocdec_status = va_video_decoder_.ExportStagingBufferHandle(pic_idx, nt_handle);
            if (rocdec_status != ROCDEC_SUCCESS) {
                ErrorLog(g_rocdec_logger, "ExportStagingBufferHandle failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
                FunctionExitLog(g_rocdec_logger);
                return rocdec_status;
            }

            resource_size = va_video_decoder_.GetStagingBufferSize(pic_idx);
            InfoLog(g_rocdec_logger, "Staging buffer size for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
                    ": " + ROCDEC_TOSTR(resource_size) + " bytes");

            hipExternalMemoryHandleDesc external_mem_handle_desc = {};
            external_mem_handle_desc.type = hipExternalMemoryHandleTypeD3D12Resource;
            external_mem_handle_desc.handle.win32.handle = nt_handle;
            external_mem_handle_desc.size = resource_size;

            CHECK_HIP(hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc));

            hipExternalMemoryBufferDesc buf_desc = {};
            buf_desc.size = resource_size;
            CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem,
                                                       hip_interop_[pic_idx].hip_ext_mem, &buf_desc));

            uint32_t d3d_pitches[3] = {}, d3d_offsets[3] = {}, d3d_num_planes = 0;
            va_video_decoder_.GetD3D12ResourceLayout(pic_idx, d3d_pitches, d3d_offsets, d3d_num_planes);

            hip_interop_[pic_idx].width = decoder_create_info_.width;
            hip_interop_[pic_idx].height = decoder_create_info_.height;
            hip_interop_[pic_idx].num_layers = d3d_num_planes;
            for (uint32_t i = 0; i < d3d_num_planes && i < 3; i++) {
                hip_interop_[pic_idx].pitch[i] = d3d_pitches[i];
                hip_interop_[pic_idx].offset[i] = d3d_offsets[i];
            }
            hip_interop_[pic_idx].nt_handle = nt_handle;
            interop_mode[pic_idx] = 2; // staging
            import_ok = true;
            InfoLog(g_rocdec_logger, "Staging D3D12↔HIP interop active for pic_idx=" + ROCDEC_TOSTR(pic_idx));
        }

        if (!import_ok) {
            WarningLog(g_rocdec_logger, "D3D12 interop failed. Falling back to CPU-staged transfer (EXPERIMENTAL).");
            hip_interop_[pic_idx].hip_ext_mem = nullptr;
        }
    } else if (interop_mode[pic_idx] == 2 && hip_interop_[pic_idx].hip_ext_mem != nullptr) {
        // Subsequent frame on staging path: re-copy tiled→linear.
        rocdec_status = va_video_decoder_.CopyToStagingBuffer(pic_idx);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "CopyToStagingBuffer failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }
    }
    // Direct path (interop_mode==1): no per-frame copy needed — HIP reads the texture directly.

    // CPU staging fallback — used when zero-copy interop is not available.
    // Must run every frame since surfaces are reused by the decoder.
    if (hip_interop_[pic_idx].hip_ext_mem == nullptr) {
        uint8_t* cpu_ptr = nullptr;
        uint32_t va_width = 0, va_height = 0, va_num_planes = 0;
        uint32_t va_pitches[3] = {}, va_offsets[3] = {};

        rocdec_status = va_video_decoder_.MapSurfaceToCPU(pic_idx, &cpu_ptr, va_width, va_height, va_pitches, va_offsets, va_num_planes);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Failed to map surface to CPU for picture idx = " + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        // Get the expected surface layout (matches sample layer).
        uint32_t layout_pitches[3] = {}, layout_offsets[3] = {}, layout_num_planes = 0;
        va_video_decoder_.GetD3D12ResourceLayout(pic_idx, layout_pitches, layout_offsets, layout_num_planes);

        // Compute total size from the layout.
        uint32_t total_size = 0;
        for (uint32_t i = 0; i < layout_num_planes; i++) {
            uint32_t plane_h = (i == 0) ? decoder_create_info_.height
                                        : static_cast<uint32_t>(std::ceil(decoder_create_info_.height *
                                          ((decoder_create_info_.chroma_format == rocDecVideoChromaFormat_420) ? 0.5f : 1.0f)));
            total_size = layout_offsets[i] + layout_pitches[i] * plane_h;
        }

        if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
            CHECK_HIP(hipMalloc(&hip_interop_[pic_idx].hip_mapped_device_mem, total_size));
            hip_interop_[pic_idx].width = decoder_create_info_.width;
            hip_interop_[pic_idx].height = decoder_create_info_.height;
            hip_interop_[pic_idx].num_layers = layout_num_planes;
            for (uint32_t i = 0; i < layout_num_planes && i < 3; i++) {
                hip_interop_[pic_idx].pitch[i] = layout_pitches[i];
                hip_interop_[pic_idx].offset[i] = layout_offsets[i];
            }
        }

        // Copy each plane from VA image to aligned device buffer with stride conversion.
        uint32_t bytes_per_pixel = (decoder_create_info_.bit_depth_minus_8 > 0) ? 2 : 1;
        uint32_t copy_planes = std::min(va_num_planes, layout_num_planes);
        for (uint32_t p = 0; p < copy_planes; p++) {
            uint32_t plane_h = (p == 0) ? decoder_create_info_.height
                                        : static_cast<uint32_t>(std::ceil(decoder_create_info_.height *
                                          ((decoder_create_info_.chroma_format == rocDecVideoChromaFormat_420) ? 0.5f : 1.0f)));
            uint8_t* dst = hip_interop_[pic_idx].hip_mapped_device_mem + layout_offsets[p];
            uint32_t row_bytes = decoder_create_info_.width * bytes_per_pixel;

            if (va_pitches[p] == layout_pitches[p]) {
                CHECK_HIP(hipMemcpy(dst, cpu_ptr + va_offsets[p],
                                    layout_pitches[p] * plane_h, hipMemcpyHostToDevice));
            } else {
                CHECK_HIP(hipMemcpy2D(dst, layout_pitches[p],
                                      cpu_ptr + va_offsets[p], va_pitches[p],
                                      row_bytes, plane_h, hipMemcpyHostToDevice));
            }
        }
        va_video_decoder_.UnmapSurface(pic_idx);
    }
#else
    // do the VA-API/HIP interop once per surface and save it for reusing
    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        hipExternalMemoryHandleDesc external_mem_handle_desc = {};
        hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
        // Linux path: export as DRM PRIME FD
        VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};

        rocdec_status = va_video_decoder_.ExportSurface(pic_idx, va_drm_prime_surface_desc);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Failed to export surface for picture idx = " + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        if (va_drm_prime_surface_desc.num_layers == 0 || va_drm_prime_surface_desc.num_layers > 3) {
            ErrorLog(g_rocdec_logger, "VA-API returned an unsupported value for num_layers. num_layers = " + ROCDEC_TOSTR(va_drm_prime_surface_desc.num_layers));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }

        external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
        external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

        CHECK_HIP(hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc));

        external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
        CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem, hip_interop_[pic_idx].hip_ext_mem, &external_mem_buffer_desc));

        hip_interop_[pic_idx].width = va_drm_prime_surface_desc.width;
        hip_interop_[pic_idx].height = va_drm_prime_surface_desc.height;

        for (int i = 0; i < va_drm_prime_surface_desc.num_layers; i++) {
            hip_interop_[pic_idx].offset[i] = va_drm_prime_surface_desc.layers[i].offset[0];
            hip_interop_[pic_idx].pitch[i] = va_drm_prime_surface_desc.layers[i].pitch[0];
        }

        hip_interop_[pic_idx].num_layers = va_drm_prime_surface_desc.num_layers;

        for (auto i = 0; i < va_drm_prime_surface_desc.num_objects; ++i) {
            close(va_drm_prime_surface_desc.objects[i].fd);
        }
    }
#endif

    for (uint32_t i = 0; i < hip_interop_[pic_idx].num_layers; i++) {
        dev_mem_ptr[i] = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[i];
        horizontal_pitch[i] = hip_interop_[pic_idx].pitch[i];
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::FreeVideoFrame(int pic_idx) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= hip_interop_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }

    if (hip_interop_[pic_idx].hip_mapped_device_mem != nullptr)
        CHECK_HIP(hipFree(hip_interop_[pic_idx].hip_mapped_device_mem));
    if (hip_interop_[pic_idx].hip_ext_mem != nullptr)
        CHECK_HIP(hipDestroyExternalMemory(hip_interop_[pic_idx].hip_ext_mem));
#ifdef _WIN32
    if (hip_interop_[pic_idx].nt_handle != nullptr)
        CloseHandle(hip_interop_[pic_idx].nt_handle);
#endif

    memset((void *)&hip_interop_[pic_idx], 0, sizeof(hip_interop_[pic_idx]));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
