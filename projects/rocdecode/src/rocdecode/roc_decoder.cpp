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
    // Windows interop: D3D12 tiled texture → linear staging buffer → HIP import.
    // The staging buffer import + layout setup is done once per surface slot.
    // The tiled→linear copy (CopyToStagingBuffer) runs every frame since surfaces are reused.
    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        // Copy decoded tiled texture to linear staging buffer, then export + import into HIP.
        rocdec_status = va_video_decoder_.CopyToStagingBuffer(pic_idx);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "CopyToStagingBuffer failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        HANDLE nt_handle = nullptr;
        rocdec_status = va_video_decoder_.ExportStagingBufferHandle(pic_idx, nt_handle);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "ExportStagingBufferHandle failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        uint64_t staging_size = va_video_decoder_.GetStagingBufferSize(pic_idx);
        InfoLog(g_rocdec_logger, "Staging buffer size for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
                ": " + ROCDEC_TOSTR(staging_size) + " bytes");

        // Import the linear staging buffer into HIP.
        struct { hipExternalMemoryHandleType type; const char* name; } handle_types[] = {
            { hipExternalMemoryHandleTypeD3D12Resource, "D3D12Resource" },
            { hipExternalMemoryHandleTypeD3D12Heap,     "D3D12Heap" },
            { hipExternalMemoryHandleTypeOpaqueWin32,   "OpaqueWin32" },
        };

        hipError_t hip_status = hipErrorInvalidValue;
        for (auto& ht : handle_types) {
            hipExternalMemoryHandleDesc external_mem_handle_desc = {};
            external_mem_handle_desc.type = ht.type;
            external_mem_handle_desc.handle.win32.handle = nt_handle;
            external_mem_handle_desc.size = staging_size;
            external_mem_handle_desc.flags = 0;

            hip_status = hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc);
            if (hip_status == hipSuccess) {
                InfoLog(g_rocdec_logger, "hipImportExternalMemory succeeded with type=" + ROCDEC_STR(ht.name));
                break;
            }
            DebugLog(g_rocdec_logger, "hipImportExternalMemory with type=" + ROCDEC_STR(ht.name) +
                     " failed: " + ROCDEC_STR(hipGetErrorName(hip_status)));
            hip_interop_[pic_idx].hip_ext_mem = nullptr;
        }

        if (hip_status == hipSuccess) {
            hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
            external_mem_buffer_desc.size = staging_size;
            CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem,
                                                       hip_interop_[pic_idx].hip_ext_mem, &external_mem_buffer_desc));

            // Use the linear staging buffer layout (from GetCopyableFootprints).
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

            InfoLog(g_rocdec_logger, "D3D12↔HIP interop active for pic_idx=" + ROCDEC_TOSTR(pic_idx));
        } else {
            CloseHandle(nt_handle);
            WarningLog(g_rocdec_logger, "All hipImportExternalMemory handle types failed. "
                       "Falling back to CPU-staged transfer (EXPERIMENTAL).");
            hip_interop_[pic_idx].hip_ext_mem = nullptr;
        }
    } else if (hip_interop_[pic_idx].hip_ext_mem != nullptr) {
        // Subsequent frame on an already-imported surface slot: re-copy tiled→linear.
        rocdec_status = va_video_decoder_.CopyToStagingBuffer(pic_idx);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "CopyToStagingBuffer failed for pic_idx=" + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }
    }

    // CPU staging fallback — used when zero-copy interop is not available.
    // Must run every frame since surfaces are reused by the decoder.
    if (hip_interop_[pic_idx].hip_ext_mem == nullptr) {
        uint8_t* cpu_ptr = nullptr;
        uint32_t va_width = 0, va_height = 0, num_planes = 0;
        uint32_t va_pitches[3] = {}, va_offsets[3] = {};

        rocdec_status = va_video_decoder_.MapSurfaceToCPU(pic_idx, &cpu_ptr, va_width, va_height, va_pitches, va_offsets, num_planes);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Failed to map surface to CPU for picture idx = " + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        uint32_t bytes_per_pixel = (decoder_create_info_.bit_depth_minus_8 > 0) ? 2 : 1;
        uint32_t aligned_pitch = ((decoder_create_info_.width + 255) & ~255u) * bytes_per_pixel;
        uint32_t aligned_vstride = (decoder_create_info_.height + 15) & ~15u;
        uint32_t chroma_vstride = aligned_vstride / 2;
        uint32_t total_size = aligned_pitch * (aligned_vstride + chroma_vstride);

        if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
            CHECK_HIP(hipMalloc(&hip_interop_[pic_idx].hip_mapped_device_mem, total_size));
            hip_interop_[pic_idx].width = decoder_create_info_.width;
            hip_interop_[pic_idx].height = decoder_create_info_.height;
            hip_interop_[pic_idx].num_layers = (num_planes >= 2) ? 2 : 1;
            hip_interop_[pic_idx].pitch[0] = aligned_pitch;
            hip_interop_[pic_idx].offset[0] = 0;
            if (num_planes >= 2) {
                hip_interop_[pic_idx].pitch[1] = aligned_pitch;
                hip_interop_[pic_idx].offset[1] = aligned_pitch * aligned_vstride;
            }
        }

        // Copy each plane with stride conversion.
        uint32_t plane_height = decoder_create_info_.height;
        if (va_pitches[0] == aligned_pitch) {
            CHECK_HIP(hipMemcpy(hip_interop_[pic_idx].hip_mapped_device_mem,
                                cpu_ptr + va_offsets[0], aligned_pitch * plane_height, hipMemcpyHostToDevice));
        } else {
            CHECK_HIP(hipMemcpy2D(hip_interop_[pic_idx].hip_mapped_device_mem, aligned_pitch,
                                  cpu_ptr + va_offsets[0], va_pitches[0],
                                  decoder_create_info_.width * bytes_per_pixel, plane_height,
                                  hipMemcpyHostToDevice));
        }
        if (num_planes >= 2) {
            uint32_t chroma_height = decoder_create_info_.height / 2;
            uint8_t* dst_uv = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[1];
            if (va_pitches[1] == aligned_pitch) {
                CHECK_HIP(hipMemcpy(dst_uv, cpu_ptr + va_offsets[1],
                                    aligned_pitch * chroma_height, hipMemcpyHostToDevice));
            } else {
                CHECK_HIP(hipMemcpy2D(dst_uv, aligned_pitch,
                                      cpu_ptr + va_offsets[1], va_pitches[1],
                                      decoder_create_info_.width * bytes_per_pixel, chroma_height,
                                      hipMemcpyHostToDevice));
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

    for (int i = 0; i < hip_interop_[pic_idx].num_layers; i++) {
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
