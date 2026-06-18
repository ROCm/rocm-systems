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
    // EXPERIMENTAL: Windows path — CPU staging (vaDeriveImage/vaMapBuffer → hipMemcpy).
    // This is a temporary workaround. Zero-copy D3D12↔HIP interop is required for
    // production but hipImportExternalMemory does not yet support D3D12 resource handles
    // on AMD HIP for Windows. Once HIP adds D3D12 interop support, this path should be
    // replaced with the zero-copy ExportSurfaceNTHandle → hipImportExternalMemory flow.
    // The copy must happen every frame since surfaces are reused by the decoder.
    {
        static bool warned_once = false;
        if (!warned_once) {
            WarningLog(g_rocdec_logger, "EXPERIMENTAL: using CPU-staged frame transfer (no D3D12↔HIP zero-copy interop).");
            warned_once = true;
        }
        uint8_t* cpu_ptr = nullptr;
        uint32_t va_width = 0, va_height = 0, num_planes = 0;
        uint32_t va_pitches[3] = {}, va_offsets[3] = {};

        rocdec_status = va_video_decoder_.MapSurfaceToCPU(pic_idx, &cpu_ptr, va_width, va_height, va_pitches, va_offsets, num_planes);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Failed to map surface to CPU for picture idx = " + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        // Use the same aligned stride that the Linux DRM/VCN path produces (align(width, 256))
        // so the downstream sample layer's size calculations match our device allocation.
        uint32_t bytes_per_pixel = (decoder_create_info_.bit_depth_minus_8 > 0) ? 2 : 1;
        uint32_t aligned_pitch = ((decoder_create_info_.width + 255) & ~255u) * bytes_per_pixel;
        uint32_t aligned_vstride = (decoder_create_info_.height + 15) & ~15u;
        uint32_t chroma_vstride = aligned_vstride / 2; // NV12/P010: chroma is half height
        uint32_t total_size = aligned_pitch * (aligned_vstride + chroma_vstride);

        // Allocate device memory once, reuse for subsequent frames on the same surface slot.
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

        // Copy each plane row-by-row from the VA image pitch to the aligned device pitch.
        // Plane 0 (Y/luma):
        uint32_t plane_height = decoder_create_info_.height;
        if (va_pitches[0] == aligned_pitch) {
            // Pitches match — bulk copy luma plane.
            CHECK_HIP(hipMemcpy(hip_interop_[pic_idx].hip_mapped_device_mem,
                                cpu_ptr + va_offsets[0], aligned_pitch * plane_height, hipMemcpyHostToDevice));
        } else {
            CHECK_HIP(hipMemcpy2D(hip_interop_[pic_idx].hip_mapped_device_mem, aligned_pitch,
                                  cpu_ptr + va_offsets[0], va_pitches[0],
                                  decoder_create_info_.width * bytes_per_pixel, plane_height,
                                  hipMemcpyHostToDevice));
        }
        // Plane 1 (UV/chroma) if present:
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
