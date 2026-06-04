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

#ifdef ROCDECODE_BUILD_LINUX
RocDecoder::RocDecoder(RocDecoderCreateInfo& decoder_create_info): va_video_decoder_{decoder_create_info}, decoder_create_info_{decoder_create_info} {
}
#elif defined(ROCDECODE_BUILD_WINDOWS)
RocDecoder::RocDecoder(RocDecoderCreateInfo& decoder_create_info): decoder_create_info_{decoder_create_info} {
    InfoLog(g_rocdec_logger, "RocDecoder: Constructor - creating PAL video decoder");
    pal_video_decoder_ = std::make_unique<rocdec::PalVideoDecoder>();
    InfoLog(g_rocdec_logger, ROCDEC_STR("RocDecoder: Constructor - pal_video_decoder_ created"));
}
#else
RocDecoder::RocDecoder(RocDecoderCreateInfo& decoder_create_info): decoder_create_info_{decoder_create_info} {
}
#endif

 RocDecoder::~RocDecoder() {
    // clean up the VA-API/HIP interop memories
    for(size_t i = 0; i < hip_interop_.size(); i++) {
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
    }
}

rocDecStatus RocDecoder::InitializeDecoder() {
    FunctionEntryLog(g_rocdec_logger);
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

    InfoLog(g_rocdec_logger, "RocDecoder: num_decode_surfaces = " + ROCDEC_TOSTR(decoder_create_info_.num_decode_surfaces));
    if (decoder_create_info_.num_decode_surfaces < 1) {
        CriticalLog(g_rocdec_logger, "Invalid number of decode surfaces.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }

    InfoLog(g_rocdec_logger, "RocDecoder: Resizing hip_interop_ to " + ROCDEC_TOSTR(decoder_create_info_.num_decode_surfaces));
    hip_interop_.resize(decoder_create_info_.num_decode_surfaces);
    for (size_t i = 0; i < hip_interop_.size(); i++) {
        memset((void *)&hip_interop_[i], 0, sizeof(hip_interop_[i]));
    }
    InfoLog(g_rocdec_logger, "RocDecoder: hip_interop_ initialized");

#ifdef ROCDECODE_BUILD_LINUX
    rocdec_status = va_video_decoder_.InitializeDecoder();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to initialize the VAAPI Video decoder.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
#elif defined(ROCDECODE_BUILD_WINDOWS)
    InfoLog(g_rocdec_logger, ROCDEC_STR("RocDecoder: Checking pal_video_decoder_ pointer"));
    if (!pal_video_decoder_) {
        CriticalLog(g_rocdec_logger, "RocDecoder: pal_video_decoder_ is NULL!");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    InfoLog(g_rocdec_logger, ROCDEC_STR("RocDecoder: pal_video_decoder_ pointer is valid"));

    InfoLog(g_rocdec_logger, "RocDecoder: Calling pal_video_decoder_->Initialize()...");
    rocdec_status = pal_video_decoder_->Initialize(
        decoder_create_info_.codec_type,
        decoder_create_info_.width,
        decoder_create_info_.height,
        decoder_create_info_.bit_depth_minus_8 + 8,
        decoder_create_info_.num_decode_surfaces
    );
    InfoLog(g_rocdec_logger, "RocDecoder: pal_video_decoder_->Initialize() returned " + ROCDEC_TOSTR((int)rocdec_status));

    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to initialize the PAL Video decoder.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
#else
    CriticalLog(g_rocdec_logger, "No decoder backend available.");
    rocdec_status = ROCDEC_NOT_SUPPORTED;
#endif
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::DecodeFrame(RocdecPicParams *pic_params) {
    FunctionEntryLog(g_rocdec_logger);
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
#ifdef ROCDECODE_BUILD_LINUX
    rocdec_status = va_video_decoder_.SubmitDecode(pic_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Decode submission is not successful.");
    }
#elif defined(ROCDECODE_BUILD_WINDOWS)
    rocdec_status = pal_video_decoder_->DecodeFrame(pic_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "PAL decode submission failed.");
    }
#else
    ErrorLog(g_rocdec_logger, "No decoder backend available.");
    rocdec_status = ROCDEC_NOT_SUPPORTED;
#endif
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status) {
    FunctionEntryLog(g_rocdec_logger);
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
#ifdef ROCDECODE_BUILD_LINUX
    rocdec_status = va_video_decoder_.GetDecodeStatus(pic_idx, decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to query the decode status.");
    }
#elif defined(ROCDECODE_BUILD_WINDOWS)
    rocdec_status = pal_video_decoder_->GetDecodeStatus(pic_idx, decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to query PAL decode status.");
    }
#else
    ErrorLog(g_rocdec_logger, "No decoder backend available.");
    rocdec_status = ROCDEC_NOT_SUPPORTED;
#endif
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    FunctionEntryLog(g_rocdec_logger);
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
#ifdef ROCDECODE_BUILD_LINUX
    rocdec_status = va_video_decoder_.ReconfigureDecoder(reconfig_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Reconfiguration of the decoder failed.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
#elif defined(ROCDECODE_BUILD_WINDOWS)
    rocdec_status = pal_video_decoder_->Reconfigure(reconfig_params->width, reconfig_params->height);
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "PAL decoder reconfiguration failed.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
#else
    CriticalLog(g_rocdec_logger, "No decoder backend available.");
    rocdec_status = ROCDEC_NOT_SUPPORTED;
#endif
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::GetVideoFrame(int pic_idx, void *dev_mem_ptr[3], uint32_t horizontal_pitch[3], RocdecProcParams *vid_postproc_params) {
    FunctionEntryLog(g_rocdec_logger);
    if (pic_idx >= hip_interop_.size() || dev_mem_ptr == nullptr || vid_postproc_params == nullptr) {
        CriticalLog(g_rocdec_logger, "Invalid parameters.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

#ifdef ROCDECODE_BUILD_LINUX

    // wait on current surface to make sure that it is ready for the HIP interop
    rocdec_status = va_video_decoder_.SyncSurface(pic_idx);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to export surface for picture idx = " + ROCDEC_TOSTR(pic_idx));
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }

    // do the VA-API/HIP interop once per surface and save it for reusing
    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        hipExternalMemoryHandleDesc external_mem_handle_desc = {};
        hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
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
        InfoLog(g_rocdec_logger, "HIP INTEROP DEBUG ----- Imported external memory for picture idx = " + ROCDEC_TOSTR(pic_idx) +
                " size = " + ROCDEC_TOSTR(external_mem_handle_desc.size));

        external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
        CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem, hip_interop_[pic_idx].hip_ext_mem, &external_mem_buffer_desc));
        InfoLog(g_rocdec_logger, "HIP INTEROP DEBUG ----- Got mapped buffer for picture idx = " + ROCDEC_TOSTR(pic_idx) +
                " size = " + ROCDEC_TOSTR(external_mem_buffer_desc.size) +
                " device pointer = " + ROCDEC_TOSTR((uintptr_t)hip_interop_[pic_idx].hip_mapped_device_mem));

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

    for (int i = 0; i < hip_interop_[pic_idx].num_layers; i++) {
        dev_mem_ptr[i] = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[i];
        horizontal_pitch[i] = hip_interop_[pic_idx].pitch[i];
    }
#elif defined(ROCDECODE_BUILD_WINDOWS)
    // Windows/PAL backend: Export PAL image for HIP interop

    // Sync decode to ensure frame is ready
    RocdecDecodeStatus decode_status = {};
    rocdec_status = pal_video_decoder_->GetDecodeStatus(pic_idx, &decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        ErrorLog(g_rocdec_logger, "Failed to query decode status for picture idx = " + ROCDEC_TOSTR(pic_idx));
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }

    // Do HIP interop once per surface and reuse
    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        // Export PAL surface
        uint32_t width = 0, height = 0;
        uint32_t pitch[3] = {0};
        uint32_t offset[3] = {0};
        uint32_t num_planes = 0;
        void* kmt_handle = nullptr;
        size_t mem_size = 0;

        rocdec_status = pal_video_decoder_->ExportSurface(
            pic_idx, width, height, pitch, offset, num_planes, kmt_handle, mem_size);
        if (rocdec_status != ROCDEC_SUCCESS) {
            ErrorLog(g_rocdec_logger, "Failed to export PAL surface for picture idx = " + ROCDEC_TOSTR(pic_idx));
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        // Import into HIP using Windows KMT handle
        hipExternalMemoryHandleDesc external_mem_handle_desc = {};
        hipExternalMemoryBufferDesc external_mem_buffer_desc = {};

        external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
        external_mem_handle_desc.handle.win32.handle = kmt_handle;
        external_mem_handle_desc.size = mem_size;
        external_mem_handle_desc.flags = 0;

        CHECK_HIP(hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc));

        external_mem_buffer_desc.size = mem_size;
        external_mem_buffer_desc.offset = 0;
        external_mem_buffer_desc.flags = 0;

        CHECK_HIP(hipExternalMemoryGetMappedBuffer(
            (void**)&hip_interop_[pic_idx].hip_mapped_device_mem,
            hip_interop_[pic_idx].hip_ext_mem,
            &external_mem_buffer_desc));

        hip_interop_[pic_idx].width = width;
        hip_interop_[pic_idx].height = height;
        hip_interop_[pic_idx].num_layers = num_planes;

        for (uint32_t i = 0; i < num_planes; i++) {
            hip_interop_[pic_idx].offset[i] = offset[i];
            hip_interop_[pic_idx].pitch[i] = pitch[i];
        }

        InfoLog(g_rocdec_logger, "PAL-HIP interop established for picture idx = " + ROCDEC_TOSTR(pic_idx) +
                " width=" + ROCDEC_TOSTR(width) + " height=" + ROCDEC_TOSTR(height) +
                " num_planes=" + ROCDEC_TOSTR(num_planes));
    }

    // Return device memory pointers
    for (int i = 0; i < hip_interop_[pic_idx].num_layers; i++) {
        dev_mem_ptr[i] = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[i];
        horizontal_pitch[i] = hip_interop_[pic_idx].pitch[i];
    }
#else
    // No decoder backend available
    ErrorLog(g_rocdec_logger, "No decoder backend available.");
    rocdec_status = ROCDEC_NOT_SUPPORTED;
#endif
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus RocDecoder::FreeVideoFrame(int pic_idx) {
    FunctionEntryLog(g_rocdec_logger);
    if (pic_idx >= hip_interop_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }

    if (hip_interop_[pic_idx].hip_mapped_device_mem != nullptr)
        CHECK_HIP(hipFree(hip_interop_[pic_idx].hip_mapped_device_mem));
    if (hip_interop_[pic_idx].hip_ext_mem != nullptr)
        CHECK_HIP(hipDestroyExternalMemory(hip_interop_[pic_idx].hip_ext_mem));

    memset((void *)&hip_interop_[pic_idx], 0, sizeof(hip_interop_[pic_idx]));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
