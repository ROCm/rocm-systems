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

RocDecoder::RocDecoder(RocDecoderCreateInfo& decoder_create_info): va_video_decoder_{decoder_create_info}, decoder_create_info_{decoder_create_info} {}

 RocDecoder::~RocDecoder() {
    // clean up the VA-API/HIP interop memories
    for(auto i = 0; i < hip_interop_.size(); i++) {
        if (hip_interop_[i].hip_mapped_device_mem != nullptr) {
            hipError_t hip_status = hipFree(hip_interop_[i].hip_mapped_device_mem);
            if (hip_status != hipSuccess) {
                logger_.CriticalLog(MakeMsg("hipFree failed for picture idx = " + TOSTR(i)));
            }
        }
        if (hip_interop_[i].hip_ext_mem != nullptr) {
            hipError_t hip_status = hipDestroyExternalMemory(hip_interop_[i].hip_ext_mem);
            if (hip_status != hipSuccess) {
                logger_.CriticalLog(MakeMsg("hipDestroyExternalMemory failed for picture idx = " + TOSTR(i)));
            }
        }
    }
 }

 rocDecStatus RocDecoder::InitializeDecoder() {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    if (decoder_create_info_.num_decode_surfaces < 1) {
        logger_.CriticalLog(MakeMsg("Invalid number of decode surfaces."));
        return ROCDEC_INVALID_PARAMETER;
    }
    hip_interop_.resize(decoder_create_info_.num_decode_surfaces);
    for (auto i = 0; i < hip_interop_.size(); i++) {
        memset((void *)&hip_interop_[i], 0, sizeof(hip_interop_[i]));
    }
    rocdec_status = va_video_decoder_.InitializeDecoder();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to initilize the VAAPI Video decoder."));
        return rocdec_status;
    }

     return rocdec_status;
 }

rocDecStatus RocDecoder::DecodeFrame(RocdecPicParams *pic_params) {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = va_video_decoder_.SubmitDecode(pic_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("Decode submission is not successful."));
    }

     return rocdec_status;
}

rocDecStatus RocDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status) {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = va_video_decoder_.GetDecodeStatus(pic_idx, decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("Failed to query the decode status."));
    }
    return rocdec_status;
}

rocDecStatus RocDecoder::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    if (reconfig_params == nullptr || reconfig_params->width == 0 || reconfig_params->height == 0 ||
        reconfig_params->num_decode_surfaces < 1 || reconfig_params->bit_depth_minus_8 > 2) {
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status;
    for (int pic_idx = 0; pic_idx < hip_interop_.size(); pic_idx++) {
        rocdec_status = FreeVideoFrame(pic_idx);
        if (rocdec_status != ROCDEC_SUCCESS) {
            logger_.ErrorLog(MakeMsg("Releasing the video frame for picture idx = " + TOSTR(pic_idx) + " failed during reconfiguration."));
            return rocdec_status;
        }
    }
    if (hip_interop_.size() != reconfig_params->num_decode_surfaces) {
        hip_interop_.resize(reconfig_params->num_decode_surfaces);
    }
    rocdec_status = va_video_decoder_.ReconfigureDecoder(reconfig_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Reconfiguration of the decoder failed."));
        return rocdec_status;
    }
    return rocdec_status;
}

rocDecStatus RocDecoder::GetVideoFrame(int pic_idx, void *dev_mem_ptr[3], uint32_t horizontal_pitch[3], RocdecProcParams *vid_postproc_params) {
    if (pic_idx >= hip_interop_.size() || &dev_mem_ptr[0] == nullptr || vid_postproc_params == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

    // wait on current surface to make sure that it is ready for the HIP interop
    rocdec_status = va_video_decoder_.SyncSurface(pic_idx);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("Failed to export surface for picture idx = " + TOSTR(pic_idx)));
        return rocdec_status;
    }

    // do the VA-API/HIP interop once per surface and save it for reusing
    if (hip_interop_[pic_idx].hip_mapped_device_mem == nullptr) {
        hipExternalMemoryHandleDesc external_mem_handle_desc = {};
        hipExternalMemoryBufferDesc external_mem_buffer_desc = {};
        VADRMPRIMESurfaceDescriptor va_drm_prime_surface_desc = {};

        rocdec_status = va_video_decoder_.ExportSurface(pic_idx, va_drm_prime_surface_desc);
        if (rocdec_status != ROCDEC_SUCCESS) {
            logger_.ErrorLog(MakeMsg("Failed to export surface for picture idx = " + TOSTR(pic_idx)));
            return rocdec_status;
        }

        external_mem_handle_desc.type = hipExternalMemoryHandleTypeOpaqueFd;
        external_mem_handle_desc.handle.fd = va_drm_prime_surface_desc.objects[0].fd;
        external_mem_handle_desc.size = va_drm_prime_surface_desc.objects[0].size;

        CHECK_HIP(hipImportExternalMemory(&hip_interop_[pic_idx].hip_ext_mem, &external_mem_handle_desc));

        external_mem_buffer_desc.size = va_drm_prime_surface_desc.objects[0].size;
        CHECK_HIP(hipExternalMemoryGetMappedBuffer((void**)&hip_interop_[pic_idx].hip_mapped_device_mem, hip_interop_[pic_idx].hip_ext_mem, &external_mem_buffer_desc));

        hip_interop_[pic_idx].width = va_drm_prime_surface_desc.width;
        hip_interop_[pic_idx].height = va_drm_prime_surface_desc.height;

        hip_interop_[pic_idx].offset[0] = va_drm_prime_surface_desc.layers[0].offset[0];
        hip_interop_[pic_idx].offset[1] = va_drm_prime_surface_desc.layers[1].offset[0];
        hip_interop_[pic_idx].offset[2] = va_drm_prime_surface_desc.layers[2].offset[0];

        hip_interop_[pic_idx].pitch[0] = va_drm_prime_surface_desc.layers[0].pitch[0];
        hip_interop_[pic_idx].pitch[1] = va_drm_prime_surface_desc.layers[1].pitch[0];
        hip_interop_[pic_idx].pitch[2] = va_drm_prime_surface_desc.layers[2].pitch[0];

        hip_interop_[pic_idx].num_layers = va_drm_prime_surface_desc.num_layers;

        for (auto i = 0; i < va_drm_prime_surface_desc.num_objects; ++i) {
            close(va_drm_prime_surface_desc.objects[i].fd);
        }
    }

    *&dev_mem_ptr[0] = hip_interop_[pic_idx].hip_mapped_device_mem;
    horizontal_pitch[0] = hip_interop_[pic_idx].pitch[0];
    if (hip_interop_[pic_idx].num_layers == 2) {
        *&dev_mem_ptr[1] = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[1];
        horizontal_pitch[1] = hip_interop_[pic_idx].pitch[1];
    } else if (hip_interop_[pic_idx].num_layers == 3) {
        *&dev_mem_ptr[2] = hip_interop_[pic_idx].hip_mapped_device_mem + hip_interop_[pic_idx].offset[2];
        horizontal_pitch[2] = hip_interop_[pic_idx].pitch[2];
    }

    return rocdec_status;
}

rocDecStatus RocDecoder::FreeVideoFrame(int pic_idx) {
    if (pic_idx >= hip_interop_.size()) {
        return ROCDEC_INVALID_PARAMETER;
    }

    if (hip_interop_[pic_idx].hip_mapped_device_mem != nullptr)
        CHECK_HIP(hipFree(hip_interop_[pic_idx].hip_mapped_device_mem));
    if (hip_interop_[pic_idx].hip_ext_mem != nullptr)
        CHECK_HIP(hipDestroyExternalMemory(hip_interop_[pic_idx].hip_ext_mem));

    memset((void *)&hip_interop_[pic_idx], 0, sizeof(hip_interop_[pic_idx]));

    return ROCDEC_SUCCESS;
}
