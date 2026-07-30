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

#include "vaapi_videodecoder.h"

#include <algorithm>
#include <cctype>
#include <stdlib.h>

VaapiVideoDecoder::VaapiVideoDecoder(RocDecoderCreateInfo &decoder_create_info) : decoder_create_info_{decoder_create_info},
    output_surface_format_override_{false}, va_display_{0}, va_config_attrib_{{}}, va_config_id_{0}, va_profile_ {VAProfileNone},
    va_context_id_{0}, va_surface_ids_{{}}, supports_modifiers_{false},
#ifdef _WIN32
    d3d12_device_{nullptr}, d3d12_shared_resources_{},
    d3d12_copy_queue_{nullptr}, d3d12_cmd_allocator_{nullptr}, d3d12_cmd_list_{nullptr},
    d3d12_fence_{nullptr}, d3d12_fence_event_{nullptr}, d3d12_fence_value_{0},
    d3d12_staging_buffers_{},
#endif
    pic_params_buf_id_{0}, iq_matrix_buf_id_{0}, num_slices_{0},
    slice_data_buf_id_{0} {
};

VaapiVideoDecoder::~VaapiVideoDecoder() {
    if (va_display_) {
        rocDecStatus rocdec_status = ROCDEC_SUCCESS;
        rocdec_status = DestroyDataBuffers();
        if (rocdec_status != ROCDEC_SUCCESS) {
            CriticalLog(g_rocdec_logger, "DestroyDataBuffers failed");
        }
        VAStatus va_status = VA_STATUS_SUCCESS;
        va_status = vaDestroySurfaces(va_display_, va_surface_ids_.data(), static_cast<int>(va_surface_ids_.size()));
        if (va_status != VA_STATUS_SUCCESS) {
            CriticalLog(g_rocdec_logger, "vaDestroySurfaces failed");
        }
#ifdef _WIN32
        // Release D3D12 resources after VA surfaces are destroyed
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
#endif
        if (va_context_id_) {
            va_status = vaDestroyContext(va_display_, va_context_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                CriticalLog(g_rocdec_logger, "vaDestroyContext failed");
            }
        }
        if (va_config_id_) {
            va_status = vaDestroyConfig(va_display_, va_config_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                CriticalLog(g_rocdec_logger, "vaDestroyConfig failed");
            }
        }
        if (vaTerminate(va_display_) != VA_STATUS_SUCCESS) {
            CriticalLog(g_rocdec_logger, "Failed to terminate VA");
        }
    }
}

void VaapiVideoDecoder::ValidateOutputFormat() {
    // When content is 8-bit; a 16-bit output format would exceed the content bit depth.
    if (decoder_create_info_.bit_depth_minus_8 == 0) {
        rocDecVideoSurfaceFormat adjusted = decoder_create_info_.output_format;
        switch (decoder_create_info_.output_format) {
            case rocDecVideoSurfaceFormat_P016:
                adjusted = rocDecVideoSurfaceFormat_NV12;
                break;
            case rocDecVideoSurfaceFormat_YUV444_16Bit:
                adjusted = rocDecVideoSurfaceFormat_YUV444;
                break;
            case rocDecVideoSurfaceFormat_YUV420_16Bit:
                adjusted = rocDecVideoSurfaceFormat_YUV420;
                break;
            case rocDecVideoSurfaceFormat_YUV422_16Bit:
                adjusted = rocDecVideoSurfaceFormat_YUV422;
                break;
            default:
                break; // already an 8-bit format, no adjustment needed
        }
        if (adjusted != decoder_create_info_.output_format) {
            WarningLog(g_rocdec_logger, ("output_format (" + ROCDEC_TOSTR(static_cast<uint32_t>(decoder_create_info_.output_format)) +
                ") bit depth exceeds content bit depth (bit_depth_minus_8 = " +
                ROCDEC_TOSTR(decoder_create_info_.bit_depth_minus_8) + "). Adjusting output_format to " +
                ROCDEC_TOSTR(static_cast<uint32_t>(adjusted)) + "."));
            decoder_create_info_.output_format = adjusted;
        }
    }
}

void VaapiVideoDecoder::SetNativeOutputFormat() {
    switch (decoder_create_info_.chroma_format) {
        case rocDecVideoChromaFormat_Monochrome:
        case rocDecVideoChromaFormat_420:
            decoder_create_info_.output_format = (decoder_create_info_.bit_depth_minus_8 > 0) ? rocDecVideoSurfaceFormat_P016 : rocDecVideoSurfaceFormat_NV12;
            break;
        case rocDecVideoChromaFormat_422:
            decoder_create_info_.output_format = (decoder_create_info_.bit_depth_minus_8 > 0) ? rocDecVideoSurfaceFormat_YUV422_16Bit : rocDecVideoSurfaceFormat_YUV422;
            break;
        case rocDecVideoChromaFormat_444:
            decoder_create_info_.output_format = (decoder_create_info_.bit_depth_minus_8 > 0) ? rocDecVideoSurfaceFormat_YUV444_16Bit : rocDecVideoSurfaceFormat_YUV444;
            break;
        default:
            decoder_create_info_.output_format = rocDecVideoSurfaceFormat_NV12;
            break;
    }
}

void VaapiVideoDecoder::CheckOutputFormat() {
    if (decoder_create_info_.output_format == rocDecVideoSurfaceFormat_Native) {
        output_surface_format_override_ = false;
        // Resolve Native to the concrete format that matches the stream's chroma format and bit depth.
        SetNativeOutputFormat();
    } else {
        output_surface_format_override_ = true;
        ValidateOutputFormat();
    }
}

rocDecStatus VaapiVideoDecoder::InitializeDecoder() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

    // Check if the output surface format is set by the user or native.
    CheckOutputFormat();

    // Before initializing the VAAPI, first check to see if the requested codec config is supported
    if (!IsCodecConfigSupported(decoder_create_info_.device_id, decoder_create_info_.codec_type, decoder_create_info_.chroma_format,
        decoder_create_info_.bit_depth_minus_8, decoder_create_info_.output_format)) {
        CriticalLog(g_rocdec_logger, "The codec config combination is not supported.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_SUPPORTED;
    }

    VaContext& va_ctx = VaContext::GetInstance();
    uint32_t va_ctx_id;
    if ((rocdec_status = va_ctx.GetVaContext(decoder_create_info_.device_id, &va_ctx_id)) != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to get VA context.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    if ((rocdec_status = va_ctx.GetVaDisplay(va_ctx_id, &va_display_)) != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to get VA display.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    rocdec_status = CreateDecoderConfig();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to create a VAAPI decoder configuration.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    rocdec_status = CreateSurfaces();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to create VAAPI surfaces.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    rocdec_status = CreateContext();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to create a VAAPI context.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

rocDecStatus VaapiVideoDecoder::SubmitDecode(RocdecPicParams *pPicParams) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(pPicParams));
    void *pic_params_ptr, *iq_matrix_ptr, *slice_params_ptr;
    uint32_t pic_params_size, iq_matrix_size, slice_params_size;
    bool scaling_list_enabled = false;
    VASurfaceID curr_surface_id;

    // Get the surface id for the current picture, assuming 1:1 mapping between DPB and VAAPI decoded surfaces.
    if (pPicParams->curr_pic_idx >= va_surface_ids_.size() || pPicParams->curr_pic_idx < 0) {
        ErrorLog(g_rocdec_logger, "curr_pic_idx exceeded the VAAPI surface pool limit.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    curr_surface_id = va_surface_ids_[pPicParams->curr_pic_idx];

    // Upload data buffers
    switch (decoder_create_info_.codec_type) {
        case rocDecVideoCodec_HEVC: {
            pPicParams->pic_params.hevc.curr_pic.pic_idx = curr_surface_id;
            for (int i = 0; i < 15; i++) {
                if (pPicParams->pic_params.hevc.ref_frames[i].pic_idx != 0xFF) {
                    if (pPicParams->pic_params.hevc.ref_frames[i].pic_idx >= va_surface_ids_.size() || pPicParams->pic_params.hevc.ref_frames[i].pic_idx < 0) {
                        ErrorLog(g_rocdec_logger, "Reference frame index exceeded the VAAPI surface pool limit.");
                        FunctionExitLog(g_rocdec_logger);
                        return ROCDEC_INVALID_PARAMETER;
                    }
                    pPicParams->pic_params.hevc.ref_frames[i].pic_idx = va_surface_ids_[pPicParams->pic_params.hevc.ref_frames[i].pic_idx];
                }
            }
            pic_params_ptr = (void*)&pPicParams->pic_params.hevc;
            pic_params_size = sizeof(RocdecHevcPicParams);

            if (pPicParams->pic_params.hevc.pic_fields.bits.scaling_list_enabled_flag) {
                scaling_list_enabled = true;
                iq_matrix_ptr = (void*)&pPicParams->iq_matrix.hevc;
                iq_matrix_size = sizeof(RocdecHevcIQMatrix);
            }

            slice_params_ptr = (void*)pPicParams->slice_params.hevc;
            slice_params_size = sizeof(RocdecHevcSliceParams);

            if ((pic_params_size != sizeof(VAPictureParameterBufferHEVC)) || (scaling_list_enabled && (iq_matrix_size != sizeof(VAIQMatrixBufferHEVC))) || 
                (slice_params_size != sizeof(VASliceParameterBufferHEVC))) {
                    ErrorLog(g_rocdec_logger, "HEVC data_buffer parameter_size not matching vaapi parameter buffer size.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_AVC: {
            pPicParams->pic_params.avc.curr_pic.pic_idx = curr_surface_id;
            for (int i = 0; i < 16; i++) {
                if (pPicParams->pic_params.avc.ref_frames[i].pic_idx != 0xFF) {
                    if (pPicParams->pic_params.avc.ref_frames[i].pic_idx >= va_surface_ids_.size() || pPicParams->pic_params.avc.ref_frames[i].pic_idx < 0) {
                        ErrorLog(g_rocdec_logger, "Reference frame index exceeded the VAAPI surface pool limit.");
                        FunctionExitLog(g_rocdec_logger);
                        return ROCDEC_INVALID_PARAMETER;
                    }
                    pPicParams->pic_params.avc.ref_frames[i].pic_idx = va_surface_ids_[pPicParams->pic_params.avc.ref_frames[i].pic_idx];
                }
            }
            pic_params_ptr = (void*)&pPicParams->pic_params.avc;
            pic_params_size = sizeof(RocdecAvcPicParams);

            scaling_list_enabled = true;
            iq_matrix_ptr = (void*)&pPicParams->iq_matrix.avc;
            iq_matrix_size = sizeof(RocdecAvcIQMatrix);

            slice_params_ptr = (void*)pPicParams->slice_params.avc;
            slice_params_size = sizeof(RocdecAvcSliceParams);

            if ((pic_params_size != sizeof(VAPictureParameterBufferH264)) || (iq_matrix_size != sizeof(VAIQMatrixBufferH264)) || (slice_params_size != sizeof(VASliceParameterBufferH264))) {
                    ErrorLog(g_rocdec_logger, "AVC data_buffer parameter_size not matching vaapi parameter buffer size.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_VP9: {
            for (int i = 0; i < 8; i++) {
                if (pPicParams->pic_params.vp9.reference_frames[i] != 0xFF) {
                    if (pPicParams->pic_params.vp9.reference_frames[i] >= va_surface_ids_.size()) {
                        ErrorLog(g_rocdec_logger, "Reference frame index exceeded the VAAPI surface pool limit.");
                        FunctionExitLog(g_rocdec_logger);
                        return ROCDEC_INVALID_PARAMETER;
                    }
                    pPicParams->pic_params.vp9.reference_frames[i] = va_surface_ids_[pPicParams->pic_params.vp9.reference_frames[i]];
                }
            }
            pic_params_ptr = (void*)&pPicParams->pic_params.vp9;
            pic_params_size = sizeof(RocdecVp9PicParams);
            slice_params_ptr = (void*)pPicParams->slice_params.vp9;
            slice_params_size = sizeof(RocdecVp9SliceParams);
            if ((pic_params_size != sizeof(VADecPictureParameterBufferVP9)) || (slice_params_size != sizeof(VASliceParameterBufferVP9))) {
                    ErrorLog(g_rocdec_logger, "VP9 data_buffer parameter_size not matching vaapi parameter buffer size.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_AV1: {
            pPicParams->pic_params.av1.current_frame = curr_surface_id;

            if (pPicParams->pic_params.av1.current_display_picture != 0xFF) {
                if (pPicParams->pic_params.av1.current_display_picture >= va_surface_ids_.size() || pPicParams->pic_params.av1.current_display_picture < 0) {
                    ErrorLog(g_rocdec_logger, "Current display picture index exceeded the VAAPI surface pool limit.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_INVALID_PARAMETER;
                }
                pPicParams->pic_params.av1.current_display_picture = va_surface_ids_[pPicParams->pic_params.av1.current_display_picture];
            }

            for (int i = 0; i < pPicParams->pic_params.av1.anchor_frames_num; i++) {
                if (pPicParams->pic_params.av1.anchor_frames_list[i] >= va_surface_ids_.size() || pPicParams->pic_params.av1.anchor_frames_list[i] < 0) {
                    ErrorLog(g_rocdec_logger, "Anchor frame index exceeded the VAAPI surface pool limit.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_INVALID_PARAMETER;
                }
                pPicParams->pic_params.av1.anchor_frames_list[i] = va_surface_ids_[pPicParams->pic_params.av1.anchor_frames_list[i]];
            }

            for (int i = 0; i < 8; i++) {
                if (pPicParams->pic_params.av1.ref_frame_map[i] != 0xFF) {
                    if (pPicParams->pic_params.av1.ref_frame_map[i] >= va_surface_ids_.size() || pPicParams->pic_params.av1.ref_frame_map[i] < 0) {
                        ErrorLog(g_rocdec_logger, "Reference frame index exceeded the VAAPI surface pool limit.");
                        FunctionExitLog(g_rocdec_logger);
                        return ROCDEC_INVALID_PARAMETER;
                    }
                    pPicParams->pic_params.av1.ref_frame_map[i] = va_surface_ids_[pPicParams->pic_params.av1.ref_frame_map[i]];
                }
            }

            pic_params_ptr = (void*)&pPicParams->pic_params.av1;
            pic_params_size = sizeof(RocdecAv1PicParams);

            slice_params_ptr = (void*)pPicParams->slice_params.av1;
            slice_params_size = sizeof(RocdecAv1SliceParams);

            if ((pic_params_size != sizeof(VADecPictureParameterBufferAV1)) || (slice_params_size != sizeof(VASliceParameterBufferAV1))) {
                    CriticalLog(g_rocdec_logger, "AV1 data_buffer parameter_size not matching vaapi parameter buffer size.");
                    FunctionExitLog(g_rocdec_logger);
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        default: {
            CriticalLog(g_rocdec_logger, "The codec type is not supported.");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_SUPPORTED;
        }
    }

    // Destroy the data buffers of the previous frame
    rocDecStatus rocdec_status = DestroyDataBuffers();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to destroy VAAPI buffer.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }

    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAPictureParameterBufferType, pic_params_size, 1, pic_params_ptr, &pic_params_buf_id_));
    if (scaling_list_enabled) {
        CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VAIQMatrixBufferType, iq_matrix_size, 1, iq_matrix_ptr, &iq_matrix_buf_id_));
    }
    // Resize if needed
    num_slices_ = pPicParams->num_slices;
    if (num_slices_ > slice_params_buf_id_.size()) {
        slice_params_buf_id_.resize(num_slices_, {0});
    }
    for (uint32_t i = 0; i < num_slices_; i++) {
        CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceParameterBufferType, slice_params_size, 1, slice_params_ptr, &slice_params_buf_id_[i]));
        slice_params_ptr = (void*)((uint8_t*)slice_params_ptr + slice_params_size);
    }
    CHECK_VAAPI(vaCreateBuffer(va_display_, va_context_id_, VASliceDataBufferType, pPicParams->bitstream_data_len, 1, (void*)pPicParams->bitstream_data, &slice_data_buf_id_));

    // Sumbmit buffers to VAAPI driver
    CHECK_VAAPI(vaBeginPicture(va_display_, va_context_id_, curr_surface_id));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &pic_params_buf_id_, 1));
    if (scaling_list_enabled) {
        CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &iq_matrix_buf_id_, 1));
    }
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, slice_params_buf_id_.data(), num_slices_));
    CHECK_VAAPI(vaRenderPicture(va_display_, va_context_id_, &slice_data_buf_id_, 1));
    CHECK_VAAPI(vaEndPicture(va_display_, va_context_id_));

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus *decode_status) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx) + ", " + RocDecFmtPtr(decode_status));
    VASurfaceStatus va_surface_status;
    if (pic_idx >= va_surface_ids_.size() || decode_status == nullptr) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaQuerySurfaceStatus(va_display_, va_surface_ids_[pic_idx], &va_surface_status));
    switch (va_surface_status) {
        case VASurfaceRendering:
            decode_status->decode_status = rocDecodeStatus_InProgress;
            break;
        case VASurfaceReady:
            decode_status->decode_status = rocDecodeStatus_Success;
            break;
        default:
           decode_status->decode_status = rocDecodeStatus_Invalid;
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

#ifdef _WIN32
rocDecStatus VaapiVideoDecoder::ExportSurfaceNTHandle(int pic_idx, HANDLE &nt_handle) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= va_surface_ids_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }

    // Always use CreateSharedHandle on our pre-created D3D12 resource to produce the NT handle.
    // This guarantees the handle type is a D3D12 shared handle, consistent with
    // hipExternalMemoryHandleTypeD3D12Resource used by the caller.
    // Note: vaExportSurfaceHandle(VA_SURFACE_ATTRIB_MEM_TYPE_NTHANDLE) is NOT used because
    // the handle origin from vaon12 is unspecified — it may not be a D3D12 CreateSharedHandle
    // handle, which would cause a type mismatch with hipImportExternalMemory.
    if (pic_idx >= d3d12_shared_resources_.size() || d3d12_shared_resources_[pic_idx] == nullptr) {
        CriticalLog(g_rocdec_logger, "No shared D3D12 resource for pic_idx=" + ROCDEC_TOSTR(pic_idx));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }

    HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_shared_resources_[pic_idx], nullptr, GENERIC_ALL, nullptr, &nt_handle);
    if (FAILED(hr)) {
        CriticalLog(g_rocdec_logger, "CreateSharedHandle failed, HRESULT=0x" +
                    ([](HRESULT h) { std::ostringstream o; o << std::hex << static_cast<uint32_t>(h); return o.str(); })(hr));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

uint64_t VaapiVideoDecoder::GetD3D12ResourceAllocationSize(int pic_idx) {
    if (d3d12_device_ == nullptr || pic_idx >= d3d12_shared_resources_.size() || d3d12_shared_resources_[pic_idx] == nullptr) {
        return 0;
    }
    D3D12_RESOURCE_DESC desc = d3d12_shared_resources_[pic_idx]->GetDesc();
    D3D12_RESOURCE_ALLOCATION_INFO alloc_info = d3d12_device_->GetResourceAllocationInfo(0, 1, &desc);
    return alloc_info.SizeInBytes;
}

VaapiVideoDecoder::SurfaceLayout VaapiVideoDecoder::GetSurfaceLayout() const {
    SurfaceLayout layout = {};
    rocDecVideoSurfaceFormat fmt = decoder_create_info_.output_format;
    uint32_t width = decoder_create_info_.width;
    uint32_t height = decoder_create_info_.height;

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

void VaapiVideoDecoder::GetD3D12ResourceLayout(int pic_idx, uint32_t pitches[3], uint32_t offsets[3], uint32_t &num_planes) {
    SurfaceLayout layout = GetSurfaceLayout();
    num_planes = layout.num_planes;
    for (uint32_t i = 0; i < num_planes && i < 3; i++) {
        pitches[i] = layout.plane_pitch[i];
        offsets[i] = layout.plane_offset[i];
    }
    InfoLog(g_rocdec_logger, "Surface layout for pic_idx=" + ROCDEC_TOSTR(pic_idx) +
            ": format=" + ROCDEC_TOSTR(decoder_create_info_.output_format) +
            " planes=" + ROCDEC_TOSTR(num_planes) +
            " pitch=" + ROCDEC_TOSTR(layout.pitch) +
            " total=" + ROCDEC_TOSTR(layout.total_size));
}

rocDecStatus VaapiVideoDecoder::CopyToStagingBuffer(int pic_idx) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= d3d12_shared_resources_.size() || d3d12_shared_resources_[pic_idx] == nullptr ||
        pic_idx >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr ||
        d3d12_copy_queue_ == nullptr) {
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
    UINT src_num_rows[3] = {};
    UINT64 src_row_sizes[3] = {};
    UINT64 src_total = 0;
    // NV12/P010: 2 subresources. Planar YUV: may need more, but D3D12 NV12 is always 2.
    UINT num_subresources = (tex_desc.Format == DXGI_FORMAT_NV12 || tex_desc.Format == DXGI_FORMAT_P010 ||
                             tex_desc.Format == DXGI_FORMAT_P016) ? 2 : 1;
    d3d12_device_->GetCopyableFootprints(&tex_desc, 0, num_subresources, 0,
                                         src_footprints, src_num_rows, src_row_sizes, &src_total);

    // Record copy commands: texture (tiled) → buffer (linear) for each subresource.
    d3d12_cmd_allocator_->Reset();
    d3d12_cmd_list_->Reset(d3d12_cmd_allocator_, nullptr);

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

    d3d12_cmd_list_->Close();
    ID3D12CommandList* lists[] = { d3d12_cmd_list_ };
    d3d12_copy_queue_->ExecuteCommandLists(1, lists);

    // Wait for copy to complete.
    d3d12_fence_value_++;
    d3d12_copy_queue_->Signal(d3d12_fence_, d3d12_fence_value_);
    if (d3d12_fence_->GetCompletedValue() < d3d12_fence_value_) {
        d3d12_fence_->SetEventOnCompletion(d3d12_fence_value_, d3d12_fence_event_);
        WaitForSingleObject(d3d12_fence_event_, INFINITE);
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::ExportStagingBufferHandle(int pic_idx, HANDLE &nt_handle) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    HRESULT hr = d3d12_device_->CreateSharedHandle(d3d12_staging_buffers_[pic_idx], nullptr, GENERIC_ALL, nullptr, &nt_handle);
    if (FAILED(hr)) {
        CriticalLog(g_rocdec_logger, "CreateSharedHandle for staging buffer failed, HRESULT=0x" +
                    ([](HRESULT h) { std::ostringstream o; o << std::hex << static_cast<uint32_t>(h); return o.str(); })(hr));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

uint64_t VaapiVideoDecoder::GetStagingBufferSize(int pic_idx) {
    if (pic_idx >= d3d12_staging_buffers_.size() || d3d12_staging_buffers_[pic_idx] == nullptr) {
        return 0;
    }
    D3D12_RESOURCE_DESC desc = d3d12_staging_buffers_[pic_idx]->GetDesc();
    return desc.Width; // For buffers, Width is the size in bytes.
}

rocDecStatus VaapiVideoDecoder::MapSurfaceToCPU(int pic_idx, uint8_t** cpu_ptr, uint32_t &width, uint32_t &height,
                                                uint32_t pitches[3], uint32_t offsets[3], uint32_t &num_planes) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= va_surface_ids_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }

    // Determine the pixel format matching our surfaces (NV12 for 8-bit 420, P010 for 10-bit, etc.)
    VAImageFormat img_fmt = {};
    if (decoder_create_info_.bit_depth_minus_8 == 0) {
        img_fmt.fourcc = VA_FOURCC_NV12;
    } else {
        img_fmt.fourcc = VA_FOURCC_P010;
    }

    // Create a CPU-accessible image, copy surface data into it, then map.
    // vaGetImage performs an internal GPU→CPU readback (unlike vaDeriveImage which
    // tries zero-copy and fails on D3D12 DEFAULT heap surfaces).
    CHECK_VAAPI(vaCreateImage(va_display_, &img_fmt,
                              decoder_create_info_.width, decoder_create_info_.height, &va_mapped_image_));
    CHECK_VAAPI(vaGetImage(va_display_, va_surface_ids_[pic_idx],
                           0, 0, decoder_create_info_.width, decoder_create_info_.height,
                           va_mapped_image_.image_id));

    void* buf = nullptr;
    CHECK_VAAPI(vaMapBuffer(va_display_, va_mapped_image_.buf, &buf));

    *cpu_ptr = static_cast<uint8_t*>(buf);
    width = va_mapped_image_.width;
    height = va_mapped_image_.height;
    num_planes = va_mapped_image_.num_planes;
    for (uint32_t i = 0; i < num_planes && i < 3; i++) {
        pitches[i] = va_mapped_image_.pitches[i];
        offsets[i] = va_mapped_image_.offsets[i];
    }

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::UnmapSurface(int pic_idx) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    CHECK_VAAPI(vaUnmapBuffer(va_display_, va_mapped_image_.buf));
    CHECK_VAAPI(vaDestroyImage(va_display_, va_mapped_image_.image_id));
    memset(&va_mapped_image_, 0, sizeof(va_mapped_image_));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
#else
rocDecStatus VaapiVideoDecoder::ExportSurface(int pic_idx, VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= va_surface_ids_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaExportSurfaceHandle(va_display_, va_surface_ids_[pic_idx],
                VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY |
                VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                &va_drm_prime_surface_desc));

    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
#endif

rocDecStatus VaapiVideoDecoder::SyncSurface(int pic_idx) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(pic_idx));
    if (pic_idx >= va_surface_ids_.size()) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaSyncSurface(va_display_, va_surface_ids_[pic_idx]));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(reconfig_params));
    if (reconfig_params == nullptr) {
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    if (va_display_ == 0) {
        CriticalLog(g_rocdec_logger, "VAAPI decoder has not been initialized but reconfiguration of the decoder has been requested.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_SUPPORTED;
    }
    CHECK_VAAPI(vaDestroySurfaces(va_display_, va_surface_ids_.data(), static_cast<int>(va_surface_ids_.size())));
    if (va_context_id_) {
        CHECK_VAAPI(vaDestroyContext(va_display_, va_context_id_));
        va_context_id_ = 0;
    }
    // Need to re-create VA config if bit deepth changes
    bool create_va_config = decoder_create_info_.bit_depth_minus_8 != reconfig_params->bit_depth_minus_8 ? true : false;
    if (create_va_config) {
        CHECK_VAAPI(vaDestroyConfig(va_display_, va_config_id_));
        va_config_id_ = 0;
    }

    va_surface_ids_.clear();
    decoder_create_info_.width = reconfig_params->width;
    decoder_create_info_.height = reconfig_params->height;
    decoder_create_info_.num_decode_surfaces = reconfig_params->num_decode_surfaces;
    decoder_create_info_.target_height = reconfig_params->target_height;
    decoder_create_info_.target_width = reconfig_params->target_width;
    decoder_create_info_.bit_depth_minus_8 = reconfig_params->bit_depth_minus_8;

    // Adjust output format if needed
    if (output_surface_format_override_) {
        ValidateOutputFormat();
    } else {
        SetNativeOutputFormat();
    }

    rocDecStatus rocdec_status;
    if (create_va_config) {
        rocdec_status = CreateDecoderConfig();
        if (rocdec_status != ROCDEC_SUCCESS) {
            CriticalLog(g_rocdec_logger, "Failed to create a VAAPI decoder configuration.");
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }
    }
    rocdec_status = CreateSurfaces();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to create VAAPI surfaces during the decoder reconfiguration.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    rocdec_status = CreateContext();
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to create a VAAPI context during the decoder reconfiguration.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    FunctionExitLog(g_rocdec_logger);
    return rocdec_status;
}

bool VaapiVideoDecoder::IsCodecConfigSupported(int device_id, rocDecVideoCodec codec_type, rocDecVideoChromaFormat chroma_format, uint32_t bit_depth_minus8, rocDecVideoSurfaceFormat output_format) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(device_id) + ", " + ROCDEC_TOSTR(codec_type) + ", " +
                             ROCDEC_TOSTR(chroma_format) + ", " + ROCDEC_TOSTR(bit_depth_minus8) + ", " + ROCDEC_TOSTR(output_format));
    RocdecDecodeCaps decode_caps;
    decode_caps.device_id = device_id;
    decode_caps.codec_type = codec_type;
    decode_caps.chroma_format = chroma_format;
    decode_caps.bit_depth_minus_8 = bit_depth_minus8;
    bool supported = (rocDecGetDecoderCaps(&decode_caps) == ROCDEC_SUCCESS) && (decode_caps.is_supported != false) && ((decode_caps.output_format_mask & (1 << output_format)) != 0);
    FunctionExitLog(g_rocdec_logger);
    return supported;
}

rocDecStatus VaapiVideoDecoder::CreateDecoderConfig() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    switch (decoder_create_info_.codec_type) {
        case rocDecVideoCodec_HEVC:
            if (decoder_create_info_.bit_depth_minus_8 == 0) {
                va_profile_ = VAProfileHEVCMain;
            } else if (decoder_create_info_.bit_depth_minus_8 == 2) {
                va_profile_ = VAProfileHEVCMain10;
            }
            break;
        case rocDecVideoCodec_AVC:
            va_profile_ = VAProfileH264Main;
            break;
        case rocDecVideoCodec_VP9:
            if (decoder_create_info_.bit_depth_minus_8 == 0) {
                va_profile_ = VAProfileVP9Profile0;
            } else if (decoder_create_info_.bit_depth_minus_8 == 2) {
                va_profile_ = VAProfileVP9Profile2;
            }
            break;
        case rocDecVideoCodec_AV1:
#if VA_CHECK_VERSION(1, 23, 0)
            if (decoder_create_info_.bit_depth_minus_8 == 4) {
                va_profile_ = VAProfileAV1Profile2;
            } else
#endif
            {
                va_profile_ = VAProfileAV1Profile0;
            }
            break;
        default:
            CriticalLog(g_rocdec_logger, "The codec type is not supported.");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_SUPPORTED;
    }
    va_config_attrib_.type = VAConfigAttribRTFormat;
    CHECK_VAAPI(vaGetConfigAttributes(va_display_, va_profile_, VAEntrypointVLD, &va_config_attrib_, 1));
    CHECK_VAAPI(vaCreateConfig(va_display_, va_profile_, VAEntrypointVLD, &va_config_attrib_, 1, &va_config_id_));
    unsigned int num_attribs = 0;
    CHECK_VAAPI(vaQuerySurfaceAttributes(va_display_, va_config_id_, nullptr, &num_attribs));
    std::vector<VASurfaceAttrib> attribs(num_attribs);
    CHECK_VAAPI(vaQuerySurfaceAttributes(va_display_, va_config_id_, attribs.data(), &num_attribs));
    for (auto attrib : attribs) {
        if (attrib.type == VASurfaceAttribDRMFormatModifiers) {
            supports_modifiers_ = true;
            break;
        }
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::CreateSurfaces() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    if (decoder_create_info_.num_decode_surfaces < 1) {
        CriticalLog(g_rocdec_logger, "Invalid number of decode surfaces.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    va_surface_ids_.resize(decoder_create_info_.num_decode_surfaces);
    std::vector<VASurfaceAttrib> surf_attribs;
    VASurfaceAttrib surf_attrib;
    surf_attrib.type = VASurfaceAttribPixelFormat;
    surf_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surf_attrib.value.type = VAGenericValueTypeInteger;
    uint32_t surface_format;
    switch (decoder_create_info_.chroma_format) {
        case rocDecVideoChromaFormat_Monochrome: {
            surface_format = VA_RT_FORMAT_YUV400;
            surf_attrib.value.value.i = VA_FOURCC_Y800;
        }
            break;
        case rocDecVideoChromaFormat_420: {
            // If the user sets the output surface format
            // Special case for 8-bit output requirement even when the stream is 10-bit and above
            if (output_surface_format_override_ && decoder_create_info_.output_format == rocDecVideoSurfaceFormat_NV12) {
                surface_format = VA_RT_FORMAT_YUV420;
                surf_attrib.value.value.i = VA_FOURCC_NV12;
            } else {
                if (decoder_create_info_.bit_depth_minus_8 == 2) {
                    surface_format = VA_RT_FORMAT_YUV420_10;
                    surf_attrib.value.value.i = VA_FOURCC_P010;
                } else if (decoder_create_info_.bit_depth_minus_8 == 4) {
                    surface_format = VA_RT_FORMAT_YUV420_12;
                    surf_attrib.value.value.i = VA_FOURCC_P012;
                } else {
                    surface_format = VA_RT_FORMAT_YUV420;
                    surf_attrib.value.value.i = VA_FOURCC_NV12;
                }
            }
        }
            break;
        case rocDecVideoChromaFormat_422: {
            if (decoder_create_info_.bit_depth_minus_8 == 2) {
                surface_format = VA_RT_FORMAT_YUV422_10;
                surf_attrib.value.value.i = VA_FOURCC_Y210;
            } else if (decoder_create_info_.bit_depth_minus_8 == 4) {
                surface_format = VA_RT_FORMAT_YUV422_12;
                surf_attrib.value.value.i = VA_FOURCC_Y212;
            } else {
                surface_format = VA_RT_FORMAT_YUV422;
                surf_attrib.value.value.i = VA_FOURCC_422H;
            }
        }
            break;
        case rocDecVideoChromaFormat_444: {
            if (decoder_create_info_.bit_depth_minus_8 == 2) {
                surface_format = VA_RT_FORMAT_YUV444_10;
                surf_attrib.value.value.i = VA_FOURCC_Y410;
            } else if (decoder_create_info_.bit_depth_minus_8 == 4) {
                surface_format = VA_RT_FORMAT_YUV444_12;
                surf_attrib.value.value.i = VA_FOURCC_Y412;
            } else {
                surface_format = VA_RT_FORMAT_YUV444;
                surf_attrib.value.value.i = VA_FOURCC_444P;
            }
        }
            break;
        default:
            CriticalLog(g_rocdec_logger, "The surface type is not supported");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_SUPPORTED;
    }
    surf_attribs.push_back(surf_attrib);
#ifdef _WIN32
    // On Windows, create D3D12 resources with D3D12_HEAP_FLAG_SHARED so that
    // CreateSharedHandle will succeed for HIP interop later.
    // Map the VA fourcc to a DXGI format for D3D12 resource creation.
    DXGI_FORMAT dxgi_format;
    switch (surf_attrib.value.value.i) {
        case VA_FOURCC_NV12: dxgi_format = DXGI_FORMAT_NV12; break;
        case VA_FOURCC_P010: dxgi_format = DXGI_FORMAT_P010; break;
        case VA_FOURCC_P012: dxgi_format = DXGI_FORMAT_P016; break; // D3D12 uses P016 for 12-bit
        default:
            CriticalLog(g_rocdec_logger, "Unsupported fourcc for D3D12 shared surface: 0x" +
                        ([](int f) { std::ostringstream o; o << std::hex << f; return o.str(); })(surf_attrib.value.value.i));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_SUPPORTED;
    }

    if (d3d12_device_ == nullptr) {
        // Get the adapter LUID from the VaContext to create a matching D3D12 device.
        VaContext& va_ctx = VaContext::GetInstance();
        LUID adapter_luid = {};
        for (auto& ctx : va_ctx.va_contexts_) {
            if (ctx.device_id == decoder_create_info_.device_id) {
                adapter_luid = ctx.adapter_luid;
                break;
            }
        }
        IDXGIFactory2* factory = nullptr;
        HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory2), reinterpret_cast<void**>(&factory));
        if (FAILED(hr)) {
            CriticalLog(g_rocdec_logger, "CreateDXGIFactory1 failed");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
        IDXGIAdapter1* adapter = nullptr;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc;
            adapter->GetDesc1(&desc);
            if (desc.AdapterLuid.HighPart == adapter_luid.HighPart &&
                desc.AdapterLuid.LowPart == adapter_luid.LowPart) {
                break;
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
        hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0,
                               __uuidof(ID3D12Device), reinterpret_cast<void**>(&d3d12_device_));
        adapter->Release();
        if (FAILED(hr) || d3d12_device_ == nullptr) {
            CriticalLog(g_rocdec_logger, "D3D12CreateDevice failed");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    uint32_t num_surfaces = decoder_create_info_.num_decode_surfaces;
    // Release any previously created shared resources (reconfigure path).
    for (auto* res : d3d12_shared_resources_) {
        if (res) res->Release();
    }
    d3d12_shared_resources_.resize(num_surfaces, nullptr);

    D3D12_RESOURCE_DESC res_desc = {};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    res_desc.Width = decoder_create_info_.width;
    res_desc.Height = decoder_create_info_.height;
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

    // Try linear layout first; fall back to driver-default layout.
    bool linear_layout = true;
    {
        HRESULT hr = d3d12_device_->CreateCommittedResource(
            &heap_props, heap_flags,
            &res_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d3d12_shared_resources_[0]));
        if (FAILED(hr)) {
            InfoLog(g_rocdec_logger, "ROW_MAJOR layout not supported (HRESULT=0x" +
                    ([](HRESULT h) { std::ostringstream o; o << std::hex << static_cast<uint32_t>(h); return o.str(); })(hr) +
                    "), falling back to LAYOUT_UNKNOWN (tiled)");
            linear_layout = false;
            res_desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
            res_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            heap_flags = D3D12_HEAP_FLAG_SHARED;
        } else {
            InfoLog(g_rocdec_logger, "Using D3D12_TEXTURE_LAYOUT_ROW_MAJOR (linear) for decode surfaces");
        }
    }

    for (uint32_t i = (linear_layout ? 1 : 0); i < num_surfaces; i++) {
        HRESULT hr = d3d12_device_->CreateCommittedResource(
            &heap_props, heap_flags,
            &res_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&d3d12_shared_resources_[i]));
        if (FAILED(hr) || d3d12_shared_resources_[i] == nullptr) {
            CriticalLog(g_rocdec_logger, "Failed to create shared D3D12 resource " + ROCDEC_TOSTR(i) +
                        ", HRESULT=0x" + ([](HRESULT h) { std::ostringstream o; o << std::hex << static_cast<uint32_t>(h); return o.str(); })(hr));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
    }

    // Tell VA-API to use our pre-created shared D3D12 resources as external surfaces.
    VASurfaceAttrib ext_attrib;
    ext_attrib.type = VASurfaceAttribMemoryType;
    ext_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    ext_attrib.value.type = VAGenericValueTypeInteger;
    ext_attrib.value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_D3D12_RESOURCE;
    surf_attribs.push_back(ext_attrib);

    VASurfaceAttrib ext_buf_attrib;
    ext_buf_attrib.type = VASurfaceAttribExternalBufferDescriptor;
    ext_buf_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    ext_buf_attrib.value.type = VAGenericValueTypePointer;
    ext_buf_attrib.value.value.p = d3d12_shared_resources_.data();
    surf_attribs.push_back(ext_buf_attrib);

    CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, decoder_create_info_.width,
        decoder_create_info_.height, va_surface_ids_.data(), static_cast<int>(va_surface_ids_.size()), surf_attribs.data(), static_cast<int>(surf_attribs.size())));

    // Log the actual D3D12 resource properties to confirm tiling mode.
    {
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
                " Flags=0x" + ([](UINT f) { std::ostringstream o; o << std::hex << f; return o.str(); })(desc.Flags) +
                " AllocSize=" + ROCDEC_TOSTR(alloc_info.SizeInBytes) +
                " Alignment=" + ROCDEC_TOSTR(alloc_info.Alignment));
    }

    // Create linear staging buffers for tiled→linear copy, and D3D12 copy infrastructure.
    if (!linear_layout) {
        // Size the staging buffer from GetSurfaceLayout (matches sample layer expectations).
        SurfaceLayout layout = GetSurfaceLayout();
        UINT64 staging_size = layout.total_size;

        for (auto* buf : d3d12_staging_buffers_) { if (buf) buf->Release(); }
        d3d12_staging_buffers_.resize(num_surfaces, nullptr);

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
                            ", HRESULT=0x" + ([](HRESULT h) { std::ostringstream o; o << std::hex << static_cast<uint32_t>(h); return o.str(); })(hr));
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
            d3d12_device_->CreateCommandQueue(&qd, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&d3d12_copy_queue_));
            d3d12_device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&d3d12_cmd_allocator_));
            d3d12_device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, d3d12_cmd_allocator_, nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&d3d12_cmd_list_));
            d3d12_cmd_list_->Close();
            d3d12_device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&d3d12_fence_));
            d3d12_fence_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            d3d12_fence_value_ = 0;
        }
        InfoLog(g_rocdec_logger, "Created D3D12 staging buffers (" + ROCDEC_TOSTR(staging_size) + " bytes each) for tiled→linear copy");
    }
#else
    uint64_t mod_linear = 0;
    VADRMFormatModifierList modifier_list = {
        .num_modifiers = 1,
        .modifiers = &mod_linear,
    };
    if (supports_modifiers_) {
        surf_attrib.type = VASurfaceAttribDRMFormatModifiers;
        surf_attrib.value.type = VAGenericValueTypePointer;
        surf_attrib.value.value.p = &modifier_list;
        surf_attribs.push_back(surf_attrib);
    }
    CHECK_VAAPI(vaCreateSurfaces(va_display_, surface_format, decoder_create_info_.width,
        decoder_create_info_.height, va_surface_ids_.data(), static_cast<int>(va_surface_ids_.size()), surf_attribs.data(), static_cast<int>(surf_attribs.size())));
#endif
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::CreateContext() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    CHECK_VAAPI(vaCreateContext(va_display_, va_config_id_, decoder_create_info_.width, decoder_create_info_.height,
        VA_PROGRESSIVE, va_surface_ids_.data(), static_cast<int>(va_surface_ids_.size()), &va_context_id_));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::DestroyDataBuffers() {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    if (pic_params_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, pic_params_buf_id_));
        pic_params_buf_id_ = 0;
    }
    if (iq_matrix_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, iq_matrix_buf_id_));
        iq_matrix_buf_id_ = 0;
    }
    for (uint32_t i = 0; i < num_slices_; i++) {
        if (slice_params_buf_id_[i]) {
            CHECK_VAAPI(vaDestroyBuffer(va_display_, slice_params_buf_id_[i]));
            slice_params_buf_id_[i] = 0;
        }
    }
    if (slice_data_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, slice_data_buf_id_));
        slice_data_buf_id_ = 0;
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

VaContext::VaContext() {
#ifndef _WIN32
    GetGpuUuids();
#endif
}

VaContext::~VaContext() {
    for (int i = 0; i < va_contexts_.size(); i++) {
#ifndef _WIN32
        if (va_contexts_[i].drm_fd != -1) {
            close(va_contexts_[i].drm_fd);
        }
#endif
        if (va_contexts_[i].va_display) {
            if (vaTerminate(va_contexts_[i].va_display) != VA_STATUS_SUCCESS) {
                CriticalLog(g_rocdec_logger, "Failed to terminate VA");
            }
        }
    }
};

rocDecStatus VaContext::GetVaContext(int device_id, uint32_t *va_ctx_id) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(device_id) + ", " + RocDecFmtPtr(va_ctx_id));
    std::lock_guard<std::mutex> lock(mutex);
    bool found_existing = false;
    uint32_t va_ctx_idx = 0;
    hipDeviceProp_t hip_dev_prop;
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = InitHIP(device_id, hip_dev_prop);
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to initialize the HIP.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }
    std::string gpu_uuid(hip_dev_prop.uuid.bytes, sizeof(hip_dev_prop.uuid.bytes));

    // Match by PCI BDF (consistent between HIP and sysfs), with unique_id as fallback.
    char pci_bus_id_buf[64] = {0};
    std::string gpu_pci_bdf;
    if (hipDeviceGetPCIBusId(pci_bus_id_buf, sizeof(pci_bus_id_buf), device_id) == hipSuccess) {
        gpu_pci_bdf = pci_bus_id_buf;
        std::transform(gpu_pci_bdf.begin(), gpu_pci_bdf.end(), gpu_pci_bdf.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        // Drop the PCI function suffix so partition children match their base BDF.
        size_t dot_pos = gpu_pci_bdf.find_last_of('.');
        if (dot_pos != std::string::npos) {
            gpu_pci_bdf = gpu_pci_bdf.substr(0, dot_pos);
        }
    }

    if (!va_contexts_.empty()) {
        for (va_ctx_idx = 0; va_ctx_idx < va_contexts_.size(); va_ctx_idx++) {
            if ((!gpu_pci_bdf.empty() && gpu_pci_bdf == va_contexts_[va_ctx_idx].gpu_pci_bdf) ||
                gpu_uuid.compare(va_contexts_[va_ctx_idx].gpu_uuid) == 0) {
                found_existing = true;
                break;
            }
        }
    }
    if (found_existing) {
        *va_ctx_id = va_ctx_idx;
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_SUCCESS;
    } else {
        va_contexts_.resize(va_contexts_.size() + 1);
        va_ctx_idx = static_cast<uint32_t>(va_contexts_.size() - 1);

        va_contexts_[va_ctx_idx].device_id = device_id;
        va_contexts_[va_ctx_idx].gpu_uuid.assign(gpu_uuid);
        va_contexts_[va_ctx_idx].gpu_pci_bdf = gpu_pci_bdf;
        va_contexts_[va_ctx_idx].hip_dev_prop = hip_dev_prop;
#ifdef _WIN32
        memcpy(&va_contexts_[va_ctx_idx].adapter_luid, hip_dev_prop.luid, sizeof(LUID));
#else
        va_contexts_[va_ctx_idx].drm_fd = -1;
#endif
        va_contexts_[va_ctx_idx].va_display = 0;
        va_contexts_[va_ctx_idx].num_dec_engines = 1;
        va_contexts_[va_ctx_idx].va_profile = VAProfileNone;
        va_contexts_[va_ctx_idx].config_attributes_probed = false;

#ifdef _WIN32
        rocdec_status = InitVAAPI(va_ctx_idx, &va_contexts_[va_ctx_idx].adapter_luid);
        if (rocdec_status != ROCDEC_SUCCESS) {
            CriticalLog(g_rocdec_logger, "Failed to initialize the VAAPI via vaon12.");
            va_contexts_.pop_back();
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }
#else
        std::vector<int> visible_devices;
        GetVisibleDevices(visible_devices);

        int offset = 0;
        ComputePartition current_compute_partition = kSpx;
        {
            auto bdf_it = gpu_pci_bdf_to_compute_partition_map_.find(gpu_pci_bdf);
            if (bdf_it != gpu_pci_bdf_to_compute_partition_map_.end()) {
                current_compute_partition = bdf_it->second;
            } else {
                auto uuid_it = gpu_uuids_to_compute_partition_map_.find(gpu_uuid);
                if (uuid_it != gpu_uuids_to_compute_partition_map_.end()) {
                    current_compute_partition = uuid_it->second;
                }
            }
        }
        GetDrmNodeOffset(va_contexts_[va_ctx_idx].hip_dev_prop.name, va_contexts_[va_ctx_idx].device_id, visible_devices, current_compute_partition, offset);

        int render_node_id = -1;
        {
            auto bdf_it = gpu_pci_bdf_to_render_nodes_map_.find(gpu_pci_bdf);
            if (bdf_it != gpu_pci_bdf_to_render_nodes_map_.end()) {
                render_node_id = bdf_it->second;
            } else {
                auto uuid_it = gpu_uuids_to_render_nodes_map_.find(gpu_uuid);
                if (uuid_it != gpu_uuids_to_render_nodes_map_.end()) {
                    render_node_id = uuid_it->second;
                }
            }
        }

        std::string drm_node;
        if (render_node_id >= 0) {
            drm_node = "/dev/dri/renderD" + std::to_string(render_node_id + offset);
        } else {
            drm_node = GetFirstAvailableDrmNode();
            if (drm_node.empty()) {
                drm_node = "/dev/dri/renderD128";
            }
        }

        if (g_rocdec_logger.GetLogLevel() >= kRocDecLogInfo) {
            InfoLog(g_rocdec_logger, "gpu_uuids_to_render_nodes_map_:");
            for (const auto& entry : gpu_uuids_to_render_nodes_map_) {
                InfoLog(g_rocdec_logger, "  " + entry.first + " -> renderD" + std::to_string(entry.second));
            }
            InfoLog(g_rocdec_logger, "gpu_pci_bdf_to_render_nodes_map_:");
            for (const auto& entry : gpu_pci_bdf_to_render_nodes_map_) {
                InfoLog(g_rocdec_logger, "  " + entry.first + " -> renderD" + std::to_string(entry.second));
            }

            auto partition_name = [](ComputePartition p) -> const char* {
                switch (p) {
                    case kSpx: return "SPX";
                    case kDpx: return "DPX";
                    case kTpx: return "TPX";
                    case kQpx: return "QPX";
                    case kCpx: return "CPX";
                    default:   return "unknown";
                }
            };
            InfoLog(g_rocdec_logger, "Selected GPU UUID: " + gpu_uuid);
            InfoLog(g_rocdec_logger, "Selected GPU BDF: " + gpu_pci_bdf);
            InfoLog(g_rocdec_logger, "Selected compute partition: " + std::string(partition_name(current_compute_partition)));
            InfoLog(g_rocdec_logger, "Selected DRM node: " + drm_node);
        }

        rocdec_status = InitVAAPI(va_ctx_idx, drm_node);
        if (rocdec_status != ROCDEC_SUCCESS) {
            CriticalLog(g_rocdec_logger, "Failed to initialize the VAAPI.");
            FunctionExitLog(g_rocdec_logger);
            return rocdec_status;
        }

        amdgpu_device_handle dev_handle;
        uint32_t major_version = 0, minor_version = 0;
        if (amdgpu_device_initialize(va_contexts_[va_ctx_idx].drm_fd, &major_version, &minor_version, &dev_handle)) {
            CriticalLog(g_rocdec_logger, "GPU device initialization failed: " + drm_node);
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_DEVICE_INVALID;
        }
        if (amdgpu_query_hw_ip_count(dev_handle, AMDGPU_HW_IP_VCN_DEC, &va_contexts_[va_ctx_idx].num_dec_engines)) {
            CriticalLog(g_rocdec_logger, "Failed to get the number of video decode engines.");
        }
        amdgpu_device_deinitialize(dev_handle);
#endif

        // Probe VA profiles
        va_contexts_[va_ctx_idx].num_va_profiles = vaMaxNumProfiles(va_contexts_[va_ctx_idx].va_display);
        va_contexts_[va_ctx_idx].va_profile_list.resize(va_contexts_[va_ctx_idx].num_va_profiles);
        CHECK_VAAPI(vaQueryConfigProfiles(va_contexts_[va_ctx_idx].va_display, va_contexts_[va_ctx_idx].va_profile_list.data(), &va_contexts_[va_ctx_idx].num_va_profiles));

        *va_ctx_id = va_ctx_idx;
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_SUCCESS;
    }
}

rocDecStatus VaContext::GetVaDisplay(uint32_t va_ctx_id, VADisplay *va_display) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(va_ctx_id) + ", " + RocDecFmtPtr(va_display));
    if (va_ctx_id >= va_contexts_.size()) {
        CriticalLog(g_rocdec_logger, "Invalid VA context Id.");
        *va_display = 0;
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    } else {
#ifdef _WIN32
        VADisplay new_va_display = vaGetDisplayWin32(&va_contexts_[va_ctx_id].adapter_luid);
#else
        VADisplay new_va_display = vaGetDisplayDRM(va_contexts_[va_ctx_id].drm_fd);
#endif
        if (!new_va_display) {
            CriticalLog(g_rocdec_logger, "Failed to create VA display.");
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_NOT_INITIALIZED;
        }
        std::string va_driver_path;
        vaSetInfoCallback(new_va_display, [](void* user_context, const char* message) {
            std::string msg(message);
            if (msg.find("Trying to open") != std::string::npos) {
                *static_cast<std::string*>(user_context) = msg;
            }
        }, &va_driver_path);
        int major_version = 0, minor_version = 0;
        VAStatus va_status = vaInitialize(new_va_display, &major_version, &minor_version);
        vaSetInfoCallback(new_va_display, nullptr, nullptr);
        if (va_status != VA_STATUS_SUCCESS) {
            CriticalLog(g_rocdec_logger, std::string("vaInitialize failed: ") + vaErrorStr(va_status));
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_RUNTIME_ERROR;
        }
        InfoLog(g_rocdec_logger, "VA-API version " + std::to_string(major_version) + "." + std::to_string(minor_version));
        const char* vendor_str = vaQueryVendorString(new_va_display);
        InfoLog(g_rocdec_logger, "VA-API vendor: " + std::string(vendor_str ? vendor_str : "<unknown>"));
        if (!va_driver_path.empty()) {
            InfoLog(g_rocdec_logger, va_driver_path);
        }
        *va_display = new_va_display;
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_SUCCESS;
    }
}

rocDecStatus VaContext::CheckDecCapForCodecType(RocdecDecodeCaps *dec_cap) {
    FunctionEntryLogWithArgs(g_rocdec_logger, RocDecFmtPtr(dec_cap));
    if (dec_cap == nullptr) {
        CriticalLog(g_rocdec_logger, "Null decode capability struct pointer.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    uint32_t va_ctx_id;
    rocdec_status = GetVaContext(dec_cap->device_id, &va_ctx_id);
    if (rocdec_status != ROCDEC_SUCCESS) {
        CriticalLog(g_rocdec_logger, "Failed to initialize.");
        FunctionExitLog(g_rocdec_logger);
        return rocdec_status;
    }

    std::lock_guard<std::mutex> lock(mutex);
    dec_cap->is_supported = 1; // init value
    VAProfile va_profile = VAProfileNone;
    switch (dec_cap->codec_type) {
        case rocDecVideoCodec_HEVC: {
            if (dec_cap->bit_depth_minus_8 == 0) {
                va_profile = VAProfileHEVCMain;
            } else if (dec_cap->bit_depth_minus_8 == 2) {
                va_profile = VAProfileHEVCMain10;
            }
            break;
        }
        case rocDecVideoCodec_AVC: {
            va_profile = VAProfileH264Main;
            break;
        }
        case rocDecVideoCodec_VP9: {
            if (dec_cap->bit_depth_minus_8 == 0) {
                va_profile = VAProfileVP9Profile0;
            } else if (dec_cap->bit_depth_minus_8 == 2) {
                va_profile = VAProfileVP9Profile2;
            }
            break;
        }
        case rocDecVideoCodec_AV1: {
#if VA_CHECK_VERSION(1, 23, 0)
            if (dec_cap->bit_depth_minus_8 == 4) {
                va_profile = VAProfileAV1Profile2;
            } else
#endif
            {
                va_profile = VAProfileAV1Profile0;
            }
            break;
        }
        default: {
            dec_cap->is_supported = 0;
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_SUCCESS;
        }
    }

    int i;
    for (i = 0; i < va_contexts_[va_ctx_id].num_va_profiles; i++) {
        if (va_contexts_[va_ctx_id].va_profile_list[i] == va_profile) {
            break;
        }
    }
    if (i == va_contexts_[va_ctx_id].num_va_profiles) {
        dec_cap->is_supported = 0;
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_SUCCESS;
    }

    // Check if the config attributes of the profile have been probed before
    if (va_profile != va_contexts_[va_ctx_id].va_profile || va_contexts_[va_ctx_id].config_attributes_probed == false) {
        va_contexts_[va_ctx_id].va_profile = va_profile;

        VAConfigAttrib va_config_attrib;
        unsigned int attr_count;
        std::vector<VASurfaceAttrib> attr_list;
        va_config_attrib.type = VAConfigAttribRTFormat;
        CHECK_VAAPI(vaGetConfigAttributes(va_contexts_[va_ctx_id].va_display, va_contexts_[va_ctx_id].va_profile, VAEntrypointVLD, &va_config_attrib, 1));
        va_contexts_[va_ctx_id].rt_format_attrib = va_config_attrib.value;

        CHECK_VAAPI(vaCreateConfig(va_contexts_[va_ctx_id].va_display, va_contexts_[va_ctx_id].va_profile, VAEntrypointVLD, &va_config_attrib, 1, &va_contexts_[va_ctx_id].va_config_id));
        CHECK_VAAPI(vaQuerySurfaceAttributes(va_contexts_[va_ctx_id].va_display, va_contexts_[va_ctx_id].va_config_id, 0, &attr_count));
        attr_list.resize(attr_count);
        CHECK_VAAPI(vaQuerySurfaceAttributes(va_contexts_[va_ctx_id].va_display, va_contexts_[va_ctx_id].va_config_id, attr_list.data(), &attr_count));
        va_contexts_[va_ctx_id].output_format_mask = 0;
        CHECK_VAAPI(vaDestroyConfig(va_contexts_[va_ctx_id].va_display, va_contexts_[va_ctx_id].va_config_id));
        for (unsigned int k = 0; k < attr_count; k++) {
            switch (attr_list[k].type) {
            case VASurfaceAttribPixelFormat: {
                switch (attr_list[k].value.value.i) {
                    case VA_FOURCC_NV12:
                        va_contexts_[va_ctx_id].output_format_mask |= 1 << rocDecVideoSurfaceFormat_NV12;
                        break;
                    case VA_FOURCC_P016:
                    case VA_FOURCC_P012:
                    case VA_FOURCC_P010:
                        va_contexts_[va_ctx_id].output_format_mask |= 1 << rocDecVideoSurfaceFormat_P016;
                        break;
                    default:
                        break;
                }
            }
                break;
            case VASurfaceAttribMinWidth:
                va_contexts_[va_ctx_id].min_width = attr_list[k].value.value.i;
                break;
            case VASurfaceAttribMinHeight:
                va_contexts_[va_ctx_id].min_height = attr_list[k].value.value.i;
                break;
            case VASurfaceAttribMaxWidth:
                va_contexts_[va_ctx_id].max_width = attr_list[k].value.value.i;
                break;
            case VASurfaceAttribMaxHeight:
                va_contexts_[va_ctx_id].max_height = attr_list[k].value.value.i;
                break;
            default:
                break;
            }
        }
        va_contexts_[va_ctx_id].config_attributes_probed = true;
    }

    // Check chroma format
    switch (dec_cap->chroma_format) {
        case rocDecVideoChromaFormat_Monochrome: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & VA_RT_FORMAT_YUV400) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_420: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV420_12)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_422: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV422 | VA_RT_FORMAT_YUV422_10 | VA_RT_FORMAT_YUV422_12)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_444: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        default: {
            dec_cap->is_supported = 0;
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_SUCCESS;
        }
    }
    // Check bit depth
    switch (dec_cap->bit_depth_minus_8) {
        case 0: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV422 | VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV400)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case 2: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV422_10 | VA_RT_FORMAT_YUV444_10)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case 4: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420_12 | VA_RT_FORMAT_YUV422_12 | VA_RT_FORMAT_YUV444_12)) == 0) {
                dec_cap->is_supported = 0;
                FunctionExitLog(g_rocdec_logger);
                return ROCDEC_SUCCESS;
            }
            break;
        }
        default: {
            dec_cap->is_supported = 0;
            FunctionExitLog(g_rocdec_logger);
            return ROCDEC_SUCCESS;
        }
    }

    dec_cap->num_decoders = va_contexts_[va_ctx_id].num_dec_engines;
    dec_cap->output_format_mask = va_contexts_[va_ctx_id].output_format_mask;
    dec_cap->max_width = va_contexts_[va_ctx_id].max_width;
    dec_cap->max_height = va_contexts_[va_ctx_id].max_height;
    dec_cap->min_width = va_contexts_[va_ctx_id].min_width;
    dec_cap->min_height = va_contexts_[va_ctx_id].min_height;
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

rocDecStatus VaContext::InitHIP(int device_id, hipDeviceProp_t& hip_dev_prop) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(device_id));
    CHECK_HIP(hipGetDeviceCount(&num_devices_));
    if (num_devices_ < 1) {
        CriticalLog(g_rocdec_logger, "Didn't find any GPU.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_DEVICE_INVALID;
    }
    if (device_id >= num_devices_) {
        CriticalLog(g_rocdec_logger, "ERROR: the requested device_id is not found!");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_DEVICE_INVALID;
    }
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}

#ifdef _WIN32
rocDecStatus VaContext::InitVAAPI(int va_ctx_idx, const LUID* adapter_luid) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(va_ctx_idx));
    InfoLog(g_rocdec_logger, "Initializing VA-API via vaon12 (LUID: " +
            ROCDEC_TOSTR(adapter_luid->HighPart) + ":" + ROCDEC_TOSTR(adapter_luid->LowPart) + ")");

    // Auto-set LIBVA_DRIVERS_PATH to the directory containing rocdecode.dll so that
    // libva can find vaon12_drv_video.dll (which is deployed alongside rocdecode.dll).
    if (std::getenv("LIBVA_DRIVERS_PATH") == nullptr) {
        HMODULE hm = nullptr;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(&VaContext::GetInstance), &hm)) {
            char module_path[MAX_PATH] = {};
            if (GetModuleFileNameA(hm, module_path, MAX_PATH) > 0) {
                std::string dir(module_path);
                auto pos = dir.find_last_of("\\/");
                if (pos != std::string::npos) {
                    dir = dir.substr(0, pos);
                    _putenv_s("LIBVA_DRIVERS_PATH", dir.c_str());
                    InfoLog(g_rocdec_logger, "Auto-set LIBVA_DRIVERS_PATH=" + dir);
                }
            }
        }
    }

    va_contexts_[va_ctx_idx].va_display = vaGetDisplayWin32(adapter_luid);
    if (!va_contexts_[va_ctx_idx].va_display) {
        CriticalLog(g_rocdec_logger, "Failed to create VA display via vaGetDisplayWin32.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    std::string va_driver_path;
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, [](void* user_context, const char* message) {
        std::string msg(message);
        if (msg.find("Trying to open") != std::string::npos) {
            *static_cast<std::string*>(user_context) = msg;
        }
    }, &va_driver_path);
    int major_version = 0, minor_version = 0;
    VAStatus va_status = vaInitialize(va_contexts_[va_ctx_idx].va_display, &major_version, &minor_version);
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, nullptr, nullptr);
    if (va_status != VA_STATUS_SUCCESS) {
        CriticalLog(g_rocdec_logger, std::string("vaInitialize failed: ") + vaErrorStr(va_status));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    InfoLog(g_rocdec_logger, "VA-API version " + std::to_string(major_version) + "." + std::to_string(minor_version));
    const char* vendor_str = vaQueryVendorString(va_contexts_[va_ctx_idx].va_display);
    InfoLog(g_rocdec_logger, "VA-API vendor: " + std::string(vendor_str ? vendor_str : "<unknown>"));
    if (!va_driver_path.empty()) {
        InfoLog(g_rocdec_logger, va_driver_path);
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
#else
rocDecStatus VaContext::InitVAAPI(int va_ctx_idx, std::string drm_node) {
    FunctionEntryLogWithArgs(g_rocdec_logger, ROCDEC_TOSTR(va_ctx_idx) + ", " + drm_node);
    InfoLog(g_rocdec_logger, "Opening DRM node: " + drm_node);
    va_contexts_[va_ctx_idx].drm_fd = open(drm_node.c_str(), O_RDWR);
    if (va_contexts_[va_ctx_idx].drm_fd < 0) {
        CriticalLog(g_rocdec_logger, "Failed to open drm node: " + drm_node);
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    va_contexts_[va_ctx_idx].va_display = vaGetDisplayDRM(va_contexts_[va_ctx_idx].drm_fd);
    if (!va_contexts_[va_ctx_idx].va_display) {
        CriticalLog(g_rocdec_logger, "Failed to create VA display.");
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_NOT_INITIALIZED;
    }
    std::string va_driver_path;
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, [](void* user_context, const char* message) {
        std::string msg(message);
        if (msg.find("Trying to open") != std::string::npos) {
            *static_cast<std::string*>(user_context) = msg;
        }
    }, &va_driver_path);
    int major_version = 0, minor_version = 0;
    VAStatus va_status = vaInitialize(va_contexts_[va_ctx_idx].va_display, &major_version, &minor_version);
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, nullptr, nullptr);
    if (va_status != VA_STATUS_SUCCESS) {
        CriticalLog(g_rocdec_logger, std::string("vaInitialize failed: ") + vaErrorStr(va_status));
        FunctionExitLog(g_rocdec_logger);
        return ROCDEC_RUNTIME_ERROR;
    }
    InfoLog(g_rocdec_logger, "VA-API version " + std::to_string(major_version) + "." + std::to_string(minor_version));
    const char* vendor_str = vaQueryVendorString(va_contexts_[va_ctx_idx].va_display);
    InfoLog(g_rocdec_logger, "VA-API vendor: " + std::string(vendor_str ? vendor_str : "<unknown>"));
    if (!va_driver_path.empty()) {
        InfoLog(g_rocdec_logger, va_driver_path);
    }
    FunctionExitLog(g_rocdec_logger);
    return ROCDEC_SUCCESS;
}
#endif

#ifndef _WIN32
void VaContext::GetVisibleDevices(std::vector<int>& visible_devices_vetor) {
    FunctionEntryLogWithArgs(g_rocdec_logger, "");
    // First, check if the ROCR_VISIBLE_DEVICES environment variable is present
    char *visible_devices = std::getenv("ROCR_VISIBLE_DEVICES");
    // If ROCR_VISIBLE_DEVICES is not present, check if HIP_VISIBLE_DEVICES is present
    if (visible_devices == nullptr) {
        visible_devices = std::getenv("HIP_VISIBLE_DEVICES");
    }
    if (visible_devices != nullptr) {
        char *token = std::strtok(visible_devices,",");
        while (token != nullptr) {
            visible_devices_vetor.push_back(std::atoi(token));
            token = std::strtok(nullptr,",");
        }
        std::sort(visible_devices_vetor.begin(), visible_devices_vetor.end());
    }
    FunctionExitLog(g_rocdec_logger);
}

void VaContext::GetDrmNodeOffset(std::string device_name, uint8_t device_id, std::vector<int>& visible_devices, ComputePartition current_compute_partition, int &offset) {
    switch (current_compute_partition) {
        case kSpx:
            offset = 0;
            break;
        case kDpx:
            if (device_id < visible_devices.size()) {
                offset = (visible_devices[device_id] % 2);
            } else {
                offset = (device_id % 2);
            }
            break;
        case kTpx:
            if (device_id < visible_devices.size()) {
               offset = (visible_devices[device_id] % 3);
            } else {
                offset = (device_id % 3);
            }
            break;
        case kQpx:
            if (device_id < visible_devices.size()) {
                offset = (visible_devices[device_id] % 4);
            } else {
                offset = (device_id % 4);
            }
            break;
        case kCpx:
            // Note: The MI300 series share the same gfx_arch_name (gfx942).
            // Therefore, we cannot use gfx942 to distinguish between MI300X, MI300A etc.
            // Instead, use the device name to identify MI300A etc.
            std::string mi300a = "MI300A";
            size_t found_mi300a = device_name.find(mi300a);
            std::string mi308 = "MI308";
            size_t found_mi308 = device_name.find(mi308);
            if (found_mi300a != std::string::npos) {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 6);
                } else {
                    offset = (device_id % 6);
                }
            } else if (found_mi308 != std::string::npos) {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 4);
                } else {
                    offset = (device_id % 4);
                }
            } else {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 8);
                } else {
                    offset = (device_id % 8);
                }
            }
            break;
    }
}

/**
 * @brief Retrieves GPU UUIDs and maps them to render node IDs and compute partitions.
 *
 * This function iterates through all render nodes in the /dev/dri directory,
 * extracts the render node ID from the filename, and then reads the unique GPU
 * UUID from the corresponding sysfs path. It maps each unique GPU UUID to its
 * corresponding render node ID and stores this mapping in the gpu_uuids_to_render_nodes_map_.
 * Additionally, it maps the unique GPU UUID to the current compute partition if available.
 */
void VaContext::GetGpuUuids() {
    std::string dri_path = "/dev/dri";
    DIR* dir = opendir(dri_path.c_str());
    if (dir) {
        struct dirent* entry;
        // Iterate through all render nodes
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            // Check if the file name starts with "renderD"
            if (filename.find("renderD") == 0) {
                // Extract the integer part from the render node name (e.g., 128 from renderD128)
                int render_id = std::stoi(filename.substr(7));
                std::string sys_device_path = "/sys/class/drm/" + filename + "/device";
                struct stat info;
                if (stat(sys_device_path.c_str(), &info) == 0) {
                    std::string bus_id = GetRenderNodeBusId(filename);
                    if (!bus_id.empty()) {
                        gpu_pci_bdf_to_render_nodes_map_[bus_id] = render_id;
                    }
                    std::string unique_id_path = sys_device_path + "/unique_id";
                    std::ifstream unique_id_file(unique_id_path);
                    std::string unique_id;
                    if (unique_id_file.is_open() && std::getline(unique_id_file, unique_id)) {
                        if (!unique_id.empty()) {
                            // Map the unique GPU UUID to the render node ID
                            gpu_uuids_to_render_nodes_map_[unique_id] = render_id;
                        }
                    }
                    unique_id_file.close();
                    if (!unique_id.empty()) {
                        unique_id_path = sys_device_path + "/current_compute_partition";
                        std::ifstream partition_file(unique_id_path);
                        std::string partition;
                        ComputePartition current_compute_partition = kSpx;
                        if (partition_file.is_open() && std::getline(partition_file, partition)) {
                            if (!partition.empty()) {
                                if (partition.compare("SPX") == 0 || partition.compare("spx") == 0) {
                                    current_compute_partition = kSpx;
                                } else if (partition.compare("DPX") == 0 || partition.compare("dpx") == 0) {
                                    current_compute_partition = kDpx;
                                } else if (partition.compare("TPX") == 0 || partition.compare("tpx") == 0) {
                                    current_compute_partition = kTpx;
                                } else if (partition.compare("QPX") == 0 || partition.compare("qpx") == 0) {
                                    current_compute_partition = kQpx;
                                } else if (partition.compare("CPX") == 0 || partition.compare("cpx") == 0) {
                                    current_compute_partition = kCpx;
                                }
                                // Map the unique GPU UUID to the compute partition
                                gpu_uuids_to_compute_partition_map_[unique_id] = current_compute_partition;
                                if (!bus_id.empty()) {
                                    gpu_pci_bdf_to_compute_partition_map_[bus_id] = current_compute_partition;
                                }
                            }
                        }
                        partition_file.close();
                    }
                }
            }
        }
        closedir(dir);
    }
}
std::string VaContext::GetRenderNodeBusId(const std::string& render_node_name) {
    std::string device_link = "/sys/class/drm/" + render_node_name + "/device";
    char* resolved = realpath(device_link.c_str(), nullptr);
    if (resolved == nullptr) {
        return "";
    }
    std::string path(resolved);
    free(resolved);
    size_t pos = path.find_last_of('/');
    std::string bus_id = (pos == std::string::npos) ? path : path.substr(pos + 1);
    if (bus_id.find(':') == std::string::npos || bus_id.find('.') == std::string::npos) {
        return "";
    }
    std::transform(bus_id.begin(), bus_id.end(), bus_id.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    size_t dot_pos = bus_id.find_last_of('.');
    if (dot_pos != std::string::npos) {
        bus_id = bus_id.substr(0, dot_pos);
    }
    return bus_id;
}

std::string VaContext::GetFirstAvailableDrmNode() {
    std::string dri_path = "/dev/dri";
    DIR* dir = opendir(dri_path.c_str());
    if (!dir) {
        return "";
    }
    int min_render_id = -1;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename = entry->d_name;
        if (filename.find("renderD") == 0 && filename.size() > 7) {
            try {
                int render_id = std::stoi(filename.substr(7));
                if (min_render_id < 0 || render_id < min_render_id) {
                    min_render_id = render_id;
                }
            } catch (...) {
            }
        }
    }
    closedir(dir);
    if (min_render_id < 0) {
        return "";
    }
    return "/dev/dri/renderD" + std::to_string(min_render_id);
}
#endif // !_WIN32
