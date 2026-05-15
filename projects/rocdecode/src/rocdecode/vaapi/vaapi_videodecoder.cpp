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

VaapiVideoDecoder::VaapiVideoDecoder(RocDecoderCreateInfo &decoder_create_info) : decoder_create_info_{decoder_create_info},
    va_display_{0}, va_config_attrib_{{}}, va_config_id_{0}, va_profile_ {VAProfileNone}, va_context_id_{0}, va_surface_ids_{{}},
    supports_modifiers_{false}, pic_params_buf_id_{0}, iq_matrix_buf_id_{0}, num_slices_{0}, slice_data_buf_id_{0} {
};

VaapiVideoDecoder::~VaapiVideoDecoder() {
    if (va_display_) {
        rocDecStatus rocdec_status = ROCDEC_SUCCESS;
        rocdec_status = DestroyDataBuffers();
        if (rocdec_status != ROCDEC_SUCCESS) {
            logger_.CriticalLog(MakeMsg("DestroyDataBuffers failed"));
        }
        VAStatus va_status = VA_STATUS_SUCCESS;
        va_status = vaDestroySurfaces(va_display_, va_surface_ids_.data(), va_surface_ids_.size());
        if (va_status != VA_STATUS_SUCCESS) {
            logger_.CriticalLog(MakeMsg("vaDestroySurfaces failed"));
        }
        if (va_context_id_) {
            va_status = vaDestroyContext(va_display_, va_context_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                logger_.CriticalLog(MakeMsg("vaDestroyContext failed"));
            }
        }
        if (va_config_id_) {
            va_status = vaDestroyConfig(va_display_, va_config_id_);
            if (va_status != VA_STATUS_SUCCESS) {
                logger_.CriticalLog(MakeMsg("vaDestroyConfig failed"));
            }
        }
        if (vaTerminate(va_display_) != VA_STATUS_SUCCESS) {
            logger_.CriticalLog(MakeMsg("Failed to termiate VA"));
        }
    }
}

rocDecStatus VaapiVideoDecoder::InitializeDecoder() {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;

    // Before initializing the VAAPI, first check to see if the requested codec config is supported
    if (!IsCodecConfigSupported(decoder_create_info_.device_id, decoder_create_info_.codec_type, decoder_create_info_.chroma_format,
        decoder_create_info_.bit_depth_minus_8, decoder_create_info_.output_format)) {
        logger_.CriticalLog(MakeMsg("The codec config combination is not supported."));
        return ROCDEC_NOT_SUPPORTED;
    }

    VaContext& va_ctx = VaContext::GetInstance();
    uint32_t va_ctx_id;
    if ((rocdec_status = va_ctx.GetVaContext(decoder_create_info_.device_id, &va_ctx_id)) != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to get VA context."));
        return rocdec_status;
    }
    if ((rocdec_status = va_ctx.GetVaDisplay(va_ctx_id, &va_display_)) != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to get VA display."));
        return rocdec_status;
    }
    rocdec_status = CreateDecoderConfig();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to create a VAAPI decoder configuration."));
        return rocdec_status;
    }
    rocdec_status = CreateSurfaces();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to create VAAPI surfaces."));
        return rocdec_status;
    }
    rocdec_status = CreateContext();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to create a VAAPI context."));
        return rocdec_status;
    }
    return rocdec_status;
}

rocDecStatus VaapiVideoDecoder::SubmitDecode(RocdecPicParams *pPicParams) {
    void *pic_params_ptr, *iq_matrix_ptr, *slice_params_ptr;
    uint32_t pic_params_size, iq_matrix_size, slice_params_size;
    bool scaling_list_enabled = false;
    VASurfaceID curr_surface_id;

    // Get the surface id for the current picture, assuming 1:1 mapping between DPB and VAAPI decoded surfaces.
    if (pPicParams->curr_pic_idx >= va_surface_ids_.size() || pPicParams->curr_pic_idx < 0) {
        logger_.ErrorLog(MakeMsg("curr_pic_idx exceeded the VAAPI surface pool limit."));
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
                        logger_.ErrorLog(MakeMsg("Reference frame index exceeded the VAAPI surface pool limit."));
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
                    logger_.ErrorLog(MakeMsg("HEVC data_buffer parameter_size not matching vaapi parameter buffer size."));
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_AVC: {
            pPicParams->pic_params.avc.curr_pic.pic_idx = curr_surface_id;
            for (int i = 0; i < 16; i++) {
                if (pPicParams->pic_params.avc.ref_frames[i].pic_idx != 0xFF) {
                    if (pPicParams->pic_params.avc.ref_frames[i].pic_idx >= va_surface_ids_.size() || pPicParams->pic_params.avc.ref_frames[i].pic_idx < 0) {
                        logger_.ErrorLog(MakeMsg("Reference frame index exceeded the VAAPI surface pool limit."));
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
                    logger_.ErrorLog(MakeMsg("AVC data_buffer parameter_size not matching vaapi parameter buffer size."));
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_VP9: {
            for (int i = 0; i < 8; i++) {
                if (pPicParams->pic_params.vp9.reference_frames[i] != 0xFF) {
                    if (pPicParams->pic_params.vp9.reference_frames[i] >= va_surface_ids_.size()) {
                        logger_.ErrorLog(MakeMsg("Reference frame index exceeded the VAAPI surface pool limit."));
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
                    logger_.ErrorLog(MakeMsg("VP9 data_buffer parameter_size not matching vaapi parameter buffer size."));
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        case rocDecVideoCodec_AV1: {
            pPicParams->pic_params.av1.current_frame = curr_surface_id;

            if (pPicParams->pic_params.av1.current_display_picture != 0xFF) {
                if (pPicParams->pic_params.av1.current_display_picture >= va_surface_ids_.size() || pPicParams->pic_params.av1.current_display_picture < 0) {
                    logger_.ErrorLog(MakeMsg("Current display picture index exceeded the VAAPI surface pool limit."));
                    return ROCDEC_INVALID_PARAMETER;
                }
                pPicParams->pic_params.av1.current_display_picture = va_surface_ids_[pPicParams->pic_params.av1.current_display_picture];
            }

            for (int i = 0; i < pPicParams->pic_params.av1.anchor_frames_num; i++) {
                if (pPicParams->pic_params.av1.anchor_frames_list[i] >= va_surface_ids_.size() || pPicParams->pic_params.av1.anchor_frames_list[i] < 0) {
                    logger_.ErrorLog(MakeMsg("Anchor frame index exceeded the VAAPI surface pool limit."));
                    return ROCDEC_INVALID_PARAMETER;
                }
                pPicParams->pic_params.av1.anchor_frames_list[i] = va_surface_ids_[pPicParams->pic_params.av1.anchor_frames_list[i]];
            }

            for (int i = 0; i < 8; i++) {
                if (pPicParams->pic_params.av1.ref_frame_map[i] != 0xFF) {
                    if (pPicParams->pic_params.av1.ref_frame_map[i] >= va_surface_ids_.size() || pPicParams->pic_params.av1.ref_frame_map[i] < 0) {
                        logger_.ErrorLog(MakeMsg("Reference frame index exceeded the VAAPI surface pool limit."));
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
                    logger_.CriticalLog(MakeMsg("AV1 data_buffer parameter_size not matching vaapi parameter buffer size."));
                    return ROCDEC_RUNTIME_ERROR;
            }
            break;
        }

        default: {
            logger_.CriticalLog(MakeMsg("The codec type is not supported."));
            return ROCDEC_NOT_SUPPORTED;
        }
    }

    // Destroy the data buffers of the previous frame
    rocDecStatus rocdec_status = DestroyDataBuffers();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to destroy VAAPI buffer."));
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
    for (int i = 0; i < num_slices_; i++) {
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

    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::GetDecodeStatus(int pic_idx, RocdecDecodeStatus *decode_status) {
    VASurfaceStatus va_surface_status;
    if (pic_idx >= va_surface_ids_.size() || decode_status == nullptr) {
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
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::ExportSurface(int pic_idx, VADRMPRIMESurfaceDescriptor &va_drm_prime_surface_desc) {
    if (pic_idx >= va_surface_ids_.size()) {
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaExportSurfaceHandle(va_display_, va_surface_ids_[pic_idx],
                VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_ONLY |
                VA_EXPORT_SURFACE_SEPARATE_LAYERS,
                &va_drm_prime_surface_desc));

   return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::SyncSurface(int pic_idx) {
    if (pic_idx >= va_surface_ids_.size()) {
        return ROCDEC_INVALID_PARAMETER;
    }
    CHECK_VAAPI(vaSyncSurface(va_display_, va_surface_ids_[pic_idx]));
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    if (reconfig_params == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    if (va_display_ == 0) {
        logger_.CriticalLog(MakeMsg("VAAPI decoder has not been initialized but reconfiguration of the decoder has been requested."));
        return ROCDEC_NOT_SUPPORTED;
    }
    CHECK_VAAPI(vaDestroySurfaces(va_display_, va_surface_ids_.data(), va_surface_ids_.size()));
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

    rocDecStatus rocdec_status;
    if (create_va_config) {
        rocdec_status = CreateDecoderConfig();
        if (rocdec_status != ROCDEC_SUCCESS) {
            logger_.CriticalLog(MakeMsg("Failed to create a VAAPI decoder configuration."));
            return rocdec_status;
        }
    }
    rocdec_status = CreateSurfaces();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to create VAAPI surfaces during the decoder reconfiguration."));
        return rocdec_status;
    }
    rocdec_status = CreateContext();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to create a VAAPI context during the decoder reconfiguration."));
        return rocdec_status;
    }
    return rocdec_status;
}

bool VaapiVideoDecoder::IsCodecConfigSupported(int device_id, rocDecVideoCodec codec_type, rocDecVideoChromaFormat chroma_format, uint32_t bit_depth_minus8, rocDecVideoSurfaceFormat output_format) {
    RocdecDecodeCaps decode_caps;
    decode_caps.device_id = device_id;
    decode_caps.codec_type = codec_type;
    decode_caps.chroma_format = chroma_format;
    decode_caps.bit_depth_minus_8 = bit_depth_minus8;
    if((rocDecGetDecoderCaps(&decode_caps) != ROCDEC_SUCCESS) || (decode_caps.is_supported == false) || ((decode_caps.output_format_mask & (1 << output_format)) == 0)) {
        return false;
    } else {
        return true;
    }
}

rocDecStatus VaapiVideoDecoder::CreateDecoderConfig() {
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
            logger_.CriticalLog(MakeMsg("The codec type is not supported."));
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
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::CreateSurfaces() {
    if (decoder_create_info_.num_decode_surfaces < 1) {
        logger_.CriticalLog(MakeMsg("Invalid number of decode surfaces."));
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
        case rocDecVideoChromaFormat_Monochrome:
            surface_format = VA_RT_FORMAT_YUV400;
            surf_attrib.value.value.i = VA_FOURCC_Y800;
            break;
        case rocDecVideoChromaFormat_420:
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
            break;
        case rocDecVideoChromaFormat_422:
            surface_format = VA_RT_FORMAT_YUV422;
            break;
        case rocDecVideoChromaFormat_444:
            surface_format = VA_RT_FORMAT_YUV444;
            break;
        default:
            logger_.CriticalLog(MakeMsg("The surface type is not supported"));
            return ROCDEC_NOT_SUPPORTED;
    }
    surf_attribs.push_back(surf_attrib);
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
        decoder_create_info_.height, va_surface_ids_.data(), va_surface_ids_.size(), surf_attribs.data(), surf_attribs.size()));
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::CreateContext() {
    CHECK_VAAPI(vaCreateContext(va_display_, va_config_id_, decoder_create_info_.width, decoder_create_info_.height,
        VA_PROGRESSIVE, va_surface_ids_.data(), va_surface_ids_.size(), &va_context_id_));
    return ROCDEC_SUCCESS;
}

rocDecStatus VaapiVideoDecoder::DestroyDataBuffers() {
    if (pic_params_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, pic_params_buf_id_));
        pic_params_buf_id_ = 0;
    }
    if (iq_matrix_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, iq_matrix_buf_id_));
        iq_matrix_buf_id_ = 0;
    }
    for (int i = 0; i < num_slices_; i++) {
        if (slice_params_buf_id_[i]) {
            CHECK_VAAPI(vaDestroyBuffer(va_display_, slice_params_buf_id_[i]));
            slice_params_buf_id_[i] = 0;
        }
    }
    if (slice_data_buf_id_) {
        CHECK_VAAPI(vaDestroyBuffer(va_display_, slice_data_buf_id_));
        slice_data_buf_id_ = 0;
    }
    return ROCDEC_SUCCESS;
}

VaContext::VaContext() {
    GetGpuUuids();
}

VaContext::~VaContext() {
    for (int i = 0; i < va_contexts_.size(); i++) {
        if (va_contexts_[i].drm_fd != -1) {
            close(va_contexts_[i].drm_fd);
        }
        if (va_contexts_[i].va_display) {
            if (vaTerminate(va_contexts_[i].va_display) != VA_STATUS_SUCCESS) {
                logger_.CriticalLog(MakeMsg("Failed to termiate VA"));
            }
        }
    }
};

rocDecStatus VaContext::GetVaContext(int device_id, uint32_t *va_ctx_id) {
    std::lock_guard<std::mutex> lock(mutex);
    bool found_existing = false;
    uint32_t va_ctx_idx = 0;
    hipDeviceProp_t hip_dev_prop;
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = InitHIP(device_id, hip_dev_prop);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to initilize the HIP."));
        return rocdec_status;
    }
    std::string gpu_uuid(hip_dev_prop.uuid.bytes, sizeof(hip_dev_prop.uuid.bytes));

    if (!va_contexts_.empty()) {
        for (va_ctx_idx = 0; va_ctx_idx < va_contexts_.size(); va_ctx_idx++) {
            if (gpu_uuid.compare(va_contexts_[va_ctx_idx].gpu_uuid) == 0) {
                found_existing = true;
                break;
            }
        }
    }
    if (found_existing) {
        *va_ctx_id = va_ctx_idx;
        return ROCDEC_SUCCESS;
    } else {
        va_contexts_.resize(va_contexts_.size() + 1);
        va_ctx_idx = va_contexts_.size() - 1;

        va_contexts_[va_ctx_idx].device_id = device_id;
        va_contexts_[va_ctx_idx].gpu_uuid.assign(gpu_uuid);
        va_contexts_[va_ctx_idx].hip_dev_prop = hip_dev_prop;
        va_contexts_[va_ctx_idx].drm_fd = -1;
        va_contexts_[va_ctx_idx].va_display = 0;
        va_contexts_[va_ctx_idx].num_dec_engines = 1;
        va_contexts_[va_ctx_idx].va_profile = VAProfileNone;
        va_contexts_[va_ctx_idx].config_attributes_probed = false;

        std::vector<int> visible_devices;
        GetVisibleDevices(visible_devices);

        int offset = 0;
        ComputePartition current_compute_partition = (gpu_uuids_to_compute_partition_map_.find(gpu_uuid) != gpu_uuids_to_compute_partition_map_.end()) ? gpu_uuids_to_compute_partition_map_[gpu_uuid] : kSpx;
        GetDrmNodeOffset(va_contexts_[va_ctx_idx].hip_dev_prop.name, va_contexts_[va_ctx_idx].device_id, visible_devices, current_compute_partition, offset);

        std::string drm_node = "/dev/dri/renderD";
        int render_node_id = (gpu_uuids_to_render_nodes_map_.find(gpu_uuid) != gpu_uuids_to_render_nodes_map_.end()) ? gpu_uuids_to_render_nodes_map_[gpu_uuid] : 128;
        drm_node += std::to_string(render_node_id + offset);
        rocdec_status = InitVAAPI(va_ctx_idx, drm_node);
        if (rocdec_status != ROCDEC_SUCCESS) {
            logger_.CriticalLog(MakeMsg("Failed to initilize the VAAPI."));
            return rocdec_status;
        }

        amdgpu_device_handle dev_handle;
        uint32_t major_version = 0, minor_version = 0;
        if (amdgpu_device_initialize(va_contexts_[va_ctx_idx].drm_fd, &major_version, &minor_version, &dev_handle)) {
            logger_.CriticalLog(MakeMsg("GPU device initialization failed: " + drm_node));
            return ROCDEC_DEVICE_INVALID;
        }
        if (amdgpu_query_hw_ip_count(dev_handle, AMDGPU_HW_IP_VCN_DEC, &va_contexts_[va_ctx_idx].num_dec_engines)) {
            logger_.CriticalLog(MakeMsg("Failed to get the number of video decode engines."));
        }
        amdgpu_device_deinitialize(dev_handle);

        // Prob VA profiles
        va_contexts_[va_ctx_idx].num_va_profiles = vaMaxNumProfiles(va_contexts_[va_ctx_idx].va_display);
        va_contexts_[va_ctx_idx].va_profile_list.resize(va_contexts_[va_ctx_idx].num_va_profiles);
        CHECK_VAAPI(vaQueryConfigProfiles(va_contexts_[va_ctx_idx].va_display, va_contexts_[va_ctx_idx].va_profile_list.data(), &va_contexts_[va_ctx_idx].num_va_profiles));

        *va_ctx_id = va_ctx_idx;
        return ROCDEC_SUCCESS;
    }
}

rocDecStatus VaContext::GetVaDisplay(uint32_t va_ctx_id, VADisplay *va_display) {
    if (va_ctx_id >= va_contexts_.size()) {
        logger_.CriticalLog(MakeMsg("Invalid VA context Id."));
        *va_display = 0;
        return ROCDEC_INVALID_PARAMETER;
    } else {
        VADisplay new_va_display = vaGetDisplayDRM(va_contexts_[va_ctx_id].drm_fd);
        if (!new_va_display) {
            logger_.CriticalLog(MakeMsg("Failed to create VA display."));
            return ROCDEC_NOT_INITIALIZED;
        }
        vaSetInfoCallback(new_va_display, NULL, NULL);
        int major_version = 0, minor_version = 0;
        CHECK_VAAPI(vaInitialize(new_va_display, &major_version, &minor_version));
        *va_display = new_va_display;
        return ROCDEC_SUCCESS;
    }
}

rocDecStatus VaContext::CheckDecCapForCodecType(RocdecDecodeCaps *dec_cap) {
    if (dec_cap == nullptr) {
        logger_.CriticalLog(MakeMsg("Null decode capability struct pointer."));
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    uint32_t va_ctx_id;
    rocdec_status = GetVaContext(dec_cap->device_id, &va_ctx_id);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to initilize."));
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
        for (int k = 0; k < attr_count; k++) {
            switch (attr_list[k].type) {
            case VASurfaceAttribPixelFormat: {
                switch (attr_list[k].value.value.i) {
                    case VA_FOURCC_NV12:
                        va_contexts_[va_ctx_id].output_format_mask |= 1 << rocDecVideoSurfaceFormat_NV12;
                        break;
                    case VA_FOURCC_P016:
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
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_420: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV420_12)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_422: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV422 | VA_RT_FORMAT_YUV422_10 | VA_RT_FORMAT_YUV422_12)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case rocDecVideoChromaFormat_444: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV444_10 | VA_RT_FORMAT_YUV444_12)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        default: {
            dec_cap->is_supported = 0;
            return ROCDEC_SUCCESS;
        }
    }
    // Check bit depth
    switch (dec_cap->bit_depth_minus_8) {
        case 0: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420 | VA_RT_FORMAT_YUV422 | VA_RT_FORMAT_YUV444 | VA_RT_FORMAT_YUV400)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case 2: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420_10 | VA_RT_FORMAT_YUV422_10 | VA_RT_FORMAT_YUV444_10)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        case 4: {
            if ((va_contexts_[va_ctx_id].rt_format_attrib & (VA_RT_FORMAT_YUV420_12 | VA_RT_FORMAT_YUV422_12 | VA_RT_FORMAT_YUV444_12)) == 0) {
                dec_cap->is_supported = 0;
                return ROCDEC_SUCCESS;
            }
            break;
        }
        default: {
            dec_cap->is_supported = 0;
            return ROCDEC_SUCCESS;
        }
    }

    dec_cap->num_decoders = va_contexts_[va_ctx_id].num_dec_engines;
    dec_cap->output_format_mask = va_contexts_[va_ctx_id].output_format_mask;
    dec_cap->max_width = va_contexts_[va_ctx_id].max_width;
    dec_cap->max_height = va_contexts_[va_ctx_id].max_height;
    dec_cap->min_width = va_contexts_[va_ctx_id].min_width;
    dec_cap->min_height = va_contexts_[va_ctx_id].min_height;
    return ROCDEC_SUCCESS;
}

rocDecStatus VaContext::InitHIP(int device_id, hipDeviceProp_t& hip_dev_prop) {
    CHECK_HIP(hipGetDeviceCount(&num_devices_));
    if (num_devices_ < 1) {
        logger_.CriticalLog(MakeMsg("Didn't find any GPU."));
        return ROCDEC_DEVICE_INVALID;
    }
    if (device_id >= num_devices_) {
        logger_.CriticalLog(MakeMsg("ERROR: the requested device_id is not found!"));
        return ROCDEC_DEVICE_INVALID;
    }   
    CHECK_HIP(hipSetDevice(device_id));
    CHECK_HIP(hipGetDeviceProperties(&hip_dev_prop, device_id));
    return ROCDEC_SUCCESS;
}

rocDecStatus VaContext::InitVAAPI(int va_ctx_idx, std::string drm_node) {
    va_contexts_[va_ctx_idx].drm_fd = open(drm_node.c_str(), O_RDWR);
    if (va_contexts_[va_ctx_idx].drm_fd < 0) {
        logger_.CriticalLog(MakeMsg("Failed to open drm node." + drm_node));
        return ROCDEC_NOT_INITIALIZED;
    }
    va_contexts_[va_ctx_idx].va_display = vaGetDisplayDRM(va_contexts_[va_ctx_idx].drm_fd);
    if (!va_contexts_[va_ctx_idx].va_display) {
        logger_.CriticalLog(MakeMsg("Failed to create VA display."));
        return ROCDEC_NOT_INITIALIZED;
    }
    vaSetInfoCallback(va_contexts_[va_ctx_idx].va_display, NULL, NULL);
    int major_version = 0, minor_version = 0;
    CHECK_VAAPI(vaInitialize(va_contexts_[va_ctx_idx].va_display, &major_version, &minor_version));
    return ROCDEC_SUCCESS;
}

void VaContext::GetVisibleDevices(std::vector<int>& visible_devices_vetor) {
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
            if (found_mi300a != std::string::npos) {
                if (device_id < visible_devices.size()) {
                    offset = (visible_devices[device_id] % 6);
                } else {
                    offset = (device_id % 6);
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