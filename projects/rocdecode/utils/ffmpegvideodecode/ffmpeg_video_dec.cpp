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


#include "ffmpeg_video_dec.h"

//helper function
static inline float GetChromaWidthFactor(rocDecVideoSurfaceFormat surface_format) {
    float factor = 0.5;
    switch (surface_format) {
    case rocDecVideoSurfaceFormat_NV12:
    case rocDecVideoSurfaceFormat_P016:
    case rocDecVideoSurfaceFormat_YUV444:
    case rocDecVideoSurfaceFormat_YUV444_16Bit:
        factor = 1.0;
        break;
    case rocDecVideoSurfaceFormat_YUV420:
    case rocDecVideoSurfaceFormat_YUV420_16Bit:
    case rocDecVideoSurfaceFormat_YUV422:
    case rocDecVideoSurfaceFormat_YUV422_16Bit:
        factor = 0.5;
        break;
    }
    return factor;
};

FFMpegVideoDecoder::FFMpegVideoDecoder(int device_id, OutputSurfaceMemoryType out_mem_type, rocDecVideoCodec codec, bool force_zero_latency,
              const Rect *p_crop_rect, bool extract_user_sei_Message, uint32_t disp_delay,  int max_width, int max_height, uint32_t clk_rate) :
               RocVideoDecoder(device_id, out_mem_type, codec, force_zero_latency, p_crop_rect, extract_user_sei_Message, disp_delay, max_width, max_height, clk_rate, true) {

    if ((out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL) || (out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED)) {
        ROCDEC_THROW("Unsupported output memory type", ROCDEC_INVALID_PARAMETER);
    }
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        if (!InitHIP(device_id_)) {
            ROCDEC_THROW("Failed to initilize the HIP", ROCDEC_DEVICE_INVALID);
        }
    }
    // many of the decoder parameters are hardcoded below for just creating the decoder.
    // In the handlevideosequence callback, the decoder will get reconfigured to the actual parameters in the sequence header
    RocDecoderHostCreateInfo create_info = {};
    create_info.codec_type = codec;
    create_info.num_decode_threads = 0;     // default
    create_info.max_width = max_width;
    create_info.max_height = max_height;
    create_info.width = max_width;
    create_info.height = max_height;
    create_info.target_width = max_width;
    create_info.target_height = max_height;
    create_info.display_rect.left = 0;
    create_info.display_rect.right = static_cast<short>(max_width);
    create_info.display_rect.top = 0;
    create_info.display_rect.bottom = static_cast<short>(max_height);
    create_info.chroma_format = rocDecVideoChromaFormat_420;
    create_info.output_format = rocDecVideoSurfaceFormat_P016;
    create_info.bit_depth_minus_8 = 2;
    create_info.num_output_surfaces = 1;
    create_info.user_data = this;
    create_info.pfn_sequence_callback = FFMpegHandleVideoSequenceProc;
    create_info.pfn_display_picture = FFMpegHandlePictureDisplayProc;
    create_info.pfn_get_sei_msg = nullptr;        // tobe supported in future
    ROCDEC_API_CALL(rocDecCreateDecoderHost(&roc_decoder_, &create_info));
    // set disp_width and height to non_zero values for it doesn't trigger decoding error before actual start of decoding
    disp_width_ = max_width;
    disp_height_ = max_height;
    // fill output_surface_info_
    output_surface_info_.output_width = max_width;
    output_surface_info_.output_height = max_height;
    output_surface_info_.output_pitch  = max_width * 2;     // bytes_per_pixel 2
    output_surface_info_.output_vstride = max_height;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = 2;
    output_surface_info_.surface_format = rocDecVideoSurfaceFormat_P016;
    output_surface_info_.num_chroma_planes = 2;
    output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
}


FFMpegVideoDecoder::~FFMpegVideoDecoder() {
    std::lock_guard<std::mutex> lock(mtx_vp_frame_);
    for (auto &p_frame : vp_frames_) {
        if (p_frame.frame_ptr) {
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                hipError_t hip_status = hipFree(p_frame.frame_ptr);
                if (hip_status != hipSuccess) {
                    std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
                }
            }
        }
    }
}

/* Return value from HandleVideoSequence() are interpreted as   :
*  0: fail, 1: succeeded, > 1: override dpb size of parser (set by RocdecParserParams::max_num_decode_surfaces while creating parser)
*/
int FFMpegVideoDecoder::HandleVideoSequence(RocdecVideoFormatHost *format_host) {
    if (format_host == nullptr) {
        ROCDEC_THROW("Rocdec:: Invalid video format in HandleVideoSequence: ", ROCDEC_INVALID_PARAMETER);
        return 0;
    }
    auto start_time = StartTimer();
    RocdecVideoFormat *p_video_format = &format_host->video_format;
    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Information" << std::endl
        << "\tCodec        : " << GetCodecFmtName(p_video_format->codec) << std::endl;
        if (p_video_format->frame_rate.numerator && p_video_format->frame_rate.denominator) {
            input_video_info_str_ << "\tFrame rate   : " << p_video_format->frame_rate.numerator << "/" << p_video_format->frame_rate.denominator << " = " << 1.0 * p_video_format->frame_rate.numerator / p_video_format->frame_rate.denominator << " fps" << std::endl;
        }
    input_video_info_str_ << "\tSequence     : " << (p_video_format->progressive_sequence ? "Progressive" : "Interlaced") << std::endl
        << "\tCoded size   : [" << p_video_format->coded_width << ", " << p_video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << p_video_format->display_area.left << ", " << p_video_format->display_area.top << ", "
            << p_video_format->display_area.right << ", " << p_video_format->display_area.bottom << "]" << std::endl
        << "\tBit depth    : " << p_video_format->bit_depth_luma_minus8 + 8
    ;
    input_video_info_str_ << std::endl;

    int num_decode_surfaces = p_video_format->min_num_decode_surfaces;
    if (curr_video_format_ptr_ == nullptr) {
        curr_video_format_ptr_ = new RocdecVideoFormat();
    }
    // store current video format: this is required to call reconfigure from application in case of random seek
    if (curr_video_format_ptr_) memcpy(curr_video_format_ptr_, p_video_format, sizeof(RocdecVideoFormat));

    if (coded_width_ && coded_height_) {
        // rocdecCreateDecoder() has been called before, and now there's possible config change
        return ReconfigureDecoder(p_video_format);
    }
    // e_codec has been set in the constructor (for parser). Here it's set again for potential correction
    codec_id_ = p_video_format->codec;
    video_chroma_format_ = p_video_format->chroma_format;
    bitdepth_minus_8_ = p_video_format->bit_depth_luma_minus8;
    byte_per_pixel_ = bitdepth_minus_8_ > 0 ? 2 : 1;

    // convert AVPixelFormat to rocDecVideoChromaFormat
    video_surface_format_ = format_host->video_surface_format;
    coded_width_ = p_video_format->coded_width;
    coded_height_ = p_video_format->coded_height;
    disp_rect_.top = p_video_format->display_area.top;
    disp_rect_.bottom = p_video_format->display_area.bottom;
    disp_rect_.left = p_video_format->display_area.left;
    disp_rect_.right = p_video_format->display_area.right;
    disp_width_ = p_video_format->display_area.right - p_video_format->display_area.left;
    disp_height_ = p_video_format->display_area.bottom - p_video_format->display_area.top;

    // AV1 has max width/height of sequence in sequence header
    if (codec_id_ == rocDecVideoCodec_AV1 && p_video_format->seqhdr_data_length > 0) {
        // dont overwrite if it is already set from cmdline or reconfig.txt
        if (!(max_width_ > p_video_format->coded_width || max_height_ > p_video_format->coded_height)) {
            RocdecVideoFormatEx *vidFormatEx = (RocdecVideoFormatEx *)p_video_format;
            max_width_ = vidFormatEx->max_width;
            max_height_ = vidFormatEx->max_height;
        }
    }
    if (max_width_ < static_cast<int>(p_video_format->coded_width))
        max_width_ = p_video_format->coded_width;
    if (max_height_ < static_cast<int>(p_video_format->coded_height))
        max_height_ = p_video_format->coded_height;
    
    if (!(crop_rect_.right && crop_rect_.bottom)) {
        target_width_ = (disp_width_ + 1) & ~1;
        target_height_ = (disp_height_ + 1) & ~1;
    } else {
        target_width_ = (crop_rect_.right - crop_rect_.left + 1) & ~1;
        target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
    }
    chroma_height_ = static_cast<int>(ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    chroma_width_ = static_cast<int>(ceil(target_width_ * GetChromaWidthFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if (video_chroma_format_ == rocDecVideoChromaFormat_Monochrome) num_chroma_planes_ = 0;
    surface_stride_ = target_width_ * byte_per_pixel_;   

    // fill output_surface_info_
    output_surface_info_.output_width = target_width_;
    output_surface_info_.output_height = target_height_;
    output_surface_info_.output_pitch  = surface_stride_;
    output_surface_info_.output_vstride = target_height_;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_COPIED;
    } else if (out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED){
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    }
    input_video_info_str_ << "Video Decoding Params:" << std::endl
        << "\tNum Surfaces : " << num_decode_surfaces << std::endl
        << "\tCrop         : [" << disp_rect_.left << ", " << disp_rect_.top << ", "
        << disp_rect_.right << ", " << disp_rect_.bottom << "]" << std::endl
        << "\tResize       : " << target_width_ << "x" << target_height_ << std::endl
    ;
    input_video_info_str_ << std::endl;
    std::cout << input_video_info_str_.str();
    double elapsed_time = StopTimer(start_time);
    AddDecoderSessionOverHead(std::this_thread::get_id(), elapsed_time);
    return num_decode_surfaces;
}

bool FFMpegVideoDecoder::GetOutputSurfaceInfo(OutputSurfaceInfo **surface_info) {
    if (!disp_width_ || !disp_height_) {
        std::cerr << "ERROR: FFMpegVideo is not initialized" << std::endl;
        return false;
    }
    *surface_info = &output_surface_info_;
    return true;
}

/**
 * @brief function to reconfigure decoder if there is a change in sequence params.
 *
 * @param p_video_format
 * @return int 1: success 0: fail
 */
int FFMpegVideoDecoder::ReconfigureDecoder(RocdecVideoFormat *p_video_format) {
    if (p_video_format->codec != codec_id_) {
        ROCDEC_THROW("Reconfigure Not supported for codec change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if (p_video_format->chroma_format != video_chroma_format_) {
        ROCDEC_THROW("Reconfigure Not supported for chroma format change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if (p_video_format->bit_depth_luma_minus8 != bitdepth_minus_8_){
        ROCDEC_THROW("Reconfigure Not supported for bit depth change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    bool is_decode_res_changed = !(p_video_format->coded_width == coded_width_ && p_video_format->coded_height == coded_height_);
    bool is_display_rect_changed = !(p_video_format->display_area.bottom == disp_rect_.bottom &&
                                     p_video_format->display_area.top == disp_rect_.top &&
                                     p_video_format->display_area.left == disp_rect_.left &&
                                     p_video_format->display_area.right == disp_rect_.right);

    if (!is_decode_res_changed && !is_display_rect_changed && !b_force_recofig_flush_) {
        return 1;
    }

    // Flush and clear internal frame store to reconfigure when either coded size or display size has changed.
    if (p_reconfig_params_ && p_reconfig_params_->p_fn_reconfigure_flush) 
        num_frames_flushed_during_reconfig_ += p_reconfig_params_->p_fn_reconfigure_flush(this, p_reconfig_params_->reconfig_flush_mode, static_cast<void *>(p_reconfig_params_->p_reconfig_user_struct));
    // clear the existing output buffers of different size
    // note that app lose the remaining frames in the vp_frames in case application didn't set p_fn_reconfigure_flush_ callback
    std::lock_guard<std::mutex> lock(mtx_vp_frame_);
    while(!vp_frames_.empty()) {
        DecFrameBuffer *p_frame = &vp_frames_.back();
        // pop decoded frame
        vp_frames_.pop_back();
        if (p_frame->frame_ptr) {
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                hipError_t hip_status = hipFree(p_frame->frame_ptr);
                if (hip_status != hipSuccess) std::cerr << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
            }
        }
    }
    output_frame_cnt_ = 0;     // reset frame_count
    if (is_decode_res_changed) {
        coded_width_ = p_video_format->coded_width;
        coded_height_ = p_video_format->coded_height;
    }
    if (is_display_rect_changed) {
        disp_rect_.left = p_video_format->display_area.left;
        disp_rect_.right = p_video_format->display_area.right;
        disp_rect_.top = p_video_format->display_area.top;
        disp_rect_.bottom = p_video_format->display_area.bottom;
        disp_width_ = p_video_format->display_area.right - p_video_format->display_area.left;
        disp_height_ = p_video_format->display_area.bottom - p_video_format->display_area.top;

        if (!(crop_rect_.right && crop_rect_.bottom)) {
            target_width_ = (disp_width_ + 1) & ~1;
            target_height_ = (disp_height_ + 1) & ~1;
        } else {
            target_width_ = (crop_rect_.right - crop_rect_.left + 1) & ~1;
            target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
        }
        is_output_surface_changed_ = true;
    }

    surface_stride_ = target_width_ * byte_per_pixel_;
    chroma_height_ = static_cast<int>(std::ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    chroma_width_ = static_cast<int>(ceil(target_width_ * GetChromaWidthFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if (p_video_format->chroma_format == rocDecVideoChromaFormat_Monochrome) num_chroma_planes_ = 0;

    // Fill output_surface_info_
    output_surface_info_.output_width = target_width_;
    output_surface_info_.output_height = target_height_;
    output_surface_info_.output_pitch  = surface_stride_;
    output_surface_info_.output_vstride = (out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL) ? surface_vstride_ : target_height_;
    output_surface_info_.bit_depth = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_COPIED;
    } else if (out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED) {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type = OUT_SURFACE_MEM_HOST_COPIED;
    }

    // If the coded_width or coded_height hasn't changed but display resolution has changed, then need to update width and height for
    // correct output with cropping. There is no need to reconfigure the decoder.
    if (!is_decode_res_changed && is_display_rect_changed) {
        return 1;
    }

    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Resolution Changed:" << std::endl
        << "\tCoded size   : [" << p_video_format->coded_width << ", " << p_video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << p_video_format->display_area.left << ", " << p_video_format->display_area.top << ", "
            << p_video_format->display_area.right << ", " << p_video_format->display_area.bottom << "]" << std::endl;
    input_video_info_str_ << std::endl;
    return 1;
}


/**
 * @brief function to handle display picture
 * 
 * @param pDispInfo 
 * @return int 0:fail 1: success
 */
int FFMpegVideoDecoder::HandlePictureDisplay(RocdecParserDispInfo *pDispInfo) {
    if (b_extract_sei_message_) {
        if (sei_message_display_q_[pDispInfo->picture_index].sei_data) {
            // Write SEI Message
            uint8_t *sei_buffer = static_cast<uint8_t *>(sei_message_display_q_[pDispInfo->picture_index].sei_data);
            uint32_t sei_num_messages = sei_message_display_q_[pDispInfo->picture_index].sei_message_count;
            RocdecSeiMessage *sei_message = sei_message_display_q_[pDispInfo->picture_index].sei_message;
            if (fp_sei_) {
                for (uint32_t i = 0; i < sei_num_messages; i++) {
                    if (codec_id_ == rocDecVideoCodec_AVC || rocDecVideoCodec_HEVC) {
                        switch (sei_message[i].sei_message_type) {
                            case SEI_TYPE_TIME_CODE: {
                                //todo:: check if we need to write timecode
                            }
                            break;
                            case SEI_TYPE_USER_DATA_UNREGISTERED: {
                                fwrite(sei_buffer, sei_message[i].sei_message_size, 1, fp_sei_);
                            }
                            break;
                        }
                    }
                    if (codec_id_ == rocDecVideoCodec_AV1) {
                        fwrite(sei_buffer, sei_message[i].sei_message_size, 1, fp_sei_);
                    }    
                    sei_buffer += sei_message[i].sei_message_size;
                }
            }
            free(sei_message_display_q_[pDispInfo->picture_index].sei_data);
            sei_message_display_q_[pDispInfo->picture_index].sei_data = NULL; // to avoid double free
            free(sei_message_display_q_[pDispInfo->picture_index].sei_message);
            sei_message_display_q_[pDispInfo->picture_index].sei_message = NULL; // to avoid double free
        }
    }

    RocdecParserDispInfo *p_disp_info = static_cast<RocdecParserDispInfo *>(pDispInfo);
    RocdecProcParams video_proc_params = {};
    video_proc_params.progressive_frame = p_disp_info->progressive_frame;
    video_proc_params.top_field_first = p_disp_info->top_field_first;
    void * src_ptr[3] = { 0 };
    uint32_t src_pitch[3] = { 0 };
    ROCDEC_API_CALL(rocDecGetVideoFrameHost(roc_decoder_, pDispInfo->picture_index, src_ptr, src_pitch, &video_proc_params));
    // copy the decoded surface info device or host
    uint8_t *p_dec_frame = nullptr;
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        // if not enough frames in stock, allocate
        if (++output_frame_cnt_ > vp_frames_.size()) {
            num_alloced_frames_++;
            DecFrameBuffer dec_frame = {0};
            if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
                // allocate device memory
                HIP_API_CALL(hipMalloc((void **)&dec_frame.frame_ptr, GetFrameSize()));
            } else {
                dec_frame.frame_ptr = new uint8_t[GetFrameSize()];
            }

            dec_frame.pts = pDispInfo->pts;
            dec_frame.picture_index = pDispInfo->picture_index;     //picture_index is not used here since it is handled within FFMpeg decoder
            vp_frames_.push_back(dec_frame);
        }
        p_dec_frame = vp_frames_[output_frame_cnt_ - 1].frame_ptr;
    }

    // Copy luma data
    int dst_pitch = disp_width_ * byte_per_pixel_;
    uint8_t *p_src_ptr_y = static_cast<uint8_t *>(src_ptr[0]) + (disp_rect_.top + crop_rect_.top) * src_pitch[0] + (disp_rect_.left + crop_rect_.left) * byte_per_pixel_;
    uint8_t *p_frame_y = p_dec_frame;
    if (!p_frame_y && !p_src_ptr_y) {
        std::cerr << "HandlePictureDisplay: Invalid Memory address for src/dst" << std::endl;
        return 0;
    }
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        if (src_pitch[0] == dst_pitch) {
            int luma_size = src_pitch[0] * disp_height_;
            HIP_API_CALL(hipMemcpyHtoDAsync(p_frame_y, p_src_ptr_y, luma_size, hip_stream_));
        } else {
            // use 2d copy to copy an ROI
            HIP_API_CALL(hipMemcpy2DAsync(p_frame_y, dst_pitch, p_src_ptr_y, src_pitch[0], dst_pitch, disp_height_, hipMemcpyHostToDevice, hip_stream_));
        }
    } else {
        if (src_pitch[0] == dst_pitch) {
            int luma_size = src_pitch[0] * disp_height_;
            memcpy(p_frame_y, p_src_ptr_y, luma_size);
        } else {
            for (int i = 0; i < disp_height_; i++) {
                memcpy(p_dec_frame, p_src_ptr_y, dst_pitch);
                p_frame_y += dst_pitch;
                p_src_ptr_y += src_pitch[0];
            }
        }
    }
    // Copy chroma plane/s
    // rocDec output gives pointer to luma and chroma pointers seperated for the decoded frame
    uint8_t *p_frame_uv = p_dec_frame + dst_pitch * disp_height_;
    uint8_t *p_src_ptr_uv = static_cast<uint8_t *>(src_ptr[1]) + ((disp_rect_.top + crop_rect_.top) >> 1) * src_pitch[1] + ((disp_rect_.left + crop_rect_.left)>>1) * byte_per_pixel_ ;
    dst_pitch = chroma_width_ *  byte_per_pixel_;          
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
        if (src_pitch[1] == dst_pitch) {
            int chroma_size = chroma_height_ * dst_pitch;
            HIP_API_CALL(hipMemcpyHtoDAsync(p_frame_uv, p_src_ptr_uv, chroma_size, hip_stream_));
        } else {
            // use 2d copy to copy an ROI
            HIP_API_CALL(hipMemcpy2DAsync(p_frame_uv, dst_pitch, p_src_ptr_uv, src_pitch[1], dst_pitch, chroma_height_, hipMemcpyHostToDevice, hip_stream_));
        }
    } else {
        if (src_pitch[1] == dst_pitch) {
            int chroma_size = chroma_height_ * dst_pitch;
            memcpy(p_frame_uv, p_src_ptr_uv, chroma_size);
        } 
        else {
            for (int i = 0; i < chroma_height_; i++) {
                memcpy(p_frame_uv, p_src_ptr_uv, dst_pitch);
                p_frame_uv += dst_pitch;
                p_src_ptr_uv += src_pitch[1];
            }
        }
    }

    if (num_chroma_planes_ == 2) {
        uint8_t *p_frame_v = p_frame_uv + dst_pitch * chroma_height_;
        uint8_t *p_src_ptr_v = static_cast<uint8_t *>(src_ptr[2]) + (disp_rect_.top + crop_rect_.top) * src_pitch[2] + ((disp_rect_.left + crop_rect_.left) >> 1) * byte_per_pixel_;
        if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) {
            if (src_pitch[2] == dst_pitch) {
                int chroma_size = chroma_height_ * dst_pitch;
                HIP_API_CALL(hipMemcpyDtoDAsync(p_frame_v, p_src_ptr_v, chroma_size, hip_stream_));
            } else {
                // use 2d copy to copy an ROI
                HIP_API_CALL(hipMemcpy2DAsync(p_frame_v, dst_pitch, p_src_ptr_v, src_pitch[2], dst_pitch, chroma_height_, hipMemcpyHostToDevice, hip_stream_));
            }            
        }
        else {
            if (src_pitch[2] == dst_pitch) {
                int chroma_size = chroma_height_ * dst_pitch;
                memcpy(p_frame_v, p_src_ptr_v, chroma_size);
            } 
            else {
                for (int i = 0; i < chroma_height_; i++) {
                    memcpy(p_frame_v, p_src_ptr_v, dst_pitch);
                    p_frame_v += dst_pitch;
                    p_src_ptr_v += src_pitch[1];
                }
            }
        }         
    }
    if (out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED) HIP_API_CALL(hipStreamSynchronize(hip_stream_));

    return 1;
}


int FFMpegVideoDecoder::DecodeFrame(const uint8_t *data, size_t size, int pkt_flags, int64_t pts, int *num_decoded_pics) {
    output_frame_cnt_ = 0, output_frame_cnt_ret_ = 0;
    decoded_pic_cnt_ = 0;
    RocdecPicParamsHost pic_params = {};
    pic_params.bitstream_data_len = size;
    pic_params.bitstream_data = data;
    if (!data || size == 0) {
        pic_params.flags = ROCDEC_PKT_ENDOFPICTURE;     // mark end_of_picture flag for last frame
    }
    ROCDEC_API_CALL(rocDecDecodeFrameHost(roc_decoder_, &pic_params));
    if (num_decoded_pics) {
        *num_decoded_pics = output_frame_cnt_;
    }
    return output_frame_cnt_;
}


uint8_t* FFMpegVideoDecoder::GetFrame(int64_t *pts) {
    if (output_frame_cnt_ > 0) {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        if (vp_frames_.size() > 0){
            output_frame_cnt_--;
            if (pts) *pts = vp_frames_[output_frame_cnt_ret_].pts;
            return vp_frames_[output_frame_cnt_ret_++].frame_ptr;
        }
    }
    return nullptr;
}

bool FFMpegVideoDecoder::ReleaseFrame(int64_t pTimestamp, bool b_flushing) {
    // if not flushing the buffers are re-used, so keep them
    if (!b_flushing)  
        return true;
    else {
        // remove frames in flushing mode
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        DecFrameBuffer *fb = &vp_frames_[0];
        if (pTimestamp != fb->pts) {
            std::cerr << "Decoded Frame is released out of order" << std::endl;
            return false;
        }
        vp_frames_.erase(vp_frames_.begin());     // get rid of the frames from the framestore
    }
    return true;
}

void FFMpegVideoDecoder::SaveFrameToFile(std::string output_file_name, void *surf_mem, OutputSurfaceInfo *surf_info, size_t rgb_image_size) {
    uint8_t *hst_ptr = nullptr;
    bool is_rgb = (rgb_image_size != 0);
    uint64_t output_image_size = is_rgb ? rgb_image_size : surf_info->output_surface_size_in_bytes;
    if (surf_info->mem_type == OUT_SURFACE_MEM_DEV_COPIED) {
        if (hst_ptr == nullptr) {
            hst_ptr = new uint8_t [output_image_size];
        }
        hipError_t hip_status = hipSuccess;
        hip_status = hipMemcpyDtoH((void *)hst_ptr, surf_mem, output_image_size);
        if (hip_status != hipSuccess) {
            std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hipGetErrorName(hip_status) << ")" << std::endl;
            delete [] hst_ptr;
            return;
        }
    } else
        hst_ptr = static_cast<uint8_t *> (surf_mem);

    
    if (current_output_filename.empty()) {
        current_output_filename = output_file_name;
    }

    // don't overwrite to the same file if reconfigure is detected for a resolution changes.
    if (is_output_surface_changed_) {
        if (fp_out_) {
            fclose(fp_out_);
            fp_out_ = nullptr;
        }
        // Append the width and height of the new stream to the old file name to create a file name to save the new frames
        // do this only if resolution changes within a stream (e.g., decoding a multi-resolution stream using the videoDecode app)
        // don't append to the output_file_name if multiple output file name is provided (e.g., decoding multi-files using the videDecodeMultiFiles)
        if (!current_output_filename.compare(output_file_name)) {
            std::string::size_type const pos(output_file_name.find_last_of('.'));
            extra_output_file_count_++;
            std::string to_append = "_" + std::to_string(surf_info->output_width) + "_" + std::to_string(surf_info->output_height) + "_" + std::to_string(extra_output_file_count_);
            if (pos != std::string::npos) {
                output_file_name.insert(pos, to_append);
            } else {
                output_file_name += to_append;
            }
        }
        is_output_surface_changed_ = false;
    } 

    if (fp_out_ == nullptr) {
        fp_out_ = fopen(output_file_name.c_str(), "wb");
    }
    if (fp_out_) {
        if (!is_rgb) {
            uint8_t *tmp_hst_ptr = hst_ptr;
            int img_width = surf_info->output_width;
            int img_height = surf_info->output_height;
            int output_stride =  surf_info->output_pitch;
            if (img_width * surf_info->bytes_per_pixel == output_stride && img_height == surf_info->output_vstride) {
                fwrite(hst_ptr, 1, output_image_size, fp_out_);
            } else {
                uint32_t width = surf_info->output_width * surf_info->bytes_per_pixel;
                if (surf_info->bit_depth <= 16) {
                    for (int i = 0; i < surf_info->output_height; i++) {
                        fwrite(tmp_hst_ptr, 1, width, fp_out_);
                        tmp_hst_ptr += output_stride;
                    }
                    // dump chroma
                    uint32_t chroma_stride = (output_stride >> 1);
                    uint8_t *u_hst_ptr = hst_ptr + output_stride * surf_info->output_height;
                    uint8_t *v_hst_ptr = u_hst_ptr + chroma_stride * chroma_height_;
                    for (int i = 0; i < chroma_height_; i++) {
                        fwrite(u_hst_ptr, 1, chroma_width_, fp_out_);
                        u_hst_ptr += chroma_stride;
                    }
                    if (num_chroma_planes_ == 2) {
                        for (int i = 0; i < chroma_height_; i++) {
                            fwrite(v_hst_ptr, 1, chroma_width_, fp_out_);
                            v_hst_ptr += chroma_stride;
                        }
                    }
                } 
            }
        } else {
            fwrite(hst_ptr, 1, rgb_image_size, fp_out_);
        }
    }

    if (hst_ptr && (surf_info->mem_type != OUT_SURFACE_MEM_HOST_COPIED)) {
        delete [] hst_ptr;
    }
}
