/*
Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include "roc_video_dec.h"

RocVideoDecoder::RocVideoDecoder(int device_id, OutputSurfaceMemoryType out_mem_type,
                                 rocDecVideoCodec codec, bool force_zero_latency,
                                 const Rect* p_crop_rect, bool extract_user_sei_Message,
                                 uint32_t disp_delay, int max_width, int max_height,
                                 uint32_t clk_rate)
: device_id_{ device_id }
, out_mem_type_(out_mem_type)
, codec_id_(codec)
, b_force_zero_latency_(force_zero_latency)
, b_extract_sei_message_(extract_user_sei_Message)
, disp_delay_(disp_delay)
, max_width_(max_width)
, max_height_(max_height)
{
    if(!InitHIP(device_id_))
    {
        THROW("Failed to initilize the HIP");
    }
    if(p_crop_rect) crop_rect_ = *p_crop_rect;
    if(b_extract_sei_message_)
    {
        fp_sei_               = fopen("rocdec_sei_message.txt", "wb");
        curr_sei_message_ptr_ = new RocdecSeiMessageInfo;
        memset(&sei_message_display_q_, 0, sizeof(sei_message_display_q_));
    }
    // create rocdec videoparser
    RocdecParserParams parser_params = {};
    parser_params.codec_type         = codec_id_;
    parser_params.max_num_decode_surfaces =
        1;  // let the parser to determine the decode buffer pool size
    parser_params.clock_rate            = clk_rate;
    parser_params.max_display_delay     = disp_delay_;
    parser_params.user_data             = this;
    parser_params.pfn_sequence_callback = HandleVideoSequenceProc;
    parser_params.pfn_decode_picture    = HandlePictureDecodeProc;
    parser_params.pfn_display_picture =
        b_force_zero_latency_ ? NULL : HandlePictureDisplayProc;
    parser_params.pfn_get_sei_msg = b_extract_sei_message_ ? HandleSEIMessagesProc : NULL;
    ROCDEC_API_CALL(rocDecCreateVideoParser(&rocdec_parser_, &parser_params));
}

RocVideoDecoder::~RocVideoDecoder()
{
    auto start_time = StartTimer();
    if(curr_sei_message_ptr_)
    {
        delete curr_sei_message_ptr_;
        curr_sei_message_ptr_ = nullptr;
    }

    if(fp_sei_)
    {
        fclose(fp_sei_);
        fp_sei_ = nullptr;
    }

    if(rocdec_parser_)
    {
        rocDecDestroyVideoParser(rocdec_parser_);
        rocdec_parser_ = nullptr;
    }

    if(roc_decoder_)
    {
        rocDecDestroyDecoder(roc_decoder_);
        roc_decoder_ = nullptr;
    }

    if(curr_video_format_ptr_)
    {
        delete curr_video_format_ptr_;
        curr_video_format_ptr_ = nullptr;
    }

    std::lock_guard<std::mutex> lock(mtx_vp_frame_);
    if(out_mem_type_ != OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        for(auto& p_frame : vp_frames_)
        {
            if(p_frame.frame_ptr)
            {
                if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
                {
                    hipError_t hip_status = hipFree(p_frame.frame_ptr);
                    if(hip_status != hipSuccess)
                    {
                        std::cerr << "ERROR: hipFree failed! (" << hip_status << ")"
                                  << std::endl;
                    }
                }
                else
                    delete[](p_frame.frame_ptr);
                p_frame.frame_ptr = nullptr;
            }
        }
    }
    if(hip_stream_)
    {
        hipError_t hip_status = hipSuccess;
        hip_status            = hipStreamDestroy(hip_stream_);
        if(hip_status != hipSuccess)
        {
            std::cerr << "ERROR: hipStream_Destroy failed! (" << hip_status << ")"
                      << std::endl;
        }
    }
    if(fp_out_)
    {
        fclose(fp_out_);
        fp_out_ = nullptr;
    }

    double elapsed_time = StopTimer(start_time);
    AddDecoderSessionOverHead(std::this_thread::get_id(), elapsed_time);
}

static const char*
GetVideoCodecString(rocDecVideoCodec e_codec)
{
    static struct
    {
        rocDecVideoCodec e_codec;
        const char*      name;
    } aCodecName[] = {
        { rocDecVideoCodec_MPEG1, "MPEG-1" },
        { rocDecVideoCodec_MPEG2, "MPEG-2" },
        { rocDecVideoCodec_MPEG4, "MPEG-4 (ASP)" },
        { rocDecVideoCodec_AVC, "AVC/H.264" },
        { rocDecVideoCodec_HEVC, "H.265/HEVC" },
        { rocDecVideoCodec_AV1, "AV1" },
        { rocDecVideoCodec_VP8, "VP8" },
        { rocDecVideoCodec_VP9, "VP9" },
        { rocDecVideoCodec_JPEG, "M-JPEG" },
        { rocDecVideoCodec_NumCodecs, "Invalid" },
    };

    if(e_codec >= 0 && e_codec <= rocDecVideoCodec_NumCodecs)
    {
        return aCodecName[e_codec].name;
    }
    for(int i = rocDecVideoCodec_NumCodecs + 1;
        i < sizeof(aCodecName) / sizeof(aCodecName[0]); i++)
    {
        if(e_codec == aCodecName[i].e_codec)
        {
            return aCodecName[e_codec].name;
        }
    }
    return "Unknown";
}

/**
 * @brief function to return the name from codec_id
 *
 * @param codec_id
 * @return const char*
 */
const char*
RocVideoDecoder::GetCodecFmtName(rocDecVideoCodec codec_id)
{
    return GetVideoCodecString(codec_id);
}

static const char*
GetSurfaceFormatString(rocDecVideoSurfaceFormat surface_format_id)
{
    static struct
    {
        rocDecVideoSurfaceFormat surf_fmt;
        const char*              name;
    } SurfName[] = {
        { rocDecVideoSurfaceFormat_NV12, "NV12" },
        { rocDecVideoSurfaceFormat_P016, "P016" },
        { rocDecVideoSurfaceFormat_YUV444, "YUV444" },
        { rocDecVideoSurfaceFormat_YUV444_16Bit, "YUV444_16Bit" },
    };

    if(surface_format_id >= rocDecVideoSurfaceFormat_NV12 &&
       surface_format_id <= rocDecVideoSurfaceFormat_YUV444_16Bit)
        return SurfName[surface_format_id].name;
    else
        return "Unknown";
}

/**
 * @brief function to return the name from surface_format_id
 *
 * @param surface_format_id - enum for surface format
 * @return const char*
 */
const char*
RocVideoDecoder::GetSurfaceFmtName(rocDecVideoSurfaceFormat surface_format_id)
{
    return GetSurfaceFormatString(surface_format_id);
}

static const char*
GetVideoChromaFormatName(rocDecVideoChromaFormat e_chroma_format)
{
    static struct
    {
        rocDecVideoChromaFormat chroma_fmt;
        const char*             name;
    } ChromaFormatName[] = {
        { rocDecVideoChromaFormat_Monochrome, "YUV 400 (Monochrome)" },
        { rocDecVideoChromaFormat_420, "YUV 420" },
        { rocDecVideoChromaFormat_422, "YUV 422" },
        { rocDecVideoChromaFormat_444, "YUV 444" },
    };

    if(e_chroma_format >= 0 && e_chroma_format <= rocDecVideoChromaFormat_444)
    {
        return ChromaFormatName[e_chroma_format].name;
    }
    return "Unknown";
}

static float
GetChromaHeightFactor(rocDecVideoSurfaceFormat surface_format)
{
    float factor = 0.5;
    switch(surface_format)
    {
        case rocDecVideoSurfaceFormat_NV12:
        case rocDecVideoSurfaceFormat_P016: factor = 0.5; break;
        case rocDecVideoSurfaceFormat_YUV444:
        case rocDecVideoSurfaceFormat_YUV444_16Bit: factor = 1.0; break;
    }

    return factor;
}

static int
GetChromaPlaneCount(rocDecVideoSurfaceFormat surface_format)
{
    int num_planes = 1;
    switch(surface_format)
    {
        case rocDecVideoSurfaceFormat_NV12:
        case rocDecVideoSurfaceFormat_P016: num_planes = 1; break;
        case rocDecVideoSurfaceFormat_YUV444:
        case rocDecVideoSurfaceFormat_YUV444_16Bit: num_planes = 2; break;
    }

    return num_planes;
}

static void
GetSurfaceStrideInternal(rocDecVideoSurfaceFormat surface_format, uint32_t width,
                         uint32_t height, uint32_t* pitch, uint32_t* vstride)
{
    switch(surface_format)
    {
        case rocDecVideoSurfaceFormat_NV12:
            *pitch   = align(width, 256);
            *vstride = align(height, 16);
            break;
        case rocDecVideoSurfaceFormat_P016:
            *pitch   = align(width, 128) * 2;
            *vstride = align(height, 16);
            break;
        case rocDecVideoSurfaceFormat_YUV444:
            *pitch   = align(width, 256);
            *vstride = align(height, 16);
            break;
        case rocDecVideoSurfaceFormat_YUV444_16Bit:
            *pitch   = align(width, 128) * 2;
            *vstride = align(height, 16);
            break;
    }
    return;
}

/* Return value from HandleVideoSequence() are interpreted as   :
 *  0: fail, 1: succeeded, > 1: override dpb size of parser (set by
 * CUVIDPARSERPARAMS::max_num_decode_surfaces while creating parser)
 */
int
RocVideoDecoder::HandleVideoSequence(RocdecVideoFormat* p_video_format)
{
    if(p_video_format == nullptr)
    {
        ROCDEC_THROW("Rocdec:: Invalid video format in HandleVideoSequence: ",
                     ROCDEC_INVALID_PARAMETER);
        return 0;
    }
    auto start_time = StartTimer();
    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Information" << std::endl
                          << "\tCodec        : " << GetCodecFmtName(p_video_format->codec)
                          << std::endl;
    if(p_video_format->frame_rate.numerator && p_video_format->frame_rate.denominator)
    {
        input_video_info_str_ << "\tFrame rate   : "
                              << p_video_format->frame_rate.numerator << "/"
                              << p_video_format->frame_rate.denominator << " = "
                              << 1.0 * p_video_format->frame_rate.numerator /
                                     p_video_format->frame_rate.denominator
                              << " fps" << std::endl;
    }
    input_video_info_str_
        << "\tSequence     : "
        << (p_video_format->progressive_sequence ? "Progressive" : "Interlaced")
        << std::endl
        << "\tCoded size   : [" << p_video_format->coded_width << ", "
        << p_video_format->coded_height << "]" << std::endl
        << "\tDisplay area : [" << p_video_format->display_area.left << ", "
        << p_video_format->display_area.top << ", " << p_video_format->display_area.right
        << ", " << p_video_format->display_area.bottom << "]" << std::endl
        << "\tChroma       : " << GetVideoChromaFormatName(p_video_format->chroma_format)
        << std::endl
        << "\tBit depth    : " << p_video_format->bit_depth_luma_minus8 + 8;
    input_video_info_str_ << std::endl;

    int num_decode_surfaces = p_video_format->min_num_decode_surfaces;

    RocdecDecodeCaps decode_caps;
    memset(&decode_caps, 0, sizeof(decode_caps));
    decode_caps.codec_type        = p_video_format->codec;
    decode_caps.chroma_format     = p_video_format->chroma_format;
    decode_caps.bit_depth_minus_8 = p_video_format->bit_depth_luma_minus8;

    rocDecGetDecoderCaps(&decode_caps);
    if(!decode_caps.is_supported)
    {
        ROCDEC_THROW("rocDecode:: Codec not supported on this GPU ",
                     ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if((p_video_format->coded_width > decode_caps.max_width) ||
       (p_video_format->coded_height > decode_caps.max_height))
    {
        std::ostringstream errorString;
        errorString << std::endl
                    << "Resolution          : " << p_video_format->coded_width << "x"
                    << p_video_format->coded_height << std::endl
                    << "Max Supported (wxh) : " << decode_caps.max_width << "x"
                    << decode_caps.max_height << std::endl
                    << "Resolution not supported on this GPU ";
        const std::string cErr = errorString.str();
        ROCDEC_THROW(cErr, ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if(curr_video_format_ptr_ == nullptr)
    {
        curr_video_format_ptr_ = new RocdecVideoFormat();
    }
    // store current video format: this is required to call reconfigure from application
    // in case of random seek
    if(curr_video_format_ptr_)
        memcpy(curr_video_format_ptr_, p_video_format, sizeof(RocdecVideoFormat));

    if(coded_width_ && coded_height_)
    {
        // rocdecCreateDecoder() has been called before, and now there's possible config
        // change
        return ReconfigureDecoder(p_video_format);
    }
    // e_codec has been set in the constructor (for parser). Here it's set again for
    // potential correction
    codec_id_            = p_video_format->codec;
    video_chroma_format_ = p_video_format->chroma_format;
    bitdepth_minus_8_    = p_video_format->bit_depth_luma_minus8;
    byte_per_pixel_      = bitdepth_minus_8_ > 0 ? 2 : 1;

    // Set the output surface format same as chroma format
    if(video_chroma_format_ == rocDecVideoChromaFormat_420 ||
       rocDecVideoChromaFormat_Monochrome)
        video_surface_format_ = bitdepth_minus_8_ ? rocDecVideoSurfaceFormat_P016
                                                  : rocDecVideoSurfaceFormat_NV12;
    else if(video_chroma_format_ == rocDecVideoChromaFormat_444)
        video_surface_format_ = bitdepth_minus_8_ ? rocDecVideoSurfaceFormat_YUV444_16Bit
                                                  : rocDecVideoSurfaceFormat_YUV444;
    else if(video_chroma_format_ == rocDecVideoChromaFormat_422)
        video_surface_format_ = rocDecVideoSurfaceFormat_NV12;

    // Check if output format supported. If not, check falback options
    if(!(decode_caps.output_format_mask & (1 << video_surface_format_)))
    {
        if(decode_caps.output_format_mask & (1 << rocDecVideoSurfaceFormat_NV12))
            video_surface_format_ = rocDecVideoSurfaceFormat_NV12;
        else if(decode_caps.output_format_mask & (1 << rocDecVideoSurfaceFormat_P016))
            video_surface_format_ = rocDecVideoSurfaceFormat_P016;
        else if(decode_caps.output_format_mask & (1 << rocDecVideoSurfaceFormat_YUV444))
            video_surface_format_ = rocDecVideoSurfaceFormat_YUV444;
        else if(decode_caps.output_format_mask &
                (1 << rocDecVideoSurfaceFormat_YUV444_16Bit))
            video_surface_format_ = rocDecVideoSurfaceFormat_YUV444_16Bit;
        else
            ROCDEC_THROW("No supported output format found", ROCDEC_NOT_SUPPORTED);
    }

    coded_width_      = p_video_format->coded_width;
    coded_height_     = p_video_format->coded_height;
    disp_rect_.top    = p_video_format->display_area.top;
    disp_rect_.bottom = p_video_format->display_area.bottom;
    disp_rect_.left   = p_video_format->display_area.left;
    disp_rect_.right  = p_video_format->display_area.right;
    disp_width_  = p_video_format->display_area.right - p_video_format->display_area.left;
    disp_height_ = p_video_format->display_area.bottom - p_video_format->display_area.top;

    // AV1 has max width/height of sequence in sequence header
    if(codec_id_ == rocDecVideoCodec_AV1 && p_video_format->seqhdr_data_length > 0)
    {
        // dont overwrite if it is already set from cmdline or reconfig.txt
        if(!(max_width_ > p_video_format->coded_width ||
             max_height_ > p_video_format->coded_height))
        {
            RocdecVideoFormatEx* vidFormatEx = (RocdecVideoFormatEx*) p_video_format;
            max_width_                       = vidFormatEx->max_width;
            max_height_                      = vidFormatEx->max_height;
        }
    }
    if(max_width_ < (int) p_video_format->coded_width)
        max_width_ = p_video_format->coded_width;
    if(max_height_ < (int) p_video_format->coded_height)
        max_height_ = p_video_format->coded_height;

    RocDecoderCreateInfo videoDecodeCreateInfo = { 0 };
    videoDecodeCreateInfo.device_id            = device_id_;
    videoDecodeCreateInfo.codec_type           = codec_id_;
    videoDecodeCreateInfo.chroma_format        = video_chroma_format_;
    videoDecodeCreateInfo.output_format        = video_surface_format_;
    videoDecodeCreateInfo.bit_depth_minus_8    = bitdepth_minus_8_;
    videoDecodeCreateInfo.num_decode_surfaces  = num_decode_surfaces;
    videoDecodeCreateInfo.width                = coded_width_;
    videoDecodeCreateInfo.height               = coded_height_;
    videoDecodeCreateInfo.max_width            = max_width_;
    videoDecodeCreateInfo.max_height           = max_height_;
    if(!(crop_rect_.right && crop_rect_.bottom))
    {
        videoDecodeCreateInfo.display_rect.top    = disp_rect_.top;
        videoDecodeCreateInfo.display_rect.bottom = disp_rect_.bottom;
        videoDecodeCreateInfo.display_rect.left   = disp_rect_.left;
        videoDecodeCreateInfo.display_rect.right  = disp_rect_.right;
        target_width_                             = (disp_width_ + 1) & ~1;
        target_height_                            = (disp_height_ + 1) & ~1;
    }
    else
    {
        videoDecodeCreateInfo.display_rect.top    = crop_rect_.top;
        videoDecodeCreateInfo.display_rect.bottom = crop_rect_.bottom;
        videoDecodeCreateInfo.display_rect.left   = crop_rect_.left;
        videoDecodeCreateInfo.display_rect.right  = crop_rect_.right;
        target_width_  = (crop_rect_.right - crop_rect_.left + 1) & ~1;
        target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
    }
    videoDecodeCreateInfo.target_width  = target_width_;
    videoDecodeCreateInfo.target_height = target_height_;

    chroma_height_ =
        (int) (ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if(video_chroma_format_ == rocDecVideoChromaFormat_Monochrome) num_chroma_planes_ = 0;
    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL ||
       out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED)
        GetSurfaceStrideInternal(video_surface_format_, p_video_format->coded_width,
                                 p_video_format->coded_height, &surface_stride_,
                                 &surface_vstride_);
    else
    {
        surface_stride_ =
            videoDecodeCreateInfo.target_width *
            byte_per_pixel_;  // todo:: check if we need pitched memory for faster copy
    }
    chroma_vstride_ =
        (int) (ceil(surface_vstride_ * GetChromaHeightFactor(video_surface_format_)));
    // fill output_surface_info_
    output_surface_info_.output_width    = target_width_;
    output_surface_info_.output_height   = target_height_;
    output_surface_info_.output_pitch    = surface_stride_;
    output_surface_info_.output_vstride  = (out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
                                               ? surface_vstride_
                                               : videoDecodeCreateInfo.target_height;
    output_surface_info_.bit_depth       = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format  = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        output_surface_info_.output_surface_size_in_bytes =
            surface_stride_ * (surface_vstride_ + (chroma_vstride_ * num_chroma_planes_));
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;
    }
    else if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
    {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type                     = OUT_SURFACE_MEM_DEV_COPIED;
    }
    else if(out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED)
    {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type                     = OUT_SURFACE_MEM_HOST_COPIED;
    }
    else
    {
        output_surface_info_.output_surface_size_in_bytes =
            surface_stride_ * (surface_vstride_ + (chroma_vstride_ * num_chroma_planes_));
        output_surface_info_.mem_type = OUT_SURFACE_MEM_NOT_MAPPED;
    }

    input_video_info_str_ << "Video Decoding Params:" << std::endl
                          << "\tNum Surfaces : "
                          << videoDecodeCreateInfo.num_decode_surfaces << std::endl
                          << "\tCrop         : ["
                          << videoDecodeCreateInfo.display_rect.left << ", "
                          << videoDecodeCreateInfo.display_rect.top << ", "
                          << videoDecodeCreateInfo.display_rect.right << ", "
                          << videoDecodeCreateInfo.display_rect.bottom << "]" << std::endl
                          << "\tResize       : " << videoDecodeCreateInfo.target_width
                          << "x" << videoDecodeCreateInfo.target_height << std::endl;
    input_video_info_str_ << std::endl;
    std::cout << input_video_info_str_.str();

    ROCDEC_API_CALL(rocDecCreateDecoder(&roc_decoder_, &videoDecodeCreateInfo));
    double elapsed_time = StopTimer(start_time);
    AddDecoderSessionOverHead(std::this_thread::get_id(), elapsed_time);
    return num_decode_surfaces;
}

/**
 * @brief Function to set the Reconfig Params object
 *
 * @param p_reconfig_params: pointer to reconfig params struct
 * @return true : success
 * @return false : fail
 */
bool
RocVideoDecoder::SetReconfigParams(ReconfigParams* p_reconfig_params,
                                   bool            b_force_reconfig_flush)
{
    if(!p_reconfig_params)
    {
        std::cerr << "ERROR: Invalid reconfig struct passed! " << std::endl;
        return false;
    }
    // save it
    p_reconfig_params_     = p_reconfig_params;
    b_force_recofig_flush_ = b_force_reconfig_flush;
    return true;
}

/**
 * @brief Function to force Reconfigure Flush: needed for random seeking to key frames
 *
 * @return int 1: Success 0: Fail
 */
int
RocVideoDecoder::FlushAndReconfigure()
{
    if(!p_reconfig_params_)
    {
        std::cerr << "ERROR: Reconfig params is not set! " << std::endl;
        return 0;
    }
    if(!curr_video_format_ptr_)
    {
        std::cerr << "ERROR: video format is not initialized! " << std::endl;
        return 0;
    }
    // call reconfigure
    b_force_recofig_flush_ = true;  // if not already set to force reconfigure
    ReconfigureDecoder(curr_video_format_ptr_);
    return true;
}

/**
 * @brief function to reconfigure decoder if there is a change in sequence params.
 *
 * @param p_video_format
 * @return int 1: success 0: fail
 */
int
RocVideoDecoder::ReconfigureDecoder(RocdecVideoFormat* p_video_format)
{
    if(p_video_format->codec != codec_id_)
    {
        ROCDEC_THROW("Reconfigure Not supported for codec change", ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if(p_video_format->chroma_format != video_chroma_format_)
    {
        ROCDEC_THROW("Reconfigure Not supported for chroma format change",
                     ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    if(p_video_format->bit_depth_luma_minus8 != bitdepth_minus_8_)
    {
        ROCDEC_THROW("Reconfigure Not supported for bit depth change",
                     ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    bool is_decode_res_changed = !(p_video_format->coded_width == coded_width_ &&
                                   p_video_format->coded_height == coded_height_);
    bool is_display_rect_changed =
        !(p_video_format->display_area.bottom == disp_rect_.bottom &&
          p_video_format->display_area.top == disp_rect_.top &&
          p_video_format->display_area.left == disp_rect_.left &&
          p_video_format->display_area.right == disp_rect_.right);

    if(!is_decode_res_changed && !is_display_rect_changed && !b_force_recofig_flush_)
    {
        return 1;
    }

    // Flush and clear internal frame store to reconfigure when either coded size or
    // display size has changed.
    if(p_reconfig_params_ && p_reconfig_params_->p_fn_reconfigure_flush)
        num_frames_flushed_during_reconfig_ += p_reconfig_params_->p_fn_reconfigure_flush(
            this, p_reconfig_params_->reconfig_flush_mode,
            static_cast<void*>(p_reconfig_params_->p_reconfig_user_struct));
    // clear the existing output buffers of different size
    // note that app lose the remaining frames in the vp_frames/vp_frames_q in case
    // application didn't set p_fn_reconfigure_flush_ callback
    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        ReleaseInternalFrames();
    }
    else
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        while(!vp_frames_.empty())
        {
            DecFrameBuffer* p_frame = &vp_frames_.back();
            // pop decoded frame
            vp_frames_.pop_back();
            if(p_frame->frame_ptr)
            {
                if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
                {
                    hipError_t hip_status = hipFree(p_frame->frame_ptr);
                    if(hip_status != hipSuccess)
                        std::cerr << "ERROR: hipFree failed! (" << hip_status << ")"
                                  << std::endl;
                }
                else
                    delete[](p_frame->frame_ptr);
            }
        }
    }
    output_frame_cnt_ = 0;  // reset frame_count
    if(is_decode_res_changed)
    {
        coded_width_  = p_video_format->coded_width;
        coded_height_ = p_video_format->coded_height;
    }
    if(is_display_rect_changed)
    {
        disp_rect_.left   = p_video_format->display_area.left;
        disp_rect_.right  = p_video_format->display_area.right;
        disp_rect_.top    = p_video_format->display_area.top;
        disp_rect_.bottom = p_video_format->display_area.bottom;
        disp_width_ =
            p_video_format->display_area.right - p_video_format->display_area.left;
        disp_height_ =
            p_video_format->display_area.bottom - p_video_format->display_area.top;
        chroma_height_ = static_cast<int>(
            std::ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
        if(!(crop_rect_.right && crop_rect_.bottom))
        {
            target_width_  = (disp_width_ + 1) & ~1;
            target_height_ = (disp_height_ + 1) & ~1;
        }
        else
        {
            target_width_  = (crop_rect_.right - crop_rect_.left + 1) & ~1;
            target_height_ = (crop_rect_.bottom - crop_rect_.top + 1) & ~1;
        }
    }

    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL ||
       out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED)
    {
        GetSurfaceStrideInternal(video_surface_format_, coded_width_, coded_height_,
                                 &surface_stride_, &surface_vstride_);
    }
    else
    {
        surface_stride_ = target_width_ * byte_per_pixel_;
    }
    chroma_height_ = static_cast<int>(
        ceil(target_height_ * GetChromaHeightFactor(video_surface_format_)));
    num_chroma_planes_ = GetChromaPlaneCount(video_surface_format_);
    if(p_video_format->chroma_format == rocDecVideoChromaFormat_Monochrome)
        num_chroma_planes_ = 0;
    chroma_vstride_ = static_cast<int>(
        std::ceil(surface_vstride_ * GetChromaHeightFactor(video_surface_format_)));
    // Fill output_surface_info_
    output_surface_info_.output_width    = target_width_;
    output_surface_info_.output_height   = target_height_;
    output_surface_info_.output_pitch    = surface_stride_;
    output_surface_info_.output_vstride  = (out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
                                               ? surface_vstride_
                                               : target_height_;
    output_surface_info_.bit_depth       = bitdepth_minus_8_ + 8;
    output_surface_info_.bytes_per_pixel = byte_per_pixel_;
    output_surface_info_.surface_format  = video_surface_format_;
    output_surface_info_.num_chroma_planes = num_chroma_planes_;
    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        output_surface_info_.output_surface_size_in_bytes =
            surface_stride_ * (surface_vstride_ + (chroma_vstride_ * num_chroma_planes_));
        output_surface_info_.mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;
    }
    else if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
    {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type                     = OUT_SURFACE_MEM_DEV_COPIED;
    }
    else if(out_mem_type_ == OUT_SURFACE_MEM_HOST_COPIED)
    {
        output_surface_info_.output_surface_size_in_bytes = GetFrameSize();
        output_surface_info_.mem_type                     = OUT_SURFACE_MEM_HOST_COPIED;
    }
    else
    {
        output_surface_info_.output_surface_size_in_bytes =
            surface_stride_ * (surface_vstride_ + (chroma_vstride_ * num_chroma_planes_));
        output_surface_info_.mem_type = OUT_SURFACE_MEM_NOT_MAPPED;
    }

    // If the coded_width or coded_height hasn't changed but display resolution has
    // changed, then need to update width and height for correct output with cropping.
    // There is no need to reconfigure the decoder.
    if(!is_decode_res_changed && is_display_rect_changed)
    {
        return 1;
    }

    RocdecReconfigureDecoderInfo reconfig_params = { 0 };
    reconfig_params.width                        = coded_width_;
    reconfig_params.height                       = coded_height_;
    reconfig_params.target_width                 = target_width_;
    reconfig_params.target_height                = target_height_;
    reconfig_params.num_decode_surfaces = p_video_format->min_num_decode_surfaces;
    if(!(crop_rect_.right && crop_rect_.bottom))
    {
        reconfig_params.display_rect.top    = disp_rect_.top;
        reconfig_params.display_rect.bottom = disp_rect_.bottom;
        reconfig_params.display_rect.left   = disp_rect_.left;
        reconfig_params.display_rect.right  = disp_rect_.right;
    }
    else
    {
        reconfig_params.display_rect.top    = crop_rect_.top;
        reconfig_params.display_rect.bottom = crop_rect_.bottom;
        reconfig_params.display_rect.left   = crop_rect_.left;
        reconfig_params.display_rect.right  = crop_rect_.right;
    }

    if(roc_decoder_ == nullptr)
    {
        ROCDEC_THROW("Reconfigurition of the decoder detected but the decoder was not "
                     "initialized previoulsy!",
                     ROCDEC_NOT_SUPPORTED);
        return 0;
    }
    ROCDEC_API_CALL(rocDecReconfigureDecoder(roc_decoder_, &reconfig_params));

    input_video_info_str_.str("");
    input_video_info_str_.clear();
    input_video_info_str_ << "Input Video Resolution Changed:" << std::endl
                          << "\tCoded size   : [" << p_video_format->coded_width << ", "
                          << p_video_format->coded_height << "]" << std::endl
                          << "\tDisplay area : [" << p_video_format->display_area.left
                          << ", " << p_video_format->display_area.top << ", "
                          << p_video_format->display_area.right << ", "
                          << p_video_format->display_area.bottom << "]" << std::endl;
    input_video_info_str_ << std::endl;
    input_video_info_str_ << "Video Decoding Params:" << std::endl
                          << "\tNum Surfaces : " << reconfig_params.num_decode_surfaces
                          << std::endl
                          << "\tResize       : " << reconfig_params.target_width << "x"
                          << reconfig_params.target_height << std::endl;
    input_video_info_str_ << std::endl;
    std::cout << input_video_info_str_.str();

    is_decoder_reconfigured_ = true;
    return 1;
}

/**
 * @brief
 *
 * @param pPicParams
 * @return int 1: success 0: fail
 */
int
RocVideoDecoder::HandlePictureDecode(RocdecPicParams* pPicParams)
{
    if(!roc_decoder_)
    {
        THROW("RocDecoder not initialized: failed with ErrCode: " +
              TOSTR(ROCDEC_NOT_INITIALIZED));
    }
    pic_num_in_dec_order_[pPicParams->curr_pic_idx] = decode_poc_++;
    ROCDEC_API_CALL(rocDecDecodeFrame(roc_decoder_, pPicParams));
    last_decode_surf_idx_ = pPicParams->curr_pic_idx;
    decoded_pic_cnt_++;
    if(b_force_zero_latency_ &&
       ((!pPicParams->field_pic_flag) || (pPicParams->second_field)))
    {
        RocdecParserDispInfo disp_info;
        memset(&disp_info, 0, sizeof(disp_info));
        disp_info.picture_index     = pPicParams->curr_pic_idx;
        disp_info.progressive_frame = !pPicParams->field_pic_flag;
        disp_info.top_field_first   = pPicParams->bottom_field_flag ^ 1;
        HandlePictureDisplay(&disp_info);
    }
    return 1;
}

/**
 * @brief function to handle display picture
 *
 * @param pDispInfo
 * @return int 0:fail 1: success
 */
int
RocVideoDecoder::HandlePictureDisplay(RocdecParserDispInfo* pDispInfo)
{
    RocdecProcParams video_proc_params  = {};
    video_proc_params.progressive_frame = pDispInfo->progressive_frame;
    video_proc_params.top_field_first   = pDispInfo->top_field_first;

    if(b_extract_sei_message_)
    {
        if(sei_message_display_q_[pDispInfo->picture_index].sei_data)
        {
            // Write SEI Message
            uint8_t* sei_buffer =
                (uint8_t*) (sei_message_display_q_[pDispInfo->picture_index].sei_data);
            uint32_t sei_num_messages =
                sei_message_display_q_[pDispInfo->picture_index].sei_message_count;
            RocdecSeiMessage* sei_message =
                sei_message_display_q_[pDispInfo->picture_index].sei_message;
            if(fp_sei_)
            {
                for(uint32_t i = 0; i < sei_num_messages; i++)
                {
                    if(codec_id_ == rocDecVideoCodec_AVC || rocDecVideoCodec_HEVC)
                    {
                        switch(sei_message[i].sei_message_type)
                        {
                            case SEI_TYPE_TIME_CODE:
                            {
                                // todo:: check if we need to write timecode
                            }
                            break;
                            case SEI_TYPE_USER_DATA_UNREGISTERED:
                            {
                                fwrite(sei_buffer, sei_message[i].sei_message_size, 1,
                                       fp_sei_);
                            }
                            break;
                        }
                    }
                    if(codec_id_ == rocDecVideoCodec_AV1)
                    {
                        fwrite(sei_buffer, sei_message[i].sei_message_size, 1, fp_sei_);
                    }
                    sei_buffer += sei_message[i].sei_message_size;
                }
            }
            free(sei_message_display_q_[pDispInfo->picture_index].sei_data);
            sei_message_display_q_[pDispInfo->picture_index].sei_data =
                NULL;  // to avoid double free
            free(sei_message_display_q_[pDispInfo->picture_index].sei_message);
            sei_message_display_q_[pDispInfo->picture_index].sei_message =
                NULL;  // to avoid double free
        }
    }
    if(out_mem_type_ != OUT_SURFACE_MEM_NOT_MAPPED)
    {
        void*    src_dev_ptr[3] = { 0 };
        uint32_t src_pitch[3]   = { 0 };
        ROCDEC_API_CALL(rocDecGetVideoFrame(roc_decoder_, pDispInfo->picture_index,
                                            src_dev_ptr, src_pitch, &video_proc_params));
        RocdecDecodeStatus dec_status;
        memset(&dec_status, 0, sizeof(dec_status));
        rocDecStatus result =
            rocDecGetDecodeStatus(roc_decoder_, pDispInfo->picture_index, &dec_status);
        if(result == ROCDEC_SUCCESS &&
           (dec_status.decode_status == rocDecodeStatus_Error ||
            dec_status.decode_status == rocDecodeStatus_Error_Concealed))
        {
            std::cerr << "Decode Error occurred for picture: "
                      << pic_num_in_dec_order_[pDispInfo->picture_index] << std::endl;
        }
        if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL)
        {
            DecFrameBuffer dec_frame = { 0 };
            dec_frame.frame_ptr      = (uint8_t*) (src_dev_ptr[0]);
            dec_frame.pts            = pDispInfo->pts;
            dec_frame.picture_index  = pDispInfo->picture_index;
            std::lock_guard<std::mutex> lock(mtx_vp_frame_);
            vp_frames_q_.push(dec_frame);
            output_frame_cnt_++;
        }
        else
        {
            // copy the decoded surface info device or host
            uint8_t* p_dec_frame = nullptr;
            {
                std::lock_guard<std::mutex> lock(mtx_vp_frame_);
                // if not enough frames in stock, allocate
                if((unsigned) ++output_frame_cnt_ > vp_frames_.size())
                {
                    num_alloced_frames_++;
                    DecFrameBuffer dec_frame = { 0 };
                    if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
                    {
                        // allocate device memory
                        HIP_API_CALL(
                            hipMalloc((void**) &dec_frame.frame_ptr, GetFrameSize()));
                    }
                    else
                    {
                        dec_frame.frame_ptr = new uint8_t[GetFrameSize()];
                    }
                    dec_frame.pts           = pDispInfo->pts;
                    dec_frame.picture_index = pDispInfo->picture_index;
                    vp_frames_.push_back(dec_frame);
                }
                p_dec_frame = vp_frames_[output_frame_cnt_ - 1].frame_ptr;
            }
            // Copy luma data
            int      dst_pitch   = disp_width_ * byte_per_pixel_;
            uint8_t* p_src_ptr_y = static_cast<uint8_t*>(src_dev_ptr[0]) +
                                   (disp_rect_.top + crop_rect_.top) * src_pitch[0] +
                                   (disp_rect_.left + crop_rect_.left) * byte_per_pixel_;
            if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
            {
                if(src_pitch[0] == dst_pitch)
                {
                    int luma_size = src_pitch[0] * coded_height_;
                    HIP_API_CALL(hipMemcpyDtoDAsync(p_dec_frame, p_src_ptr_y, luma_size,
                                                    hip_stream_));
                }
                else
                {
                    // use 2d copy to copy an ROI
                    HIP_API_CALL(hipMemcpy2DAsync(p_dec_frame, dst_pitch, p_src_ptr_y,
                                                  src_pitch[0], dst_pitch, disp_height_,
                                                  hipMemcpyDeviceToDevice, hip_stream_));
                }
            }
            else
                HIP_API_CALL(hipMemcpy2DAsync(p_dec_frame, dst_pitch, p_src_ptr_y,
                                              src_pitch[0], dst_pitch, disp_height_,
                                              hipMemcpyDeviceToHost, hip_stream_));

            // Copy chroma plane ( )
            // rocDec output gives pointer to luma and chroma pointers seperated for the
            // decoded frame
            uint8_t* p_frame_uv = p_dec_frame + dst_pitch * disp_height_;
            uint8_t* p_src_ptr_uv =
                (num_chroma_planes_ == 1)
                    ? static_cast<uint8_t*>(src_dev_ptr[1]) +
                          ((disp_rect_.top + crop_rect_.top) >> 1) * src_pitch[1] +
                          (disp_rect_.left + crop_rect_.left) * byte_per_pixel_
                    : static_cast<uint8_t*>(src_dev_ptr[1]) +
                          (disp_rect_.top + crop_rect_.top) * src_pitch[1] +
                          (disp_rect_.left + crop_rect_.left) * byte_per_pixel_;
            if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
            {
                if(src_pitch[1] == dst_pitch)
                {
                    int chroma_size = chroma_height_ * dst_pitch;
                    HIP_API_CALL(hipMemcpyDtoDAsync(p_frame_uv, p_src_ptr_uv, chroma_size,
                                                    hip_stream_));
                }
                else
                {
                    // use 2d copy to copy an ROI
                    HIP_API_CALL(hipMemcpy2DAsync(p_frame_uv, dst_pitch, p_src_ptr_uv,
                                                  src_pitch[1], dst_pitch, chroma_height_,
                                                  hipMemcpyDeviceToDevice, hip_stream_));
                }
            }
            else
                HIP_API_CALL(hipMemcpy2DAsync(p_frame_uv, dst_pitch, p_src_ptr_uv,
                                              src_pitch[1], dst_pitch, chroma_height_,
                                              hipMemcpyDeviceToHost, hip_stream_));

            if(num_chroma_planes_ == 2)
            {
                uint8_t* p_frame_v =
                    p_dec_frame + dst_pitch * (disp_height_ + chroma_height_);
                uint8_t* p_src_ptr_v =
                    static_cast<uint8_t*>(src_dev_ptr[2]) +
                    (disp_rect_.top + crop_rect_.top) * src_pitch[2] +
                    (disp_rect_.left + crop_rect_.left) * byte_per_pixel_;
                if(out_mem_type_ == OUT_SURFACE_MEM_DEV_COPIED)
                {
                    if(src_pitch[2] == dst_pitch)
                    {
                        int chroma_size = chroma_height_ * dst_pitch;
                        HIP_API_CALL(hipMemcpyDtoDAsync(p_frame_v, p_src_ptr_v,
                                                        chroma_size, hip_stream_));
                    }
                    else
                    {
                        // use 2d copy to copy an ROI
                        HIP_API_CALL(hipMemcpy2DAsync(
                            p_frame_v, dst_pitch, p_src_ptr_v, src_pitch[2], dst_pitch,
                            chroma_height_, hipMemcpyDeviceToDevice, hip_stream_));
                    }
                }
                else
                    HIP_API_CALL(hipMemcpy2DAsync(p_frame_v, dst_pitch, p_src_ptr_v,
                                                  src_pitch[2], dst_pitch, chroma_height_,
                                                  hipMemcpyDeviceToHost, hip_stream_));
            }

            HIP_API_CALL(hipStreamSynchronize(hip_stream_));
        }
    }
    else
    {
        RocdecDecodeStatus dec_status;
        memset(&dec_status, 0, sizeof(dec_status));
        rocDecStatus result =
            rocDecGetDecodeStatus(roc_decoder_, pDispInfo->picture_index, &dec_status);
        if(result == ROCDEC_SUCCESS &&
           (dec_status.decode_status == rocDecodeStatus_Error ||
            dec_status.decode_status == rocDecodeStatus_Error_Concealed))
        {
            std::cerr << "Decode Error occurred for picture: "
                      << pic_num_in_dec_order_[pDispInfo->picture_index] << std::endl;
        }
        output_frame_cnt_++;
    }

    return 1;
}

int
RocVideoDecoder::GetSEIMessage(RocdecSeiMessageInfo* pSEIMessageInfo)
{
    uint32_t sei_num_mesages = pSEIMessageInfo->sei_message_count;
    if(sei_num_mesages)
    {
        RocdecSeiMessage* p_sei_msg_info      = pSEIMessageInfo->sei_message;
        size_t            total_SEI_buff_size = 0;
        if((pSEIMessageInfo->picIdx < 0) || (pSEIMessageInfo->picIdx >= MAX_FRAME_NUM))
        {
            ERR("Invalid picture index for SEI message: " +
                TOSTR(pSEIMessageInfo->picIdx));
            return 0;
        }
        for(uint32_t i = 0; i < sei_num_mesages; i++)
        {
            total_SEI_buff_size += p_sei_msg_info[i].sei_message_size;
        }
        if(!curr_sei_message_ptr_)
        {
            ERR("Out of Memory, Allocation failed for m_pCurrSEIMessage");
            return 0;
        }
        curr_sei_message_ptr_->sei_data = malloc(total_SEI_buff_size);
        if(!curr_sei_message_ptr_->sei_data)
        {
            ERR("Out of Memory, Allocation failed for SEI Buffer");
            return 0;
        }
        memcpy(curr_sei_message_ptr_->sei_data, pSEIMessageInfo->sei_data,
               total_SEI_buff_size);
        curr_sei_message_ptr_->sei_message =
            (RocdecSeiMessage*) malloc(sizeof(RocdecSeiMessage) * sei_num_mesages);
        if(!curr_sei_message_ptr_->sei_message)
        {
            free(curr_sei_message_ptr_->sei_data);
            curr_sei_message_ptr_->sei_data = NULL;
            return 0;
        }
        memcpy(curr_sei_message_ptr_->sei_message, pSEIMessageInfo->sei_message,
               sizeof(RocdecSeiMessage) * sei_num_mesages);
        curr_sei_message_ptr_->sei_message_count = pSEIMessageInfo->sei_message_count;
        sei_message_display_q_[pSEIMessageInfo->picIdx] = *curr_sei_message_ptr_;
    }
    return 1;
}

int
RocVideoDecoder::DecodeFrame(const uint8_t* data, size_t size, int pkt_flags, int64_t pts,
                             int* num_decoded_pics)
{
    output_frame_cnt_ = 0, output_frame_cnt_ret_ = 0;
    decoded_pic_cnt_              = 0;
    RocdecSourceDataPacket packet = { 0 };
    packet.payload                = data;
    packet.payload_size           = size;
    packet.flags                  = pkt_flags | ROCDEC_PKT_TIMESTAMP;
    packet.pts                    = pts;
    if(!data || size == 0)
    {
        packet.flags |= ROCDEC_PKT_ENDOFSTREAM;
    }
    ROCDEC_API_CALL(rocDecParseVideoData(rocdec_parser_, &packet));
    if(num_decoded_pics)
    {
        *num_decoded_pics = decoded_pic_cnt_;
    }
    return output_frame_cnt_;
}

uint8_t*
RocVideoDecoder::GetFrame(int64_t* pts)
{
    if(output_frame_cnt_ > 0)
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        output_frame_cnt_--;
        if(out_mem_type_ == OUT_SURFACE_MEM_DEV_INTERNAL && !vp_frames_q_.empty())
        {
            DecFrameBuffer* fb = &vp_frames_q_.front();
            if(pts) *pts = fb->pts;
            return fb->frame_ptr;
        }
        else if(vp_frames_.size() > 0)
        {
            if(pts) *pts = vp_frames_[output_frame_cnt_ret_].pts;
            return vp_frames_[output_frame_cnt_ret_++].frame_ptr;
        }
    }
    return nullptr;
}

/**
 * @brief function to release frame after use by the application: Only used with
 * "OUT_SURFACE_MEM_DEV_INTERNAL"
 *
 * @param pTimestamp - timestamp of the frame to be released (unmapped)
 * @return true      - success
 * @return false     - falied
 */

bool
RocVideoDecoder::ReleaseFrame(int64_t pTimestamp, bool b_flushing)
{
    if(out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED) return true;  // nothing to do
    if(out_mem_type_ != OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        if(!b_flushing)   // if not flushing the buffers are re-used, so keep them
            return true;  // nothing to do
        else
        {
            DecFrameBuffer* fb = &vp_frames_[0];
            if(pTimestamp != fb->pts)
            {
                std::cerr << "Decoded Frame is released out of order" << std::endl;
                return false;
            }
            vp_frames_.erase(
                vp_frames_.begin());  // get rid of the frames from the framestore
        }
    }
    // only needed when using internal mapped buffer
    if(!vp_frames_q_.empty())
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        DecFrameBuffer*             fb               = &vp_frames_q_.front();
        void*                       mapped_frame_ptr = fb->frame_ptr;

        if(pTimestamp != fb->pts)
        {
            std::cerr << "Decoded Frame is released out of order" << std::endl;
            return false;
        }
        // pop decoded frame
        vp_frames_q_.pop();
    }
    return true;
}

/**
 * @brief function to release all internal frames and clear the q (used with reconfigure):
 * Only used with "OUT_SURFACE_MEM_DEV_INTERNAL"
 *
 * @return true      - success
 * @return false     - falied
 */
bool
RocVideoDecoder::ReleaseInternalFrames()
{
    if(out_mem_type_ != OUT_SURFACE_MEM_DEV_INTERNAL ||
       out_mem_type_ == OUT_SURFACE_MEM_NOT_MAPPED)
        return true;  // nothing to do
    // only needed when using internal mapped buffer
    while(!vp_frames_q_.empty())
    {
        std::lock_guard<std::mutex> lock(mtx_vp_frame_);
        // pop decoded frame
        vp_frames_q_.pop();
    }
    return true;
}

void
RocVideoDecoder::SaveFrameToFile(std::string output_file_name, void* surf_mem,
                                 OutputSurfaceInfo* surf_info, size_t rgb_image_size)
{
    uint8_t* hst_ptr = nullptr;
    bool     is_rgb  = (rgb_image_size != 0);
    uint64_t output_image_size =
        is_rgb ? rgb_image_size : surf_info->output_surface_size_in_bytes;
    if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL ||
       surf_info->mem_type == OUT_SURFACE_MEM_DEV_COPIED)
    {
        if(hst_ptr == nullptr)
        {
            hst_ptr = new uint8_t[output_image_size];
        }
        hipError_t hip_status = hipSuccess;
        hip_status = hipMemcpyDtoH((void*) hst_ptr, surf_mem, output_image_size);
        if(hip_status != hipSuccess)
        {
            std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hipGetErrorName(hip_status)
                      << ")" << std::endl;
            delete[] hst_ptr;
            return;
        }
    }
    else
        hst_ptr = static_cast<uint8_t*>(surf_mem);

    if(current_output_filename.empty())
    {
        current_output_filename = output_file_name;
    }

    // don't overwrite to the same file if reconfigure is detected for a resolution
    // changes.
    if(is_decoder_reconfigured_)
    {
        if(fp_out_)
        {
            fclose(fp_out_);
            fp_out_ = nullptr;
        }
        // Append the width and height of the new stream to the old file name to create a
        // file name to save the new frames do this only if resolution changes within a
        // stream (e.g., decoding a multi-resolution stream using the videoDecode app)
        // don't append to the output_file_name if multiple output file name is provided
        // (e.g., decoding multi-files using the videDecodeMultiFiles)
        if(!current_output_filename.compare(output_file_name))
        {
            std::string::size_type const pos(output_file_name.find_last_of('.'));
            extra_output_file_count_++;
            std::string to_append = "_" + std::to_string(surf_info->output_width) + "_" +
                                    std::to_string(surf_info->output_height) + "_" +
                                    std::to_string(extra_output_file_count_);
            if(pos != std::string::npos)
            {
                output_file_name.insert(pos, to_append);
            }
            else
            {
                output_file_name += to_append;
            }
        }
        is_decoder_reconfigured_ = false;
    }

    if(fp_out_ == nullptr)
    {
        fp_out_ = fopen(output_file_name.c_str(), "wb");
    }
    if(fp_out_)
    {
        if(!is_rgb)
        {
            uint8_t* tmp_hst_ptr = hst_ptr;
            if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL)
            {
                tmp_hst_ptr +=
                    ((disp_rect_.top + crop_rect_.top) * surf_info->output_pitch) +
                    (disp_rect_.left + crop_rect_.left) * surf_info->bytes_per_pixel;
            }
            int img_width     = surf_info->output_width;
            int img_height    = surf_info->output_height;
            int output_stride = surf_info->output_pitch;
            if(img_width * surf_info->bytes_per_pixel == output_stride &&
               img_height == surf_info->output_vstride)
            {
                fwrite(hst_ptr, 1, output_image_size, fp_out_);
            }
            else
            {
                uint32_t width = surf_info->output_width * surf_info->bytes_per_pixel;
                if(surf_info->bit_depth <= 16)
                {
                    for(int i = 0; i < surf_info->output_height; i++)
                    {
                        fwrite(tmp_hst_ptr, 1, width, fp_out_);
                        tmp_hst_ptr += output_stride;
                    }
                    // dump chroma
                    uint8_t* uv_hst_ptr =
                        hst_ptr + output_stride * surf_info->output_vstride;
                    if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL)
                    {
                        uv_hst_ptr += (num_chroma_planes_ == 1)
                                          ? (((disp_rect_.top + crop_rect_.top) >> 1) *
                                             surf_info->output_pitch) +
                                                ((disp_rect_.left + crop_rect_.left) *
                                                 surf_info->bytes_per_pixel)
                                          : ((disp_rect_.top + crop_rect_.top) *
                                             surf_info->output_pitch) +
                                                ((disp_rect_.left + crop_rect_.left) *
                                                 surf_info->bytes_per_pixel);
                    }
                    for(int i = 0; i < chroma_height_; i++)
                    {
                        fwrite(uv_hst_ptr, 1, width, fp_out_);
                        uv_hst_ptr += output_stride;
                    }
                    if(num_chroma_planes_ == 2)
                    {
                        uv_hst_ptr =
                            hst_ptr +
                            output_stride * (surf_info->output_vstride + chroma_vstride_);
                        if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL)
                        {
                            uv_hst_ptr += ((disp_rect_.top + crop_rect_.top) *
                                           surf_info->output_pitch) +
                                          ((disp_rect_.left + crop_rect_.left) *
                                           surf_info->bytes_per_pixel);
                        }
                        for(int i = 0; i < chroma_height_; i++)
                        {
                            fwrite(uv_hst_ptr, 1, width, fp_out_);
                            uv_hst_ptr += output_stride;
                        }
                    }
                }
            }
        }
        else
        {
            fwrite(hst_ptr, 1, rgb_image_size, fp_out_);
        }
    }

    if(hst_ptr && (surf_info->mem_type != OUT_SURFACE_MEM_HOST_COPIED))
    {
        delete[] hst_ptr;
    }
}

void
RocVideoDecoder::ResetSaveFrameToFile()
{
    if(fp_out_)
    {
        fclose(fp_out_);
        fp_out_ = nullptr;
    }
}

void
RocVideoDecoder::InitMd5()
{
    md5_ctx_ = av_md5_alloc();
    av_md5_init(md5_ctx_);
}

void
RocVideoDecoder::UpdateMd5ForDataBuffer(void* pDevMem, int rgb_image_size)
{
    uint8_t* hstPtr       = nullptr;
    hstPtr                = new uint8_t[rgb_image_size];
    hipError_t hip_status = hipSuccess;
    hip_status            = hipMemcpyDtoH((void*) hstPtr, pDevMem, rgb_image_size);
    if(hip_status != hipSuccess)
    {
        std::cout << "ERROR: hipMemcpyDtoH failed! (" << hip_status << ")" << std::endl;
        delete[] hstPtr;
        return;
    }
    av_md5_update(md5_ctx_, hstPtr, rgb_image_size);
    if(hstPtr)
    {
        delete[] hstPtr;
    }
}

void
RocVideoDecoder::UpdateMd5ForFrame(void* surf_mem, OutputSurfaceInfo* surf_info)
{
    int      i;
    uint8_t* hst_ptr           = nullptr;
    uint64_t output_image_size = surf_info->output_surface_size_in_bytes;
    if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL ||
       surf_info->mem_type == OUT_SURFACE_MEM_DEV_COPIED)
    {
        if(hst_ptr == nullptr)
        {
            hst_ptr = new uint8_t[output_image_size];
        }
        hipError_t hip_status = hipSuccess;
        hip_status = hipMemcpyDtoH((void*) hst_ptr, surf_mem, output_image_size);
        if(hip_status != hipSuccess)
        {
            std::cerr << "ERROR: hipMemcpyDtoH failed! (" << hip_status << ")"
                      << std::endl;
            delete[] hst_ptr;
            return;
        }
    }
    else
        hst_ptr = static_cast<uint8_t*>(surf_mem);

    // Need to covert interleaved planar to stacked planar, assuming 4:2:0 chroma
    // sampling.
    uint8_t* stacked_ptr = new uint8_t[output_image_size];

    uint8_t* tmp_hst_ptr   = hst_ptr;
    int      output_stride = surf_info->output_pitch;
    tmp_hst_ptr +=
        (disp_rect_.top * output_stride) + disp_rect_.left * surf_info->bytes_per_pixel;
    uint8_t* tmp_stacked_ptr = stacked_ptr;
    int      img_width       = surf_info->output_width;
    int      img_height      = surf_info->output_height;
    // Luma
    if(img_width * surf_info->bytes_per_pixel == output_stride &&
       img_height == surf_info->output_vstride)
    {
        memcpy(stacked_ptr, hst_ptr, img_width * surf_info->bytes_per_pixel * img_height);
    }
    else
    {
        for(i = 0; i < img_height; i++)
        {
            memcpy(tmp_stacked_ptr, tmp_hst_ptr, img_width * surf_info->bytes_per_pixel);
            tmp_hst_ptr += output_stride;
            tmp_stacked_ptr += img_width * surf_info->bytes_per_pixel;
        }
    }
    // Chroma
    int img_width_chroma = img_width >> 1;
    tmp_hst_ptr          = hst_ptr + output_stride * surf_info->output_vstride;
    if(surf_info->mem_type == OUT_SURFACE_MEM_DEV_INTERNAL)
    {
        tmp_hst_ptr += ((disp_rect_.top >> 1) * output_stride) +
                       (disp_rect_.left * surf_info->bytes_per_pixel);
    }
    tmp_stacked_ptr =
        stacked_ptr + img_width * surf_info->bytes_per_pixel * img_height;  // Cb
    uint8_t* tmp_stacked_ptr_v = tmp_stacked_ptr + img_width_chroma *
                                                       surf_info->bytes_per_pixel *
                                                       chroma_height_;  // Cr
    for(i = 0; i < chroma_height_; i++)
    {
        for(int j = 0; j < img_width_chroma; j++)
        {
            uint8_t *src_ptr, *dst_ptr;
            // Cb
            src_ptr = &tmp_hst_ptr[j * surf_info->bytes_per_pixel * 2];
            dst_ptr = &tmp_stacked_ptr[j * surf_info->bytes_per_pixel];
            memcpy(dst_ptr, src_ptr, surf_info->bytes_per_pixel);
            // Cr
            src_ptr += surf_info->bytes_per_pixel;
            dst_ptr = &tmp_stacked_ptr_v[j * surf_info->bytes_per_pixel];
            memcpy(dst_ptr, src_ptr, surf_info->bytes_per_pixel);
        }
        tmp_hst_ptr += output_stride;
        tmp_stacked_ptr += img_width_chroma * surf_info->bytes_per_pixel;
        tmp_stacked_ptr_v += img_width_chroma * surf_info->bytes_per_pixel;
    }

    int img_size = img_width * surf_info->bytes_per_pixel * (img_height + chroma_height_);

    // For 10 bit, convert from P010 to little endian to match reference decoder output
    if(surf_info->bytes_per_pixel == 2)
    {
        uint16_t* ptr = reinterpret_cast<uint16_t*>(stacked_ptr);
        for(i = 0; i < img_size / 2; i++)
        {
            ptr[i] = ptr[i] >> 6;
        }
    }

    av_md5_update(md5_ctx_, stacked_ptr, img_size);

    if(hst_ptr && (surf_info->mem_type != OUT_SURFACE_MEM_HOST_COPIED))
    {
        delete[] hst_ptr;
    }
    delete[] stacked_ptr;
}

void
RocVideoDecoder::FinalizeMd5(uint8_t** digest)
{
    av_md5_final(md5_ctx_, md5_digest_);
    av_freep(&md5_ctx_);
    *digest = md5_digest_;
}

void
RocVideoDecoder::GetDeviceinfo(std::string& device_name, std::string& gcn_arch_name,
                               int& pci_bus_id, int& pci_domain_id, int& pci_device_id)
{
    device_name   = hip_dev_prop_.name;
    gcn_arch_name = hip_dev_prop_.gcnArchName;
    pci_bus_id    = hip_dev_prop_.pciBusID;
    pci_domain_id = hip_dev_prop_.pciDomainID;
    pci_device_id = hip_dev_prop_.pciDeviceID;
}

bool
RocVideoDecoder::GetOutputSurfaceInfo(OutputSurfaceInfo** surface_info)
{
    if(!disp_width_ || !disp_height_)
    {
        std::cerr << "ERROR: RocVideoDecoder is not intialized" << std::endl;
        return false;
    }
    *surface_info = &output_surface_info_;
    return true;
}

bool
RocVideoDecoder::InitHIP(int device_id)
{
    HIP_API_CALL(hipGetDeviceCount(&num_devices_));
    if(num_devices_ < 1)
    {
        std::cerr << "ERROR: didn't find any GPU!" << std::endl;
        return false;
    }
    HIP_API_CALL(hipSetDevice(device_id));
    HIP_API_CALL(hipGetDeviceProperties(&hip_dev_prop_, device_id));
    HIP_API_CALL(hipStreamCreate(&hip_stream_));
    return true;
}

std::chrono::_V2::system_clock::time_point
RocVideoDecoder::StartTimer()
{
    return std::chrono::_V2::system_clock::now();
}

double
RocVideoDecoder::StopTimer(const std::chrono::_V2::system_clock::time_point& start_time)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::_V2::system_clock::now() - start_time)
        .count();
}

bool
RocVideoDecoder::CodecSupported(int device_id, rocDecVideoCodec codec_id,
                                uint32_t bit_depth)
{
    RocdecDecodeCaps decode_caps;
    decode_caps.device_id         = device_id;
    decode_caps.codec_type        = codec_id;
    decode_caps.chroma_format     = rocDecVideoChromaFormat_420;
    decode_caps.bit_depth_minus_8 = bit_depth - 8;
    if(rocDecGetDecoderCaps(&decode_caps) != ROCDEC_SUCCESS)
    {
        return false;
    }
    return true;
}

void
RocVideoDecoder::WaitForDecodeCompletion()
{
    RocdecDecodeStatus dec_status;
    memset(&dec_status, 0, sizeof(dec_status));
    do
    {
        rocDecStatus result =
            rocDecGetDecodeStatus(roc_decoder_, last_decode_surf_idx_, &dec_status);
    } while(dec_status.decode_status == rocDecodeStatus_InProgress);
}
