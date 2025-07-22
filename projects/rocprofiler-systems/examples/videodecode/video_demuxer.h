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

#pragma once

#include <iostream>
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#if USE_AVCODEC_GREATER_THAN_58_134
#    include <libavcodec/bsf.h>
#endif
}

#include <rocdecode/rocdecode.h>

/*!
 * \file
 * \brief The AMD Video Demuxer for rocDecode Library.
 *
 * \defgroup group_amd_rocdecode_videodemuxer videoDemuxer: AMD rocDecode Video Demuxer
 * API \brief AMD The rocDecode video demuxer API.
 */

/**
 * @brief Enum for Seek mode
 *
 */
typedef enum SeekModeEnum
{
    SEEK_MODE_EXACT_FRAME    = 0,
    SEEK_MODE_PREV_KEY_FRAME = 1,
    SEEK_MODE_NUM,
} SeekMode;

/**
 * @brief Enum for Seek Criteria
 *
 */
typedef enum SeekCriteriaEnum
{
    SEEK_CRITERIA_FRAME_NUM  = 0,
    SEEK_CRITERIA_TIME_STAMP = 1,
    SEEK_CRITERIA_NUM,
} SeekCriteria;

struct PacketData
{
    int32_t   key;
    int64_t   pts;
    int64_t   dts;
    uint64_t  pos;
    uintptr_t bsl_data;
    uint64_t  bsl;
    uint64_t  duration;
};

class VideoSeekContext
{
public:
    VideoSeekContext()
    : use_seek_(false)
    , seek_frame_(0)
    , seek_mode_(SEEK_MODE_PREV_KEY_FRAME)
    , seek_crit_(SEEK_CRITERIA_FRAME_NUM)
    , out_frame_pts_(0)
    , out_frame_duration_(0)
    , num_frames_decoded_(0U)
    {}

    VideoSeekContext(uint64_t frame_id)
    : use_seek_(true)
    , seek_frame_(frame_id)
    , seek_mode_(SEEK_MODE_PREV_KEY_FRAME)
    , seek_crit_(SEEK_CRITERIA_FRAME_NUM)
    , out_frame_pts_(0)
    , out_frame_duration_(0)
    , num_frames_decoded_(0U)
    {}

    VideoSeekContext& operator=(const VideoSeekContext& other)
    {
        use_seek_           = other.use_seek_;
        seek_frame_         = other.seek_frame_;
        seek_mode_          = other.seek_mode_;
        seek_crit_          = other.seek_crit_;
        out_frame_pts_      = other.out_frame_pts_;
        out_frame_duration_ = other.out_frame_duration_;
        num_frames_decoded_ = other.num_frames_decoded_;
        return *this;
    }

    /* Will be set to false when not seeking, true otherwise;
     */
    bool use_seek_;

    /* Frame we want to get. Set by user.
     * Shall be set to frame timestamp in case seek is done by time.
     */
    uint64_t seek_frame_;

    /* Mode in which we seek. */
    SeekMode seek_mode_;

    /* Criteria by which we seek. */
    SeekCriteria seek_crit_;

    /* PTS of frame found after seek. */
    int64_t out_frame_pts_;

    /* Duration of frame found after seek. */
    int64_t out_frame_duration_;

    /* Number of frames that were decoded during seek. */
    uint64_t num_frames_decoded_;
};

// Video Demuxer Interface class
class VideoDemuxer
{
public:
    class StreamProvider
    {
    public:
        virtual ~StreamProvider() {}
        virtual int    GetData(uint8_t* buf, int buf_size) = 0;
        virtual size_t GetBufferSize()                     = 0;
    };
    AVCodecID GetCodecID() { return av_video_codec_id_; };
    VideoDemuxer(const char* input_file_path)
    : VideoDemuxer(CreateFmtContextUtil(input_file_path))
    {}
    VideoDemuxer(StreamProvider* stream_provider)
    : VideoDemuxer(CreateFmtContextUtil(stream_provider))
    {
        av_io_ctx_ = av_fmt_input_ctx_->pb;
    }
    ~VideoDemuxer()
    {
        if(!av_fmt_input_ctx_)
        {
            return;
        }
        if(packet_)
        {
            av_packet_free(&packet_);
        }
        if(packet_filtered_)
        {
            av_packet_free(&packet_filtered_);
        }
        if(av_bsf_ctx_)
        {
            av_bsf_free(&av_bsf_ctx_);
        }
        avformat_close_input(&av_fmt_input_ctx_);
        if(av_io_ctx_)
        {
            av_freep(&av_io_ctx_->buffer);
            av_freep(&av_io_ctx_);
        }
        if(data_with_header_)
        {
            av_free(data_with_header_);
        }
    }
    bool Demux(uint8_t** video, int* video_size, int64_t* pts = nullptr)
    {
        if(!av_fmt_input_ctx_)
        {
            return false;
        }
        *video_size = 0;
        if(packet_->data)
        {
            av_packet_unref(packet_);
        }
        int ret = 0;
        while((ret = av_read_frame(av_fmt_input_ctx_, packet_)) >= 0 &&
              packet_->stream_index != av_stream_)
        {
            av_packet_unref(packet_);
        }
        if(ret < 0)
        {
            return false;
        }
        if(is_h264_ || is_hevc_)
        {
            if(packet_filtered_->data)
            {
                av_packet_unref(packet_filtered_);
            }
            if(av_bsf_send_packet(av_bsf_ctx_, packet_) != 0)
            {
                std::cerr << "ERROR: av_bsf_send_packet failed!" << std::endl;
                return false;
            }
            if(av_bsf_receive_packet(av_bsf_ctx_, packet_filtered_) != 0)
            {
                std::cerr << "ERROR: av_bsf_receive_packet failed!" << std::endl;
                return false;
            }
            *video      = packet_filtered_->data;
            *video_size = packet_filtered_->size;
            if(packet_filtered_->dts != AV_NOPTS_VALUE)
            {
                pkt_dts_ = packet_filtered_->dts;
            }
            else
            {
                pkt_dts_ = packet_filtered_->pts;
            }
            if(pts)
            {
                *pts =
                    (int64_t) (packet_filtered_->pts * default_time_scale_ * time_base_);
                pkt_duration_ = packet_filtered_->duration;
            }
        }
        else
        {
            if(is_mpeg4_ && (frame_count_ == 0))
            {
                int ext_data_size =
                    av_fmt_input_ctx_->streams[av_stream_]->codecpar->extradata_size;
                if(ext_data_size > 0)
                {
                    data_with_header_ = (uint8_t*) av_malloc(
                        ext_data_size + packet_->size - 3 * sizeof(uint8_t));
                    if(!data_with_header_)
                    {
                        std::cerr << "ERROR: av_malloc failed!" << std::endl;
                        return false;
                    }
                    memcpy(data_with_header_,
                           av_fmt_input_ctx_->streams[av_stream_]->codecpar->extradata,
                           ext_data_size);
                    memcpy(data_with_header_ + ext_data_size, packet_->data + 3,
                           packet_->size - 3 * sizeof(uint8_t));
                    *video      = data_with_header_;
                    *video_size = ext_data_size + packet_->size - 3 * sizeof(uint8_t);
                }
            }
            else
            {
                *video      = packet_->data;
                *video_size = packet_->size;
            }
            if(packet_->dts != AV_NOPTS_VALUE)
            {
                pkt_dts_ = packet_->dts;
            }
            else
            {
                pkt_dts_ = packet_->pts;
            }
            if(pts)
            {
                *pts = (int64_t) (packet_->pts * default_time_scale_ * time_base_);
                pkt_duration_ = packet_->duration;
            }
        }
        frame_count_++;
        return true;
    }
    bool Seek(VideoSeekContext& seek_ctx, uint8_t** pp_video, int* video_size)
    {
        /* !!! IMPORTANT !!!
         * Across this function, packet decode timestamp (DTS) values are used to
         * compare given timestamp against. This is done because DTS values shall
         * monotonically increase during the course of decoding unlike PTS values
         * which may be affected by frame reordering due to B frames.
         */

        if(!is_seekable_)
        {
            std::cerr << "ERROR: Seek isn't supported for this input." << std::endl;
            return false;
        }

        if(IsVFR() && (SEEK_CRITERIA_FRAME_NUM == seek_ctx.seek_crit_))
        {
            std::cerr << "ERROR: Can't seek by frame number in VFR sequences. Seek by "
                         "timestamp instead."
                      << std::endl;
            return false;
        }

        // Seek for single frame;
        auto seek_frame = [&](VideoSeekContext const& seek_ctx, int flags) {
            bool    seek_backward = true;
            int64_t timestamp     = 0;
            int     ret           = 0;

            switch(seek_ctx.seek_crit_)
            {
                case SEEK_CRITERIA_FRAME_NUM:
                    timestamp = TsFromFrameNumber(seek_ctx.seek_frame_);
                    ret       = av_seek_frame(av_fmt_input_ctx_, av_stream_, timestamp,
                                        seek_backward ? AVSEEK_FLAG_BACKWARD | flags
                                                            : flags);
                    break;
                case SEEK_CRITERIA_TIME_STAMP:
                    timestamp = TsFromTime(seek_ctx.seek_frame_);
                    ret       = av_seek_frame(av_fmt_input_ctx_, av_stream_, timestamp,
                                        seek_backward ? AVSEEK_FLAG_BACKWARD | flags
                                                            : flags);
                    break;
                default: std::cerr << "ERROR: Invalid seek mode" << std::endl; ret = -1;
            }

            if(ret < 0)
            {
                throw std::runtime_error("ERROR: seeking for frame");
            }
        };

        // Check if frame satisfies seek conditions;
        auto is_seek_done = [&](PacketData& pkt_data, VideoSeekContext const& seek_ctx) {
            int64_t target_ts = 0;

            switch(seek_ctx.seek_crit_)
            {
                case SEEK_CRITERIA_FRAME_NUM:
                    target_ts = TsFromFrameNumber(seek_ctx.seek_frame_);
                    break;
                case SEEK_CRITERIA_TIME_STAMP:
                    target_ts = TsFromTime(seek_ctx.seek_frame_);
                    break;
                default:
                    std::cerr << "ERROR::Invalid seek criteria" << std::endl;
                    return -1;
            }

            if(pkt_dts_ == target_ts)
            {
                return 0;
            }
            else if(pkt_dts_ > target_ts)
            {
                return 1;
            }
            else
            {
                return -1;
            };
        };

        /* This will seek for exact frame number;
         * Note that decoder may not be able to decode such frame; */
        auto seek_for_exact_frame = [&](PacketData&       pkt_data,
                                        VideoSeekContext& seek_ctx) {
            // Repetititive seek until seek condition is satisfied;
            VideoSeekContext tmp_ctx(seek_ctx.seek_frame_);
            seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

            int seek_done = 0;
            do
            {
                if(!Demux(pp_video, video_size, &pkt_data.pts))
                {
                    throw std::runtime_error("ERROR: Demux failed trying to seek for "
                                             "specified frame number/timestamp");
                }
                seek_done = is_seek_done(pkt_data, seek_ctx);
                // TODO: one last condition, check for a target too high than available
                // for timestamp
                if(seek_done > 0)
                {  // We've gone too far and need to seek backwards;
                    if((tmp_ctx.seek_frame_--) >= 0)
                    {
                        seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
                    }
                }
                else if(seek_done < 0)
                {  // Need to read more frames until we reach requested number;
                    tmp_ctx.seek_frame_++;
                    seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
                }
                if(tmp_ctx.seek_frame_ ==
                   seek_ctx.seek_frame_)  // if frame 'N' is too far and frame 'N-1' is
                                          // too less from target. Avoids infinite loop
                                          // between N & N-1
                    break;
            } while(seek_done != 0);

            seek_ctx.out_frame_pts_      = pkt_data.pts;
            seek_ctx.out_frame_duration_ = pkt_data.duration = pkt_duration_;
        };

        // Seek for closest key frame in the past;
        auto seek_for_prev_key_frame = [&](PacketData&       pkt_data,
                                           VideoSeekContext& seek_ctx) {
            seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);
            Demux(pp_video, video_size, &pkt_data.pts);
            seek_ctx.num_frames_decoded_ =
                static_cast<uint64_t>(pkt_data.pts / 1000 * frame_rate_);
            seek_ctx.out_frame_pts_      = pkt_data.pts;
            seek_ctx.out_frame_duration_ = pkt_data.duration = pkt_duration_;
        };

        PacketData pktData;
        pktData.bsl_data = size_t(*pp_video);
        pktData.bsl      = *video_size;

        switch(seek_ctx.seek_mode_)
        {
            case SEEK_MODE_EXACT_FRAME: seek_for_exact_frame(pktData, seek_ctx); break;
            case SEEK_MODE_PREV_KEY_FRAME:
                seek_for_prev_key_frame(pktData, seek_ctx);
                break;
            default: throw std::runtime_error("ERROR::Unsupported seek mode"); break;
        }

        return true;
    }
    const uint32_t GetWidth() const { return width_; }
    const uint32_t GetHeight() const { return height_; }
    const uint32_t GetChromaHeight() const { return chroma_height_; }
    const uint32_t GetBitDepth() const { return bit_depth_; }
    const uint32_t GetBytePerPixel() const { return byte_per_pixel_; }
    const uint32_t GetBitRate() const { return bit_rate_; }
    const double   GetFrameRate() const { return frame_rate_; };
    bool           IsVFR() const { return frame_rate_ != avg_frame_rate_; };
    int64_t        TsFromTime(double ts_sec)
    {
        // Convert integer timestamp representation to AV_TIME_BASE and switch to
        // fixed_point
        auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);
        // Rescale the timestamp to value represented in stream base units;
        AVRational time_factor = { 1, AV_TIME_BASE };
        return av_rescale_q(ts_tbu, time_factor,
                            av_fmt_input_ctx_->streams[av_stream_]->time_base);
    }

    int64_t TsFromFrameNumber(int64_t frame_num)
    {
        auto const ts_sec = static_cast<double>(frame_num) / frame_rate_;
        return TsFromTime(ts_sec);
    }

private:
    VideoDemuxer(AVFormatContext* av_fmt_input_ctx)
    : av_fmt_input_ctx_(av_fmt_input_ctx)
    {
        av_log_set_level(AV_LOG_QUIET);
        if(!av_fmt_input_ctx_)
        {
            std::cerr << "ERROR: av_fmt_input_ctx_ is not vaild!" << std::endl;
            return;
        }
        packet_          = av_packet_alloc();
        packet_filtered_ = av_packet_alloc();
        if(!packet_ || !packet_filtered_)
        {
            std::cerr << "ERROR: av_packet_alloc failed!" << std::endl;
            return;
        }
        if(avformat_find_stream_info(av_fmt_input_ctx_, nullptr) < 0)
        {
            std::cerr << "ERROR: avformat_find_stream_info failed!" << std::endl;
            return;
        }
        av_stream_ = av_find_best_stream(av_fmt_input_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1,
                                         nullptr, 0);
        if(av_stream_ < 0)
        {
            std::cerr << "ERROR: av_find_best_stream failed!" << std::endl;
            av_packet_free(&packet_);
            av_packet_free(&packet_filtered_);
            return;
        }
        av_video_codec_id_ = av_fmt_input_ctx_->streams[av_stream_]->codecpar->codec_id;
        width_             = av_fmt_input_ctx_->streams[av_stream_]->codecpar->width;
        height_            = av_fmt_input_ctx_->streams[av_stream_]->codecpar->height;
        chroma_format_ =
            (AVPixelFormat) av_fmt_input_ctx_->streams[av_stream_]->codecpar->format;
        bit_rate_ = av_fmt_input_ctx_->streams[av_stream_]->codecpar->bit_rate;
        if(av_fmt_input_ctx_->streams[av_stream_]->r_frame_rate.den != 0)
            frame_rate_ = static_cast<double>(
                              av_fmt_input_ctx_->streams[av_stream_]->r_frame_rate.num) /
                          static_cast<double>(
                              av_fmt_input_ctx_->streams[av_stream_]->r_frame_rate.den);
        if(av_fmt_input_ctx_->streams[av_stream_]->avg_frame_rate.den != 0)
            avg_frame_rate_ =
                static_cast<double>(
                    av_fmt_input_ctx_->streams[av_stream_]->avg_frame_rate.num) /
                static_cast<double>(
                    av_fmt_input_ctx_->streams[av_stream_]->avg_frame_rate.den);

        switch(chroma_format_)
        {
            case AV_PIX_FMT_YUV420P10LE:
            case AV_PIX_FMT_GRAY10LE:
                bit_depth_      = 10;
                chroma_height_  = (height_ + 1) >> 1;
                byte_per_pixel_ = 2;
                break;
            case AV_PIX_FMT_YUV420P12LE:
                bit_depth_      = 12;
                chroma_height_  = (height_ + 1) >> 1;
                byte_per_pixel_ = 2;
                break;
            case AV_PIX_FMT_YUV444P10LE:
                bit_depth_      = 10;
                chroma_height_  = height_ << 1;
                byte_per_pixel_ = 2;
                break;
            case AV_PIX_FMT_YUV444P12LE:
                bit_depth_      = 12;
                chroma_height_  = height_ << 1;
                byte_per_pixel_ = 2;
                break;
            case AV_PIX_FMT_YUV444P:
                bit_depth_      = 8;
                chroma_height_  = height_ << 1;
                byte_per_pixel_ = 1;
                break;
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUVJ420P:
            case AV_PIX_FMT_YUVJ422P:
            case AV_PIX_FMT_YUVJ444P:
            case AV_PIX_FMT_GRAY8:
                bit_depth_      = 8;
                chroma_height_  = (height_ + 1) >> 1;
                byte_per_pixel_ = 1;
                break;
            default:
                chroma_format_  = AV_PIX_FMT_YUV420P;
                bit_depth_      = 8;
                chroma_height_  = (height_ + 1) >> 1;
                byte_per_pixel_ = 1;
        }

        AVRational time_base = av_fmt_input_ctx_->streams[av_stream_]->time_base;
        time_base_           = av_q2d(time_base);

        is_h264_ = av_video_codec_id_ == AV_CODEC_ID_H264 &&
                   (!strcmp(av_fmt_input_ctx_->iformat->long_name, "QuickTime / MOV") ||
                    !strcmp(av_fmt_input_ctx_->iformat->long_name, "FLV (Flash Video)") ||
                    !strcmp(av_fmt_input_ctx_->iformat->long_name, "Matroska / WebM"));
        is_hevc_ = av_video_codec_id_ == AV_CODEC_ID_HEVC &&
                   (!strcmp(av_fmt_input_ctx_->iformat->long_name, "QuickTime / MOV") ||
                    !strcmp(av_fmt_input_ctx_->iformat->long_name, "FLV (Flash Video)") ||
                    !strcmp(av_fmt_input_ctx_->iformat->long_name, "Matroska / WebM"));
        is_mpeg4_ =
            av_video_codec_id_ == AV_CODEC_ID_MPEG4 &&
            (!strcmp(av_fmt_input_ctx_->iformat->long_name, "QuickTime / MOV") ||
             !strcmp(av_fmt_input_ctx_->iformat->long_name, "FLV (Flash Video)") ||
             !strcmp(av_fmt_input_ctx_->iformat->long_name, "Matroska / WebM"));

        // Check if the input file allow seek functionality.
        is_seekable_ = av_fmt_input_ctx_->iformat->read_seek ||
                       av_fmt_input_ctx_->iformat->read_seek2;

        if(is_h264_)
        {
            const AVBitStreamFilter* bsf = av_bsf_get_by_name("h264_mp4toannexb");
            if(!bsf)
            {
                std::cerr << "ERROR: av_bsf_get_by_name() failed" << std::endl;
                av_packet_free(&packet_);
                av_packet_free(&packet_filtered_);
                return;
            }
            if(av_bsf_alloc(bsf, &av_bsf_ctx_) != 0)
            {
                std::cerr << "ERROR: av_bsf_alloc failed!" << std::endl;
                return;
            }
            avcodec_parameters_copy(av_bsf_ctx_->par_in,
                                    av_fmt_input_ctx_->streams[av_stream_]->codecpar);
            if(av_bsf_init(av_bsf_ctx_) < 0)
            {
                std::cerr << "ERROR: av_bsf_init failed!" << std::endl;
                return;
            }
        }
        if(is_hevc_)
        {
            const AVBitStreamFilter* bsf = av_bsf_get_by_name("hevc_mp4toannexb");
            if(!bsf)
            {
                std::cerr << "ERROR: av_bsf_get_by_name() failed" << std::endl;
                av_packet_free(&packet_);
                av_packet_free(&packet_filtered_);
                return;
            }
            if(av_bsf_alloc(bsf, &av_bsf_ctx_) != 0)
            {
                std::cerr << "ERROR: av_bsf_alloc failed!" << std::endl;
                return;
            }
            avcodec_parameters_copy(av_bsf_ctx_->par_in,
                                    av_fmt_input_ctx_->streams[av_stream_]->codecpar);
            if(av_bsf_init(av_bsf_ctx_) < 0)
            {
                std::cerr << "ERROR: av_bsf_init failed!" << std::endl;
                return;
            }
        }
    }
    AVFormatContext* CreateFmtContextUtil(StreamProvider* stream_provider)
    {
        AVFormatContext* ctx = nullptr;
        if(!(ctx = avformat_alloc_context()))
        {
            std::cerr << "ERROR: avformat_alloc_context failed" << std::endl;
            return nullptr;
        }
        uint8_t* avioc_buffer      = nullptr;
        int      avioc_buffer_size = stream_provider->GetBufferSize();
        avioc_buffer               = (uint8_t*) av_malloc(avioc_buffer_size);
        if(!avioc_buffer)
        {
            std::cerr << "ERROR: av_malloc failed!" << std::endl;
            return nullptr;
        }
        av_io_ctx_ = avio_alloc_context(avioc_buffer, avioc_buffer_size, 0,
                                        stream_provider, &ReadPacket, nullptr, nullptr);
        if(!av_io_ctx_)
        {
            std::cerr << "ERROR: avio_alloc_context failed!" << std::endl;
            return nullptr;
        }
        ctx->pb = av_io_ctx_;

        if(avformat_open_input(&ctx, nullptr, nullptr, nullptr) != 0)
        {
            std::cerr << "ERROR: avformat_open_input failed!" << std::endl;
            return nullptr;
        }
        return ctx;
    }
    AVFormatContext* CreateFmtContextUtil(const char* input_file_path)
    {
        avformat_network_init();
        AVFormatContext* ctx = nullptr;
        if(avformat_open_input(&ctx, input_file_path, nullptr, nullptr) != 0)
        {
            std::cerr << "ERROR: avformat_open_input failed!" << std::endl;
            return nullptr;
        }
        return ctx;
    }
    static int ReadPacket(void* data, uint8_t* buf, int buf_size)
    {
        return ((StreamProvider*) data)->GetData(buf, buf_size);
    }
    AVFormatContext* av_fmt_input_ctx_ = nullptr;
    AVIOContext*     av_io_ctx_        = nullptr;
    AVPacket*        packet_           = nullptr;
    AVPacket*        packet_filtered_  = nullptr;
    AVBSFContext*    av_bsf_ctx_       = nullptr;
    AVCodecID        av_video_codec_id_;
    AVPixelFormat    chroma_format_;
    double           frame_rate_         = 0.0;
    double           avg_frame_rate_     = 0.0;
    uint8_t*         data_with_header_   = nullptr;
    int              av_stream_          = 0;
    bool             is_h264_            = false;
    bool             is_hevc_            = false;
    bool             is_mpeg4_           = false;
    bool             is_seekable_        = false;
    int64_t          default_time_scale_ = 1000;
    double           time_base_          = 0.0;
    uint32_t         frame_count_        = 0;
    uint32_t         width_              = 0;
    uint32_t         height_             = 0;
    uint32_t         chroma_height_      = 0;
    uint32_t         bit_depth_          = 0;
    uint32_t         byte_per_pixel_     = 0;
    uint32_t         bit_rate_           = 0;
    // used for Seek Exact frame
    int64_t pkt_dts_      = 0;
    int64_t pkt_duration_ = 0;
};

static inline rocDecVideoCodec
AVCodec2RocDecVideoCodec(AVCodecID av_codec)
{
    switch(av_codec)
    {
        case AV_CODEC_ID_MPEG1VIDEO: return rocDecVideoCodec_MPEG1;
        case AV_CODEC_ID_MPEG2VIDEO: return rocDecVideoCodec_MPEG2;
        case AV_CODEC_ID_MPEG4: return rocDecVideoCodec_MPEG4;
        case AV_CODEC_ID_H264: return rocDecVideoCodec_AVC;
        case AV_CODEC_ID_HEVC: return rocDecVideoCodec_HEVC;
        case AV_CODEC_ID_VP8: return rocDecVideoCodec_VP8;
        case AV_CODEC_ID_VP9: return rocDecVideoCodec_VP9;
        case AV_CODEC_ID_MJPEG: return rocDecVideoCodec_JPEG;
        case AV_CODEC_ID_AV1: return rocDecVideoCodec_AV1;
        default: return rocDecVideoCodec_NumCodecs;
    }
}
