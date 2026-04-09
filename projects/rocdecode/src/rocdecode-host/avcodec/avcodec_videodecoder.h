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

#pragma once

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libavutil/pixdesc.h>
    #if USE_AVCODEC_GREATER_THAN_58_134
        #include <libavcodec/bsf.h>
    #endif
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <thread>
#include "../src/commons.h"
#include "../api/rocdecode/rocdecode.h"
#include "../api/rocdecode/rocdecode_host.h"


#define MAX_AV_PACKET_DATA_SIZE     4096

typedef struct DecFrameBufferFFMpeg_ {
    AVFrame *av_frame_ptr;      /**< av_frame pointer for the decoded frame */
    uint8_t *frame_ptr;       /**< host memory pointer for the decoded frame depending on mem_type*/
    int64_t  pts;             /**<  timestamp for the decoded frame */
    int picture_index;         /**<  surface index for the decoded frame */
} DecFrameBufferFFMpeg;

typedef struct DecPacketBuffer_
{
    AVPacket *av_pckt;
    int av_frame_index;
} DecPacketBuffer;

typedef struct Rect_{
    int16_t left;
    int16_t top;
    int16_t right;
    int16_t bottom;
} Rect;     

/**
 * Class definition for AVcodec based video decoder class. This uses FFMpeg avcodec decoder to decode video frames
 */
class AvcodecVideoDecoder {
public:
    AvcodecVideoDecoder(RocDecoderHostCreateInfo &decoder_create_info);
    ~AvcodecVideoDecoder();
    rocDecStatus InitializeDecoder();
    rocDecStatus SubmitDecode(RocdecPicParamsHost *pPicParams);
    rocDecStatus GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status);
    rocDecStatus ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params);
    rocDecStatus GetVideoFrame(int pic_idx, void **frame_ptr, uint32_t *line_size, RocdecProcParams *vid_postproc_params);

protected:
    RocDecoderHostCreateInfo decoder_create_info_;
    RocdecVideoFormatHost video_format_host_;
    
    /*! \brief callback function pointers for the parser
     */
    PFNVIDSEQUENCECHOSTALLBACK pfn_sequece_cb_ = nullptr;             /**< Called before decoding frames and/or whenever there is a fmt change */
    PFNVIDDISPLAYCALLBACK pfn_display_picture_cb_ = nullptr;      /**< Called whenever a picture is ready to be displayed (display order)  */
    PFNVIDSEIMSGCALLBACK pfn_get_sei_message_cb_ = nullptr;       /**< Called when all SEI messages are parsed for particular frame        */

private:
    void DecodeThread();
    int DecodeAvFrame(AVPacket *av_pkt, AVFrame *p_frame);
    int FlushDecoder();
    rocDecStatus NotifyNewSequence(AVFrame *p_frame);
    rocDecStatus NotifySeiMesage(AVFrame *p_frame);
    rocDecStatus NotifyPictureDisplay();
    rocDecStatus SendSeiMsgPayload(AVFrame *p_frame);

    void PushPacket(AVPacket *pkt) {
        {
            std::lock_guard<std::mutex> lock(mtx_pkt_q_);
            av_packet_q_.push(pkt);
        }
        cv_pkt_.notify_one();
    }
    
    AVPacket *PopPacket() {
        AVPacket *pkt;
        std::unique_lock<std::mutex> lock(mtx_pkt_q_);
        cv_pkt_.wait(lock, [&] { return !av_packet_q_.empty(); });
        pkt = av_packet_q_.front();
        av_packet_q_.pop();
        return pkt;
    }

    DecFrameBufferFFMpeg *GetDisplayFrame() {
        std::unique_lock<std::mutex> lock(mtx_frame_q_);
        cv_frame_.wait(lock, [&] { return !disp_frames_q_.empty() || end_of_stream_; });
        if (end_of_stream_ && disp_frames_q_.empty())
            return nullptr;
        DecFrameBufferFFMpeg *p_disp_frame = &disp_frames_q_.front();
        disp_frames_q_.pop();
        return p_disp_frame;
    }

    void PushDisplayFrame(DecFrameBufferFFMpeg& frame) {
        {
            std::lock_guard<std::mutex> lock(mtx_frame_q_);
            disp_frames_q_.push(frame);
        }
        cv_frame_.notify_one();
    };

    int decoded_pic_cnt_ = 0;
    int coded_width_ = 0, coded_height_ = 0;        // need to detect resolution changes for sps callback function
    Rect disp_rect_ = {}; // displayable area specified in the bitstream
    int av_sample_format = -1;
    bool b_multithreading_ = true;
    uint32_t av_frame_cnt_ = 0;
    uint32_t av_pkt_cnt_ = 0;
    RocdecSourceDataPacket last_packet_;
    std::thread *ffmpeg_decoder_thread_ = nullptr;
    std::queue<AVPacket *> av_packet_q_;        // queue for compressed packets
    std::queue<DecFrameBufferFFMpeg> disp_frames_q_;      // vector of decoded frames
    std::vector<AVFrame *> dec_frames_;      // vector of AVFrame * for decoded frames
    std::vector<AVPacket *> av_packets_;    // store of AVPackets for decoding
    std::vector<std::pair<uint8_t *, int>> av_packet_data_;
    std::mutex mtx_pkt_q_, mtx_frame_q_;               //for packet and frames
    std::condition_variable cv_pkt_, cv_frame_;
    DecFrameBufferFFMpeg *p_disp_frame_ = nullptr;      // frame to display
    std::atomic<bool> end_of_stream_ = false;
    // Variables for FFMpeg decoding
    AVCodecContext * dec_context_ = nullptr;
    AVPixelFormat decoder_pixel_format_;
#if USE_AVCODEC_GREATER_THAN_58_134 || USE_AVCODEC_GREATER_THAN_60_31
    const AVCodec *decoder_ = nullptr;
#else
    AVCodec *decoder_ = nullptr;
#endif
    AVFormatContext * formatContext = nullptr;
    AVInputFormat * inputFormat = nullptr;

    RocDecLogger logger_;
};
