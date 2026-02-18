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
#include "roc_decoder_host.h"

RocDecoderHost::RocDecoderHost(RocDecoderHostCreateInfo& decoder_create_info): avcodec_video_decoder_{decoder_create_info}, decoder_create_info_{decoder_create_info} {}

RocDecoderHost::~RocDecoderHost() {}

rocDecStatus RocDecoderHost::InitializeDecoder() {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    if (!decoder_create_info_.user_data) {
        logger_.CriticalLog(MakeMsg("Invalid function callback pointer passed"));
        return ROCDEC_NOT_INITIALIZED;
    }
    rocdec_status = avcodec_video_decoder_.InitializeDecoder();
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Failed to initialize the FFMpeg Video decoder."));
        return rocdec_status;
    }
     return rocdec_status;
 }

rocDecStatus RocDecoderHost::DecodeFrame(RocdecPicParamsHost *pic_params) {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = avcodec_video_decoder_.SubmitDecode(pic_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("Decode submission is not successful."));
    }

     return rocdec_status;
}

rocDecStatus RocDecoderHost::GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status) {
    rocDecStatus rocdec_status = ROCDEC_SUCCESS;
    rocdec_status = avcodec_video_decoder_.GetDecodeStatus(pic_idx, decode_status);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("Failed to query the decode status."));
    }
    return rocdec_status;
}

rocDecStatus RocDecoderHost::ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params) {
    if (reconfig_params == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = avcodec_video_decoder_.ReconfigureDecoder(reconfig_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.CriticalLog(MakeMsg("Reconfiguration of the decoder failed."));
        return rocdec_status;
    }
    return rocdec_status;
}

rocDecStatus RocDecoderHost::GetVideoFrame(int pic_idx, void **frame_ptr, uint32_t *line_size, RocdecProcParams *vid_postproc_params) {
    if (vid_postproc_params == nullptr || frame_ptr == nullptr) {
        return ROCDEC_INVALID_PARAMETER;
    }
    rocDecStatus rocdec_status = avcodec_video_decoder_.GetVideoFrame(pic_idx, frame_ptr, line_size, vid_postproc_params);
    if (rocdec_status != ROCDEC_SUCCESS) {
        logger_.ErrorLog(MakeMsg("GetVideoFrame failed."));
        return rocdec_status;
    }
    return rocdec_status;
}