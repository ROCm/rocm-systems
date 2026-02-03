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

#include <assert.h>
#include <stdint.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <string.h>
#include <map>
#include "avcodec/avcodec_videodecoder.h"

/**
 * RocDecoderHost class: Wrapper class for rocDecoderHost API implementation
 */
class RocDecoderHost {
public:
    RocDecoderHost(RocDecoderHostCreateInfo &decoder_create_info);
    ~RocDecoderHost();
    rocDecStatus InitializeDecoder();
    rocDecStatus DecodeFrame(RocdecPicParamsHost *pic_params);
    rocDecStatus GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status);
    rocDecStatus ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params);
    rocDecStatus GetVideoFrame(int pic_idx, void *frame_ptr[3], uint32_t line_size[3], RocdecProcParams *vid_postproc_params);

private:
    AvcodecVideoDecoder avcodec_video_decoder_;
    RocDecoderHostCreateInfo decoder_create_info_;

    RocDecLogger logger_;
};