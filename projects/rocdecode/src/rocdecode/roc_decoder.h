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
#include "../api/rocdecode/rocdecode.h"
#include <hip/hip_runtime.h>
#include "vaapi/vaapi_videodecoder.h"

struct HipInteropDeviceMem {
    hipExternalMemory_t hip_ext_mem; // Interface to the vaapi-hip interop
    uint8_t* hip_mapped_device_mem; // Mapped device memory for the YUV plane
    uint32_t width; // Width of the surface in pixels.
    uint32_t height; // Height of the surface in pixels.
    uint32_t offset[3]; // Offset of each plane
    uint32_t pitch[3]; // Pitch of each plane
    uint32_t num_layers; // Number of layers making up the surface
};

class RocDecoder {
public:
    RocDecoder(RocDecoderCreateInfo &decoder_create_info);
    ~RocDecoder();
    rocDecStatus InitializeDecoder();
    rocDecStatus DecodeFrame(RocdecPicParams *pic_params);
    rocDecStatus GetDecodeStatus(int pic_idx, RocdecDecodeStatus* decode_status);
    rocDecStatus ReconfigureDecoder(RocdecReconfigureDecoderInfo *reconfig_params);
    rocDecStatus GetVideoFrame(int pic_idx, void *dev_mem_ptr[3], uint32_t horizontal_pitch[3], RocdecProcParams *vid_postproc_params);

private:
    rocDecStatus FreeVideoFrame(int pic_idx);
    VaapiVideoDecoder va_video_decoder_;
    RocDecoderCreateInfo decoder_create_info_;
    std::vector<HipInteropDeviceMem> hip_interop_;

    RocDecLogger logger_;
};