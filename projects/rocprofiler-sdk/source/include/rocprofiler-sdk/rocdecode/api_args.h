// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/rocdecode/api_trace.h>

#include <rocprofiler-sdk/rocdecode/details/rocdecode_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_ROCDECODE > 0
#    include <rocdecode/roc_bitstream_reader.h>
#    include <rocdecode/rocdecode.h>
#    include <rocdecode/rocparser.h>
#else
#    include <rocprofiler-sdk/rocdecode/details/roc_bitstream_reader.h>
#    include <rocprofiler-sdk/rocdecode/details/rocdecode.h>
#    include <rocprofiler-sdk/rocdecode/details/rocparser.h>
#endif

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_rocdecode_api_no_args
{
    char empty;
} rocprofiler_rocdecode_api_no_args;

typedef union rocprofiler_rocdecode_api_retval_t
{
    uint64_t    uint64_t_retval;
    int32_t     rocDecStatus_retval;
    const char* const_charp_retval;
} rocprofiler_rocdecode_api_retval_t;

typedef union rocprofiler_rocdecode_api_args_t
{
    struct
    {
        RocdecVideoParser*  parser_handle;
        RocdecParserParams* params;
    } rocDecCreateVideoParser;

    struct
    {
        RocdecVideoParser       parser_handle;
        RocdecSourceDataPacket* packet;
    } rocDecParseVideoData;

    struct
    {
        RocdecVideoParser parser_handle;
    } rocDecDestroyVideoParser;

    struct
    {
        rocDecDecoderHandle*  decoder_handle;
        RocDecoderCreateInfo* decoder_create_info;
    } rocDecCreateDecoder;

    struct
    {
        rocDecDecoderHandle decoder_handle;
    } rocDecDestroyDecoder;

    struct
    {
        RocdecDecodeCaps* decode_caps;
    } rocDecGetDecoderCaps;

    struct
    {
        rocDecDecoderHandle decoder_handle;
        RocdecPicParams*    pic_params;
    } rocDecDecodeFrame;

    struct
    {
        rocDecDecoderHandle decoder_handle;
        int                 pic_idx;
        RocdecDecodeStatus* decode_status;
    } rocDecGetDecodeStatus;

    struct
    {
        rocDecDecoderHandle           decoder_handle;
        RocdecReconfigureDecoderInfo* reconfig_params;
    } rocDecReconfigureDecoder;

    struct
    {
        rocDecDecoderHandle decoder_handle;
        int                 pic_idx;
        void**              dev_mem_ptr;
        uint32_t*           horizontal_pitch;
        RocdecProcParams*   vid_postproc_params;
    } rocDecGetVideoFrame;
    struct
    {
        rocDecStatus rocdec_status;
    } rocDecGetErrorName;

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION >= 1
    struct
    {
        RocdecBitstreamReader* bs_reader_handle;
        const char*            input_file_path;
    } rocDecCreateBitstreamReader;
    struct
    {
        RocdecBitstreamReader bs_reader_handle;
        rocDecVideoCodec*     codec_type;
    } rocDecGetBitstreamCodecType;
    struct
    {
        RocdecBitstreamReader bs_reader_handle;
        int*                  bit_depth;
    } rocDecGetBitstreamBitDepth;
    struct
    {
        RocdecBitstreamReader bs_reader_handle;
        uint8_t**             pic_data;
        int*                  pic_size;
        int64_t*              pts;
    } rocDecGetBitstreamPicData;
    struct
    {
        RocdecBitstreamReader bs_reader_handle;
    } rocDecDestroyBitstreamReader;
#endif
} rocprofiler_rocdecode_api_args_t;

ROCPROFILER_EXTERN_C_FINI
