/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

// Define version macros for the rocDecode API dispatch table, specifying the MAJOR and STEP
// versions.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     IMPORTANT    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// 1. When adding new functions to the rocDecode API dispatch table, always append the new function
// pointer
//    to the end of the table and increment the dispatch table's version number. Never rearrange the
//    order of the member variables in the dispatch table, as doing so will break the Application
//    Binary Interface (ABI).
// 2. In critical situations where the type of an existing member variable in a dispatch table has
// been changed
//    or removed due to a data type modification, it is important to increment the major version
//    number of the rocDecode API dispatch table. If the function pointer type can no longer be
//    declared, do not remove it. Instead, change the function pointer type to `void*` and ensure it
//    is always initialized to `nullptr`.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//

// The major version number should ideally remain unchanged. Increment the
// ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION only for fundamental changes to the
// rocDecodeDispatchTable struct, such as altering the type or name of an existing member variable.
// Please DO NOT REMOVE it.
#define ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION 0

// Increment the ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION when new runtime API functions are added.
// If the corresponding ROCDECODE_RUNTIME_API_TABLE_MAJOR_VERSION increases reset the
// ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION to zero.
#define ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION 1

// rocDecode API interface
typedef rocDecStatus(ROCDECAPI* PfnRocDecCreateVideoParser)(RocdecVideoParser*  parser_handle,
                                                            RocdecParserParams* params);
typedef rocDecStatus(ROCDECAPI* PfnRocDecParseVideoData)(RocdecVideoParser       parser_handle,
                                                         RocdecSourceDataPacket* packet);
typedef rocDecStatus(ROCDECAPI* PfnRocDecDestroyVideoParser)(RocdecVideoParser parser_handle);
typedef rocDecStatus(ROCDECAPI* PfnRocDecCreateDecoder)(rocDecDecoderHandle*  decoder_handle,
                                                        RocDecoderCreateInfo* decoder_create_info);
typedef rocDecStatus(ROCDECAPI* PfnRocDecDestroyDecoder)(rocDecDecoderHandle decoder_handle);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetDecoderCaps)(RocdecDecodeCaps* decode_caps);
typedef rocDecStatus(ROCDECAPI* PfnRocDecDecodeFrame)(rocDecDecoderHandle decoder_handle,
                                                      RocdecPicParams*    pic_params);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetDecodeStatus)(rocDecDecoderHandle decoder_handle,
                                                          int                 pic_idx,
                                                          RocdecDecodeStatus* decode_status);
typedef rocDecStatus(ROCDECAPI* PfnRocDecReconfigureDecoder)(
    rocDecDecoderHandle           decoder_handle,
    RocdecReconfigureDecoderInfo* reconfig_params);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetVideoFrame)(rocDecDecoderHandle decoder_handle,
                                                        int                 pic_idx,
                                                        void*               dev_mem_ptr[3],
                                                        uint32_t*           horizontal_pitch,
                                                        RocdecProcParams*   vid_postproc_params);
typedef const char*(ROCDECAPI* PfnRocDecGetErrorName)(rocDecStatus rocdec_status);
typedef rocDecStatus(ROCDECAPI* PfnRocDecCreateBitstreamReader)(
    RocdecBitstreamReader* bs_reader_handle,
    const char*            input_file_path);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetBitstreamCodecType)(
    RocdecBitstreamReader bs_reader_handle,
    rocDecVideoCodec*     codec_type);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetBitstreamBitDepth)(
    RocdecBitstreamReader bs_reader_handle,
    int*                  bit_depth);
typedef rocDecStatus(ROCDECAPI* PfnRocDecGetBitstreamPicData)(
    RocdecBitstreamReader bs_reader_handle,
    uint8_t**             pic_data,
    int*                  pic_size,
    int64_t*              pts);
typedef rocDecStatus(ROCDECAPI* PfnRocDecDestroyBitstreamReader)(
    RocdecBitstreamReader bs_reader_handle);

// rocDecode API dispatch table
struct RocDecodeDispatchTable
{
    // ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 0
    size_t                      size;
    PfnRocDecCreateVideoParser  pfn_rocdec_create_video_parser;
    PfnRocDecParseVideoData     pfn_rocdec_parse_video_data;
    PfnRocDecDestroyVideoParser pfn_rocdec_destroy_video_parser;
    PfnRocDecCreateDecoder      pfn_rocdec_create_decoder;
    PfnRocDecDestroyDecoder     pfn_rocdec_destroy_decoder;
    PfnRocDecGetDecoderCaps     pfn_rocdec_get_gecoder_caps;
    PfnRocDecDecodeFrame        pfn_rocdec_decode_frame;
    PfnRocDecGetDecodeStatus    pfn_rocdec_get_decode_status;
    PfnRocDecReconfigureDecoder pfn_rocdec_reconfigure_decoder;
    PfnRocDecGetVideoFrame      pfn_rocdec_get_video_frame;
    PfnRocDecGetErrorName       pfn_rocdec_get_error_name;
    // PLEASE DO NOT EDIT ABOVE!
    // ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 1
    PfnRocDecCreateBitstreamReader  pfn_rocdec_create_bitstream_reader;
    PfnRocDecGetBitstreamCodecType  pfn_rocdec_get_bitstream_codec_type;
    PfnRocDecGetBitstreamBitDepth   pfn_rocdec_get_bitstream_bit_depth;
    PfnRocDecGetBitstreamPicData    pfn_rocdec_get_bitstream_pic_data;
    PfnRocDecDestroyBitstreamReader pfn_rocdec_destroy_bitstream_reader;
    // PLEASE DO NOT EDIT ABOVE!
    // ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION == 2

    // *******************************************************************************************
    // //
    //                                            READ BELOW
    // *******************************************************************************************
    // // Please keep this text at the end of the structure:

    // 1. Do not reorder any existing members.
    // 2. Increase the step version definition before adding new members.
    // 3. Insert new members under the appropriate step version comment.
    // 4. Generate a comment for the next step version.
    // 5. Add a "PLEASE DO NOT EDIT ABOVE!" comment.
    // *******************************************************************************************
    // //
};
