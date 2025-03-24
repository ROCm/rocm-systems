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

#include <rocprofiler-sdk/rocjpeg/details/rocjpeg_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG > 0
#    include <rocjpeg/rocjpeg.h>
#else
#    include <rocprofiler-sdk/rocjpeg/details/rocjpeg.h>
#endif

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_rocjpeg_api_no_args
{
    char empty;
} rocprofiler_rocjpeg_api_no_args;

typedef union rocprofiler_rocjpeg_api_retval_t
{
    int32_t     rocJpegStatus_retval;
    const char* const_charp_retval;
} rocprofiler_rocjpeg_api_retval_t;

typedef union rocprofiler_rocjpeg_api_args_t
{
    struct
    {
        RocJpegStreamHandle* jpeg_stream_handle;
    } rocJpegStreamCreate;

    struct
    {
        const unsigned char* data;
        size_t               length;
        RocJpegStreamHandle  jpeg_stream_handle;
    } rocJpegStreamParse;

    struct
    {
        RocJpegStreamHandle jpeg_stream_handle;
    } rocJpegStreamDestroy;

    struct
    {
        RocJpegBackend backend;
        int            device_id;
        RocJpegHandle* handle;
    } rocJpegCreate;

    struct
    {
        RocJpegHandle handle;
    } rocJpegDestroy;

    struct
    {
        RocJpegHandle             handle;
        RocJpegStreamHandle       jpeg_stream_handle;
        uint8_t*                  num_components;
        RocJpegChromaSubsampling* subsampling;
        uint32_t*                 widths;
        uint32_t*                 heights;
    } rocJpegGetImageInfo;

    struct
    {
        RocJpegHandle              handle;
        RocJpegStreamHandle        jpeg_stream_handle;
        const RocJpegDecodeParams* decode_params;
        RocJpegImage*              destination;
    } rocJpegDecode;

    struct
    {
        RocJpegHandle              handle;
        RocJpegStreamHandle*       jpeg_stream_handles;
        int                        batch_size;
        const RocJpegDecodeParams* decode_params;
        RocJpegImage*              destinations;
    } rocJpegDecodeBatched;

    struct
    {
        RocJpegStatus rocjpeg_status;
    } rocJpegGetErrorName;
} rocprofiler_rocjpeg_api_args_t;

ROCPROFILER_EXTERN_C_FINI
