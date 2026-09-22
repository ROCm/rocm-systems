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

#include <rocprofiler-sdk/rocjpeg/details/rocjpeg_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG > 0
#    include <rocjpeg/rocjpeg.h>
#else
#    include <rocprofiler-sdk/rocjpeg/details/rocjpeg.h>
#endif

// Define version macros for the rocJPEG API dispatch table, specifying the MAJOR and STEP versions.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     IMPORTANT    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// 1. When adding new functions to the rocJPEG API dispatch table, always append the new function
// pointer
//    to the end of the table and increment the dispatch table's version number. Never rearrange the
//    order of the member variables in the dispatch table, as doing so will break the Application
//    Binary Interface (ABI).
// 2. In critical situations where the type of an existing member variable in a dispatch table has
// been changed
//    or removed due to a data type modification, it is important to increment the major version
//    number of the rocJPEG API dispatch table. If the function pointer type can no longer be
//    declared, do not remove it. Instead, change the function pointer type to `void*` and ensure it
//    is always initialized to `nullptr`.
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//

// The major version number should ideally remain unchanged. Increment the
// ROCJPEG_RUNTIME_API_TABLE_MAJOR_VERSION only for fundamental changes to the rocJPEGDispatchTable
// struct, such as altering the type or name of an existing member variable. Please DO NOT REMOVE
// it.
#define ROCJPEG_RUNTIME_API_TABLE_MAJOR_VERSION 0

// Increment the ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION when new runtime API functions are added.
// If the corresponding ROCJPEG_RUNTIME_API_TABLE_MAJOR_VERSION increases reset the
// ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION to zero.
#define ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION 0

// rocJPEG API interface
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegStreamCreate)(RocJpegStreamHandle* jpeg_stream_handle);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegStreamParse)(const unsigned char* data,
                                                         size_t               length,
                                                         RocJpegStreamHandle  jpeg_stream_handle);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegStreamDestroy)(RocJpegStreamHandle jpeg_stream_handle);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegCreate)(RocJpegBackend backend,
                                                    int            device_id,
                                                    RocJpegHandle* handle);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegDestroy)(RocJpegHandle handle);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegGetImageInfo)(RocJpegHandle       handle,
                                                          RocJpegStreamHandle jpeg_stream_handle,
                                                          uint8_t*            num_components,
                                                          RocJpegChromaSubsampling* subsampling,
                                                          uint32_t*                 widths,
                                                          uint32_t*                 heights);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegDecode)(RocJpegHandle              handle,
                                                    RocJpegStreamHandle        jpeg_stream_handle,
                                                    const RocJpegDecodeParams* decode_params,
                                                    RocJpegImage*              destination);
typedef RocJpegStatus(ROCJPEGAPI* PfnRocJpegDecodeBatched)(RocJpegHandle        handle,
                                                           RocJpegStreamHandle* jpeg_stream_handles,
                                                           int                  batch_size,
                                                           const RocJpegDecodeParams* decode_params,
                                                           RocJpegImage*              destinations);
typedef const char*(ROCJPEGAPI* PfnRocJpegGetErrorName)(RocJpegStatus rocjpeg_status);

// rocJPEG API dispatch table
struct RocJpegDispatchTable
{
    // ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION == 0
    size_t                  size;
    PfnRocJpegStreamCreate  pfn_rocjpeg_stream_create;
    PfnRocJpegStreamParse   pfn_rocjpeg_stream_parse;
    PfnRocJpegStreamDestroy pfn_rocjpeg_stream_destroy;
    PfnRocJpegCreate        pfn_rocjpeg_create;
    PfnRocJpegDestroy       pfn_rocjpeg_destroy;
    PfnRocJpegGetImageInfo  pfn_rocjpeg_get_image_info;
    PfnRocJpegDecode        pfn_rocjpeg_decode;
    PfnRocJpegDecodeBatched pfn_rocjpeg_decode_batched;
    PfnRocJpegGetErrorName  pfn_rocjpeg_get_error_name;

    // PLEASE DO NOT EDIT ABOVE!
    // ROCJPEG_RUNTIME_API_TABLE_STEP_VERSION == 1

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
