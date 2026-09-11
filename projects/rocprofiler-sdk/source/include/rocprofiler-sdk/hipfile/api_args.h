// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>

#include <rocprofiler-sdk/hipfile/details/hipfile_headers.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

typedef struct rocprofiler_hipfile_api_no_args
{
    char empty;
} rocprofiler_hipfile_api_no_args;

typedef union rocprofiler_hipfile_api_retval_t
{
    const char*    const_charp_retval;
    hipFileError_t hipFileError_t_retval;
    ssize_t        ssize_t_retval;
    int64_t        int64_t_retval;
} rocprofiler_hipfile_api_retval_t;

typedef union rocprofiler_hipfile_api_args_t
{
    struct
    {
        hipFileOpError_t status;
    } hipFileGetOpErrorString;

    struct
    {
        hipFileHandle_t* fh;
        hipFileDescr_t*  descr;
    } hipFileHandleRegister;

    struct
    {
        hipFileHandle_t fh;
    } hipFileHandleDeregister;

    struct
    {
        const void* buffer_base;
        size_t      length;
        int         flags;
    } hipFileBufRegister;

    struct
    {
        const void* buffer_base;
    } hipFileBufDeregister;

    struct
    {
        hipFileHandle_t fh;
        void*           buffer_base;
        size_t          size;
        hoff_t          file_offset;
        hoff_t          buffer_offset;
    } hipFileRead;

    struct
    {
        hipFileHandle_t fh;
        const void*     buffer_base;
        size_t          size;
        hoff_t          file_offset;
        hoff_t          buffer_offset;
    } hipFileWrite;

    rocprofiler_hipfile_api_no_args hipFileDriverOpen;
    rocprofiler_hipfile_api_no_args hipFileDriverClose;
    rocprofiler_hipfile_api_no_args hipFileUseCount;

    struct
    {
        hipFileDriverProps_t* props;
    } hipFileDriverGetProperties;

    struct
    {
        bool   poll;
        size_t poll_threshold_size;
    } hipFileDriverSetPollMode;

    struct
    {
        size_t max_direct_io_size;
    } hipFileDriverSetMaxDirectIOSize;

    struct
    {
        size_t max_cache_size;
    } hipFileDriverSetMaxCacheSize;

    struct
    {
        size_t max_pinned_size;
    } hipFileDriverSetMaxPinnedMemSize;

    struct
    {
        hipFileBatchHandle_t* batch_idp;
        unsigned              max_nr;
    } hipFileBatchIOSetUp;

    struct
    {
        hipFileBatchHandle_t batch_idp;
        unsigned             nr;
        hipFileIOParams_t*   iocbp;
        unsigned             flags;
    } hipFileBatchIOSubmit;

    struct
    {
        hipFileBatchHandle_t batch_idp;
        unsigned             min_nr;
        unsigned*            nr;
        hipFileIOEvents_t*   iocbp;
        struct timespec*     timeout;
    } hipFileBatchIOGetStatus;

    struct
    {
        hipFileBatchHandle_t batch_idp;
    } hipFileBatchIOCancel;

    struct
    {
        hipFileBatchHandle_t batch_idp;
    } hipFileBatchIODestroy;

    struct
    {
        hipFileHandle_t fh;
        void*           buffer_base;
        size_t*         size_p;
        hoff_t*         file_offset_p;
        hoff_t*         buffer_offset_p;
        ssize_t*        bytes_read_p;
        hipStream_t     stream;
    } hipFileReadAsync;

    struct
    {
        hipFileHandle_t fh;
        void*           buffer_base;
        size_t*         size_p;
        hoff_t*         file_offset_p;
        hoff_t*         buffer_offset_p;
        ssize_t*        bytes_written_p;
        hipStream_t     stream;
    } hipFileWriteAsync;

    struct
    {
        hipStream_t stream;
        unsigned    flags;
    } hipFileStreamRegister;

    struct
    {
        hipStream_t stream;
    } hipFileStreamDeregister;

    struct
    {
        unsigned* major;
        unsigned* minor;
        unsigned* patch;
    } hipFileGetVersion;

    struct
    {
        hipFileSizeTConfigParameter_t param;
        size_t*                       value;
    } hipFileGetParameterSizeT;

    struct
    {
        hipFileBoolConfigParameter_t param;
        bool*                        value;
    } hipFileGetParameterBool;

    struct
    {
        hipFileStringConfigParameter_t param;
        char*                          desc_str;
        int                            len;
    } hipFileGetParameterString;

    struct
    {
        hipFileSizeTConfigParameter_t param;
        size_t                        value;
    } hipFileSetParameterSizeT;

    struct
    {
        hipFileBoolConfigParameter_t param;
        bool                         value;
    } hipFileSetParameterBool;

    struct
    {
        hipFileStringConfigParameter_t param;
        const char*                    desc_str;
    } hipFileSetParameterString;

#if HIPFILE_RUNTIME_API_TABLE_STEP_VERSION >= 1
    struct
    {
        hipFileStatsLevel1_t* stats;
    } hipFileGetStatsL1;

    struct
    {
        hipFileStatsLevel2_t* stats;
    } hipFileGetStatsL2;

    struct
    {
        hipFileStatsLevel3_t* stats;
    } hipFileGetStatsL3;
#endif
} rocprofiler_hipfile_api_args_t;

ROCPROFILER_EXTERN_C_FINI
