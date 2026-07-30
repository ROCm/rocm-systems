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

/**
 * @brief ROCProfiler enumeration of hipFILE API tracing operations
 */
typedef enum rocprofiler_hipfile_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HIPFILE_API_ID_NONE = -1,

    ROCPROFILER_HIPFILE_API_ID_hipFileGetOpErrorString = 0,
    ROCPROFILER_HIPFILE_API_ID_hipFileHandleRegister,
    ROCPROFILER_HIPFILE_API_ID_hipFileHandleDeregister,
    ROCPROFILER_HIPFILE_API_ID_hipFileBufRegister,
    ROCPROFILER_HIPFILE_API_ID_hipFileBufDeregister,
    ROCPROFILER_HIPFILE_API_ID_hipFileRead,
    ROCPROFILER_HIPFILE_API_ID_hipFileWrite,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverOpen,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverClose,
    ROCPROFILER_HIPFILE_API_ID_hipFileUseCount,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverGetProperties,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetPollMode,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxDirectIOSize,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxCacheSize,
    ROCPROFILER_HIPFILE_API_ID_hipFileDriverSetMaxPinnedMemSize,
    ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOSetUp,
    ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOSubmit,
    ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOGetStatus,
    ROCPROFILER_HIPFILE_API_ID_hipFileBatchIOCancel,
    ROCPROFILER_HIPFILE_API_ID_hipFileBatchIODestroy,
    ROCPROFILER_HIPFILE_API_ID_hipFileReadAsync,
    ROCPROFILER_HIPFILE_API_ID_hipFileWriteAsync,
    ROCPROFILER_HIPFILE_API_ID_hipFileStreamRegister,
    ROCPROFILER_HIPFILE_API_ID_hipFileStreamDeregister,
    ROCPROFILER_HIPFILE_API_ID_hipFileGetVersion,
    ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterSizeT,
    ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterBool,
    ROCPROFILER_HIPFILE_API_ID_hipFileGetParameterString,
    ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterSizeT,
    ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterBool,
    ROCPROFILER_HIPFILE_API_ID_hipFileSetParameterString,

    ROCPROFILER_HIPFILE_API_ID_LAST,
} rocprofiler_hipfile_api_id_t;
