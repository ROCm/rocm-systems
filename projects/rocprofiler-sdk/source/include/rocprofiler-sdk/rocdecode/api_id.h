

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

#include <rocprofiler-sdk/rocdecode/api_trace.h>

/**
 * @brief ROCProfiler enumeration of rocDecode API tracing operations
 */
typedef enum rocprofiler_rocdecode_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_ROCDECODE_API_ID_NONE = -1,

    ROCPROFILER_ROCDECODE_API_ID_rocDecCreateVideoParser = 0,
    ROCPROFILER_ROCDECODE_API_ID_rocDecParseVideoData,
    ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyVideoParser,
    ROCPROFILER_ROCDECODE_API_ID_rocDecCreateDecoder,
    ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyDecoder,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecoderCaps,
    ROCPROFILER_ROCDECODE_API_ID_rocDecDecodeFrame,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetDecodeStatus,
    ROCPROFILER_ROCDECODE_API_ID_rocDecReconfigureDecoder,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetVideoFrame,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetErrorName,

#if ROCDECODE_RUNTIME_API_TABLE_STEP_VERSION >= 1
    ROCPROFILER_ROCDECODE_API_ID_rocDecCreateBitstreamReader,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamCodecType,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamBitDepth,
    ROCPROFILER_ROCDECODE_API_ID_rocDecGetBitstreamPicData,
    ROCPROFILER_ROCDECODE_API_ID_rocDecDestroyBitstreamReader,
#endif
    ROCPROFILER_ROCDECODE_API_ID_LAST,
} rocprofiler_rocdecode_api_id_t;
