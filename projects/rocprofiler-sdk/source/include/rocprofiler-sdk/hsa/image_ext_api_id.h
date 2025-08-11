// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/ext_version.h>

/**
 * @brief ROCProfiler enumeration of HSA Image Extended API tracing operations
 */
typedef enum rocprofiler_hsa_image_ext_api_id_t  // NOLINT(performance-enum-size)
{
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_NONE = -1,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_get_capability,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_data_get_info,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_create,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_import,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_export,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_copy,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_clear,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_destroy,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_sampler_create,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_sampler_destroy,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_get_capability_with_layout,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_data_get_info_with_layout,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_hsa_ext_image_create_with_layout,
    ROCPROFILER_HSA_IMAGE_EXT_API_ID_LAST,
} rocprofiler_hsa_image_ext_api_id_t;
