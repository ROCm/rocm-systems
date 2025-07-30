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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/experimental/spm/core.h>

ROCPROFILER_EXTERN_C_INIT

// TODO: Reuse PMC dimensions
struct rocprofiler_spm_coord_t
{
    uint64_t    coord;
    const char* name;
};

typedef void (*rocprofiler_spm_decode_callback_t)(rocprofiler_counter_id_t counter_id,
                                                  rocprofiler_spm_coord_t* dimensions,
                                                  uint64_t                 num_dimensions,
                                                  const uint64_t*          timestamps,
                                                  const uint64_t*          values,
                                                  uint64_t                 count,
                                                  void*                    userdata);

rocprofiler_status_t
rocprofiler_spm_decode(rocprofiler_spm_descriptor_t      desc,
                       rocprofiler_spm_buffer_id_t       buffer_id,
                       rocprofiler_spm_decode_callback_t decode_cb,
                       void*                             data,
                       size_t                            size,
                       void* userdata) ROCPROFILER_NONNULL(3, 4) ROCPROFILER_API;

ROCPROFILER_EXTERN_C_FINI
