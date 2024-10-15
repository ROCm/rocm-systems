// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/hip_runtime.hpp"
#include "core/perfetto.hpp"
#include "library/components/roctracer.hpp"
#include "library/ptl.hpp"

#include <iostream>
#include <memory>

// Macro to check ROC-tracer calls status
#define ROCPROFSYS_ROCTRACER_CALL(call)                                                  \
    {                                                                                    \
        ROCPROFSYS_DEBUG_F(#call);                                                       \
        int err = call;                                                                  \
        if(err != 0)                                                                     \
        {                                                                                \
            ROCPROFSYS_PRINT_F("%s in: %s\n", roctracer_error_string(), #call);          \
        }                                                                                \
    }

namespace rocprofsys
{
using roctracer_hip_bundle_t =
    tim::component_bundle<category::rocm_hip, comp::roctracer_data, comp::wall_clock>;
using roctracer_hsa_bundle_t =
    tim::component_bundle<category::rocm_hsa, comp::roctracer_data>;
using roctracer_functions_t = std::vector<std::pair<std::string, std::function<void()>>>;

// HSA API callback function
void
hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);

void
hsa_activity_callback(uint32_t op, const void* record, void* arg);

void
hip_exec_activity_callbacks(int64_t _tid);

// HIP API callback function
void
hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);

void
roctx_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg);

// Activity tracing callback
void
hip_activity_callback(const char* begin, const char* end, void*);

bool&
roctracer_is_init();

bool&
roctracer_is_setup();

int64_t
get_clock_skew();

roctracer_functions_t&
roctracer_setup_routines();

roctracer_functions_t&
roctracer_shutdown_routines();
}  // namespace rocprofsys
