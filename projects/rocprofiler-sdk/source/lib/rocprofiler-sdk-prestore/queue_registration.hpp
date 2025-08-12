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

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>

using hsa_amd_queue_intercept_packet_writer_t = void(*)(const void*, uint64_t);
using write_interceptor_t = void(*)(const void*, uint64_t, uint64_t, void*, hsa_amd_queue_intercept_packet_writer_t);

namespace rocprofiler {
namespace prestore {
    
struct queue_prestore_export_t {
    hsa_agent_t agent;
    hsa_queue_t* queue;
};

void
queue_registration_init(HsaApiTable* table);

}}

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_prestore_export_all_queues(
    rocprofiler::prestore::queue_prestore_export_t* queues,
    uint64_t* num_queues) ROCPROFILER_API;

int rocprofiler_prestore_set_write_interceptor(
    hsa_queue_t* queue,
    write_interceptor_t func,
    void* data) ROCPROFILER_API;

ROCPROFILER_EXTERN_C_FINI
