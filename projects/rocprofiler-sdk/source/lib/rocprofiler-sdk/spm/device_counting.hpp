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

#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <cstddef>

namespace rocprofiler
{
namespace context
{
struct context;
}

namespace SPM
{
struct spm_counter_config;

struct spm_agent_callback_data
{
    uint64_t                                     context_idx  = 0;
    hsa_queue_t*                                 queue        = nullptr;
    hsa_signal_t                                 start_signal = {.handle = 0};
    hsa_signal_t                                 stop_signal  = {.handle = 0};
    std::unique_ptr<rocprofiler::hsa::SPMPacket> packet;

    rocprofiler_user_data_t user_data     = {.value = 0};
    rocprofiler_user_data_t callback_data = {.value = 0};

    std::shared_ptr<rocprofiler::SPM::spm_counter_config> profile     = {};
    rocprofiler_agent_id_t                                agent_id    = {.handle = 0};
    rocprofiler_spm_device_counting_service_cb_t          cb          = nullptr;
    rocprofiler_buffer_id_t                               buffer      = {.handle = 0};
    bool                                                  set_profile = false;

    spm_agent_callback_data() = default;
    spm_agent_callback_data(spm_agent_callback_data&& rhs) noexcept
    : queue(rhs.queue)
    , user_data(rhs.user_data)
    , callback_data(rhs.callback_data)
    , profile(rhs.profile)
    , agent_id(rhs.agent_id)
    , cb(rhs.cb)
    , buffer(rhs.buffer)
    {
        start_signal.handle = 0;
        stop_signal.handle  = 0;
    };

    spm_agent_callback_data& operator=(const spm_agent_callback_data&) = delete;
    spm_agent_callback_data(const spm_agent_callback_data&)            = delete;

    ~spm_agent_callback_data();
};

rocprofiler_status_t
spm_start_agent_ctx(const context::context* ctx);

// Send the AQL end packet to a queue on the agent to stop
// collecting counter data. This function is synchronous and will
// return when the agent has stopped collecting data (or if there
// is an error).
rocprofiler_status_t
spm_stop_agent_ctx(const context::context* ctx);

uint64_t
submitPacket(hsa_queue_t* queue, const void* packet);

rocprofiler_status_t
spm_device_counting_service_hsa_registration();

rocprofiler_status_t
spm_device_counting_service_finalize();

}  // namespace SPM
}  // namespace rocprofiler
