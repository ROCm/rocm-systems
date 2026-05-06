// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <array>
#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
constexpr size_t NUM_CPU_BUFFERS = 3;

// Lifecycle
hsa_signal_t
signal_create();

hsa_signal_t
signal_create(hsa_ext_amd_aql_pm4_packet_t* packet);

void
signal_destroy(hsa_signal_t sig);

// Operations
void
signal_wait(hsa_signal_t sig);

void
signal_reset(hsa_signal_t sig);

// RAII helpers
struct signal_deleter_t
{
    void operator()(hsa_signal_t* s) const;
};
using signal_ptr_t = std::unique_ptr<hsa_signal_t, signal_deleter_t>;

signal_ptr_t
make_signal();

signal_ptr_t
make_signal(hsa_ext_amd_aql_pm4_packet_t* packet);

/// Plain data struct for the async DMA queue used by thread trace copies.
struct att_queue_t
{
    hsa_queue_t*                       hsa_queue{nullptr};
    std::array<void*, NUM_CPU_BUFFERS> triple_buffer_memory{};
    rocprofiler_agent_id_t             agent_id{};
    size_t                             buffer_size{0};
    hsa_agent_t                        hsa_agent{};
    hsa_agent_t                        near_cpu{};

    /// Function pointer for submit — allows test injection (replaces virtual dispatch).
    void (*submit_fn)(const att_queue_t&            self,
                      hsa_ext_amd_aql_pm4_packet_t* packet,
                      hsa_signal_t*                 completion){nullptr};
};

att_queue_t
att_queue_create(const hsa::AgentCache& agent, size_t triple_buffer_size);

void
att_queue_destroy(att_queue_t& q);

signal_ptr_t
att_queue_submit(const att_queue_t& q, hsa_ext_amd_aql_pm4_packet_t* packet, bool wait);

void
att_queue_submit(const att_queue_t&            q,
                 hsa_ext_amd_aql_pm4_packet_t* packet,
                 hsa_signal_t*                 completion);

/// Enqueues a sequence of packets and returns the completion signal of the last
/// entry without waiting on it. Useful when the caller wants to fan out submissions
/// to multiple queues and wait on them in parallel afterwards, or when a worker
/// thread will be the one to observe completion.
template <typename VecType>
signal_ptr_t
att_queue_submit_signal_last(const att_queue_t& q, VecType& vec)
{
    for(size_t i = 0; i < vec.size(); i++)
    {
        auto sig = att_queue_submit(q, &vec.at(i), i == vec.size() - 1);
        if(sig) return sig;
    }
    return nullptr;
}

/// Enqueues a sequence of packets, waits for the last packet to complete, and
/// returns its completion signal. AQL packets execute in submission order, so
/// waiting on the last signal guarantees the entire batch has drained.
template <typename VecType>
signal_ptr_t
att_queue_submit_and_wait_last(const att_queue_t& q, VecType& vec)
{
    auto sig = att_queue_submit_signal_last(q, vec);
    if(sig) signal_wait(*sig);
    return sig;
}

struct att_queue_deleter_t
{
    void operator()(att_queue_t* q) const;
};
using att_queue_ptr_t = std::unique_ptr<att_queue_t, att_queue_deleter_t>;

att_queue_ptr_t
make_att_queue(const hsa::AgentCache& agent, size_t triple_buffer_size);

};  // namespace thread_trace
};  // namespace rocprofiler
