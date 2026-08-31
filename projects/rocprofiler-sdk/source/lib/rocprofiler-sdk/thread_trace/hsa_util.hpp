// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <memory>
#include <vector>

namespace rocprofiler
{
namespace thread_trace
{
class kfd_copy_queue_t;
class kfd_memory_pool_t;
class kfd_signal_t;

class att_signal_t
{
public:
    explicit att_signal_t(std::shared_ptr<kfd_memory_pool_t> kfd_memory = {});
    ~att_signal_t();

    att_signal_t(const att_signal_t&) = delete;
    att_signal_t& operator=(const att_signal_t&) = delete;

    hsa_signal_t handle() const;
    void         reset();
    void         wait() const;

private:
    hsa_signal_t                  _hsa_signal{};
    std::unique_ptr<kfd_signal_t> _kfd_signal{};
};

using signal_ptr_t = std::unique_ptr<att_signal_t>;

struct att_queue_t
{
    std::shared_ptr<kfd_memory_pool_t> kfd_memory{};
    std::shared_ptr<kfd_copy_queue_t>  kfd_copy_queue{};
    signal_ptr_t                       copy_signal{};
    hsa_queue_t*                       hsa_queue{nullptr};
    std::vector<void*>                 cpu_buffers{};
    rocprofiler_agent_id_t             agent_id{};
    size_t                             buffer_size{0};
    hsa_agent_t                        hsa_agent{};
    hsa_agent_t                        near_cpu{};

    void (*submit_fn)(const att_queue_t&            self,
                      hsa_ext_amd_aql_pm4_packet_t* packet,
                      att_signal_t*                 completion){nullptr};
};

void
signal_wait(const att_signal_t& signal);

signal_ptr_t
make_signal(const att_queue_t& queue);

att_queue_t
att_queue_create(rocprofiler_agent_id_t             agent_id,
                 size_t                             buffer_size,
                 size_t                             num_buffers = 0,
                 std::shared_ptr<kfd_memory_pool_t> kfd_memory  = {});

void
att_queue_destroy(att_queue_t& queue);

signal_ptr_t
att_queue_submit(const att_queue_t& queue, hsa_ext_amd_aql_pm4_packet_t* packet, bool wait);

void
att_queue_submit(const att_queue_t&            queue,
                 hsa_ext_amd_aql_pm4_packet_t* packet,
                 att_signal_t*                 completion);

void
att_queue_copy(att_queue_t& queue, void* dst, const void* src, size_t size);

template <typename VecType>
signal_ptr_t
att_queue_submit_signal_last(const att_queue_t& queue, VecType& packets)
{
    for(size_t i = 0; i < packets.size(); ++i)
    {
        auto signal = att_queue_submit(queue, &packets.at(i), i + 1 == packets.size());
        if(signal) return signal;
    }
    return nullptr;
}

struct att_queue_deleter_t
{
    void operator()(att_queue_t* queue) const;
};

using att_queue_ptr_t = std::unique_ptr<att_queue_t, att_queue_deleter_t>;

att_queue_ptr_t
make_att_queue(rocprofiler_agent_id_t             agent_id,
               size_t                             buffer_size,
               size_t                             num_buffers = 0,
               std::shared_ptr<kfd_memory_pool_t> kfd_memory  = {});

}  // namespace thread_trace
}  // namespace rocprofiler
