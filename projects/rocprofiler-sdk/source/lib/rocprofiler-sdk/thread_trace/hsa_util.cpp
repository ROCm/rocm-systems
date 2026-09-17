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

#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kfd/resource.hpp"

#include <atomic>
#include <cstring>
#include <thread>
#include <utility>

namespace rocprofiler
{
namespace thread_trace
{
using kfd::kfd_memory_kind_t;
namespace
{
constexpr size_t QUEUE_SIZE = 512;

bool
hsa_submit(const att_queue_t& queue, hsa_ext_amd_aql_pm4_packet_t* packet, att_signal_t* completion)
{
    auto*          core        = CHECK_NOTNULL(hsa::get_core_table());
    const uint64_t write_index = core->hsa_queue_add_write_index_relaxed_fn(queue.hsa_queue, 1);
    const size_t   index       = (write_index % queue.hsa_queue->size) * sizeof(*packet);
    auto*          slot =
        reinterpret_cast<uint32_t*>(static_cast<char*>(queue.hsa_queue->base_address) + index);
    const auto* packet_words = reinterpret_cast<const uint32_t*>(packet);

    std::memcpy(&slot[1], &packet_words[1], sizeof(*packet) - sizeof(uint32_t));
    if(completion)
    {
        completion->reset();
        reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(slot)->completion_signal =
            completion->handle();
    }
    reinterpret_cast<std::atomic<uint32_t>*>(slot)->store(packet_words[0],
                                                          std::memory_order_release);
    core->hsa_signal_store_screlease_fn(queue.hsa_queue->doorbell_signal, write_index);
    return true;
}

bool
kfd_submit(const att_queue_t& queue, hsa_ext_amd_aql_pm4_packet_t* packet, att_signal_t* completion)
{
    auto completion_handle = hsa_signal_t{};
    if(completion)
    {
        completion->reset();
        completion_handle = completion->handle();
    }
    if(!CHECK_NOTNULL(queue.kfd_copy_queue.get())->submit(*packet, completion_handle))
    {
        if(completion) completion->reset(0);
        ROCP_CI_LOG(ERROR) << "KFD ATT packet submission rejected";
        return false;
    }
    return true;
}
}  // namespace

att_signal_t::att_signal_t(const std::shared_ptr<kfd_memory_pool_t>& kfd_memory)
{
    if(kfd_memory)
    {
        _kfd_signal = kfd_signal_t::create(kfd_memory);
        return;
    }

    auto* ext    = CHECK_NOTNULL(hsa::get_amd_ext_table());
    auto  status = ext->hsa_amd_signal_create_fn(0, 0, nullptr, 0, &_hsa_signal);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create thread trace signal";
}

att_signal_t::~att_signal_t()
{
    if(_kfd_signal || _hsa_signal.handle == 0) return;
    wait();
    auto status = CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_destroy_fn(_hsa_signal);
    ROCP_CI_LOG_IF(WARNING, status != HSA_STATUS_SUCCESS)
        << "Failed to destroy thread trace signal";
}

hsa_signal_t
att_signal_t::handle() const
{
    return _kfd_signal ? _kfd_signal->handle() : _hsa_signal;
}

void
att_signal_t::reset(int64_t value)
{
    if(_kfd_signal)
        _kfd_signal->reset(value);
    else
        CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_store_screlease_fn(_hsa_signal, value);
}

void
att_signal_t::wait() const
{
    if(_kfd_signal)
    {
        _kfd_signal->wait();
        return;
    }

    auto wait_fn = CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_wait_scacquire_fn;
    if(_hsa_signal.handle == 0) return;
    while(wait_fn(_hsa_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
        std::this_thread::yield();
}

void
signal_wait(const att_signal_t& signal)
{
    signal.wait();
}

signal_ptr_t
make_signal(const att_queue_t& queue)
{
    auto result = std::make_unique<att_signal_t>(queue.kfd_memory);
    if(result->handle().handle == 0) return nullptr;
    return result;
}

att_queue_ptr_t
make_att_queue(rocprofiler_agent_id_t             agent_id,
               size_t                             buffer_size,
               size_t                             num_buffers,
               std::shared_ptr<kfd_memory_pool_t> kfd_memory)
{
    auto  result      = std::make_unique<att_queue_t>();
    auto& queue       = *result;
    queue.agent_id    = agent_id;
    queue.buffer_size = buffer_size;
    queue.kfd_memory  = std::move(kfd_memory);

    if(queue.kfd_memory)
    {
        queue.kfd_copy_queue = kfd_copy_queue_t::create(queue.kfd_memory, buffer_size);
        if(!queue.kfd_copy_queue) return nullptr;
        queue.submit_fn = kfd_submit;
        queue.cpu_buffers.resize(num_buffers, nullptr);
        for(auto& memory : queue.cpu_buffers)
        {
            memory = queue.kfd_memory->allocate(buffer_size, kfd_memory_kind_t::host);
            if(!memory) return nullptr;
        }
        return result;
    }

    const auto* cache =
        CHECK_NOTNULL(rocprofiler::agent::get_agent_cache(rocprofiler::agent::get_agent(agent_id)));
    queue.hsa_agent = cache->get_hsa_agent();
    queue.near_cpu  = cache->near_cpu();

    queue.submit_fn = hsa_submit;
    auto* core      = CHECK_NOTNULL(hsa::get_core_table());
    auto* ext       = CHECK_NOTNULL(hsa::get_amd_ext_table());
    auto  status    = core->hsa_queue_create_fn(queue.hsa_agent,
                                            QUEUE_SIZE,
                                            HSA_QUEUE_TYPE_MULTI,
                                            nullptr,
                                            nullptr,
                                            UINT32_MAX,
                                            UINT32_MAX,
                                            &queue.hsa_queue);
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to create thread trace async queue";

    queue.cpu_buffers.resize(num_buffers, nullptr);
    for(auto& memory : queue.cpu_buffers)
    {
        status = ext->hsa_amd_memory_pool_allocate_fn(cache->cpu_pool(), buffer_size, 0, &memory);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to allocate thread trace memory";
        status = ext->hsa_amd_agents_allow_access_fn(1, &queue.near_cpu, nullptr, memory);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to allow CPU access";
        status = ext->hsa_amd_agents_allow_access_fn(1, &queue.hsa_agent, nullptr, memory);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS) << "Failed to allow GPU access";
    }
    if(num_buffers > 0) queue.copy_signal = make_signal(queue);
    return result;
}

att_queue_t::~att_queue_t()
{
    copy_signal.reset();
    if(kfd_memory)
    {
        if(kfd_copy_queue) kfd_copy_queue->close();
        for(auto* memory : cpu_buffers)
            kfd_memory->deallocate(memory);
    }
    else
    {
        auto* core   = CHECK_NOTNULL(hsa::get_core_table());
        auto* ext    = CHECK_NOTNULL(hsa::get_amd_ext_table());
        auto  status = hsa_queue ? core->hsa_queue_destroy_fn(hsa_queue) : HSA_STATUS_SUCCESS;
        ROCP_CI_LOG_IF(WARNING, status != HSA_STATUS_SUCCESS)
            << "Failed to destroy thread trace queue";
        for(auto* memory : cpu_buffers)
        {
            status = ext->hsa_amd_memory_pool_free_fn(memory);
            ROCP_CI_LOG_IF(WARNING, status != HSA_STATUS_SUCCESS)
                << "Failed to free thread trace memory";
        }
    }
}

bool
att_queue_enabled(const att_queue_t& queue)
{
    auto lock = std::unique_lock{queue.submit_mutex};
    return queue.submit_fn != nullptr;
}

bool
att_queue_submit(const att_queue_t&            queue,
                 hsa_ext_amd_aql_pm4_packet_t* packet,
                 att_signal_t*                 completion)
{
    auto lock = std::unique_lock{queue.submit_mutex};
    if(!queue.submit_fn)
    {
        ROCP_TRACE << "Discarding ATT packet submission on disabled queue for agent "
                   << queue.agent_id.handle;
        return false;
    }
    return queue.submit_fn(queue, packet, completion);
}

signal_ptr_t
att_queue_submit(const att_queue_t& queue, hsa_ext_amd_aql_pm4_packet_t* packet, bool wait)
{
    auto signal = signal_ptr_t{};
    if(wait)
    {
        signal = make_signal(queue);
        if(!signal) return nullptr;
    }
    if(!att_queue_submit(queue, packet, signal.get())) return nullptr;
    return signal;
}

bool
att_queue_copy(att_queue_t& queue, void* dst, const void* src, size_t size)
{
    if(size == 0) return true;
    if(queue.kfd_copy_queue)
    {
        return queue.kfd_copy_queue->copy(dst, src, size);
    }

    auto& completion = *CHECK_NOTNULL(queue.copy_signal.get());
    completion.reset();
    auto status =
        CHECK_NOTNULL(hsa::get_amd_ext_table())
            ->hsa_amd_memory_async_copy_fn(
                dst, queue.near_cpu, src, queue.hsa_agent, size, 0, nullptr, completion.handle());
    if(status != HSA_STATUS_SUCCESS)
    {
        CHECK_NOTNULL(hsa::get_core_table())->hsa_signal_store_screlease_fn(completion.handle(), 0);
        ROCP_CI_LOG(ERROR) << "Failed to copy thread trace memory: " << status;
        return false;
    }
    completion.wait();
    return true;
}

}  // namespace thread_trace
}  // namespace rocprofiler
