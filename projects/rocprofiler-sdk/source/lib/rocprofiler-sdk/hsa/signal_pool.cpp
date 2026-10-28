// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <hsa/amd_hsa_signal.h>

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace
{
bool
signal_pool_exists()
{
    return (common::static_object<common::container::pool<signal_t>>::get() != nullptr);
}

signal_t&
reset_hsa_signal(signal_t& signal, hsa_signal_value_t initial_value)
{
    get_core_table()->hsa_signal_store_screlease_fn(signal.value, initial_value);
    return signal;
}

constexpr size_t   agent_signal_pool_size   = 512;
constexpr uint32_t device_signal_batch_size = 256;
// counts down one per dispatch and is never reset, so it must not reach zero
constexpr hsa_signal_value_t device_signal_initial_value =
    std::numeric_limits<hsa_signal_value_t>::max();

struct agent_device_signals
{
    device_signal_source                               source = {};
    std::once_flag                                     once   = {};
    std::unique_ptr<common::container::pool<signal_t>> pool   = {};
};

// One entry per agent with signal-capable device memory. Filled once, then only read, so the
// per-dispatch lookup takes no lock.
using agent_device_signals_vec_t = std::vector<std::unique_ptr<agent_device_signals>>;

std::atomic<bool>&
get_device_signals_in_use()
{
    static auto _v = std::atomic<bool>{false};
    return _v;
}

std::atomic<agent_device_signals_vec_t*>&
agent_device_signals_storage()
{
    static auto _v = std::atomic<agent_device_signals_vec_t*>{nullptr};
    return _v;
}

agent_device_signals_vec_t&
get_agent_device_signals()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        auto* _v = new agent_device_signals_vec_t{};
        for(const auto* rocp_agent : agent::get_agents())
        {
            const auto* cache = agent::get_agent_cache(rocp_agent);
            if(!cache || cache->signal_pool().handle == 0) continue;

            auto& entry               = _v->emplace_back(std::make_unique<agent_device_signals>());
            entry->source.agent       = cache->get_hsa_agent();
            entry->source.memory_pool = cache->signal_pool();
        }
        agent_device_signals_storage().store(_v, std::memory_order_release);
    });
    return *agent_device_signals_storage().load(std::memory_order_acquire);
}
}  // namespace

void
device_signal_source::construct(signal_t& signal)
{
    if(registration::get_fini_status() != 0) return;

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x17
    auto lk = std::lock_guard<std::mutex>{mtx};
    if(unused.empty() && memory_pool.handle != 0)
    {
        auto values =
            std::vector<hsa_signal_value_t>(device_signal_batch_size, device_signal_initial_value);
        auto signals = std::vector<hsa_signal_t>(device_signal_batch_size);
        auto batch   = hsa_amd_signal_batch_t{};
        auto status =
            get_amd_ext_table()->hsa_amd_signal_batch_create_fn(device_signal_batch_size,
                                                                values.data(),
                                                                1,
                                                                &agent,
                                                                HSA_AMD_SIGNAL_AMD_GPU_ONLY,
                                                                memory_pool,
                                                                &batch,
                                                                signals.data());
        if(status == HSA_STATUS_SUCCESS)
        {
            // each member keeps the batch memory alive until it is destroyed
            get_amd_ext_table()->hsa_amd_signal_batch_destroy_fn(batch);
            created += signals.size();
            unused.assign(signals.begin(), signals.end());
        }
        else
        {
            ROCP_WARNING << fmt::format("hsa_amd_signal_batch_create failed ({}), kernel dispatch "
                                        "completion signals fall back to host memory",
                                        hsa::get_hsa_status_string(status));
            memory_pool = {};
        }
    }
    if(!unused.empty())
    {
        signal.value         = unused.back();
        signal.device_memory = true;
        signal.idle_value    = device_signal_initial_value;
        unused.pop_back();
        ++handed_out;
        // the timestamp reads take the whole amd_signal_t as one cache line
        ROCP_FATAL_IF(signal.value.handle % alignof(amd_signal_t) != 0) << fmt::format(
            "device signal handle {:#x} is not an aligned amd_signal_t", signal.value.handle);
        return;
    }
#endif
    construct_hsa_signal(signal, 0, 0, nullptr, 0);
}

size_t
device_signal_source::destroy_unused()
{
    auto lk = std::lock_guard<std::mutex>{mtx};
    auto n  = unused.size();
    if(get_core_table() && get_core_table()->hsa_signal_destroy_fn)
    {
        for(auto itr : unused)
            get_core_table()->hsa_signal_destroy_fn(itr);
    }
    unused.clear();
    return n;
}

bool
retire_device_signal(signal_t& signal, hsa_signal_value_t observed)
{
    if(observed != signal.idle_value - 1) return false;
    signal.idle_value = observed;
    return true;
}

signal_t&
construct_hsa_signal(signal_t&          signal,
                     hsa_signal_value_t initial_value,
                     uint32_t           num_consumers,
                     const hsa_agent_t* consumers,
                     uint64_t           attributes)
{
    if(signal.value != hsa_signal_t{})
    {
        ROCP_WARNING_IF(num_consumers != 0 || consumers != nullptr || attributes != 0)
            << "Ignoring HSA signal creation arguments when reusing an existing signal";
        return reset_hsa_signal(signal, initial_value);
    }

    auto status = HSA_STATUS_SUCCESS;
    if(!get_amd_ext_table() || !get_amd_ext_table()->hsa_amd_signal_create_fn)
        status = HSA_STATUS_ERROR;
    else
        status = get_amd_ext_table()->hsa_amd_signal_create_fn(
            initial_value, num_consumers, consumers, attributes, &signal.value);

    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS)
        << fmt::format("Error: hsa_amd_signal_create failed with error code {} :: {}",
                       static_cast<int>(status),
                       hsa::get_hsa_status_string(status));

    return signal;
}

common::container::pool<signal_t>*
get_signal_pool()
{
    constexpr size_t default_signal_pool_size = (1 << 12);  // 4096 signals per pool batch

    static auto*& pool = common::static_object<common::container::pool<signal_t>>::construct(
        std::piecewise_construct, default_signal_pool_size, [](signal_t& signal) {
            if(registration::get_fini_status() == 0) construct_hsa_signal(signal, 0, 0, nullptr, 0);
        });

    return pool;
}

common::container::pool<signal_t>*
get_device_signal_pool(const AgentCache& agent)
{
    if(agent.signal_pool().handle == 0) return nullptr;

    for(const auto& entry : get_agent_device_signals())
    {
        if(entry->source.agent.handle != agent.get_hsa_agent().handle) continue;

        std::call_once(entry->once, [&entry]() {
            entry->pool = std::make_unique<common::container::pool<signal_t>>(
                std::piecewise_construct,
                agent_signal_pool_size,
                [src = &entry->source](signal_t& signal) { src->construct(signal); });
            get_device_signals_in_use().store(true, std::memory_order_release);
        });
        return entry->pool.get();
    }
    return nullptr;
}

bool
device_signals_in_use()
{
    return get_device_signals_in_use().load(std::memory_order_acquire);
}

void
signal_pool_init()
{
    common::consume_args(get_signal_pool());
}

/**
 * @brief Finalize the signal pool, destroying any remaining signals and printing a usage report.
 *
 */
void
signal_pool_fini()
{
    // without building the list if no dispatch ever asked for it
    static auto _none           = agent_device_signals_vec_t{};
    auto*       _device_signals = agent_device_signals_storage().load(std::memory_order_acquire);
    for(auto& entry : (_device_signals) ? *_device_signals : _none)
    {
        auto pooled = size_t{0};
        if(entry->pool)
        {
            entry->pool->clear([&pooled](auto& signal) {
                if(signal.get().device_memory) ++pooled;
                Queue::destroy_signal(&signal);
            });
        }
        const auto unused = entry->source.destroy_unused();
        ROCP_WARNING_IF(pooled + unused != entry->source.created)
            << fmt::format("device signals of agent {:#x}: created {}, destroyed {} pooled and {} "
                           "never used",
                           entry->source.agent.handle,
                           entry->source.created,
                           pooled,
                           unused);
        // fini can run more than once
        entry->pool.reset();
        entry->source.created = 0;
    }

    // checks if pool exists without constructing it if it doesn't exist
    if(!signal_pool_exists()) return;

    if(auto* pool = get_signal_pool(); pool != nullptr)
    {
        // only report once
        static auto _once = std::once_flag{};
        std::call_once(_once, [&]() { ROCP_INFO << pool->get_usage_report(); });

        // always try to clear
        pool->clear([](auto& signal) { Queue::destroy_signal(&signal); });
    }
}

}  // namespace hsa
}  // namespace rocprofiler
