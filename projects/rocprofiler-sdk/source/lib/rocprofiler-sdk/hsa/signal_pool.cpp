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
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
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

// One agent's signals in its signal-capable device pool, created a batch at a time.
struct agent_signal_pool
{
    hsa_amd_memory_pool_t                              memory_pool = {};
    std::mutex                                         mtx         = {};
    std::vector<hsa_signal_t>                          unused      = {};
    std::unique_ptr<common::container::pool<signal_t>> pool        = {};
};

void
construct_device_signal(signal_t& signal, agent_signal_pool& src)
{
    if(registration::get_fini_status() != 0) return;

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x17
    auto lk = std::lock_guard<std::mutex>{src.mtx};
    if(src.unused.empty() && src.memory_pool.handle != 0)
    {
        auto values =
            std::vector<hsa_signal_value_t>(device_signal_batch_size, device_signal_initial_value);
        auto signals = std::vector<hsa_signal_t>(device_signal_batch_size);
        auto batch   = hsa_amd_signal_batch_t{};
        auto status  = get_amd_ext_table()->hsa_amd_signal_batch_create_fn(device_signal_batch_size,
                                                                          values.data(),
                                                                          0,
                                                                          nullptr,
                                                                          0,
                                                                          src.memory_pool,
                                                                          &batch,
                                                                          signals.data());
        if(status == HSA_STATUS_SUCCESS)
        {
            // each member keeps the batch memory alive until it is destroyed
            get_amd_ext_table()->hsa_amd_signal_batch_destroy_fn(batch);
            src.unused.assign(signals.begin(), signals.end());
        }
        else
        {
            ROCP_WARNING << fmt::format("hsa_amd_signal_batch_create failed ({}), kernel dispatch "
                                        "completion signals fall back to host memory",
                                        hsa::get_hsa_status_string(status));
            src.memory_pool = {};
        }
    }
    if(!src.unused.empty())
    {
        signal.value         = src.unused.back();
        signal.device_memory = true;
        signal.idle_value    = device_signal_initial_value;
        src.unused.pop_back();
        return;
    }
#endif
    construct_hsa_signal(signal, 0, 0, nullptr, 0);
}

auto*
get_agent_signal_pools()
{
    static auto* pools = new std::unordered_map<uint64_t, std::unique_ptr<agent_signal_pool>>{};
    return pools;
}

std::mutex&
get_agent_signal_pools_mutex()
{
    static auto* mtx = new std::mutex{};
    return *mtx;
}
}  // namespace

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
get_signal_pool(const AgentCache& agent)
{
    if(agent.signal_pool().handle == 0) return get_signal_pool();

    auto  lk    = std::lock_guard<std::mutex>{get_agent_signal_pools_mutex()};
    auto& entry = (*get_agent_signal_pools())[agent.get_hsa_agent().handle];
    if(!entry)
    {
        entry              = std::make_unique<agent_signal_pool>();
        entry->memory_pool = agent.signal_pool();
        entry->pool        = std::make_unique<common::container::pool<signal_t>>(
            std::piecewise_construct,
            agent_signal_pool_size,
            [src = entry.get()](signal_t& signal) { construct_device_signal(signal, *src); });
    }
    return entry->pool.get();
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
    {
        auto lk = std::lock_guard<std::mutex>{get_agent_signal_pools_mutex()};
        for(auto& [handle, entry] : *get_agent_signal_pools())
        {
            entry->pool->clear([](auto& signal) { Queue::destroy_signal(&signal); });
            for(auto signal : entry->unused)
                get_core_table()->hsa_signal_destroy_fn(signal);
            entry->unused.clear();
        }
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
