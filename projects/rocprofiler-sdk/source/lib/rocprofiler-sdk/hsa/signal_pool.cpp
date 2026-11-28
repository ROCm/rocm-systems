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
}  // namespace

signal_t&
construct_hsa_signal(signal_t&          signal,
                     hsa_signal_value_t initial_value,
                     uint32_t           num_consumers,
                     const hsa_agent_t* consumers,
                     uint64_t           attributes)
{
    // A handle from a previous runtime generation is not a reusable signal: once the runtime closed
    // it, the handle is invalid, and reusing it would substitute a dead signal for a fresh one.
    // Drop it so the create path below allocates for the current generation.
    if(signal.value != hsa_signal_t{} && is_runtime_shutdown()) signal.value = hsa_signal_t{};

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
    constexpr size_t default_signal_pool_size = (1 << 10);

    static auto*& pool = common::static_object<common::container::pool<signal_t>>::construct(
        std::piecewise_construct, default_signal_pool_size, [](signal_t& signal) {
            if(registration::get_fini_status() == 0) construct_hsa_signal(signal, 0, 0, nullptr, 0);
        });

    return pool;
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
    // checks if pool exists without constructing it if it doesn't exist
    if(!signal_pool_exists()) return;

    if(auto* pool = get_signal_pool(); pool != nullptr)
    {
        // only report once
        static auto _once = std::once_flag{};
        std::call_once(_once, [&]() { ROCP_INFO << pool->get_usage_report(); });

        // A checked-out entry keeps a pointer to this pool, and release() writes through it, so
        // tearing the storage down while one is borrowed leaves that release with a dangling pool.
        // clear_if_idle() performs the check and the teardown under one lock; if an entry is still
        // borrowed the pool is kept intact and the owning handler's runtime reclaims its signal.
        if(!pool->clear_if_idle([](auto& signal) { Queue::destroy_signal(&signal); }))
        {
            ROCP_WARNING << "signal pool still has entries in use; deferring pool teardown";
        }
    }
}

}  // namespace hsa
}  // namespace rocprofiler
