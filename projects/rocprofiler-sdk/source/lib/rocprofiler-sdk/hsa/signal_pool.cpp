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
}  // namespace

signal_t&
construct_hsa_signal(signal_t&          signal,
                     hsa_signal_value_t initial_value,
                     uint32_t           num_consumers,
                     const hsa_agent_t* consumers,
                     uint64_t           attributes)
{
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

        // always try to clear
        pool->clear([](auto& signal) { Queue::destroy_signal(&signal); });
    }
}

}  // namespace hsa
}  // namespace rocprofiler
