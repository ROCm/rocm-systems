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

// the other half of the pooled signal's lifecycle: construct_hsa_signal creates the handle on
// acquire, this destroys it once pool<Tp>::clear() retires the object.
void
destroy_hsa_signal(signal_t& signal)
{
    // the handle is left in place rather than nulled. The only caller is pool::clear(), which
    // retires its storage instead of freeing it, so AsyncSignalHandler can still reach this
    // object through packet_data_t::pooled_signal afterwards -- and it exempts the pooled
    // completion signal from destruction by comparing the dispatch-time copy of the handle
    // against this field. Nulling turns that exemption off and destroys the handle a second
    // time. Nor is that read a data race: the only other writer is construct_hsa_signal, which
    // pool::acquire() runs on every acquire, and a retired object can never be re-acquired.
    if(get_core_table() && get_core_table()->hsa_signal_destroy_fn)
        get_core_table()->hsa_signal_destroy_fn(signal.value);
}
}  // namespace

signal_t&
construct_hsa_signal(signal_t&          signal,
                     hsa_signal_value_t initial_value,
                     uint32_t           num_consumers,
                     const hsa_agent_t* consumers,
                     uint64_t           attributes)
{
    // pool<Tp>::acquire(FuncT&&, Args&&...) runs this on every acquire, reused objects
    // included, and nothing in the pool destroys a handle between acquires: release() is
    // bookkeeping only and destroy_hsa_signal() runs from clear(), which retires the object
    // rather than returning it to the free list. Creating unconditionally therefore overwrote a
    // live handle with a fresh one on every reuse and leaked the old one -- one signal per
    // intercepted dispatch, against a 4096 KFD event limit per process that HSA exhausts
    // silently. Reuse the handle instead. A null handle still takes the create path: it is the
    // pool's batch constructor that skips creating once finalization has started, and this call
    // is the lazy-creation path those objects depend on.
    if(signal.value.handle != 0) return signal;

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
        pool->clear(destroy_hsa_signal);
    }
}

}  // namespace hsa
}  // namespace rocprofiler
