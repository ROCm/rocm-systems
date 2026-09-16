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
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

namespace rocprofiler
{
namespace hsa
{
namespace
{
struct signal_pool_state
{
    std::mutex                     mutex{};
    std::shared_ptr<signal_pool_t> current{};
    bool                           retired = false;
};

signal_pool_state&
get_pool_state()
{
    static auto* state = common::static_object<signal_pool_state>::construct();
    return *state;
}

std::shared_ptr<signal_pool_t>
make_signal_pool()
{
    constexpr size_t default_signal_pool_size = 1 << 12;
    auto             destroy_signal           = get_core_table()->hsa_signal_destroy_fn;
    return std::shared_ptr<signal_pool_t>{
        new signal_pool_t{std::piecewise_construct,
                          default_signal_pool_size,
                          [](signal_t& signal) { construct_hsa_signal(signal); }},
        [destroy_signal](signal_pool_t* pool) {
            if(internal_threading::fork_stale()) return;
            pool->for_each([destroy_signal](auto& object) {
                if(object.in_use())
                {
                    ROCP_WARNING << "Retaining HSA signal for unfinished GPU work: "
                                 << object.get().value.handle;
                    return;
                }
                if(object.get().value.handle != 0 && destroy_signal)
                    destroy_signal(object.get().value);
            });
            delete pool;
        }};
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

std::shared_ptr<signal_pool_t>
get_signal_pool()
{
    if(internal_threading::fork_stale() || registration::get_fini_status() != 0) return {};
    auto& state = get_pool_state();
    auto  lock  = std::lock_guard{state.mutex};
    if(!state.current && !state.retired) state.current = make_signal_pool();
    return state.current;
}

pooled_signal_ptr_t
acquire_pooled_signal(hsa_signal_value_t initial_value)
{
    auto pool = get_signal_pool();
    if(!pool) return {};
    auto& signal = pool->acquire(construct_hsa_signal, initial_value, 0, nullptr, 0);
    return pooled_signal_ptr_t{pool, &signal};
}

void
signal_pool_init()
{
    if(internal_threading::fork_stale() || registration::get_fini_status() != 0) return;
    auto& state   = get_pool_state();
    auto  lock    = std::lock_guard{state.mutex};
    state.retired = false;
    if(!state.current) state.current = make_signal_pool();
}

void
signal_pool_fini()
{
    if(internal_threading::fork_stale() || registration::get_fini_status() > 0) return;
    auto& state   = get_pool_state();
    auto  retired = std::shared_ptr<signal_pool_t>{};
    {
        auto lock     = std::lock_guard{state.mutex};
        retired       = std::move(state.current);
        state.retired = true;
    }
    if(retired)
    {
        static auto once = std::once_flag{};
        std::call_once(once, [&]() { ROCP_INFO << retired->get_usage_report(); });
    }
}

}  // namespace hsa
}  // namespace rocprofiler
