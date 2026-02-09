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

#pragma once

#include "lib/common/logging.hpp"

#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

struct CoreApiTable;

namespace rocprofiler
{
namespace hsa
{
enum class wait_result
{
    completed,
    shutdown,
    timeout
};

class async_wait_manager
{
public:
    static async_wait_manager& instance();

    bool is_shutdown() const;
    void notify_shutdown();
    void reset();

private:
    std::atomic<bool> _shutdown{false};
};

/// Register for HSA_AMD_SYSTEM_ASYNC_HANDLER_DESTROY_EVENT via the internal AMD ext table.
void
async_wait_manager_init();

/// HSA signal wait with shutdown awareness. Polls the signal at poll_interval_ns intervals
/// and checks the shutdown flag between polls.
wait_result
wait_or_shutdown(hsa_signal_t           signal,
                 hsa_signal_condition_t cond,
                 hsa_signal_value_t     value,
                 const std::string&     callsite,
                 uint64_t               timeout_ns       = UINT64_MAX,
                 uint64_t               poll_interval_ns = 100000000ULL,
                 const CoreApiTable*    core_api         = nullptr);

/// Condition variable wait with shutdown awareness.
wait_result
wait_or_shutdown(std::condition_variable&      cv,
                 std::unique_lock<std::mutex>& lock,
                 std::function<bool()>         predicate,
                 const std::string&            callsite,
                 uint64_t                      timeout_ns = UINT64_MAX,
                 std::chrono::milliseconds     interval   = std::chrono::milliseconds{100});

/// Atomic wait with shutdown awareness. Polls the atomic value and yields between checks.
template <typename T>
wait_result
wait_or_shutdown(std::atomic<T>&    atomic_val,
                 T                  expected,
                 const std::string& callsite,
                 uint64_t           timeout_ns = UINT64_MAX)
{
    auto& mgr   = async_wait_manager::instance();
    auto  start = std::chrono::steady_clock::now();

    while(atomic_val.load(std::memory_order_acquire) != expected)
    {
        if(mgr.is_shutdown())
        {
            ROCP_WARNING << "atomic wait interrupted by HSA async handler shutdown at " << callsite;
            return wait_result::shutdown;
        }
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
        if(timeout_ns != UINT64_MAX && static_cast<uint64_t>(elapsed) >= timeout_ns)
        {
            ROCP_ERROR << "atomic wait timed out at " << callsite;
            return wait_result::timeout;
        }
        std::this_thread::yield();
    }
    return wait_result::completed;
}

}  // namespace hsa
}  // namespace rocprofiler
