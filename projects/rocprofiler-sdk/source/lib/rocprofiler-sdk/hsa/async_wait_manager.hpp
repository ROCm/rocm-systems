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

#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string_view>

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

/// Register for HSA_AMD_SYSTEM_ASYNC_HANDLER_DESTROY_EVENT via the internal AMD ext table.
void
async_wait_manager_init();

/// Notify that the async handler has been destroyed (for testing).
void
notify_async_shutdown();

/// Reset the async shutdown flag (for testing).
void
reset_async_shutdown();

/// Check if the async handler has been destroyed.
bool
is_async_shutdown();

/// Check if async_wait_manager_init() has been called (i.e. HSA was loaded).
bool
is_async_wait_initialized();

/// Register a callback to be invoked when HSA async handler shutdown is detected.
/// Used for deferred finalization: if finalize() is called before HSA shuts down,
/// it registers a callback here and returns. When the event fires, the callback
/// triggers finalization at the right time.
using shutdown_callback_t = std::function<void()>;

void
register_shutdown_callback(shutdown_callback_t callback);

/// HSA signal wait with shutdown awareness. Polls the signal at poll_interval_ns intervals
/// and checks the shutdown flag between polls.
wait_result
wait_or_shutdown(hsa_signal_t           signal,
                 hsa_signal_condition_t cond,
                 hsa_signal_value_t     value,
                 std::string_view       callsite,
                 uint64_t               timeout_ns       = UINT64_MAX,
                 uint64_t               poll_interval_ns = 100000000ULL,
                 const CoreApiTable*    core_api         = nullptr);

/// Condition variable wait with shutdown awareness.
wait_result
wait_or_shutdown(std::condition_variable&      cv,
                 std::unique_lock<std::mutex>& lock,
                 const std::function<bool()>&  predicate,
                 std::string_view              callsite,
                 uint64_t                      timeout_ns = UINT64_MAX,
                 std::chrono::milliseconds     interval   = std::chrono::milliseconds{100});

/// Atomic wait with shutdown awareness. Polls the atomic value and yields between checks.
template <typename T>
wait_result
wait_or_shutdown(std::atomic<T>&  atomic_val,
                 T                expected,
                 std::string_view callsite,
                 uint64_t         timeout_ns = UINT64_MAX);

}  // namespace hsa
}  // namespace rocprofiler
