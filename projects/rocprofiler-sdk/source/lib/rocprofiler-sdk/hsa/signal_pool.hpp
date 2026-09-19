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

#pragma once

#include "lib/common/container/pool.hpp"
#include "lib/rocprofiler-sdk/hsa/signal.hpp"

#include <rocprofiler-sdk/hsa.h>

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
/**
 * @brief Construct @p signal's HSA signal if it has none, and set its value to @p initial_value.
 *
 * Constructs a signal only when @p signal has none, i.e. when its handle is zero. An existing
 * handle is kept and reused rather than replaced, because nothing in the pool destroys a
 * handle between acquires and replacing it would leak the old one. Either way the signal's
 * value is @p initial_value on return, so a reused signal never carries the value its
 * previous user left behind.
 *
 * Written to be the callable given to pool<Tp>::acquire() and to the signal pool's batch
 * constructor. acquire() runs it on every acquire and not only on the objects it had to
 * create, which is what makes the two halves above the whole contract: idempotent on the
 * handle, authoritative on the value.
 *
 * @p num_consumers, @p consumers and @p attributes are hsa_amd_signal_create arguments and so
 * apply on the create path only. They cannot be changed on a signal that already exists, so
 * passing different ones for a non-zero handle has no effect.
 *
 * Example:
 * @code{.cpp}
 *      pool->acquire(construct_hsa_signal, 0, 0, nullptr, 0);
 * @endcode
 *
 * @param signal in/out; created when its handle is zero, otherwise reused
 * @param initial_value the value @p signal holds on return
 * @param num_consumers create path only: number of consumer agents
 * @param consumers create path only: consumer agents, or nullptr for all
 * @param attributes create path only: hsa_amd_signal_create attribute flags
 * @return signal_t& the same @p signal
 */
signal_t&
construct_hsa_signal(signal_t&          signal,
                     hsa_signal_value_t initial_value = 0,
                     uint32_t           num_consumers = 0,
                     const hsa_agent_t* consumers     = nullptr,
                     uint64_t           attributes    = 0);

/**
 * @brief Get the signal pool object
 *
 * @return common::container::pool<signal_t>*
 */
common::container::pool<signal_t>*
get_signal_pool();

/**
 * @brief Initialize the signal pool.
 *
 */
void
signal_pool_init();

/**
 * @brief Finalize the signal pool, destroying any remaining signals and printing a usage report.
 *
 */
void
signal_pool_fini();
}  // namespace hsa
}  // namespace rocprofiler
