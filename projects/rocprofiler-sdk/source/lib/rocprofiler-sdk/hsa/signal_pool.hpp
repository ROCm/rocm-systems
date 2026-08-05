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
 * @brief A function passed to the signal pool constructor and acquire function.
 *
 * Example:
 * @code{.cpp}
 *      pool->acquire(construct_hsa_signal, 0, 0, nullptr, 0);
 * @endcode
 *
 * @param signal
 * @param initial_value
 * @param num_consumers
 * @param consumers
 * @param attributes
 * @return signal_t&
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
