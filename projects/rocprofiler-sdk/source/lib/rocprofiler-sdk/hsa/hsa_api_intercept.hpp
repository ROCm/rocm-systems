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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <hsa/hsa_api_trace.h>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief Layer 4 of the queue_intercept refactor: HSA core API interposition.
 *
 * Owns the wrapper functions installed into the HSA CoreApiTable (add/store/cas/
 * load write-index variants and signal store variants) and the saved
 * next-in-chain table. The wrappers delegate to the cursor/doorbell plumbing
 * in queue_intercept.cpp and to the registry in queue_state_registry.cpp.
 */

/**
 * @brief Install interposition wrappers into the HSA core API table
 *
 * Saves original function pointers and replaces them with wrappers that
 * route through the SDK's write-pointer virtualization when the queue is
 * tracked, or fall through to the original HSA implementation otherwise.
 *
 * @param core_table The HSA core API table to intercept
 */
void
install_intercept(CoreApiTable& core_table);

/**
 * @brief Query whether inline interception is currently enabled
 */
bool
is_intercepting_inline();

/**
 * @brief Disable inline queue interception and clear tracked state
 *
 * This leaves the wrapped function pointers installed but removes all tracked
 * queue state so wrappers always pass through to the next function table.
 * Intended for finalization to avoid teardown-order hazards in static objects.
 */
void
shutdown_intercept();

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
