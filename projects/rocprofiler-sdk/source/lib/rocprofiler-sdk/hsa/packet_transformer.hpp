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

#include <hsa/hsa.h>

#include <cstdint>
#include <functional>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
/**
 * @brief C-ABI trampoline for hsa_amd_queue_intercept_packet_writer.
 *
 * Reads a thread_local std::function slot set by a `ScopedWriter` and
 * forwards to it. This provides a stable function pointer that can be
 * handed to `Queue::invoke_write_interceptor`, while the underlying
 * writer body is free to capture arbitrary per-call state (cursor,
 * reservation, etc.) via a closure.
 *
 * If no writer is currently installed (i.e. called outside of any
 * `ScopedWriter` scope on this thread), the call is a silent no-op.
 */
void
packet_writer_trampoline(const void* pkts, uint64_t pkt_count);

/**
 * @brief RAII guard installing a thread_local writer closure.
 *
 * The previous thread_local writer (if any) is saved on construction and
 * restored on destruction, so nesting is safe — a callback invoked from
 * inside a WriteInterceptor that itself opens another `ScopedWriter`
 * scope will not clobber the outer scope's writer.
 *
 * This replaces the bare TLS globals `tls_cursor` / `tls_pkt_size`
 * previously used in `process_doorbell_impl` (fixes bug #10).
 */
class ScopedWriter
{
public:
    using writer_fn_t = std::function<void(const void*, uint64_t)>;

    explicit ScopedWriter(writer_fn_t fn);
    ~ScopedWriter();

    ScopedWriter(const ScopedWriter&)            = delete;
    ScopedWriter& operator=(const ScopedWriter&) = delete;
    ScopedWriter(ScopedWriter&&)                 = delete;
    ScopedWriter& operator=(ScopedWriter&&)      = delete;

private:
    writer_fn_t prev_;
};

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
