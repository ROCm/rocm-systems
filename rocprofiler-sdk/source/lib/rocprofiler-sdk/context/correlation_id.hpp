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

#include "lib/common/mpl.hpp"
#include "lib/rocprofiler-sdk/external_correlation.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace context
{
struct correlation_id
{
    // reference count starts at 5:
    // - decrement after begin callback/buffer API
    // - decrement after end callback/buffer API
    // - decrement after kernel dispatch/HW counters
    // - if PC sampling is not enabled, we can "retire" correlation id at ref count at 2
    // - if PC sampling is enabled, we decrement after each HSA buffer flush once ref count hits 2
    //   - after the kernel dispatch completes, we know no more PC samples will be generated and
    //     thus, after two HSA buffer flushes, we will have received all the PC samples for
    //     the
    correlation_id(uint32_t _cnt, rocprofiler_thread_id_t _tid, uint64_t _internal) noexcept
    : thread_idx{_tid}
    , internal{_internal}
    , m_ref_count{_cnt}
    {}

    correlation_id()                              = default;
    ~correlation_id()                             = default;
    correlation_id(correlation_id&& val) noexcept = delete;
    correlation_id(const correlation_id&)         = delete;

    correlation_id& operator=(const correlation_id&) = delete;
    correlation_id& operator=(correlation_id&&) noexcept = delete;

    rocprofiler_thread_id_t thread_idx = 0;
    uint64_t                internal   = 0;
    uint64_t                ancestor   = 0;

    uint32_t get_ref_count() const { return m_ref_count.load(); }
    uint32_t add_ref_count();
    uint32_t sub_ref_count();

    uint32_t get_kern_count() const { return m_kern_count.load(); }
    uint32_t add_kern_count();
    uint32_t sub_kern_count();

private:
    std::atomic<uint32_t> m_kern_count = {0};
    std::atomic<uint32_t> m_ref_count  = {0};
};

// latest correlation id for thread
correlation_id*
get_latest_correlation_id();

const correlation_id*
pop_latest_correlation_id(correlation_id*);

// push correlation id
correlation_id*
push_correlation_id(correlation_id*);

// dump the cid stack for debugging
void
dump_correlation_stack(const char*);

void
correlation_id_finalize();

/// permits tools opportunity to modify the correlation id based on the domain, op, and
/// the rocprofiler generated correlation id
struct correlation_tracing_service
{
    external_correlation::external_correlation external_correlator = {};
    static correlation_id*                     construct(uint32_t init_ref_count);
};
}  // namespace context
}  // namespace rocprofiler
