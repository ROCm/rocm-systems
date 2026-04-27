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

#define LTTNG_UST_TRACEPOINT_CREATE_PROBES
#define LTTNG_UST_TRACEPOINT_DEFINE
#include "lib/common/container/lttng_record_header_buffer.hpp"

namespace rocprofiler::common::container
{
lttng_record_header_buffer::lttng_record_header_buffer(size_t nbytes) { allocate(nbytes); }

lttng_record_header_buffer::lttng_record_header_buffer(
    lttng_record_header_buffer&& rhs) noexcept
{
    this->operator=(std::move(rhs));
}

lttng_record_header_buffer&
lttng_record_header_buffer::operator=(lttng_record_header_buffer&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_buffer   = std::move(rhs.m_buffer);
        m_sequence = rhs.m_sequence.exchange(0, std::memory_order_acq_rel);
    }
    return *this;
}

void
lttng_record_header_buffer::emit(uint32_t category,
                                 uint32_t kind,
                                 uint64_t hash,
                                 size_t   payload_size,
                                 size_t   payload_alignment)
{
    auto sequence = m_sequence.fetch_add(1, std::memory_order_acq_rel);
    lttng_ust_tracepoint(rocprofiler_sdk_buffer,
                         record,
                         category,
                         kind,
                         hash,
                         payload_size,
                         payload_alignment,
                         sequence);
}
}  // namespace rocprofiler::common::container
