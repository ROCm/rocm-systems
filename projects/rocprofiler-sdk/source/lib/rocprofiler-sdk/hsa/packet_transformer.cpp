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

#include "lib/rocprofiler-sdk/hsa/packet_transformer.hpp"

#include <utility>

namespace rocprofiler
{
namespace hsa
{
namespace queue_intercept
{
namespace
{
thread_local ScopedWriter::writer_fn_t tls_writer;
}  // namespace

void
packet_writer_trampoline(const void* pkts, uint64_t pkt_count)
{
    if(tls_writer)
    {
        tls_writer(pkts, pkt_count);
    }
    // else: silent no-op. Should not happen in intended use — a writer must
    // be installed via ScopedWriter before any callback path that funnels
    // back through this trampoline.
}

ScopedWriter::ScopedWriter(writer_fn_t fn)
: prev_{std::move(tls_writer)}
{
    tls_writer = std::move(fn);
}

ScopedWriter::~ScopedWriter() { tls_writer = std::move(prev_); }

}  // namespace queue_intercept
}  // namespace hsa
}  // namespace rocprofiler
