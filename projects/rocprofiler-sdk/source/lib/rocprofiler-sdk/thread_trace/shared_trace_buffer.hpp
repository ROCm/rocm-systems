// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
struct TraceMemoryPool;
}  // namespace hsa

namespace thread_trace
{
// Process-global manager of per-agent SQTT output buffers: one trace is active per
// agent at a time, so all contexts on an agent share the same buffer(s), sized to the
// largest requested buffer_size. Triple buffering uses one ring slot per buffer.

/// Record the max buffer size for @p agent; call for every context before the first
/// acquire so the shared buffer fits all contexts.
void
register_shared_buffer_size(hsa_agent_t agent, uint64_t buffer_size);

/// Return @p pool's agent's @p index -th shared buffer, allocating it (sized to the
/// agent max, at least @p size) on first use and reusing it after. nullptr on failure.
void*
acquire_shared_buffer(const hsa::TraceMemoryPool& pool, size_t index, uint64_t size);

/// Whether @p ptr is a manager-owned shared buffer; TraceMemoryPool::Free uses this to
/// skip it (freed once in free_shared_buffers()).
bool
is_shared_buffer(void* ptr);

/// Free all shared per-agent buffers. Called from thread_trace::finalize().
void
free_shared_buffers();

}  // namespace thread_trace
}  // namespace rocprofiler
