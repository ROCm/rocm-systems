// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifdef hip_event_tracing_client_EXPORTS
#    define CLIENT_API __attribute__((visibility("default")))
#else
#    define CLIENT_API
#endif

#include <cstdint>

namespace client
{
void
setup() CLIENT_API;

void
shutdown() CLIENT_API;

void
start() CLIENT_API;

void
stop() CLIENT_API;

/// Force delivery of any buffered records collected so far. Buffered tracing is otherwise
/// delivered only when the buffer fills or at finalization, which would push every buffer
/// record to the end of the run; flushing at a known point keeps them interleaved with the
/// application narration that produced them.
void
flush() CLIENT_API;

/// Emit a line of application narration into the same timestamped, line-atomic output
/// stream used for the trace records. Sharing the stream and its lock with the tracer is
/// what puts narration in order relative to the records it produces.
void
narrate(const char* msg) CLIENT_API;

/// Same as narrate(), but framed by rules so that the major phases of the run stand out.
void
banner(const char* msg) CLIENT_API;

/// Running count of HIP_EVENT_WAIT barrier completions (::ROCPROFILER_CALLBACK_PHASE_NONE)
/// the tool has observed. The deferred-wait case uses this to notice that its long kernel
/// finished too early for a wait barrier to have been needed at all.
///
/// The completion callback is delivered on the HSA async signal handler thread, so this
/// count can lag the hipStreamSynchronize that guarantees the barrier itself has run.
/// Poll it rather than sampling it once.
uint64_t
wait_barriers_completed() CLIENT_API;
}  // namespace client
