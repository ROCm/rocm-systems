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

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace hip
{
namespace etw
{
// HIP API tracing on Windows is out-of-process: amdhip64 is an ETW provider and
// rocprofiler-sdk is the consumer. These open and close the consumer session and are
// reference counted, so every context that traces a HIP domain can call them independently.
//
// Declared on both platforms and no-ops off Windows so that the context start/stop call
// sites need no preprocessor guards.
rocprofiler_status_t
start_session();

rocprofiler_status_t
stop_session();

// The session hands its events over in one go when it is stopped, so unlike an in-process
// runtime it produces all of its records during teardown. Those records need active contexts
// and a live correlation service to land anywhere, which makes the start of finalization --
// before either is torn down -- the last moment this can happen. Idempotent: a subsequent
// stop_session() finds the session already closed.
void
drain_sessions();
}  // namespace etw
}  // namespace hip
}  // namespace rocprofiler
