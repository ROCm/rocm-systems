// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/defines.hpp"

#include <functional>

namespace rocprofsys
{
namespace control
{
/// Callback function type for control events
using callback_t = std::function<void()>;

/// Register a callback to be triggered when tracing should start.
/// Called when:
/// - roctxRangeStartA matches a target region (ROCPROFSYS_TRACE_REGION)
/// - First target region becomes active (0→1 active regions)
void
register_start_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

/// Register a callback to be triggered when tracing should stop.
/// Called when:
/// - roctxRangeStop exits the last active target region (1→0 active regions)
void
register_stop_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

/// Register a callback to be triggered when tracing should pause.
/// Called when:
/// - roctxProfilerPause(0) is called
void
register_pause_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

/// Register a callback to be triggered when tracing should resume.
/// Called when:
/// - roctxProfilerResume(0) is called
void
register_resume_callback(callback_t callback) ROCPROFSYS_INTERNAL_API;

/// Setup the control client (rocprofiler-sdk client for marker watching).
/// Called during library initialization.
void
setup() ROCPROFSYS_INTERNAL_API;

/// Shutdown the control client.
/// Called during library finalization.
void
shutdown() ROCPROFSYS_INTERNAL_API;

}  // namespace control
}  // namespace rocprofsys
