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

#include "lib/rocprofiler-sdk/hsa/signal_monitor.hpp"

#include "lib/common/environment.hpp"

namespace rocprofiler::hsa
{
bool
evaluate_signal_condition(hsa_signal_condition_t condition,
                          hsa_signal_value_t     current,
                          hsa_signal_value_t     compare_value)
{
    switch(condition)
    {
        case HSA_SIGNAL_CONDITION_EQ: return current == compare_value;
        case HSA_SIGNAL_CONDITION_LT: return current < compare_value;
        default:
            // Phase 1 intentionally supports only EQ/LT; other conditions are treated as false.
            return false;
    }
}

SignalMonitorBackend
parse_signal_monitor_backend_env()
{
    auto backend = common::get_env("ROCPROF_SIGNAL_MONITOR_BACKEND", std::string{"auto"});
    if(backend == "poll") return SignalMonitorBackend::poll;
    if(backend == "ioctl") return SignalMonitorBackend::ioctl;
    return SignalMonitorBackend::auto_select;
}

std::shared_ptr<SignalMonitor>
create_signal_monitor(SignalMonitorBackend backend, const SignalMonitorConfig& cfg, SignalMonitorOps ops)
{
    (void) backend;
    (void) cfg;
    (void) ops;
    return {};
}
}  // namespace rocprofiler::hsa
