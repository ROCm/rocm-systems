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

#include "lib/rocprofiler-sdk/kfd/signal_less.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_correlation.hpp"

namespace rocprofiler
{
namespace kfd
{
signal_less_hub_t&
signal_less_hub()
{
    static auto*& _v = common::static_object<signal_less_hub_t>::construct();
    return *_v;
}

bool
signal_less_feature_enabled()
{
    // Read once: the answer must not change under a running process, and the
    // enqueue path cannot afford an env lookup per batch.
    static const bool _enabled = []() {
        auto _v = common::get_env_optional("ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS");
        if(!_v || !parse_signal_less_env(*_v)) return false;
        ROCP_WARNING << "KFD dispatch-log: signal-less kernel-dispatch completion requested via "
                        "ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS; the feature is still being "
                        "landed and remains inactive until every stage is present";
        return true;
    }();
    return _enabled;
}

bool
kfd_selection_enabled()
{
    // Emitting a KFD timestamp requires the whole signal-less path, not just the
    // env flag: until it is fully wired this stays false and every dispatch keeps
    // the Phase 1 behavior (HSA timestamps, signals retained).
    return signal_less_feature_enabled() && signal_less_fully_wired();
}
}  // namespace kfd
}  // namespace rocprofiler
