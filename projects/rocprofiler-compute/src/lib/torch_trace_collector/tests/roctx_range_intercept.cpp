// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "roctx_range_intercept.h"

#include <mutex>

namespace
{

struct RecordingState
{
    std::mutex                       mutex;
    bool                             recording   = false;
    bool                             reject_push = false;
    roctx_range_intercept::Recording recorded;
};

// ROCTX range callbacks have no user-data parameter; tests reset this state.
RecordingState& recording_state() noexcept
{
    static RecordingState state;
    return state;
}

}  // namespace

namespace roctx_range_intercept
{

void start_recording()
{
    auto&                             state = recording_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.recorded    = {};
    state.reject_push = false;
    state.recording   = true;
}

void fail_next_push()
{
    auto&                             state = recording_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.reject_push = true;
}

Recording stop_recording()
{
    auto&                             state = recording_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    state.recording = false;
    return state.recorded;
}

}  // namespace roctx_range_intercept

extern "C"
{
int __real_roctxRangePushA(const char* message);
int __real_roctxRangePop();

int __wrap_roctxRangePushA(const char* message)
{
    auto&                             state = recording_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (state.reject_push)
    {
        state.reject_push = false;
        return -1;
    }
    const int result = __real_roctxRangePushA(message);
    if (state.recording && result >= 0 && message != nullptr)
    {
        state.recorded.messages.emplace_back(message);
    }
    return result;
}

int __wrap_roctxRangePop()
{
    auto&                             state = recording_state();
    const std::lock_guard<std::mutex> lock(state.mutex);
    if (state.recording)
    {
        ++state.recorded.pops;
    }
    return __real_roctxRangePop();
}
}
