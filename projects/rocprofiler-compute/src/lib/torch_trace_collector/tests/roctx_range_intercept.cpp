// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "roctx_range_intercept.h"

#include <mutex>

namespace
{

std::mutex                       recording_mutex;
bool                             recording   = false;
bool                             reject_push = false;
roctx_range_intercept::Recording recorded;

}  // namespace

namespace roctx_range_intercept
{

void start_recording()
{
    const std::lock_guard<std::mutex> lock(recording_mutex);
    recorded    = {};
    reject_push = false;
    recording   = true;
}

void fail_next_push()
{
    const std::lock_guard<std::mutex> lock(recording_mutex);
    reject_push = true;
}

Recording stop_recording()
{
    const std::lock_guard<std::mutex> lock(recording_mutex);
    recording = false;
    return recorded;
}

}  // namespace roctx_range_intercept

extern "C"
{
int __real_roctxRangePushA(const char* message);
int __real_roctxRangePop();

int __wrap_roctxRangePushA(const char* message)
{
    const std::lock_guard<std::mutex> lock(recording_mutex);
    if (reject_push)
    {
        reject_push = false;
        return -1;
    }
    const int result = __real_roctxRangePushA(message);
    if (recording && result >= 0 && message != nullptr)
    {
        recorded.messages.emplace_back(message);
    }
    return result;
}

int __wrap_roctxRangePop()
{
    const std::lock_guard<std::mutex> lock(recording_mutex);
    if (recording)
    {
        ++recorded.pops;
    }
    return __real_roctxRangePop();
}
}
