// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "roctx_range_intercept.h"

#include <mutex>

namespace
{

std::mutex               recording_mutex;
bool                     recording = false;
std::vector<std::string> recorded_messages;

}  // namespace

namespace roctx_range_intercept
{

void start_recording()
{
    std::lock_guard<std::mutex> lock(recording_mutex);
    recorded_messages.clear();
    recording = true;
}

std::vector<std::string> stop_recording()
{
    std::lock_guard<std::mutex> lock(recording_mutex);
    recording = false;
    return recorded_messages;
}

}  // namespace roctx_range_intercept

extern "C"
{

int __real_roctxRangePushA(const char* message);

int __wrap_roctxRangePushA(const char* message)
{
    {
        std::lock_guard<std::mutex> lock(recording_mutex);
        if (recording && message != nullptr)
        {
            recorded_messages.emplace_back(message);
        }
    }
    return __real_roctxRangePushA(message);
}

}
