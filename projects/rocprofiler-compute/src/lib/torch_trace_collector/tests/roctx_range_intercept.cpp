// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Link-wrap of roctxRangePushA for test-torch-trace-collector.

#include "roctx_range_intercept.h"

#include <mutex>
#include <string>
#include <vector>

namespace
{

std::mutex               recording_mutex;
bool                     recording = false;
std::vector<std::string> recorded;

}  // namespace

namespace roctx_range_intercept
{

void start()
{
    std::lock_guard<std::mutex> lock(recording_mutex);
    recorded.clear();
    recording = true;
}

std::vector<std::string> stop()
{
    std::lock_guard<std::mutex> lock(recording_mutex);
    recording = false;
    return recorded;
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
            recorded.emplace_back(message);
        }
    }
    return __real_roctxRangePushA(message);
}

}
