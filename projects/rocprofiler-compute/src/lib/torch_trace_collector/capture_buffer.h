// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "synchronized.hpp"

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// Buffer of emitted wire strings for the test capture hook.
class CaptureBuffer
{
public:
    void capture(const std::string& wire_string)
    {
        // Read outside the lock so the callback path takes no lock while
        // capture is off, which is the case outside the tests.
        if (!capturing_.load(std::memory_order_relaxed))
            return;
        captured_.wlock(
            [&](std::vector<std::string>& captured)
            {
                if (captured.size() < kCap)
                {
                    captured.push_back(wire_string);
                }
            });
    }

    void start()
    {
        captured_.wlock(
            [this](std::vector<std::string>& captured)
            {
                captured.clear();
                capturing_.store(true, std::memory_order_release);
            });
    }

    std::vector<std::string> stop()
    {
        capturing_.store(false, std::memory_order_release);
        return captured_.wlock(
            [](std::vector<std::string>& captured)
            {
                std::vector<std::string> out = captured;
                captured.clear();
                return out;
            });
    }

private:
    static constexpr std::size_t kCap = 4096;

    std::atomic<bool>                        capturing_{false};
    synchronized_t<std::vector<std::string>> captured_;
};

// Start and stop recording of the emitted wire strings.
void                     start_capture();
std::vector<std::string> stop_capture();

}  // namespace torch_trace_collector::detail
