// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace roctx_recordfn::detail
{

// Buffer of emitted wire strings for the test capture hook.
class CaptureBuffer
{
public:
    void capture(const std::string& wire_string)
    {
        if (!capturing_.load(std::memory_order_relaxed))
            return;
        std::lock_guard<std::mutex> guard(mutex_);
        if (captured_.size() < kCap)
        {
            captured_.push_back(wire_string);
        }
    }

    void start()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        captured_.clear();
        capturing_.store(true, std::memory_order_release);
    }

    std::vector<std::string> stop()
    {
        capturing_.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<std::string>    out = captured_;
        captured_.clear();
        return out;
    }

private:
    static constexpr std::size_t kCap = 4096;

    std::atomic<bool>        capturing_{false};
    std::mutex               mutex_;
    std::vector<std::string> captured_;
};

inline CaptureBuffer g_capture;

}  // namespace roctx_recordfn::detail
