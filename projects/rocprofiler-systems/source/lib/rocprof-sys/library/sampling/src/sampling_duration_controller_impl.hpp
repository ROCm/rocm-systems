// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Implementation body for sampling_duration_controller<ClockPolicy>.

#include "logger/debug.hpp"

namespace rocprofsys::sampling
{

template <class ClockPolicy>
void
sampling_duration_controller<ClockPolicy>::start(double duration_sec)
{
    if(duration_sec <= 0.0) return;

    std::unique_lock<std::mutex> lk(mutex_);
    if(thread_) return;  // already started

    duration_sec_ = duration_sec;
    LOG_INFO("Sampling will be disabled after {:.6f} seconds", duration_sec);

    ClockPolicy clk;
    deadline_ = clk.now_steady() +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(duration_sec));

    stop_requested_ = false;
    thread_         = std::make_unique<std::thread>([this]() {
        std::unique_lock<std::mutex> inner(mutex_);
        while(!stop_requested_)
        {
            if(test_finalized_) break;

            auto status = cv_.wait_until(inner, deadline_);
            if(stop_requested_) break;
            if(test_finalized_) break;

            if(status == std::cv_status::timeout)
            {
                disabled_.store(true, std::memory_order_release);
                LOG_INFO("Sampling duration of {:.6f} seconds has elapsed. "
                                         "Shutting down sampling...",
                                 duration_sec_);
                inner.unlock();
                if(callback_) callback_();
                return;
            }
            // Spurious wakeup
            LOG_WARNING("Spurious wakeup of sampling duration thread...");
        }
    });
}

template <class ClockPolicy>
void
sampling_duration_controller<ClockPolicy>::notify() noexcept
{
    cv_.notify_one();
}

template <class ClockPolicy>
void
sampling_duration_controller<ClockPolicy>::stop() noexcept
{
    {
        std::lock_guard<std::mutex> lk(mutex_);
        stop_requested_ = true;
    }
    cv_.notify_one();
    if(thread_ && thread_->joinable()) thread_->join();
    thread_.reset();
}

template <class ClockPolicy>
void
sampling_duration_controller<ClockPolicy>::tick_for_test()
{
    // Directly check deadline against the fake clock (no sleeping).
    // Used exclusively in tests that control time via fake_clock.
    if(test_finalized_) return;

    ClockPolicy clk;
    if(clk.now_steady() >= deadline_)
    {
        if(!disabled_.exchange(true, std::memory_order_acq_rel))
        {
            LOG_INFO("Sampling duration of {:.6f} seconds has elapsed. "
                     "Shutting down sampling...",
                     duration_sec_);
            if(callback_) callback_();
        }
    }
    else
    {
        LOG_WARNING("Spurious wakeup of sampling duration thread...");
    }
}

}  // namespace rocprofsys::sampling
