// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"

#include <pthread.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace rocprofsys::sampling
{

// sampling_duration_controller — runs a background thread that fires a
// callback after a configured duration, handling spurious wakeups internally.
// DEC-5: DurationMutex is order-index 3.
template <class ClockPolicy>
class sampling_duration_controller
{
public:
    using shutdown_callback = std::function<void()>;

    // Callback is supplied at construction time.
    explicit sampling_duration_controller(shutdown_callback callback)
    : callback_(std::move(callback))
    {}

    ~sampling_duration_controller() { stop(); }

    sampling_duration_controller(sampling_duration_controller const&)            = delete;
    sampling_duration_controller& operator=(sampling_duration_controller const&) = delete;

    // Start the duration thread. No-op if duration_sec <= 0 or already started.
    // Fires callback_ once when the deadline elapses.
    void start(double duration_sec);

    // Notify the duration thread (e.g. to let it observe shutdown state).
    void notify() noexcept;

    // Stop and join the duration thread. Idempotent.
    void stop() noexcept;

    // Returns true if the duration timer has already fired.
    [[nodiscard]] bool is_disabled() const noexcept
    {
        return disabled_.load(std::memory_order_acquire);
    }

    // Synchronously check whether the deadline has elapsed and fire the
    // callback if so. Used when the clock is advanced externally (e.g.
    // fake_clock in tests) rather than waiting for the background thread.
    void check_deadline();

private:
    [[nodiscard]] bool deadline_reached() const noexcept;

    // Set disabled flag, emit L09 log line, and invoke callback exactly once.
    // Shared between the production deadline-thread path and check_deadline().
    void fire_deadline_callback();

    shutdown_callback                     callback_;
    std::atomic<bool>                     disabled_{ false };
    std::mutex                            mutex_;
    std::condition_variable               cv_;
    std::unique_ptr<std::thread>          thread_;
    bool                                  stop_requested_ = false;
    double                                duration_sec_   = 0.0;
    std::chrono::steady_clock::time_point deadline_{};
};

template <class ClockPolicy>
void
sampling_duration_controller<ClockPolicy>::fire_deadline_callback()
{
    // Set disabled exactly once; the bool result of exchange() prevents a
    // double-fire across the production thread and a concurrent check_deadline().
    if(disabled_.exchange(true, std::memory_order_acq_rel)) return;

    LOG_INFO("Sampling duration of {:.6f} seconds has elapsed. "
             "Shutting down sampling...",
             duration_sec_);
    if(callback_) callback_();
}

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
        ::pthread_setname_np(::pthread_self(), "omni.samp.dur");

        sigset_t ss;
        sigemptyset(&ss);
        sigaddset(&ss, SIGPROF);
        sigaddset(&ss, SIGRTMIN);
        sigaddset(&ss, SIGRTMIN + 1);
        ::pthread_sigmask(SIG_BLOCK, &ss, nullptr);

        std::unique_lock<std::mutex> inner(mutex_);
        while(!stop_requested_)
        {
            auto status = cv_.wait_until(inner, deadline_);
            if(stop_requested_) break;

            if(status == std::cv_status::timeout)
            {
                inner.unlock();
                fire_deadline_callback();
                return;
            }
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
sampling_duration_controller<ClockPolicy>::check_deadline()
{
    ClockPolicy clk;
    if(clk.now_steady() >= deadline_) fire_deadline_callback();
}

}  // namespace rocprofsys::sampling
