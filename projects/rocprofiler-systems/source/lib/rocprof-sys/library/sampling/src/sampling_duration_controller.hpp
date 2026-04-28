// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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

    // Callback is supplied at construction time (makes test injection cleaner).
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

    // ── Test seams (no-ops in production) ────────────────────────────────────
    // tick_for_test: directly runs one iteration of the deadline check
    // (tests with fake_clock call this instead of waiting for the real thread).
    void tick_for_test();

    // set_finalized_for_test: inject a "process finalized" flag to break the loop.
    void set_finalized_for_test(bool v) noexcept { test_finalized_ = v; }

private:
    [[nodiscard]] bool deadline_reached() const noexcept;

    shutdown_callback                     callback_;
    std::atomic<bool>                     disabled_{ false };
    std::mutex                            mutex_;
    std::condition_variable               cv_;
    std::unique_ptr<std::thread>          thread_;
    bool                                  stop_requested_ = false;
    bool                                  test_finalized_ = false;
    double                                duration_sec_   = 0.0;
    std::chrono::steady_clock::time_point deadline_{};
};

}  // namespace rocprofsys::sampling

#include "sampling_duration_controller_impl.hpp"
