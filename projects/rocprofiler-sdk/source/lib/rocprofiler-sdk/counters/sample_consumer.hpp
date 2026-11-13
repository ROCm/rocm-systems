// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

namespace rocprofiler
{
namespace counters
{
template <typename DataType>
class consumer_thread_t
{
    static constexpr size_t SIZE = 128;
    using consume_func_t         = std::function<void(DataType&&)>;

public:
    consumer_thread_t(consume_func_t func) { this->consume_fn = std::move(func); }
    virtual ~consumer_thread_t() { exit(); }

    void start()
    {
        std::unique_lock<std::mutex> lk(mut);

        if(valid.exchange(true)) return;
        exited.store(false);

        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        consumer = std::thread(&consumer_thread_t::consumer_loop, this);
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
    }

    void exit()
    {
        std::unique_lock<std::mutex> lk(mut);

        valid.store(false);
        cv.notify_all();
        // A drain waiting on a consumer that is shutting down would otherwise never be woken.
        drain_cv.notify_all();

        if(!exited) cv.wait(lk, [&] { return exited.load(); });
        if(consumer.joinable()) consumer.join();
    }

    void add(DataType&& params)
    {
        bool selfconsume = true;
        {
            auto lk = std::unique_lock{mut};
            if(valid && read_ptr + buffer.size() > write_ptr)
            {
                buffer.at(write_ptr.fetch_add(1) % buffer.size()) = std::move(params);
                selfconsume                                       = false;
            }
        }
        if(selfconsume)
            consume_fn(std::move(params));
        else
            cv.notify_all();
    }

    // Blocks until samples enqueued before this call have been consumed, or the timeout expires.
    // Stops nothing: samples arriving during the wait are post-call work and are not waited for.
    uint64_t drain(const uint64_t timeout_ns)
    {
        auto         lk              = std::unique_lock{mut};
        const size_t target          = write_ptr.load(std::memory_order_acquire);
        const auto   timeout_enabled = timeout_ns > 0;
        const auto   timeout =
            std::chrono::steady_clock::now() + std::chrono::nanoseconds{timeout_ns};
        const auto predicate = [&] {
            return consumed_ptr.load(std::memory_order_acquire) >= target ||
                   !valid.load(std::memory_order_acquire);
        };

        if(!timeout_enabled)
            drain_cv.wait(lk, predicate);
        else
            drain_cv.wait_until(lk, timeout, predicate);

        // Outstanding is what the consumer has NOT reached, so target minus consumed -- the other
        // order returns zero on exactly the timeout this exists to report. Computed on both arms:
        // the untimed wait also returns early on shutdown, leaving samples never consumed.
        const auto consumed = consumed_ptr.load(std::memory_order_relaxed);
        return target > consumed ? target - consumed : 0;
    }

    uint64_t peek_pending_records()
    {
        const auto target   = write_ptr.load(std::memory_order_relaxed);
        const auto consumed = consumed_ptr.load(std::memory_order_relaxed);
        return target > consumed ? target - consumed : 0;
    }

protected:
    void consumer_loop()
    {
        while(true)
        {
            {
                auto lk = std::unique_lock{mut};
                cv.wait(lk, [&] { return read_ptr != write_ptr || !valid; });
                if(read_ptr == write_ptr)
                {
                    exited.store(true);
                    cv.notify_all();
                    drain_cv.notify_all();
                    return;
                }
            }

            auto retrieved = std::move(buffer.at(read_ptr % buffer.size()));
            read_ptr.fetch_add(1);
            consume_fn(std::move(retrieved));

            // Counted under mut, after consume_fn: a drain holds mut while evaluating its
            // predicate, so incrementing outside the lock could land between that check and the
            // wait, losing the wakeup.
            {
                auto lk = std::unique_lock{mut};
                consumed_ptr.fetch_add(1, std::memory_order_release);
            }
            drain_cv.notify_all();
        }
    }

    consume_func_t      consume_fn;
    std::atomic<bool>   valid{false};
    std::atomic<bool>   exited{true};
    std::mutex          mut;
    std::atomic<size_t> write_ptr{0};
    std::atomic<size_t> read_ptr{0};
    // Pairs with write_ptr: advanced after consume_fn returns, where read_ptr advances before
    // it. A self-consumed sample enters no ring and is counted by neither.
    std::atomic<size_t>        consumed_ptr{0};
    std::array<DataType, SIZE> buffer;
    std::thread                consumer{};
    std::condition_variable    cv;
    std::condition_variable    drain_cv;
};

}  // namespace counters
}  // namespace rocprofiler
