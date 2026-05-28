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
        exited = 0;

        for (auto& consumer : consumers)
        {
            internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
            consumer = std::thread{&consumer_thread_t::consumer_loop, this};
            internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
        }
    }

    void exit()
    {
        std::unique_lock<std::mutex> lk(mut);

        valid.store(false);
        cv.notify_all();
        cv.wait(lk, [&] { return exited >= consumers.size(); });
        for(auto& consumer : consumers) if(consumer.joinable()) consumer.join();
    }

    void add(DataType&& params)
    {
        bool selfconsume = true;
        {
            auto lk = std::unique_lock{mut};
            if(valid && read_ptr + buffer.size() > write_ptr)
            {
                buffer.at((write_ptr++) % buffer.size()) = std::move(params);
                selfconsume                              = false;
            }
        }
        if(selfconsume)
            consume_fn(std::move(params));
        else
            cv.notify_all();
    }

protected:
    void consumer_loop()
    {
        while(true)
        {
            DataType retrieved{};
            {
                auto lk = std::unique_lock{mut};
                cv.wait(lk, [&] { return read_ptr != write_ptr || !valid; });
                if(read_ptr == write_ptr)
                {
                    exited++;
                    cv.notify_all();
                    return;
                }

                retrieved = std::move(buffer.at((read_ptr++) % buffer.size()));
            }

            consume_fn(std::move(retrieved));
        }
    }

    size_t exited{0};
    size_t write_ptr{0};
    size_t read_ptr{0};

    consume_func_t             consume_fn{};
    std::atomic<bool>          valid{false};
    std::mutex                 mut;
    std::array<DataType, SIZE> buffer{};
    std::array<std::thread, 2> consumers{};
    std::condition_variable    cv{};
};

}  // namespace counters
}  // namespace rocprofiler
