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

#include <condition_variable>
#include <mutex>
#include <thread>

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
    consumer_thread_t(consume_func_t func) { this->consume_fn = func; }
    virtual ~consumer_thread_t() { exit(); }

    void start()
    {
        std::unique_lock<std::mutex> lk(mut);

        if(valid.exchange(true)) return;
        exited.store(false);

        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        consumer = std::thread{&consumer_thread_t::consumer_loop, this};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
    }

    void exit()
    {
        std::unique_lock<std::mutex> lk(mut);

        valid.store(false);
        cv.notify_all();

        if(!exited) cv.wait(lk, [&] { return exited.load(); });
        if(consumer.joinable()) consumer.join();
    }

    void add(DataType&& params)
    {
        std::unique_lock<std::mutex> lk(mut);

        if(read_ptr + buffer.size() <= write_ptr || !valid)
        {
            // If not possible to use consumer thread, proccess with this thread
            consume_fn(std::move(params));
            return;
        }

        buffer.at(write_ptr % buffer.size()) = std::move(params);
        write_ptr.fetch_add(1);
        cv.notify_all();
    }

protected:
    void consumer_loop()
    {
        while(true)
        {
            while(read_ptr == write_ptr)
            {
                std::unique_lock<std::mutex> lk(mut);
                cv.wait(lk, [&] { return read_ptr != write_ptr || !valid; });
                if(!valid && read_ptr == write_ptr)
                {
                    exited.store(true);
                    cv.notify_all();
                    return;
                }
            }

            auto retrieved = std::move(buffer.at(read_ptr % buffer.size()));
            read_ptr.fetch_add(1);
            consume_fn(std::move(retrieved));
        }
    }

    consume_func_t             consume_fn;
    std::atomic<bool>          valid{false};
    std::atomic<bool>          exited{true};
    std::mutex                 mut;
    std::atomic<size_t>        write_ptr{0};
    std::atomic<size_t>        read_ptr{0};
    std::array<DataType, SIZE> buffer;
    std::thread                consumer;
    std::condition_variable    cv;
};

}  // namespace counters
}  // namespace rocprofiler
