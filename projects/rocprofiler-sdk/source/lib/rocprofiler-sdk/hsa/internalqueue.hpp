// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hsa/hsa_api_trace.h>
#include "lib/common/utility.hpp"

#include <atomic>
#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace internal_queue
{
void
initialize(HsaApiTable* table);

class Signal
{
public:
    Signal(void* packet);
    Signal(Signal&& other);
    ~Signal();

    Signal(Signal& other)       = delete;
    Signal(const Signal& other) = delete;
    Signal& operator=(Signal& other) = delete;
    Signal& operator=(const Signal& other) = delete;

    void WaitOn() const;

    hsa_signal_t      signal;
    std::atomic<bool> released{false};
};

class Queue
{
public:
    Queue(hsa_agent_t agent, size_t queue_size = 128);
    Queue(Queue&& other)
    : queue(other.queue)
    {
        other.queue = nullptr;
    }
    virtual ~Queue();

    Queue(Queue& other) = delete;
    Queue& operator=(Queue& other) = delete;

    std::unique_ptr<Signal> Submit(void* packet, bool bWait);

    template <typename VecType>
    [[nodiscard]] std::unique_ptr<Signal> SubmitAndSignalLast(VecType vec)
    {
        for(size_t i = 0; i < vec.size(); i++)
        {
            auto sig = Submit(&vec.at(i), i == vec.size() - 1);
            if(sig) return sig;
        }
        return nullptr;
    }

    hsa_queue_t* queue = nullptr;
};
}  // namespace internal_queue
}  // namespace hsa
}  // namespace rocprofiler
