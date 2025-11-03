// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <atomic>

namespace rocprofiler
{
namespace thread_trace
{
/// RAII wrapper around an HSA signal used to synchronize packet submission.
class Signal
{
public:
    Signal(hsa_ext_amd_aql_pm4_packet_t* packet);
    Signal();
    ~Signal();
    Signal(Signal& other) = delete;
    Signal& operator=(Signal& other) = delete;

    void WaitOn() const;

    hsa_signal_t getSignal() const { return signal; }
    void         reset();

private:
    hsa_signal_t      signal{};
    std::atomic<bool> released{false};
};

/// Helper queue that owns the async DMA path used by thread trace copies.
class HsaATTQueue
{
    using code_object_id_t = uint64_t;

public:
    HsaATTQueue(const hsa::AgentCache& agent, size_t triple_buffer_size);
    ~HsaATTQueue();
    HsaATTQueue(HsaATTQueue& other) = delete;
    HsaATTQueue& operator=(HsaATTQueue& other) = delete;

    std::unique_ptr<Signal> Submit(hsa_ext_amd_aql_pm4_packet_t* packet, bool bWait) const;
    virtual void            Submit(hsa_ext_amd_aql_pm4_packet_t* packet, Signal* completion) const;

    /// Enqueues a sequence of packets and returns the completion signal of the last entry.
    template <typename VecType>
    std::unique_ptr<Signal> SubmitAndSignalLast(VecType vec)
    {
        for(size_t i = 0; i < vec.size(); i++)
        {
            auto sig = Submit(&vec.at(i), i == vec.size() - 1);
            if(sig) return sig;
        }
        return nullptr;
    }

    std::array<void*, 3> get_triple_buffer_memory() const { return triple_buffer_memory; }

    const rocprofiler_agent_id_t agent_id;
    const size_t                 buffer_size;

    const hsa_agent_t hsa_agent;
    const hsa_agent_t near_cpu;

protected:
    hsa_queue_t*         queue{nullptr};
    std::array<void*, 3> triple_buffer_memory{};
};

};  // namespace thread_trace
};  // namespace rocprofiler
