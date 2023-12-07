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

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_set>

#include "correlation.hpp"

#define CHECK_PARSER(x)                                                                            \
    {                                                                                              \
        int val = (x);                                                                             \
        if(val != PCSAMPLE_STATUS_SUCCESS)                                                         \
        {                                                                                          \
            std::cerr << __FILE__ << ':' << __LINE__ << " Parser error: " << val << std::endl;     \
            exit(val);                                                                             \
        }                                                                                          \
    }

/**
 * Mimics the rocprofiler buffer sent to the parser.
 */
class MockRuntimeBuffer
{
public:
    MockRuntimeBuffer() { packets = {}; };

    //! Adds a packet to the buffer
    void submit(const packet_union_t& packet) { packets.push_back(packet); };

    //! Submits a "upcoming_samples_t" packet signaling the next num_samples packets are PC samples
    void genUpcomingSamples(int num_samples)
    {
        packet_union_t uni;
        ::memset(&uni, 0, sizeof(uni));
        uni.upcoming.type              = AMD_UPCOMING_SAMPLES;
        uni.upcoming.which_sample_type = AMD_SNAPSHOT_V1;
        uni.upcoming.num_samples       = num_samples;
        submit(uni);
    }

    std::vector<std::vector<pcsample_v1_t>> get_parsed_buffer(int GFXIP_MAJOR)
    {
        parsed_data = {};

        CHECK_PARSER(parse_buffer((generic_sample_t*) packets.data(),
                                  packets.size(),
                                  GFXIP_MAJOR,
                                  &alloc_parse_memory,
                                  this));

        return parsed_data;
    }

    static uint64_t alloc_parse_memory(pcsample_v1_t** sample, uint64_t req_size, void* userdata)
    {
        auto* buffer = reinterpret_cast<MockRuntimeBuffer*>(userdata);
        buffer->parsed_data.push_back(std::vector<pcsample_v1_t>(req_size));
        *sample = buffer->parsed_data.back().data();
        return req_size;
    }

    std::vector<packet_union_t>             packets;
    std::vector<std::vector<pcsample_v1_t>> parsed_data;
};

/**
 * Mimics a HSA doorbell. Every live instance of this class has an unique ID (handler).
 * The handler itself may be not unique considering dead instances.
 */
class MockDoorBell
{
public:
    MockDoorBell()
    : handler(getUniqueId())
    {
        available_ids.erase(handler);
    };
    ~MockDoorBell() { available_ids.insert(handler); }

    const size_t            handler;
    static constexpr size_t num_unique_bells = 4;

private:
    static size_t getUniqueId()
    {
        assert(available_ids.size() > 0);
        return *available_ids.begin();
    }
    static std::unordered_set<size_t> reset_available_ids()
    {
        std::unordered_set<size_t> set;
        for(size_t i = 0; i < num_unique_bells; i++)
            set.insert(i);
        return set;
    };
    static std::unordered_set<size_t> available_ids;
};
std::unordered_set<size_t> MockDoorBell::available_ids = MockDoorBell::reset_available_ids();

/**
 * Mimics a HSA queue. Every live instance of this class has an unique ID and a doorbell.
 * The read and write indexes mimics the locations in the queue (modulo queue_size) for the
 * read and write pointers.
 * Creating an instance of this class automatically adds a queue creation packet to the buffer.
 */
class MockQueue
{
public:
    MockQueue(int size_, std::shared_ptr<MockRuntimeBuffer>& buffer_)
    : id(cur_unique_id)
    , size(size_)
    , doorbell()
    , buffer(buffer_){};

    //! Submits a packet to the runtime buffer
    void submit(const packet_union_t& pkt) { buffer->submit(pkt); }
    void print() { std::cout << "Queue - id:" << id << " bell:" << doorbell.handler << std::endl; }

    //! Increments the read_index.
    void inc_read_index(int dispatch_id)
    {
        async_read_index.insert(dispatch_id);
        while(async_read_index.erase(read_index))
            read_index++;
    }

    int    read_index  = 0;
    int    write_index = 0;
    size_t active_dispatches =
        0;  //! Number of dispatches that are still able to generate PC samples
    int                     last_known_read_pkt = 0;
    std::unordered_set<int> async_read_index{};

    const size_t                             id;
    const size_t                             size;
    const MockDoorBell                       doorbell;
    std::shared_ptr<MockRuntimeBuffer> const buffer;

private:
    static size_t cur_unique_id;
};
size_t MockQueue::cur_unique_id = 1;

/**
 * Mimics a kernel dispatch.
 * Creating an instance of this class automatically adds a dispatch creation packet to the buffer.
 */
class MockDispatch
{
public:
    MockDispatch(std::shared_ptr<MockQueue>& queue_)
    : queue(queue_)
    , dispatch_id(queue->write_index)
    , doorbell_id(queue->doorbell.handler)
    , unique_id(cur_unique_id)
    {
        // Ensure queues are not holding more dispatches than queue_size.
        assert(queue->active_dispatches < queue->size);
        queue->active_dispatches++;
        cur_unique_id++;

        packet_union_t uni;
        ::memset(&uni, 0, sizeof(uni));
        uni.dispatch_id.type           = AMD_DISPATCH_PKT_ID;
        uni.dispatch_id.doorbell_id    = doorbell_id;
        uni.dispatch_id.queue_size     = queue->size;
        uni.dispatch_id.write_index    = dispatch_id;
        uni.dispatch_id.read_index     = queue->read_index;
        uni.dispatch_id.correlation_id = unique_id;
        queue->submit(uni);
        queue->write_index++;
    };

    virtual ~MockDispatch()
    {
        queue->active_dispatches--;
        if(queue_read_inc) return;

        queue->inc_read_index((int) dispatch_id);
        queue_read_inc = true;
    }

    //! Returns the "correlation_id" seen by the trap handler.
    uint64_t getMockId()
    {
        return Parser::CorrelationMap::wrap_correlation_id(doorbell_id, dispatch_id, queue->size);
    };

    //! Submits a packet to the buffer
    void submit(const packet_union_t& pkt) { queue->submit(pkt); }
    void submit(const perf_sample_snapshot_v1& snap)
    {
        queue->submit(packet_union_t{.snap = snap});
    }
    void print()
    {
        std::cout << "Dispatch - un_id:" << unique_id << " bell:" << doorbell_id
                  << " ds_id:" << dispatch_id << std::endl;
    }

    std::shared_ptr<MockQueue> const queue;
    const size_t                     dispatch_id;
    const size_t                     doorbell_id;
    const size_t                     unique_id;
    static size_t                    cur_unique_id;

private:
    bool queue_read_inc = false;
};
size_t MockDispatch::cur_unique_id = 0;

/**
 * Lightweight class to represent a wave in the particular dispatch.
 * Capable of generating PC samples and submiting them to the buffer.
 * Instead of generating a valid program counter, this class uses the snapshot.pc field to
 * store the original dispatch's unique_id for later correctness verification.
 */
class MockWave
{
public:
    MockWave(const std::shared_ptr<MockDispatch>& dispatch_)
    : dispatch(dispatch_)
    {}

    void genPCSample()
    {
        packet_union_t uni;
        ::memset(&uni, 0, sizeof(uni));
        uni.snap.pc             = dispatch->unique_id;
        uni.snap.correlation_id = dispatch->getMockId();
        dispatch->submit(uni);
    };
    void print()
    {
        std::cout << "Gen: " << dispatch->doorbell_id << " "
                  << (dispatch->dispatch_id % dispatch->queue->size) << " from "
                  << dispatch->unique_id << std::endl;
    }

    std::shared_ptr<MockDispatch> const dispatch;
};
