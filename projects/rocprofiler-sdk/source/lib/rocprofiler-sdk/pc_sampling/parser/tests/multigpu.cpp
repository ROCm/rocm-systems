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

#include <gtest/gtest.h>
#include <cstddef>

#include <future>
#include "lib/rocprofiler-sdk/pc_sampling/code_object.hpp"
#include "mocks.hpp"

#define GFXIP_MAJOR 9
constexpr size_t NUM_THREADS = 8;

class Latch
{
public:
    Latch(size_t num) { counter.store(num); };
    void sync()
    {
        counter.fetch_sub(1);
        while(counter.load())
            ;
    };
    std::atomic<size_t> counter;
};

/**
 * Sample user memory allocation callback.
 * It expects userdata to be cast-able to a pointer to
 * std::vector<std::pair<rocprofiler_pc_sampling_record_t*, uint64_t>>
 */
static uint64_t
alloc_callback(rocprofiler_pc_sampling_record_t** buffer, uint64_t size, void* userdata)
{
    *buffer = new rocprofiler_pc_sampling_record_t[size];
    auto& vector =
        *reinterpret_cast<std::vector<std::pair<rocprofiler_pc_sampling_record_t*, uint64_t>>*>(
            userdata);
    vector.push_back({*buffer, size});
    return size;
}

void
multithread_queue_hammer(size_t tid, Latch* latch)
{
    static auto  corr_map = Parser::CorrelationMap{};
    std::mt19937 rdgen(tid);

    constexpr int NUM_ACTIONS = 100000;
    constexpr int QSIZE       = 16;
    constexpr int NUM_QUEUES  = MockDoorBell::num_unique_bells / NUM_THREADS;
    constexpr int ACTION_MAX  = QSIZE * NUM_QUEUES / 2;

    std::shared_ptr<MockRuntimeBuffer> buffer = std::make_shared<MockRuntimeBuffer>(tid);

    std::array<std::shared_ptr<MockQueue>, NUM_QUEUES>                 queues;
    std::array<std::vector<std::shared_ptr<MockDispatch>>, NUM_QUEUES> active_dispatches;

    int    num_reset_queues         = 0;
    int    num_samples_generated    = 0;
    int    num_dispatches_generated = 0;
    double avg_q_occupancy          = 0;
    size_t max_q_occupancy          = 0;

    for(int i = 0; i < NUM_QUEUES; i++)
        queues[i] = std::make_shared<MockQueue>(QSIZE, buffer);
    for(int i = 0; i < NUM_QUEUES; i++)
        active_dispatches[i].push_back(std::make_shared<MockDispatch>(queues[i]));

    for(int i = 0; i < NUM_ACTIONS; i++)
    {
        int q      = rdgen() % NUM_QUEUES;
        int action = rdgen() % ACTION_MAX;
        if(action == 0)
        {
            // Delete queue and create new one
            active_dispatches[q] = {};
            queues[q].reset();
            queues[q] = std::make_shared<MockQueue>(QSIZE, buffer);
            num_reset_queues++;
        }
        else if(action > ACTION_MAX / 2 && active_dispatches[q].size() > 1)
        {
            // Delete dispatch
            active_dispatches[q].erase(active_dispatches[q].begin(),
                                       active_dispatches[q].begin() + 1);
        }

        // Add new dispatch
        if(active_dispatches[q].size() < QSIZE)
        {
            active_dispatches[q].push_back(std::make_shared<MockDispatch>(queues[q]));
            num_dispatches_generated += 1;
        }

        // Generate one "pc" sample for each queue
        buffer->genUpcomingSamples(NUM_QUEUES);
        for(auto& queue : active_dispatches)
        {
            EXPECT_NE(queue.size(), 0);
            std::shared_ptr<MockDispatch> rand_dispatch = queue[rdgen() % queue.size()];
            MockWave(rand_dispatch).genPCSample();
            num_samples_generated += 1;
            avg_q_occupancy += queue.size();
            max_q_occupancy = std::max(max_q_occupancy, queue.size());
        }
    }

    latch->sync();

    std::vector<std::pair<rocprofiler_pc_sampling_record_t*, uint64_t>> all_allocations;

    CHECK_PARSER(_parse_buffer<GFX9>((generic_sample_t*) buffer->packets.data(),
                                     buffer->packets.size(),
                                     alloc_callback,
                                     (void*) &all_allocations,
                                     &corr_map));

    EXPECT_EQ(all_allocations.size(), NUM_ACTIONS);  // Incorrect number of callbacks
    for(auto sb = 0ul; sb < all_allocations.size(); sb++)
    {
        rocprofiler_pc_sampling_record_t* samples     = all_allocations[sb].first;
        size_t                            num_samples = all_allocations[sb].second;

        EXPECT_EQ(num_samples, NUM_QUEUES);
        for(size_t i = 0; i < num_samples; i++)
            EXPECT_EQ(samples[i].correlation_id.internal, samples[i].pc.loaded_code_object_offset);
        delete[] samples;
    }
}

/**
 * Benchmarks how fast the parser can process samples on a single threaded case
 * Current: 5600X with -Ofast, up to >140 million samples/s or ~9GB/s R/W (18GB/s bidirectional)
 */
static std::pair<size_t, size_t>
MultiThread_BenchMark(size_t tid, Latch* latch)
{
    static auto corr_map = Parser::CorrelationMap{};

    constexpr size_t SAMPLE_PER_DISPATCH = 4096;
    constexpr size_t DISP_PER_QUEUE      = 16;
    constexpr size_t NUM_QUEUES          = 1;

    std::shared_ptr<MockRuntimeBuffer> buffer = std::make_shared<MockRuntimeBuffer>(tid);
    std::array<std::vector<std::shared_ptr<MockDispatch>>, NUM_QUEUES> active_dispatches;

    for(size_t q = 0; q < NUM_QUEUES; q++)
    {
        std::shared_ptr<MockQueue> queue = std::make_shared<MockQueue>(DISP_PER_QUEUE * 2, buffer);
        for(size_t d = 0; d < DISP_PER_QUEUE; d++)
            active_dispatches[q].push_back(std::make_shared<MockDispatch>(queue));
    }

    constexpr size_t TOTAL_NUM_SAMPLES = NUM_QUEUES * DISP_PER_QUEUE * SAMPLE_PER_DISPATCH;
    buffer->genUpcomingSamples(TOTAL_NUM_SAMPLES);

    for(auto& queue : active_dispatches)
        for(auto& dispatch : queue)
            for(size_t i = 0; i < SAMPLE_PER_DISPATCH; i++)
                MockWave(dispatch).genPCSample();

    std::pair<rocprofiler_pc_sampling_record_t*, size_t> userdata;
    userdata.first  = new rocprofiler_pc_sampling_record_t[TOTAL_NUM_SAMPLES];
    userdata.second = TOTAL_NUM_SAMPLES;

    latch->sync();

    auto t0 = std::chrono::system_clock::now();
    CHECK_PARSER(_parse_buffer<GFX9>(
        (generic_sample_t*) buffer->packets.data(),
        buffer->packets.size(),
        [](rocprofiler_pc_sampling_record_t** sample, uint64_t size, void* userdata_) {
            auto* pair =
                reinterpret_cast<std::pair<rocprofiler_pc_sampling_record_t*, size_t>*>(userdata_);
            *sample = pair->first;
            return size;
        },
        &userdata,
        &corr_map));
    auto t1 = std::chrono::system_clock::now();
    delete[] userdata.first;
    return {TOTAL_NUM_SAMPLES, (t1 - t0).count()};
}

void
multithread_codeobj(size_t tid, Latch* latch)
{
    using addr_range_t = rocprofiler::sdk::codeobj::segment::address_range_t;
    auto* table = rocprofiler::pc_sampling::code_object::CodeobjTableTranslatorSynchronized::Get();

    static auto  corr_map = Parser::CorrelationMap{};
    std::mt19937 rdgen(tid);

    constexpr int NUM_DISPATCH = 20000;
    constexpr int NUM_SAMPLES  = 50;
    constexpr int QSIZE        = 16;

    auto buffer = std::make_shared<MockRuntimeBuffer>(tid);
    auto queue  = std::make_shared<MockQueue>(QSIZE, buffer);

    std::pair<rocprofiler_pc_sampling_record_t*, size_t> userdata;
    userdata.first  = new rocprofiler_pc_sampling_record_t[NUM_SAMPLES];
    userdata.second = NUM_SAMPLES;

    latch->sync();

    for(int d = 0; d < NUM_DISPATCH; d++)
    {
        buffer->packets.clear();
        auto dispatch = std::make_shared<MockDispatch>(queue);

        const size_t pc_base_addr = NUM_SAMPLES * dispatch->unique_id;
        table->insert(addr_range_t{pc_base_addr, NUM_SAMPLES, dispatch->unique_id});

        packet_union_t uni{};
        uni.snap.correlation_id = dispatch->getMockId().raw;

        buffer->genUpcomingSamples(NUM_SAMPLES);
        for(int s = 0; s < NUM_SAMPLES; s++)
        {
            uni.snap.pc = pc_base_addr + s;
            dispatch->submit(uni);
        }

        CHECK_PARSER(_parse_buffer<GFX9>(
            (generic_sample_t*) buffer->packets.data(),
            buffer->packets.size(),
            [](rocprofiler_pc_sampling_record_t** sample, uint64_t size, void* userdata_) {
                auto* pair =
                    reinterpret_cast<std::pair<rocprofiler_pc_sampling_record_t*, size_t>*>(
                        userdata_);
                *sample = pair->first;
                assert(size <= NUM_SAMPLES);
                return size;
            },
            &userdata,
            &corr_map));

        for(int s = 0; s < NUM_SAMPLES; s++)
        {
            const auto& pc = userdata.first[s].pc;
            EXPECT_EQ(pc.loaded_code_object_id, dispatch->unique_id);
            EXPECT_EQ(pc.loaded_code_object_offset, s);
        }

        table->remove(addr_range_t{pc_base_addr, NUM_SAMPLES, dispatch->unique_id});
    }

    delete[] userdata.first;
}

TEST(pcs_parser, bench_test)
{
    size_t time    = 0;
    size_t samples = 0;

    for(int it = 0; it < 4; it++)
    {
        Latch latch(NUM_THREADS);

        std::vector<std::future<std::pair<size_t, size_t>>> threads{};
        for(size_t t = 0; t < NUM_THREADS; t++)
            threads.push_back(std::async(std::launch::async, MultiThread_BenchMark, t, &latch));

        if(it == 0) continue;  // Skip warmup

        for(auto& t : threads)
        {
            auto result = t.get();
            samples += result.first;
            time += result.second;
        }
    }

    double mean = 1E3 * NUM_THREADS * samples / time;

    std::cout << "Benchmark: Parsed " << int(mean * 1E3 + 0.5) * 1E-3f << " Msample/s (";
    std::cout << int(sizeof(rocprofiler_pc_sampling_record_t) * mean) << " MB/s)" << std::endl;
};

TEST(pcs_parser, hammer_test)
{
    Latch latch(NUM_THREADS);

    std::vector<std::future<void>> threads{};
    for(size_t i = 0; i < NUM_THREADS; i++)
        threads.push_back(std::async(std::launch::async, multithread_queue_hammer, i, &latch));
};

TEST(pcs_parser, codeobj_test)
{
    Latch latch(NUM_THREADS);

    std::vector<std::future<void>> threads{};
    for(size_t i = 0; i < NUM_THREADS; i++)
        threads.push_back(std::async(std::launch::async, multithread_codeobj, i, &latch));
};
