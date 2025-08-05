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

#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

#define GFXIP_MAJOR 9
#define GFXIP_MINOR 4

std::mt19937 rdgen(1);

/**
 * Sample user memory allocation callback.
 * It expects userdata to be cast-able to a pointer to
 * std::vector<std::pair<PcSamplingRecordT*, uint64_t>>
 */
template <typename PcSamplingRecordT>
static uint64_t
alloc_callback(PcSamplingRecordT** buffer, uint64_t size, void* userdata)
{
    *buffer = new PcSamplingRecordT[size];
    auto& vector =
        *reinterpret_cast<std::vector<std::pair<PcSamplingRecordT*, uint64_t>>*>(userdata);
    vector.push_back({*buffer, size});
    return size;
}

/**
 * Uses the MockWave dispatch's unique_id store in the pc field to verify
 * the reconstructed correlation_id.
 */
template <typename PcSamplingRecordT>
static bool
check_samples(PcSamplingRecordT* samples, uint64_t size)
{
    // TODO: replace with (code_obj_id, pc)
    for(size_t i = 0; i < size; i++)
        if(samples[i].correlation_id.internal != samples[i].pc.code_object_offset) return false;
    return true;
}

template <typename PcSamplingRecordT>
void
pcs_parser_hello_world()
{
    auto buffer   = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    auto queue    = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
    auto dispatch = std::make_shared<MockDispatch<PcSamplingRecordT>>(queue);

    buffer->genUpcomingSamples(2);
    MockWave(dispatch).genPCSample();
    MockWave(dispatch).genPCSample();

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(), 1);  // HelloWorld: Incorrect number of callbacks
    for(auto& sample : all_allocations)
    {
        EXPECT_EQ(sample.second, 2);  // HelloWorld: Incorrect number of samples
        EXPECT_EQ(check_samples(sample.first, sample.second),
                  true);  // HelloWorld: parsed ID does not match correct ID
        delete[] sample.first;
    }
}

/**
 * Simplest mock classes use, generates a single queue+dispatch with 2 PC samples.
 */
TEST(pcs_parser, hello_world)
{
    pcs_parser_hello_world<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_hello_world<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

/**
 * A little more complicated.
 * Generates a few dispatches for 2 different queues and samples in forward and reverse order.
 * Checks if the reconstructed correlation_id is correct.
 */
template <typename PcSamplingRecordT>
void
pcs_parser_reverse_wave_order()
{
    auto buffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    auto queue1 = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
    auto queue2 = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);

    std::vector<std::shared_ptr<MockDispatch<PcSamplingRecordT>>> dispatches;
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));

    buffer->genUpcomingSamples(dispatches.size());
    for(auto it = dispatches.rbegin(); it != dispatches.rend(); it++)
        MockWave(*it).genPCSample();
    buffer->genUpcomingSamples(dispatches.size());
    for(auto it = dispatches.begin(); it != dispatches.end(); it++)
        MockWave(*it).genPCSample();

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(), 2);  // ReverseWaveOrder test: Incorrect number of callbacks
    for(auto& sample : all_allocations)
    {
        EXPECT_EQ(sample.second,
                  dispatches.size());  // ReverseWaveOrder: Incorrect number of samples
        EXPECT_EQ(check_samples(sample.first, sample.second),
                  true);  // ReverseWaveOrder: parsed ID does not match correct ID
        delete[] sample.first;
    }
}

TEST(pcs_parser, reverse_wave_order)
{
    pcs_parser_reverse_wave_order<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_reverse_wave_order<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

template <typename PcSamplingRecordT>
void
pcs_parser_dispatch_wrapping()
{
    const int num_samples = 32;
    auto      buffer      = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    auto      queue       = std::make_shared<MockQueue<PcSamplingRecordT>>(5, buffer);

    for(int i = 0; i < num_samples; i++)
    {
        auto dispatch = std::make_shared<MockDispatch<PcSamplingRecordT>>(queue);
        buffer->genUpcomingSamples(1);
        MockWave(dispatch).genPCSample();
    }

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(),
              num_samples);  // RandomSamples test: Incorrect number of callbacks
    for(auto& sample : all_allocations)
    {
        EXPECT_EQ(sample.second, 1);  // RandomSamples: Incorrect number of samples
        EXPECT_EQ(check_samples(sample.first, sample.second),
                  true);  // RandomSamples: parsed ID does not match correct ID
        delete[] sample.first;
    }
}

/**
 * Creates a small queue and causes the dispatch_ids to wrap around a few times, and generates
 * a single sample per dispatch. Checks the parser is properly handling the wrapping of queues.
 */
TEST(pcs_parser, dispatch_wrapping)
{
    pcs_parser_dispatch_wrapping<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_dispatch_wrapping<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

template <typename PcSamplingRecordT>
void
pcs_parser_random_samples()
{
    const int num_samples = 1024;
    auto      buffer      = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    auto      queue1      = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
    auto      queue2      = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
    auto      queue3      = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
    auto      queue4      = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);

    std::vector<std::shared_ptr<MockDispatch<PcSamplingRecordT>>> dispatches;
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch<PcSamplingRecordT>>(queue1));

    buffer->genUpcomingSamples(num_samples);
    for(int i = 0; i < num_samples; i++)
        MockWave(dispatches[rdgen() % dispatches.size()]).genPCSample();

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(), 1);  // RandomSamples test: Incorrect number of callbacks
    for(auto& sample : all_allocations)
    {
        EXPECT_EQ(sample.second, num_samples);  // RandomSamples: Incorrect number of samples
        EXPECT_EQ(check_samples(sample.first, sample.second),
                  true);  // RandomSamples: parsed ID does not match correct ID
        delete[] sample.first;
    }
}

/**
 * Creates a few queues with a few dispatchs per queue.
 * Adds random samples per dispatch, and checks the result.
 */
TEST(pcs_parser, random_samples)
{
    pcs_parser_random_samples<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_random_samples<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

template <typename PcSamplingRecordT>
void
pcs_parser_queue_hammer()
{
    constexpr int NUM_ACTIONS = 10000;
    constexpr int QSIZE       = 16;
    constexpr int NUM_QUEUES  = MockDoorBell::num_unique_bells;
    constexpr int ACTION_MAX  = QSIZE * NUM_QUEUES / 2;

    auto buffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();

    std::array<std::shared_ptr<MockQueue<PcSamplingRecordT>>, NUM_QUEUES> queues;
    std::array<std::vector<std::shared_ptr<MockDispatch<PcSamplingRecordT>>>, NUM_QUEUES>
        active_dispatches;

    int    num_reset_queues         = 0;
    int    num_samples_generated    = 0;
    int    num_dispatches_generated = 0;
    double avg_q_occupancy          = 0;
    size_t max_q_occupancy          = 0;

    for(int i = 0; i < NUM_QUEUES; i++)
        queues[i] = std::make_shared<MockQueue<PcSamplingRecordT>>(QSIZE, buffer);
    for(int i = 0; i < NUM_QUEUES; i++)
        active_dispatches[i].push_back(
            std::make_shared<MockDispatch<PcSamplingRecordT>>(queues[i]));

    for(int i = 0; i < NUM_ACTIONS; i++)
    {
        int q      = rdgen() % NUM_QUEUES;
        int action = rdgen() % ACTION_MAX;
        if(action == 0)
        {
            // Delete queue and create new one
            active_dispatches[q] = {};
            queues[q].reset();
            queues[q] = std::make_shared<MockQueue<PcSamplingRecordT>>(QSIZE, buffer);
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
            active_dispatches[q].push_back(
                std::make_shared<MockDispatch<PcSamplingRecordT>>(queues[q]));
            num_dispatches_generated += 1;
        }

        // Generate one "pc" sample for each queue
        buffer->genUpcomingSamples(NUM_QUEUES);
        for(auto& queue : active_dispatches)
        {
            EXPECT_NE(queue.size(), 0);
            std::shared_ptr<MockDispatch<PcSamplingRecordT>> rand_dispatch =
                queue[rdgen() % queue.size()];
            MockWave(rand_dispatch).genPCSample();
            num_samples_generated += 1;
            avg_q_occupancy += queue.size();
            max_q_occupancy = std::max(max_q_occupancy, queue.size());
        }
    }

    std::cout << "Hammer Stats: " << std::endl;
    std::cout << "num_reset_queues: " << num_reset_queues << std::endl;
    std::cout << "num_samples_generated: " << num_samples_generated << std::endl;
    std::cout << "num_dispatches_generated: " << num_dispatches_generated << std::endl;
    std::cout << "Avg queue occupancy: " << avg_q_occupancy / (NUM_ACTIONS * NUM_QUEUES)
              << std::endl;
    std::cout << "Max queue occupancy: " << max_q_occupancy << "\n\n" << std::endl;

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(),
              NUM_ACTIONS);  // QueueHammer test: Incorrect number of callbacks
    for(auto sb = 0ul; sb < all_allocations.size(); sb++)
    {
        PcSamplingRecordT* samples     = all_allocations[sb].first;
        size_t             num_samples = all_allocations[sb].second;

        EXPECT_EQ(num_samples, NUM_QUEUES);  // QueueHammer: Incorrect number of samples
        EXPECT_EQ(check_samples(samples, num_samples),
                  true);  // QueueHammer: parsed ID does not match correct ID
        delete[] samples;
    }
}

/**
 * Hammers the parser by creating and destrying queues at random, adding dispatches at random
 * and generating PC samples at random. By default we use all 4 unique doorbells,
 * queue size is 16 and we generate 10k samples dispatch.
 */
TEST(pcs_parser, queue_hammer)
{
    pcs_parser_queue_hammer<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_queue_hammer<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

template <typename PcSamplingRecordT>
void
pcs_parser_multi_buffer()
{
    auto firstBuffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    auto queue       = std::make_shared<MockQueue<PcSamplingRecordT>>(16, firstBuffer);
    auto dispatch1   = std::make_shared<MockDispatch<PcSamplingRecordT>>(queue);
    auto dispatch2   = std::make_shared<MockDispatch<PcSamplingRecordT>>(queue);

    firstBuffer->genUpcomingSamples(4);
    MockWave(dispatch1).genPCSample();
    MockWave(dispatch2).genPCSample();
    MockWave(dispatch1).genPCSample();
    MockWave(dispatch2).genPCSample();

    auto        secondBuffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
    const auto& packets      = firstBuffer->packets;
    secondBuffer->packets    = std::vector<packet_union_t>(packets.begin() + 2, packets.end());

    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) firstBuffer->packets.data(),
                              firstBuffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));
    CHECK_PARSER(parse_buffer((generic_sample_t*) secondBuffer->packets.data(),
                              secondBuffer->packets.size(),
                              GFXIP_MAJOR,
                              GFXIP_MINOR,
                              alloc_callback<PcSamplingRecordT>,
                              (void*) &all_allocations));

    EXPECT_EQ(all_allocations.size(), 2);  // MultiBuffer: Incorrect number of callbacks
    auto& sample = all_allocations[1];
    EXPECT_EQ(sample.second, 4);  // MultiBuffer: Incorrect number of samples
    EXPECT_EQ(check_samples(sample.first, sample.second),
              true);  // MultiBuffer: parsed ID does not match correct ID

    for(auto& allocation : all_allocations)
    {
        delete[] allocation.first;
    }
}

TEST(pcs_parser, multi_buffer)
{
    pcs_parser_multi_buffer<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_multi_buffer<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}
