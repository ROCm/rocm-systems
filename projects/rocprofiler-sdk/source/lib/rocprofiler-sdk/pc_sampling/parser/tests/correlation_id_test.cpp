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
 * Creates a few queues with a few dispatches per queue.
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
 * Hammers the parser by creating and destroying queues at random, adding dispatches at random
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

    delete[] all_allocations[0].first;
    delete[] all_allocations[1].first;
}

TEST(pcs_parser, multi_buffer)
{
    pcs_parser_multi_buffer<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_multi_buffer<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}

/**
 * Regression test for AIPROFSDK-1154.
 *
 * The trap-correlation-id lookup in add_upcoming_samples() legitimately fails for samples
 * that cannot be tied to any tracked dispatch (e.g. blit-kernel / self-modifying-code
 * samples). That is intentional: such samples are still published, just with
 * Dispatch_Id == 0 and Correlation_Id == 0 (see PR #878 "PC Sampling - blit kernels
 * handling"). Deliberately dropping every sample that hits this path would silently
 * regress that feature.
 *
 * On gfx942 (MI300A), a fraction of samples arrive with the same "no known correlation"
 * signature, but their contents (hw_id, workgroup_id, exec_mask, timestamp) are leftover/
 * uninitialized hardware memory rather than a real measurement. Every single one of these
 * genuinely-corrupt samples has bit 63 of the timestamp set (i.e. is negative when read as
 * signed), which a real timestamp can never be. This test verifies that samples are
 * distinguished by that signal: a negative-timestamp sample is dropped (size == 0), while an
 * otherwise-identical, merely-uncorrelated sample with a normal timestamp is preserved.
 */
template <typename PcSamplingRecordT>
void
pcs_parser_negative_timestamp_dropped()
{
    auto buffer = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();

    // Neither sample below corresponds to a registered dispatch, so both take the
    // correlation-lookup exception path inside add_upcoming_samples().
    buffer->genUpcomingSamples(2);

    packet_union_t uncorrelated_but_real;
    ::memset(&uncorrelated_but_real, 0, sizeof(uncorrelated_but_real));
    uncorrelated_but_real.snap.pc                 = 0xbadd;
    uncorrelated_but_real.snap.correlation_id     = 0xdeadbeef;  // never registered
    uncorrelated_but_real.snap.timestamp          = 123456789ULL;  // plausible, non-negative
    uncorrelated_but_real.snap.perf_snapshot_data = 0x1;  // stochastic "valid" bit
    buffer->submit(uncorrelated_but_real);

    packet_union_t garbage_slot;
    ::memset(&garbage_slot, 0, sizeof(garbage_slot));
    garbage_slot.snap.pc                 = 0xbadd;
    garbage_slot.snap.correlation_id     = 0xdeadbeef;  // never registered
    garbage_slot.snap.timestamp          = 0x8000000000000001ULL;  // sign bit set
    garbage_slot.snap.perf_snapshot_data = 0x1;  // stochastic "valid" bit
    buffer->submit(garbage_slot);

    // Both samples take the correlation-lookup exception path, so parse_buffer() is expected
    // to report PCSAMPLE_STATUS_PARSER_ERROR (this is how blit-kernel samples are normally
    // surfaced; it is not treated as fatal by callers). We can't use
    // MockRuntimeBuffer::get_parsed_buffer()/CHECK_PARSER here since those abort on anything
    // other than PCSAMPLE_STATUS_SUCCESS.
    std::vector<std::pair<PcSamplingRecordT*, uint64_t>> all_allocations;
    auto                                                 status =
        parse_buffer((generic_sample_t*) buffer->packets.data(),
                     buffer->packets.size(),
                     GFXIP_MAJOR,
                     GFXIP_MINOR,
                     alloc_callback<PcSamplingRecordT>,
                     (void*) &all_allocations);
    EXPECT_EQ(status, PCSAMPLE_STATUS_PARSER_ERROR);

    ASSERT_EQ(all_allocations.size(), 1);
    ASSERT_EQ(all_allocations[0].second, 2);

    const auto& uncorrelated = all_allocations[0].first[0];
    const auto& garbage      = all_allocations[0].first[1];

    // Legitimate uncorrelated ("blit kernel") sample: kept, with the existing sentinel
    // values, i.e. NOT invalidated by this fix.
    EXPECT_NE(uncorrelated.size, 0u);
    EXPECT_EQ(uncorrelated.dispatch_id, 0u);
    EXPECT_EQ(uncorrelated.correlation_id.internal, ROCPROFILER_CORRELATION_ID_INTERNAL_NONE);

    // Genuinely corrupt trap slot (negative timestamp): must be dropped.
    EXPECT_EQ(garbage.size, 0u);

    delete[] all_allocations[0].first;
}

TEST(pcs_parser, negative_timestamp_dropped)
{
    pcs_parser_negative_timestamp_dropped<rocprofiler_pc_sampling_record_host_trap_v0_t>();
    pcs_parser_negative_timestamp_dropped<rocprofiler_pc_sampling_record_stochastic_v0_t>();
}
