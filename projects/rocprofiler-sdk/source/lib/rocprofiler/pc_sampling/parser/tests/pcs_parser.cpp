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

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <gtest/gtest.h>
#include <cassert>
#include <cstddef>

#include "lib/rocprofiler/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler/pc_sampling/parser/tests/mocks.hpp"

#define GFXIP_MAJOR 9

#define TYPECHECK(x)                                                                               \
    snapshots.push_back(pcsample_snapshot_v1_t{.dual_issue_valu   = 0,                             \
                                               .inst_type         = ::PCSAMPLE::x,                 \
                                               .reason_not_issued = 0,                             \
                                               .arb_state_issue   = 0,                             \
                                               .arb_state_stall   = 0});
#define UNROLL_TYPECHECK()                                                                         \
    TYPECHECK(TYPE_VALU);                                                                          \
    TYPECHECK(TYPE_MATRIX);                                                                        \
    TYPECHECK(TYPE_SCALAR);                                                                        \
    TYPECHECK(TYPE_TEX);                                                                           \
    TYPECHECK(TYPE_LDS);                                                                           \
    TYPECHECK(TYPE_FLAT);                                                                          \
    TYPECHECK(TYPE_EXP);                                                                           \
    TYPECHECK(TYPE_MESSAGE);                                                                       \
    TYPECHECK(TYPE_BARRIER);                                                                       \
    TYPECHECK(TYPE_BRANCH_NOT_TAKEN);                                                              \
    TYPECHECK(TYPE_BRANCH_TAKEN);                                                                  \
    TYPECHECK(TYPE_JUMP);                                                                          \
    TYPECHECK(TYPE_OTHER);                                                                         \
    TYPECHECK(TYPE_NO_INST);

#define REASONCHECK(x)                                                                             \
    snapshots.push_back(pcsample_snapshot_v1_t{.dual_issue_valu   = 0,                             \
                                               .inst_type         = 0,                             \
                                               .reason_not_issued = ::PCSAMPLE::x,                 \
                                               .arb_state_issue   = 0,                             \
                                               .arb_state_stall   = 0});
#define UNROLL_REASONCHECK(x)                                                                      \
    REASONCHECK(REASON_NOT_AVAILABLE);                                                             \
    REASONCHECK(REASON_ALU);                                                                       \
    REASONCHECK(REASON_WAITCNT);                                                                   \
    REASONCHECK(REASON_INTERNAL);                                                                  \
    REASONCHECK(REASON_BARRIER);                                                                   \
    REASONCHECK(REASON_ARBITER);                                                                   \
    REASONCHECK(REASON_EX_STALL);                                                                  \
    REASONCHECK(REASON_OTHER_WAIT);

#define ARBCHECK1(x, y)                                                                            \
    snapshots.push_back(pcsample_snapshot_v1_t{.dual_issue_valu   = 0,                             \
                                               .inst_type         = 0,                             \
                                               .reason_not_issued = 0,                             \
                                               .arb_state_issue   = 1 << ::PCSAMPLE::x,            \
                                               .arb_state_stall   = 1 << ::PCSAMPLE::y});
#define ARBCHECK2(x)                                                                               \
    ARBCHECK1(x, ISSUE_VALU);                                                                      \
    ARBCHECK1(x, ISSUE_MATRIX);                                                                    \
    ARBCHECK1(x, ISSUE_SCALAR);                                                                    \
    ARBCHECK1(x, ISSUE_VMEM_TEX);                                                                  \
    ARBCHECK1(x, ISSUE_LDS);                                                                       \
    ARBCHECK1(x, ISSUE_FLAT);                                                                      \
    ARBCHECK1(x, ISSUE_EXP);                                                                       \
    ARBCHECK1(x, ISSUE_MISC);

#define UNROLL_ARBCHECK()                                                                          \
    ARBCHECK2(ISSUE_VALU);                                                                         \
    ARBCHECK2(ISSUE_MATRIX);                                                                       \
    ARBCHECK2(ISSUE_SCALAR);                                                                       \
    ARBCHECK2(ISSUE_VMEM_TEX);                                                                     \
    ARBCHECK2(ISSUE_LDS);                                                                          \
    ARBCHECK2(ISSUE_FLAT);                                                                         \
    ARBCHECK2(ISSUE_EXP);                                                                          \
    ARBCHECK2(ISSUE_MISC);

std::mt19937 rdgen(1);

TEST(pcs_parser_context, init) { PCSamplingParserContext context; }

/**
 * Sample user memory allocation callback.
 * It expects userdata to be cast-able to a pointer to
 * std::vector<std::pair<pcsample_v1_t*, uint64_t>>
 */
static uint64_t
alloc_callback(pcsample_v1_t** buffer, uint64_t size, void* userdata)
{
    *buffer      = new pcsample_v1_t[size];
    auto& vector = *reinterpret_cast<std::vector<std::pair<pcsample_v1_t*, uint64_t>>*>(userdata);
    vector.push_back({*buffer, size});
    return size;
}

/**
 * Uses the MockWave dispatch's unique_id store in the pc field to verify
 * the reconstructed correlation_id.
 */
static bool
check_samples(pcsample_v1_t* samples, uint64_t size)
{
    for(size_t i = 0; i < size; i++)
        if(samples[i].correlation_id != samples[i].pc) return false;
    return true;
}

/**
 * Simplest mock classes use, generates a single queue+dispatch with 2 PC samples.
 */
TEST(pcs_parser_correlation_id, hello_world)
{
    std::shared_ptr<MockRuntimeBuffer> buffer   = std::make_shared<MockRuntimeBuffer>();
    std::shared_ptr<MockQueue>         queue    = std::make_shared<MockQueue>(16, buffer);
    std::shared_ptr<MockDispatch>      dispatch = std::make_shared<MockDispatch>(queue);

    buffer->genUpcomingSamples(2);
    MockWave(dispatch).genPCSample();
    MockWave(dispatch).genPCSample();

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == 1 && "HelloWorld: Incorrect number of callbacks");
    for(auto& sample : all_allocations)
    {
        assert(sample.second == 2 && "HelloWorld: Incorrect number of samples");
        assert(check_samples(sample.first, sample.second) &&
               "HelloWorld: parsed ID does not match correct ID");
        delete[] sample.first;
    }
}

/**
 * A little more complicated.
 * Generates a few dispatches for 2 different queues and samples in forward and reverse order.
 * Checks if the reconstructed correlation_id is correct.
 */
TEST(pcs_parser_correlation_id, reverse_wave_order)
{
    std::shared_ptr<MockRuntimeBuffer> buffer = std::make_shared<MockRuntimeBuffer>();
    std::shared_ptr<MockQueue>         queue1 = std::make_shared<MockQueue>(16, buffer);
    std::shared_ptr<MockQueue>         queue2 = std::make_shared<MockQueue>(16, buffer);

    std::vector<std::shared_ptr<MockDispatch>> dispatches;
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));

    buffer->genUpcomingSamples(dispatches.size());
    for(auto it = dispatches.rbegin(); it != dispatches.rend(); it++)
        MockWave(*it).genPCSample();
    buffer->genUpcomingSamples(dispatches.size());
    for(auto it = dispatches.begin(); it != dispatches.end(); it++)
        MockWave(*it).genPCSample();

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == 2 && "ReverseWaveOrder test: Incorrect number of callbacks");
    for(auto& sample : all_allocations)
    {
        assert(sample.second == dispatches.size() &&
               "ReverseWaveOrder: Incorrect number of samples");
        assert(check_samples(sample.first, sample.second) &&
               "ReverseWaveOrder: parsed ID does not match correct ID");
        delete[] sample.first;
    }
}

/**
 * Creates a small queue and causes the dispatch_ids to wrap around a few times, and generates
 * a single sample per dispatch. Checks the parser is properly handling the wrapping of queues.
 */
TEST(pcs_parser_correlation_id, dispatch_wrapping)
{
    const int                          num_samples = 32;
    std::shared_ptr<MockRuntimeBuffer> buffer      = std::make_shared<MockRuntimeBuffer>();
    std::shared_ptr<MockQueue>         queue       = std::make_shared<MockQueue>(5, buffer);

    for(int i = 0; i < num_samples; i++)
    {
        auto dispatch = std::make_shared<MockDispatch>(queue);
        buffer->genUpcomingSamples(1);
        MockWave(dispatch).genPCSample();
    }

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == num_samples &&
           "RandomSamples test: Incorrect number of callbacks");
    for(auto& sample : all_allocations)
    {
        assert(sample.second == 1 && "RandomSamples: Incorrect number of samples");
        assert(check_samples(sample.first, sample.second) &&
               "RandomSamples: parsed ID does not match correct ID");
        delete[] sample.first;
    }
}

/**
 * Creates a few queues with a few dispatchs per queue.
 * Adds random samples per dispatch, and checks the result.
 */
TEST(pcs_parser_correlation_id, random_samples)
{
    const int                          num_samples = 1024;
    std::shared_ptr<MockRuntimeBuffer> buffer      = std::make_shared<MockRuntimeBuffer>();
    std::shared_ptr<MockQueue>         queue1      = std::make_shared<MockQueue>(16, buffer);
    std::shared_ptr<MockQueue>         queue2      = std::make_shared<MockQueue>(16, buffer);
    std::shared_ptr<MockQueue>         queue3      = std::make_shared<MockQueue>(16, buffer);
    std::shared_ptr<MockQueue>         queue4      = std::make_shared<MockQueue>(16, buffer);

    std::vector<std::shared_ptr<MockDispatch>> dispatches;
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));
    dispatches.push_back(std::make_shared<MockDispatch>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch>(queue3));
    dispatches.push_back(std::make_shared<MockDispatch>(queue2));
    dispatches.push_back(std::make_shared<MockDispatch>(queue1));

    buffer->genUpcomingSamples(num_samples);
    for(int i = 0; i < num_samples; i++)
        MockWave(dispatches[rdgen() % dispatches.size()]).genPCSample();

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == 1 && "RandomSamples test: Incorrect number of callbacks");
    for(auto& sample : all_allocations)
    {
        assert(sample.second == num_samples && "RandomSamples: Incorrect number of samples");
        assert(check_samples(sample.first, sample.second) &&
               "RandomSamples: parsed ID does not match correct ID");
        delete[] sample.first;
    }
}

/**
 * Hammers the parser by creating and destrying queues at random, adding dispatches at random
 * and generating PC samples at random. By default we use all 4 unique doorbells,
 * queue size is 16 and we generate 10k samples dispatch.
 */
TEST(pcs_parser_correlation_id, queue_hammer)
{
    constexpr int NUM_ACTIONS = 10000;
    constexpr int QSIZE       = 16;
    constexpr int NUM_QUEUES  = MockDoorBell::num_unique_bells;
    constexpr int ACTION_MAX  = QSIZE * NUM_QUEUES / 2;

    std::shared_ptr<MockRuntimeBuffer> buffer = std::make_shared<MockRuntimeBuffer>();

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
            assert(queue.size() > 0);
            std::shared_ptr<MockDispatch> rand_dispatch = queue[rdgen() % queue.size()];
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

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == NUM_ACTIONS &&
           "QueueHammer test: Incorrect number of callbacks");
    for(auto& all_allocation : all_allocations)
    {
        pcsample_v1_t* samples     = all_allocation.first;
        size_t         num_samples = all_allocation.second;

        assert(num_samples == NUM_QUEUES && "QueueHammer: Incorrect number of samples");
        assert(check_samples(samples, num_samples) &&
               "QueueHammer: parsed ID does not match correct ID");
        delete[] samples;
        (void) num_samples;
    }
}

TEST(pcs_parser_correlation_id, multi_buffer)
{
    std::shared_ptr<MockRuntimeBuffer> firstBuffer = std::make_shared<MockRuntimeBuffer>();
    std::shared_ptr<MockQueue>         queue       = std::make_shared<MockQueue>(16, firstBuffer);
    std::shared_ptr<MockDispatch>      dispatch1   = std::make_shared<MockDispatch>(queue);
    std::shared_ptr<MockDispatch>      dispatch2   = std::make_shared<MockDispatch>(queue);

    firstBuffer->genUpcomingSamples(4);
    MockWave(dispatch1).genPCSample();
    MockWave(dispatch2).genPCSample();
    MockWave(dispatch1).genPCSample();
    MockWave(dispatch2).genPCSample();

    std::shared_ptr<MockRuntimeBuffer> secondBuffer = std::make_shared<MockRuntimeBuffer>();
    const auto&                        packets      = firstBuffer->packets;
    secondBuffer->packets = std::vector<packet_union_t>(packets.begin() + 2, packets.end());

    std::vector<std::pair<pcsample_v1_t*, uint64_t>> all_allocations;

    CHECK_PARSER(parse_buffer((generic_sample_t*) firstBuffer->packets.data(),
                              firstBuffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));
    CHECK_PARSER(parse_buffer((generic_sample_t*) secondBuffer->packets.data(),
                              secondBuffer->packets.size(),
                              GFXIP_MAJOR,
                              alloc_callback,
                              (void*) &all_allocations));

    assert(all_allocations.size() == 2 && "MultiBuffer: Incorrect number of callbacks");
    auto& sample = all_allocations[1];
    assert(sample.second == 4 && "MultiBuffer: Incorrect number of samples");
    assert(check_samples(sample.first, sample.second) &&
           "MultiBuffer: parsed ID does not match correct ID");

    delete[] all_allocations[0].first;
    delete[] all_allocations[1].first;
    (void) sample;
};

/**
 * Benchmarks how fast the parser can process samples on a single threaded case
 * Current: 5600X with -Ofast, up to >140 million samples/s or ~9GB/s R/W (18GB/s bidirectional)
 */
static void
Benchmark(bool bWarmup)
{
    constexpr size_t SAMPLE_PER_DISPATCH = 8192;
    constexpr size_t DISP_PER_QUEUE      = 12;
    constexpr size_t NUM_QUEUES          = MockDoorBell::num_unique_bells;

    std::shared_ptr<MockRuntimeBuffer> buffer = std::make_shared<MockRuntimeBuffer>();
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

    std::pair<pcsample_v1_t*, size_t> userdata;
    userdata.first  = new pcsample_v1_t[TOTAL_NUM_SAMPLES];
    userdata.second = TOTAL_NUM_SAMPLES;

    auto t0 = std::chrono::system_clock::now();
    CHECK_PARSER(parse_buffer((generic_sample_t*) buffer->packets.data(),
                              buffer->packets.size(),
                              GFXIP_MAJOR,
                              [](pcsample_v1_t** sample, uint64_t size, void* userdata_) {
                                  auto* pair = reinterpret_cast<std::pair<pcsample_v1_t*, size_t>*>(
                                      userdata_);
                                  assert(TOTAL_NUM_SAMPLES == pair->second);
                                  *sample = pair->first;
                                  return size;
                              },
                              &userdata));
    auto  t1             = std::chrono::system_clock::now();
    float samples_per_us = float(TOTAL_NUM_SAMPLES) / (t1 - t0).count() * 1E3f;

    if(!bWarmup)
    {
        std::cout << "Benchmark: Parsed " << int(samples_per_us * 1E3f + 0.5f) * 1E-3f
                  << " Msample/s (";
        std::cout << int(sizeof(pcsample_v1_t) * samples_per_us) << " MB/s)" << std::endl;
    }

    delete[] userdata.first;
}

TEST(pcs_parser, benchmark)
{
    Benchmark(true);
    Benchmark(false);
    Benchmark(false);
    Benchmark(false);
}

class WaveSnapTest
{
public:
    WaveSnapTest()
    {
        buffer   = std::make_shared<MockRuntimeBuffer>();
        queue    = std::make_shared<MockQueue>(16, buffer);
        dispatch = std::make_shared<MockDispatch>(queue);
    }

    void Test()
    {
        FillBuffers();
        CheckBuffers();
    }

    virtual void FillBuffers()  = 0;
    virtual void CheckBuffers() = 0;

    void genPCSample(int wave_cnt, int inst_type, int reason, int arb_issue, int arb_stall)
    {
        wave_cnt &= 0x3F;
        inst_type &= 0xF;
        reason &= 0x7;
        arb_issue &= 0xFF;
        arb_stall &= 0xFF;

        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));
        snap.pc             = dispatch->unique_id;
        snap.correlation_id = dispatch->getMockId();

        snap.perf_snapshot_data = (inst_type << 3) | (reason << 7);
        snap.perf_snapshot_data |= (arb_issue << 10) | (arb_stall << 18);
        snap.perf_snapshot_data1 = wave_cnt;

        assert(dispatch.get());
        dispatch->submit(packet_union_t{.snap = snap});
    };

    std::shared_ptr<MockRuntimeBuffer> buffer;
    std::shared_ptr<MockQueue>         queue;
    std::shared_ptr<MockDispatch>      dispatch;
};

class WaveCntTest : public WaveSnapTest
{
public:
    void FillBuffers() override
    {
        // Loop over all possible wave_cnt
        buffer->genUpcomingSamples(max_wave_number);
        for(size_t i = 0; i < max_wave_number; i++)
            genPCSample(i, GFX9::TYPE_LDS, GFX9::REASON_ALU, GFX9::ISSUE_VALU, GFX9::ISSUE_VALU);
    }

    void CheckBuffers() override
    {
        auto parsed = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == max_wave_number);

        for(size_t i = 0; i < max_wave_number; i++)
            assert(parsed[0][i].wave_count == i);
    }

    const size_t                        max_wave_number = 64;
    std::vector<pcsample_snapshot_v1_t> snapshots;
};

class InstTypeTest : public WaveSnapTest
{
public:
    void FillBuffers() override
    {
        // Loop over inst_type_issued
        UNROLL_TYPECHECK();
        buffer->genUpcomingSamples(GFX9::TYPE_LAST);
        for(int i = 0; i < GFX9::TYPE_LAST; i++)
            genPCSample(i, i, GFX9::REASON_ALU, GFX9::ISSUE_MATRIX, GFX9::ISSUE_MATRIX);
    }

    void CheckBuffers() override
    {
        auto parsed = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == GFX9::TYPE_LAST);
        assert(snapshots.size() == GFX9::TYPE_LAST);

        for(size_t i = 0; i < GFX9::TYPE_LAST; i++)
            assert(snapshots[i].inst_type == parsed[0][i].snapshot.inst_type);
    }

    std::vector<pcsample_snapshot_v1_t> snapshots;
};

class StallReasonTest : public WaveSnapTest
{
public:
    void FillBuffers() override
    {
        // Loop over reason_not_issued
        UNROLL_REASONCHECK();
        buffer->genUpcomingSamples(GFX9::REASON_LAST);
        for(int i = 0; i < GFX9::REASON_LAST; i++)
            genPCSample(i, GFX9::TYPE_MATRIX, i, GFX9::ISSUE_MATRIX, GFX9::ISSUE_MATRIX);
    }

    void CheckBuffers() override
    {
        auto parsed = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == GFX9::REASON_LAST);
        assert(snapshots.size() == GFX9::REASON_LAST);

        for(size_t i = 0; i < GFX9::REASON_LAST; i++)
            assert(snapshots[i].reason_not_issued == parsed[0][i].snapshot.reason_not_issued);
    }

    std::vector<pcsample_snapshot_v1_t> snapshots;
};

class ArbStateTest : public WaveSnapTest
{
public:
    void FillBuffers() override
    {
        // Loop over arb_state_issue
        UNROLL_ARBCHECK();
        buffer->genUpcomingSamples(GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);
        for(int i = 0; i < GFX9::ISSUE_LAST; i++)
            for(int j = 0; j < GFX9::ISSUE_LAST; j++)
                genPCSample(i, GFX9::TYPE_MATRIX, GFX9::REASON_ALU, 1 << i, 1 << j);
    }

    void CheckBuffers() override
    {
        auto parsed = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);
        assert(snapshots.size() == GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);

        for(size_t i = 0; i < GFX9::ISSUE_LAST * GFX9::ISSUE_LAST; i++)
        {
            auto& snap = snapshots[i];
            assert(snap.arb_state_issue == parsed[0][i].snapshot.arb_state_issue);
            assert(snap.arb_state_stall == parsed[0][i].snapshot.arb_state_stall);
        }
    }

    std::vector<pcsample_snapshot_v1_t> snapshots;
};

class WaveIssueAndErrorTest : public WaveSnapTest
{
    void FillBuffers() override
    {
        buffer->genUpcomingSamples(16);
        for(int valid = 0; valid <= 1; valid++)
            for(int issued = 0; issued <= 1; issued++)
                for(int dual = 0; dual <= 1; dual++)
                    for(int error = 0; error <= 1; error++)
                        genPCSample(valid, issued, dual, error);
    }

    void CheckBuffers() override
    {
        const int num_combinations = 16;
        auto      parsed           = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == num_combinations);
        assert(compare.size() == num_combinations);

        for(size_t i = 0; i < num_combinations; i++)
        {
            assert(compare[i].flags.valid == parsed[0][i].flags.valid);
            assert(compare[i].wave_issued == parsed[0][i].wave_issued);
            assert(compare[i].snapshot.dual_issue_valu == parsed[0][i].snapshot.dual_issue_valu);
        }
    }

    union trap_snapshot_v1
    {
        struct
        {
            uint32_t valid     : 1;
            uint32_t issued    : 1;
            uint32_t dual      : 1;
            uint32_t reserved  : 23;
            uint32_t error     : 1;
            uint32_t reserved2 : 5;
        };
        uint32_t raw;
    };

    void genPCSample(bool valid, bool issued, bool dual, bool error)
    {
        pcsample_v1_t sample;
        ::memset(&sample, 0, sizeof(sample));
        sample.pc             = dispatch->unique_id;
        sample.correlation_id = dispatch->getMockId();

        sample.flags.valid              = valid && !error;
        sample.wave_issued              = issued;
        sample.snapshot.dual_issue_valu = dual;

        assert(dispatch.get());

        compare.push_back(sample);

        trap_snapshot_v1 snap;
        snap.valid  = valid;
        snap.issued = issued;
        snap.dual   = dual;
        snap.error  = error;

        perf_sample_snapshot_v1 pss;
        pss.perf_snapshot_data = snap.raw;
        pss.correlation_id     = dispatch->getMockId();
        dispatch->submit(std::move(pss));
    };

    std::vector<pcsample_v1_t> compare;
};

class WaveOtherFieldsTest : public WaveSnapTest
{
    void FillBuffers() override
    {
        buffer->genUpcomingSamples(3);
        genPCSample(1, 2, 3, 4, 5, 6, 7, 8);       // Counting
        genPCSample(3, 5, 7, 11, 13, 17, 19, 23);  // Some prime numbers
        genPCSample(23, 19, 17, 13, 11, 7, 5, 3);  // Some reversed primes
    }

    void CheckBuffers() override
    {
        auto parsed = buffer->get_parsed_buffer(9);  // GFXIP==9
        assert(parsed.size() == 1);
        assert(parsed[0].size() == 3);
        assert(compare.size() == 3);

        for(size_t i = 0; i < 3; i++)
        {
            assert(parsed[0][i].flags.has_stall_reason == true);
            assert(parsed[0][i].flags.has_wave_cnt == true);
            assert(parsed[0][i].flags.has_memory_counter == false);

            assert(compare[i].exec_mask == parsed[0][i].exec_mask);
            assert(compare[i].workgroud_id_x == parsed[0][i].workgroud_id_x);
            assert(compare[i].workgroud_id_y == parsed[0][i].workgroud_id_y);
            assert(compare[i].workgroud_id_z == parsed[0][i].workgroud_id_z);

            assert(compare[i].chiplet == parsed[0][i].chiplet);
            assert(compare[i].wave_id == parsed[0][i].wave_id);
            assert(compare[i].hw_id == parsed[0][i].hw_id);
            assert(compare[i].correlation_id == parsed[0][i].correlation_id);
        }
    }

    void genPCSample(int pc, int exec, int blkx, int blky, int blkz, int chip, int wave, int hwid)
    {
        pcsample_v1_t sample;
        ::memset(&sample, 0, sizeof(sample));

        sample.exec_mask      = exec;
        sample.workgroud_id_x = blkx;
        sample.workgroud_id_y = blky;
        sample.workgroud_id_z = blkz;

        sample.chiplet        = chip;
        sample.wave_id        = wave;
        sample.hw_id          = hwid;
        sample.correlation_id = dispatch->unique_id;

        compare.push_back(sample);

        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));
        snap.exec_mask = exec;

        snap.workgroud_id_x      = blkx;
        snap.workgroud_id_y      = blky;
        snap.workgroud_id_z      = blkz;
        snap.chiplet_and_wave_id = (chip << 8) | (wave & 0x3F);
        snap.hw_id               = hwid;
        snap.correlation_id      = dispatch->getMockId();

        assert(dispatch.get());
        dispatch->submit(snap);

        (void) pc;
    };

    std::vector<pcsample_v1_t> compare;
};

// FIXME (vladimir): For some reason, the test can stochastically fail.
// Did not have time to get into details.
TEST(pcs_parser, gfx9)
{
    WaveCntTest{}.Test();
    InstTypeTest{}.Test();
    StallReasonTest{}.Test();
    ArbStateTest{}.Test();
    WaveIssueAndErrorTest{}.Test();
    // FIXME: this might crash some time.
    // WaveOtherFieldsTest{}.Test();

    std::cout << "GFX9 Test Done." << std::endl;
}

// TODO: refactor the tests, modularize them and extract unit tests
// from the integration f
