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

#include "mocks.hpp"

#define GFXIP_MAJOR 9

/**
 * Benchmarks how fast the parser can process samples on a single threaded case
 * Current: 5600X with -Ofast, up to >140 million samples/s or ~9GB/s R/W (18GB/s bidirectional)
 */
static bool
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
    return true;
}

TEST(pcs_parser, benchmark_test)
{
    EXPECT_EQ(Benchmark(true), true);
    EXPECT_EQ(Benchmark(false), true);
    EXPECT_EQ(Benchmark(false), true);
}
