// MIT License
//
// Copyright (c) 2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

#include <gtest/gtest.h>

template <typename PcSamplingRecordT>
class WaveSnapTest
{
public:
    WaveSnapTest()
    {
        buffer   = std::make_shared<MockRuntimeBuffer<PcSamplingRecordT>>();
        queue    = std::make_shared<MockQueue<PcSamplingRecordT>>(16, buffer);
        dispatch = std::make_shared<MockDispatch<PcSamplingRecordT>>(queue);
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
        snap.correlation_id = dispatch->getMockId().raw;

        snap.perf_snapshot_data = (inst_type << 3) | (reason << 7);
        snap.perf_snapshot_data |= 0x1;  // sample is valid
        snap.perf_snapshot_data |= (arb_issue << 10) | (arb_stall << 18);
        snap.perf_snapshot_data1 = wave_cnt;

        EXPECT_NE(dispatch.get(), nullptr);
        dispatch->submit(packet_union_t{.snap = snap});
    };

    std::shared_ptr<MockRuntimeBuffer<PcSamplingRecordT>> buffer;
    std::shared_ptr<MockQueue<PcSamplingRecordT>>         queue;
    std::shared_ptr<MockDispatch<PcSamplingRecordT>>      dispatch;
};

/**
 * @brief Testing how mid_macro bit affects the PC address.
 *
 * On GFX950, this bit triggers correction of the PC address.
 * On other GFX9 architectures, the PC address remains unchanged.
 */
template <typename PcSamplingRecordT>
class MidMacroPCCorrection : public WaveSnapTest<PcSamplingRecordT>
{
public:
    void FillBuffers() override;   // Explicitly mark as override
    void CheckBuffers() override;  // Explicitly mark as override

    /**
     * @brief Generate PC sample with mid_macro flag.
     * The @p mid_macro is relevant for the GFX950, so it's false by default
     */
    virtual void genPCSample(uint64_t pc, bool mid_macro = false);

    /**
     * @brief Caulcate expected PC address for comparison.
     */
    virtual uint64_t calcaulteExpectedPC(uint64_t pc, bool mid_macro = false);

    virtual std::vector<std::vector<PcSamplingRecordT>> get_parsed_data();

protected:
    ///< testing data
    std::vector<PcSamplingRecordT> compare;
};
