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

template <typename GFX, typename PcSamplingRecordT>
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

        if constexpr(std::is_same_v<GFX, GFX9>)
        {
            snap.perf_snapshot_data = (inst_type << 3) | (reason << 7);
            snap.perf_snapshot_data |= 0x1;  // sample is valid
            snap.perf_snapshot_data |= (arb_issue << 10) | (arb_stall << 18);
            snap.perf_snapshot_data1 = wave_cnt;
        }
        else if constexpr(std::is_same_v<GFX, GFX12>)
        {
            snap.perf_snapshot_data = (inst_type << 2) | (reason << 6);
            snap.perf_snapshot_data |= 0x1;  // sample is valid
            snap.perf_snapshot_data1 = wave_cnt;
            snap.perf_snapshot_data1 |= (arb_issue << 6) | (arb_stall << 14);
        }

        EXPECT_NE(dispatch.get(), nullptr);
        dispatch->submit(packet_union_t{.snap = snap});
    };

protected:
    std::shared_ptr<MockRuntimeBuffer<PcSamplingRecordT>> buffer;
    std::shared_ptr<MockQueue<PcSamplingRecordT>>         queue;
    std::shared_ptr<MockDispatch<PcSamplingRecordT>>      dispatch;

    std::vector<PcSamplingRecordT> snapshots;
};

template <typename GFX, typename PcSamplingRecordT>
class WaveCntTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over all possible wave_cnt
        this->buffer->genUpcomingSamples(max_wave_number);
        // Only wave_cnt is relevant for this test
        for(size_t i = 0; i < max_wave_number; i++)
            this->genPCSample(
                i, GFX::TYPE_LDS, GFX::REASON_ALU_DEPENDENCY, GFX::ISSUE_SCALAR, GFX::ISSUE_VALU);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), max_wave_number);

        for(size_t i = 0; i < max_wave_number; i++)
            EXPECT_EQ(parsed[0][i].wave_count, i);
    }

protected:
    const size_t max_wave_number = GFX::max_wave_cnt;
};

template <typename GFX, typename PcSamplingRecordT>
class InstTypeTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over inst_type_issued
        generate_records_inst_type();
        this->buffer->genUpcomingSamples(GFX::TYPE_LAST);
        // Only inst_type is relevant for this test
        for(int i = 0; i < GFX::TYPE_LAST; i++)
            this->genPCSample(
                i, i, GFX::REASON_NO_INSTRUCTION_AVAILABLE, GFX::ISSUE_SCALAR, GFX::ISSUE_VALU);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX::TYPE_LAST);
        EXPECT_EQ(this->snapshots.size(), GFX::TYPE_LAST);

        for(size_t i = 0; i < GFX::TYPE_LAST; i++)
            EXPECT_EQ(this->snapshots[i].inst_type, parsed[0][i].inst_type);
    }

    virtual void generate_records_inst_type() = 0;
};

template <typename GFX, typename PcSamplingRecordT>
class StallReasonTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over reason_not_issued
        generate_records_not_issued_reason();
        this->buffer->genUpcomingSamples(GFX::REASON_LAST);
        // no issue reason is the only relevant for this test
        for(int i = 0; i < GFX::REASON_LAST; i++)
            this->genPCSample(i, GFX::TYPE_MATRIX, i, GFX::ISSUE_VALU, GFX::ISSUE_LDS);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX::REASON_LAST);
        EXPECT_EQ(this->snapshots.size(), GFX::REASON_LAST);

        for(size_t i = 0; i < GFX::REASON_LAST; i++)
            EXPECT_EQ(this->snapshots[i].snapshot.reason_not_issued,
                      parsed[0][i].snapshot.reason_not_issued);
    }

    virtual void generate_records_not_issued_reason() = 0;
};

template <typename GFX, typename PcSamplingRecordT>
class ArbStateTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over arb_state_issue
        generate_records_arbstate_issue();
        this->buffer->genUpcomingSamples(GFX::ISSUE_LAST * GFX::ISSUE_LAST);
        // To match the order of instantiating snapshots inside `generate_records_arbstate_issue`
        // we loop over GFX::
        for(int i = 0; i < GFX::ISSUE_LAST; i++)
            for(int j = 0; j < GFX::ISSUE_LAST; j++)
                this->genPCSample(i, GFX::TYPE_MATRIX, GFX::REASON_ALU_DEPENDENCY, 1 << i, 1 << j);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX::ISSUE_LAST * GFX::ISSUE_LAST);
        EXPECT_EQ(this->snapshots.size(), GFX::ISSUE_LAST * GFX::ISSUE_LAST);

        for(size_t i = 0; i < GFX::ISSUE_LAST * GFX::ISSUE_LAST; i++)
        {
            auto& snap = this->snapshots[i];
            match_arbstate(snap, parsed[0][i]);
        }
    }

    virtual void generate_records_arbstate_issue()                          = 0;
    virtual void match_arbstate(PcSamplingRecordT& x, PcSamplingRecordT& y) = 0;
};

template <typename GFX, typename PcSamplingRecordT, typename PcSamplingRecordInvalidT>
class WaveIssueAndErrorTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    struct pc_sampling_test_record_t
    {
        bool valid;
        union
        {
            PcSamplingRecordT        valid_record;
            PcSamplingRecordInvalidT invalid_record;
        };
    };

protected:
    std::vector<pc_sampling_test_record_t> compare;
};

template <typename GFX, typename PcSamplingRecordT>
class WaveOtherFieldsTest : public WaveSnapTest<GFX, PcSamplingRecordT>
{
protected:
    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(3);
        this->genPCSample(1, 2, 3, 4, 5, 6, 7);       // Counting
        this->genPCSample(3, 5, 7, 11, 13, 17, 19);   // Some prime numbers
        this->genPCSample(23, 19, 17, 13, 11, 7, 5);  // Some reversed primes
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), 3);
        EXPECT_EQ(this->snapshots.size(), 3);

        for(size_t i = 0; i < 3; i++)
        {
            if constexpr(std::is_same<GFX, GFX12>::value)
            {
                // GFX12 has no chiplets
                EXPECT_EQ(0, parsed[0][i].hw_id.chiplet);
                if constexpr(std::is_same<PcSamplingRecordT,
                                          rocprofiler_pc_sampling_record_stochastic_v0_t>::value)
                {
                    // Memory counters are introduced in GFX12 stochastc
                    EXPECT_EQ(1, parsed[0][i].flags.has_memory_counter);
                }
            }

            EXPECT_EQ(this->snapshots[i].exec_mask, parsed[0][i].exec_mask);
            EXPECT_EQ(this->snapshots[i].workgroup_id, parsed[0][i].workgroup_id);

            // No matter what we passed to the genPCSample, chiplet is 0 on Navi4x
            EXPECT_EQ(this->snapshots[i].hw_id.chiplet, parsed[0][i].hw_id.chiplet);
            EXPECT_EQ(this->snapshots[i].wave_in_group, parsed[0][i].wave_in_group);
            EXPECT_EQ(this->snapshots[i].correlation_id.internal,
                      parsed[0][i].correlation_id.internal);
        }
    }

    virtual void genPCSample(int pc, int exec, int blkx, int blky, int blkz, int chip, int wave)
    {
        PcSamplingRecordT sample;
        ::memset(&sample, 0, sizeof(sample));

        sample.exec_mask      = exec;
        sample.workgroup_id.x = blkx;
        sample.workgroup_id.y = blky;
        sample.workgroup_id.z = blkz;

        sample.hw_id.chiplet           = chip;
        sample.wave_in_group           = wave;
        sample.correlation_id.internal = this->dispatch->unique_id;

        this->snapshots.push_back(sample);

        // We're testing fields commong for both perf_sample_host_trap_v1 and
        // perf_sample_snapshot_v1, so either struct is suitable here. No need to make
        // specialization,
        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));
        snap.exec_mask = exec;

        snap.workgroup_id_x      = blkx;
        snap.workgroup_id_y      = blky;
        snap.workgroup_id_z      = blkz;
        snap.chiplet_and_wave_id = (chip << 8) | (wave & 0x3F);
        snap.correlation_id      = this->dispatch->getMockId().raw;

        // to ensure all stochastic samples are generated properly,
        // marked them as valid
        snap.perf_snapshot_data |= 0x1;  // set the bit indicating the sample is valid

        EXPECT_NE(this->dispatch.get(), nullptr);
        this->dispatch->submit(snap);

        (void) pc;
    };
};

/**
 * @brief Testing how mid_macro bit affects the PC address.
 *
 * On GFX950, this bit triggers correction of the PC address.
 * On other architectures, the PC address remains unchanged.
 */
template <typename GFX, typename PcSamplingRecordT>
class MidMacroPCCorrection : public WaveSnapTest<GFX, PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(3);
        // NOTE: mid_macro is relevant only on GFX950
        genPCSample(0x800, true);
        genPCSample(0x900, false);
        genPCSample(0x1000, true);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFX::gfx_ip_major, GFX::gfx_ip_minor);
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), 3);
        EXPECT_EQ(compare.size(), 3);

        for(size_t i = 0; i < 3; i++)
        {
            // verifying PC address
            EXPECT_EQ(parsed[0][i].pc.code_object_offset, compare[i].pc.code_object_offset);
        }
    }

    /**
     * @brief Generate PC sample with mid_macro flag.
     * The @p mid_macro is relevant for the GFX950, so it's false by default
     */
    virtual void genPCSample(uint64_t pc, bool mid_macro = false)
    {
        PcSamplingRecordT sample;
        ::memset(&sample, 0, sizeof(sample));
        // Calculate the expected PC address
        sample.pc.code_object_offset = calculateExpectedPC(pc, mid_macro);
        compare.push_back(sample);

        // This test considers only PC address.
        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));
        snap.pc = pc;
        // Mandatory for correlation mapping. Otherwise, parsing error occurs.
        snap.correlation_id = this->dispatch->getMockId().raw;

        // to ensure all stochastic samples are generated properly,
        // marked them as valid
        snap.perf_snapshot_data |= 0x1;  // set the bit indicating the sample is valid

        // the mid_macro is the bit at the position 31
        snap.perf_snapshot_data1 = (mid_macro << 31);

        EXPECT_NE(this->dispatch.get(), nullptr);
        this->dispatch->submit(snap);
    }

    /**
     * @brief Calculate expected PC address for comparison.
     */
    virtual uint64_t calculateExpectedPC(uint64_t pc, bool /*mid_macro*/) { return pc; }

protected:
    ///< testing data
    std::vector<PcSamplingRecordT> compare;
};
