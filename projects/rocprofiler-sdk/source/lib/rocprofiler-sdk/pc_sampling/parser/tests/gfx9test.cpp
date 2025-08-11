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

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/gfx9test.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <gtest/gtest.h>
#include <cstddef>

#define GFXIP_MAJOR 9

#define RECORD_INST_TYPE(x)                                                                        \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.inst_type = ROCPROFILER_PC_SAMPLING_INSTRUCTION##_##x;                              \
        snapshots.push_back(sample);                                                               \
    }

#define GENERATE_RECORDS_INST_TYPE()                                                               \
    RECORD_INST_TYPE(TYPE_VALU);                                                                   \
    RECORD_INST_TYPE(TYPE_MATRIX);                                                                 \
    RECORD_INST_TYPE(TYPE_SCALAR);                                                                 \
    RECORD_INST_TYPE(TYPE_TEX);                                                                    \
    RECORD_INST_TYPE(TYPE_LDS);                                                                    \
    RECORD_INST_TYPE(TYPE_FLAT);                                                                   \
    RECORD_INST_TYPE(TYPE_EXPORT);                                                                 \
    RECORD_INST_TYPE(TYPE_MESSAGE);                                                                \
    RECORD_INST_TYPE(TYPE_BARRIER);                                                                \
    RECORD_INST_TYPE(TYPE_BRANCH_NOT_TAKEN);                                                       \
    RECORD_INST_TYPE(TYPE_BRANCH_TAKEN);                                                           \
    RECORD_INST_TYPE(TYPE_JUMP);                                                                   \
    RECORD_INST_TYPE(TYPE_OTHER);                                                                  \
    RECORD_INST_TYPE(TYPE_NO_INST);

#define RECORD_NOT_ISSUED_REASON(x)                                                                \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.snapshot.reason_not_issued = ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED##_##x;  \
        snapshots.push_back(sample);                                                               \
    }

#define GENERATE_RECORDS_NOT_ISSUED_REASON(x)                                                      \
    RECORD_NOT_ISSUED_REASON(REASON_NO_INSTRUCTION_AVAILABLE);                                     \
    RECORD_NOT_ISSUED_REASON(REASON_ALU_DEPENDENCY);                                               \
    RECORD_NOT_ISSUED_REASON(REASON_WAITCNT);                                                      \
    RECORD_NOT_ISSUED_REASON(REASON_INTERNAL_INSTRUCTION);                                         \
    RECORD_NOT_ISSUED_REASON(REASON_BARRIER_WAIT);                                                 \
    RECORD_NOT_ISSUED_REASON(REASON_ARBITER_NOT_WIN);                                              \
    RECORD_NOT_ISSUED_REASON(REASON_ARBITER_WIN_EX_STALL);                                         \
    RECORD_NOT_ISSUED_REASON(REASON_OTHER_WAIT);

#define RECORD_ARBSTATE_ISSUE_STALL(x, y)                                                          \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.snapshot.arb_state##_##x = 1;                                                       \
        sample.snapshot.arb_state##_##y = 1;                                                       \
        snapshots.push_back(sample);                                                               \
    }

// Respecting the order of elements in GFX9:arb_state that match the order of arb_state bits
// in perf_snapshot_data register
#define RECORD_ARBSTATE_ISSUE(x)                                                                   \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_misc);                                                    \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_exp);                                                     \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_flat);                                                    \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_lds);                                                     \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_vmem_tex);                                                \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_scalar);                                                  \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_matrix);                                                  \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_valu);

// Respecting the order of elements in GFX9:arb_state that match the order of arb_state bits
// in perf_snapshot_data register
#define GENERATE_RECORDS_ARBSTATE_ISSUE()                                                          \
    RECORD_ARBSTATE_ISSUE(issue_misc);                                                             \
    RECORD_ARBSTATE_ISSUE(issue_exp);                                                              \
    RECORD_ARBSTATE_ISSUE(issue_flat);                                                             \
    RECORD_ARBSTATE_ISSUE(issue_lds);                                                              \
    RECORD_ARBSTATE_ISSUE(issue_vmem_tex);                                                         \
    RECORD_ARBSTATE_ISSUE(issue_scalar);                                                           \
    RECORD_ARBSTATE_ISSUE(issue_matrix);                                                           \
    RECORD_ARBSTATE_ISSUE(issue_valu);

#define NON_GFX9_ARBSTATE_IS_ZERO(x, y)                                                            \
    EXPECT_EQ(x.snapshot.arb_state_issue_lds_direct, 0);                                           \
    EXPECT_EQ(y.snapshot.arb_state_issue_lds_direct, 0);                                           \
    EXPECT_EQ(x.snapshot.arb_state_issue_brmsg, 0);                                                \
    EXPECT_EQ(y.snapshot.arb_state_issue_brmsg, 0);                                                \
                                                                                                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_lds_direct, 0);                                           \
    EXPECT_EQ(y.snapshot.arb_state_stall_lds_direct, 0);                                           \
    EXPECT_EQ(x.snapshot.arb_state_stall_brmsg, 0);                                                \
    EXPECT_EQ(y.snapshot.arb_state_stall_brmsg, 0);

#define MATCH_ARBSTATE(x, y)                                                                       \
    EXPECT_EQ(x.snapshot.arb_state_issue_valu, y.snapshot.arb_state_issue_valu);                   \
    EXPECT_EQ(x.snapshot.arb_state_issue_matrix, y.snapshot.arb_state_issue_matrix);               \
    EXPECT_EQ(x.snapshot.arb_state_issue_lds, y.snapshot.arb_state_issue_lds);                     \
    EXPECT_EQ(x.snapshot.arb_state_issue_scalar, y.snapshot.arb_state_issue_scalar);               \
    EXPECT_EQ(x.snapshot.arb_state_issue_vmem_tex, y.snapshot.arb_state_issue_vmem_tex);           \
    EXPECT_EQ(x.snapshot.arb_state_issue_flat, y.snapshot.arb_state_issue_flat);                   \
    EXPECT_EQ(x.snapshot.arb_state_issue_exp, y.snapshot.arb_state_issue_exp);                     \
    EXPECT_EQ(x.snapshot.arb_state_issue_misc, y.snapshot.arb_state_issue_misc);                   \
                                                                                                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_valu, y.snapshot.arb_state_stall_valu);                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_matrix, y.snapshot.arb_state_stall_matrix);               \
    EXPECT_EQ(x.snapshot.arb_state_stall_lds, y.snapshot.arb_state_stall_lds);                     \
    EXPECT_EQ(x.snapshot.arb_state_stall_scalar, y.snapshot.arb_state_stall_scalar);               \
    EXPECT_EQ(x.snapshot.arb_state_stall_vmem_tex, y.snapshot.arb_state_stall_vmem_tex);           \
    EXPECT_EQ(x.snapshot.arb_state_stall_flat, y.snapshot.arb_state_stall_flat);                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_exp, y.snapshot.arb_state_stall_exp);                     \
    EXPECT_EQ(x.snapshot.arb_state_stall_misc, y.snapshot.arb_state_stall_misc);                   \
                                                                                                   \
    NON_GFX9_ARBSTATE_IS_ZERO(x, y)

template <typename PcSamplingRecordT>
class WaveCntTest : public WaveSnapTest<PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over all possible wave_cnt
        this->buffer->genUpcomingSamples(max_wave_number);
        for(size_t i = 0; i < max_wave_number; i++)
            this->genPCSample(
                i, GFX9::TYPE_LDS, GFX9::REASON_ALU_DEPENDENCY, GFX9::ISSUE_VALU, GFX9::ISSUE_VALU);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), max_wave_number);

        for(size_t i = 0; i < max_wave_number; i++)
            EXPECT_EQ(parsed[0][i].wave_count, i);
    }

    const size_t                   max_wave_number = 64;
    std::vector<PcSamplingRecordT> snapshots;
};

template <typename PcSamplingRecordT>
class InstTypeTest : public WaveSnapTest<PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over inst_type_issued
        GENERATE_RECORDS_INST_TYPE();
        this->buffer->genUpcomingSamples(GFX9::TYPE_LAST);
        for(int i = 0; i < GFX9::TYPE_LAST; i++)
            this->genPCSample(
                i, i, GFX9::REASON_ALU_DEPENDENCY, GFX9::ISSUE_MATRIX, GFX9::ISSUE_MATRIX);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX9::TYPE_LAST);
        EXPECT_EQ(snapshots.size(), GFX9::TYPE_LAST);

        for(size_t i = 0; i < GFX9::TYPE_LAST; i++)
            EXPECT_EQ(snapshots[i].inst_type, parsed[0][i].inst_type);
    }

    std::vector<PcSamplingRecordT> snapshots;
};

template <typename PcSamplingRecordT>
class StallReasonTest : public WaveSnapTest<PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over reason_not_issued
        GENERATE_RECORDS_NOT_ISSUED_REASON();
        this->buffer->genUpcomingSamples(GFX9::REASON_LAST);
        for(int i = 0; i < GFX9::REASON_LAST; i++)
            this->genPCSample(i, GFX9::TYPE_MATRIX, i, GFX9::ISSUE_MATRIX, GFX9::ISSUE_MATRIX);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX9::REASON_LAST);
        EXPECT_EQ(snapshots.size(), GFX9::REASON_LAST);

        for(size_t i = 0; i < GFX9::REASON_LAST; i++)
            EXPECT_EQ(snapshots[i].snapshot.reason_not_issued,
                      parsed[0][i].snapshot.reason_not_issued);
    }

    std::vector<PcSamplingRecordT> snapshots;
};

template <typename PcSamplingRecordT>
class ArbStateTest : public WaveSnapTest<PcSamplingRecordT>
{
public:
    void FillBuffers() override
    {
        // Loop over arb_state_issue
        GENERATE_RECORDS_ARBSTATE_ISSUE();
        this->buffer->genUpcomingSamples(GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);
        // To match the order of instantiating snapshots inside `GENERATE_RECORDS_ARBSTATE_ISSUE`
        // we loop over GFX9::
        for(int i = 0; i < GFX9::ISSUE_LAST; i++)
            for(int j = 0; j < GFX9::ISSUE_LAST; j++)
                this->genPCSample(
                    i, GFX9::TYPE_MATRIX, GFX9::REASON_ALU_DEPENDENCY, 1 << i, 1 << j);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);
        EXPECT_EQ(snapshots.size(), GFX9::ISSUE_LAST * GFX9::ISSUE_LAST);

        for(size_t i = 0; i < GFX9::ISSUE_LAST * GFX9::ISSUE_LAST; i++)
        {
            auto& snap = snapshots[i];
            MATCH_ARBSTATE(snap, parsed[0][i])
        }
    }

    std::vector<PcSamplingRecordT> snapshots;
};

template <typename PcSamplingRecordT, typename PcSamplingRecordInvalidT>
class WaveIssueAndErrorTest : public WaveSnapTest<PcSamplingRecordT>
{
    struct pc_sampling_test_record_t
    {
        bool valid;
        union
        {
            PcSamplingRecordT        valid_record;
            PcSamplingRecordInvalidT invalid_record;
        };
    };

    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(16);
        for(int valid = 0; valid <= 1; valid++)
            for(int issued = 0; issued <= 1; issued++)
                for(int dual = 0; dual <= 1; dual++)
                    for(int error = 0; error <= 1; error++)
                        genPCSample(valid, issued, dual, error);
    }

    void CheckBuffers() override
    {
        const int num_combinations = 16;
        auto      parsed           = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), num_combinations);
        EXPECT_EQ(compare.size(), num_combinations);

        for(size_t i = 0; i < num_combinations; i++)
        {
            if(compare[i].valid)
            {
                EXPECT_EQ(compare[i].valid_record.wave_issued, parsed[0][i].wave_issued);
                EXPECT_EQ(compare[i].valid_record.snapshot.dual_issue_valu,
                          parsed[0][i].snapshot.dual_issue_valu);
            }
            else
            {
                // Internally (inside the parser) invalid samples are represented with
                // PcSamplingRecordT of size 0. Eventually, those records are replaced with the
                // PcSamplingRecordInvalidT prior to putting inside the SDK buffer.
                EXPECT_EQ(parsed[0][i].size, 0);
            }
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
        pc_sampling_test_record_t record{};
        record.valid = valid && !error;
        if(record.valid)
        {
            // Fill in the data for the valid record.
            auto& sample = record.valid_record;

            // TODO: Since code objects are not mocked, use pc.code_object_offset
            // as the absolute physical address of the mocked PC.
            sample.pc.code_object_offset = this->dispatch->unique_id;

            sample.correlation_id.internal = this->dispatch->getMockId().raw;

            sample.wave_issued              = issued;
            sample.snapshot.dual_issue_valu = dual;

            EXPECT_NE(this->dispatch.get(), nullptr);
        }

        compare.push_back(record);

        trap_snapshot_v1 snap;
        snap.valid  = valid;
        snap.issued = issued;
        snap.dual   = dual;
        snap.error  = error;

        perf_sample_snapshot_v1 pss;
        pss.perf_snapshot_data = snap.raw;
        pss.correlation_id     = this->dispatch->getMockId().raw;
        this->dispatch->submit(std::move(pss));
    };

    std::vector<pc_sampling_test_record_t> compare;
};

template <typename PcSamplingRecordT>
class HwIdTest : public WaveSnapTest<PcSamplingRecordT>
{
    union gfx9_hw_id_t
    {
        uint32_t raw;
        struct
        {
            uint32_t wave_id : 4;  ///< wave slot index
            uint32_t simd_id : 2;  ///< SIMD index
            uint32_t pipe_id : 2;  ///< pipe index
            uint32_t cu_id : 4;  ///< Index of compute unit on GFX9 or workgroup processer on other
                                 ///< architectures
            uint32_t shader_array_id  : 1;  ///< Shared array index
            uint32_t shader_engine_id : 3;  ///< shared engine index
            uint32_t
                threadgroup_id : 4;  ///< thread_group index on GFX9, and workgroup index on GFX10+
            uint32_t vm_id     : 4;  ///< virtual memory ID
            uint32_t queue_id  : 3;  ///< queue id
            uint32_t gfx_context_state_id : 3;  ///< GFX context (state) id (only on GFX9) - ignored
            uint32_t microengine_id       : 2;  ///< ACE (microengine) index
        };
    };

    void FillBuffers() override
    {
        gfx9_hw_id_t hw_id_val0;
        hw_id_val0.wave_id              = 0;
        hw_id_val0.simd_id              = 0;
        hw_id_val0.pipe_id              = 0;
        hw_id_val0.cu_id                = 0;
        hw_id_val0.shader_array_id      = 0;
        hw_id_val0.shader_engine_id     = 0;
        hw_id_val0.threadgroup_id       = 0;
        hw_id_val0.vm_id                = 0;
        hw_id_val0.queue_id             = 0;
        hw_id_val0.gfx_context_state_id = 0;
        hw_id_val0.microengine_id       = 0;

        gfx9_hw_id_t hw_id_val1;
        hw_id_val0.wave_id              = 15;
        hw_id_val0.simd_id              = 3;
        hw_id_val0.pipe_id              = 3;
        hw_id_val0.cu_id                = 15;
        hw_id_val0.shader_array_id      = 1;
        hw_id_val0.shader_engine_id     = 7;
        hw_id_val0.threadgroup_id       = 15;
        hw_id_val0.vm_id                = 15;
        hw_id_val0.queue_id             = 7;
        hw_id_val0.gfx_context_state_id = 7;
        hw_id_val0.microengine_id       = 3;

        gfx9_hw_id_t hw_id_val2;
        hw_id_val2.wave_id              = 7;
        hw_id_val2.simd_id              = 2;
        hw_id_val2.pipe_id              = 2;
        hw_id_val2.cu_id                = 6;
        hw_id_val2.shader_array_id      = 0;
        hw_id_val2.shader_engine_id     = 3;
        hw_id_val2.threadgroup_id       = 8;
        hw_id_val2.vm_id                = 9;
        hw_id_val2.queue_id             = 3;
        hw_id_val2.gfx_context_state_id = 2;
        hw_id_val2.microengine_id       = 1;

        this->buffer->genUpcomingSamples(3);
        genPCSample(hw_id_val0);
        genPCSample(hw_id_val1);
        genPCSample(hw_id_val2);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), 3);
        EXPECT_EQ(compare.size(), 3);

        for(size_t i = 0; i < 3; i++)
        {
            // Comparing individual fields
            EXPECT_EQ(compare[i].hw_id.wave_id, parsed[0][i].hw_id.wave_id);
            EXPECT_EQ(compare[i].hw_id.simd_id, parsed[0][i].hw_id.simd_id);
            EXPECT_EQ(compare[i].hw_id.pipe_id, parsed[0][i].hw_id.pipe_id);
            EXPECT_EQ(compare[i].hw_id.cu_or_wgp_id, parsed[0][i].hw_id.cu_or_wgp_id);
            EXPECT_EQ(compare[i].hw_id.shader_array_id, parsed[0][i].hw_id.shader_array_id);
            EXPECT_EQ(compare[i].hw_id.shader_engine_id, parsed[0][i].hw_id.shader_engine_id);
            EXPECT_EQ(compare[i].hw_id.workgroup_id, parsed[0][i].hw_id.workgroup_id);
            EXPECT_EQ(compare[i].hw_id.vm_id, parsed[0][i].hw_id.vm_id);
            EXPECT_EQ(compare[i].hw_id.queue_id, parsed[0][i].hw_id.queue_id);
            EXPECT_EQ(compare[i].hw_id.microengine_id, parsed[0][i].hw_id.microengine_id);
        }
    }

    void genPCSample(gfx9_hw_id_t hw_id)
    {
        PcSamplingRecordT sample;
        ::memset(&sample, 0, sizeof(sample));
        // Unpacking individual fields
        // NOTE: chiplet is tested in a WaveOtherFieldsTest test, becuase it's not
        // transferred via hw_id, but chiplet_and_wave_id field.
        sample.hw_id.wave_id          = hw_id.wave_id;
        sample.hw_id.simd_id          = hw_id.simd_id;
        sample.hw_id.pipe_id          = hw_id.pipe_id;
        sample.hw_id.cu_or_wgp_id     = hw_id.cu_id;
        sample.hw_id.shader_array_id  = hw_id.shader_array_id;
        sample.hw_id.shader_engine_id = hw_id.shader_engine_id;
        sample.hw_id.workgroup_id     = hw_id.threadgroup_id;
        sample.hw_id.vm_id            = hw_id.vm_id;
        sample.hw_id.queue_id         = hw_id.queue_id;
        sample.hw_id.microengine_id   = hw_id.microengine_id;

        compare.push_back(sample);

        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));

        // raw register value
        snap.hw_id          = hw_id.raw;
        snap.correlation_id = this->dispatch->getMockId().raw;
        snap.perf_snapshot_data |= 0x1;  // sample is valid

        EXPECT_NE(this->dispatch.get(), nullptr);
        this->dispatch->submit(snap);
    };

    std::vector<PcSamplingRecordT> compare;
};

template <typename PcSamplingRecordT>
class WaveOtherFieldsTest : public WaveSnapTest<PcSamplingRecordT>
{
    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(3);
        genPCSample(1, 2, 3, 4, 5, 6, 7);       // Counting
        genPCSample(3, 5, 7, 11, 13, 17, 19);   // Some prime numbers
        genPCSample(23, 19, 17, 13, 11, 7, 5);  // Some reversed primes
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(9);  // GFXIP==9
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), 3);
        EXPECT_EQ(compare.size(), 3);

        for(size_t i = 0; i < 3; i++)
        {
            // TODO: if we decide to test flags, make specialization for
            // rocprofiler_pc_sampling_record_stochastic_v0_t
            // EXPECT_EQ(parsed[0][i].flags.has_stall_reason, true);
            // EXPECT_EQ(parsed[0][i].flags.has_wave_cnt, true);
            // EXPECT_EQ(parsed[0][i].flags.reserved, false);

            EXPECT_EQ(compare[i].exec_mask, parsed[0][i].exec_mask);
            EXPECT_EQ(compare[i].workgroup_id, parsed[0][i].workgroup_id);

            EXPECT_EQ(compare[i].hw_id.chiplet, parsed[0][i].hw_id.chiplet);
            EXPECT_EQ(compare[i].wave_in_group, parsed[0][i].wave_in_group);
            // TODO: handle HW_ID as well.
            // EXPECT_EQ(compare[i].hw_id, parsed[0][i].hw_id);
            EXPECT_EQ(compare[i].correlation_id.internal, parsed[0][i].correlation_id.internal);
        }
    }

    void genPCSample(int pc, int exec, int blkx, int blky, int blkz, int chip, int wave)
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

        compare.push_back(sample);

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

    std::vector<PcSamplingRecordT> compare;
};

/**
 * @brief This test verifies that the PC address remains unchanged for GFX9.
 */
template <typename PcSamplingRecordT>
void
MidMacroPCCorrection<PcSamplingRecordT>::FillBuffers()
{
    this->buffer->genUpcomingSamples(3);
    // NOTE: mid_macro is relevant only on GFX950
    genPCSample(0x800, true);
    genPCSample(0x900, false);
    genPCSample(0x1000, true);
}

template <typename PcSamplingRecordT>
std::vector<std::vector<PcSamplingRecordT>>
MidMacroPCCorrection<PcSamplingRecordT>::get_parsed_data()
{
    return this->buffer->get_parsed_buffer(9);  // GFXIP==9
}

template <typename PcSamplingRecordT>
void
MidMacroPCCorrection<PcSamplingRecordT>::CheckBuffers()
{
    auto parsed = get_parsed_data();
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
 * @brief By default, PC address remains unchanged.
 */
template <typename PcSamplingRecordT>
uint64_t
MidMacroPCCorrection<PcSamplingRecordT>::calcaulteExpectedPC(uint64_t pc, bool /*mid_macro*/)
{
    return pc;
}

template <typename PcSamplingRecordT>
void
MidMacroPCCorrection<PcSamplingRecordT>::genPCSample(uint64_t pc, bool mid_macro)
{
    PcSamplingRecordT sample;
    ::memset(&sample, 0, sizeof(sample));
    // Calculate the expected PC address
    sample.pc.code_object_offset = calcaulteExpectedPC(pc, mid_macro);
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

TEST(pcs_parser, gfx9_test)
{
    // Tests specific to stochastic sampling only
    WaveCntTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    InstTypeTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    StallReasonTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    ArbStateTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    WaveIssueAndErrorTest<rocprofiler_pc_sampling_record_stochastic_v0_t,
                          rocprofiler_pc_sampling_record_invalid_t>{}
        .Test();

    // Tests commong for both host trap and stochastic sampling.
    HwIdTest<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    HwIdTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    WaveOtherFieldsTest<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    WaveOtherFieldsTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();

    MidMacroPCCorrection<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    MidMacroPCCorrection<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();

    std::cout << "GFX9 Test Done." << std::endl;
}
