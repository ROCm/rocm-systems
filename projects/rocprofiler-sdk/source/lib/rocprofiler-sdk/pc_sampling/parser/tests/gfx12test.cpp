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

#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/gfxtest.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

#include <rocprofiler-sdk/cxx/operators.hpp>

#include <gtest/gtest.h>
#include <cstddef>

#define GFXIP_MAJOR 12

#define RECORD_INST_TYPE(x)                                                                        \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.inst_type = ROCPROFILER_PC_SAMPLING_INSTRUCTION##_##x;                              \
        this->snapshots.push_back(sample);                                                         \
    }

#define GENERATE_RECORDS_INST_TYPE()                                                               \
    RECORD_INST_TYPE(TYPE_VALU);                                                                   \
    RECORD_INST_TYPE(TYPE_SCALAR);                                                                 \
    RECORD_INST_TYPE(TYPE_TEX);                                                                    \
    RECORD_INST_TYPE(TYPE_LDS);                                                                    \
    RECORD_INST_TYPE(TYPE_LDS_DIRECT);                                                             \
    RECORD_INST_TYPE(TYPE_EXPORT);                                                                 \
    RECORD_INST_TYPE(TYPE_MESSAGE);                                                                \
    RECORD_INST_TYPE(TYPE_BARRIER);                                                                \
    RECORD_INST_TYPE(TYPE_BRANCH_NOT_TAKEN);                                                       \
    RECORD_INST_TYPE(TYPE_BRANCH_TAKEN);                                                           \
    RECORD_INST_TYPE(TYPE_JUMP);                                                                   \
    RECORD_INST_TYPE(TYPE_OTHER);                                                                  \
    RECORD_INST_TYPE(TYPE_NO_INST);                                                                \
    RECORD_INST_TYPE(TYPE_DUAL_VALU);                                                              \
    RECORD_INST_TYPE(TYPE_FLAT);                                                                   \
    RECORD_INST_TYPE(TYPE_MATRIX);

#define RECORD_NOT_ISSUED_REASON(x)                                                                \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.snapshot.reason_not_issued = ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED##_##x;  \
        this->snapshots.push_back(sample);                                                         \
    }

#define GENERATE_RECORDS_NOT_ISSUED_REASON(x)                                                      \
    RECORD_NOT_ISSUED_REASON(REASON_NO_INSTRUCTION_AVAILABLE);                                     \
    RECORD_NOT_ISSUED_REASON(REASON_ALU_DEPENDENCY);                                               \
    RECORD_NOT_ISSUED_REASON(REASON_WAITCNT);                                                      \
    RECORD_NOT_ISSUED_REASON(REASON_ARBITER_NOT_WIN);                                              \
    RECORD_NOT_ISSUED_REASON(REASON_SLEEP_WAIT);                                                   \
    RECORD_NOT_ISSUED_REASON(REASON_BARRIER_WAIT);                                                 \
    RECORD_NOT_ISSUED_REASON(REASON_OTHER_WAIT);                                                   \
    RECORD_NOT_ISSUED_REASON(REASON_INTERNAL_INSTRUCTION);

#define RECORD_ARBSTATE_ISSUE_STALL(x, y)                                                          \
    {                                                                                              \
        PcSamplingRecordT sample{};                                                                \
        sample.snapshot.arb_state##_##x = 1;                                                       \
        sample.snapshot.arb_state##_##y = 1;                                                       \
        this->snapshots.push_back(sample);                                                         \
    }

// Respecting the order of elements in GFX12:arb_state that match the order of arb_state bits
// in perf_snapshot_data register
#define RECORD_ARBSTATE_ISSUE(x)                                                                   \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_brmsg);                                                   \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_exp);                                                     \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_lds_direct);                                              \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_lds);                                                     \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_vmem_tex);                                                \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_scalar);                                                  \
    RECORD_ARBSTATE_ISSUE_STALL(x, stall_valu);

// Respecting the order of elements in GFX12:arb_state that match the order of arb_state bits
// in perf_snapshot_data register
#define GENERATE_RECORDS_ARBSTATE_ISSUE()                                                          \
    RECORD_ARBSTATE_ISSUE(issue_brmsg);                                                            \
    RECORD_ARBSTATE_ISSUE(issue_exp);                                                              \
    RECORD_ARBSTATE_ISSUE(issue_lds_direct);                                                       \
    RECORD_ARBSTATE_ISSUE(issue_lds);                                                              \
    RECORD_ARBSTATE_ISSUE(issue_vmem_tex);                                                         \
    RECORD_ARBSTATE_ISSUE(issue_scalar);                                                           \
    RECORD_ARBSTATE_ISSUE(issue_valu);

#define NON_GFX12_ARBSTATE_IS_ZERO(x, y)                                                           \
    EXPECT_EQ(x.snapshot.arb_state_issue_misc, 0);                                                 \
    EXPECT_EQ(y.snapshot.arb_state_issue_misc, 0);                                                 \
    EXPECT_EQ(x.snapshot.arb_state_issue_matrix, 0);                                               \
    EXPECT_EQ(y.snapshot.arb_state_issue_matrix, 0);                                               \
    EXPECT_EQ(x.snapshot.arb_state_issue_flat, 0);                                                 \
    EXPECT_EQ(y.snapshot.arb_state_issue_flat, 0);                                                 \
                                                                                                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_misc, 0);                                                 \
    EXPECT_EQ(y.snapshot.arb_state_stall_misc, 0);                                                 \
    EXPECT_EQ(x.snapshot.arb_state_stall_matrix, 0);                                               \
    EXPECT_EQ(y.snapshot.arb_state_stall_matrix, 0);                                               \
    EXPECT_EQ(x.snapshot.arb_state_stall_flat, 0);                                                 \
    EXPECT_EQ(y.snapshot.arb_state_stall_flat, 0);

#define MATCH_ARBSTATE(x, y)                                                                       \
    EXPECT_EQ(x.snapshot.arb_state_issue_valu, y.snapshot.arb_state_issue_valu);                   \
    EXPECT_EQ(x.snapshot.arb_state_issue_lds, y.snapshot.arb_state_issue_lds);                     \
    EXPECT_EQ(x.snapshot.arb_state_issue_lds_direct, y.snapshot.arb_state_issue_lds_direct);       \
    EXPECT_EQ(x.snapshot.arb_state_issue_scalar, y.snapshot.arb_state_issue_scalar);               \
    EXPECT_EQ(x.snapshot.arb_state_issue_vmem_tex, y.snapshot.arb_state_issue_vmem_tex);           \
    EXPECT_EQ(x.snapshot.arb_state_issue_exp, y.snapshot.arb_state_issue_exp);                     \
    EXPECT_EQ(x.snapshot.arb_state_issue_brmsg, y.snapshot.arb_state_issue_brmsg);                 \
                                                                                                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_valu, y.snapshot.arb_state_stall_valu);                   \
    EXPECT_EQ(x.snapshot.arb_state_stall_lds, y.snapshot.arb_state_stall_lds);                     \
    EXPECT_EQ(x.snapshot.arb_state_stall_lds_direct, y.snapshot.arb_state_stall_lds_direct);       \
    EXPECT_EQ(x.snapshot.arb_state_stall_scalar, y.snapshot.arb_state_stall_scalar);               \
    EXPECT_EQ(x.snapshot.arb_state_stall_vmem_tex, y.snapshot.arb_state_stall_vmem_tex);           \
    EXPECT_EQ(x.snapshot.arb_state_stall_exp, y.snapshot.arb_state_stall_exp);                     \
    EXPECT_EQ(x.snapshot.arb_state_stall_brmsg, y.snapshot.arb_state_stall_brmsg);                 \
                                                                                                   \
    NON_GFX12_ARBSTATE_IS_ZERO(x, y)

template <typename PcSamplingRecordT>
class InstTypeTestGFX12 : public InstTypeTest<GFX12, PcSamplingRecordT>
{
public:
    void generate_records_inst_type() override { GENERATE_RECORDS_INST_TYPE(); }
};

template <typename PcSamplingRecordT>
class StallReasonTestGFX12 : public StallReasonTest<GFX12, PcSamplingRecordT>
{
public:
    void generate_records_not_issued_reason() override { GENERATE_RECORDS_NOT_ISSUED_REASON(); }
};

template <typename PcSamplingRecordT>
class ArbStateTestGFX12 : public ArbStateTest<GFX12, PcSamplingRecordT>
{
public:
    void generate_records_arbstate_issue() override { GENERATE_RECORDS_ARBSTATE_ISSUE(); }

    void match_arbstate(PcSamplingRecordT& x, PcSamplingRecordT& y) override
    {
        MATCH_ARBSTATE(x, y);
    }
};

template <typename PcSamplingRecordT, typename PcSamplingRecordInvalidT>
class WaveIssueAndErrorTestGFX12
: public WaveIssueAndErrorTest<GFX12, PcSamplingRecordT, PcSamplingRecordInvalidT>
{
    // Encodes bits from the perf_snapshot_data register
    union perf_snapshot_data_t
    {
        struct
        {
            uint32_t valid    : 1;
            uint32_t issued   : 1;
            uint32_t reserved : 30;
        };
        uint32_t raw;
    };

    // specific
    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(4);
        for(int valid = 0; valid <= 1; valid++)
            for(int issued = 0; issued <= 1; issued++)
                genPCSample(valid, issued);
    }

    // Could be reused with assumption that the num_combinations will be overriden
    void CheckBuffers() override
    {
        const int num_combinations = 4;
        auto      parsed           = this->buffer->get_parsed_buffer(GFXIP_MAJOR);  // GFXIP==12
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), num_combinations);
        EXPECT_EQ(this->compare.size(), num_combinations);

        for(size_t i = 0; i < num_combinations; i++)
        {
            if(this->compare[i].valid)
            {
                EXPECT_EQ(this->compare[i].valid_record.wave_issued, parsed[0][i].wave_issued);
                // dual_issue_valu doesn't exist on GFX12, so we expect it to be 0 always
                EXPECT_EQ(0, parsed[0][i].snapshot.dual_issue_valu);
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

    void genPCSample(bool valid, bool issued)
    {
        typename WaveIssueAndErrorTest<GFX12, PcSamplingRecordT, PcSamplingRecordInvalidT>::
            pc_sampling_test_record_t record{};
        record.valid = valid;
        if(record.valid)
        {
            // Fill in the data for the valid record.
            auto& sample = record.valid_record;

            // TODO: Since code objects are not mocked, use pc.code_object_offset
            // as the absolute physical address of the mocked PC.
            sample.pc.code_object_offset = this->dispatch->unique_id;

            sample.correlation_id.internal = this->dispatch->getMockId().raw;

            sample.wave_issued = issued;

            EXPECT_NE(this->dispatch.get(), nullptr);
        }

        this->compare.push_back(record);

        perf_snapshot_data_t perf_snapshot_data{};
        perf_snapshot_data.valid  = valid;
        perf_snapshot_data.issued = issued;

        perf_sample_snapshot_v1 pss;
        pss.perf_snapshot_data = perf_snapshot_data.raw;
        pss.correlation_id     = this->dispatch->getMockId().raw;
        this->dispatch->submit(std::move(pss));
    };
};

template <typename PcSamplingRecordT>
class HwIdTest : public WaveSnapTest<GFX12, PcSamplingRecordT>
{
    // The combined hw_id1 and hw_id2 encoded by ROCr's 2nd level trap hadler
    union gfx12_hw_id_t
    {
        uint32_t raw;
        struct
        {
            uint32_t wave_id          : 5;  ///< wave_id[4:0]
            uint32_t queue_id         : 4;  ///< queue_id[8:5]
            uint32_t reserved0        : 1;  ///< reserved [9]
            uint32_t cu_or_wgp_id     : 4;  ///< wgp_id[13:10]
            uint32_t simd_id          : 2;  ///< simd_id[15:14]
            uint32_t shader_array_id  : 1;  ///< sa_id[16]
            uint32_t microengine_id   : 1;  ///< me_id[17]
            uint32_t shader_engine_id : 2;  ///< se_id[19:18]
            uint32_t pipe_id          : 2;  ///< pipe_id[21:20]
            uint32_t reserved1        : 1;  ///< reserved [22]
            uint32_t workgroup_id     : 5;  ///< wg_id[27:23]
            uint32_t vm_id            : 4;  ///< vm_id[31:28]
        };
    };

    void FillBuffers() override
    {
        gfx12_hw_id_t hw_id_val0{};
        hw_id_val0.wave_id          = 0;
        hw_id_val0.simd_id          = 0;
        hw_id_val0.cu_or_wgp_id     = 0;
        hw_id_val0.shader_array_id  = 0;
        hw_id_val0.shader_engine_id = 0;
        hw_id_val0.queue_id         = 0;
        hw_id_val0.pipe_id          = 0;
        hw_id_val0.microengine_id   = 0;
        hw_id_val0.workgroup_id     = 0;
        hw_id_val0.vm_id            = 0;

        gfx12_hw_id_t hw_id_val1{};
        hw_id_val1.wave_id          = 15;
        hw_id_val1.simd_id          = 3;
        hw_id_val1.cu_or_wgp_id     = 15;
        hw_id_val1.shader_array_id  = 1;
        hw_id_val1.shader_engine_id = 2;
        hw_id_val1.queue_id         = 7;
        hw_id_val1.pipe_id          = 3;
        hw_id_val1.microengine_id   = 1;
        hw_id_val1.workgroup_id     = 15;
        hw_id_val1.vm_id            = 15;

        gfx12_hw_id_t hw_id_val2{};
        hw_id_val2.wave_id          = 7;
        hw_id_val2.simd_id          = 2;
        hw_id_val2.cu_or_wgp_id     = 6;
        hw_id_val2.shader_array_id  = 0;
        hw_id_val2.shader_engine_id = 3;
        hw_id_val2.queue_id         = 3;
        hw_id_val2.pipe_id          = 2;
        hw_id_val2.microengine_id   = 1;
        hw_id_val2.workgroup_id     = 8;
        hw_id_val2.vm_id            = 9;

        this->buffer->genUpcomingSamples(3);
        genPCSample(hw_id_val0);
        genPCSample(hw_id_val1);
        genPCSample(hw_id_val2);
    }

    void CheckBuffers() override
    {
        auto parsed = this->buffer->get_parsed_buffer(GFXIP_MAJOR);  // GFXIP==12
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

    void genPCSample(gfx12_hw_id_t hw_id)
    {
        // ROCr doesn't deliver the info store in hw_id2
        PcSamplingRecordT sample;
        ::memset(&sample, 0, sizeof(sample));

        // Unpacking individual fields
        // NOTE: chiplet is tested in a WaveOtherFieldsTest test, becuase it's not
        // transferred via hw_id, but chiplet_and_wave_id field.
        sample.hw_id.wave_id          = hw_id.wave_id;
        sample.hw_id.simd_id          = hw_id.simd_id;
        sample.hw_id.cu_or_wgp_id     = hw_id.cu_or_wgp_id;
        sample.hw_id.shader_array_id  = hw_id.shader_array_id;
        sample.hw_id.shader_engine_id = hw_id.shader_engine_id;
        sample.hw_id.pipe_id          = hw_id.pipe_id;
        sample.hw_id.workgroup_id     = hw_id.workgroup_id;
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
class WaveOtherFieldsTestGFX12 : public WaveOtherFieldsTest<GFX12, PcSamplingRecordT>
{
public:
    void genPCSample(int pc, int exec, int blkx, int blky, int blkz, int chip, int wave) override
    {
        // chiplet is not used on GFX12, so we set it to 0
        chip = 0;
        WaveOtherFieldsTest<GFX12, PcSamplingRecordT>::genPCSample(
            pc, exec, blkx, blky, blkz, chip, wave);
    }
};

template <typename PcSamplingRecordT>
class MemoryCountersTest : public WaveSnapTest<GFX12, PcSamplingRecordT>
{
    union perf_snapshot_data2
    {
        uint32_t raw;
        struct
        {
            uint32_t load_cnt   : 6;  ///< bits 5:0
            uint32_t store_cnt  : 6;  ///< bits 11:6
            uint32_t bvh_cnt    : 3;  ///< bits 14:12
            uint32_t sample_cnt : 6;  ///< bits 20:15
            uint32_t ds_cnt     : 6;  ///< bits 26:21
            uint32_t km_cnt     : 5;  ///< bits 31:27
        };
    };

    void FillBuffers() override
    {
        this->buffer->genUpcomingSamples(4);
        genPCSample(0, 0, 0, 0, 0, 0);       // All zeros
        genPCSample(1, 2, 3, 4, 5, 6);       // Counting
        genPCSample(3, 5, 7, 11, 13, 17);    // Some prime numbers
        genPCSample(23, 19, 17, 13, 11, 7);  // Some reversed primes
    }

    void CheckBuffers() override
    {
        // Test appliclable only to stochastic sampling records
        bool is_stoch_sampling_record =
            std::is_same<PcSamplingRecordT, rocprofiler_pc_sampling_record_stochastic_v0_t>::value;

        EXPECT_EQ(is_stoch_sampling_record, true);

        auto parsed = this->buffer->get_parsed_buffer(GFXIP_MAJOR);  // GFXIP==12
        EXPECT_EQ(parsed.size(), 1);
        EXPECT_EQ(parsed[0].size(), 4);
        EXPECT_EQ(compare.size(), 4);

        for(size_t i = 0; i < 4; i++)
        {
            EXPECT_EQ(1, parsed[0][i].flags.has_memory_counter);
            EXPECT_EQ(compare[i].memory_counters.load_cnt, parsed[0][i].memory_counters.load_cnt);
            EXPECT_EQ(compare[i].memory_counters.store_cnt, parsed[0][i].memory_counters.store_cnt);
            EXPECT_EQ(compare[i].memory_counters.bvh_cnt, parsed[0][i].memory_counters.bvh_cnt);
            EXPECT_EQ(compare[i].memory_counters.sample_cnt,
                      parsed[0][i].memory_counters.sample_cnt);
            EXPECT_EQ(compare[i].memory_counters.ds_cnt, parsed[0][i].memory_counters.ds_cnt);
            EXPECT_EQ(compare[i].memory_counters.km_cnt, parsed[0][i].memory_counters.km_cnt);
        }
    }

    void genPCSample(int load_cnt,
                     int store_cnt,
                     int bvh_cnt,
                     int sample_cnt,
                     int ds_cnt,
                     int km_cnt)
    {
        PcSamplingRecordT sample;
        ::memset(&sample, 0, sizeof(sample));

        sample.flags.has_memory_counter   = 1;
        sample.memory_counters.load_cnt   = load_cnt;
        sample.memory_counters.store_cnt  = store_cnt;
        sample.memory_counters.bvh_cnt    = bvh_cnt;
        sample.memory_counters.sample_cnt = sample_cnt;
        sample.memory_counters.ds_cnt     = ds_cnt;
        sample.memory_counters.km_cnt     = km_cnt;

        compare.push_back(sample);

        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));

        perf_snapshot_data2 data2{};
        data2.load_cnt   = load_cnt;
        data2.store_cnt  = store_cnt;
        data2.bvh_cnt    = bvh_cnt;
        data2.sample_cnt = sample_cnt;
        data2.ds_cnt     = ds_cnt;
        data2.km_cnt     = km_cnt;

        snap.perf_snapshot_data2 = data2.raw;
        snap.correlation_id      = this->dispatch->getMockId().raw;

        // to ensure all stochastic samples are generated properly,
        // marked them as valid
        snap.perf_snapshot_data |= 0x1;  // set the bit indicating the sample is valid

        EXPECT_NE(this->dispatch.get(), nullptr);
        this->dispatch->submit(snap);
    };

    std::vector<PcSamplingRecordT> compare;
};

TEST(pcs_parser, gfx12_test)
{
    // Tests specific to stochastic sampling only
    WaveCntTest<GFX12, rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    InstTypeTestGFX12<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    StallReasonTestGFX12<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    ArbStateTestGFX12<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    WaveIssueAndErrorTestGFX12<rocprofiler_pc_sampling_record_stochastic_v0_t,
                               rocprofiler_pc_sampling_record_invalid_t>{}
        .Test();

    // Tests common for both host trap and stochastic sampling.
    HwIdTest<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    HwIdTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    WaveOtherFieldsTestGFX12<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();
    WaveOtherFieldsTestGFX12<rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();

    MidMacroPCCorrection<GFX12, rocprofiler_pc_sampling_record_host_trap_v0_t>{}.Test();
    MidMacroPCCorrection<GFX12, rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();

    // test specific to GFX12
    MemoryCountersTest<rocprofiler_pc_sampling_record_stochastic_v0_t>{}.Test();

    std::cout << "GFX12 Test Done." << std::endl;
}
