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

#include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/tests/mocks.hpp"

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
            assert(compare[i].workgroup_id_x == parsed[0][i].workgroup_id_x);
            assert(compare[i].workgroup_id_y == parsed[0][i].workgroup_id_y);
            assert(compare[i].workgroup_id_z == parsed[0][i].workgroup_id_z);

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
        sample.workgroup_id_x = blkx;
        sample.workgroup_id_y = blky;
        sample.workgroup_id_z = blkz;

        sample.chiplet        = chip;
        sample.wave_id        = wave;
        sample.hw_id          = hwid;
        sample.correlation_id = dispatch->unique_id;

        compare.push_back(sample);

        perf_sample_snapshot_v1 snap;
        ::memset(&snap, 0, sizeof(snap));
        snap.exec_mask = exec;

        snap.workgroup_id_x      = blkx;
        snap.workgroup_id_y      = blky;
        snap.workgroup_id_z      = blkz;
        snap.chiplet_and_wave_id = (chip << 8) | (wave & 0x3F);
        snap.hw_id               = hwid;
        snap.correlation_id      = dispatch->getMockId();

        assert(dispatch.get());
        dispatch->submit(snap);

        (void) pc;
    };

    std::vector<pcsample_v1_t> compare;
};

TEST(pcs_parser, gfx9_test)
{
    WaveCntTest{}.Test();
    InstTypeTest{}.Test();
    StallReasonTest{}.Test();
    ArbStateTest{}.Test();
    WaveIssueAndErrorTest{}.Test();
    WaveOtherFieldsTest{}.Test();

    std::cout << "GFX9 Test Done." << std::endl;
}
