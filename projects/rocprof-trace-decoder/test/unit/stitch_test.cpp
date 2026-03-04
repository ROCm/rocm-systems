// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "stitch/stitch.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// Mock ICodeServicer for testing
class MockCodeServicer : public ICodeServicer
{
public:
    MOCK_METHOD(assemblyLine, GetInstruction, (pcinfo_t addr, int gfxip), (override));
};

// Test callback to capture Stitcher output
struct TestCallbackData
{
    std::vector<rocprofiler_thread_trace_decoder_record_type_t> record_types;
    std::vector<uint64_t> record_sizes;
    uint64_t gfxip_received = 0;
    int wave_count = 0;
};

rocprofiler_thread_trace_decoder_status_t test_callback(
    rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t size, void* userdata
)
{
    auto* cbdata = static_cast<TestCallbackData*>(userdata);
    cbdata->record_types.push_back(type);
    cbdata->record_sizes.push_back(size);

    if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_GFXIP)
        cbdata->gfxip_received = reinterpret_cast<uint64_t>(data);
    else if (type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
        cbdata->wave_count++;

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

class StitcherTest : public ::testing::Test
{
protected:
    std::shared_ptr<MockCodeServicer> mock_service;
    TestCallbackData cbdata;

    void SetUp() override
    {
        mock_service = std::make_shared<MockCodeServicer>();
        cbdata = TestCallbackData{};
    }
};

TEST_F(StitcherTest, ConstructorInitializesRawCode)
{
    Stitcher stitcher(mock_service, test_callback, &cbdata);

    ASSERT_FALSE(stitcher.raw_code.empty());
    EXPECT_EQ(stitcher.raw_code[0]->line, "; Begin ASM");
}

TEST_F(StitcherTest, SetGfxipSetsValueAndCallsCallback)
{
    Stitcher stitcher(mock_service, test_callback, &cbdata);

    stitcher.setgfxip(9);
    EXPECT_EQ(stitcher.getgfxip(), 9);
    EXPECT_EQ(cbdata.gfxip_received, 9);

    // Second call should not invoke callback again (std::once_flag)
    cbdata.gfxip_received = 0;
    stitcher.setgfxip(10); // Value changes but callback not called
    EXPECT_EQ(cbdata.gfxip_received, 0);
}

TEST_F(StitcherTest, SetGfxipMultipleVersions)
{
    {
        Stitcher stitcher(mock_service, test_callback, &cbdata);
        stitcher.setgfxip(9);
        EXPECT_EQ(stitcher.getgfxip(), 9);
    }

    cbdata = TestCallbackData{};
    {
        Stitcher stitcher(mock_service, test_callback, &cbdata);
        stitcher.setgfxip(10);
        EXPECT_EQ(stitcher.getgfxip(), 10);
        EXPECT_EQ(cbdata.gfxip_received, 10);
    }

    cbdata = TestCallbackData{};
    {
        Stitcher stitcher(mock_service, test_callback, &cbdata);
        stitcher.setgfxip(11);
        EXPECT_EQ(stitcher.getgfxip(), 11);
    }

    cbdata = TestCallbackData{};
    {
        Stitcher stitcher(mock_service, test_callback, &cbdata);
        stitcher.setgfxip(12);
        EXPECT_EQ(stitcher.getgfxip(), 12);
    }
}

TEST_F(StitcherTest, StitchEmptyWave)
{
    Stitcher stitcher(mock_service, test_callback, &cbdata);
    stitcher.setgfxip(10);

    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.bIsComplete = true;

    stitcher.stitch(wave);

    EXPECT_EQ(cbdata.wave_count, 1);
    EXPECT_TRUE(wave.callbackComplete);
}

TEST_F(StitcherTest, StitchDoesNotCallbackTwice)
{
    Stitcher stitcher(mock_service, test_callback, &cbdata);
    stitcher.setgfxip(10);

    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.bIsComplete = true;

    stitcher.stitch(wave);
    EXPECT_EQ(cbdata.wave_count, 1);

    // Second stitch should not callback again
    stitcher.stitch(wave);
    EXPECT_EQ(cbdata.wave_count, 1);
}

TEST_F(StitcherTest, StitchWaveWithInstructions)
{
    Stitcher stitcher(mock_service, test_callback, &cbdata);
    stitcher.setgfxip(10);

    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(1, 2, 3, 100, start_pc, false);
    wave.bIsComplete = true;

    // Add some instructions
    wave.instructions.push_back(Instruction(100, WaveInstCategory::VALU, 4, 0));
    wave.instructions.push_back(Instruction(104, WaveInstCategory::SMEM, 8, 2));
    wave.instructions.push_back(Instruction(112, WaveInstCategory::VMEM, 16, 4));

    stitcher.stitch(wave);

    EXPECT_EQ(cbdata.wave_count, 1);
    EXPECT_EQ(wave.instructions_size, 3);
    EXPECT_EQ(wave.instructions_array, wave.instructions.data());
}

// Tests for insert_excluded_gfx12_barrier_wait
class InsertGfx12BarrierWaitTest : public ::testing::Test
{
protected:
    WaveDataInternal createWaveWithInstructions(int num_instructions)
    {
        pcinfo_t start_pc{0, 0};
        WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
        wave.begin_time = 0;

        for (int i = 0; i < num_instructions; i++)
        {
            Instruction inst;
            inst.time = i * 10;
            inst.duration = 5;
            inst.category = WaveInstCategory::VALU;
            inst.pc = {static_cast<uint64_t>(i * 4), 1};
            wave.instructions.push_back(inst);
        }
        wave.end_time = num_instructions * 10;
        return wave;
    }
};

TEST_F(InsertGfx12BarrierWaitTest, EmptyBarrierListDoesNothing)
{
    auto wave = createWaveWithInstructions(5);
    size_t original_size = wave.instructions.size();

    barrier_list_t barriers;
    insert_gfx12_barrier_wait(wave, barriers);

    EXPECT_EQ(wave.instructions.size(), original_size);
}

TEST_F(InsertGfx12BarrierWaitTest, EmptyWaveDoesNothing)
{
    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);

    barrier_list_t barriers = {
        {2, {100, 1}}
    };
    insert_gfx12_barrier_wait(wave, barriers);

    EXPECT_TRUE(wave.instructions.empty());
}

TEST_F(InsertGfx12BarrierWaitTest, InsertsBarrierInstruction)
{
    auto wave = createWaveWithInstructions(5);

    barrier_list_t barriers = {
        {2, {100, 1}}
    };
    insert_gfx12_barrier_wait(wave, barriers);

    // Should have inserted one barrier instruction
    EXPECT_EQ(wave.instructions.size(), 6);
}

TEST_F(InsertGfx12BarrierWaitTest, MultipleBarriers)
{
    auto wave = createWaveWithInstructions(10);

    barrier_list_t barriers = {
        {2, {100, 1}},
        {5, {200, 1}},
        {8, {300, 1}}
    };
    insert_gfx12_barrier_wait(wave, barriers);

    // Should have inserted three barrier instructions
    EXPECT_EQ(wave.instructions.size(), 13);
}

TEST_F(InsertGfx12BarrierWaitTest, BarrierAtIndex1)
{
    auto wave = createWaveWithInstructions(5);

    barrier_list_t barriers = {
        {1, {100, 1}}
    };
    insert_gfx12_barrier_wait(wave, barriers);

    EXPECT_EQ(wave.instructions.size(), 6);
}

TEST_F(InsertGfx12BarrierWaitTest, SkipsDuplicateIndices)
{
    auto wave = createWaveWithInstructions(5);

    // Second barrier at same index should be skipped
    barrier_list_t barriers = {
        {2, {100, 1}},
        {2, {200, 1}}
    };
    insert_gfx12_barrier_wait(wave, barriers);

    EXPECT_EQ(wave.instructions.size(), 6); // Only one inserted
}

// Test WaveDataInternal construction
TEST(WaveDataInternalTest, ConstructorSetsFields)
{
    pcinfo_t start_pc{100, 5};
    WaveDataInternal wave(1, 2, 3, 1000, start_pc, true);

    EXPECT_EQ(wave.cu, 1);
    EXPECT_EQ(wave.simd, 2);
    EXPECT_EQ(wave.wave_id, 3);
    EXPECT_EQ(wave.begin_time, 1000);
    EXPECT_TRUE(wave.exclude_barrier_wait);
    EXPECT_FALSE(wave.bIsComplete);
    EXPECT_FALSE(wave.callbackComplete);
}

TEST(WaveDataInternalTest, ConstructorWithValidPcAddsPcInfo)
{
    pcinfo_t start_pc{100, 5};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);

    ASSERT_EQ(wave.pc_infos.size(), 1);
    EXPECT_EQ(wave.pc_infos[0].first, 0);
    EXPECT_EQ(wave.pc_infos[0].second.address, 100);
    EXPECT_EQ(wave.pc_infos[0].second.code_object_id, 5);
}

TEST(WaveDataInternalTest, ConstructorWithInvalidPcDoesNotAddPcInfo)
{
    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);

    EXPECT_TRUE(wave.pc_infos.empty());
}

TEST(WaveDataInternalTest, ConstructorWithZeroCodeObjAddsToUnattribPcs)
{
    pcinfo_t start_pc{100, 0}; // Valid address but no code object
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);

    ASSERT_EQ(wave.pc_infos.size(), 1);
    ASSERT_EQ(wave.unattrib_pcs.size(), 1);
    EXPECT_EQ(wave.unattrib_pcs[0], 0);
}

// Test Instruction
TEST(InstructionTest, ConstructorSetsFields)
{
    Instruction inst(1000, WaveInstCategory::SMEM, 10, 5);

    EXPECT_EQ(inst.time, 1000);
    EXPECT_EQ(inst.category, WaveInstCategory::SMEM);
    EXPECT_EQ(inst.duration, 10);
    EXPECT_EQ(inst.stall, 5);
    EXPECT_EQ(inst.pc.code_object_id, 0);
    EXPECT_EQ(inst.pc.address, 0);
}

TEST(InstructionTest, EqualityOperator)
{
    Instruction a(100, WaveInstCategory::VALU, 4, 0);
    Instruction b(100, WaveInstCategory::VALU, 4, 0);
    Instruction c(100, WaveInstCategory::SMEM, 4, 0);

    a.pc = {10, 1};
    b.pc = {10, 1};
    c.pc = {10, 1};

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
    EXPECT_TRUE(a != c);
}

TEST(InstructionTest, DifferentPcNotEqual)
{
    Instruction a(100, WaveInstCategory::VALU, 4, 0);
    Instruction b(100, WaveInstCategory::VALU, 4, 0);

    a.pc = {10, 1};
    b.pc = {20, 1};

    EXPECT_FALSE(a == b);
}

// Test callback returning error status
rocprofiler_thread_trace_decoder_status_t error_callback(
    rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t size, void* userdata
)
{
    (void) type;
    (void) data;
    (void) size;
    (void) userdata;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR;
}

TEST(StitcherEdgeCaseTest, CallbackReturningErrorDoesNotCrash)
{
    auto mock_service = std::make_shared<MockCodeServicer>();
    TestCallbackData cbdata{};

    Stitcher stitcher(mock_service, error_callback, &cbdata);
    stitcher.setgfxip(10);

    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.bIsComplete = true;

    // Should not crash even if callback returns error
    stitcher.stitch(wave);
    EXPECT_TRUE(wave.callbackComplete);
}

// Test wave completion flag behavior
TEST(StitcherEdgeCaseTest, WaveCompletionBehavior)
{
    auto mock_service = std::make_shared<MockCodeServicer>();
    TestCallbackData cbdata{};

    Stitcher stitcher(mock_service, test_callback, &cbdata);
    stitcher.setgfxip(10);

    pcinfo_t start_pc{100, 1};
    WaveDataInternal wave(1, 2, 3, 100, start_pc, false);
    wave.bIsComplete = false; // Initially incomplete

    // Stitch may still process wave - implementation dependent
    stitcher.stitch(wave);

    // Just verify no crash occurs
    // Actual behavior depends on implementation
}

// Test insert_gfx12_barrier_wait with valid indices only
TEST(StitcherEdgeCaseTest, BarrierAtLastValidIndex)
{
    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.begin_time = 0;

    // Add 5 instructions
    for (int i = 0; i < 5; i++)
    {
        Instruction inst;
        inst.time = i * 10;
        inst.duration = 5;
        inst.category = WaveInstCategory::VALU;
        inst.pc = {static_cast<uint64_t>(i * 4), 1};
        wave.instructions.push_back(inst);
    }
    wave.end_time = 50;

    // Barrier at last valid index
    barrier_list_t barriers = {
        {4, {100, 1}}
    }; // Index 4, size is 5

    insert_gfx12_barrier_wait(wave, barriers);

    // Should insert one barrier
    EXPECT_EQ(wave.instructions.size(), 6);
}

// Test barrier at index 0
TEST(StitcherEdgeCaseTest, BarrierAtIndexZero)
{
    pcinfo_t start_pc{0, 0};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.begin_time = 0;

    for (int i = 0; i < 5; i++)
    {
        Instruction inst;
        inst.time = i * 10;
        inst.duration = 5;
        inst.category = WaveInstCategory::VALU;
        inst.pc = {static_cast<uint64_t>(i * 4), 1};
        wave.instructions.push_back(inst);
    }
    wave.end_time = 50;

    barrier_list_t barriers = {
        {0, {100, 1}}
    }; // Index 0

    // Should handle index 0 without issue
    insert_gfx12_barrier_wait(wave, barriers);

    // Behavior at index 0 may vary - just ensure no crash
    EXPECT_GE(wave.instructions.size(), 5);
}

// Helper callback for standalone stitch tests
namespace
{
rocprofiler_thread_trace_decoder_status_t
noop_stitch_cb(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}
} // namespace

// Tests that is_elaborate_match is triggered during barrier_wait exclusion lookahead.
// The elaborate match extends trivial match with:
//   - JUMP wave + BRANCH line
//   - IMMED wave + skippable line (VALU/VMEM/FLAT/IMMED/LDS)
TEST(StitcherElaborateMatchTest, BarrierExclusionLookaheadUsesElaborateMatch)
{
    auto mock = std::make_shared<MockCodeServicer>();

    // Line 1: VALU — primes the stitcher (matched by inst[0])
    assemblyLine line1{};
    line1.line = "v_add_f32 v0, v1, v2";
    line1.cat = InstCategory::VALU;
    line1.addr = {0x100, 1};
    line1.next = {0x104, 1};

    // Line 2: s_barrier_wait — triggers barrier exclusion path
    assemblyLine line2{};
    line2.line = "s_barrier_wait 0x1";
    line2.cat = InstCategory::IMMED;
    line2.parsed = false;
    line2.addr = {0x104, 1};
    line2.next = {0x108, 1};

    // Line 3: VALU — lookahead reads this line's cat, then advances to line4.
    //   is_elaborate_match(IMMED, VALU): trivial_match false, then skippable(VALU) → true
    assemblyLine line3{};
    line3.line = "v_mov_b32 v0, 0";
    line3.cat = InstCategory::VALU;
    line3.addr = {0x108, 1};
    line3.next = {0x10C, 1};

    // Line 4: needed so getcode(line3.next) succeeds before is_elaborate_match is called
    assemblyLine line4{};
    line4.line = "v_mov_b32 v1, 1";
    line4.cat = InstCategory::VALU;
    line4.addr = {0x10C, 1};
    line4.next = {0, 0};

    ON_CALL(*mock, GetInstruction(testing::_, testing::_))
        .WillByDefault(
            [&](pcinfo_t addr, int) -> assemblyLine
            {
                if (addr.address == 0x100) return line1;
                if (addr.address == 0x104) return line2;
                if (addr.address == 0x108) return line3;
                if (addr.address == 0x10C) return line4;
                throw std::runtime_error("unknown address");
            }
        );

    Stitcher stitcher(mock, noop_stitch_cb, nullptr);
    stitcher.setgfxip(12);

    pcinfo_t start_pc{0x100, 1};
    // exclude_barrier_wait = true (last arg)
    WaveDataInternal wave(0, 0, 0, 0, start_pc, true);
    wave.bIsComplete = true;

    // inst[0]: VALU with valid PC → primes line/next in stitchWave
    // inst[1]: IMMED without PC → enters barrier exclusion, lookahead calls is_elaborate_match
    //   lookahead: line2cat=VALU (line3), advances to line4, then is_elaborate_match(IMMED, VALU)
    // inst[2]: VALU → continues stitching after barrier handled
    wave.instructions.push_back(Instruction(0, WaveInstCategory::VALU, 4, 0));
    wave.instructions.push_back(Instruction(4, WaveInstCategory::IMMED, 4, 0));
    wave.instructions.push_back(Instruction(8, WaveInstCategory::VALU, 4, 0));

    size_t orig_size = wave.instructions.size();
    stitcher.stitch(wave);

    // If the elaborate match succeeded, the barrier_wait was excluded and
    // insert_gfx12_barrier_wait was called (gfxip=12), inserting a barrier instruction.
    EXPECT_GT(wave.instructions.size(), orig_size);
}

// Tests for PCTranslator::try_match_swapped
TEST(PCTranslatorTest, TryMatchSwapped)
{
    auto mock = std::make_shared<MockCodeServicer>();

    // Set up two assembly lines: line A (VALU) -> line B (SMEM)
    assemblyLine lineA{};
    lineA.line = "v_add_f32 v0, v1, v2";
    lineA.cat = InstCategory::VALU;
    lineA.addr = {0x100, 1};
    lineA.next = {0x104, 1};

    assemblyLine lineB{};
    lineB.line = "s_load_b64 s[0:1], s[2:3]";
    lineB.cat = InstCategory::SMEM;
    lineB.addr = {0x104, 1};
    lineB.next = {0x108, 1};

    // Mock returns lineB when asked for lineA.next
    ON_CALL(*mock, GetInstruction(testing::_, testing::_)).WillByDefault(testing::Return(lineB));

    std::vector<assemblyLinePtr> code;
    code.push_back(std::make_shared<assemblyLine>(lineA));
    code.push_back(std::make_shared<assemblyLine>(lineB));

    std::shared_ptr<ICodeServicer> svc = mock;
    PCTranslator pct(code, svc, 9);

    // first=SMEM, second=VALU, line=VALU => second matches line, first matches next(SMEM)
    Instruction first(0, WaveInstCategory::SMEM, 4, 0);
    Instruction second(4, WaveInstCategory::VALU, 4, 0);
    EXPECT_TRUE(pct.try_match_swapped(first, second, lineA));

    // Reverse should not match
    EXPECT_FALSE(pct.try_match_swapped(second, first, lineA));
}

// Tests for s_waitcnt/_load_ stitch break path
TEST(StitcherWaitcntTest, SWaitcntBreaksStitch)
{
    auto mock = std::make_shared<MockCodeServicer>();

    // Line 1: VALU (matches first inst)
    assemblyLine line1{};
    line1.line = "v_add_f32 v0, v1, v2";
    line1.cat = InstCategory::VALU;
    line1.addr = {0x100, 1};
    line1.next = {0x104, 1};

    // Line 2: s_waitcnt (should trigger the s_waitcnt break path)
    assemblyLine line2{};
    line2.line = "s_waitcnt vmcnt(0)";
    line2.cat = InstCategory::IMMED;
    line2.addr = {0x104, 1};
    line2.next = {0x108, 1};

    // Line 3: another instruction
    assemblyLine line3{};
    line3.line = "v_mov_b32 v0, 0";
    line3.cat = InstCategory::VALU;
    line3.addr = {0x108, 1};
    line3.next = {0, 0};

    ON_CALL(*mock, GetInstruction(testing::_, testing::_))
        .WillByDefault(
            [&](pcinfo_t addr, int) -> assemblyLine
            {
                if (addr.address == 0x100) return line1;
                if (addr.address == 0x104) return line2;
                if (addr.address == 0x108) return line3;
                return assemblyLine{};
            }
        );

    Stitcher stitcher(mock, noop_stitch_cb, nullptr);
    stitcher.setgfxip(10); // gfxip != 9 for IMMED/skippable path

    pcinfo_t start_pc{0x100, 1};
    WaveDataInternal wave(0, 0, 0, 0, start_pc, false);
    wave.bIsComplete = true;

    // Add instructions: VALU (matches line1), then SMEM (won't match s_waitcnt)
    wave.instructions.push_back(Instruction(0, WaveInstCategory::VALU, 4, 0));
    wave.instructions.push_back(Instruction(4, WaveInstCategory::SMEM, 4, 0));

    stitcher.stitch(wave);

    // The first instruction should get a valid PC, but the s_waitcnt break
    // should prevent full stitching
    EXPECT_NE(wave.instructions[0].pc.address, 0u);
}
