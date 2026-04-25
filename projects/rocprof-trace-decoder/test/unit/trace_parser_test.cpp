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

#include "trace_parser.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "stitch/stitch.hpp"
#include "trace_decoder_api.h"

// Forward declarations for internal functions
std::unique_ptr<SQTTParser> AnalyseBinary_GFX9_internal(
    CppReturnInfo& info,
    const uint8_t* tokendata,
    uint64_t buffersize,
    int target_cu,
    class Stitcher& stitch,
    bool double_buffer
);

// Note: ToPcV2 is defined in segment_test.cpp

// Tests for CppReturnInfo
TEST(CppReturnInfoTest, DefaultConstruction)
{
    CppReturnInfo info;

    EXPECT_TRUE(info.perfevents.empty());
    EXPECT_EQ(info.realtime_frequency, 0);
    EXPECT_EQ(info.counter_frequency, 0);
    EXPECT_FALSE(info.bPacketLost);
    EXPECT_EQ(info.globaltime, 0);
    EXPECT_EQ(info.basetime, 0);
}

// Tests for occupancy_info_t
// Note: 'start' is a 1-bit field: 1 = wave_start, 0 = wave_end
TEST(OccupancyInfoTest, ConstructorWithUint64WaveStart)
{
    pcinfo_t pc{100, 5};
    occupancy_info_t occ(pc, 1000, uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(1));

    EXPECT_EQ(occ.pc.address, 100);
    EXPECT_EQ(occ.pc.code_object_id, 5);
    EXPECT_EQ(occ.time, 1000);
    EXPECT_EQ(occ.cu, 1);
    EXPECT_EQ(occ.simd, 2);
    EXPECT_EQ(occ.wave_id, 3);
    EXPECT_EQ(occ.start, 1); // wave_start
}

TEST(OccupancyInfoTest, ConstructorWithUint64WaveEnd)
{
    pcinfo_t pc{100, 5};
    occupancy_info_t occ(pc, 1000, uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(0));

    EXPECT_EQ(occ.start, 0); // wave_end
}

TEST(OccupancyInfoTest, ConstructorWithInt8)
{
    pcinfo_t pc{200, 10};
    occupancy_info_t occ(pc, 2000, int8_t(4), int8_t(1), int8_t(7), uint64_t(1));

    EXPECT_EQ(occ.pc.address, 200);
    EXPECT_EQ(occ.pc.code_object_id, 10);
    EXPECT_EQ(occ.time, 2000);
    EXPECT_EQ(occ.cu, 4);
    EXPECT_EQ(occ.simd, 1);
    EXPECT_EQ(occ.wave_id, 7);
    EXPECT_EQ(occ.start, 1); // wave_start
}

// Note: bValid tests are in stitch_utils_test.cpp

// Tests for PipeArray
TEST(PipeArrayTest, AtRegAccessesCorrectElement)
{
    PipeArray<int> arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };

    MockToken t1{0, 0};
    MockToken t2{0, 1};
    MockToken t3{1, 2};
    MockToken t4{1, 3};

    arr.at_reg(t1) = 10;
    arr.at_reg(t2) = 20;
    arr.at_reg(t3) = 30;
    arr.at_reg(t4) = 40;

    EXPECT_EQ(arr[0][0], 10);
    EXPECT_EQ(arr[0][1], 20);
    EXPECT_EQ(arr[1][2], 30);
    EXPECT_EQ(arr[1][3], 40);
}

TEST(PipeArrayTest, MeModulo2)
{
    PipeArray<int> arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };

    // me=2 should map to me&1 = 0
    MockToken t1{2, 0};
    // me=3 should map to me&1 = 1
    MockToken t2{3, 1};

    arr.at_reg(t1) = 100;
    arr.at_reg(t2) = 200;

    EXPECT_EQ(arr[0][0], 100);
    EXPECT_EQ(arr[1][1], 200);
}

// Tests for PipeArray64
TEST(PipeArray64Test, SetLoSetsLowBits)
{
    PipeArray64 arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };
    MockToken t{0, 0};

    arr.setlo(t, 0xDEADBEEF);

    EXPECT_EQ(arr.at_reg(t) & 0xFFFFFFFF, 0xDEADBEEF);
}

TEST(PipeArray64Test, SetHiSetsHighBits)
{
    PipeArray64 arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };
    MockToken t{0, 0};

    arr.sethi(t, 0xCAFEBABE);

    EXPECT_EQ(arr.at_reg(t) >> 32, 0xCAFEBABE);
}

TEST(PipeArray64Test, SetLoPreservesHighBits)
{
    PipeArray64 arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };
    MockToken t{0, 0};

    arr.sethi(t, 0xAAAAAAAA);
    arr.setlo(t, 0xBBBBBBBB);

    EXPECT_EQ(arr.at_reg(t), 0xAAAAAAAABBBBBBBBull);
}

TEST(PipeArray64Test, SetHiPreservesLowBits)
{
    PipeArray64 arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };
    MockToken t{0, 0};

    arr.setlo(t, 0xCCCCCCCC);
    arr.sethi(t, 0xDDDDDDDD);

    EXPECT_EQ(arr.at_reg(t), 0xDDDDDDDDCCCCCCCCull);
}

// Tests for TokenGenerator base class behavior
class TestTokenGenerator : public TokenGenerator
{
public:
    TestTokenGenerator(const uint8_t* buf, size_t size, int64_t gt, int64_t bt) : TokenGenerator(buf, size, gt, bt) {}
};

TEST(TokenGeneratorTest, ConstructorSetsFields)
{
    uint8_t buffer[100];
    TestTokenGenerator gen(buffer, 100, 5000, 1000);

    EXPECT_EQ(gen.BUFFER_SIZE, 100);
    EXPECT_EQ(gen.get_time(), 5000);
    EXPECT_EQ(gen.get_base_time(), 1000);
}

// Edge case: minimum buffer size
TEST(TokenGeneratorEdgeCaseTest, MinimumBufferSize)
{
    uint8_t buffer[16]; // Minimum practical size
    TestTokenGenerator gen(buffer, 16, 0, 0);

    EXPECT_EQ(gen.BUFFER_SIZE, 16);
}

// Edge case: large time values
TEST(TokenGeneratorEdgeCaseTest, LargeTimeValues)
{
    uint8_t buffer[100];
    int64_t large_time = 0x7FFFFFFFFFFFF; // Large but not max
    TestTokenGenerator gen(buffer, 100, large_time, large_time - 1000);

    EXPECT_EQ(gen.get_time(), large_time);
    EXPECT_EQ(gen.get_base_time(), large_time - 1000);
}

// Edge case: PipeArray with all pipe values
TEST(PipeArrayEdgeCaseTest, AllPipeValues)
{
    PipeArray<int> arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };

    // Test all combinations
    for (int me = 0; me < 2; me++)
    {
        for (int pipe = 0; pipe < 4; pipe++)
        {
            MockToken t{me, pipe};
            arr.at_reg(t) = me * 100 + pipe;
        }
    }

    // Verify all values
    for (int me = 0; me < 2; me++)
    {
        for (int pipe = 0; pipe < 4; pipe++) { EXPECT_EQ(arr[me][pipe], me * 100 + pipe); }
    }
}

// Edge case: PipeArray64 combines hi and lo correctly
TEST(PipeArray64EdgeCaseTest, HiLoComposition)
{
    PipeArray64 arr{};

    struct MockToken
    {
        int me;
        int pipe;
    };
    MockToken t{0, 0};

    arr.sethi(t, 0x12345678);
    arr.setlo(t, 0xABCDEF00);

    EXPECT_EQ(arr.at_reg(t), 0x12345678ABCDEF00ull);
}

// Tests for CSRegisterHandler
struct FakeRegToken
{
    int me = 0;
    int pipe = 0;
    uint32_t regaddr = 0;
    uint64_t regdata = 0;
};

TEST(CSRegisterHandlerTest, AddressClassification)
{
    CSRegisterHandler h;
    EXPECT_TRUE(h.IsPgmLo(0xC));
    EXPECT_TRUE(h.IsPgmHi(0xD));
    EXPECT_FALSE(h.IsPgmLo(0xD));
    EXPECT_TRUE(h.IsUserdata(0xC340));
    EXPECT_TRUE(h.IsUserdata(0xC343));
    EXPECT_FALSE(h.IsUserdata(0xC344));
    EXPECT_TRUE(h.IsUserdata0(0xC340));
    EXPECT_TRUE(h.IsUserdata1(0xC341));
    EXPECT_TRUE(h.IsUserdata2(0xC342));
    EXPECT_TRUE(h.IsUserdata3(0xC343));
}

TEST(CSRegisterHandlerTest, UpdateRegNoCSNonUserdata2Ignored)
{
    CSRegisterHandler h;
    FakeRegToken tok{0, 0, 0xC340, 0};
    EXPECT_FALSE(h.UpdateRegNoCS(tok));
}

TEST(CSRegisterHandlerTest, ROCHeaderAndCodeObjLoadUnload)
{
    CSRegisterHandler h;
    FakeRegToken tok{0, 0, 0xC342, 0};

    // Activate ROC format
    rocprof_trace_decoder_instrument_enable_t en{};
    en.char1 = '\0';
    en.char2 = 'R';
    en.char3 = 'O';
    en.char4 = 'C';
    tok.regdata = en.u32All;
    h.UpdateRegNoCS(tok);
    ASSERT_TRUE(h.bIsROCMFormat);

    // CODEOBJ ADDR_LO header + payload
    rocprof_trace_decoder_packet_header_t pkt{};
    pkt.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
    pkt.type = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO;
    tok.regdata = pkt.u32All;
    h.UpdateRegNoCS(tok);
    tok.regdata = 0x1000;
    h.UpdateRegNoCS(tok);

    // ADDR_HI
    pkt.type = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI;
    tok.regdata = pkt.u32All;
    h.UpdateRegNoCS(tok);
    tok.regdata = 0;
    h.UpdateRegNoCS(tok);

    // SIZE_LO
    pkt.type = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO;
    tok.regdata = pkt.u32All;
    h.UpdateRegNoCS(tok);
    tok.regdata = 0x100;
    h.UpdateRegNoCS(tok);

    // TAIL (load)
    pkt.type = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL;
    tok.regdata = pkt.u32All;
    h.UpdateRegNoCS(tok);
    rocprof_trace_decoder_codeobj_marker_tail_t tail{};
    tail.isUnload = 0;
    tail.bFromStart = 1;
    tail.legacy_id = 42;
    tok.regdata = tail.raw;
    h.UpdateRegNoCS(tok);
    EXPECT_FALSE(h.active_codeobj_id.empty());

    // TAIL (unload)
    tok.regdata = pkt.u32All;
    h.UpdateRegNoCS(tok);
    tail.isUnload = 1;
    tok.regdata = tail.raw;
    h.UpdateRegNoCS(tok);
    EXPECT_TRUE(h.active_codeobj_id.empty());
}

TEST(CSRegisterHandlerTest, AgentInfoCounterAndRtFrequency)
{
    CSRegisterHandler h;
    h.bIsROCMFormat = true;
    h.userdata_state = ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_LAST;
    FakeRegToken tok{0, 0, 0xC342, 0};

    rocprof_trace_decoder_packet_header_t pkt{};
    pkt.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_AGENT_INFO;

    // Counter interval
    pkt.type = ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_COUNTER_INTERVAL;
    pkt.data20 = 256;
    tok.regdata = pkt.u32All;
    EXPECT_TRUE(h.UpdateRegNoCS(tok));
    EXPECT_EQ(h.counter_frequency, 256u);

    // RT frequency
    pkt.type = ROCPROF_TRACE_DECODER_AGENT_INFO_TYPE_RT_FREQUENCY_KHZ;
    pkt.data20 = 100;
    tok.regdata = pkt.u32All;
    EXPECT_FALSE(h.UpdateRegNoCS(tok));
    EXPECT_EQ(h.realtime_frequency, 100u * 1000);
}

// Tests for AnalyseBinary dispatch
namespace
{
rocprofiler_thread_trace_decoder_status_t
noop_cb(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

class AnalyseBinaryMock : public ICodeServicer
{
public:
    MOCK_METHOD(assemblyLine, GetInstruction, (pcinfo_t addr, int gfxip), (override));
};
} // namespace

TEST(AnalyseBinaryTest, InvalidBufferReturnsNull)
{
    auto mock = std::make_shared<AnalyseBinaryMock>();
    Stitcher stitch(mock, noop_cb, nullptr);
    uint8_t buf[8] = {0};
    CppReturnInfo info;
    EXPECT_EQ(AnalyseBinary_internal(info, buf, sizeof(buf), -1, stitch), nullptr);
}

TEST(AnalyseBinaryTest, Gfx9PositiveTargetCu)
{
    auto mock = std::make_shared<AnalyseBinaryMock>();
    Stitcher stitch(mock, noop_cb, nullptr);
    uint8_t buf[64] = {0};
    CppReturnInfo info;
    EXPECT_NE(AnalyseBinary_GFX9_internal(info, buf, sizeof(buf), 0, stitch, false), nullptr);
}
