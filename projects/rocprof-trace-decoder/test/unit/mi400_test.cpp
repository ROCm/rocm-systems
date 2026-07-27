/* Copyright (c) 2025 Advanced Micro Devices, Inc. */

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "gfx10/gfx10wave.h"
#include "mi400/mi400parser.h"
#include "mi400/mi400token.h"
#include "mi400/mi400wave.h"

//=============================================================================
// MI400 Token Type Tests
//=============================================================================

TEST(MI400TokenTypeTest, WendTypeMi400GetMethod)
{
    mi400::wend_type wend{};
    wend.tm = 3;
    wend.sa = 1;
    wend.simd = 2;
    wend.wgp = 5;
    wend.wid = 20;

    auto common = wend.get();
    EXPECT_EQ(common.tm, wend.tm);
    EXPECT_EQ(common.sa, wend.sa);
    EXPECT_EQ(common.simd, wend.simd);
    EXPECT_EQ(common.wgp, wend.wgp);
    EXPECT_EQ(common.wid, wend.wid);
}

TEST(MI400TokenTypeTest, DiagnosticPrintsReportDecodedFields)
{
    mi400::immed_one_type immed{};
    immed.header = 0b1101;
    immed.wid = 17;

    mi400::misc_type misc{};
    misc.header = 0b1010001;
    misc.CLF = 1;
    misc.CLID = 9;

    mi400::header_type header{};
    header.header = 0b0010001;
    header.version = 5;
    header.DWGP = 3;
    header.DSIMD = 2;
    header.DSA = 1;
    header.UCF = 1;
    header.DPRate = 4;
    header.WSM = 2;

    EXPECT_STREQ(immed.typestr(), "IMMEDONE");
    EXPECT_NE(immed.print().str().find("wid:17"), std::string::npos);

    EXPECT_STREQ(misc.typestr(), "MISC");
    EXPECT_NE(misc.print().str().find("raw:0x"), std::string::npos);

    EXPECT_STREQ(header.typestr(), "HEADER");
    const std::string header_text = header.print().str();
    EXPECT_NE(header_text.find("TT Version:5"), std::string::npos);
    EXPECT_NE(header_text.find("DWGP:3"), std::string::npos);
    EXPECT_NE(header_text.find("DSIMD:2"), std::string::npos);
    EXPECT_NE(header_text.find("DSA:1"), std::string::npos);
    EXPECT_NE(header_text.find("UCF:1"), std::string::npos);
    EXPECT_NE(header_text.find("DPRate:4"), std::string::npos);
    EXPECT_NE(header_text.find("WSM:2"), std::string::npos);
}

//=============================================================================
// MI400 Token Lookup Table Tests
//=============================================================================

TEST(MI400LookupTableTest, BasicEncodingLookups)
{
    mi400::TokenLookupTable lookup;

    EXPECT_EQ(lookup.lookup(0b0010).type, RdnaType::INST);
    EXPECT_EQ(lookup.lookup(0b011).type, RdnaType::VALU_INST);
    EXPECT_EQ(lookup.lookup(0b1000001).type, RdnaType::WAVE_END);
}

TEST(MI400LookupTableTest, GetTimeForTimestamp)
{
    mi400::TokenLookupTable lookup;

    gfx12::timestamp_type ts{};
    ts.pl = 0;
    ts.rt = 0;
    ts.time = 1000;

    bool packetlost = false;
    int64_t realtime = 0;
    int64_t cur_time = 500;

    auto result = lookup.getTime(lookup.lookup(0b0000001), ts.raw, cur_time, packetlost, realtime);
    EXPECT_EQ(result, ts.time + cur_time);
}

TEST(MI400LookupTableTest, GetTimeForRealtimeTimestamp)
{
    mi400::TokenLookupTable lookup;

    gfx12::timestamp_type ts{};
    ts.pl = 0;
    ts.rt = 1;
    ts.time = 12345;

    bool packetlost = false;
    int64_t realtime = 0;
    int64_t cur_time = 500;

    auto result = lookup.getTime(lookup.lookup(0b0000001), ts.raw, cur_time, packetlost, realtime);
    EXPECT_EQ(result, cur_time);
    EXPECT_EQ(realtime, ts.time);
}

TEST(MI400LookupTableTest, GetTimeWithPacketLoss)
{
    mi400::TokenLookupTable lookup;

    gfx12::timestamp_type ts{};
    ts.pl = 1;
    ts.rt = 0;
    ts.time = 1000;

    bool packetlost = false;
    int64_t realtime = 0;
    int64_t cur_time = 500;

    lookup.getTime(lookup.lookup(0b0000001), ts.raw, cur_time, packetlost, realtime);
    EXPECT_TRUE(packetlost);
}

TEST(MI400LookupTableTest, GetTimeForTimeToken)
{
    mi400::TokenLookupTable lookup;

    bool packetlost = false;
    int64_t realtime = 0;
    int64_t cur_time = 500;

    auto result = lookup.getTime(lookup.lookup(0b1110), 0, cur_time, packetlost, realtime);
    EXPECT_GT(result, cur_time);
}

//=============================================================================
// MI400 Wave Tests
//=============================================================================

TEST(MI400WaveTest, MapToCommonTypeSalu)
{
    auto mapped = mi400::map_to_common_type(0, 0); // salu
    EXPECT_EQ(mapped.category, WaveInstCategory::SALU);
    EXPECT_EQ(mapped.cycles, 1);
}

TEST(MI400WaveTest, MapToCommonTypeTrap)
{
    auto mapped = mi400::map_to_common_type(6, 0); // trap
    EXPECT_EQ(mapped.category, WaveInstCategory::TRAP);
    EXPECT_EQ(mapped.cycles, 1);
}

TEST(MI400WaveTest, MapToCommonTypeValutWithTrans2)
{
    // valut (11) returns 4 normally, but 2 with trans2=1
    auto mapped1 = mi400::map_to_common_type(11, 0);
    EXPECT_EQ(mapped1.category, WaveInstCategory::VALU);
    EXPECT_EQ(mapped1.cycles, 4);

    auto mapped2 = mi400::map_to_common_type(11, 1);
    EXPECT_EQ(mapped2.category, WaveInstCategory::VALU);
    EXPECT_EQ(mapped2.cycles, 2);
}

TEST(MI400WaveTest, MapToCommonTypeOtherSimdLds)
{
    // einst 80-84 → LDS_OTHER_SIMD
    for (int einst = 80; einst <= 84; einst++)
    {
        auto mapped = mi400::map_to_common_type(einst, 0);
        EXPECT_EQ(mapped.category, WaveInstCategory::LDS_OTHER_SIMD) << "Failed for einst=" << einst;
        EXPECT_GT(mapped.cycles, 0) << "Failed for einst=" << einst;
    }
}

TEST(MI400WaveTest, MapToCommonTypeOtherSimdFlat)
{
    // einst 85-89 → FLAT_OTHER_SIMD
    for (int einst = 85; einst <= 89; einst++)
    {
        auto mapped = mi400::map_to_common_type(einst, 0);
        EXPECT_EQ(mapped.category, WaveInstCategory::FLAT_OTHER_SIMD) << "Failed for einst=" << einst;
        EXPECT_GT(mapped.cycles, 0) << "Failed for einst=" << einst;
    }
}

TEST(MI400WaveTest, MapToCommonTypeOtherSimdRemainder)
{
    // einst 90-102 → NONE (no range matches)
    for (int einst = 90; einst <= 102; einst++)
    {
        auto mapped = mi400::map_to_common_type(einst, 0);
        EXPECT_EQ(mapped.category, WaveInstCategory::NONE) << "Failed for einst=" << einst;
        EXPECT_EQ(mapped.cycles, 0) << "Failed for einst=" << einst;
    }
}

TEST(MI400WaveTest, MapToCommonTypeVmemOtherSimd)
{
    // vmem_other_simd range (188 to 221) returns VMEM_OTHER_SIMD
    for (int einst = 188; einst < 222; einst++)
    {
        auto mapped = mi400::map_to_common_type(einst, 0);
        EXPECT_EQ(mapped.category, WaveInstCategory::VMEM_OTHER_SIMD) << "Failed for einst=" << einst;
        EXPECT_GT(mapped.cycles, 0) << "Failed for einst=" << einst;
    }
}

TEST(MI400WaveTest, MapToCommonTypeBlockStore)
{
    auto mapped = mi400::map_to_common_type(222, 0);
    EXPECT_EQ(mapped.category, WaveInstCategory::VMEM);
    EXPECT_EQ(mapped.cycles, 1);

    auto mapped2 = mi400::map_to_common_type(225, 0);
    EXPECT_EQ(mapped2.category, WaveInstCategory::VMEM);
    EXPECT_EQ(mapped2.cycles, 4);
}

TEST(MI400WaveTest, MapToCommonTypeWmma)
{
    auto xdl4 = mi400::map_to_common_type(184, 0);
    EXPECT_EQ(xdl4.category, WaveInstCategory::VALU);
    EXPECT_EQ(xdl4.cycles, 4);

    auto dp32 = mi400::map_to_common_type(187, 0);
    EXPECT_EQ(dp32.category, WaveInstCategory::VALU);
    EXPECT_EQ(dp32.cycles, 32);
}

//=============================================================================
// MI400 Token Generator Tests
//=============================================================================

class TestMI400TokenGenerator : public mi400::TokenGenerator
{
public:
    using mi400::TokenGenerator::current;
    using mi400::TokenGenerator::FIFO;
    using mi400::TokenGenerator::TokenGenerator;
};

static std::array<uint8_t, 16> make_token_buffer(uint64_t raw, size_t num_bytes)
{
    std::array<uint8_t, 16> buffer{};
    for (size_t i = 0; i < num_bytes && i < buffer.size(); i++)
        buffer[i] = static_cast<uint8_t>((raw >> (8 * i)) & 0xFF);
    return buffer;
}

TEST(MI400TokenGeneratorTest, ConstructorThrowsOnNullBuffer)
{
    EXPECT_THROW(mi400::TokenGenerator(nullptr, 10, 0, 0), std::exception);
}

TEST(MI400TokenGeneratorTest, ConstructorThrowsOnZeroSize)
{
    uint8_t dummy[10] = {0};
    EXPECT_THROW(mi400::TokenGenerator(dummy, 0, 0, 0), std::exception);
}

TEST(MI400TokenGeneratorTest, UpdateFifoAndGetValuInst)
{
    uint8_t buffer[16] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    TestMI400TokenGenerator gen(buffer, sizeof(buffer), 0, 0);

    gen.update_fifo(5);
    EXPECT_EQ(gen.FIFO[0], 5);

    gen.update_fifo(10);
    EXPECT_EQ(gen.FIFO[0], 10);
    EXPECT_EQ(gen.FIFO[1], 5);

    gen.update_fifo(15);
    EXPECT_EQ(gen.FIFO[0], 15);
    EXPECT_EQ(gen.FIFO[1], 10);
    EXPECT_EQ(gen.FIFO[2], 5);

    // Set current to a valu_inst with wavetm=0 (maps to FIFO[0]=15)
    mi400::valu_inst_type valu{};
    valu.wavetm = 0;
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 15);

    // wavetm=4 maps to FIFO[1]=10
    valu.wavetm = 4;
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 10);

    // wavetm=8 maps to FIFO[2]=5
    valu.wavetm = 8;
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 5);
}

TEST(MI400TokenGeneratorTest, FifoDuplicateMovesFront)
{
    uint8_t buffer[16] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    TestMI400TokenGenerator gen(buffer, sizeof(buffer), 0, 0);

    gen.update_fifo(5);
    gen.update_fifo(10);
    gen.update_fifo(15);

    gen.update_fifo(5);
    EXPECT_EQ(gen.FIFO[0], 5);
    EXPECT_EQ(gen.FIFO[1], 15);
    EXPECT_EQ(gen.FIFO[2], 10);

    mi400::valu_inst_type valu{};
    valu.wavetm = 0;
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 5);
}

TEST(MI400TokenGeneratorTest, GetValuInstUsesHighWaveTmBuckets)
{
    uint8_t buffer[16] = {
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    TestMI400TokenGenerator gen(buffer, sizeof(buffer), 0, 0);

    gen.FIFO = {9, 8, 7, 6, 5, -1};

    mi400::valu_inst_type valu{};
    valu.wavetm = 12; // maps to FIFO[3]
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 6);
    EXPECT_EQ(gen.FIFO[0], 6);

    gen.FIFO = {9, 8, 7, 6, 5, -1};
    valu.wavetm = 14; // maps to FIFO[4]
    gen.current = valu.raw;
    EXPECT_EQ(gen.get_valu_inst_mi400(), 5);
    EXPECT_EQ(gen.FIFO[0], 5);
}

TEST(MI400TokenGeneratorTest, NextInstPathsUpdateFifo)
{
    struct inst_case_t
    {
        bool has_prefix;
        uint8_t prefix;
        size_t buffer_size;
        int wid;
        int inst;
        int64_t min_time;
    };

    const std::array<inst_case_t, 4> cases{
        inst_case_t{false, 0,    16, 13, 10, 0}, // unsafe path
        inst_case_t{false, 0,    3,  5,  11, 0}, // safe path
        inst_case_t{true,  0x00, 16, 7,  12, 0}, // NOP skip then INST
        inst_case_t{true,  0x0E, 16, 6,  13, 1}, // TIME skip then INST
    };

    for (const auto& c : cases)
    {
        gfx12::inst_type inst{};
        inst.header = 2; // low 4 bits become 0b0010 -> INST token type
        inst.tm = 0;
        inst.w64h = 0;
        inst.wid = c.wid;
        inst.inst = c.inst;

        std::array<uint8_t, 16> buffer{};
        size_t offset = 0;
        if (c.has_prefix)
        {
            buffer[0] = c.prefix;
            offset = 1;
        }

        buffer[offset + 0] = static_cast<uint8_t>(inst.raw & 0xFF);
        buffer[offset + 1] = static_cast<uint8_t>((inst.raw >> 8) & 0xFF);
        buffer[offset + 2] = static_cast<uint8_t>((inst.raw >> 16) & 0xFF);

        TestMI400TokenGenerator gen(buffer.data(), c.buffer_size, 0, 0);
        auto token = gen.next();

        EXPECT_EQ(token.type, RdnaType::INST);
        EXPECT_EQ(gen.FIFO[0], c.wid);
        EXPECT_GE(token.time, c.min_time);
    }
}

TEST(MI400TokenGeneratorTest, NextValuAndLongTokenPaths)
{
    // VALU_INST remaps wave id from FIFO
    {
        mi400::valu_inst_type valu{};
        valu.header = 3; // low 3 bits become 0b011 -> VALU_INST token type
        valu.tp = 0;
        valu.wavetm = 0;

        auto buffer = make_token_buffer(valu.raw, 1);
        TestMI400TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);
        gen.FIFO[0] = 9;

        auto token = gen.next();
        EXPECT_EQ(token.type, RdnaType::VALU_INST);

        ::valu_inst_type mapped{};
        mapped.raw = token.contents;
        EXPECT_EQ(mapped.wid, 9);
    }

    // NEW_PC_GFX12 path reads the extra byte when token length is > 64 bits
    {
        std::array<uint8_t, 16> buffer{};
        buffer[0] = 0x21; // NEW_PC_GFX12 encoding: {1,0,0,0,0,1,0}
        buffer[8] = 0xAB; // consumed by bits_toread > 64 path

        TestMI400TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);
        auto token = gen.next();

        EXPECT_EQ(token.type, RdnaType::NEW_PC_GFX12);
        EXPECT_EQ(token.contents, 0xAB00000000000000ull);
    }
}

TEST(MI400TokenGeneratorTest, SafeReaderHandlesRealtimeValuAndExtendedPcTokens)
{
    {
        gfx12::timestamp_type ts{};
        ts.header = 0b0000001;
        ts.rt = 1;
        ts.pl = 0;
        ts.time = 0x123456789ull;

        std::array<uint8_t, sizeof(uint64_t)> buffer{};
        std::memcpy(buffer.data(), &ts.raw, sizeof(ts.raw));

        mi400::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);
        auto token = gen.next();

        EXPECT_EQ(token.type, RdnaType::TIMESTAMP);
        ASSERT_EQ(gen.realtime.size(), 1u);
        EXPECT_EQ(gen.realtime.front().shader_clock, 0);
        EXPECT_EQ(gen.realtime.front().realtime_clock, ts.time);
    }

    {
        mi400::valu_inst_type valu{};
        valu.header = 0b011;
        valu.wavetm = 0;

        std::array<uint8_t, 1> buffer{static_cast<uint8_t>(valu.raw)};
        TestMI400TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);
        gen.FIFO[0] = 9;

        auto token = gen.next();
        EXPECT_EQ(token.type, RdnaType::VALU_INST);
        EXPECT_EQ(token.time, 1);

        ::valu_inst_type mapped{};
        mapped.raw = token.contents;
        EXPECT_EQ(mapped.wid, 9);
    }

    {
        std::array<uint8_t, 9> buffer{};
        buffer[0] = 0x21; // NEW_PC_GFX12 encoding: {1,0,0,0,0,1,0}
        buffer[8] = 0xAB;

        TestMI400TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);
        auto token = gen.next();

        EXPECT_EQ(token.type, RdnaType::NEW_PC_GFX12);
        EXPECT_EQ(token.contents, 0xAB00000000000000ull);
    }
}

TEST(MI400TokenGeneratorTest, EmptyBufferTerminates)
{
    uint8_t buffer[1] = {0};
    mi400::TokenGenerator gen(buffer, 1, 0, 0);

    int count = 0;
    while (gen.nextValid() && count < 100)
    {
        gen.next();
        count++;
    }
    EXPECT_LT(count, 100);
}

//=============================================================================
// MI400 XNACK tests
//=============================================================================

// Helper: simulate what rdna_sqtt.cpp does for an MI400 INST token
static void apply_mi400_inst(gfx10::wave_t& wave, int einst, int64_t time, int derate)
{
    mapped_inst_t mapped{WaveInstCategory::NONE, 0};
    try
    {
        mapped = mi400::map_to_common_type(einst, derate);
    }
    catch (std::exception&)
    {
        mi400::handle_xnack_rewind(wave);
        return;
    }

    if (mapped.category == WaveInstCategory::LD_SCALE)
    {
        wave.next_ld_scale = true;
        return;
    }

    int64_t inst_time = time;
    if (wave.next_ld_scale)
    {
        wave.next_ld_scale = false;
        inst_time -= 1;
        mapped.cycles++;
    }

    wave.apply_inst(inst_time, einst, mapped, 5);
}

TEST(MI400xnack, InstReplayThrowsForXnack) { EXPECT_THROW(mi400::map_to_common_type(79, 0), std::exception); }

TEST(MI400xnack, XnackRewindRemovesImmedInstructions)
{
    gfx10::Token token{};
    token.time = 100;
    token.type = RdnaType::WAVE_START;

    pcinfo_t addr{1, 0x1000};
    gfx10::wave_t wave(0, 0, 0, addr, token, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    // SALU
    apply_mi400_inst(wave, 0, 150, 0);

    // VMEM (buf_rd_1 = 47)
    apply_mi400_inst(wave, 47, 200, 0);

    // IMMED (barrier_wait = 19)
    apply_mi400_inst(wave, 19, 300, 0);
    apply_mi400_inst(wave, 19, 400, 0);

    size_t instCountBefore = wave.instructions.size();
    EXPECT_GE(instCountBefore, 1);

    int immedCountBefore = 0;
    for (const auto& instr : wave.instructions)
        if (instr.category == (uint32_t) WaveInstCategory::IMMED) immedCountBefore++;

    // Trigger XNACK rewind with replay instruction (79)
    apply_mi400_inst(wave, 79, 500, 0);

    int immedCountAfter = 0;
    for (const auto& instr : wave.instructions)
        if (instr.category == (uint32_t) WaveInstCategory::IMMED) immedCountAfter++;

    EXPECT_EQ(immedCountAfter, 0) << "XNACK rewind should remove all trailing IMMED instructions";
    EXPECT_EQ(wave.cur_state, WaveslotState::WS_STALL) << "Wave should be stalled after XNACK rewind";

    // Now add a VMEM instruction after the stall
    apply_mi400_inst(wave, 47, 700, 0);

    ASSERT_FALSE(wave.instructions.empty());
    const auto& lastInst = wave.instructions.back();
    EXPECT_EQ(lastInst.category, (uint32_t) WaveInstCategory::VMEM);

    int64_t expectedStall = 700 - 500;
    EXPECT_GE((int64_t) lastInst.stall, expectedStall - 1) << "Stall should reflect time since XNACK rewind";
}

TEST(MI400xnack, XnackRecoversFromWaveReady)
{
    gfx10::Token token{};
    token.time = 10;
    token.type = RdnaType::WAVE_START;
    pcinfo_t addr{1, 0x1000};
    gfx10::wave_t wave(0, 0, 0, addr, token, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    wave.apply_immediate(20);

    // salu
    apply_mi400_inst(wave, 0, 30, 1);

    // buf_rd_1
    wave.apply_wave_rdy(40);
    apply_mi400_inst(wave, 47, 40, 1);
    wave.apply_wave_rdy(40);

    wave.apply_immediate(50);
    wave.apply_immediate(60);

    // replay -> XNACK rewind
    apply_mi400_inst(wave, 79, 70, 0);

    int immedCountAfter = 0;
    for (const auto& instr : wave.instructions)
        if (instr.category == (uint32_t) WaveInstCategory::IMMED) immedCountAfter++;

    EXPECT_EQ(immedCountAfter, 1) << "XNACK rewind should remove all trailing IMMED instructions";
    EXPECT_EQ(wave.cur_state, WaveslotState::WS_STALL);

    EXPECT_EQ(wave.instructions.size(), 2) << "We only expect SALU and IMMED";
    EXPECT_EQ(wave.instructions.back().category, (uint32_t) WaveInstCategory::SALU) << "Last=SALU";
    EXPECT_EQ(wave.timeline.size(), 3) << "We only expect Idle -> Wait -> Exec";
}

TEST(MI400xnack, MultipleXnack)
{
    gfx10::Token token{};
    token.time = 10;
    token.type = RdnaType::WAVE_START;
    pcinfo_t addr{1, 0x1000};
    gfx10::wave_t wave(0, 0, 0, addr, token, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    // salu
    apply_mi400_inst(wave, 0, 10, 1);

    for (int i = 0; i < 19; i++)
    {
        int64_t base_time = i * 20 + 20;
        // buf_rd_1
        if (i % 2) wave.apply_wave_rdy(base_time);
        apply_mi400_inst(wave, 47, base_time, 1);
        if (i % 3) wave.apply_wave_rdy(base_time);

        if (i % 4) wave.apply_immediate(base_time + 5);

        // replay -> XNACK rewind
        apply_mi400_inst(wave, 79, base_time + 19, 0);
        // buf_rd_1 after XNACK
        apply_mi400_inst(wave, 47, base_time + 19, 1);
    }

    int immedCountAfter = 0;
    for (const auto& instr : wave.instructions)
        if (instr.category == (uint32_t) WaveInstCategory::IMMED) immedCountAfter++;

    EXPECT_EQ(immedCountAfter, 0);
    EXPECT_EQ(wave.cur_state, WaveslotState::WS_EXEC);

    EXPECT_EQ(wave.instructions.size(), 20) << "We expect SALU and 19xVMEM";
    EXPECT_EQ(wave.instructions.front().category, (uint32_t) WaveInstCategory::SALU);
    EXPECT_EQ(wave.instructions.back().category, (uint32_t) WaveInstCategory::VMEM);

    int exec_time = 0;
    for (size_t i = 1; i < wave.timeline.size(); i++)
        if (wave.timeline[i].type != WaveslotState::WS_STALL) exec_time += wave.timeline[i].duration;

    EXPECT_LE(exec_time, 20) << "Execution time should be small!";
}

TEST(MI400xnack, WaveReadyAfterXnack)
{
    gfx10::Token token{};
    token.time = 10;
    token.type = RdnaType::WAVE_START;
    pcinfo_t addr{1, 0x1000};
    gfx10::wave_t wave(0, 0, 0, addr, token, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    // salu
    apply_mi400_inst(wave, 0, 20, 0);

    // replay -> XNACK rewind
    apply_mi400_inst(wave, 79, 30, 0);

    // buf_rd_1 with wave_rdy
    wave.apply_wave_rdy(30);
    apply_mi400_inst(wave, 47, 30, 1);
    wave.apply_wave_rdy(30);

    EXPECT_EQ(wave.cur_state, WaveslotState::WS_EXEC);
}
