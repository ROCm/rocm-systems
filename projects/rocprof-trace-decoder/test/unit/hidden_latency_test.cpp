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

#include "hidden_latency.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "rocprof_trace_decoder/trace_decoder_instrument.h"
#include "stitch/stitch.hpp"

namespace
{
constexpr uint64_t CODE_OBJECT = 1;

pcinfo_t make_pc(uint64_t address, uint64_t code_object_id = CODE_OBJECT)
{
    pcinfo_t pc{};
    pc.address = address;
    pc.code_object_id = code_object_id;
    return pc;
}

/// Serves fixed ISA text per address, so matrix classification can be exercised without a
/// disassembler.
class FakeCodeServicer : public ICodeServicer
{
public:
    void set(uint64_t address, std::string instruction) { lines[address] = std::move(instruction); }

    assemblyLine GetInstruction(pcinfo_t addr, int) override
    {
        assemblyLine line{};
        line.addr = addr;
        auto found = lines.find(addr.address);
        line.line = found == lines.end() ? "s_nop 0" : found->second;
        return line;
    }

private:
    std::map<uint64_t, std::string> lines;
};

/// Drives HiddenLatencyAnalysis and collects the records it emits.
class Harness
{
public:
    Harness() :
    servicer(std::make_shared<FakeCodeServicer>()),
    servicer_interface(servicer),
    translator(code, servicer_interface, 12)
    {}

    void set_isa(uint64_t address, std::string instruction) { servicer->set(address, std::move(instruction)); }

    WaveDataInternal& add_wave(int simd, int64_t begin_time)
    {
        waves.push_back(std::make_unique<WaveDataInternal>(0, simd, 0, begin_time, make_pc(0, 0), false));
        return *waves.back();
    }

    void add_instruction(WaveDataInternal& wave, uint64_t address, WaveInstCategory category, int64_t time,
                         int64_t duration, int64_t stall = 0, uint64_t code_object_id = CODE_OBJECT)
    {
        Instruction inst{time, category, duration, stall};
        inst.pc = make_pc(address, code_object_id);
        // Stitching resolves and caches the ISA for every program counter it matches, and the
        // analysis reads that cache rather than disassembling. Prime it the same way.
        translator.getcode(inst.pc);
        wave.instructions.push_back(inst);
    }

    void add_other_simd(int64_t time, uint16_t cycles, WaveInstCategory category)
    {
        att_other_simd_t record{};
        record.size = sizeof(record);
        record.time = time;
        record.cycles = cycles;
        record.category = static_cast<uint8_t>(category);
        std::vector<att_other_simd_t> batch{record};
        analysis.add_other_simd(batch);
    }

    /// Feeds every wave to the analysis and returns the emitted records.
    std::vector<rocprofiler_thread_trace_decoder_hidden_latency_t> run()
    {
        for (auto& wave : waves)
            analysis.add_wave(*wave, translator);
        emitted.clear();
        analysis.finalize(collect, this);
        return emitted;
    }

    /// Convenience lookup for the single record matching a program counter.
    const rocprofiler_thread_trace_decoder_hidden_latency_t* find(
        const std::vector<rocprofiler_thread_trace_decoder_hidden_latency_t>& records,
        uint64_t address,
        uint8_t simd = 0
    )
    {
        for (const auto& record : records)
            if (record.pc.address == address && record.simd == simd) return &record;
        return nullptr;
    }

private:
    static rocprofiler_thread_trace_decoder_status_t collect(
        rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t size, void* userdata
    )
    {
        auto* self = static_cast<Harness*>(userdata);
        EXPECT_EQ(type, ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY);
        auto* records = static_cast<rocprofiler_thread_trace_decoder_hidden_latency_t*>(data);
        self->emitted.assign(records, records + size);
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }

    std::shared_ptr<FakeCodeServicer> servicer;
    std::shared_ptr<ICodeServicer> servicer_interface;
    std::vector<assemblyLinePtr> code;
    PCTranslator translator;
    HiddenLatencyAnalysis analysis;
    std::vector<std::unique_ptr<WaveDataInternal>> waves;
    std::vector<rocprofiler_thread_trace_decoder_hidden_latency_t> emitted;
};
} // namespace

TEST(HiddenLatencyMatrix, RecognizesEveryMatrixPrefix)
{
    EXPECT_TRUE(is_matrix_instruction("v_mfma_f32_16x16x16f16 v[0:3], v0, v1, v[0:3]"));
    EXPECT_TRUE(is_matrix_instruction("v_smfma_f32_16x16x32_f16 v[0:3], v0, v1, v[0:3]"));
    EXPECT_TRUE(is_matrix_instruction("v_wmma_f32_16x16x16_f16 v0, v1, v2"));
    EXPECT_TRUE(is_matrix_instruction("v_swmma_f32_16x16x32_f16 v0, v1, v2"));
}

TEST(HiddenLatencyMatrix, RejectsOrdinaryInstructions)
{
    EXPECT_FALSE(is_matrix_instruction("v_mov_b32_e32 v0, v1"));
    EXPECT_FALSE(is_matrix_instruction("s_nop 0"));
    EXPECT_FALSE(is_matrix_instruction(""));
    // The prefix must be at the start, not merely present.
    EXPECT_FALSE(is_matrix_instruction("; comment about v_mfma"));
}

TEST(HiddenLatencyAnalysisTest, EmitsNothingWithoutWaves)
{
    Harness harness;
    EXPECT_TRUE(harness.run().empty());
}

TEST(HiddenLatencyAnalysisTest, LoneInstructionHidesNothing)
{
    Harness harness;
    auto& wave = harness.add_wave(0, 0);
    harness.add_instruction(wave, 0x100, WaveInstCategory::IMMED, 0, 8);

    auto records = harness.run();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].idle, 0);
    EXPECT_EQ(records[0].stall, 0);
    EXPECT_EQ(records[0].issue, 0);
}

TEST(HiddenLatencyAnalysisTest, ConcurrentValuHidesOtherPipeIssue)
{
    Harness harness;
    auto& target_wave = harness.add_wave(0, 0);
    harness.add_instruction(target_wave, 0x100, WaveInstCategory::IMMED, 0, 4);
    auto& busy_wave = harness.add_wave(0, 0);
    harness.add_instruction(busy_wave, 0x200, WaveInstCategory::VALU, 0, 4);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100);
    ASSERT_NE(target, nullptr);
    // The whole window overlaps VALU work from another wave on the same SIMD.
    EXPECT_EQ(target->issue, 4);
}

TEST(HiddenLatencyAnalysisTest, SamePipeDoesNotHideOwnExecution)
{
    Harness harness;
    auto& first = harness.add_wave(0, 0);
    harness.add_instruction(first, 0x100, WaveInstCategory::VALU, 0, 4);
    auto& second = harness.add_wave(0, 0);
    harness.add_instruction(second, 0x200, WaveInstCategory::VALU, 0, 4);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100);
    ASSERT_NE(target, nullptr);
    // VALU cannot hide VALU execution, and no matrix work exists to hide it either.
    EXPECT_EQ(target->issue, 0);
}

TEST(HiddenLatencyAnalysisTest, MatrixWorkHidesValuExecution)
{
    Harness harness;
    harness.set_isa(0x200, "v_mfma_f32_16x16x16f16 v[0:3], v0, v1, v[0:3]");
    auto& target_wave = harness.add_wave(0, 0);
    harness.add_instruction(target_wave, 0x100, WaveInstCategory::VALU, 0, 8);
    auto& matrix_wave = harness.add_wave(0, 0);
    harness.add_instruction(matrix_wave, 0x200, WaveInstCategory::VALU, 0, 8);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100);
    ASSERT_NE(target, nullptr);
    // MATRIX is a separate pipe, so a VALU instruction's execution can hide behind it.
    EXPECT_EQ(target->issue, 8);
}

TEST(HiddenLatencyAnalysisTest, IdleBeforeAnInstructionCanBeHidden)
{
    Harness harness;
    auto& target_wave = harness.add_wave(0, 0);
    harness.add_instruction(target_wave, 0x100, WaveInstCategory::IMMED, 8, 2);
    auto& busy_wave = harness.add_wave(0, 0);
    harness.add_instruction(busy_wave, 0x200, WaveInstCategory::VALU, 4, 4);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100);
    ASSERT_NE(target, nullptr);
    // Cycles 4 through 8 were busy on the VALU pipe while this wave sat idle.
    EXPECT_EQ(target->idle, 4);
}

TEST(HiddenLatencyAnalysisTest, OtherSimdRecordsFeedVectorMemory)
{
    Harness harness;
    auto& wave = harness.add_wave(0, 0);
    harness.add_instruction(wave, 0x100, WaveInstCategory::IMMED, 0, 4);
    harness.add_other_simd(0, 4, WaveInstCategory::VMEM);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100);
    ASSERT_NE(target, nullptr);
    EXPECT_EQ(target->issue, 4);
}

TEST(HiddenLatencyAnalysisTest, UnresolvedProgramCountersAreNotReported)
{
    Harness harness;
    auto& wave = harness.add_wave(0, 0);
    harness.add_instruction(wave, 0, WaveInstCategory::IMMED, 0, 4, 0, 0);

    EXPECT_TRUE(harness.run().empty());
}

TEST(HiddenLatencyAnalysisTest, SimdsAreScopedSeparatelyAndPcsAccumulate)
{
    Harness harness;
    for (int simd : {0, 1})
    {
        auto& target_wave = harness.add_wave(simd, 0);
        harness.add_instruction(target_wave, 0x100, WaveInstCategory::IMMED, 0, 4);
        harness.add_instruction(target_wave, 0x100, WaveInstCategory::IMMED, 4, 4);
        auto& busy_wave = harness.add_wave(simd, 0);
        harness.add_instruction(busy_wave, 0x200, WaveInstCategory::VALU, 0, 8);
    }

    auto records = harness.run();
    const auto* first = harness.find(records, 0x100, 0);
    const auto* second = harness.find(records, 0x100, 1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    // Two executions of the same PC accumulate within a SIMD, and each SIMD reports its own.
    EXPECT_EQ(first->issue, 8);
    EXPECT_EQ(second->issue, 8);
}

TEST(HiddenLatencyAnalysisTest, BusyPipeOnAnotherSimdDoesNotHide)
{
    Harness harness;
    auto& target_wave = harness.add_wave(0, 0);
    harness.add_instruction(target_wave, 0x100, WaveInstCategory::IMMED, 0, 4);
    auto& busy_wave = harness.add_wave(1, 0);
    harness.add_instruction(busy_wave, 0x200, WaveInstCategory::VALU, 0, 4);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x100, 0);
    ASSERT_NE(target, nullptr);
    // Concurrency is scoped per SIMD, so VALU work on SIMD 1 competes for a different pipe.
    EXPECT_EQ(target->idle, 0);
    EXPECT_EQ(target->stall, 0);
    EXPECT_EQ(target->issue, 0);
}

TEST(HiddenLatencyAnalysisTest, VectorMemoryHidesScalarExecution)
{
    for (auto category : {WaveInstCategory::VMEM, WaveInstCategory::LDS, WaveInstCategory::FLAT})
    {
        Harness harness;
        auto& target_wave = harness.add_wave(0, 0);
        harness.add_instruction(target_wave, 0x100, WaveInstCategory::SALU, 10, 10, 4);
        auto& busy_wave = harness.add_wave(0, 0);
        harness.add_instruction(busy_wave, 0x200, category, 14, 6);

        auto records = harness.run();
        const auto* target = harness.find(records, 0x100);
        ASSERT_NE(target, nullptr);
        // Every vector-memory category outranks the scalar pipe, so it hides scalar execution.
        EXPECT_EQ(target->issue, 6);
    }
}

TEST(HiddenLatencyAnalysisTest, IdleWindowStartsAtThePreviousInstructionsEnd)
{
    Harness harness;
    auto& wave = harness.add_wave(0, 0);
    harness.add_instruction(wave, 0x100, WaveInstCategory::IMMED, 0, 10);
    harness.add_instruction(wave, 0x200, WaveInstCategory::IMMED, 2, 2);
    harness.add_instruction(wave, 0x300, WaveInstCategory::IMMED, 8, 2);
    auto& busy_wave = harness.add_wave(0, 0);
    harness.add_instruction(busy_wave, 0x400, WaveInstCategory::VALU, 4, 4);

    auto records = harness.run();
    const auto* target = harness.find(records, 0x300);
    ASSERT_NE(target, nullptr);
    // The idle window opens where the immediately preceding instruction ended, at 4, not where
    // the longer instruction before that ended, at 10.
    EXPECT_EQ(target->idle, 4);
}

namespace
{
/// One gfx9 capture: the software header the producer prepends, then a little payload.
std::vector<uint8_t> gfx9_buffer(int shader_engine)
{
    rocprof_trace_decoder_gfx9_header_t header{};
    header.raw = 0;
    header.legacy_version = 0x11;
    header.gfx9_version2 = 5; // MI300
    header.SEID = static_cast<uint64_t>(shader_engine);
    header.DCU = 0;
    header.DSIMDM = 0xF;

    std::vector<uint8_t> buffer(sizeof(header) + 64, 0);
    std::memcpy(buffer.data(), &header, sizeof(header));
    return buffer;
}

/// Hands each prepared capture to the decoder in turn, which is what a caller combining
/// several shader engines into one parse would do.
struct SeDataFeed
{
    std::vector<std::vector<uint8_t>> buffers;
    size_t next = 0;
};

uint64_t feed_se_data(uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    auto* feed = static_cast<SeDataFeed*>(userdata);
    if (feed->next >= feed->buffers.size())
    {
        *buffer = nullptr;
        *buffer_size = 0;
        return 0;
    }
    auto& current = feed->buffers[feed->next++];
    *buffer = current.data();
    *buffer_size = current.size();
    return current.size();
}

rocprofiler_thread_trace_decoder_status_t
stub_isa(char* instruction, uint64_t* memory_size, uint64_t* size, rocprofiler_thread_trace_decoder_pc_t, void*)
{
    static constexpr char text[] = "s_nop 0";
    if (*size < sizeof(text)) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    std::memcpy(instruction, text, sizeof(text));
    *size = sizeof(text) - 1;
    *memory_size = 4;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
collect_info(rocprofiler_thread_trace_decoder_record_type_t type, void* data, uint64_t size, void* userdata)
{
    if (type != ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    auto* seen = static_cast<std::vector<rocprofiler_thread_trace_decoder_info_t>*>(userdata);
    auto* records = static_cast<rocprofiler_thread_trace_decoder_info_t*>(data);
    seen->insert(seen->end(), records, records + size);
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

/// Parses the given captures in one call with the analysis on, returning the info records.
std::vector<rocprofiler_thread_trace_decoder_info_t> parse_with_analysis(std::vector<std::vector<uint8_t>> buffers)
{
    rocprof_trace_decoder_handle_t handle{};
    EXPECT_EQ(rocprof_trace_decoder_create_handle(&handle), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    EXPECT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle, stub_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );
    EXPECT_EQ(
        rocprof_trace_decoder_set_analysis(handle, ROCPROF_TRACE_DECODER_ANALYSIS_HIDDEN_LATENCY),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    SeDataFeed feed{std::move(buffers), 0};
    EXPECT_EQ(
        rocprof_trace_decoder_set_se_data_callback(handle, feed_se_data, &feed),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    std::vector<rocprofiler_thread_trace_decoder_info_t> seen;
    rocprof_trace_decoder_parse(handle, nullptr, 0, collect_info, &seen);
    rocprof_trace_decoder_destroy_handle(handle);
    return seen;
}

bool warned_about_multiple_buffers(const std::vector<rocprofiler_thread_trace_decoder_info_t>& seen)
{
    return std::find(seen.begin(), seen.end(), ROCPROFILER_THREAD_TRACE_DECODER_INFO_ANALYSIS_MULTIPLE_BUFFERS) !=
           seen.end();
}
} // namespace

TEST(HiddenLatencyScopeTest, WarnsOnRepeatedCapturesFromOneShaderEngine)
{
    // Each buffer is a complete capture, so two captures describe separate timelines even from
    // one shader engine. Counting buffers is also what covers gfx10+, whose traces name no
    // shader engine for a comparison to use.
    EXPECT_TRUE(warned_about_multiple_buffers(parse_with_analysis({gfx9_buffer(3), gfx9_buffer(3)})));
}

TEST(HiddenLatencyScopeTest, SingleBufferNeverWarns)
{
    EXPECT_FALSE(warned_about_multiple_buffers(parse_with_analysis({gfx9_buffer(0)})));
}

TEST(HiddenLatencyAnalysisTest, RecordsCarrySizeForAbiGrowth)
{
    Harness harness;
    auto& wave = harness.add_wave(0, 0);
    harness.add_instruction(wave, 0x100, WaveInstCategory::IMMED, 0, 4);

    auto records = harness.run();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].size, sizeof(rocprofiler_thread_trace_decoder_hidden_latency_t));
}
