// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-detector/race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <format>
#include <optional>
#include <regex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu::plugins::race_detector;
using rocjitsu::amdgpu::WaitCounterType;
namespace cdna4 = rocjitsu::cdna4;

using Trace = RingBuffer<InstructionSnapshot, kTraceHistorySize>;

MarkedInstruction mark(const Trace &trace, size_t index, int wave = 0) {
  return {trace[index], wave, 0};
}

std::string text(const InstructionSnapshot &instruction) {
  return std::format("instruction {}", instruction.encoding()[0]);
}

InstructionSnapshot snapshot(uint64_t pc, uint32_t word, size_t width = 1,
                             bool preserve_in_trace = false) {
  std::array<uint32_t, InstructionSnapshot::kMaxWords> words{word};
  return {pc, std::span(words).first(width), preserve_in_trace};
}

struct PrintedRow {
  bool marked;
  uint64_t pc;
  size_t sequence;
};

std::vector<PrintedRow> rows(const std::string &report) {
  const std::regex pattern(R"(^\s*(==>\s*)?0x([0-9a-f]+)\s+instruction ([0-9]+))");
  std::vector<PrintedRow> result;
  std::istringstream lines(report);
  std::string line;
  while (std::getline(lines, line)) {
    std::smatch match;
    if (std::regex_search(line, match, pattern))
      result.push_back(
          {match[1].matched, std::stoull(match[2], nullptr, 16), std::stoull(match[3])});
  }
  return result;
}

size_t elided_count(const std::string &report) {
  const std::regex pattern(R"(\.\.\. ([0-9]+) instructions elided)");
  size_t result = 0;
  for (auto it = std::sregex_iterator(report.begin(), report.end(), pattern);
       it != std::sregex_iterator(); ++it)
    result += std::stoull((*it)[1]);
  return result;
}

TEST(RaceReportTraceTest, PreservesExecutionOrderAcrossBranchesAndRepeatedPcs) {
  Trace trace;
  for (uint32_t i = 0; uint64_t pc : {0x100u, 0x120u, 0x108u, 0x108u})
    trace.push(snapshot(pc, i++, 1));
  EXPECT_EQ(formatTrace(trace, text, mark(trace, 0), mark(trace, 3)),
            "       PC     Instruction\n"
            "  ==>  0x100  instruction 0  ; <-- wave 0 lane 0\n"
            "       0x120  instruction 1\n"
            "       0x108  instruction 2\n"
            "  ==>  0x108  instruction 3  ; <-- wave 0 lane 0\n");
}

TEST(RaceReportTraceTest, PadsPcsSoInstructionsShareAColumn) {
  Trace trace;
  trace.push(snapshot(0xfc, 0, 1));
  trace.push(snapshot(0x100, 1, 1));
  EXPECT_EQ(
      formatTrace(trace, text, MarkedInstruction{snapshot(0x2000, 9, 1), 1, 0}, mark(trace, 1)),
      "       PC      Instruction\n"
      "  ==>  0x2000  instruction 9  ; <-- wave 1 lane 0\n"
      "       Execution trace below is for wave 0.\n"
      "       0xfc    instruction 0\n"
      "  ==>  0x100   instruction 1  ; <-- wave 0 lane 0\n");
}

TEST(RaceReportTraceTest, ContextLimitAndRingWrapPreserveRowsAndAccountForEveryGap) {
  for (size_t length : {1u, 2u, 127u, 128u, 129u, 255u, 256u, 257u, 513u}) {
    SCOPED_TRACE(std::format("length={}", length));
    Trace trace;
    for (uint32_t i = 0; i < length; ++i)
      trace.push(snapshot(0x100 + i * 4, i, 1));
    const std::string report =
        formatTrace(trace, text, mark(trace, 0), mark(trace, trace.size() - 1));
    const std::vector<PrintedRow> printed = rows(report);
    // A one-instruction gap is printed rather than elided.
    ASSERT_EQ(printed.size(),
              trace.size() <= kDefaultTraceContext + 1 ? trace.size() : kDefaultTraceContext);
    ASSERT_TRUE(printed.front().marked);
    ASSERT_TRUE(printed.back().marked);
    EXPECT_EQ(printed.front().sequence, length - trace.size());
    EXPECT_EQ(printed.back().sequence, length - 1);
    EXPECT_EQ(printed.size() + elided_count(report), trace.size());
  }
}

TEST(RaceReportTraceTest, ElisionPreservesProtectedRowsAcrossPcDiscontinuities) {
  Trace trace;
  for (uint32_t i = 0; i < 200; ++i)
    trace.push(snapshot((i < 100 ? 0x1000u : 0x800u) + i * 4, i, 1, i == 80 || i == 120));
  for (size_t context : {size_t{2}, kDefaultTraceContext}) {
    SCOPED_TRACE(context);
    const std::string report = formatTrace(trace, text, mark(trace, 0), mark(trace, 199), context);
    const std::vector<PrintedRow> printed = rows(report);
    ASSERT_GT(elided_count(report), 0u);
    for (size_t required : {0u, 80u, 120u, 199u})
      EXPECT_TRUE(std::ranges::any_of(
          printed, [=](const PrintedRow &row) { return row.sequence == required; }));
    for (size_t elided : {99u, 100u})
      EXPECT_FALSE(std::ranges::any_of(
          printed, [=](const PrintedRow &row) { return row.sequence == elided; }));
    EXPECT_EQ(printed.size() + elided_count(report), 200u);
  }
}

TEST(RaceReportTraceTest, ElisionNeverHidesASingleInstruction) {
  Trace trace;
  for (uint32_t i = 0; i < 200; ++i)
    trace.push(snapshot(0x1000 + i * 4, i, 1, i == 80 || i == 82));
  const std::string report = formatTrace(trace, text, mark(trace, 0), mark(trace, 199));
  const std::vector<PrintedRow> printed = rows(report);
  EXPECT_TRUE(
      std::ranges::any_of(printed, [](const PrintedRow &row) { return row.sequence == 81; }));
  EXPECT_EQ(report.find("... 1 instructions elided"), std::string::npos);
  EXPECT_EQ(printed.size() + elided_count(report), 200u);
}

TEST(RaceReportTraceTest, CompactsNopsWithoutAbsorbingMarkedInstructions) {
  Trace trace;
  const std::array<uint32_t, 1> nop = cdna4::build_sopp(cdna4::kSNopSopp);
  for (uint64_t pc = 0x100; pc <= 0x114; pc += 4)
    trace.push(InstructionSnapshot{pc, nop, /*preserve_in_trace=*/false, /*is_nop=*/true});
  const auto disassemble = [](const InstructionSnapshot &) { return "s_nop 0"; };
  EXPECT_EQ(formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 5)),
            "       PC     Instruction\n"
            "  ==>  0x100  s_nop 0  ; <-- wave 0 lane 0\n"
            "       0x104  s_nop 0  ; 4 executions, through PC 0x110\n"
            "  ==>  0x114  s_nop 0  ; <-- wave 0 lane 0\n");
}

TEST(RaceReportTraceTest, NopRunsRespectEncodingAndPcContinuity) {
  Trace trace;
  const std::array<uint32_t, 1> nop = cdna4::build_sopp(cdna4::kSNopSopp);
  const std::array<uint32_t, 1> longer_nop = cdna4::build_sopp(cdna4::kSNopSopp, {.simm16 = 7});
  trace.push(snapshot(0x100, 0));
  for (uint64_t pc : {0x104u, 0x108u, 0x10cu, 0x110u, 0x200u, 0x204u})
    trace.push(InstructionSnapshot{pc, pc < 0x10c ? nop : longer_nop,
                                   /*preserve_in_trace=*/false, /*is_nop=*/true});
  trace.push(snapshot(0x208, 1));
  const std::string report = formatTrace(trace, text, mark(trace, 0), mark(trace, 7));
  EXPECT_NE(report.find("2 executions, through PC 0x108\n"), std::string::npos);
  EXPECT_NE(report.find("2 executions, through PC 0x110\n"), std::string::npos);
  EXPECT_NE(report.find("2 executions, through PC 0x204\n"), std::string::npos);
  EXPECT_EQ(rows(report).size(), 5u);
}

TEST(RaceReportTraceTest, ContextCountsExecutionsBeforeCompactionAndKeepsGapsSeparate) {
  Trace trace;
  const std::array<uint32_t, 1> nop = cdna4::build_sopp(cdna4::kSNopSopp);
  for (uint64_t pc = 0x100; pc <= 0x11c; pc += 4)
    trace.push(InstructionSnapshot{pc, nop, /*preserve_in_trace=*/false, /*is_nop=*/true});
  const auto disassemble = [](const InstructionSnapshot &) { return "s_nop 0"; };
  const std::string short_report =
      formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 4);
  EXPECT_EQ(short_report, "       PC     Instruction\n"
                          "  ==>  0x100  s_nop 0  ; <-- wave 0 lane 0\n"
                          "       0x104  s_nop 0\n"
                          "       ... 4 instructions elided ...\n"
                          "       0x118  s_nop 0\n"
                          "  ==>  0x11c  s_nop 0  ; <-- wave 0 lane 0\n");
  const std::string full_report =
      formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 256);
  EXPECT_NE(full_report.find("6 executions, through PC 0x118\n"), std::string::npos);
  EXPECT_EQ(elided_count(full_report), 0u);
  const std::string odd_report = formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 5);
  EXPECT_EQ(elided_count(odd_report), 3u);
  EXPECT_NE(odd_report.find("2 executions, through PC 0x118\n"), std::string::npos);
}

TEST(RaceReportConfigTest, ValidatesTraceContextAtConstruction) {
  for (const char *config : {static_cast<const char *>(nullptr), "", "{}", R"({"trace_context":2})",
                             R"({"trace_context":3})", R"({"trace_context":256})"}) {
    SCOPED_TRACE(config ? config : "null");
    EXPECT_NO_THROW({ RaceDetectorPlugin plugin(config); });
  }
  for (const char *config :
       {"{", "[]", "null", R"({"trace_context":-1})", R"({"trace_context":0})",
        R"({"trace_context":1})", R"({"trace_context":257})", R"({"trace_context":1e30})",
        R"({"trace_context":2.5})", R"({"trace_context":true})", R"({"trace_context":"128"})"}) {
    SCOPED_TRACE(config);
    EXPECT_THROW({ RaceDetectorPlugin plugin(config); }, std::invalid_argument);
  }
}

TEST(RaceReportTraceTest, RepeatedPcWithDifferentBytesSelectsTheMatchingEncoding) {
  Trace trace;
  for (uint32_t i = 0; i < 6; ++i)
    trace.push(snapshot(0x100 + (i % 2) * 4, i, 1));
  const std::string report = formatTrace(trace, text, mark(trace, 2), mark(trace, 5));
  const std::vector<PrintedRow> printed = rows(report);
  ASSERT_EQ(printed.size(), 4u);
  EXPECT_EQ(printed[0].sequence, 2u);
  EXPECT_EQ(printed.back().sequence, 5u);
}

TEST(RaceReportTraceTest, OtherWaveConflictIsSeparateEvenWhenPcsAndEncodingsMatch) {
  Trace trace;
  trace.push(snapshot(0x100, 0, 1));
  trace.push(snapshot(0x104, 1, 1));
  const std::string report = formatTrace(trace, text, mark(trace, 0, 1), mark(trace, 1));
  const std::vector<PrintedRow> printed = rows(report);
  ASSERT_EQ(printed.size(), 3u);
  EXPECT_NE(report.find("Execution trace below is for wave 0."), std::string::npos);
}

TEST(RaceReportTraceTest, OtherWaveConflictStartsAtThisWavesLatestRunOfThatInstruction) {
  Trace trace;
  for (uint32_t i = 0; i < 40; ++i)
    trace.push(snapshot(0x100 + (i % 20) * 4, i % 20, 1));
  const MarkedInstruction producer{snapshot(0x104, 1, 1), 1, 0};
  const std::vector<PrintedRow> printed = rows(formatTrace(trace, text, producer, mark(trace, 39)));
  ASSERT_EQ(printed.size(), 20u);
  EXPECT_TRUE(printed[0].marked);
  EXPECT_EQ(printed[1].pc, 0x104u);
  EXPECT_FALSE(printed[1].marked);
  EXPECT_EQ(printed.back().pc, 0x14cu);
}

TEST(RaceReportTraceTest, OtherWaveConflictNeverRunHereShowsOnlyRecentRows) {
  Trace trace;
  for (uint32_t i = 0; i < 40; ++i)
    trace.push(snapshot(0x100 + i * 4, i, 1));
  const MarkedInstruction producer{snapshot(0x2000, 99, 1), 1, 0};
  for (size_t context : {size_t{2}, size_t{3}, kDefaultTraceContext, kTraceHistorySize}) {
    SCOPED_TRACE(context);
    const std::string report = formatTrace(trace, text, producer, mark(trace, 39), context);
    const std::vector<PrintedRow> printed = rows(report);
    const size_t expected_context = std::min(context, kOtherWaveTraceContext);
    ASSERT_EQ(printed.size(), 1 + expected_context);
    EXPECT_EQ(printed[1].sequence, 40 - expected_context);
    EXPECT_EQ(printed.back().sequence, 39u);
    EXPECT_EQ(report.find("elided"), std::string::npos);
  }
}

TEST(RaceReportTraceTest, ExpiredProducerRetainsIdentityAndMissingHistory) {
  Trace trace;
  const InstructionSnapshot producer = snapshot(0x100, 0, 1);
  trace.push(producer);
  for (uint32_t i = 1; i < 300; ++i)
    trace.push(snapshot(0x100 + i * 4, i, 1));
  const std::string report =
      formatTrace(trace, text, MarkedInstruction{producer, 0, 0}, mark(trace, 255));
  const std::vector<PrintedRow> printed = rows(report);
  ASSERT_GE(printed.size(), 2u);
  EXPECT_EQ(printed[0].sequence, 0u);
  EXPECT_EQ(printed[1].sequence, 44u);
  EXPECT_NE(report.find("before trace window"), std::string::npos);
  EXPECT_NE(report.find("earlier instructions not recorded"), std::string::npos);
  EXPECT_EQ(printed.back().sequence, 299u);
  EXPECT_EQ(printed.size() - 1 + elided_count(report), trace.size());
}

TEST(RaceReportTraceTest, EmptyTraceMarksCurrentAccessAsOutsideRecordedTrace) {
  const MarkedInstruction current{snapshot(0x104, 1, 1), 0, 0};
  const std::string report = formatTrace({}, text, std::nullopt, current);
  ASSERT_EQ(rows(report).size(), 1u);
  EXPECT_NE(report.find("outside recorded trace"), std::string::npos);
}

#ifndef NDEBUG
TEST(RaceReportTraceDeathTest, AssertsReversedExecutionOrder) {
  Trace trace;
  trace.push(snapshot(0x100, 0, 1));
  trace.push(snapshot(0x104, 1, 1));
  EXPECT_DEATH(formatTrace(trace, text, mark(trace, 1), mark(trace, 0)),
               "race conflict follows the current access");
}
#endif

TEST(RaceReportDescriptionTest, DescribesRegisterAndLdsDirectionsAcrossCompletionStates) {
  using Space = RaceViolation::Space;
  const struct {
    Space space;
    MemoryEventType event_type;
    bool write;
    int owner;
    bool complete;
    const char *resource;
    const char *hazard;
    const char *operation;
  } cases[] = {
      {Space::VGPR, MemoryEventType::GLOBAL_TO_VGPR, false, 0, false, "VGPR v2", "RAW",
       "load into VGPRs"},
      {Space::VGPR, MemoryEventType::LDS_TO_VGPR, true, 0, false, "VGPR v2", "WAW",
       "load into VGPRs"},
      {Space::SGPR, MemoryEventType::GLOBAL_TO_SGPR, false, 0, false, "SGPR s2", "RAW",
       "load into SGPRs"},
      {Space::TTMP, MemoryEventType::GLOBAL_TO_TTMP, true, 0, false, "TTMP ttmp2", "WAW",
       "load into TTMPs"},
      {Space::LDS, MemoryEventType::VGPR_TO_LDS, false, 1, false, "LDS byte 2", "RAW",
       "transfer from VGPRs into LDS"},
      {Space::LDS, MemoryEventType::VGPR_TO_LDS, false, 1, true, "LDS byte 2", "RAW",
       "transfer from VGPRs into LDS"},
      {Space::LDS, MemoryEventType::LDS_TO_VGPR, true, 1, false, "LDS byte 2", "WAR",
       "transfer from LDS into VGPRs"},
      {Space::LDS, MemoryEventType::LDS_TO_VGPR, true, 1, true, "LDS byte 2", "WAR",
       "transfer from LDS into VGPRs"},
      {Space::LDS, MemoryEventType::LDS_TO_GLOBAL, true, 1, false, "LDS byte 2", "WAR",
       "transfer from LDS into global memory"},
      {Space::LDS, MemoryEventType::GLOBAL_TO_LDS, false, 0, false, "LDS byte 2", "RAW",
       "load into LDS"},
      {Space::LDS, MemoryEventType::GLOBAL_TO_LDS, false, 1, true, "LDS byte 2", "RAW",
       "load into LDS"},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.operation);
    RaceDetector detector(2, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
    const EventId event = detector.allocateEventId(
        WaveId{test.owner}, testInstruction(0x100), test.event_type, {2}, 1, 0xf, {},
        CounterObligations(WaitCounterType::LGKMCNT, MemoryOrderClass::LDS), MemoryOrderClass::LDS);
    if (test.complete) {
      for (const auto obligation : detector.events().counterObligations(event))
        detector.satisfyEventWaitCounter(event, obligation.wait_counter_type());
      detector.markEventWaveComplete(event);
    }
    const RaceViolation violation{
        test.space, 2,
        0,          test.space == Space::SGPR || test.space == Space::TTMP ? -1 : 3,
        test.write, Dim3d(1, 2, 3),
        event};
    const std::string report =
        formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4);
    EXPECT_NE(report.find(test.resource), std::string::npos);
    EXPECT_NE(report.find(std::string("(") + test.hazard + ")"), std::string::npos);
    EXPECT_NE(report.find(std::string("Conflict: ") + test.operation + " at PC 0x100"),
              std::string::npos);
    EXPECT_NE(report.find("Current access: " + std::string(test.write ? "write" : "read") +
                          " at PC 0x200"),
              std::string::npos);
    EXPECT_NE(report.find("workgroup (1, 2, 3)"), std::string::npos);
    if (test.complete) {
      EXPECT_NE(report.find("Wait requirements satisfied for this operation in wave 1.\n"),
                std::string::npos);
      EXPECT_EQ(report.find("pending counters"), std::string::npos);
    } else {
      EXPECT_NE(
          report.find("Completion for wave " + std::to_string(test.owner) + " is not guaranteed"),
          std::string::npos);
      EXPECT_NE(report.find("pending counters"), std::string::npos);
    }
    EXPECT_EQ(report.find("Waves 1 and 0 have not synchronized this access.\n") !=
                  std::string::npos,
              test.owner == 1);
    EXPECT_EQ(report.find("lane -1"), std::string::npos);
  }
}

TEST(RaceReportDescriptionTest, ReportsOnlyRemainingCounterObligationsAfterAPartialWait) {
  RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  const EventId event = detector.allocateEventId(
      WaveId{0}, testInstruction(0x100), MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
      CounterObligations(WaitCounterType::LOADCNT, MemoryOrderClass::UNORDERED,
                         WaitCounterType::DSCNT),
      MemoryOrderClass::UNORDERED);
  const RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), event};
  EXPECT_NE(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_RDNA4)
                .find("pending counters: loadcnt, dscnt."),
            std::string::npos);
  EXPECT_FALSE(detector.satisfyEventWaitCounter(event, WaitCounterType::DSCNT));
  const std::string report =
      formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_NE(report.find("pending counters: loadcnt."), std::string::npos);
  EXPECT_EQ(report.find("dscnt"), std::string::npos);
}

TEST(RaceReportDescriptionTest, ArchitecturalCounterNamesPreservePendingSubsetChecks) {
  struct Case {
    rj_code_arch_t arch;
    WaitCounterType first;
    WaitCounterType second;
    const char *both;
    const char *remaining;
  };
  const Case cases[] = {
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterType::DSCNT, WaitCounterType::KMCNT, "lgkmcnt",
       "lgkmcnt"},
      {ROCJITSU_CODE_ARCH_RDNA3_5, WaitCounterType::LOADCNT, WaitCounterType::DSCNT,
       "vmcnt, lgkmcnt", "lgkmcnt"},
      {ROCJITSU_CODE_ARCH_RDNA3, WaitCounterType::STORECNT, WaitCounterType::EXPCNT,
       "vscnt, expcnt", "expcnt"},
      {ROCJITSU_CODE_ARCH_RDNA4, WaitCounterType::DSCNT, WaitCounterType::KMCNT, "dscnt, kmcnt",
       "kmcnt"},
      {ROCJITSU_CODE_ARCH_CDNA5, WaitCounterType::ASYNCCNT, WaitCounterType::TENSORCNT,
       "asynccnt, tensorcnt", "tensorcnt"},
      {ROCJITSU_CODE_ARCH_CDNA3, WaitCounterType::VMCNT, WaitCounterType::LGKMCNT, "vmcnt, lgkmcnt",
       "lgkmcnt"},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.both);
    RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
    const EventId event = detector.allocateEventId(
        WaveId{0}, testInstruction(0x100), MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
        CounterObligations(test.first, MemoryOrderClass::UNORDERED, test.second),
        MemoryOrderClass::UNORDERED);
    const RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), event};
    const std::string prefix =
        "Race on VGPR v2 [workgroup (0, 0, 0), wave 0, lane 0]: read-after-write (RAW).\n"
        "  Current access: read at PC 0x200.\n"
        "  Conflict: load into VGPRs at PC 0x100, wave 0.\n"
        "  Completion for wave 0 is not guaranteed; pending counters: ";
    EXPECT_EQ(formatRaceDescription(violation, detector.events(), 0x200, test.arch),
              prefix + test.both + ".\n");
    EXPECT_FALSE(detector.satisfyEventWaitCounter(event, test.first));
    EXPECT_EQ(formatRaceDescription(violation, detector.events(), 0x200, test.arch),
              prefix + test.remaining + ".\n");
  }
}

TEST(RaceReportDescriptionTest, ExactCrossWaveCompletedReadConflict) {
  RaceDetector detector(2, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  const EventId event = detector.allocateEventId(
      WaveId{1}, testInstruction(0x100), MemoryEventType::LDS_TO_VGPR, {2}, 1, 0xf, {},
      CounterObligations(WaitCounterType::LGKMCNT, MemoryOrderClass::LDS), MemoryOrderClass::LDS);
  for (const auto obligation : detector.events().counterObligations(event))
    detector.satisfyEventWaitCounter(event, obligation.wait_counter_type());
  detector.markEventWaveComplete(event);
  const RaceViolation violation{RaceViolation::Space::LDS, 16, 0, 3, true, Dim3d(1, 2, 3), event};
  EXPECT_EQ(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_RDNA3),
            "Race on LDS byte 16 [workgroup (1, 2, 3), wave 0, lane 3]: write-after-read (WAR).\n"
            "  Current access: write at PC 0x200.\n"
            "  Conflict: transfer from LDS into VGPRs at PC 0x100, wave 1.\n"
            "  Wait requirements satisfied for this operation in wave 1.\n"
            "  Waves 1 and 0 have not synchronized this access.\n");
}

TEST(RaceReportDescriptionTest, RejectsUnavailableAndRetiredConflicts) {
  RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), {}};
  EXPECT_THROW(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4),
               std::out_of_range);
  violation.conflictingEvent = detector.allocateEventId(
      WaveId{0}, testInstruction(0x100), MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
      CounterObligations(WaitCounterType::VMCNT, MemoryOrderClass::VMEM), MemoryOrderClass::VMEM);
  EXPECT_THROW(
      formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_INVALID),
      std::invalid_argument);
  detector.retireEvent(violation.conflictingEvent);
  EXPECT_THROW(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4),
               std::logic_error);
}

static_assert(sizeof(InstructionSnapshot) == 32);

} // namespace
