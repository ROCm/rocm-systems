// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "race-detector/race_test_instruction.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/plugins/race_detector/core/common_register.h"
#include "rocjitsu/vm/plugins/race_detector/plugin.h"
#include "util/result.h"

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
#include <string_view>
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
  EXPECT_EQ(formatTrace(trace, text, mark(trace, 0), mark(trace, 3)).value(),
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
      formatTrace(trace, text, MarkedInstruction{snapshot(0x2000, 9, 1), 1, 0}, mark(trace, 1))
          .value(),
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
    const util::FailureOr<std::string> report_result =
        formatTrace(trace, text, mark(trace, 0), mark(trace, trace.size() - 1));
    ASSERT_TRUE(report_result.succeeded());
    const std::string &report = report_result.value();
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
    const util::FailureOr<std::string> report_result =
        formatTrace(trace, text, mark(trace, 0), mark(trace, 199), context);
    ASSERT_TRUE(report_result.succeeded());
    const std::string &report = report_result.value();
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
  const util::FailureOr<std::string> report_result =
      formatTrace(trace, text, mark(trace, 0), mark(trace, 199));
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
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
  const auto disassemble = [](const InstructionSnapshot &) { return std::string("s_nop 0"); };
  EXPECT_EQ(formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 5)).value(),
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
  const util::FailureOr<std::string> report_result =
      formatTrace(trace, text, mark(trace, 0), mark(trace, 7));
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
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
  const auto disassemble = [](const InstructionSnapshot &) { return std::string("s_nop 0"); };
  const util::FailureOr<std::string> short_report_result =
      formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 4);
  ASSERT_TRUE(short_report_result.succeeded());
  const std::string &short_report = short_report_result.value();
  EXPECT_EQ(short_report, "       PC     Instruction\n"
                          "  ==>  0x100  s_nop 0  ; <-- wave 0 lane 0\n"
                          "       0x104  s_nop 0\n"
                          "       ... 4 instructions elided ...\n"
                          "       0x118  s_nop 0\n"
                          "  ==>  0x11c  s_nop 0  ; <-- wave 0 lane 0\n");
  const util::FailureOr<std::string> full_report_result =
      formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 256);
  ASSERT_TRUE(full_report_result.succeeded());
  const std::string &full_report = full_report_result.value();
  EXPECT_NE(full_report.find("6 executions, through PC 0x118\n"), std::string::npos);
  EXPECT_EQ(elided_count(full_report), 0u);
  const util::FailureOr<std::string> odd_report_result =
      formatTrace(trace, disassemble, mark(trace, 0), mark(trace, 7), 5);
  ASSERT_TRUE(odd_report_result.succeeded());
  const std::string &odd_report = odd_report_result.value();
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
  const util::FailureOr<std::string> report_result =
      formatTrace(trace, text, mark(trace, 2), mark(trace, 5));
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
  const std::vector<PrintedRow> printed = rows(report);
  ASSERT_EQ(printed.size(), 4u);
  EXPECT_EQ(printed[0].sequence, 2u);
  EXPECT_EQ(printed.back().sequence, 5u);
}

TEST(RaceReportTraceTest, OtherWaveConflictIsSeparateEvenWhenPcsAndEncodingsMatch) {
  Trace trace;
  trace.push(snapshot(0x100, 0, 1));
  trace.push(snapshot(0x104, 1, 1));
  const util::FailureOr<std::string> report_result =
      formatTrace(trace, text, mark(trace, 0, 1), mark(trace, 1));
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
  const std::vector<PrintedRow> printed = rows(report);
  ASSERT_EQ(printed.size(), 3u);
  EXPECT_NE(report.find("Execution trace below is for wave 0."), std::string::npos);
}

TEST(RaceReportTraceTest, OtherWaveConflictStartsAtThisWavesLatestRunOfThatInstruction) {
  Trace trace;
  for (uint32_t i = 0; i < 40; ++i)
    trace.push(snapshot(0x100 + (i % 20) * 4, i % 20, 1));
  const MarkedInstruction producer{snapshot(0x104, 1, 1), 1, 0};
  const std::vector<PrintedRow> printed =
      rows(formatTrace(trace, text, producer, mark(trace, 39)).value());
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
    const util::FailureOr<std::string> report_result =
        formatTrace(trace, text, producer, mark(trace, 39), context);
    ASSERT_TRUE(report_result.succeeded());
    const std::string &report = report_result.value();
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
  const util::FailureOr<std::string> report_result =
      formatTrace(trace, text, MarkedInstruction{producer, 0, 0}, mark(trace, 255));
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
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
  const util::FailureOr<std::string> report_result = formatTrace({}, text, std::nullopt, current);
  ASSERT_TRUE(report_result.succeeded());
  const std::string &report = report_result.value();
  ASSERT_EQ(rows(report).size(), 1u);
  EXPECT_NE(report.find("outside recorded trace"), std::string::npos);
}

#ifndef NDEBUG
TEST(RaceReportTraceDeathTest, AssertsReversedExecutionOrder) {
  Trace trace;
  trace.push(snapshot(0x100, 0, 1));
  trace.push(snapshot(0x104, 1, 1));
  EXPECT_DEATH(static_cast<void>(formatTrace(trace, text, mark(trace, 1), mark(trace, 0))),
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
        formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Cdna4);
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
    EXPECT_EQ(
        report.find(
            "The detector observed no synchronization between waves 1 and 0 for this access.\n") !=
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
  EXPECT_NE(formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Rdna4)
                .find("pending counters: loadcnt, dscnt."),
            std::string::npos);
  EXPECT_FALSE(detector.satisfyEventWaitCounter(event, WaitCounterType::DSCNT));
  const std::string report =
      formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Rdna4);
  EXPECT_NE(report.find("pending counters: loadcnt."), std::string::npos);
  EXPECT_EQ(report.find("dscnt"), std::string::npos);
}

TEST(RaceReportDescriptionTest, ArchitecturalCounterNamesPreservePendingSubsetChecks) {
  struct Case {
    rocjitsu::IsaFamily arch;
    WaitCounterType first;
    WaitCounterType second;
    const char *both;
    const char *remaining;
  };
  const Case cases[] = {
      {rocjitsu::IsaFamily::Rdna3, WaitCounterType::DSCNT, WaitCounterType::KMCNT, "lgkmcnt",
       "lgkmcnt"},
      {rocjitsu::IsaFamily::Rdna3_5, WaitCounterType::LOADCNT, WaitCounterType::DSCNT,
       "vmcnt, lgkmcnt", "lgkmcnt"},
      {rocjitsu::IsaFamily::Rdna3, WaitCounterType::STORECNT, WaitCounterType::EXPCNT,
       "vscnt, expcnt", "expcnt"},
      {rocjitsu::IsaFamily::Rdna4, WaitCounterType::DSCNT, WaitCounterType::KMCNT, "dscnt, kmcnt",
       "kmcnt"},
      {rocjitsu::IsaFamily::Cdna5, WaitCounterType::ASYNCCNT, WaitCounterType::TENSORCNT,
       "asynccnt, tensorcnt", "tensorcnt"},
      {rocjitsu::IsaFamily::Cdna3, WaitCounterType::VMCNT, WaitCounterType::LGKMCNT,
       "vmcnt, lgkmcnt", "lgkmcnt"},
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
  EXPECT_EQ(formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Rdna3),
            "Race on LDS byte 16 [workgroup (1, 2, 3), wave 0, lane 3]: write-after-read (WAR).\n"
            "  Current access: write at PC 0x200.\n"
            "  Conflict: transfer from LDS into VGPRs at PC 0x100, wave 1.\n"
            "  Wait requirements satisfied for this operation in wave 1.\n"
            "  The detector observed no synchronization between waves 1 and 0 for this access.\n");
}

TEST(RaceReportDescriptionTest, RejectsUnavailableAndRetiredConflicts) {
  RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), {}};
  EXPECT_THROW(
      formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Cdna4),
      std::out_of_range);
  violation.conflictingEvent = detector.allocateEventId(
      WaveId{0}, testInstruction(0x100), MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
      CounterObligations(WaitCounterType::VMCNT, MemoryOrderClass::VMEM), MemoryOrderClass::VMEM);
  EXPECT_THROW(
      formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Invalid),
      std::invalid_argument);
  detector.retireEvent(violation.conflictingEvent);
  EXPECT_THROW(
      formatRaceDescription(violation, detector.events(), 0x200, rocjitsu::IsaFamily::Cdna4),
      std::logic_error);
}

// ----- The barrier a conflicting access crossed in flight -----
//
// A barrier signal waits for no memory counter (CDNA5 ISA 5.6 and 15.3,
// S_BARRIER_SIGNAL), so an LDS access in flight when its wave signals is not
// ordered by that barrier. The detector puts the barrier on the violation when
// such an access races with a wave that has passed it, and the report then
// ends with a line naming the wave, the barrier and the counters to wait for
// before it.

constexpr uint64_t kBarrierPc = 0x180;

/// What one description of a cross-wave LDS race is built from. Wave 1 owns
/// the conflicting event, issued at PC 0x100; wave 0 makes the access at 0x200.
struct BarrierRace {
  MemoryEventType type = MemoryEventType::VGPR_TO_LDS;
  WaitCounterType counter = WaitCounterType::DSCNT;
  MemoryOrderClass order = MemoryOrderClass::LDS;
  std::optional<WaitCounterType> second_counter = std::nullopt;
  LdsAccess access = {};
  rocjitsu::IsaFamily arch = rocjitsu::IsaFamily::Cdna5;
  /// Whether wave 1 has waited for the event since the barrier.
  bool completed = false;
};

/// The description of `race`, with the violation carrying the barrier at
/// kBarrierPc or none, and the barrier's disassembly `instruction` supplied to
/// the formatter.
std::string describeBarrierRace(const BarrierRace &race, bool crossed_barrier,
                                std::string_view instruction = "s_barrier_signal -1") {
  RaceDetector detector(2, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  const EventId event = detector.allocateEventId(
      WaveId{1}, testInstruction(0x100), race.type, /*registers=*/{2}, /*execMask=*/1,
      /*byteMask=*/0xf, /*ldsIntervals=*/{},
      CounterObligations(race.counter, race.order, race.second_counter), race.order,
      /*lastRegisterByteMask=*/0, race.access);
  if (race.completed) {
    for (const auto obligation : detector.events().counterObligations(event))
      detector.satisfyEventWaitCounter(event, obligation.wait_counter_type());
    detector.markEventWaveComplete(event);
  }
  const bool reads_lds =
      race.type == MemoryEventType::LDS_TO_VGPR || race.type == MemoryEventType::LDS_TO_GLOBAL;
  RaceViolation violation{RaceViolation::Space::LDS,
                          16,
                          /*wave=*/0,
                          3,
                          /*isWrite=*/reads_lds,
                          Dim3d(1, 2, 3),
                          event};
  if (crossed_barrier)
    violation.crossedBarrier = testInstruction(kBarrierPc);
  return formatRaceDescription(violation, detector.events(), 0x200, race.arch, instruction);
}

/// The line that names the barrier, for wave 1's access pending on `counters`.
std::string barrierLine(std::string_view counters,
                        std::string_view instruction = " (s_barrier_signal -1)") {
  return std::format("  Wave 1 arrived at the barrier at PC 0x180{} with this access in flight, so "
                     "that barrier does not order it. Wait for {} before the barrier.\n",
                     instruction, counters);
}

TEST(RaceReportBarrierTest, NamesTheBarrierAPendingConflictCrossedInFlight) {
  const BarrierRace race{};
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/true),
            "Race on LDS byte 16 [workgroup (1, 2, 3), wave 0, lane 3]: read-after-write (RAW).\n"
            "  Current access: read at PC 0x200.\n"
            "  Conflict: transfer from VGPRs into LDS at PC 0x100, wave 1.\n"
            "  Completion for wave 1 is not guaranteed; pending counters: dscnt.\n"
            "  The detector observed no synchronization between waves 1 and 0 for this access.\n"
            "  Wave 1 arrived at the barrier at PC 0x180 (s_barrier_signal -1) with this access in "
            "flight, so that barrier does not order it. Wait for dscnt before the barrier.\n");
}

// The line is appended to the description the report has without it.
TEST(RaceReportBarrierTest, AppendsToTheExistingDescription) {
  const BarrierRace race{};
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/true),
            describeBarrierRace(race, /*crossed_barrier=*/false) + barrierLine("dscnt"));
}

// A wait after the signal completes the access for its own wave, and the
// barrier still did not order it for the others.
TEST(RaceReportBarrierTest, NamesTheBarrierAfterTheConflictCompleted) {
  const BarrierRace race{.completed = true};
  const std::string without = describeBarrierRace(race, /*crossed_barrier=*/false);
  ASSERT_TRUE(without.ends_with(
      "  Wait requirements satisfied for this operation in wave 1.\n"
      "  The detector observed no synchronization between waves 1 and 0 for this access.\n"))
      << without;
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/true), without + barrierLine("dscnt"));
}

TEST(RaceReportBarrierTest, NamesAReadThatCrossedTheBarrier) {
  const BarrierRace race{.type = MemoryEventType::LDS_TO_VGPR};
  const std::string without = describeBarrierRace(race, /*crossed_barrier=*/false);
  ASSERT_TRUE(without.starts_with(
      "Race on LDS byte 16 [workgroup (1, 2, 3), wave 0, lane 3]: write-after-read (WAR).\n"))
      << without;
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/true), without + barrierLine("dscnt"));
}

// The counters named are the ones the LDS access completes on, under the
// architecture's own names.
TEST(RaceReportBarrierTest, NamesTheCountersTheLdsAccessCompletesOn) {
  struct Case {
    BarrierRace race;
    const char *counters;
    const char *barrier;
  };
  const Case cases[] = {
      // s_barrier and the combined counter of the targets before GFX12.
      {{.counter = WaitCounterType::LGKMCNT, .arch = rocjitsu::IsaFamily::Cdna4},
       "lgkmcnt",
       "s_barrier"},
      // The LDS access of a FLAT store completes on DScnt (CDNA5 ISA 5.7.1.3);
      // its STOREcnt does not matter to the barrier.
      {{.counter = WaitCounterType::STORECNT,
        .order = MemoryOrderClass::UNORDERED,
        .second_counter = WaitCounterType::DSCNT,
        .access = LdsAccess{.uses_flat_lds = true}},
       "dscnt",
       "s_barrier_signal -1"},
      {{.counter = WaitCounterType::VMCNT,
        .order = MemoryOrderClass::UNORDERED,
        .second_counter = WaitCounterType::LGKMCNT,
        .access = LdsAccess{.uses_flat_lds = true},
        .arch = rocjitsu::IsaFamily::Cdna4},
       "lgkmcnt",
       "s_barrier"},
      // Tensor DMA and async copies complete on their own counters.
      {{.type = MemoryEventType::GLOBAL_TO_LDS,
        .counter = WaitCounterType::TENSORCNT,
        .order = MemoryOrderClass::TENSOR},
       "tensorcnt",
       "s_barrier_signal -1"},
      {{.type = MemoryEventType::LDS_TO_GLOBAL,
        .counter = WaitCounterType::TENSORCNT,
        .order = MemoryOrderClass::TENSOR},
       "tensorcnt",
       "s_barrier_signal -1"},
      {{.type = MemoryEventType::GLOBAL_TO_LDS,
        .counter = WaitCounterType::ASYNCCNT,
        .order = MemoryOrderClass::ASYNC_LOAD},
       "asynccnt",
       "s_barrier_signal -1"},
      {{.type = MemoryEventType::LDS_TO_GLOBAL,
        .counter = WaitCounterType::ASYNCCNT,
        .order = MemoryOrderClass::ASYNC_STORE},
       "asynccnt",
       "s_barrier_signal -1"},
      // Every counter the access completes on is named, in the order of its
      // obligations.
      {{.type = MemoryEventType::GLOBAL_TO_LDS,
        .counter = WaitCounterType::ASYNCCNT,
        .order = MemoryOrderClass::ASYNC_LOAD,
        .second_counter = WaitCounterType::DSCNT},
       "asynccnt, dscnt",
       "s_barrier_signal -1"},
      // Two obligations on the same counter name it once: before GFX12 the DS
      // and constant-load counters are the one LGKMcnt.
      {{.counter = WaitCounterType::LGKMCNT,
        .second_counter = WaitCounterType::DSCNT,
        .arch = rocjitsu::IsaFamily::Cdna4},
       "lgkmcnt",
       "s_barrier"},
      // A direct-to-LDS load completes with the vector-memory loads.
      {{.type = MemoryEventType::GLOBAL_TO_LDS,
        .counter = WaitCounterType::VMCNT,
        .order = MemoryOrderClass::VMEM,
        .arch = rocjitsu::IsaFamily::Cdna4},
       "vmcnt",
       "s_barrier"},
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(std::string(test.counters) + " " + test.barrier);
    EXPECT_EQ(describeBarrierRace(test.race, /*crossed_barrier=*/true, test.barrier),
              describeBarrierRace(test.race, /*crossed_barrier=*/false, test.barrier) +
                  barrierLine(test.counters, std::string(" (") + test.barrier + ")"));
  }
}

// Without the barrier's disassembly the line gives its PC alone.
TEST(RaceReportBarrierTest, OmitsTheInstructionWhenItIsNotSupplied) {
  const BarrierRace race{};
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/true, /*instruction=*/{}),
            describeBarrierRace(race, /*crossed_barrier=*/false, /*instruction=*/{}) +
                barrierLine("dscnt", /*instruction=*/{}));
}

// The barrier a violation names is disassembled through the caller's
// disassembler, and one that names none is not disassembled at all.
TEST(RaceReportBarrierTest, DisassemblesTheBarrierTheViolationNames) {
  std::vector<uint64_t> disassembled;
  const auto disassemble =
      [&](const InstructionSnapshot &instruction) -> util::FailureOr<std::string> {
    disassembled.push_back(instruction.pc());
    return std::string("s_barrier_signal -1");
  };
  RaceViolation violation{
      RaceViolation::Space::LDS, 16, /*wave=*/0, 3, /*isWrite=*/false, Dim3d(1, 2, 3), EventId{0}};
  util::FailureOr<std::string> none = formatCrossedBarrier(violation, disassemble);
  ASSERT_TRUE(none.succeeded());
  EXPECT_EQ(none.value(), "");
  EXPECT_EQ(disassembled, std::vector<uint64_t>{});

  violation.crossedBarrier = testInstruction(kBarrierPc);
  util::FailureOr<std::string> named = formatCrossedBarrier(violation, disassemble);
  ASSERT_TRUE(named.succeeded());
  EXPECT_EQ(named.value(), "s_barrier_signal -1");
  EXPECT_EQ(disassembled, std::vector<uint64_t>{kBarrierPc});
}

// A barrier that does not disassemble is a failure, which the plugin reports
// in place of the race report, as it does for the trace.
TEST(RaceReportBarrierTest, ReportsABarrierThatDoesNotDisassemble) {
  RaceViolation violation{
      RaceViolation::Space::LDS, 16, /*wave=*/0, 3, /*isWrite=*/false, Dim3d(1, 2, 3), EventId{0}};
  violation.crossedBarrier = testInstruction(kBarrierPc);
  const util::FailureOr<std::string> failed = formatCrossedBarrier(
      violation, [](const InstructionSnapshot &) -> util::FailureOr<std::string> {
        return util::Result::failure();
      });
  EXPECT_TRUE(failed.failed());
}

// A violation that carries no barrier gets no line, whatever the caller
// supplies.
TEST(RaceReportBarrierTest, NoLineWithoutABarrierOnTheViolation) {
  const BarrierRace race{};
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/false, "s_barrier_signal -1"),
            describeBarrierRace(race, /*crossed_barrier=*/false, /*instruction=*/{}));
  EXPECT_EQ(describeBarrierRace(race, /*crossed_barrier=*/false).find("arrived at the barrier"),
            std::string::npos);
}

static_assert(sizeof(InstructionSnapshot) == 32);

} // namespace
