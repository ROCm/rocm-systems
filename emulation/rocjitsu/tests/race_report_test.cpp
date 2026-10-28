// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/race_detector/plugin.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <format>
#include <limits>
#include <random>
#include <regex>
#include <sstream>

namespace {

using namespace rocjitsu::plugins::race_detector;
using rocjitsu::amdgpu::WaitCounterType;

using Trace = RingBuffer<InstructionSnapshot, 256>;

MarkedInstruction mark(const Trace &trace, size_t index, int wave = 0) {
  return {{trace[index], trace.sequence(index)}, wave, 0};
}

std::string text(const InstructionSnapshot &instruction) {
  return std::format("instruction {}", instruction.words[0]);
}

struct PrintedRow {
  bool marked;
  std::string transition;
  uint64_t pc;
  size_t sequence;
};

std::vector<PrintedRow> rows(const std::string &report) {
  const std::regex pattern(R"(^\s*(==>\s*)?0x([0-9a-f]+)\s+instruction ([0-9]+))");
  const std::regex marker(R"(^\s*--- (jump forward|jump backward|same PC executed again) ---$)");
  std::vector<PrintedRow> result;
  std::istringstream lines(report);
  std::string line, transition;
  while (std::getline(lines, line)) {
    EXPECT_EQ(line.find("Trace window:"), std::string::npos);
    EXPECT_EQ(line.find("Same-wave instruction distance:"), std::string::npos);
    std::smatch match;
    if (std::regex_search(line, match, marker)) {
      EXPECT_TRUE(transition.empty());
      transition = match[1];
    } else if (std::regex_search(line, match, pattern)) {
      result.push_back({match[1].matched, transition, std::stoull(match[2], nullptr, 16),
                        std::stoull(match[3])});
      transition.clear();
    }
  }
  EXPECT_TRUE(transition.empty());
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

TEST(RaceReportTraceTest, VariableWidthSequentialInstructionsHaveNoJumpMarkers) {
  Trace trace;
  trace.push({.pc = 0x100, .words = {0}, .word_count = 2});
  trace.push({.pc = 0x108, .words = {1}, .word_count = 1});
  trace.push({.pc = 0x10c, .words = {2}, .word_count = 3});
  trace.push({.pc = 0x118, .words = {3}, .word_count = 4});
  trace.push({.pc = 0x128, .words = {4}, .word_count = 1});
  const auto report = formatTrace(trace, text, mark(trace, 0), mark(trace, 4));
  const auto printed = rows(report);
  ASSERT_EQ(printed.size(), 5u);
  for (const auto &row : printed)
    EXPECT_TRUE(row.transition.empty());
  EXPECT_EQ(report.find("PC diff"), std::string::npos);
  EXPECT_EQ(elided_count(report), 0u);
}

TEST(RaceReportTraceTest, ExactForwardBackwardAndRepeatedPcSeparators) {
  Trace trace;
  for (uint32_t i = 0; auto pc : {0x100u, 0x120u, 0x108u, 0x108u})
    trace.push({.pc = pc, .words = {i++}, .word_count = 1});
  EXPECT_EQ(formatTrace(trace, text, mark(trace, 0), mark(trace, 3)),
            "       PC  Instruction\n"
            "  ==>  0x100  instruction 0  ; <-- wave 0 lane 0\n"
            "       --- jump forward ---\n"
            "       0x120  instruction 1\n"
            "       --- jump backward ---\n"
            "       0x108  instruction 2\n"
            "       --- same PC executed again ---\n"
            "  ==>  0x108  instruction 3  ; <-- wave 0 lane 0\n");
}

TEST(RaceReportTraceTest, FullAddressRangeAndSelfLoopHaveCorrectJumpMarkers) {
  Trace trace;
  trace.push({.pc = 0, .words = {0}, .word_count = 1});
  trace.push({.pc = UINT64_MAX, .words = {1}, .word_count = 1});
  trace.push({.pc = 0, .words = {2}, .word_count = 1});
  trace.push({.pc = 0, .words = {3}, .word_count = 1});
  const auto report = formatTrace(trace, text, mark(trace, 0), mark(trace, 3));
  const auto printed = rows(report);
  ASSERT_EQ(printed.size(), 4u);
  EXPECT_TRUE(printed[0].transition.empty());
  EXPECT_EQ(printed[1].transition, "jump forward");
  EXPECT_EQ(printed[2].transition, "jump backward");
  EXPECT_EQ(printed[3].transition, "same PC executed again");
}

TEST(RaceReportTraceTest, ContextLimitsAndRingWrapPreserveRowsAndAccountForEveryGap) {
  for (size_t length : {1u, 2u, 31u, 32u, 33u, 127u, 128u, 129u, 255u, 256u, 257u, 513u}) {
    Trace trace;
    for (uint32_t i = 0; i < length; ++i)
      trace.push({.pc = 0x100 + i * 4, .words = {i}, .word_count = 1});
    for (size_t limit : {2u, 3u, 32u, 127u, 128u, 129u, 255u, 256u}) {
      SCOPED_TRACE(std::format("length={} limit={}", length, limit));
      const auto report =
          formatTrace(trace, text, mark(trace, 0), mark(trace, trace.size() - 1), limit);
      const auto printed = rows(report);
      ASSERT_EQ(printed.size(), std::min(trace.size(), limit));
      ASSERT_TRUE(printed.front().marked);
      ASSERT_TRUE(printed.back().marked);
      EXPECT_EQ(printed.front().sequence, length - trace.size());
      EXPECT_EQ(printed.back().sequence, length - 1);
      for (const auto &row : printed)
        EXPECT_TRUE(row.transition.empty());
      EXPECT_EQ(printed.size() + elided_count(report), trace.size());
    }
  }
}

TEST(RaceReportTraceTest, DefaultShows128AndFullHistoryOptionShows256) {
  Trace trace;
  for (uint32_t i = 0; i < 256; ++i)
    trace.push({.pc = 0x100 + i * 4, .words = {i}, .word_count = 1});
  EXPECT_EQ(rows(formatTrace(trace, text, mark(trace, 0), mark(trace, 255))).size(), 128u);
  EXPECT_EQ(rows(formatTrace(trace, text, mark(trace, 0), mark(trace, 255), 256)).size(), 256u);
}

TEST(RaceReportTraceTest, ElisionNeverHidesSynchronizationOrEitherSideOfAJump) {
  Trace trace;
  for (uint32_t i = 0; i < 200; ++i)
    trace.push({.pc = (i < 100 ? 0x1000u : 0x800u) + i * 4,
                .words = {i},
                .word_count = 1,
                .keep_in_trace = i == 60 || i == 140});
  const auto report = formatTrace(trace, text, mark(trace, 0), mark(trace, 199), 32);
  const auto printed = rows(report);
  for (size_t required : {0u, 60u, 99u, 100u, 140u, 199u})
    EXPECT_TRUE(
        std::ranges::any_of(printed, [=](const auto &row) { return row.sequence == required; }));
  EXPECT_NE(report.find("instruction 99\n       --- jump backward ---\n       0x"),
            std::string::npos);
  EXPECT_EQ(printed.size() + elided_count(report), 200u);
}

TEST(RaceReportTraceTest, RepeatedPcSelectsExactExecutionWithoutAJumpBeforeTheWindow) {
  Trace trace;
  for (uint32_t i = 0; i < 6; ++i)
    trace.push({.pc = 0x100 + (i % 2) * 4, .words = {i}, .word_count = 1});
  const auto report = formatTrace(trace, text, mark(trace, 2), mark(trace, 5));
  const auto printed = rows(report);
  ASSERT_EQ(printed.size(), 4u);
  EXPECT_EQ(printed[0].sequence, 2u);
  EXPECT_TRUE(printed[0].transition.empty());
  EXPECT_EQ(printed[2].transition, "jump backward");
  EXPECT_EQ(printed.back().sequence, 5u);
}

TEST(RaceReportTraceTest, OtherWaveConflictIsSeparateEvenWhenPcsAndSequencesMatch) {
  Trace trace;
  trace.push({.pc = 0x100, .words = {0}, .word_count = 1});
  trace.push({.pc = 0x104, .words = {1}, .word_count = 1});
  const auto report = formatTrace(trace, text, mark(trace, 0, 1), mark(trace, 1));
  const auto printed = rows(report);
  ASSERT_EQ(printed.size(), 3u);
  for (const auto &row : printed)
    EXPECT_TRUE(row.transition.empty());
  EXPECT_NE(report.find("outside this wave's trace"), std::string::npos);
}

TEST(RaceReportTraceTest, ExpiredProducerRetainsIdentityAndMissingHistoryWithoutInventingAJump) {
  Trace trace;
  const InstructionSnapshot producer{.pc = 0x100, .words = {0}, .word_count = 1};
  trace.push(producer);
  for (uint32_t i = 1; i < 300; ++i)
    trace.push({.pc = 0x100 + i * 4, .words = {i}, .word_count = 1});
  const auto report =
      formatTrace(trace, text, MarkedInstruction{{producer, 0}, 0, 0}, mark(trace, 255));
  const auto printed = rows(report);
  ASSERT_GE(printed.size(), 2u);
  EXPECT_EQ(printed[0].sequence, 0u);
  EXPECT_EQ(printed[1].sequence, 44u);
  EXPECT_TRUE(printed[1].transition.empty());
  EXPECT_NE(report.find("before trace window"), std::string::npos);
  EXPECT_NE(report.find("43 earlier instructions not recorded"), std::string::npos);
  EXPECT_EQ(printed.back().sequence, 299u);
  EXPECT_EQ(printed.size() - 1 + elided_count(report), trace.size());
}

TEST(RaceReportTraceTest, EmptyAndMissingCurrentTraceDoNotFabricateExecutionContinuity) {
  const MarkedInstruction current{{{.pc = 0x104, .words = {1}, .word_count = 1}}, 0, 0};
  const auto report = formatTrace({}, text, std::nullopt, current);
  ASSERT_EQ(rows(report).size(), 1u);
  EXPECT_TRUE(rows(report)[0].transition.empty());
  EXPECT_NE(report.find("outside recorded trace"), std::string::npos);
}

TEST(RaceReportTraceTest, RejectsInvalidLimitAndReversedExecutionOrder) {
  Trace trace;
  trace.push({.pc = 0x100, .words = {0}, .word_count = 1});
  trace.push({.pc = 0x104, .words = {1}, .word_count = 1});
  for (size_t limit : {0u, 1u, 257u})
    EXPECT_THROW(formatTrace(trace, text, mark(trace, 0), mark(trace, 1), limit),
                 std::invalid_argument);
  EXPECT_THROW(formatTrace(trace, text, mark(trace, 1), mark(trace, 0)), std::invalid_argument);
}

TEST(RaceReportTraceTest, SeededExecutionPathsPreserveEveryControlTransferAndMarkOnlyActualJumps) {
  std::mt19937 generator(14330);
  for (size_t trial = 0; trial < 100; ++trial) {
    Trace trace;
    std::vector<uint64_t> pcs;
    uint64_t pc = 0x10000;
    uint32_t previous_width = 1;
    for (uint32_t i = 0; i < 400; ++i) {
      if (i)
        pc = generator() % 8 ? pc + previous_width * 4 : 0x10000 + generator() % 1024;
      pcs.push_back(pc);
      previous_width = 1 + generator() % 4;
      trace.push({.pc = pc,
                  .words = {i},
                  .word_count = previous_width,
                  .keep_in_trace = generator() % 13 == 0});
    }
    const auto report =
        formatTrace(trace, text, mark(trace, 0), mark(trace, 255), 2 + generator() % 255);
    const auto printed = rows(report);
    ASSERT_FALSE(printed.empty());
    EXPECT_EQ(printed.size() + elided_count(report), 256u);
    size_t previous_sequence = 143;
    for (const auto &row : printed) {
      ASSERT_GT(row.sequence, previous_sequence);
      EXPECT_EQ(row.pc, pcs[row.sequence]);
      if (row.sequence == 144) {
        EXPECT_TRUE(row.transition.empty());
      } else {
        const size_t index = row.sequence - 144;
        const auto previous = trace[index - 1];
        const bool jump = row.pc != previous.pc + previous.word_count * 4;
        const std::string expected = !jump                  ? ""
                                     : row.pc < previous.pc ? "jump backward"
                                     : row.pc > previous.pc ? "jump forward"
                                                            : "same PC executed again";
        EXPECT_EQ(row.transition, expected);
      }
      previous_sequence = row.sequence;
    }
    for (size_t i = 1; i < trace.size(); ++i) {
      const bool jump = trace[i].pc != trace[i - 1].pc + trace[i - 1].word_count * 4;
      if (jump || trace[i].keep_in_trace)
        EXPECT_TRUE(std::ranges::any_of(
            printed, [&](const auto &row) { return row.sequence == trace.sequence(i); }));
      if (jump)
        EXPECT_TRUE(std::ranges::any_of(
            printed, [&](const auto &row) { return row.sequence == trace.sequence(i - 1); }));
    }
  }
}

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
       "global memory into VGPRs"},
      {Space::VGPR, MemoryEventType::LDS_TO_VGPR, true, 0, false, "VGPR v2", "WAW",
       "LDS into VGPRs"},
      {Space::SGPR, MemoryEventType::GLOBAL_TO_SGPR, false, 0, false, "SGPR s2", "RAW",
       "global memory into SGPRs"},
      {Space::TTMP, MemoryEventType::GLOBAL_TO_TTMP, true, 0, false, "TTMP ttmp2", "WAW",
       "global memory into TTMPs"},
      {Space::LDS, MemoryEventType::VGPR_TO_LDS, false, 1, false, "LDS byte 2", "RAW",
       "VGPRs into LDS"},
      {Space::LDS, MemoryEventType::VGPR_TO_LDS, false, 1, true, "LDS byte 2", "RAW",
       "VGPRs into LDS"},
      {Space::LDS, MemoryEventType::LDS_TO_VGPR, true, 1, false, "LDS byte 2", "WAR",
       "LDS into VGPRs"},
      {Space::LDS, MemoryEventType::LDS_TO_VGPR, true, 1, true, "LDS byte 2", "WAR",
       "LDS into VGPRs"},
      {Space::LDS, MemoryEventType::GLOBAL_TO_LDS, false, 0, false, "LDS byte 2", "RAW",
       "global memory into LDS"},
      {Space::LDS, MemoryEventType::GLOBAL_TO_LDS, false, 1, true, "LDS byte 2", "RAW",
       "global memory into LDS"},
  };
  for (const auto &test : cases) {
    SCOPED_TRACE(test.operation);
    RaceDetector detector(2, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
    const auto event = detector.allocateEventId(WaveId{test.owner}, 0x100, test.event_type, {2}, 1);
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
    const auto report =
        formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4);
    EXPECT_NE(report.find(test.resource), std::string::npos);
    EXPECT_NE(report.find(std::string("(") + test.hazard + ")"), std::string::npos);
    EXPECT_NE(report.find(test.operation), std::string::npos);
    EXPECT_NE(report.find("Current access: " + std::string(test.write ? "write" : "read") +
                          " at PC 0x200"),
              std::string::npos);
    EXPECT_NE(report.find("workgroup (1, 2, 3)"), std::string::npos);
    if (test.complete) {
      EXPECT_NE(report.find("Completed for wave 1, but not synchronized with wave 0"),
                std::string::npos);
      EXPECT_EQ(report.find("pending counters"), std::string::npos);
    } else {
      EXPECT_NE(
          report.find("Completion for wave " + std::to_string(test.owner) + " is not guaranteed"),
          std::string::npos);
      EXPECT_NE(report.find("pending counters"), std::string::npos);
    }
    EXPECT_EQ(report.find("lane -1"), std::string::npos);
  }
}

TEST(RaceReportDescriptionTest, ReportsOnlyRemainingCounterObligationsAfterAPartialWait) {
  RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  const auto event = detector.allocateEventId(WaveId{0}, 0x100, MemoryEventType::GLOBAL_TO_VGPR,
                                              {2}, 1, 0xf, {}, WaitCounterType::LOADCNT,
                                              MemoryOrderClass::UNORDERED, WaitCounterType::DSCNT);
  const RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), event};
  EXPECT_NE(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_RDNA4)
                .find("pending counters: loadcnt, dscnt."),
            std::string::npos);
  EXPECT_FALSE(detector.satisfyEventWaitCounter(event, WaitCounterType::DSCNT));
  const auto report =
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
    const auto event =
        detector.allocateEventId(WaveId{0}, 0x100, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1, 0xf, {},
                                 test.first, MemoryOrderClass::UNORDERED, test.second);
    const RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), event};
    const std::string prefix =
        "Race on VGPR v2 [workgroup (0, 0, 0), wave 0, lane 0]: read-after-write (RAW).\n"
        "  Current access: read at PC 0x200.\n"
        "  Conflict: load from global memory into VGPRs at PC 0x100, wave 0.\n"
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
  const auto event =
      detector.allocateEventId(WaveId{1}, 0x100, MemoryEventType::LDS_TO_VGPR, {2}, 1);
  for (const auto obligation : detector.events().counterObligations(event))
    detector.satisfyEventWaitCounter(event, obligation.wait_counter_type());
  detector.markEventWaveComplete(event);
  const RaceViolation violation{RaceViolation::Space::LDS, 16, 0, 3, true, Dim3d(1, 2, 3), event};
  EXPECT_EQ(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_RDNA3),
            "Race on LDS byte 16 [workgroup (1, 2, 3), wave 0, lane 3]: write-after-read (WAR).\n"
            "  Current access: write at PC 0x200.\n"
            "  Conflict: load from LDS into VGPRs at PC 0x100, wave 1.\n"
            "  Completed for wave 1, but not synchronized with wave 0.\n");
}

TEST(RaceReportDescriptionTest, RejectsUnavailableAndRetiredConflicts) {
  RaceDetector detector(1, 4, 4, Dim3d(0), [](RaceViolation) {}, {}, 65536);
  RaceViolation violation{RaceViolation::Space::VGPR, 2, 0, 0, false, Dim3d(0), {}};
  EXPECT_THROW(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4),
               std::out_of_range);
  violation.conflictingEvent =
      detector.allocateEventId(WaveId{0}, 0x100, MemoryEventType::GLOBAL_TO_VGPR, {2}, 1);
  EXPECT_THROW(
      formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_INVALID),
      std::invalid_argument);
  detector.retireEvent(violation.conflictingEvent);
  EXPECT_THROW(formatRaceDescription(violation, detector.events(), 0x200, ROCJITSU_CODE_ARCH_CDNA4),
               std::logic_error);
}

TEST(RaceReportConfigTest, AcceptsDefaultAndBoundaryContextLimits) {
  for (const char *config :
       {static_cast<const char *>(nullptr), "{}", "{\"trace_context_instructions\":2}",
        "{\"trace_context_instructions\":128}", "{\"trace_context_instructions\":256}"})
    EXPECT_NO_THROW(RaceDetectorPlugin plugin(config));
}

TEST(RaceReportConfigTest, RejectsMalformedObjectsUnknownKeysAndNonIntegerOrOutOfRangeLimits) {
  for (const char *config :
       {"{", "[]", "true", "null", "{\"max_trace_instruction\":256}",
        "{\"trace_context_instructions\":null}", "{\"trace_context_instructions\":true}",
        "{\"trace_context_instructions\":\"128\"}", "{\"trace_context_instructions\":2.5}",
        "{\"trace_context_instructions\":-1}", "{\"trace_context_instructions\":0}",
        "{\"trace_context_instructions\":1}", "{\"trace_context_instructions\":257}",
        "{\"trace_context_instructions\":18446744073709551615}"}) {
    SCOPED_TRACE(config);
    EXPECT_THROW(RaceDetectorPlugin plugin(config), std::invalid_argument);
  }
}

static_assert(sizeof(InstructionSnapshot) == 32);

} // namespace
