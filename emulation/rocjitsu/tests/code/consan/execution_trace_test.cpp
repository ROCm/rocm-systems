// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "capture_fixture.h"
#include "rocjitsu/code/consan/analysis/analyze.h"
#include "rocjitsu/code/consan/analysis/execution_trace.h"
#include "rocjitsu/code/consan/encoding.h"
#include "util/diagnostic.h"

#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

namespace rocjitsu::consan {
namespace {

class ConSanExecutionTraceAnalysis : public ::testing::Test {
protected:
  Capture make_capture(uint32_t groups = 16, uint32_t threads = 32) {
    Capture capture;
    capture.header = test::trace_capture_header();
    const Manifest &facts = test::manifest(capture.header);
    capture.executions.push_back(test::execution(
        facts, 1, 1, {{groups * threads, 1, 1}, {threads, 1, 1}}, ExecutionMode::Application));
    std::vector<uint8_t> &report = capture.executions[0].report;
    const uint32_t waves = (threads + 31) / 32;
    for (uint32_t index = 0; index < groups * waves; ++index) {
      const size_t row = kReportHeaderBytes + index * kRowHeaderBytes;
      wire::write_u64(report, row + wire::kRowSerial, 1);
      wire::write_u32(report, row + wire::kRowEntry, facts.entry.id);
      wire::write_u32(report, row + wire::kRowWave, index);
      wire::write_u32(report, row + wire::kRowStatus, wire::kRowEntered | wire::kRowExited);
      wire::write_u32(report, row + wire::kTraceRowWorkgroup, index / waves);
      wire::write_u32(report, row + wire::kTraceRowLocalWave, index % waves);
      wire::write_u32(report, row + wire::kTraceRowEntries, 1);
      wire::write_u32(report, row + wire::kTraceRowExits, 1);
    }
    return capture;
  }

  FailureOr<Capture> roundtrip(const Capture &capture) {
    CaptureEncoder encoder;
    const FailureOr<std::vector<uint8_t>> prefix = encoder.start(capture.header, error_.emitter());
    if (prefix.failed())
      return Result::failure();
    std::vector<uint8_t> bytes = prefix.value();
    for (const CapturedExecution &execution : capture.executions) {
      const FailureOr<std::vector<uint8_t>> frame = encoder.append(execution, error_.emitter());
      if (frame.failed())
        return Result::failure();
      bytes.insert(bytes.end(), frame.value().begin(), frame.value().end());
    }
    const FailureOr<std::vector<uint8_t>> end = encoder.finish(error_.emitter());
    if (end.failed())
      return Result::failure();
    bytes.insert(bytes.end(), end.value().begin(), end.value().end());
    encoded_ = bytes;
    return decode_capture(bytes, error_.emitter());
  }
  util::StringDiagnostic error_;
  std::vector<uint8_t> encoded_;
};

TEST_F(ConSanExecutionTraceAnalysis, CompleteBoundedGroupsAndPartialWavesRoundTrip) {
  for (const uint32_t groups : {1u, 16u, 512u}) {
    for (const uint32_t threads : {32u, 33u, 64u}) {
      const FailureOr<Capture> capture = roundtrip(make_capture(groups, threads));
      ASSERT_TRUE(capture.succeeded()) << error_.message();
      const FailureOr<ExecutionTraceResult> result =
          analyze_execution_trace_capture(capture.value(), error_.emitter());
      ASSERT_TRUE(result.succeeded()) << error_.message();
      EXPECT_TRUE(result.value().transport_complete);
      EXPECT_EQ(result.value().expected_rows, groups * ((threads + 31) / 32));
      EXPECT_EQ(result.value().completed_rows, result.value().expected_rows);
      EXPECT_EQ(result.value().entry_events, result.value().completed_rows);
      EXPECT_EQ(result.value().exit_events, result.value().completed_rows);
      const std::string json = render_execution_trace(result.value());
      EXPECT_NE(json.find("\"lds_analysis\":\"not_performed\""), std::string::npos);
      for (const char *forbidden :
           {"verdict", "coverage_complete", "cross_wave", "collision", "findings"})
        EXPECT_EQ(json.find(forbidden), std::string::npos);
      EXPECT_TRUE(analyze_capture(capture.value(), {}, error_.emitter()).failed());
    }
  }
}

TEST_F(ConSanExecutionTraceAnalysis, EveryGpuRowFieldAndBoundaryMultiplicityIsChecked) {
  const Capture saved = make_capture(16, 33);
  const std::array<std::pair<size_t, uint32_t>, 18> fields = {{{wire::kRowSerial, 0},
                                                               {wire::kRowEntry, 99},
                                                               {wire::kRowWave, 1},
                                                               {wire::kRowStatus, 0},
                                                               {wire::kRowStatus, 1},
                                                               {wire::kRowStatus, 2},
                                                               {wire::kRowStatus, 7},
                                                               {wire::kRowLoss, 1},
                                                               {wire::kRowFinalSequence, 1},
                                                               {wire::kTraceRowWorkgroup, 1},
                                                               {wire::kTraceRowLocalWave, 1},
                                                               {wire::kTraceRowEntries, 0},
                                                               {wire::kTraceRowEntries, 2},
                                                               {wire::kTraceRowExits, 0},
                                                               {wire::kTraceRowExits, 2},
                                                               {wire::kTraceRowReservedBegin, 1},
                                                               {60, 1},
                                                               {wire::kRowSerial + 4, 1}}};
  for (const auto &[offset, value] : fields) {
    Capture changed = saved;
    wire::write_u32(changed.executions[0].report, kReportHeaderBytes + offset, value);
    const FailureOr<Capture> decoded = roundtrip(changed);
    ASSERT_TRUE(decoded.succeeded()) << offset << ' ' << error_.message();
    const FailureOr<ExecutionTraceResult> result =
        analyze_execution_trace_capture(decoded.value(), error_.emitter());
    ASSERT_TRUE(result.succeeded()) << error_.message();
    EXPECT_FALSE(result.value().transport_complete) << offset;
    EXPECT_EQ(result.value().completed_rows, result.value().expected_rows - 1) << offset;
    EXPECT_EQ(result.value().executions[0].integrity_events, 1u) << offset;
  }
}

TEST_F(ConSanExecutionTraceAnalysis, CorruptParentCannotCertifyChildRows) {
  const Capture saved = make_capture();
  const std::array<std::pair<size_t, uint32_t>, 8> fields = {{{wire::kReportSerial, 0},
                                                              {wire::kReportSerial + 4, 1},
                                                              {wire::kReportEntry, 99},
                                                              {wire::kReportRows, 1},
                                                              {wire::kReportRowStride, 128},
                                                              {wire::kReportSites, 1},
                                                              {wire::kReportReservedBegin, 1},
                                                              {60, 1}}};
  for (const auto &[offset, value] : fields) {
    Capture changed = saved;
    wire::write_u32(changed.executions[0].report, offset, value);
    const FailureOr<Capture> decoded = roundtrip(changed);
    ASSERT_TRUE(decoded.succeeded());
    const FailureOr<ExecutionTraceResult> result =
        analyze_execution_trace_capture(decoded.value(), error_.emitter());
    ASSERT_TRUE(result.succeeded());
    EXPECT_FALSE(result.value().transport_complete);
    EXPECT_EQ(result.value().completed_rows, 0u);
  }
}

TEST_F(ConSanExecutionTraceAnalysis, InvalidTypedEnvelopesRejectBeforeInspectingRows) {
  const Capture saved = make_capture();
  for (unsigned kind = 0; kind < 11; ++kind) {
    Capture changed = saved;
    CapturedExecution &execution = changed.executions[0];
    switch (kind) {
    case 0:
      execution.completion = kCompletionVisibleForwarded;
      break;
    case 1:
      execution.serial = 2;
      break;
    case 2:
      execution.entry_key = 999;
      break;
    case 3:
      execution.storage_slot = 0;
      break;
    case 4:
      execution.storage_generation = 0;
      break;
    case 5:
      execution.arguments.pop_back();
      break;
    case 6:
      execution.report.pop_back();
      break;
    case 7:
      wire::write_u64(execution.context, wire::kContextSerial, 2);
      break;
    case 8:
      changed.header.execution_conditions = kControlledExecutionConditions;
      break;
    case 9:
      changed.header.run_identity = {};
      break;
    case 10:
      execution.dimensions.grid[0] = 513;
      break;
    }
    EXPECT_TRUE(analyze_execution_trace_capture(changed, error_.emitter()).failed()) << kind;
    EXPECT_TRUE(roundtrip(changed).failed()) << kind;
  }
  Capture repeated = saved;
  repeated.executions.push_back(saved.executions[0]);
  repeated.executions[1].serial = 2;
  EXPECT_TRUE(analyze_execution_trace_capture(repeated, error_.emitter()).failed());
  EXPECT_TRUE(analyze_execution_trace_capture({.header = test::capture_header(), .executions = {}},
                                              error_.emitter())
                  .failed());
}

TEST_F(ConSanExecutionTraceAnalysis, BarrierAndLoopAnalysisRemainIndependentOfExecutionTransport) {
  for (const ScopeKind scope :
       {ScopeKind::NativeLdsWorkgroupBarrier, ScopeKind::NativeLdsWorkgroupBarrierLoop}) {
    Capture capture;
    capture.header = test::capture_header();
    Manifest &facts = test::manifest(capture.header);
    facts.target = 1250;
    facts.scope = scope;
    facts.entry.workgroup_id_scalar = kWorkgroupIdTtmp9Selector;
    facts.barriers = {{.original_drain = 32,
                       .original_signal = 36,
                       .original_wait = 40,
                       .prepared_signal = 320,
                       .prepared_wait = 324,
                       .completion_probe = 328,
                       .probe_size = 32}};
    test::refresh_image(capture.header);
    EXPECT_EQ(profile_options_digest(scope), mvp_options_digest());
    EXPECT_EQ(capture_execution_conditions(ExecutionMode::Controlled),
              kControlledExecutionConditions);
    EXPECT_EQ(completion_evidence(ExecutionMode::Controlled), kCompletionVisibleForwarded);
    capture.executions.push_back(test::execution(facts, 1, 1, {{256, 1, 1}, {64, 1, 1}}));
    const FailureOr<Capture> decoded = roundtrip(capture);
    ASSERT_TRUE(decoded.succeeded()) << error_.message();
    const FailureOr<LdsRaceResult> lds = analyze_capture(decoded.value(), {}, error_.emitter());
    ASSERT_TRUE(lds.succeeded()) << error_.message();
    EXPECT_EQ(lds.value().scope, scope);
    ASSERT_EQ(lds.value().executions.size(), 1u);
    EXPECT_EQ(lds.value().executions.front().workgroup_count, 4u);
    EXPECT_EQ(lds.value().executions.front().waves_per_workgroup, 2u);
    EXPECT_TRUE(analyze_execution_trace_capture(decoded.value(), error_.emitter()).failed());
    capture.header.execution_conditions = kApplicationExecutionConditions;
    EXPECT_TRUE(roundtrip(capture).failed());
    capture.executions.front().completion = kCompletionVisible;
    const FailureOr<Capture> application = roundtrip(capture);
    ASSERT_TRUE(application.succeeded()) << error_.message();
    const FailureOr<LdsRaceResult> application_lds =
        analyze_capture(application.value(), {}, error_.emitter());
    ASSERT_TRUE(application_lds.succeeded()) << error_.message();
    EXPECT_EQ(application_lds.value().scope, scope);
    EXPECT_EQ(application_lds.value().execution_conditions, kApplicationExecutionConditions);
    capture.header.execution_conditions = kControlledExecutionConditions;
    EXPECT_TRUE(roundtrip(capture).failed());
  }
}

TEST_F(ConSanExecutionTraceAnalysis, ControlledTransportRetainsForwardingEvidenceForTrace) {
  Capture capture = make_capture();
  capture.header.execution_conditions = capture_execution_conditions(ExecutionMode::Controlled);
  capture.executions.front().completion = completion_evidence(ExecutionMode::Controlled);
  const FailureOr<Capture> decoded = roundtrip(capture);
  ASSERT_TRUE(decoded.succeeded()) << error_.message();
  const FailureOr<ExecutionTraceResult> result =
      analyze_execution_trace_capture(decoded.value(), error_.emitter());
  ASSERT_TRUE(result.succeeded()) << error_.message();
  EXPECT_TRUE(result.value().transport_complete);
  EXPECT_EQ(result.value().completed_rows, 16u);
}

TEST_F(ConSanExecutionTraceAnalysis, EmptyExecutionMembershipCannotCertifyTransport) {
  Capture capture;
  capture.header = test::trace_capture_header();
  const FailureOr<ExecutionTraceResult> result = analyze_execution_trace_capture(capture);
  ASSERT_TRUE(result.succeeded());
  EXPECT_FALSE(result.value().transport_complete);
}

TEST_F(ConSanExecutionTraceAnalysis, JsonNamesUseSharedEscapingAndUtf8Validation) {
  Capture capture = make_capture();
  test::manifest(capture.header).entry.name = "quote\" slash\\ line\n tab\t";
  test::refresh_image(capture.header);
  const FailureOr<ExecutionTraceResult> result = analyze_execution_trace_capture(capture);
  ASSERT_TRUE(result.succeeded());
  EXPECT_NE(
      render_execution_trace(result.value()).find("quote\\\" slash\\\\ line\\u000a tab\\u0009"),
      std::string::npos);
  test::manifest(capture.header).entry.name = std::string(1, static_cast<char>(0xff));
  test::refresh_image(capture.header);
  EXPECT_TRUE(analyze_execution_trace_capture(capture).failed());
}

#if defined(CONSAN_ANALYZER_FIXTURE)
TEST_F(ConSanExecutionTraceAnalysis, FreshCliSeparatesTransportGapsAndInvalidContainers) {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "consan-trace-cli-XXXXXX").string();
  std::vector<char> temporary(pattern.begin(), pattern.end());
  temporary.push_back(0);
  ASSERT_NE(::mkdtemp(temporary.data()), nullptr);
  const std::filesystem::path directory = temporary.data();
  const std::filesystem::path capture_path = directory / "capture.bin";
  const std::filesystem::path output_path = directory / "analysis.json";
  const auto execute = [&](int expected_status) {
    std::ofstream file(capture_path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(encoded_.data()), encoded_.size());
    file.close();
    ASSERT_TRUE(file);
    const pid_t child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      const int output = ::open(output_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
      if (output < 0 || ::dup2(output, STDOUT_FILENO) < 0 || ::chdir(directory.c_str()) != 0)
        std::_Exit(90);
      ::close(output);
      ::execl(CONSAN_ANALYZER_FIXTURE, CONSAN_ANALYZER_FIXTURE, "capture.bin", "--examples", "0",
              nullptr);
      std::_Exit(91);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), expected_status);
  };
  ASSERT_TRUE(roundtrip(make_capture()).succeeded());
  execute(0);
  std::ifstream output(output_path);
  const std::string rendered{std::istreambuf_iterator<char>(output),
                             std::istreambuf_iterator<char>()};
  EXPECT_NE(rendered.find("\"transport_complete\":true"), std::string::npos);
  EXPECT_EQ(rendered.find("verdict"), std::string::npos);
  Capture gap = make_capture();
  wire::write_u32(gap.executions[0].report, kReportHeaderBytes + wire::kTraceRowExits, 0);
  ASSERT_TRUE(roundtrip(gap).succeeded());
  execute(1);
  encoded_.pop_back();
  execute(2);
  ASSERT_TRUE(roundtrip(make_capture()).succeeded());
  encoded_.back() ^= 1;
  execute(2);
  std::filesystem::remove_all(directory);
}
#endif

} // namespace
} // namespace rocjitsu::consan
