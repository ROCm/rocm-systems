// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_diagnostic_report.h"

#include "util/log.h"

#include <format>
#include <fstream>

namespace rocjitsu::amdgpu {
namespace {

using Finding = MemoryRaceChecker::Finding;
using Kind = MemoryRaceChecker::Kind;
using Space = MemoryRaceChecker::Space;

const char *kind_name(Kind kind) {
  switch (kind) {
  case Kind::RAW:
    return "RAW";
  case Kind::WAR:
    return "WAR";
  case Kind::WAW:
    return "WAW";
  }
  return "RAW";
}

const char *action_name(const Finding &finding) {
  if (finding.space == Space::Global)
    return "cross-workgroup synchronization";
  return finding.requires_barrier ? "barrier" : "wait";
}

} // namespace

MemoryDiagnosticReport::MemoryDiagnosticReport(std::string path) : path_(std::move(path)) {}

MemoryDiagnosticReport::~MemoryDiagnosticReport() {
  std::lock_guard lock(mutex_);
  if (dirty_)
    write_locked();
}

void MemoryDiagnosticReport::add_wait(const MemoryRaceChecker::WaveKey &wave,
                                      const MemoryWaitScoreboard::Hazard &hazard,
                                      std::string_view counter, uint32_t required_wait) {
  std::lock_guard lock(mutex_);
  ++observed_;
  dirty_ = true;
  if (wait_findings_.size() < kMaxFindings)
    wait_findings_.push_back({wave, hazard, std::string(counter), required_wait});
  else
    ++suppressed_;
}

void MemoryDiagnosticReport::add_race(const Finding &finding) {
  std::lock_guard lock(mutex_);
  ++observed_;
  dirty_ = true;
  if (race_findings_.size() < kMaxFindings)
    race_findings_.push_back(finding);
  else
    ++suppressed_;
}

void MemoryDiagnosticReport::add_suppressed_race() {
  std::lock_guard lock(mutex_);
  ++observed_;
  ++suppressed_;
  dirty_ = true;
}

void MemoryDiagnosticReport::set_coverage_degraded() {
  std::lock_guard lock(mutex_);
  dirty_ |= !coverage_degraded_;
  coverage_degraded_ = true;
}

void MemoryDiagnosticReport::flush() {
  std::lock_guard lock(mutex_);
  if (dirty_ || !written_)
    write_locked();
}

void MemoryDiagnosticReport::write_locked() {
  std::ofstream out(path_);
  if (!out) {
    if (!write_error_reported_)
      util::Logger::warn(
          std::format("memory-diagnostics: cannot write memory_diagnostic_report {}", path_));
    write_error_reported_ = true;
    return;
  }
  out << "{\n  \"version\": 1,\n  \"findings\": [";
  bool first = true;
  for (const WaitFinding &finding : wait_findings_) {
    out << (first ? "" : ",");
    first = false;
    out << std::format("\n    {{\"checker\":\"memory-wait\",\"kind\":\"wait\",\"resource\":\"{}\","
                       "\"producer_pc\":{},\"consumer_pc\":{},\"process\":{},\"dispatch\":{},"
                       "\"workgroup\":{},\"wave\":{},\"required_wait\":{}}}",
                       finding.counter, finding.hazard.producer.pc, finding.hazard.consumer_pc,
                       finding.wave.process, finding.wave.dispatch, finding.wave.workgroup,
                       finding.wave.wave, finding.required_wait);
  }
  for (const Finding &finding : race_findings_) {
    out << (first ? "" : ",");
    first = false;
    out << std::format("\n    {{\"checker\":\"memory-race\",\"kind\":\"{}\",\"resource\":\"{}\","
                       "\"address\":{},\"size\":{},\"producer_pc\":{},\"consumer_pc\":{},"
                       "\"process\":{},\"dispatch\":{},\"producer_workgroup\":{},"
                       "\"producer_wave\":{},\"producer_lane\":{},\"consumer_workgroup\":{},"
                       "\"consumer_wave\":{},\"consumer_lane\":{},\"action\":\"{}\"}}",
                       kind_name(finding.kind), finding.space == Space::LDS ? "lds" : "global",
                       finding.address, finding.size, finding.producer.pc, finding.consumer.pc,
                       finding.consumer.wave.process, finding.consumer.wave.dispatch,
                       finding.producer.wave.workgroup, finding.producer.wave.wave,
                       finding.producer.lane, finding.consumer.wave.workgroup,
                       finding.consumer.wave.wave, finding.consumer.lane, action_name(finding));
  }
  out << std::format("\n  ],\n  \"observed\": {},\n  \"suppressed\": {},\n  "
                     "\"coverage_degraded\": {}\n}}\n",
                     observed_, suppressed_, coverage_degraded_ ? "true" : "false");
  out.flush();
  if (!out) {
    if (!write_error_reported_)
      util::Logger::warn(
          std::format("memory-diagnostics: cannot write memory_diagnostic_report {}", path_));
    write_error_reported_ = true;
    return;
  }
  written_ = true;
  dirty_ = false;
  write_error_reported_ = false;
}

} // namespace rocjitsu::amdgpu
