/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/**
 * Shared helpers for tests that assert on the runtime's AQL packet trace.
 *
 * The trace is an internal CLR debug format with no stability contract.
 * Centralizing its parser means a format change breaks one header rather than
 * every test that reads the trace.
 *
 * CLR reads AMD_LOG_LEVEL and AMD_LOG_MASK while the HIP runtime initializes.
 * Re-execute each workload in a fresh child so the variables are set before
 * runtime initialization.
 */

#include "hip_test_common.hh"
#include "hip_test_aqllog_trace_region.hh"
#include "hip_test_process.hh"

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace hip {
namespace aqllog {

enum class PacketKind { kDispatch, kBarrierAnd, kBarrierOr, kBarrierValue };

struct Packet {
  PacketKind kind = PacketKind::kDispatch;
  std::optional<int> barrier;  //!< AQL barrier bit
  std::optional<int> acquire;  //!< HSA_FENCE_SCOPE_NONE/AGENT/SYSTEM, i.e. 0/1/2
  std::optional<int> release;
  std::string line;  //!< the raw trace line, for fields this struct does not parse
};

//! Return the text between matching child trace markers, or nullopt when either
//! marker is missing or out of order.
inline std::optional<std::string> ExtractRegion(const std::string& log, std::string_view name) {
  const std::string begin = std::string{kRegionBegin} + std::string{name};
  const std::string end = std::string{kRegionEnd} + std::string{name};
  const size_t begin_pos = log.find(begin);
  if (begin_pos == std::string::npos) {
    return std::nullopt;
  }
  const size_t content_pos = log.find('\n', begin_pos + begin.size());
  if (content_pos == std::string::npos) {
    return std::nullopt;
  }
  const size_t end_pos = log.find(end, content_pos + 1);
  if (end_pos == std::string::npos) {
    return std::nullopt;
  }
  return log.substr(content_pos + 1, end_pos - content_pos - 1);
}

//! Run an executable with AQL tracing on and return its captured output.
//! |arguments| must not contain quoted whitespace because SpawnProc tokenizes
//! it before exec.
inline std::string RunExecutable(const std::string& executable, const std::string& arguments,
                                 const std::map<std::string, std::string>& extra_env = {}) {
  constexpr bool kCaptureOutput = true;
  constexpr bool kCaptureStderr = true;  //!< the trace is written to stderr
  SpawnProc child(executable, kCaptureOutput, kCaptureStderr);

  child.setEnv("AMD_LOG_LEVEL", "5");
  child.setEnv("AMD_LOG_MASK", "8");
  child.setEnv("AMD_LOG_ASYNC", "0");
  for (const auto& entry : extra_env) {
    child.setEnv(entry.first, entry.second);
  }

  const int exit_code = child.run(arguments);
  INFO("AQL log child '" << executable << "' exit code " << exit_code);
  REQUIRE(exit_code == 0);
  return child.getOutput();
}

// Same-binary Catch child support
//
// The environment marker lets parent and child cases avoid invoking each
// other recursively. Standalone helper executables do not need it.
constexpr char kChildEnv[] = "HIP_TEST_AQLLOG_CHILD";

inline bool RunningAsChild() { return std::getenv(kChildEnv) != nullptr; }

inline std::string RunChildTest(const std::string& child_test_case,
                                const std::map<std::string, std::string>& extra_env = {}) {
  std::map<std::string, std::string> child_env = extra_env;
  child_env[kChildEnv] = "1";
  return RunExecutable(getSelfExePath(), child_test_case, child_env);
}

//! The digits written immediately after |key|, or nullopt when |key| is absent
//! or is not followed by one.
inline std::optional<int> FieldValue(const std::string& line, const std::string& key) {
  const size_t key_pos = line.find(key);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }

  size_t pos = key_pos + key.size();
  if (pos >= line.size() || std::isdigit(static_cast<unsigned char>(line[pos])) == 0) {
    return std::nullopt;
  }

  int value = 0;
  while (pos < line.size() && std::isdigit(static_cast<unsigned char>(line[pos])) != 0) {
    value = value * 10 + (line[pos] - '0');
    ++pos;
  }
  return value;
}

//! The trace token identifying a dispatch of a 1D kernel with |size| threads.
//! Some tests use distinct workgroup sizes to locate their own dispatches,
//! because runtime initialization also emits packets.
inline std::string WorkgroupToken(uint32_t size) {
  return "workgroup=[" + std::to_string(size) + ", 1, 1]";
}

inline bool IsDispatchLine(const std::string& line) {
  return line.find("Dispatch Header") != std::string::npos;
}

inline std::optional<Packet> ParseLine(const std::string& line) {
  Packet packet;
  if (IsDispatchLine(line)) {
    packet.kind = PacketKind::kDispatch;
  } else if (line.find("Barrier-AND Header") != std::string::npos) {
    packet.kind = PacketKind::kBarrierAnd;
  } else if (line.find("Barrier-OR Header") != std::string::npos) {
    packet.kind = PacketKind::kBarrierOr;
  } else if (line.find("BarrierValue Header") != std::string::npos) {
    packet.kind = PacketKind::kBarrierValue;
  } else {
    return std::nullopt;
  }

  packet.barrier = FieldValue(line, "barrier=");
  packet.acquire = FieldValue(line, "acquire=");
  packet.release = FieldValue(line, "release=");
  packet.line = line;
  return packet;
}

inline std::vector<Packet> ParsePackets(const std::string& log) {
  std::vector<Packet> packets;
  std::istringstream lines(log);
  std::string line;
  while (std::getline(lines, line)) {
    if (std::optional<Packet> packet = ParseLine(line)) {
      packets.push_back(*packet);
    }
  }
  return packets;
}

//! Packets emitted after the dispatch carrying |begin_token| and before the
//! next dispatch carrying |end_token|. Scanning runs to the end of |log| when
//! |end_token| never appears, and returns nothing when |begin_token| never does.
inline std::vector<Packet> PacketsBetweenDispatches(const std::string& log,
                                                    const std::string& begin_token,
                                                    const std::string& end_token) {
  std::vector<Packet> packets;
  std::istringstream lines(log);
  std::string line;
  bool collecting = false;
  while (std::getline(lines, line)) {
    const bool is_dispatch = IsDispatchLine(line);
    if (!collecting) {
      collecting = is_dispatch && line.find(begin_token) != std::string::npos;
      continue;
    }
    if (is_dispatch && line.find(end_token) != std::string::npos) {
      break;
    }
    if (std::optional<Packet> packet = ParseLine(line)) {
      packets.push_back(*packet);
    }
  }
  return packets;
}

//! Barrier packets in the same dispatch-delimited window as PacketsBetweenDispatches().
inline std::vector<Packet> BarriersBetweenDispatches(const std::string& log,
                                                     const std::string& begin_token,
                                                     const std::string& end_token) {
  std::vector<Packet> barriers;
  for (const Packet& packet : PacketsBetweenDispatches(log, begin_token, end_token)) {
    if (packet.kind != PacketKind::kDispatch) {
      barriers.push_back(packet);
    }
  }
  return barriers;
}

//! The dispatch packet for a 1D kernel of |size| threads, or nullopt when the
//! trace holds no such dispatch.
inline std::optional<Packet> DispatchForWorkgroup(const std::string& log, uint32_t size) {
  const std::string token = WorkgroupToken(size);
  for (const Packet& packet : ParsePackets(log)) {
    if (packet.kind == PacketKind::kDispatch && packet.line.find(token) != std::string::npos) {
      return packet;
    }
  }
  return std::nullopt;
}

//! The AQL barrier bit on the dispatch for a 1D kernel of |size| threads,
//! or -1 when |log| holds no such dispatch that reports the bit.
inline int DispatchBarrierBitForWorkgroup(const std::string& log, uint32_t size) {
  const std::optional<Packet> packet = DispatchForWorkgroup(log, size);
  if (packet.has_value() && packet->barrier.has_value()) {
    return *packet->barrier;
  }
  return -1;
}

}  // namespace aqllog
}  // namespace hip
