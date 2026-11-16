// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "util/sha256.h"

#include <map>

namespace rocjitsu::consan {

// The runtime enforces this execution contract. Queue, overlap and profiling
// restrictions describe admission; system/fine-grained transport supports host
// visibility. These conditions do not establish guest LDS ordering or race freedom.
inline constexpr uint32_t kExecutionSingleQueue = 1;
inline constexpr uint32_t kExecutionNonoverlapping = 2;
inline constexpr uint32_t kExecutionProfilingDisabled = 4;
inline constexpr uint32_t kExecutionSystemFineGrain = 8;
inline constexpr uint32_t kSupportedExecutionConditions =
    kExecutionSingleQueue | kExecutionNonoverlapping | kExecutionProfilingDisabled |
    kExecutionSystemFineGrain;
// Evidence of completion, host visibility and signal forwarding. The contract
// accepts only this complete value; individual bits are not separate states.
inline constexpr uint32_t kCompletionVisibleForwarded = 7;

class CaptureHeader {
public:
  std::array<uint8_t, 16> run_identity{};
  Manifest manifest;
  uint32_t execution_conditions = kSupportedExecutionConditions;
  bool operator==(const CaptureHeader &) const = default;
};

/// @brief Owning evidence for one execution in a capture, not live process state.
class CapturedExecution {
public:
  // Nonzero, increasing identity within the capture's run_identity.
  uint64_t serial = 0;
  LaunchDimensions dimensions;
  uint32_t group_bytes = 0;
  uint32_t private_bytes = 0;
  // Run-local logical report slot. Replacing its allocation during growth does
  // not reset the generation; successive uses of this slot increase it.
  uint32_t storage_slot = 0;
  uint32_t storage_generation = 0;
  // Completion/visibility/forwarding evidence, not a live HSA signal value.
  uint32_t completion = kCompletionVisibleForwarded;
  // Copied dispatch context. Its saved addresses must not be dereferenced by
  // offline readers, which need no live application or HSA allocations.
  std::array<uint8_t, kContextBytes> context{};
  // Original kernarg bytes, including pointer values, without pointed-to data.
  std::vector<uint8_t> arguments;
  // Empty in the pre-submit envelope; snapshot fills an independent host copy
  // of raw report headers/rows/records after completion and visibility proof.
  std::vector<uint8_t> report;
};

struct CaptureLimits {
  uint64_t file_bytes = 512 * 1024 * 1024;
  uint32_t executions = 1024;
  uint32_t argument_bytes = 65536;
  ContractLimits contract;
};

struct Capture {
  CaptureHeader header;
  std::vector<CapturedExecution> executions;
};

/// @brief Incremental encoding with an explicit, checksummed End dependency.
/// @details Owns immutable run facts. Returned chunks belong to the caller;
/// failed I/O must poison its writer and prohibit publishing finish(). It never
/// interprets row contents or produces a race diagnosis.
class CaptureEncoder {
public:
  [[nodiscard]] FailureOr<std::vector<uint8_t>> start(const CaptureHeader &header,
                                                      ContractError *error = nullptr);
  [[nodiscard]] FailureOr<std::vector<uint8_t>> append(const CapturedExecution &execution,
                                                       ContractError *error = nullptr);
  [[nodiscard]] FailureOr<std::vector<uint8_t>> finish(ContractError *error = nullptr);

private:
  CaptureHeader header_;
  util::Sha256 hash_;
  uint64_t count_ = 0;
  uint64_t bytes_ = 0;
  std::map<uint32_t, uint32_t> generations_;
  bool started_ = false;
  bool finished_ = false;
};

/// @brief Verify the entire container before exposing owning facts/observations.
/// @details A malformed local GPU row remains raw evidence for the analyzer;
/// framing, missing End/facts, execution identity and bounds reject the file.
[[nodiscard]] FailureOr<Capture> decode_capture(std::span<const uint8_t> bytes,
                                                ContractError *error = nullptr,
                                                const CaptureLimits &limits = {});

} // namespace rocjitsu::consan
