// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "util/diagnostic.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::amdgpu::matrix_coexecution {
class ExecutionResources;
}

namespace rocjitsu::execution_fuzz {

inline constexpr size_t kHeaderSize = 2;
inline constexpr size_t kMaxInputSize = 1024;
inline constexpr size_t kGeneratedKernelSeedSize = 256;
inline constexpr uint8_t kAsyncSelectBit = 0x1;
inline constexpr uint8_t kConcurrentIssuerMask = 0x6;
inline constexpr uint8_t kFullExecMaskBit = 0x8;
inline constexpr uint8_t kWaitCheckingBit = 0x10;
inline constexpr uint8_t kCacheMemoryBit = 0x20;

enum class FuzzTarget : uint8_t {
  Gfx908,
  Gfx90a,
  Gfx942,
  Gfx950,
  Gfx1010,
  Gfx1030,
  Gfx1100,
  Gfx1151,
  Gfx1200,
  Gfx1201,
  Gfx1250,
  Gfx1251,
  Count,
};

enum class AsyncMode : uint8_t {
  Auto,
  Disabled,
  Enabled,
};

struct RunStats {
  uint64_t executed = 0;
  uint64_t async_executed = 0;
  uint64_t async_capable_inputs = 0;
  uint64_t async_enabled_inputs = 0;
  uint64_t concurrent_inputs = 0;
  uint64_t issuer_executions = 0;
  uint64_t waves_halted = 0;
  uint64_t rounds = 0;
  uint64_t budget_exhausted = 0;
  uint64_t inputs = 0;
  uint64_t inputs_executed = 0;
  uint64_t inputs_halted = 0;
  uint64_t inputs_stopped = 0;
  /// @brief L2 backing reads observed for the cache-backed data route.
  /// @details Used by the routing test and omitted from the CLI summary.
  uint64_t l2_backing_reads = 0;
};

std::optional<FuzzTarget> parse_target(std::string_view name);
/// @brief Query metadata for a valid FuzzTarget (excluding Count).
std::string_view target_name(FuzzTarget target);
/// @brief Query architecture for a valid FuzzTarget (excluding Count).
rj_code_arch_t target_arch(FuzzTarget target);

/// @brief Return the canonical registry ID for an execution fuzz target.
rj_code_target_id_t target_id(FuzzTarget target);

/// @brief Return the target-specific encoding of `s_endpgm`.
uint32_t terminating_instruction(FuzzTarget target);

/// @brief Align a raw program and append terminators plus decoder and CU fetch padding.
std::vector<uint8_t> terminated_program(std::span<const uint8_t> instructions, FuzzTarget target);
bool targets_cover_amdgpu_execution_registry();
std::optional<FuzzTarget> indexed_target(uint8_t index);

/// @brief Execute an arbitrary instruction stream.
/// @details Byte zero selects a target, byte one selects execution modes, and
/// the remaining bytes are instructions. The wait-checking and cache bits can
/// be flipped without changing the initial register and memory contents.
/// Unassigned targets, decode rejection, unsupported execution, and the
/// instruction budget are normal outcomes.
/// Invalid harness arguments and setup failures return failure and emit a diagnostic;
/// host failures remain visible to the fuzzing runtime. All process-scoped GWS
/// resources start armed with a zero count so a barrier without a guest INIT is
/// a defined guest outcome.
util::FailureOr<RunStats>
run_input(std::span<const uint8_t> input, std::optional<FuzzTarget> target = std::nullopt,
          std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources> async_resources = nullptr,
          AsyncMode async_mode = AsyncMode::Auto, std::optional<unsigned> issuers = std::nullopt,
          util::DiagnosticEmitter diagnostic = {});
/// @brief Create helper resources, or fail with a diagnostic for a count outside 1..128.
util::FailureOr<std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>>
make_async_execution_resources(unsigned helpers, util::DiagnosticEmitter diagnostic = {});

/// @brief Generate a deterministic input for a random execution campaign.
/// @details Mixes arbitrary bytes, target encodings, and generated kernels.
/// Invalid sizes or targets and unavailable decoders fail with a diagnostic.
util::FailureOr<std::vector<uint8_t>> random_input(uint64_t seed, size_t size = kMaxInputSize,
                                                   std::optional<FuzzTarget> target = std::nullopt,
                                                   util::DiagnosticEmitter diagnostic = {});

/// @brief Generate a bounded program as a replay input.
/// @details Used by the random mix and seed corpora. Invalid sizes or targets
/// fail with a diagnostic.
util::FailureOr<std::vector<uint8_t>>
generated_kernel_input(uint64_t seed, size_t size, FuzzTarget target,
                       util::DiagnosticEmitter diagnostic = {});

} // namespace rocjitsu::execution_fuzz
