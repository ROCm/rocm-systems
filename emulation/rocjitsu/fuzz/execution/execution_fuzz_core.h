// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
};

std::optional<FuzzTarget> parse_target(std::string_view name);
std::string_view target_name(FuzzTarget target);
FuzzTarget indexed_target(uint8_t index);

// Execute an arbitrary instruction stream. Byte zero selects a target, byte
// one selects execution modes, and the remaining bytes are instructions.
// Decode rejection, unsupported execution, and the instruction budget are
// normal outcomes; host failures remain visible to the fuzzing runtime.
RunStats
run_input(std::span<const uint8_t> input, std::optional<FuzzTarget> target = std::nullopt,
          std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources> async_resources = nullptr,
          AsyncMode async_mode = AsyncMode::Auto, std::optional<unsigned> issuers = std::nullopt);
std::shared_ptr<amdgpu::matrix_coexecution::ExecutionResources>
make_async_execution_resources(unsigned helpers);
std::vector<uint8_t> random_input(uint64_t seed, size_t size = kMaxInputSize,
                                  std::optional<FuzzTarget> target = std::nullopt);

} // namespace rocjitsu::execution_fuzz
