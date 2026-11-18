// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/lds_race.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::consan {

enum class InputErrorCode {
  None,
  InvalidImage,
  UnsupportedTarget,
  UnsupportedEntry,
  UnsupportedAccess,
  UnsupportedSynchronization,
  UnsupportedInstruction,
  UnsupportedFlow,
  DecodeFailure,
  ResourceLimit,
};

struct InputError {
  InputErrorCode code = InputErrorCode::None;
  std::optional<uint64_t> text_offset;
  std::string message;
};

struct InputLimits {
  uint64_t image_bytes = 256 * 1024 * 1024;
  uint64_t text_bytes = 1024 * 1024;
  uint32_t instructions = 65536;
};

/// @brief Compiler-owned original-program analysis, not a prepared artifact.
/// @details No prepared mappings or register reservations are fabricated here.
/// The shared instrumentor plans placement and preservation during lowering.
struct InputProgram {
  std::vector<uint8_t> image;
  ImageIdentity identity;
  uint64_t text_file_offset = 0;
  uint64_t text_bytes = 0;
  KernelDescriptorInfo kernel;
  // Canonical static table entry; never borrows the caller's aggregate/name.
  const TargetProfile *target = nullptr;
  uint32_t wave_size = 0;
  uint32_t maximum_workgroup_size = 0;
  uint32_t original_sgprs = 0;
  uint32_t original_vgprs = 0;
  std::vector<uint64_t> exit_offsets;
  std::vector<NativeLdsAccess> accesses;
  std::vector<BarrierFacts> barriers;
};

/// @brief Build an owning, qualified input program for the native LDS b32 MVP.
/// @details The closed, acyclic reachable CFG admits a deliberately explicit
/// integer/argument/output instruction subset. Unsupported memory or sync is a
/// preparation failure, never an omitted operation interpreted as no ordering.
/// The profile must match a supported table entry; copies/temporaries are valid.
/// A successful result retains the static entry, independent of caller lifetime.
[[nodiscard]] FailureOr<InputProgram> build_input_program(std::span<const uint8_t> image,
                                                          const TargetProfile &target,
                                                          std::string_view entry, InputError &error,
                                                          const InputLimits &limits = {});

} // namespace rocjitsu::consan
