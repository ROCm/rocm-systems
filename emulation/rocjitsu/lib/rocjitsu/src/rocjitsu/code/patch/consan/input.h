// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/image/facts.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/lds_race.h"

#include <array>
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
  UnsupportedControlTransferRange,
  DecodeFailure,
  ResourceLimit,
};

/// @brief Provenance for the narrow recovery of an already-qualified lowering failure.
enum class InputErrorRecovery { None, NativeCallableCapability };

struct InputError {
  InputErrorCode code = InputErrorCode::None;
  std::optional<uint64_t> text_offset;
  std::string message;
  InputErrorRecovery recovery = InputErrorRecovery::None;
};

struct InputLimits {
  uint64_t image_bytes = 256 * 1024 * 1024;
  uint64_t text_bytes = 1024 * 1024;
  uint32_t instructions = 65536;
};

/// @brief Inspect LDS applicability independently of instrumentation ABI support.
/// @details A certificate requires complete decoded direct control flow and no
/// LDS, generic-address-space, cross-kernel or unknown memory effects. A missing
/// certificate is not a proof that LDS executes; normal qualification must still
/// decide whether the requested instrumentation can be provided.
[[nodiscard]] FailureOr<std::optional<NoProbesNeededProof>>
inspect_no_lds_memory(std::span<const uint8_t> image, const TargetProfile &target, EntryKey entry,
                      InputError &error, const InputLimits &limits = {});

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
  ScopeKind scope = ScopeKind::NativeLdsNoSynchronization;
  uint32_t wave_size = 0;
  uint32_t maximum_workgroup_size = 0;
  uint32_t kernarg_alignment = 8;
  std::array<uint32_t, 3> required_workgroup_size = {};
  uint32_t original_sgprs = 0;
  uint32_t original_vgprs = 0;
  std::vector<uint64_t> exit_offsets;
  // KernelExit replays these two-instruction terminal bundles contiguously
  // after the callback, while their VGPR allocation is still available.
  struct TerminalExit {
    uint64_t original_exit = 0;
    uint64_t deallocation = 0;
  };
  std::vector<TerminalExit> terminal_exits;
  std::vector<NativeLdsAccess> accesses;
  std::vector<BarrierFacts> barriers;
  bool repeated = false;
};

/// @brief Build an owning, qualified input program for the bounded native LDS profile.
/// @details A closed acyclic CFG, or a qualified single-block barrier loop, admits an explicit
/// bounded native-LDS application instruction subset. Unsupported memory or sync is a
/// preparation failure, never an omitted operation interpreted as no ordering.
/// The profile must match a supported table entry; copies/temporaries are valid.
/// A successful result retains the static entry, independent of caller lifetime.
[[nodiscard]] FailureOr<InputProgram> build_input_program(std::span<const uint8_t> image,
                                                          const TargetProfile &target,
                                                          std::string_view entry, InputError &error,
                                                          const InputLimits &limits = {});

/// @brief Qualify one selected entry inside an intact multi-entry source image.
/// @details Generic SBI independently proves disjoint ownership for the whole
/// image. This analysis rejects selected control flow that enters a sibling.
[[nodiscard]] FailureOr<InputProgram>
build_image_entry_program(std::span<const uint8_t> image, const TargetProfile &target,
                          std::string_view entry, InputError &error, const InputLimits &limits = {},
                          ScopeKind scope = ScopeKind::NativeLdsNoSynchronization);

} // namespace rocjitsu::consan
