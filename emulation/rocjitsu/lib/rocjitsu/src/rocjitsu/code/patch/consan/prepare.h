// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/input.h"

namespace rocjitsu::consan {

struct PreparedArtifact {
  Manifest manifest;
  std::vector<uint8_t> image;
};

/// @brief Check the shared prologue against ConSan's runtime wrapper offsets.
/// @details The two layouts have independent payload declarations. Reject drift
/// before publishing an artifact whose prologue would read a different slot.
[[nodiscard]] Result validate_entry_kernarg_layout(const DbiEntryPrologue &prologue,
                                                   const KernargExtensionLayout &kernarg,
                                                   InputError &error);

/// @brief Prepare one exact original image without runtime allocations or handles.
[[nodiscard]] FailureOr<PreparedArtifact> prepare(std::span<const uint8_t> original,
                                                  const TargetProfile &target,
                                                  std::string_view entry, InputError &error);

/// @brief Validate shared images/CFG/maps and analysis-specific preservation.
/// @details Verify image identity, descriptor/note resources, branch boundaries,
/// original operation multiplicity, probe placement and owning static mappings.
/// Canonical re-lowering checks consistency, not the correctness of its own
/// algorithm; independent state/publication tests exercise that behavior.
[[nodiscard]] Result validate_prepared(std::span<const uint8_t> original,
                                       const PreparedArtifact &artifact, InputError &error);

} // namespace rocjitsu::consan
