// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/image/facts.h"
#include "rocjitsu/code/patch/consan/input.h"

namespace rocjitsu::consan {

struct PreparedArtifact {
  Manifest manifest;
  std::vector<uint8_t> image;
};

/// @brief Prepare one exact original image without runtime allocations or handles.
[[nodiscard]] FailureOr<PreparedArtifact> prepare(std::span<const uint8_t> original,
                                                  const TargetProfile &target,
                                                  std::string_view entry, InputError &error);

struct PreparedImageArtifact {
  ImageFacts facts;
  std::vector<uint8_t> image;
};

enum class PreparationStage { Applicability, Qualification, Lowering };

struct EntryPreparation {
  std::string entry;
  EntryKey key = 0;
  bool instrumented = false;
  std::optional<NoProbesNeededProof> no_probes_needed = {};
  PreparationStage stage = PreparationStage::Qualification;
  InputError reason;
};

/// @brief Selection evidence independent of whether instrumentation was possible.
/// @details A missing artifact means all requested entries remain native. It is
/// not an empty prepared image or a capture claiming successful analysis.
struct PreparationResult {
  ImageIdentity original;
  uint32_t target = 0;
  ScopeKind requested_scope = ScopeKind::NativeLdsNoSynchronization;
  uint32_t inventory_entries = 0;
  std::vector<EntryPreparation> entries;
  std::optional<PreparedImageArtifact> artifact;
};

/// @brief Prepare one coherent image with explicit selected entry names.
[[nodiscard]] FailureOr<PreparedImageArtifact>
prepare_image(std::span<const uint8_t> original, const TargetProfile &target,
              std::span<const std::string_view> entries, InputError &error,
              ScopeKind scope = ScopeKind::NativeLdsNoSynchronization);

/// @brief Qualify every image entry by default, retaining unsupported entries as native.
/// @details Invalid images/selections, exhausted budgets and failed rewriting
/// remain errors. Empty entries selects the complete inventory; explicit names
/// limit the request. Supported entries undergo the same strict transactional
/// lowering and preservation checks as prepare_image().
[[nodiscard]] FailureOr<PreparationResult>
prepare_image_best_effort(std::span<const uint8_t> original, const TargetProfile &target,
                          std::span<const std::string_view> entries, InputError &error,
                          ScopeKind scope = ScopeKind::NativeLdsNoSynchronization);

/// @brief Render schema-v2 outcomes, or return empty for inconsistent evidence.
[[nodiscard]] std::string render_preparation_report(const PreparationResult &result);

/// @brief Validate shared images/CFG/maps and analysis-specific preservation.
/// @details Verify image identity, descriptor/note resources, branch boundaries,
/// original operation multiplicity, probe placement and owning static mappings.
/// Canonical re-lowering checks consistency, not the correctness of its own
/// algorithm; independent state/publication tests exercise that behavior.
[[nodiscard]] Result validate_prepared(std::span<const uint8_t> original,
                                       const PreparedArtifact &artifact, InputError &error);

} // namespace rocjitsu::consan
