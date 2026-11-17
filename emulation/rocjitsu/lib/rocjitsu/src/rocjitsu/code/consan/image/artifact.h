// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/image/facts.h"

#include <filesystem>

namespace rocjitsu::consan {

struct LoadedArtifact {
  Manifest manifest;
  std::vector<uint8_t> prepared;
};

/// @brief Check the complete entry inventory and ELF/descriptor/note consistency.
/// @details Shared image inspection has no decoder, CFG analysis or lowering.
[[nodiscard]] Result verify_artifact_images(const ImageFacts &facts,
                                            std::span<const uint8_t> original,
                                            std::span<const uint8_t> prepared,
                                            util::DiagnosticEmitter diagnostic = {});

/// @brief Single-entry compiler convenience; original metadata is read explicitly.
[[nodiscard]] Result verify_artifact_images(const Manifest &manifest,
                                            std::span<const uint8_t> original,
                                            std::span<const uint8_t> prepared,
                                            util::DiagnosticEmitter diagnostic = {});

/// @brief Read one ready artifact with bounded owning storage; never compile.
[[nodiscard]] FailureOr<LoadedArtifact> load_artifact(const std::filesystem::path &directory,
                                                      std::span<const uint8_t> original,
                                                      std::string_view concrete_target,
                                                      util::DiagnosticEmitter diagnostic = {});

} // namespace rocjitsu::consan
