// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/image/facts.h"

#include <filesystem>

namespace rocjitsu::consan {

struct LoadedArtifact {
  ImageFacts facts;
  std::vector<uint8_t> prepared;
};

/// @brief Original bindings for an explicitly configured native image.
/// @details This inventory has no selected manifests or detector coverage and
/// cannot be published as a prepared artifact or selected-execution capture.
struct NativeImageInventory {
  ImageIdentity identity;
  uint32_t target = 0;
  uint32_t wave_size = 0;
  std::vector<ImageEntryFacts> entries;
};

struct LoadedNativeImage {
  NativeImageInventory inventory;
  std::vector<uint8_t> original;
};

[[nodiscard]] FailureOr<NativeImageInventory>
inspect_native_image(std::span<const uint8_t> original, std::string_view concrete_target,
                     util::DiagnosticEmitter diagnostic = {});

/// @brief Own and validate an independently configured, unmodified native image.
[[nodiscard]] FailureOr<LoadedNativeImage>
load_native_image(const std::filesystem::path &path, util::DiagnosticEmitter diagnostic = {});

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

/// @brief Read one ready artifact with independently configured selection identity.
[[nodiscard]] FailureOr<LoadedArtifact> load_artifact(const std::filesystem::path &directory,
                                                      std::span<const uint8_t> original,
                                                      std::string_view concrete_target,
                                                      const Digest &expected_selection_profile,
                                                      util::DiagnosticEmitter diagnostic = {});

} // namespace rocjitsu::consan
