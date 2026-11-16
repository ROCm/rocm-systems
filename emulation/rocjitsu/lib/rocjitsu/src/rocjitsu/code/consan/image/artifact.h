// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"

#include <filesystem>

namespace rocjitsu::consan {

struct LoadedArtifact {
  Manifest manifest;
  std::vector<uint8_t> prepared;
};

/// @brief Check exact artifacts and required ELF/descriptor/note consistency.
/// @details Shared image inspection has no decoder, CFG analysis or lowering.
[[nodiscard]] Result verify_artifact_images(const Manifest &manifest,
                                            std::span<const uint8_t> original,
                                            std::span<const uint8_t> prepared,
                                            ContractError *error = nullptr);

/// @brief Read one ready artifact with bounded owning storage; never compile.
[[nodiscard]] FailureOr<LoadedArtifact> load_artifact(const std::filesystem::path &directory,
                                                      std::span<const uint8_t> original,
                                                      std::string_view concrete_target,
                                                      ContractError *error = nullptr);

} // namespace rocjitsu::consan
