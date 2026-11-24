// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"

#include <filesystem>

namespace rocjitsu::consan {

enum class ApplicationImageKind : uint32_t { Selected = 1, Native = 2 };

struct ApplicationImage {
  ImageIdentity original;
  uint32_t target = 0;
  ApplicationImageKind kind = ApplicationImageKind::Native;
  // Prepared artifact directory for Selected, exact original HSACO for Native.
  std::string path;
  Digest selection_profile{};
  std::string capture_path;
  bool operator==(const ApplicationImage &) const = default;
};

struct ApplicationCatalog {
  std::vector<ApplicationImage> images;
  bool operator==(const ApplicationCatalog &) const = default;
};

inline constexpr uint32_t kApplicationCatalogImages = 1024;
inline constexpr uint32_t kApplicationCatalogPathBytes = 4096;
inline constexpr uint64_t kApplicationCatalogBytes = 16 * 1024 * 1024;
inline constexpr uint64_t kApplicationCatalogImageBytes = 512 * 1024 * 1024;

[[nodiscard]] FailureOr<std::vector<uint8_t>>
encode_application_catalog(const ApplicationCatalog &catalog,
                           util::DiagnosticEmitter diagnostic = {});
[[nodiscard]] FailureOr<ApplicationCatalog>
decode_application_catalog(std::span<const uint8_t> bytes, util::DiagnosticEmitter diagnostic = {});
[[nodiscard]] FailureOr<ApplicationCatalog>
load_application_catalog(const std::filesystem::path &path,
                         util::DiagnosticEmitter diagnostic = {});

} // namespace rocjitsu::consan
