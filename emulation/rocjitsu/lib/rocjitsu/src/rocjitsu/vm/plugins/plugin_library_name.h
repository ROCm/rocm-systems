// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file plugin_library_name.h
/// @brief Shared execution-plugin shared-object naming convention.

#pragma once

#include <string>
#include <string_view>

namespace rocjitsu {

inline constexpr std::string_view kPluginLibraryPrefix = "librocjitsu_plugin_";
inline constexpr std::string_view kPluginLibrarySuffix = ".so";

inline std::string plugin_library_name(std::string_view plugin_name) {
  std::string name(kPluginLibraryPrefix);
  name.append(plugin_name);
  name.append(kPluginLibrarySuffix);
  return name;
}

} // namespace rocjitsu
