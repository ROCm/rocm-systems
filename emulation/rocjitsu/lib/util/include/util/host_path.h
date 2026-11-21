// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_HOST_PATH_H_
#define UTIL_HOST_PATH_H_

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>

namespace util {

/// @brief Resolve a host filesystem path, returning an empty path on failure.
inline std::filesystem::path canonical_host_path(const char *path) {
#if defined(__GLIBC__)
  // librocjitsu exports a realpath interposer. When Mirage loads it with dlopen,
  // unversioned realpath calls, including from static libstdc++, can bind to
  // glibc's compatibility ABI, which rejects a null output buffer.
  const std::unique_ptr<char, decltype(&std::free)> resolved(::canonicalize_file_name(path),
                                                             &std::free);
  return resolved ? std::filesystem::path(resolved.get()) : std::filesystem::path{};
#else
  std::error_code error;
  return std::filesystem::canonical(path, error);
#endif
}

} // namespace util

#endif // UTIL_HOST_PATH_H_
