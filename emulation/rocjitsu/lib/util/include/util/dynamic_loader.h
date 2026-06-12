// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dynamic_loader.h
/// @brief Typed wrappers for runtime symbol resolution.

#ifndef UTIL_DYNAMIC_LOADER_H_
#define UTIL_DYNAMIC_LOADER_H_

#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <dlfcn.h>
#else
#error "Unsupported platform: expected _WIN32 or __linux__"
#endif

namespace util {

namespace detail {
#ifdef _WIN32
constexpr bool is_windows = true;
constexpr bool is_linux = false;
#elif defined(__linux__)
constexpr bool is_windows = false;
constexpr bool is_linux = true;
#endif
} // namespace detail

/// @brief Look up a typed function pointer from a loaded library handle.
/// @tparam T Function pointer type (e.g., int(*)(const char*, int)).
/// @param handle Platform library handle (void* on Linux, HMODULE on Windows).
/// @param name Symbol name to resolve.
/// @returns Typed function pointer, or nullptr if not found.
template <typename T>
  requires std::is_pointer_v<T>
T lookup_symbol([[maybe_unused]] auto handle, const char *name) {
  if constexpr (detail::is_windows)
    return reinterpret_cast<T>(GetProcAddress(handle, name));
  else if constexpr (detail::is_linux)
    return reinterpret_cast<T>(dlsym(handle, name));
}

} // namespace util

#endif // UTIL_DYNAMIC_LOADER_H_
