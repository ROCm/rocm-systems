// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dynamic_loader.h
/// @brief Typed wrappers for runtime symbol resolution.

#ifndef UTIL_DYNAMIC_LOADER_H_
#define UTIL_DYNAMIC_LOADER_H_

#include <string>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>

namespace util::detail {
constexpr bool is_windows = true;
constexpr bool is_linux = false;
constexpr int library_flags = 0;
using LibraryHandle = HMODULE;
struct PlatformApi {
  static auto last_error() { return GetLastError(); }
};
} // namespace util::detail
#elif defined(__linux__)
#include <dlfcn.h>

namespace util::detail {
constexpr bool is_windows = false;
constexpr bool is_linux = true;
constexpr int library_flags = RTLD_NOW | RTLD_LOCAL;
using LibraryHandle = void *;
struct PlatformApi {
  static auto last_error() { return dlerror(); }
};
} // namespace util::detail
#else
#error "Unsupported platform: expected _WIN32 or __linux__"
#endif

namespace util::detail {

template <typename Name> LibraryHandle open_library(Name name) {
  if constexpr (is_windows)
    return LoadLibraryA(name);
  else if constexpr (is_linux)
    return dlopen(name, library_flags);
}

template <typename Handle> void close_library(Handle handle) {
  if constexpr (is_windows)
    FreeLibrary(handle);
  else if constexpr (is_linux)
    dlclose(handle);
}

template <typename Api> std::string last_library_error() {
  if constexpr (is_windows) {
    return "error code " + std::to_string(Api::last_error());
  } else {
    const char *err = Api::last_error();
    return err ? std::string(err) : std::string("unknown error");
  }
}

} // namespace util::detail

namespace util {

/// @brief Opaque handle to a dynamically loaded library.
/// @details void* on Linux (dlopen), HMODULE on Windows (LoadLibrary).
using LibraryHandle = detail::LibraryHandle;

/// @brief Load a shared library by name or path.
/// @param name Library name (resolved via the platform search path) or path.
/// @returns A non-null handle on success, or nullptr on failure. Call
///          last_library_error() for a human-readable failure reason.
inline LibraryHandle open_library(const char *name) { return detail::open_library(name); }

/// @brief Close a library handle previously returned by open_library().
inline void close_library(LibraryHandle handle) {
  if (!handle)
    return;
  detail::close_library(handle);
}

/// @brief Return a human-readable description of the last library error.
inline auto last_library_error() { return detail::last_library_error<detail::PlatformApi>(); }

/// @brief Look up a typed function pointer from a loaded library handle.
/// @tparam T Function pointer type (e.g., int(*)(const char*, int)).
/// @param handle Platform library handle (void* on Linux, HMODULE on Windows).
/// @param name Symbol name to resolve.
/// @returns Typed function pointer, or nullptr if not found.
template <typename T, typename Handle>
T lookup_symbol([[maybe_unused]] Handle handle, const char *name) {
  static_assert(std::is_pointer_v<T>, "symbol type must be a pointer");
  if constexpr (detail::is_windows)
    return reinterpret_cast<T>(GetProcAddress(handle, name));
  else if constexpr (detail::is_linux)
    return reinterpret_cast<T>(dlsym(handle, name));
}

} // namespace util

#endif // UTIL_DYNAMIC_LOADER_H_
