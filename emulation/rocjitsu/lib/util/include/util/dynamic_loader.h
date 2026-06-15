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

/// @brief Opaque handle to a dynamically loaded library.
/// @details void* on Linux (dlopen), HMODULE on Windows (LoadLibrary).
#ifdef _WIN32
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void *;
#endif

/// @brief Load a shared library by name or path.
/// @param name Library name (resolved via the platform search path) or path.
/// @returns A non-null handle on success, or nullptr on failure. Call
///          last_library_error() for a human-readable failure reason.
inline LibraryHandle open_library(const char *name) {
#ifdef _WIN32
  return LoadLibraryA(name);
#else
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

/// @brief Close a library handle previously returned by open_library().
inline void close_library(LibraryHandle handle) {
  if (!handle)
    return;
#ifdef _WIN32
  FreeLibrary(handle);
#else
  dlclose(handle);
#endif
}

/// @brief Return a human-readable description of the last library error.
inline std::string last_library_error() {
#ifdef _WIN32
  return "error code " + std::to_string(GetLastError());
#else
  const char *err = dlerror();
  return err ? std::string(err) : std::string("unknown error");
#endif
}

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
