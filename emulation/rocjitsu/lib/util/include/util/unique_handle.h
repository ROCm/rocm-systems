// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file unique_handle.h
/// @brief Move-only ownership for platform handles and handle-like resources.

#ifndef UTIL_UNIQUE_HANDLE_H_
#define UTIL_UNIQUE_HANDLE_H_

#include <utility>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
#else
#error "Unsupported platform: expected _WIN32 or __linux__"
#endif

namespace util {

/// @brief Move-only RAII owner for a handle described by @p Traits.
///
/// @details Traits must provide a handle_type and noexcept static functions
/// invalid(), is_valid(handle_type), and close(handle_type). Invalid handles are
/// normalized so APIs with multiple invalid sentinel values can be supported.
/// Cleanup is best-effort because destructors cannot report close failures.
template <typename Traits> class BasicUniqueHandle {
public:
  using handle_type = typename Traits::handle_type;

  BasicUniqueHandle() noexcept = default;
  explicit BasicUniqueHandle(handle_type handle) noexcept : handle_(normalize(handle)) {}
  ~BasicUniqueHandle() noexcept { reset(); }

  BasicUniqueHandle(const BasicUniqueHandle &) = delete;
  BasicUniqueHandle &operator=(const BasicUniqueHandle &) = delete;

  BasicUniqueHandle(BasicUniqueHandle &&other) noexcept : handle_(other.release()) {}
  BasicUniqueHandle &operator=(BasicUniqueHandle &&other) noexcept {
    if (this != &other)
      reset(other.release());
    return *this;
  }

  /// @brief Return whether this object owns a valid handle.
  [[nodiscard]] explicit operator bool() const noexcept { return Traits::is_valid(handle_); }

  /// @brief Return the owned handle, or Traits::invalid() if empty.
  [[nodiscard]] handle_type get() const noexcept { return handle_; }

  /// @brief Release ownership and return the handle.
  [[nodiscard]] handle_type release() noexcept { return std::exchange(handle_, Traits::invalid()); }

  /// @brief Close the current handle and take ownership of @p handle.
  void reset(handle_type handle = Traits::invalid()) noexcept {
    handle = normalize(handle);
    if (handle_ == handle)
      return;
    handle_type previous = std::exchange(handle_, handle);
    if (Traits::is_valid(previous))
      Traits::close(previous);
  }

private:
  static handle_type normalize(handle_type handle) noexcept {
    return Traits::is_valid(handle) ? handle : Traits::invalid();
  }

  handle_type handle_ = Traits::invalid();
};

namespace detail {

#ifdef _WIN32
struct NativeHandleTraits {
  using handle_type = HANDLE;

  static handle_type invalid() noexcept { return nullptr; }
  static bool is_valid(handle_type handle) noexcept {
    return handle != nullptr && handle != INVALID_HANDLE_VALUE;
  }
  static void close(handle_type handle) noexcept { static_cast<void>(::CloseHandle(handle)); }
};
#elif defined(__linux__)
struct NativeHandleTraits {
  using handle_type = int;

  static handle_type invalid() noexcept { return -1; }
  static bool is_valid(handle_type handle) noexcept { return handle >= 0; }
  static void close(handle_type handle) noexcept { static_cast<void>(::close(handle)); }
};
#endif

} // namespace detail

/// @brief Move-only owner for a native platform handle.
///
/// @details Owns a POSIX file descriptor on Linux and a Win32 HANDLE on
/// Windows. On Windows, both nullptr and INVALID_HANDLE_VALUE are treated as
/// invalid, and invalid values are normalized to nullptr.
using UniqueHandle = BasicUniqueHandle<detail::NativeHandleTraits>;

} // namespace util

#endif // UTIL_UNIQUE_HANDLE_H_
