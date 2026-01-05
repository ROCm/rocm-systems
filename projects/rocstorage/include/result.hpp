// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocstorage/error.hpp>

#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace rocstorage {

/// Result type for operations that can fail (similar to std::expected<T, E>)
///
/// Usage:
///   result<int> divide(int a, int b) {
///     if (b == 0) return error(error_code::invalid_parameter, "division by zero");
///     return a / b;
///   }
///
///   auto r = divide(10, 2);
///   if (r) {
///     std::cout << "Result: " << r.value() << std::endl;
///   } else {
///     std::cerr << "Error: " << r.get_error().to_string() << std::endl;
///   }
template <typename T> class result {
public:
  using value_type = T;
  using error_type = error;

  /// Construct with a value (success)
  result(T value) : data_(std::move(value)) {}

  /// Construct with an error (failure)
  result(error err) : data_(std::move(err)) {}

  /// Construct with error code and message (convenience)
  result(error_code code, std::string message = {})
      : data_(error(code, std::move(message))) {}

  // Default copy/move operations
  result(const result &) = default;
  result(result &&) noexcept = default;
  result &operator=(const result &) = default;
  result &operator=(result &&) noexcept = default;

  /// Check if the result contains a value
  explicit operator bool() const noexcept {
    return std::holds_alternative<T>(data_);
  }

  /// Check if the result contains a value
  bool has_value() const noexcept { return std::holds_alternative<T>(data_); }

  /// Check if the result contains an error
  bool has_error() const noexcept {
    return std::holds_alternative<error>(data_);
  }

  /// Get the value (throws if error)
  T &value() & {
    if (!has_value()) {
      throw std::runtime_error("result::value() called on error: " +
                               std::get<error>(data_).to_string());
    }
    return std::get<T>(data_);
  }

  const T &value() const & {
    if (!has_value()) {
      throw std::runtime_error("result::value() called on error: " +
                               std::get<error>(data_).to_string());
    }
    return std::get<T>(data_);
  }

  T &&value() && {
    if (!has_value()) {
      throw std::runtime_error("result::value() called on error: " +
                               std::get<error>(data_).to_string());
    }
    return std::move(std::get<T>(data_));
  }

  /// Get the value or a default
  template <typename U> T value_or(U &&default_value) const & {
    if (has_value()) {
      return std::get<T>(data_);
    }
    return static_cast<T>(std::forward<U>(default_value));
  }

  template <typename U> T value_or(U &&default_value) && {
    if (has_value()) {
      return std::move(std::get<T>(data_));
    }
    return static_cast<T>(std::forward<U>(default_value));
  }

  /// Get the error (throws std::logic_error if result contains a value)
  error &get_error() & {
    if (!has_error()) {
      throw std::logic_error("result::get_error() called on value");
    }
    return std::get<error>(data_);
  }

  const error &get_error() const & {
    if (!has_error()) {
      throw std::logic_error("result::get_error() called on value");
    }
    return std::get<error>(data_);
  }

  /// Pointer-like access to value
  T *operator->() { return &value(); }
  const T *operator->() const { return &value(); }

  /// Reference access to value
  T &operator*() & { return value(); }
  const T &operator*() const & { return value(); }
  T &&operator*() && { return std::move(*this).value(); }

  /// Transform the value if present
  template <typename F> auto map(F &&f) const & -> result<decltype(f(value()))> {
    using U = decltype(f(value()));
    if (has_value()) {
      return result<U>(f(std::get<T>(data_)));
    }
    return result<U>(std::get<error>(data_));
  }

  /// Chain another operation that returns a result
  template <typename F> auto and_then(F &&f) const & -> decltype(f(value())) {
    if (has_value()) {
      return f(std::get<T>(data_));
    }
    return decltype(f(value()))(std::get<error>(data_));
  }

private:
  std::variant<T, error> data_;
};

/// Specialization for void (status-only operations)
template <> class result<void> {
public:
  using value_type = void;
  using error_type = error;

  /// Construct success
  result() = default;

  /// Construct with an error (failure)
  result(error err) : err_(std::move(err)) {}

  /// Construct with error code and message (convenience)
  result(error_code code, std::string message = {})
      : err_(error(code, std::move(message))) {}

  // Default copy/move operations
  result(const result &) = default;
  result(result &&) noexcept = default;
  result &operator=(const result &) = default;
  result &operator=(result &&) noexcept = default;

  /// Check if the result is success
  explicit operator bool() const noexcept { return !err_.has_value(); }

  /// Check if the result is success
  bool has_value() const noexcept { return !err_.has_value(); }

  /// Check if the result contains an error
  bool has_error() const noexcept { return err_.has_value(); }

  /// Get the error (throws std::logic_error if result is success)
  const error &get_error() const & {
    if (!err_.has_value()) {
      throw std::logic_error("result::get_error() called on success");
    }
    return *err_;
  }

  error &get_error() & {
    if (!err_.has_value()) {
      throw std::logic_error("result::get_error() called on success");
    }
    return *err_;
  }

private:
  std::optional<error> err_;
};

/// Alias for void result (status-only operations)
using status = result<void>;

/// Helper to create a success result
template <typename T> result<T> ok(T &&value) {
  return result<T>(std::forward<T>(value));
}

/// Helper to create a void success result
inline status ok() { return status(); }

/// Helper to create an error result
template <typename T = void>
result<T> err(error_code code, std::string message = {}) {
  return result<T>(code, std::move(message));
}

/// Helper to create an error result from an error object
template <typename T = void> result<T> err(error e) {
  return result<T>(std::move(e));
}

} // namespace rocstorage
