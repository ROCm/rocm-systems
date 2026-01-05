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

#include <string>

namespace rocstorage {

/// Error codes for rocstorage operations.
/// Values 0-10 are aligned with rocprofvis_dm_result_t C API for interoperability.
/// Values 100+ are rocstorage-specific extensions.
enum class error_code : int {
  // C API compatible codes (0-10)
  success = 0,
  unknown_error = 1,
  timeout = 2,
  not_loaded = 3,
  alloc_failure = 4,
  invalid_parameter = 5,
  db_access_failed = 6,
  invalid_property = 7,
  not_supported = 8,
  resource_busy = 9,
  db_abort = 10,

  // Extended codes (100+)
  file_not_found = 100,
  invalid_database = 101,
  constraint_violation = 102,
  schema_error = 103,
  query_error = 104,
  connection_failed = 105,
  index_out_of_bounds = 106,
};

/// Rich error type with context for rocstorage operations.
class error {
public:
  /// Construct an error with just an error code
  explicit error(error_code code) noexcept;

  /// Construct an error with code and message
  error(error_code code, std::string message);

  /// Construct an error with code, message, and query context
  error(error_code code, std::string message, std::string query);

  /// Construct an error with code, message, query context, and sqlite error code
  error(error_code code, std::string message, std::string query,
        int sqlite_code);

  /// Get the error code
  error_code code() const noexcept { return code_; }

  /// Get the error message
  const std::string &message() const noexcept { return message_; }

  /// Get the associated query (if any)
  const std::string &query() const noexcept { return query_; }

  /// Get the underlying SQLite error code (0 if not applicable)
  int sqlite_code() const noexcept { return sqlite_code_; }

  /// Convert to C API result code (for interoperability)
  int to_c_result() const noexcept;

  /// Get human-readable error description
  std::string to_string() const;

  /// Check if this represents a success code (useful when converting from C API)
  bool is_success() const noexcept { return code_ == error_code::success; }

private:
  error_code code_;
  std::string message_;
  std::string query_;
  int sqlite_code_ = 0;
};

} // namespace rocstorage
