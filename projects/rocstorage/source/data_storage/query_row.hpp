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

#include <sqlite3.h>

#include <cstdint>
#include <functional>
#include <string_view>

namespace rocstorage {
namespace data_storage {

/// Accessor for a single row from a query result
///
/// This class provides type-safe access to column values from a SQLite query
/// result. It does not own the underlying statement and is only valid during
/// the callback invocation.
class query_row {
public:
  /// Construct a query_row from a SQLite statement
  explicit query_row(sqlite3_stmt *stmt) noexcept : stmt_(stmt) {}

  /// Get the number of columns in this row
  int column_count() const noexcept { return sqlite3_column_count(stmt_); }

  /// Get the name of a column by index (0-based)
  std::string_view column_name(int column) const noexcept {
    const char *name = sqlite3_column_name(stmt_, column);
    return name ? std::string_view(name) : std::string_view();
  }

  /// Get the SQLite data type of a column
  /// @return One of: SQLITE_INTEGER, SQLITE_FLOAT, SQLITE_TEXT, SQLITE_BLOB, SQLITE_NULL
  int column_type(int column) const noexcept {
    return sqlite3_column_type(stmt_, column);
  }

  /// Check if a column value is NULL
  bool is_null(int column) const noexcept {
    return sqlite3_column_type(stmt_, column) == SQLITE_NULL;
  }

  /// Get a column value as int32
  int32_t get_int(int column) const noexcept {
    return sqlite3_column_int(stmt_, column);
  }

  /// Get a column value as int64
  int64_t get_int64(int column) const noexcept {
    return sqlite3_column_int64(stmt_, column);
  }

  /// Get a column value as double
  double get_double(int column) const noexcept {
    return sqlite3_column_double(stmt_, column);
  }

  /// Get a column value as text (string_view)
  /// Returns empty string_view if NULL
  std::string_view get_text(int column) const noexcept {
    const unsigned char *text = sqlite3_column_text(stmt_, column);
    if (!text) {
      return std::string_view();
    }
    int bytes = sqlite3_column_bytes(stmt_, column);
    return std::string_view(reinterpret_cast<const char *>(text), bytes);
  }

  /// Get a column value as blob (returns pointer and size)
  /// Returns nullptr if NULL
  const void *get_blob(int column, int &size) const noexcept {
    size = sqlite3_column_bytes(stmt_, column);
    return sqlite3_column_blob(stmt_, column);
  }

  /// Get the underlying SQLite statement (for advanced usage)
  sqlite3_stmt *raw_stmt() const noexcept { return stmt_; }

private:
  sqlite3_stmt *stmt_;
};

/// Callback type for iterating over query results
/// Return true to continue iteration, false to stop
using row_callback = std::function<bool(const query_row &)>;

} // namespace data_storage
} // namespace rocstorage
