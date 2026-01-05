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

#include "data_storage/database.hpp"
#include "rocprofvis_db_rocprof.h"

#include <memory>

namespace rocstorage {

/// Adapter that allows RocprofDatabase to use a shared data_storage::database
///
/// This adapter enables the unified database approach where both writer and
/// reader can share the same underlying database connection. It inherits from
/// RocprofDatabase to reuse all the complex query building and parsing logic,
/// but overrides connection management to use the provided database.
///
/// Usage:
///   auto db = std::make_shared<data_storage::database>(...);
///   auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);
///   // adapter can now be used as a Database for the reader
class UnifiedDatabaseAdapter : public RocProfVis::DataModel::RocprofDatabase {
public:
  /// Construct adapter from a shared database
  /// @param db The unified database to use for queries
  explicit UnifiedDatabaseAdapter(std::shared_ptr<data_storage::database> db);

  ~UnifiedDatabaseAdapter() override;

  // Non-copyable, non-movable
  UnifiedDatabaseAdapter(const UnifiedDatabaseAdapter&) = delete;
  UnifiedDatabaseAdapter& operator=(const UnifiedDatabaseAdapter&) = delete;
  UnifiedDatabaseAdapter(UnifiedDatabaseAdapter&&) = delete;
  UnifiedDatabaseAdapter& operator=(UnifiedDatabaseAdapter&&) = delete;

  /// Open the adapter for reading
  /// Initializes the parent's connection pool using the existing database file.
  rocprofvis_dm_result_t Open() override;

  /// Close the adapter
  /// Releases the parent's connection pool but does not close the underlying database.
  rocprofvis_dm_result_t Close() override;

  /// Get the underlying unified database
  std::shared_ptr<data_storage::database> unified_database() const {
    return unified_db_;
  }

private:
  std::shared_ptr<data_storage::database> unified_db_;
  bool opened_ = false;
};

} // namespace rocstorage
