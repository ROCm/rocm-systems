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

#include "unified_db_adapter.hpp"

#include <spdlog/spdlog.h>

namespace rocstorage {

UnifiedDatabaseAdapter::UnifiedDatabaseAdapter(
    std::shared_ptr<data_storage::database> db)
    : RocProfVis::DataModel::RocprofDatabase(db->get_path().c_str()),
      unified_db_(std::move(db)) {
  spdlog::debug("UnifiedDatabaseAdapter created with path: {}",
                unified_db_->get_path());
}

UnifiedDatabaseAdapter::~UnifiedDatabaseAdapter() {
  spdlog::debug("UnifiedDatabaseAdapter destroyed");
}

rocprofvis_dm_result_t UnifiedDatabaseAdapter::Open() {
  if (opened_) {
    // Already opened, return success
    return kRocProfVisDmResultSuccess;
  }

  // The unified database is already open and the database file exists.
  // Call the parent's Open() to initialize its connection pool.
  rocprofvis_dm_result_t result =
      RocProfVis::DataModel::RocprofDatabase::Open();

  if (result == kRocProfVisDmResultSuccess) {
    opened_ = true;
    spdlog::debug("UnifiedDatabaseAdapter opened successfully");
  } else {
    spdlog::error("UnifiedDatabaseAdapter failed to open (result: {})",
                  static_cast<int>(result));
  }

  return result;
}

rocprofvis_dm_result_t UnifiedDatabaseAdapter::Close() {
  if (!opened_) {
    return kRocProfVisDmResultSuccess;
  }

  // Call parent's Close() to release the connection pool
  rocprofvis_dm_result_t result =
      RocProfVis::DataModel::RocprofDatabase::Close();

  opened_ = false;
  spdlog::debug("UnifiedDatabaseAdapter closed");

  return result;
}

} // namespace rocstorage
