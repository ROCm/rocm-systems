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

#include "data_storage/database.hpp"
#include "unified_db_adapter.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace {

using rocstorage::UnifiedDatabaseAdapter;
using rocstorage::data_storage::database;
using rocstorage::data_storage::database_mode;

class UnifiedDatabaseAdapterTest : public ::testing::Test {
protected:
  void SetUp() override {
    m_database_path =
        "test_unified_adapter_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
        ".db";
    m_uuid = "testuuid12345";
  }

  void TearDown() override { std::remove(m_database_path.c_str()); }

  std::string m_database_path;
  std::string m_uuid;
};

TEST_F(UnifiedDatabaseAdapterTest, ConstructWithValidDatabase) {
  // Create a database in WAL mode (so it exists on disk)
  auto db = std::make_shared<database>(m_database_path, m_uuid, database_mode::wal);
  db->initialize_schema();

  // Create adapter from the database
  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);
  ASSERT_NE(adapter, nullptr);

  // The unified database should be accessible
  EXPECT_EQ(adapter->unified_database(), db);
}

TEST_F(UnifiedDatabaseAdapterTest, OpenSucceedsAfterFlush) {
  // Create database and flush to disk
  {
    auto db = std::make_unique<database>(m_database_path, m_uuid);
    db->initialize_schema();
    db->flush();
  }

  // Open from disk in read-only mode
  auto result = database::open_readonly(m_database_path);
  ASSERT_TRUE(result);

  // Convert unique_ptr to shared_ptr (shared_ptr can take ownership of unique_ptr)
  std::shared_ptr<database> db(std::move(result.value()));
  ASSERT_NE(db, nullptr);

  // Create adapter with the shared read-only database
  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);
  ASSERT_NE(adapter, nullptr);

  // Verify the adapter has the correct database
  EXPECT_EQ(adapter->unified_database(), db);
}

TEST_F(UnifiedDatabaseAdapterTest, OpenAndCloseSucceeds) {
  // Create a database in WAL mode
  auto db = std::make_shared<database>(m_database_path, m_uuid, database_mode::wal);
  db->initialize_schema();

  // Create adapter and open
  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);
  EXPECT_EQ(adapter->Open(), kRocProfVisDmResultSuccess);

  // Close should also succeed
  EXPECT_EQ(adapter->Close(), kRocProfVisDmResultSuccess);
}

TEST_F(UnifiedDatabaseAdapterTest, DoubleOpenSucceeds) {
  // Create a database in WAL mode
  auto db = std::make_shared<database>(m_database_path, m_uuid, database_mode::wal);
  db->initialize_schema();

  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);

  // First open
  EXPECT_EQ(adapter->Open(), kRocProfVisDmResultSuccess);

  // Second open should also succeed (idempotent)
  EXPECT_EQ(adapter->Open(), kRocProfVisDmResultSuccess);
}

TEST_F(UnifiedDatabaseAdapterTest, AdapterSharesDatabaseConnection) {
  // Create a database in WAL mode
  auto db = std::make_shared<database>(m_database_path, m_uuid, database_mode::wal);
  db->initialize_schema();

  // Create and open adapter
  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);
  EXPECT_EQ(adapter->Open(), kRocProfVisDmResultSuccess);

  // The unified database should still be accessible and working
  EXPECT_NO_THROW(
      db->execute_query("SELECT name FROM sqlite_master WHERE type='table'"));

  // Cleanup
  adapter->Close();
}

TEST_F(UnifiedDatabaseAdapterTest, AdapterRetrievesDatabasePath) {
  // Create a database in WAL mode
  auto db = std::make_shared<database>(m_database_path, m_uuid, database_mode::wal);
  db->initialize_schema();

  auto adapter = std::make_unique<UnifiedDatabaseAdapter>(db);

  // The adapter should have the same path as the unified database
  EXPECT_EQ(adapter->unified_database()->get_path(), m_database_path);
}

} // namespace
