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

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <string>

namespace {

using namespace rocstorage::data_storage;

class database_test : public ::testing::Test {
protected:
  void SetUp() override {
    m_database_path =
        "test_database_" +
        std::to_string(
            ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
        ".db";
    m_uuid = "test_uuid_12345";
  }

  void TearDown() override { std::remove(m_database_path.c_str()); }

  std::string m_database_path;
  std::string m_uuid;
};

TEST_F(database_test, construct_database_instance) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  ASSERT_NE(db, nullptr);
}

TEST_F(database_test, get_uuid_returns_correct_value) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  EXPECT_EQ(db->get_uuid(), m_uuid);
}

TEST_F(database_test, initialize_schema_succeeds) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  EXPECT_NO_THROW(db->initialize_schema());
}

TEST_F(database_test, double_initialize_schema_throws) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->initialize_schema();
  EXPECT_THROW(db->initialize_schema(), std::runtime_error);
}

TEST_F(database_test, execute_query_creates_table) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  EXPECT_NO_THROW(db->execute_query(
      "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)"));
}

TEST_F(database_test, execute_query_inserts_data) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)");
  EXPECT_NO_THROW(
      db->execute_query("INSERT INTO test_tbl (id, name) VALUES (1, 'test')"));
}

TEST_F(database_test, get_last_insert_id_returns_correct_value) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)");
  db->execute_query("INSERT INTO test_tbl (name) VALUES ('first')");
  auto first_id = db->get_last_insert_id();
  EXPECT_EQ(first_id, 1u);

  db->execute_query("INSERT INTO test_tbl (name) VALUES ('second')");
  auto second_id = db->get_last_insert_id();
  EXPECT_EQ(second_id, 2u);
}

TEST_F(database_test, invalid_query_throws) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  EXPECT_THROW(db->execute_query("INVALID SQL SYNTAX"), std::runtime_error);
}

TEST_F(database_test, flush_creates_file) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)");
  db->flush();

  FILE *file = std::fopen(m_database_path.c_str(), "r");
  ASSERT_NE(file, nullptr);
  std::fclose(file);
}

TEST_F(database_test, double_flush_throws) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)");
  db->flush();
  EXPECT_THROW(db->flush(), std::runtime_error);
}

TEST_F(database_test, create_statement_executor_with_int32) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE int32_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

  auto executor = db->create_statement_executor<int32_t>(
      "INSERT INTO int32_tbl (val) VALUES (?)");
  EXPECT_NO_THROW(executor(42));
  EXPECT_NO_THROW(executor(-100));
  EXPECT_EQ(db->get_last_insert_id(), 2u);
}

TEST_F(database_test, create_statement_executor_with_int64) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE int64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

  auto executor = db->create_statement_executor<int64_t>(
      "INSERT INTO int64_tbl (val) VALUES (?)");
  EXPECT_NO_THROW(executor(int64_t{9223372036854775807LL}));
}

TEST_F(database_test, create_statement_executor_with_uint64) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE uint64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

  auto executor = db->create_statement_executor<uint64_t>(
      "INSERT INTO uint64_tbl (val) VALUES (?)");
  EXPECT_NO_THROW(executor(uint64_t{12345678901234567890ULL}));
}

TEST_F(database_test, create_statement_executor_with_double) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE double_tbl (id INTEGER PRIMARY KEY, val REAL)");

  auto executor = db->create_statement_executor<double>(
      "INSERT INTO double_tbl (val) VALUES (?)");
  EXPECT_NO_THROW(executor(3.14159265359));
  EXPECT_NO_THROW(executor(-2.71828));
}

TEST_F(database_test, create_statement_executor_with_text) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query("CREATE TABLE text_tbl (id INTEGER PRIMARY KEY, val TEXT)");

  auto executor = db->create_statement_executor<const char *>(
      "INSERT INTO text_tbl (val) VALUES (?)");
  EXPECT_NO_THROW(executor("hello world"));
  EXPECT_NO_THROW(executor(""));
}

TEST_F(database_test, create_statement_executor_with_multiple_params) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE multi_tbl (id INTEGER PRIMARY KEY, int_val INTEGER, "
      "real_val REAL, text_val TEXT)");

  auto executor = db->create_statement_executor<int32_t, double, const char *>(
      "INSERT INTO multi_tbl (int_val, real_val, text_val) VALUES (?, ?, ?)");
  EXPECT_NO_THROW(executor(42, 3.14, "test"));
  EXPECT_NO_THROW(executor(-1, 0.0, "another"));
  EXPECT_EQ(db->get_last_insert_id(), 2u);
}

TEST_F(database_test, statement_executor_can_be_reused) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->execute_query(
      "CREATE TABLE reuse_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

  auto executor = db->create_statement_executor<int32_t>(
      "INSERT INTO reuse_tbl (val) VALUES (?)");

  for (int i = 0; i < 100; ++i) {
    EXPECT_NO_THROW(executor(i));
  }
  EXPECT_EQ(db->get_last_insert_id(), 100u);
}

TEST_F(database_test, database_with_initialized_schema_can_flush) {
  auto db = std::make_unique<database>(m_database_path, m_uuid);
  db->initialize_schema();
  EXPECT_NO_THROW(db->flush());

  FILE *file = std::fopen(m_database_path.c_str(), "r");
  ASSERT_NE(file, nullptr);
  std::fclose(file);
}

TEST_F(database_test, multiple_databases_independent) {
  std::string path1 = m_database_path + "_1";
  std::string path2 = m_database_path + "_2";

  auto db1 = std::make_unique<database>(path1, "uuid1");
  auto db2 = std::make_unique<database>(path2, "uuid2");

  EXPECT_EQ(db1->get_uuid(), "uuid1");
  EXPECT_EQ(db2->get_uuid(), "uuid2");

  db1->execute_query("CREATE TABLE tbl1 (id INTEGER PRIMARY KEY)");
  db2->execute_query("CREATE TABLE tbl2 (id INTEGER PRIMARY KEY)");

  db1->execute_query("INSERT INTO tbl1 (id) VALUES (1)");
  db2->execute_query("INSERT INTO tbl2 (id) VALUES (100)");

  EXPECT_EQ(db1->get_last_insert_id(), 1u);
  EXPECT_EQ(db2->get_last_insert_id(), 100u);

  std::remove(path1.c_str());
  std::remove(path2.c_str());
}

} // namespace
