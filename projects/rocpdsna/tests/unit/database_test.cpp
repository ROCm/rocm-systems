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

#include "data_storage/backends/sqlite_backend.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

namespace
{

using namespace rocpdsna::data_storage;

class sqlite_backend_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
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

TEST_F(sqlite_backend_test, construct_instance)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    ASSERT_NE(db, nullptr);
}

TEST_F(sqlite_backend_test, get_uuid_returns_correct_value)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_EQ(db->get_uuid(), m_uuid);
}

TEST_F(sqlite_backend_test, initialize_schema_succeeds)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_NO_THROW(db->initialize_schema());
}

TEST_F(sqlite_backend_test, double_initialize_schema_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema();
    EXPECT_THROW(db->initialize_schema(), std::runtime_error);
}

TEST_F(sqlite_backend_test, execute_creates_table)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_NO_THROW(
        db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)"));
}

TEST_F(sqlite_backend_test, execute_inserts_data)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)");
    EXPECT_NO_THROW(db->execute("INSERT INTO test_tbl (id, name) VALUES (1, 'test')"));
}

TEST_F(sqlite_backend_test, invalid_query_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_THROW(db->execute("INVALID SQL SYNTAX"), std::runtime_error);
}

TEST_F(sqlite_backend_test, flush_creates_file)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)");
    db->flush();

    FILE* file = std::fopen(m_database_path.c_str(), "r");
    ASSERT_NE(file, nullptr);
    std::fclose(file);
}

TEST_F(sqlite_backend_test, double_flush_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)");
    db->flush();
    EXPECT_THROW(db->flush(), std::runtime_error);
}

TEST_F(sqlite_backend_test, create_statement_executor_with_int32)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE int32_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int32_t>(
        "INSERT INTO int32_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(42));
    EXPECT_NO_THROW(executor(-100));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_int64)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE int64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int64_t>(
        "INSERT INTO int64_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(int64_t{ 9223372036854775807LL }));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_uint64)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE uint64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<uint64_t>(
        "INSERT INTO uint64_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(uint64_t{ 12345678901234567890ULL }));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_double)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE double_tbl (id INTEGER PRIMARY KEY, val REAL)");

    auto executor = db->create_write_statement_executor<double>(
        "INSERT INTO double_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(3.14159265359));
    EXPECT_NO_THROW(executor(-2.71828));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_text)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE text_tbl (id INTEGER PRIMARY KEY, val TEXT)");

    auto executor = db->create_write_statement_executor<const char*>(
        "INSERT INTO text_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor("hello world"));
    EXPECT_NO_THROW(executor(""));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_multiple_params)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE multi_tbl (id INTEGER PRIMARY KEY, int_val INTEGER, "
                "real_val REAL, text_val TEXT)");

    auto executor = db->create_write_statement_executor<int32_t, double, const char*>(
        "INSERT INTO multi_tbl (int_val, real_val, text_val) VALUES (?, ?, ?)");
    EXPECT_NO_THROW(executor(42, 3.14, "test"));
    EXPECT_NO_THROW(executor(-1, 0.0, "another"));
}

TEST_F(sqlite_backend_test, statement_executor_can_be_reused)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE reuse_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int32_t>(
        "INSERT INTO reuse_tbl (val) VALUES (?)");

    for(int i = 0; i < 100; ++i)
    {
        EXPECT_NO_THROW(executor(i));
    }
}

TEST_F(sqlite_backend_test, initialized_schema_can_flush)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema();
    EXPECT_NO_THROW(db->flush());

    FILE* file = std::fopen(m_database_path.c_str(), "r");
    ASSERT_NE(file, nullptr);
    std::fclose(file);
}

TEST_F(sqlite_backend_test, multiple_backends_independent)
{
    std::string path1 = m_database_path + "_1";
    std::string path2 = m_database_path + "_2";

    auto db1 = sqlite_backend::create(path1, "uuid1");
    auto db2 = sqlite_backend::create(path2, "uuid2");

    EXPECT_EQ(db1->get_uuid(), "uuid1");
    EXPECT_EQ(db2->get_uuid(), "uuid2");

    db1->execute("CREATE TABLE tbl1 (id INTEGER PRIMARY KEY)");
    db2->execute("CREATE TABLE tbl2 (id INTEGER PRIMARY KEY)");

    db1->execute("INSERT INTO tbl1 (id) VALUES (1)");
    db2->execute("INSERT INTO tbl2 (id) VALUES (100)");

    std::remove(path1.c_str());
    std::remove(path2.c_str());
}

TEST_F(sqlite_backend_test, check_is_uuid_correct)
{
    std::string database_path = ROCPD_DB_PATH;
    if(!std::filesystem::exists(database_path))
    {
        ASSERT_TRUE(false) << "Database file does not exist.";
    }
    const auto* expected_uuid = "3224963d0bd2e790224c3b2186eb8bd0";
    auto        db            = sqlite_backend::create(database_path, "");
    EXPECT_EQ(db->get_uuid(), expected_uuid);
}

TEST_F(sqlite_backend_test, statement_outlives_backend)
{
    std::function<void(int32_t)> executor;
    {
        auto db = sqlite_backend::create(m_database_path, m_uuid);
        db->execute("CREATE TABLE outlive_tbl (id INTEGER PRIMARY KEY, val INTEGER)");
        executor = db->create_write_statement_executor<int32_t>(
            "INSERT INTO outlive_tbl (val) VALUES (?)");
        // db goes out of scope here, but executor keeps it alive via shared_ptr
    }
    // This should still work -- the connection stays alive through the lambda capture
    EXPECT_NO_THROW(executor(42));
}

TEST_F(sqlite_backend_test, on_disk_engages_wal_journal_mode)
{
    auto db = sqlite_backend::create(
        m_database_path, m_uuid, sqlite_backend::storage_mode_t::on_disk);

    // Open a fresh handle to the same file and read the journal mode back.
    // sqlite_backend's constructor already raises if WAL did not engage,
    // so reaching this point implies the active mode is WAL; the second
    // open re-confirms it is persisted on disk.
    sqlite3* handle = nullptr;
    ASSERT_EQ(sqlite3_open(m_database_path.c_str(), &handle), SQLITE_OK);
    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(handle, "PRAGMA journal_mode", -1, &stmt, nullptr),
              SQLITE_OK);
    ASSERT_EQ(sqlite3_step(stmt), SQLITE_ROW);
    std::string mode{ reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)) };
    sqlite3_finalize(stmt);
    sqlite3_close(handle);
    EXPECT_EQ(mode, "wal");
}

}  // namespace
