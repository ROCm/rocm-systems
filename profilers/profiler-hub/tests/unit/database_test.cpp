// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "data_storage/backends/sqlite_backend.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub::data_storage;

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

// ============================================================================
// transaction_guard tests
// ============================================================================
//
// transaction_guard issues BEGIN on construction and either COMMIT (when the
// scope exits normally) or ROLLBACK (when std::uncaught_exceptions() grew
// during the scope) on destruction. The guard is reachable through the public
// sqlite_backend::begin_transaction() factory, which returns it by value.

namespace
{

struct row_count
{
    int value{ 0 };
};

int
query_row_count(const std::shared_ptr<sqlite_backend>& db, const std::string& table)
{
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM " + table, &row_count::value);
    auto rows = reader().to_vector();
    return rows.empty() ? 0 : rows.front().value;
}

}  // namespace

TEST_F(sqlite_backend_test, transaction_guard_commits_on_normal_exit)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE tx_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto insert = db->create_write_statement_executor<int32_t>(
        "INSERT INTO tx_tbl (val) VALUES (?)");

    {
        auto guard = db->begin_transaction();
        for(int i = 0; i < 5; ++i)
        {
            insert(i);
        }
    }

    EXPECT_EQ(query_row_count(db, "tx_tbl"), 5);
}

TEST_F(sqlite_backend_test, transaction_guard_rolls_back_on_exception)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE tx_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto insert = db->create_write_statement_executor<int32_t>(
        "INSERT INTO tx_tbl (val) VALUES (?)");

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            for(int i = 0; i < 5; ++i)
            {
                insert(i);
            }
            throw std::runtime_error("simulated mid-transaction failure");
        },
        std::runtime_error);

    EXPECT_EQ(query_row_count(db, "tx_tbl"), 0);
}

TEST_F(sqlite_backend_test, transaction_guard_independent_scopes)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE tx_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto insert = db->create_write_statement_executor<int32_t>(
        "INSERT INTO tx_tbl (val) VALUES (?)");

    {
        auto guard = db->begin_transaction();
        for(int i = 0; i < 3; ++i)
        {
            insert(i);
        }
    }

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            for(int i = 100; i < 105; ++i)
            {
                insert(i);
            }
            throw std::runtime_error("rollback this scope only");
        },
        std::runtime_error);

    EXPECT_EQ(query_row_count(db, "tx_tbl"), 3);
}

}  // namespace
