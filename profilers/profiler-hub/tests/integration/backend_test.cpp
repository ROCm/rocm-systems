// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests that require a REAL SQLite engine.
//
// Two groups live here:
//
//  1. Backend tests (sqlite_backend_integration_test): drive the high-level
//     basic_sqlite_backend<sqlite_api_policy> for behavior the mock seam cannot
//     verify : real schema loading, real CREATE/INSERT/SELECT round-trips, real
//     backup (flush) file creation, and UUID discovery from a populated
//     rocpd.db. Pure backend control-flow (bind/step/reset ordering,
//     commit-vs-rollback, error translation) is covered by the mock-based
//     database_test.cpp in the unit target.
//
//  2. Policy contract tests (sqlite_api_policy_contract): exercise the
//     PRODUCTION SqlitePolicy (sqlite_api_policy) directly against a real
//     engine. This is the production-side mirror of mock_sqlite3_test.cpp: both
//     policies must expose the same member names, signatures, associated types,
//     and status constants, so each gets a direct contract test (the mock via
//     call forwarding, the real policy via real round-trips).

#include "data_storage/backends/sqlite_api_policy.hpp"
#include "data_storage/backends/sqlite_backend.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace
{

using namespace profiler_hub::data_storage;

class sqlite_backend_integration_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_database_path =
            "itest_database_" +
            std::to_string(
                ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
            ".db";
        m_uuid = "test_uuid_12345";
    }

    void TearDown() override { std::remove(m_database_path.c_str()); }

    std::string m_database_path;
    std::string m_uuid;
};

TEST_F(sqlite_backend_integration_test, initialize_schema_succeeds)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_NO_THROW(db->initialize_schema());
}

TEST_F(sqlite_backend_integration_test, double_initialize_schema_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema();
    EXPECT_THROW(db->initialize_schema(), std::runtime_error);
}

TEST_F(sqlite_backend_integration_test, execute_creates_table)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    ASSERT_NO_THROW(
        db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)"));

    // Confirm the table actually exists in the schema catalog : not merely that
    // execute() did not throw.
    struct row_count
    {
        int value{ 0 };
    };
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='test_tbl'",
        &row_count::value);
    EXPECT_EQ(reader().to_vector().front().value, 1);
}

TEST_F(sqlite_backend_integration_test, execute_inserts_data)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)");
    ASSERT_NO_THROW(db->execute("INSERT INTO test_tbl (id, name) VALUES (1, 'test')"));

    // Confirm the data actually exists in the table : not merely that execute() did not
    // throw.
    struct row_count
    {
        int value{ 0 };
    };
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM test_tbl WHERE id=1 AND name='test'", &row_count::value);
    EXPECT_EQ(reader().to_vector().front().value, 1);  // Should be exactly one row.
}

TEST_F(sqlite_backend_integration_test, invalid_query_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_THROW(db->execute("INVALID SQL SYNTAX"), std::runtime_error);
}

TEST_F(sqlite_backend_integration_test, flush_creates_file)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)");
    db->execute("INSERT INTO test_tbl (id) VALUES (7)");
    db->flush();

    // The flush target exists on disk ...
    FILE* file = std::fopen(m_database_path.c_str(), "r");
    ASSERT_NE(file, nullptr);
    std::fclose(file);

    // ... and is a real SQLite database whose schema and data survived the
    // flush. Reopen the on-disk file directly through the production policy
    // (independent of the backend that wrote it) and read the row back.
    sqlite_api_policy::database_t disk = nullptr;
    ASSERT_EQ(sqlite_api_policy::open(m_database_path.c_str(), &disk),
              sqlite_api_policy::result_ok);
    ASSERT_NE(disk, nullptr);

    sqlite_api_policy::statement_t stmt = nullptr;
    ASSERT_EQ(sqlite_api_policy::prepare(disk, "SELECT id FROM test_tbl", &stmt),
              sqlite_api_policy::result_ok);
    ASSERT_EQ(sqlite_api_policy::step(stmt), sqlite_api_policy::result_row);
    EXPECT_EQ(sqlite_api_policy::column_int(stmt, 0), 7);
    // Exactly one row was written.
    EXPECT_EQ(sqlite_api_policy::step(stmt), sqlite_api_policy::result_done);

    sqlite_api_policy::finalize(stmt);
    sqlite_api_policy::close(disk);
}

TEST_F(sqlite_backend_integration_test, initialized_schema_can_flush)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema();
    ASSERT_NO_THROW(db->flush());

    // The flush target exists on disk ...
    FILE* file = std::fopen(m_database_path.c_str(), "r");
    ASSERT_NE(file, nullptr);
    std::fclose(file);
}

TEST_F(sqlite_backend_integration_test, real_insert_and_count_round_trip)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->execute("CREATE TABLE rt_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto insert = db->create_write_statement_executor<int32_t>(
        "INSERT INTO rt_tbl (val) VALUES (?)");
    for(int i = 0; i < 5; ++i)
    {
        insert(i);
    }

    struct row_count
    {
        int value{ 0 };
    };
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM rt_tbl", &row_count::value);
    auto rows = reader().to_vector();
    ASSERT_FALSE(rows.empty());
    EXPECT_EQ(rows.front().value, 5);
}

TEST_F(sqlite_backend_integration_test, transaction_guard_commits_on_normal_exit)
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

    struct row_count
    {
        int value{ 0 };
    };
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM tx_tbl", &row_count::value);
    EXPECT_EQ(reader().to_vector().front().value, 5);
}

TEST_F(sqlite_backend_integration_test, transaction_guard_rolls_back_on_exception)
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

    struct row_count
    {
        int value{ 0 };
    };
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM tx_tbl", &row_count::value);
    EXPECT_EQ(reader().to_vector().front().value, 0);
}

TEST_F(sqlite_backend_integration_test, check_is_uuid_correct)
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

// ============================================================================
// Production policy contract tests
//
// Drive sqlite_api_policy's static operations directly against a real engine,
// using only the policy's own associated types (database_t/statement_t) and
// status constants (result_ok/result_row/result_done/column_null) : never raw
// sqlite3_* names or SQLITE_* macros. This pins the exact behavior the mock
// policy is required to mirror.
// ============================================================================

using policy_t = sqlite_api_policy;

class sqlite_api_policy_contract : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(policy_t::open(":memory:", &m_db), policy_t::result_ok);
        ASSERT_NE(m_db, nullptr);
    }

    void TearDown() override
    {
        if(m_db != nullptr)
        {
            policy_t::close(m_db);
        }
    }

    policy_t::database_t m_db{ nullptr };
};

TEST_F(sqlite_api_policy_contract, exec_creates_table_and_reports_ok)
{
    EXPECT_EQ(policy_t::exec(m_db, "CREATE TABLE t (id INTEGER, name TEXT)"),
              policy_t::result_ok);
}

TEST_F(sqlite_api_policy_contract, exec_reports_error_on_invalid_sql)
{
    EXPECT_NE(policy_t::exec(m_db, "NOT VALID SQL"), policy_t::result_ok);
    // errmsg/errstr must return non-empty diagnostics the backend can surface.
    EXPECT_FALSE(policy_t::errmsg(m_db).empty());
    EXPECT_FALSE(policy_t::errstr(policy_t::result_done).empty());
}

TEST_F(sqlite_api_policy_contract, bind_all_types_and_read_back_columns)
{
    ASSERT_EQ(policy_t::exec(m_db,
                             "CREATE TABLE t (i32 INTEGER, i64 INTEGER, dbl REAL, "
                             "txt TEXT, opt INTEGER)"),
              policy_t::result_ok);

    policy_t::statement_t ins = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "INSERT INTO t VALUES (?, ?, ?, ?, ?)", &ins),
              policy_t::result_ok);

    EXPECT_EQ(policy_t::bind_int(ins, 1, 42), policy_t::result_ok);
    EXPECT_EQ(policy_t::bind_int64(ins, 2, std::int64_t{ 9223372036854775807LL }),
              policy_t::result_ok);
    EXPECT_EQ(policy_t::bind_double(ins, 3, 3.5), policy_t::result_ok);
    EXPECT_EQ(policy_t::bind_text(ins, 4, std::string_view{ "hello" }),
              policy_t::result_ok);
    EXPECT_EQ(policy_t::bind_null(ins, 5), policy_t::result_ok);
    EXPECT_EQ(policy_t::step(ins), policy_t::result_done);
    policy_t::finalize(ins);

    policy_t::statement_t sel = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "SELECT i32, i64, dbl, txt, opt FROM t", &sel),
              policy_t::result_ok);
    ASSERT_EQ(policy_t::step(sel), policy_t::result_row);

    EXPECT_EQ(policy_t::column_int(sel, 0), 42);
    EXPECT_EQ(policy_t::column_int64(sel, 1), std::int64_t{ 9223372036854775807LL });
    EXPECT_DOUBLE_EQ(policy_t::column_double(sel, 2), 3.5);
    EXPECT_EQ(policy_t::column_text(sel, 3), "hello");
    EXPECT_EQ(policy_t::column_type(sel, 4), policy_t::column_null);

    EXPECT_EQ(policy_t::step(sel), policy_t::result_done);
    policy_t::finalize(sel);
}

TEST_F(sqlite_api_policy_contract, column_count_and_name_describe_result_set)
{
    ASSERT_EQ(policy_t::exec(m_db, "CREATE TABLE t (id INTEGER, label TEXT)"),
              policy_t::result_ok);

    policy_t::statement_t sel = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "SELECT id, label FROM t", &sel),
              policy_t::result_ok);

    EXPECT_EQ(policy_t::column_count(sel), 2);
    EXPECT_EQ(policy_t::column_name(sel, 0), "id");
    EXPECT_EQ(policy_t::column_name(sel, 1), "label");

    policy_t::finalize(sel);
}

TEST_F(sqlite_api_policy_contract, column_text_on_null_returns_empty_string)
{
    ASSERT_EQ(policy_t::exec(m_db, "CREATE TABLE t (txt TEXT)"), policy_t::result_ok);
    ASSERT_EQ(policy_t::exec(m_db, "INSERT INTO t (txt) VALUES (NULL)"),
              policy_t::result_ok);

    policy_t::statement_t sel = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "SELECT txt FROM t", &sel), policy_t::result_ok);
    ASSERT_EQ(policy_t::step(sel), policy_t::result_row);

    EXPECT_EQ(policy_t::column_type(sel, 0), policy_t::column_null);
    EXPECT_EQ(policy_t::column_text(sel, 0), std::string{});

    policy_t::finalize(sel);
}

TEST_F(sqlite_api_policy_contract, reset_and_clear_bindings_allow_statement_reuse)
{
    ASSERT_EQ(policy_t::exec(m_db, "CREATE TABLE t (val INTEGER)"), policy_t::result_ok);

    policy_t::statement_t ins = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "INSERT INTO t (val) VALUES (?)", &ins),
              policy_t::result_ok);

    for(int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(policy_t::bind_int(ins, 1, i), policy_t::result_ok);
        EXPECT_EQ(policy_t::step(ins), policy_t::result_done);
        EXPECT_EQ(policy_t::reset(ins), policy_t::result_ok);
        EXPECT_EQ(policy_t::clear_bindings(ins), policy_t::result_ok);
    }
    policy_t::finalize(ins);

    policy_t::statement_t sel = nullptr;
    ASSERT_EQ(policy_t::prepare(m_db, "SELECT COUNT(*) FROM t", &sel),
              policy_t::result_ok);
    ASSERT_EQ(policy_t::step(sel), policy_t::result_row);
    EXPECT_EQ(policy_t::column_int(sel, 0), 3);
    policy_t::finalize(sel);
}

TEST_F(sqlite_api_policy_contract, backup_to_file_writes_a_real_database_file)
{
    ASSERT_EQ(policy_t::exec(m_db, "CREATE TABLE t (id INTEGER)"), policy_t::result_ok);

    const std::string out_path = "policy_contract_backup.db";
    std::remove(out_path.c_str());

    EXPECT_EQ(policy_t::backup_to_file(m_db, out_path.c_str()), policy_t::result_ok);
    EXPECT_TRUE(std::filesystem::exists(out_path));

    std::remove(out_path.c_str());
}

}  // namespace
