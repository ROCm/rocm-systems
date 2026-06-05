// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Backend behavior tests driven entirely through the mockable sqlite3 seam.
//
// These tests instantiate basic_sqlite_backend<mock_sqlite3> so the backend's
// control flow (open/prepare on construction, bind/step/reset ordering on
// writes, column extraction on reads, commit-vs-rollback on transactions,
// backup on flush, and error translation) is verified WITHOUT a real SQLite
// database. No real sqlite3 symbol is linked or called; every operation is
// recorded by a gmock sqlite3_recorder.
//
// Behavior that is inherently tied to a real SQL engine (actual table
// creation, real schema loading, real UUID discovery from a populated rocpd.db)
// lives in the integration test target instead.

#include "data_storage/backends/sqlite_backend_impl.hpp"
#include "mocks/mock_sqlite3.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <string>

namespace
{

using namespace profiler_hub::data_storage;
using namespace profiler_hub::data_storage::mocks;

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrEq;

using mock_backend = basic_sqlite_backend<mock_sqlite3>;

// =============================================================================
// Fixture
//
// Installs default behaviors on a NiceMock recorder so that construction and
// destruction of a backend succeed with no per-test boilerplate. Individual
// tests layer EXPECT_CALL expectations on top to assert the behavior under
// test. The recorder and its scoped_bind outlive every backend created in a
// test body, so backend destructors still find an active recorder.
// =============================================================================
class sqlite_backend_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_database_path =
            "mock_backend_" +
            std::to_string(
                ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
            ".db";
        m_uuid = "test_uuid_12345";

        ON_CALL(m_recorder, open(_, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(&m_connection), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, prepare(_, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_statement), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, exec(_, _)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, backup_to_file(_, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, step(_)).WillByDefault(Return(mock_sqlite3::result_done));
        ON_CALL(m_recorder, reset(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, clear_bindings(_))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, finalize(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, close(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, bind_null(_, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, bind_int(_, _, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, bind_int64(_, _, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, bind_double(_, _, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, bind_text(_, _, _))
            .WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, column_type(_, _))
            .WillByDefault(Return(mock_sqlite3::column_null));
        ON_CALL(m_recorder, column_int(_, _)).WillByDefault(Return(0));
        ON_CALL(m_recorder, column_int64(_, _)).WillByDefault(Return(0));
        ON_CALL(m_recorder, column_double(_, _)).WillByDefault(Return(0.0));
        ON_CALL(m_recorder, column_text(_, _)).WillByDefault(Return(std::string{}));
        ON_CALL(m_recorder, errmsg(_))
            .WillByDefault(Return(std::string{ "mock errmsg" }));
        ON_CALL(m_recorder, errstr(_))
            .WillByDefault(Return(std::string{ "mock errstr" }));
    }

    NiceMock<sqlite3_recorder> m_recorder;
    mock_sqlite3::scoped_bind  m_bind{ m_recorder };
    mock_connection            m_connection;
    mock_statement             m_statement;

    std::string m_database_path;
    std::string m_uuid;
};

// =============================================================================
// Construction / lifecycle
// =============================================================================

TEST_F(sqlite_backend_test, construct_instance)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    ASSERT_NE(db, nullptr);
}

TEST_F(sqlite_backend_test, get_uuid_returns_correct_value)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    EXPECT_EQ(db->get_uuid(), m_uuid);
}

TEST_F(sqlite_backend_test, create_opens_in_memory_and_prepares_control_statements)
{
    EXPECT_CALL(m_recorder, open(StrEq(":memory:"), _))
        .WillOnce(
            DoAll(SetArgPointee<1>(&m_connection), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_connection, StrEq("BEGIN TRANSACTION"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_statement), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_connection, StrEq("COMMIT"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_statement), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_connection, StrEq("ROLLBACK"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_statement), Return(mock_sqlite3::result_ok)));

    auto db = mock_backend::create(m_database_path, m_uuid);
    ASSERT_NE(db, nullptr);
}

TEST_F(sqlite_backend_test, destructor_finalizes_control_statements_and_closes)
{
    EXPECT_CALL(m_recorder, finalize(_))
        .Times(3)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, close(&m_connection))
        .WillOnce(Return(mock_sqlite3::result_ok));

    {
        auto db = mock_backend::create(m_database_path, m_uuid);
    }
}

// =============================================================================
// execute()
// =============================================================================

TEST_F(sqlite_backend_test, execute_forwards_query_to_exec)
{
    auto db = mock_backend::create(m_database_path, m_uuid);

    EXPECT_CALL(m_recorder,
                exec(&m_connection,
                     StrEq("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)")))
        .WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(
        db->execute("CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)"));
}

TEST_F(sqlite_backend_test, invalid_query_throws)
{
    auto db = mock_backend::create(m_database_path, m_uuid);

    EXPECT_CALL(m_recorder, exec(&m_connection, StrEq("INVALID SQL SYNTAX")))
        .WillOnce(Return(1 /* generic non-ok error */));

    EXPECT_THROW(db->execute("INVALID SQL SYNTAX"), std::runtime_error);
}

// =============================================================================
// Write statement executor: bind / step / reset
// =============================================================================

TEST_F(sqlite_backend_test, create_statement_executor_with_int32)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    auto executor =
        db->create_write_statement_executor<int32_t>("INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_int(&m_statement, 1, 42))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, reset(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(executor(42));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_int64)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    auto executor =
        db->create_write_statement_executor<int64_t>("INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_int64(&m_statement, 1, int64_t{ 9223372036854775807LL }))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_NO_THROW(executor(int64_t{ 9223372036854775807LL }));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_double)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    auto executor =
        db->create_write_statement_executor<double>("INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_double(&m_statement, 1, 3.14))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_NO_THROW(executor(3.14));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_text)
{
    auto db       = mock_backend::create(m_database_path, m_uuid);
    auto executor = db->create_write_statement_executor<const char*>(
        "INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_text(&m_statement, 1, std::string_view{ "hello world" }))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_NO_THROW(executor("hello world"));
}

TEST_F(sqlite_backend_test, create_statement_executor_with_multiple_params)
{
    auto db       = mock_backend::create(m_database_path, m_uuid);
    auto executor = db->create_write_statement_executor<int32_t, double, const char*>(
        "INSERT INTO t (a, b, c) VALUES (?, ?, ?)");

    EXPECT_CALL(m_recorder, bind_int(&m_statement, 1, 42))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, bind_double(&m_statement, 2, 3.14))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, bind_text(&m_statement, 3, std::string_view{ "test" }))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_NO_THROW(executor(42, 3.14, "test"));
}

TEST_F(sqlite_backend_test, statement_executor_can_be_reused)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    auto executor =
        db->create_write_statement_executor<int32_t>("INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_int(&m_statement, 1, _))
        .Times(100)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .Times(100)
        .WillRepeatedly(Return(mock_sqlite3::result_done));

    for(int i = 0; i < 100; ++i)
    {
        EXPECT_NO_THROW(executor(i));
    }
}

TEST_F(sqlite_backend_test, write_executor_failure_throws)
{
    auto db = mock_backend::create(m_database_path, m_uuid);
    auto executor =
        db->create_write_statement_executor<int32_t>("INSERT INTO t (val) VALUES (?)");

    EXPECT_CALL(m_recorder, bind_int(&m_statement, 1, 7))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement)).WillOnce(Return(1 /* non-ok */));

    EXPECT_THROW(executor(7), std::runtime_error);
}

// =============================================================================
// Read statement executor: reset / clear_bindings / step / column extraction
// =============================================================================

namespace
{
struct row_count
{
    int value{ 0 };
};
}  // namespace

TEST_F(sqlite_backend_test, read_executor_extracts_rows)
{
    auto db     = mock_backend::create(m_database_path, m_uuid);
    auto reader = db->create_read_statement_executor<row_count, bind_types<>>(
        "SELECT COUNT(*) FROM t", &row_count::value);

    // Two rows then done; each row reads column 0 as int.
    EXPECT_CALL(m_recorder, reset(&m_statement))
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, clear_bindings(&m_statement))
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_statement))
        .WillOnce(Return(mock_sqlite3::result_row))
        .WillOnce(Return(mock_sqlite3::result_row))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, column_int(&m_statement, 0))
        .WillOnce(Return(11))
        .WillOnce(Return(22));

    auto rows = reader().to_vector();
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].value, 11);
    EXPECT_EQ(rows[1].value, 22);
}

// =============================================================================
// flush() -> backup_to_file
// =============================================================================

TEST_F(sqlite_backend_test, flush_calls_backup_to_file)
{
    auto db = mock_backend::create(m_database_path, m_uuid);

    EXPECT_CALL(m_recorder, backup_to_file(&m_connection, StrEq(m_database_path.c_str())))
        .WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(db->flush());
}

TEST_F(sqlite_backend_test, double_flush_throws)
{
    auto db = mock_backend::create(m_database_path, m_uuid);

    EXPECT_CALL(m_recorder, backup_to_file(&m_connection, _))
        .WillOnce(Return(mock_sqlite3::result_ok));

    db->flush();
    EXPECT_THROW(db->flush(), std::runtime_error);
}

TEST_F(sqlite_backend_test, flush_failure_throws)
{
    auto db = mock_backend::create(m_database_path, m_uuid);

    EXPECT_CALL(m_recorder, backup_to_file(&m_connection, _))
        .WillOnce(Return(1 /* non-ok */));

    EXPECT_THROW(db->flush(), std::runtime_error);
}

// =============================================================================
// transaction_guard: BEGIN on entry, COMMIT on normal exit, ROLLBACK on throw
//
// Each control statement is a distinct mock_statement so the test can assert
// which one was stepped.
// =============================================================================
class transaction_guard_test : public sqlite_backend_test
{
protected:
    std::shared_ptr<mock_backend> make_backend_with_distinct_control_statements()
    {
        EXPECT_CALL(m_recorder, prepare(_, StrEq("BEGIN TRANSACTION"), _))
            .WillOnce(DoAll(SetArgPointee<2>(&m_begin), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, prepare(_, StrEq("COMMIT"), _))
            .WillOnce(
                DoAll(SetArgPointee<2>(&m_commit), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, prepare(_, StrEq("ROLLBACK"), _))
            .WillOnce(
                DoAll(SetArgPointee<2>(&m_rollback), Return(mock_sqlite3::result_ok)));
        return mock_backend::create(m_database_path, m_uuid);
    }

    mock_statement m_begin;
    mock_statement m_commit;
    mock_statement m_rollback;
};

TEST_F(transaction_guard_test, commits_on_normal_exit)
{
    auto db = make_backend_with_distinct_control_statements();

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback)).Times(0);

    {
        auto guard = db->begin_transaction();
    }
}

TEST_F(transaction_guard_test, rolls_back_on_exception)
{
    auto db = make_backend_with_distinct_control_statements();

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).Times(0);

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            throw std::runtime_error("simulated mid-transaction failure");
        },
        std::runtime_error);
}

// =============================================================================
// Lifetime: a prepared statement keeps the backend alive after the owning
// shared_ptr is dropped (deleter holds shared_from_this()).
// =============================================================================

TEST_F(sqlite_backend_test, statement_outlives_backend)
{
    std::function<void(int32_t)> executor;
    {
        auto db  = mock_backend::create(m_database_path, m_uuid);
        executor = db->create_write_statement_executor<int32_t>(
            "INSERT INTO t (val) VALUES (?)");
        // db shared_ptr drops here, but executor keeps the connection alive.
    }

    EXPECT_CALL(m_recorder, bind_int(_, 1, 42)).WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(_)).WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_NO_THROW(executor(42));
}

TEST_F(sqlite_backend_test, multiple_backends_independent)
{
    auto db1 = mock_backend::create(m_database_path + "_1", "uuid1");
    auto db2 = mock_backend::create(m_database_path + "_2", "uuid2");

    EXPECT_EQ(db1->get_uuid(), "uuid1");
    EXPECT_EQ(db2->get_uuid(), "uuid2");
}

}  // namespace
