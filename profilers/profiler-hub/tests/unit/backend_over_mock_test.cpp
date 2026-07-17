// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "data_storage/backends/sqlite_backend_impl.hpp"
#include "mocks/mock_sqlite3.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>

namespace
{
using namespace profiler_hub::data_storage;
using namespace profiler_hub::data_storage::mocks;

using ::testing::_;
using ::testing::DoAll;
using ::testing::HasSubstr;
using ::testing::InSequence;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrEq;

using mock_backend = database_backend<mock_sqlite3>;

class backend_over_mock_test : public ::testing::Test
{
protected:
    // The path must not exist on disk: the constructor flips to on_disk mode
    // (and runs discover_uuids) when the file is already present.
    void SetUp() override { std::filesystem::remove(m_db_path); }
    void TearDown() override { std::filesystem::remove(m_db_path); }

    // Minimal recorder wiring the ctor/dtor need: open the in-memory
    // connection, prepare the three transaction statements, and accept the
    // finalize/close issued during destruction.
    void expect_lifecycle()
    {
        EXPECT_CALL(m_recorder, open(StrEq(":memory:"), _))
            .WillOnce(DoAll(SetArgPointee<1>(&m_conn), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("BEGIN TRANSACTION"), _))
            .WillOnce(DoAll(SetArgPointee<2>(&m_begin), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("COMMIT"), _))
            .WillOnce(
                DoAll(SetArgPointee<2>(&m_commit), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("ROLLBACK"), _))
            .WillOnce(
                DoAll(SetArgPointee<2>(&m_rollback), Return(mock_sqlite3::result_ok)));
        EXPECT_CALL(m_recorder, finalize(_))
            .Times(3)
            .WillRepeatedly(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, close(&m_conn)).WillOnce(Return(mock_sqlite3::result_ok));
    }

    // Lenient construction wiring for tests that assert a specific operation
    // rather than the full lifecycle: open succeeds, every prepare/finalize/
    // reset/close returns ok. NiceMock suppresses uninteresting-call noise.
    void allow_lifecycle()
    {
        ON_CALL(m_recorder, open(_, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(&m_conn), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, prepare(_, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_begin), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, finalize(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, reset(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(m_recorder, close(_)).WillByDefault(Return(mock_sqlite3::result_ok));
    }

    // Route each transaction-control statement to its own handle so a test can
    // assert which one was stepped (allow_lifecycle alone maps them all to one).
    void route_txn_handles()
    {
        ON_CALL(m_recorder, prepare(_, StrEq("BEGIN TRANSACTION"), _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_begin), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, prepare(_, StrEq("COMMIT"), _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_commit), Return(mock_sqlite3::result_ok)));
        ON_CALL(m_recorder, prepare(_, StrEq("ROLLBACK"), _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_rollback), Return(mock_sqlite3::result_ok)));
    }

    // Same ON_CALL pattern as write_executor_binds... for other INSERT queries.
    void route_insert_prepare(const std::string& query)
    {
        ON_CALL(m_recorder, prepare(_, StrEq(query), _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&m_insert), Return(mock_sqlite3::result_ok)));
    }

    // initialize_schema() issues one exec per schema kind (five total today).
    void expect_schema_exec_calls(int times = 5)
    {
        EXPECT_CALL(m_recorder, exec(&m_conn, _))
            .Times(times)
            .WillRepeatedly(Return(mock_sqlite3::result_ok));
    }

    std::shared_ptr<mock_backend> create_in_memory_backend()
    {
        return mock_backend::create(
            m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);
    }

    std::string                m_db_path{ "backend_over_mock_test.db" };
    std::string                m_uuid{ "uuid-mock" };
    NiceMock<sqlite3_recorder> m_recorder;
    mock_sqlite3::scoped_bind  m_bind{ m_recorder };
    mock_connection            m_conn;
    mock_statement             m_begin;
    mock_statement             m_commit;
    mock_statement             m_rollback;
    mock_statement             m_insert;
};

TEST_F(backend_over_mock_test, create_drives_open_and_txn_prepares_without_real_db)
{
    expect_lifecycle();

    auto db =
        mock_backend::create(m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);

    ASSERT_NE(db, nullptr);
    EXPECT_EQ(db->get_uuid(), m_uuid);
}

TEST_F(backend_over_mock_test, write_executor_binds_each_value_then_steps_and_resets)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO strings(id, value) VALUES (?, ?)";

    // Route the INSERT prepare to a distinct handle so the bind/step/reset
    // expectations below target it specifically. Left as ON_CALL (not
    // EXPECT_CALL) so the ctor's BEGIN/COMMIT/ROLLBACK prepares stay covered by
    // allow_lifecycle()'s default rather than counting as unexpected calls.
    ON_CALL(m_recorder, prepare(_, StrEq(query), _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(&m_insert), Return(mock_sqlite3::result_ok)));

    auto db =
        mock_backend::create(m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);
    auto insert =
        db->create_write_statement_executor<std::int64_t, std::string_view>(query);

    {
        InSequence seq;
        EXPECT_CALL(m_recorder, bind_int64(&m_insert, 1, 42))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_text(&m_insert, 2, std::string_view{ "abc" }))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    insert(std::int64_t{ 42 }, std::string_view{ "abc" });
}

TEST_F(backend_over_mock_test, write_executor_throws_runtime_error_when_step_fails)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO strings(id, value) VALUES (?, ?)";
    ON_CALL(m_recorder, prepare(_, StrEq(query), _))
        .WillByDefault(
            DoAll(SetArgPointee<2>(&m_insert), Return(mock_sqlite3::result_ok)));

    // Any result that is not ok/done/row is an error the backend must surface.
    constexpr int error_code = 1;
    EXPECT_CALL(m_recorder, step(&m_insert)).WillOnce(Return(error_code));
    EXPECT_CALL(m_recorder, errmsg(&m_conn))
        .WillOnce(Return(std::string{ "disk I/O error" }));
    EXPECT_CALL(m_recorder, errstr(error_code))
        .WillOnce(Return(std::string{ "SQLITE_ERROR" }));

    auto db =
        mock_backend::create(m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);
    auto insert =
        db->create_write_statement_executor<std::int64_t, std::string_view>(query);

    EXPECT_THROW(
        {
            try
            {
                insert(std::int64_t{ 1 }, std::string_view{ "x" });
            } catch(const std::runtime_error& e)
            {
                const std::string what = e.what();
                EXPECT_THAT(what, HasSubstr("disk I/O error"));
                EXPECT_THAT(what, HasSubstr("SQLITE_ERROR"));
                throw;
            }
        },
        std::runtime_error);
}

TEST_F(backend_over_mock_test, transaction_guard_commits_on_normal_scope_exit)
{
    allow_lifecycle();
    route_txn_handles();

    auto db =
        mock_backend::create(m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback)).Times(0);

    {
        auto guard = db->begin_transaction();
    }
}

TEST_F(backend_over_mock_test,
       transaction_guard_rolls_back_when_scope_exits_via_exception)
{
    allow_lifecycle();
    route_txn_handles();

    auto db =
        mock_backend::create(m_db_path, m_uuid, mock_backend::storage_mode_t::in_memory);

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).Times(0);

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            throw std::runtime_error("boom");
        },
        std::runtime_error);
}

TEST_F(backend_over_mock_test, initialize_schema_succeeds)
{
    allow_lifecycle();
    expect_schema_exec_calls();

    auto db = create_in_memory_backend();
    EXPECT_NO_THROW(db->initialize_schema());
}

TEST_F(backend_over_mock_test, double_initialize_schema_throws)
{
    allow_lifecycle();
    expect_schema_exec_calls();

    auto db = create_in_memory_backend();
    db->initialize_schema();
    EXPECT_THROW(db->initialize_schema(), std::runtime_error);
}

TEST_F(backend_over_mock_test, execute_creates_table)
{
    allow_lifecycle();

    const char* create_sql = "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)";
    EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(create_sql)))
        .WillOnce(Return(mock_sqlite3::result_ok));

    auto db = create_in_memory_backend();
    EXPECT_NO_THROW(db->execute(create_sql));
}

TEST_F(backend_over_mock_test, execute_inserts_data)
{
    allow_lifecycle();

    const char* create_sql = "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY, name TEXT)";
    const char* insert_sql = "INSERT INTO test_tbl (id, name) VALUES (1, 'test')";
    {
        InSequence seq;
        EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(create_sql)))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(insert_sql)))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    auto db = create_in_memory_backend();
    db->execute(create_sql);
    EXPECT_NO_THROW(db->execute(insert_sql));
}

TEST_F(backend_over_mock_test, invalid_query_throws)
{
    allow_lifecycle();

    const char* invalid_sql = "INVALID SQL SYNTAX";
    EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(invalid_sql))).WillOnce(Return(1));
    EXPECT_CALL(m_recorder, errmsg(&m_conn))
        .WillOnce(Return(std::string{ "near \"INVALID\": syntax error" }));
    EXPECT_CALL(m_recorder, errstr(1)).WillOnce(Return(std::string{ "SQLITE_ERROR" }));

    auto db = create_in_memory_backend();
    EXPECT_THROW(db->execute(invalid_sql), std::runtime_error);
}

TEST_F(backend_over_mock_test, flush_calls_backup_to_file)
{
    allow_lifecycle();

    const char* create_sql = "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)";
    EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(create_sql)))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, backup_to_file(&m_conn, StrEq(m_db_path.c_str()), _))
        .WillOnce(Return(mock_sqlite3::result_ok));

    auto db = create_in_memory_backend();
    db->execute(create_sql);
    EXPECT_NO_THROW(db->flush());
}

TEST_F(backend_over_mock_test, double_flush_throws)
{
    allow_lifecycle();

    const char* create_sql = "CREATE TABLE test_tbl (id INTEGER PRIMARY KEY)";
    EXPECT_CALL(m_recorder, exec(&m_conn, StrEq(create_sql)))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, backup_to_file(&m_conn, StrEq(m_db_path.c_str()), _))
        .WillOnce(Return(mock_sqlite3::result_ok));

    auto db = create_in_memory_backend();
    db->execute(create_sql);
    db->flush();
    EXPECT_THROW(db->flush(), std::runtime_error);
}

TEST_F(backend_over_mock_test, create_statement_executor_with_int32)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO int32_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<std::int32_t>(query);

    {
        InSequence seq;
        EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, 42))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, -100))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    EXPECT_NO_THROW(executor(42));
    EXPECT_NO_THROW(executor(-100));
}

TEST_F(backend_over_mock_test, create_statement_executor_with_int64)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO int64_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<std::int64_t>(query);

    EXPECT_CALL(m_recorder, bind_int64(&m_insert, 1, 9223372036854775807LL))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, reset(&m_insert)).WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(executor(std::int64_t{ 9223372036854775807LL }));
}

TEST_F(backend_over_mock_test, create_statement_executor_with_uint64)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO uint64_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<std::uint64_t>(query);

    EXPECT_CALL(m_recorder, bind_int64(&m_insert, 1, 12345678901234567890ULL))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, reset(&m_insert)).WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(executor(std::uint64_t{ 12345678901234567890ULL }));
}

TEST_F(backend_over_mock_test, create_statement_executor_with_double)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO double_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<double>(query);

    {
        InSequence seq;
        EXPECT_CALL(m_recorder, bind_double(&m_insert, 1, 3.14159265359))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_double(&m_insert, 1, -2.71828))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    EXPECT_NO_THROW(executor(3.14159265359));
    EXPECT_NO_THROW(executor(-2.71828));
}

TEST_F(backend_over_mock_test, create_statement_executor_with_text)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO text_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<const char*>(query);

    {
        InSequence seq;
        EXPECT_CALL(m_recorder,
                    bind_text(&m_insert, 1, std::string_view{ "hello world" }))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_text(&m_insert, 1, std::string_view{ "" }))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    EXPECT_NO_THROW(executor("hello world"));
    EXPECT_NO_THROW(executor(""));
}

TEST_F(backend_over_mock_test, create_statement_executor_with_multiple_params)
{
    allow_lifecycle();

    const std::string query =
        "INSERT INTO multi_tbl (int_val, real_val, text_val) VALUES (?, ?, ?)";
    route_insert_prepare(query);

    auto db = create_in_memory_backend();
    auto executor =
        db->create_write_statement_executor<std::int32_t, double, const char*>(query);

    {
        InSequence seq;
        EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, 42))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_double(&m_insert, 2, 3.14))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_text(&m_insert, 3, std::string_view{ "test" }))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, -1))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_double(&m_insert, 2, 0.0))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, bind_text(&m_insert, 3, std::string_view{ "another" }))
            .WillOnce(Return(mock_sqlite3::result_ok));
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_done));
        EXPECT_CALL(m_recorder, reset(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_ok));
    }

    EXPECT_NO_THROW(executor(42, 3.14, "test"));
    EXPECT_NO_THROW(executor(-1, 0.0, "another"));
}

TEST_F(backend_over_mock_test, statement_executor_can_be_reused)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO reuse_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db       = create_in_memory_backend();
    auto executor = db->create_write_statement_executor<std::int32_t>(query);

    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, _))
        .Times(100)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert))
        .Times(100)
        .WillRepeatedly(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, reset(&m_insert))
        .Times(100)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));

    for(int i = 0; i < 100; ++i)
    {
        EXPECT_NO_THROW(executor(i));
    }
}

TEST_F(backend_over_mock_test, initialized_schema_can_flush)
{
    allow_lifecycle();
    expect_schema_exec_calls();
    EXPECT_CALL(m_recorder, backup_to_file(&m_conn, StrEq(m_db_path.c_str()), _))
        .WillOnce(Return(mock_sqlite3::result_ok));

    auto db = create_in_memory_backend();
    db->initialize_schema();
    EXPECT_NO_THROW(db->flush());
}

TEST_F(backend_over_mock_test, multiple_backends_independent)
{
    const std::string path1 = m_db_path + "_1";
    const std::string path2 = m_db_path + "_2";
    std::filesystem::remove(path1);
    std::filesystem::remove(path2);

    NiceMock<sqlite3_recorder> recorder1;
    mock_connection            conn1;
    mock_statement             begin1;
    mock_statement             commit1;
    mock_statement             rollback1;

    {
        mock_sqlite3::scoped_bind bind1{ recorder1 };
        ON_CALL(recorder1, open(_, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(&conn1), Return(mock_sqlite3::result_ok)));
        ON_CALL(recorder1, prepare(_, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&begin1), Return(mock_sqlite3::result_ok)));
        ON_CALL(recorder1, finalize(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(recorder1, close(_)).WillByDefault(Return(mock_sqlite3::result_ok));

        auto db1 =
            mock_backend::create(path1, "uuid1", mock_backend::storage_mode_t::in_memory);
        EXPECT_EQ(db1->get_uuid(), "uuid1");
        EXPECT_CALL(recorder1,
                    exec(&conn1, StrEq("CREATE TABLE tbl1 (id INTEGER PRIMARY KEY)")))
            .WillOnce(Return(mock_sqlite3::result_ok));
        db1->execute("CREATE TABLE tbl1 (id INTEGER PRIMARY KEY)");
    }

    NiceMock<sqlite3_recorder> recorder2;
    mock_connection            conn2;
    mock_statement             begin2;
    mock_statement             commit2;
    mock_statement             rollback2;

    {
        mock_sqlite3::scoped_bind bind2{ recorder2 };
        ON_CALL(recorder2, open(_, _))
            .WillByDefault(
                DoAll(SetArgPointee<1>(&conn2), Return(mock_sqlite3::result_ok)));
        ON_CALL(recorder2, prepare(_, _, _))
            .WillByDefault(
                DoAll(SetArgPointee<2>(&begin2), Return(mock_sqlite3::result_ok)));
        ON_CALL(recorder2, finalize(_)).WillByDefault(Return(mock_sqlite3::result_ok));
        ON_CALL(recorder2, close(_)).WillByDefault(Return(mock_sqlite3::result_ok));

        auto db2 =
            mock_backend::create(path2, "uuid2", mock_backend::storage_mode_t::in_memory);
        EXPECT_EQ(db2->get_uuid(), "uuid2");
        EXPECT_CALL(recorder2,
                    exec(&conn2, StrEq("CREATE TABLE tbl2 (id INTEGER PRIMARY KEY)")))
            .WillOnce(Return(mock_sqlite3::result_ok));
        db2->execute("CREATE TABLE tbl2 (id INTEGER PRIMARY KEY)");
    }

    std::filesystem::remove(path1);
    std::filesystem::remove(path2);
}

TEST_F(backend_over_mock_test, create_on_existing_file_discovers_uuid)
{
    std::ofstream{ m_db_path };
    const std::string expected_uuid = "3224963d0bd2e790224c3b2186eb8bd0";

    EXPECT_CALL(m_recorder, open(StrEq(m_db_path.c_str()), _))
        .WillOnce(DoAll(SetArgPointee<1>(&m_conn), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("BEGIN TRANSACTION"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_begin), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("COMMIT"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_commit), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_conn, StrEq("ROLLBACK"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_rollback), Return(mock_sqlite3::result_ok)));
    EXPECT_CALL(m_recorder, prepare(&m_conn, HasSubstr("sqlite_master"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&m_insert), Return(mock_sqlite3::result_ok)));
    {
        InSequence seq;
        EXPECT_CALL(m_recorder, step(&m_insert))
            .WillOnce(Return(mock_sqlite3::result_row))
            .WillOnce(Return(mock_sqlite3::result_done));
    }
    EXPECT_CALL(m_recorder, column_text(&m_insert, 0)).WillOnce(Return(expected_uuid));
    EXPECT_CALL(m_recorder, finalize(_))
        .Times(4)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, close(&m_conn)).WillOnce(Return(mock_sqlite3::result_ok));

    auto db = mock_backend::create(m_db_path, "");
    EXPECT_EQ(db->get_uuid(), expected_uuid);
}

TEST_F(backend_over_mock_test, statement_outlives_backend)
{
    allow_lifecycle();

    const std::string query = "INSERT INTO outlive_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    mock_backend::prepared_insert_statement<std::int32_t> executor;
    {
        auto db  = create_in_memory_backend();
        executor = db->create_write_statement_executor<std::int32_t>(query);
    }

    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, 42))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, reset(&m_insert)).WillOnce(Return(mock_sqlite3::result_ok));

    EXPECT_NO_THROW(executor(42));
}

TEST_F(backend_over_mock_test, transaction_guard_commits_after_multiple_inserts)
{
    allow_lifecycle();
    route_txn_handles();

    const std::string query = "INSERT INTO tx_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db     = create_in_memory_backend();
    auto insert = db->create_write_statement_executor<std::int32_t>(query);

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, _))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback)).Times(0);

    {
        auto guard = db->begin_transaction();
        for(int i = 0; i < 5; ++i)
        {
            insert(i);
        }
        (void) guard;
    }
}

TEST_F(backend_over_mock_test, transaction_guard_rolls_back_after_failed_scope)
{
    allow_lifecycle();
    route_txn_handles();

    const std::string query = "INSERT INTO tx_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db     = create_in_memory_backend();
    auto insert = db->create_write_statement_executor<std::int32_t>(query);

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, _))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).Times(0);

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            for(int i = 0; i < 5; ++i)
            {
                insert(i);
            }
            (void) guard;
            throw std::runtime_error("simulated mid-transaction failure");
        },
        std::runtime_error);
}

TEST_F(backend_over_mock_test, transaction_guard_independent_scopes)
{
    allow_lifecycle();
    route_txn_handles();

    const std::string query = "INSERT INTO tx_tbl (val) VALUES (?)";
    route_insert_prepare(query);

    auto db     = create_in_memory_backend();
    auto insert = db->create_write_statement_executor<std::int32_t>(query);

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, _))
        .Times(3)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert))
        .Times(3)
        .WillRepeatedly(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).WillOnce(Return(mock_sqlite3::result_done));

    {
        auto guard = db->begin_transaction();
        for(int i = 0; i < 3; ++i)
        {
            insert(i);
        }
        (void) guard;
    }

    EXPECT_CALL(m_recorder, step(&m_begin)).WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, bind_int(&m_insert, 1, _))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&m_insert))
        .Times(5)
        .WillRepeatedly(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_rollback))
        .WillOnce(Return(mock_sqlite3::result_done));
    EXPECT_CALL(m_recorder, step(&m_commit)).Times(0);

    EXPECT_THROW(
        {
            auto guard = db->begin_transaction();
            for(int i = 100; i < 105; ++i)
            {
                insert(i);
            }
            (void) guard;
            throw std::runtime_error("rollback this scope only");
        },
        std::runtime_error);
}
}  // namespace
