// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "data_storage/backends/sqlite_backend_impl.hpp"
#include "mocks/mock_sqlite3.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
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
}  // namespace
