// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mocks/mock_sqlite3.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace
{
using namespace profiler_hub::data_storage::mocks;
using ::testing::_;
using ::testing::DoAll;
using ::testing::Return;
using ::testing::SetArgPointee;
using ::testing::StrEq;

class mock_sqlite3_test : public ::testing::Test
{
protected:
    sqlite3_recorder          m_recorder;
    mock_sqlite3::scoped_bind m_bind{ m_recorder };
};

TEST_F(mock_sqlite3_test, prepare_forwards_to_recorder_and_returns_out_handle)
{
    mock_statement stmt;
    EXPECT_CALL(m_recorder, prepare(_, StrEq("SELECT 1"), _))
        .WillOnce(DoAll(SetArgPointee<2>(&stmt), Return(mock_sqlite3::result_ok)));

    mock_sqlite3::statement_t out = nullptr;
    const int                 rc  = mock_sqlite3::prepare(nullptr, "SELECT 1", &out);

    EXPECT_EQ(rc, mock_sqlite3::result_ok);
    EXPECT_EQ(out, &stmt);
}

TEST_F(mock_sqlite3_test, bind_and_step_forward_to_recorder)
{
    mock_statement stmt;
    EXPECT_CALL(m_recorder, bind_int(&stmt, 1, 42))
        .WillOnce(Return(mock_sqlite3::result_ok));
    EXPECT_CALL(m_recorder, step(&stmt)).WillOnce(Return(mock_sqlite3::result_done));

    EXPECT_EQ(mock_sqlite3::bind_int(&stmt, 1, 42), mock_sqlite3::result_ok);
    EXPECT_EQ(mock_sqlite3::step(&stmt), mock_sqlite3::result_done);
}

TEST_F(mock_sqlite3_test, column_text_returns_recorded_string)
{
    mock_statement stmt;
    EXPECT_CALL(m_recorder, column_text(&stmt, 0))
        .WillOnce(Return(std::string{ "uuid-x" }));

    EXPECT_EQ(mock_sqlite3::column_text(&stmt, 0), "uuid-x");
}
}  // namespace
