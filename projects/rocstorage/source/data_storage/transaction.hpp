// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <sqlite3.h>
#include <stdexcept>

namespace rocstorage::data_storage
{

class transaction_block
{
public:
    explicit transaction_block(sqlite3* db)
    : m_db{ db }
    , m_uncaught_on_entry{ std::uncaught_exceptions() }
    {
        if(sqlite3_exec(m_db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr) !=
           SQLITE_OK)
        {
            throw std::runtime_error(sqlite3_errmsg(m_db));
        }
    }

    ~transaction_block()
    {
        if(std::uncaught_exceptions() > m_uncaught_on_entry)
        {
            sqlite3_exec(m_db, "ROLLBACK", nullptr, nullptr, nullptr);
        }
        else
        {
            sqlite3_exec(m_db, "COMMIT", nullptr, nullptr, nullptr);
        }
    }

    transaction_block(const transaction_block&)            = delete;
    transaction_block& operator=(const transaction_block&) = delete;
    transaction_block(transaction_block&&)                 = delete;
    transaction_block& operator=(transaction_block&&)      = delete;

private:
    sqlite3* m_db;
    int      m_uncaught_on_entry;
};

}  // namespace rocstorage::data_storage
