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
#include <functional>
#include <memory>
#include <string>

namespace
{

using namespace rocpdsna::data_storage;

// Schema versions to test
static const std::vector<rocpdsna::version_t> kSchemaVersions = {
    { 3, 0, 0 },  // Schema v3
    { 4, 0, 0 },  // Schema v4
};

// =============================================================================
// Parameterized tests (run for each schema version: v3 and v4)
// =============================================================================

class sqlite_backend_versioned_test
: public ::testing::Test
, public ::testing::WithParamInterface<rocpdsna::version_t>
{
protected:
    void SetUp() override
    {
        m_version = GetParam();
        m_database_path =
            "test_database_v" + std::to_string(m_version.major) + "_" +
            std::to_string(
                ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
            ".db";
        m_uuid = "test_uuid_12345";
    }

    void TearDown() override { std::remove(m_database_path.c_str()); }

    std::string         m_database_path;
    std::string         m_uuid;
    rocpdsna::version_t m_version;
};

// -----------------------------------------------------------------------------
// Parameterized schema tests (run for v3 and v4)
// -----------------------------------------------------------------------------

TEST_P(sqlite_backend_versioned_test, initialize_schema_succeeds)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    EXPECT_NO_THROW(db->initialize_schema(m_version));
}

TEST_P(sqlite_backend_versioned_test, double_initialize_schema_throws)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    EXPECT_THROW(db->initialize_schema(m_version), std::runtime_error);
}

TEST_P(sqlite_backend_versioned_test, initialized_schema_can_flush)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    EXPECT_NO_THROW(db->flush());

    FILE* file = std::fopen(m_database_path.c_str(), "r");
    ASSERT_NE(file, nullptr);
    std::fclose(file);
}

TEST_P(sqlite_backend_versioned_test, execute_custom_table_after_schema_init)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    // Verify we can still create custom tables after schema initialization
    EXPECT_NO_THROW(
        db->execute("CREATE TABLE custom_tbl (id INTEGER PRIMARY KEY, name TEXT)"));
    EXPECT_NO_THROW(db->execute("INSERT INTO custom_tbl (id, name) VALUES (1, 'test')"));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_after_schema_init_int32)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE int32_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int32_t>(
        "INSERT INTO int32_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(42));
    EXPECT_NO_THROW(executor(-100));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_after_schema_init_int64)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE int64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int64_t>(
        "INSERT INTO int64_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(int64_t{ 9223372036854775807LL }));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_after_schema_init_uint64)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE uint64_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<uint64_t>(
        "INSERT INTO uint64_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(uint64_t{ 12345678901234567890ULL }));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_after_schema_init_double)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE double_tbl (id INTEGER PRIMARY KEY, val REAL)");

    auto executor = db->create_write_statement_executor<double>(
        "INSERT INTO double_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor(3.14159265359));
    EXPECT_NO_THROW(executor(-2.71828));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_after_schema_init_text)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE text_tbl (id INTEGER PRIMARY KEY, val TEXT)");

    auto executor = db->create_write_statement_executor<const char*>(
        "INSERT INTO text_tbl (val) VALUES (?)");
    EXPECT_NO_THROW(executor("hello world"));
    EXPECT_NO_THROW(executor(""));
}

TEST_P(sqlite_backend_versioned_test,
       statement_executor_after_schema_init_multiple_params)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE multi_tbl (id INTEGER PRIMARY KEY, int_val INTEGER, "
                "real_val REAL, text_val TEXT)");

    auto executor = db->create_write_statement_executor<int32_t, double, const char*>(
        "INSERT INTO multi_tbl (int_val, real_val, text_val) VALUES (?, ?, ?)");
    EXPECT_NO_THROW(executor(42, 3.14, "test"));
    EXPECT_NO_THROW(executor(-1, 0.0, "another"));
}

TEST_P(sqlite_backend_versioned_test, statement_executor_reuse_after_schema_init)
{
    auto db = sqlite_backend::create(m_database_path, m_uuid);
    db->initialize_schema(m_version);
    db->execute("CREATE TABLE reuse_tbl (id INTEGER PRIMARY KEY, val INTEGER)");

    auto executor = db->create_write_statement_executor<int32_t>(
        "INSERT INTO reuse_tbl (val) VALUES (?)");

    for(int i = 0; i < 100; ++i)
    {
        EXPECT_NO_THROW(executor(i));
    }
}

TEST_P(sqlite_backend_versioned_test, statement_outlives_backend_after_schema_init)
{
    std::function<void(int32_t)> executor;
    {
        auto db = sqlite_backend::create(m_database_path, m_uuid);
        db->initialize_schema(m_version);
        db->execute("CREATE TABLE outlive_tbl (id INTEGER PRIMARY KEY, val INTEGER)");
        executor = db->create_write_statement_executor<int32_t>(
            "INSERT INTO outlive_tbl (val) VALUES (?)");
        // db goes out of scope here, but executor keeps it alive via shared_ptr
    }
    // This should still work -- the connection stays alive through the lambda capture
    EXPECT_NO_THROW(executor(42));
}

// Instantiate parameterized tests for both schema versions
INSTANTIATE_TEST_SUITE_P(SchemaVersions,
                         sqlite_backend_versioned_test,
                         ::testing::ValuesIn(kSchemaVersions),
                         [](const ::testing::TestParamInfo<rocpdsna::version_t>& info) {
                             return "v" + std::to_string(info.param.major) + "_" +
                                    std::to_string(info.param.minor) + "_" +
                                    std::to_string(info.param.patch);
                         });

// =============================================================================
// Non-parameterized tests (version-independent, no schema initialization)
// =============================================================================

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

// =============================================================================
// Foreign Key (PRAGMA foreign_keys) Tests
//
// Step 1: Create schema with FK constraints
// Step 2: FK ON  - insert with valid reference succeeds
// Step 3: FK ON  - insert with invalid reference fails
// Step 4: FK OFF - insert with invalid reference succeeds
// Step 5: FK OFF - bulk inserts with invalid references all succeed
// Step 6: Verify rows inserted while FK OFF are persisted
// Step 7: FK ON again - insert with invalid reference fails again
// =============================================================================

class sqlite_backend_foreign_key_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_database_path =
            "test_fk_" +
            std::to_string(
                ::testing::UnitTest::GetInstance()->current_test_info()->line()) +
            ".db";
        m_db = sqlite_backend::create(m_database_path, "fk_test_uuid");

        m_db->execute("CREATE TABLE parent (id INTEGER PRIMARY KEY);"
                      "CREATE TABLE child ("
                      "  id        INTEGER PRIMARY KEY,"
                      "  parent_id INTEGER REFERENCES parent(id)"
                      ")");
        m_db->execute("INSERT INTO parent VALUES (1)");
    }

    void TearDown() override
    {
        m_db.reset();
        std::remove(m_database_path.c_str());
    }

    int count_rows(const std::string& table)
    {
        int           count = 0;
        sqlite3*      raw   = nullptr;
        sqlite3_stmt* stmt  = nullptr;

        m_db->flush();

        sqlite3_open(m_database_path.c_str(), &raw);
        sqlite3_prepare_v2(
            raw, ("SELECT COUNT(*) FROM " + table).c_str(), -1, &stmt, nullptr);
        if(sqlite3_step(stmt) == SQLITE_ROW)
        {
            count = sqlite3_column_int(stmt, 0);
        }
        sqlite3_finalize(stmt);
        sqlite3_close(raw);
        return count;
    }

    std::string                     m_database_path;
    std::shared_ptr<sqlite_backend> m_db;
};

// Step 1: Schema with FK constraints is created successfully
TEST_F(sqlite_backend_foreign_key_test, step1_schema_with_fk_constraints_created)
{
    EXPECT_NO_THROW(m_db->execute("INSERT INTO parent VALUES (2)"));
    EXPECT_NO_THROW(m_db->execute("INSERT INTO child VALUES (1, 2)"));
}

// Step 2: FK ON - insert with valid reference succeeds
TEST_F(sqlite_backend_foreign_key_test, step2_fk_on_valid_reference_succeeds)
{
    m_db->set_foreign_keys_enabled(true);
    EXPECT_NO_THROW(m_db->execute("INSERT INTO child VALUES (1, 1)"));
}

// Step 3: FK ON - insert with invalid reference throws
TEST_F(sqlite_backend_foreign_key_test, step3_fk_on_invalid_reference_throws)
{
    m_db->set_foreign_keys_enabled(true);
    EXPECT_THROW(m_db->execute("INSERT INTO child VALUES (1, 999)"), std::runtime_error);
}

// Step 4: FK OFF - insert with invalid reference succeeds
TEST_F(sqlite_backend_foreign_key_test, step4_fk_off_invalid_reference_succeeds)
{
    m_db->set_foreign_keys_enabled(false);
    EXPECT_NO_THROW(m_db->execute("INSERT INTO child VALUES (1, 999)"));
}

// Step 5: FK OFF - bulk inserts with invalid references all succeed
TEST_F(sqlite_backend_foreign_key_test, step5_fk_off_bulk_invalid_references_succeed)
{
    m_db->set_foreign_keys_enabled(false);

    for(int i = 1; i <= 5; ++i)
    {
        EXPECT_NO_THROW(
            m_db->execute("INSERT INTO child VALUES (" + std::to_string(i) + ", 888)"));
    }
}

// Step 6: Rows inserted while FK OFF are actually persisted
TEST_F(sqlite_backend_foreign_key_test, step6_fk_off_rows_are_persisted)
{
    m_db->set_foreign_keys_enabled(false);

    m_db->execute("INSERT INTO child VALUES (10, 777)");
    m_db->execute("INSERT INTO child VALUES (11, 888)");
    m_db->execute("INSERT INTO child VALUES (12, 999)");

    EXPECT_EQ(count_rows("child"), 3);
}

// Step 7: FK re-enabled - invalid reference throws again
TEST_F(sqlite_backend_foreign_key_test, step7_fk_reenabled_invalid_reference_throws)
{
    m_db->set_foreign_keys_enabled(false);
    EXPECT_NO_THROW(m_db->execute("INSERT INTO child VALUES (1, 999)"));

    m_db->set_foreign_keys_enabled(true);
    EXPECT_THROW(m_db->execute("INSERT INTO child VALUES (2, 999)"), std::runtime_error);
}

}  // namespace
