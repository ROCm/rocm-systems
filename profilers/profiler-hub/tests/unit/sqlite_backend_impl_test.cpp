// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/embedded_schema.hpp"
#include "backends/sqlite_backend_impl.hpp"
#include "profiler-hub/storage.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace
{

TEST(sqlite_backend_impl_test, get_schema_query_returns_non_empty_sql_for_rocpd_tables)
{
    const std::string query =
        get_schema_query(::rocpd::data_storage::schema::v3_0_0::ROCPD_TABLES_SQL,
                         "test-uuid",
                         profiler_hub::version_t{ 3, 0, 0 });
    EXPECT_FALSE(query.empty());
}

TEST(sqlite_backend_impl_test, get_schema_query_for_metadata_embeds_requested_version)
{
    const std::string query =
        get_schema_query(::rocpd::data_storage::schema::v3_0_1::ROCPD_METADATA_SQL,
                         "test-uuid",
                         profiler_hub::version_t{ 3, 0, 1 });

    EXPECT_NE(query.find(R"(("schema_version", "3.0.1"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_major", "3"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_minor", "0"))"), std::string::npos);
    EXPECT_NE(query.find(R"(("schema_version_patch", "1"))"), std::string::npos);
}

TEST(sqlite_backend_impl_test, get_schema_query_substitutes_metadata_uuid_and_guid)
{
    const std::string query =
        get_schema_query(::rocpd::data_storage::schema::v3_0_1::ROCPD_METADATA_SQL,
                         "test-uuid",
                         profiler_hub::version_t{ 3, 0, 1 });

    EXPECT_NE(query.find("rocpd_metadata_test-uuid"), std::string::npos);
    EXPECT_NE(query.find("(\"uuid\", \"_test-uuid\")"), std::string::npos);
    EXPECT_NE(query.find("(\"guid\", \"test-uuid\")"), std::string::npos);
    EXPECT_EQ(query.find("{{uuid}}"), std::string::npos);
    EXPECT_EQ(query.find("{{guid}}"), std::string::npos);
}

TEST(sqlite_backend_impl_test, get_schema_query_returns_empty_for_null_schema_content)
{
    const std::string query =
        get_schema_query(nullptr, "test-uuid", profiler_hub::version_t{ 3, 0, 0 });
    EXPECT_TRUE(query.empty());
}

// database_backend calls SqlitePolicy static methods. Google Mock needs an
// instance, so this policy forwards every call to the mock installed for the test.
class sqlite_api
{
public:
    struct connection
    {};
    struct statement
    {};

    using database_t  = connection*;
    using statement_t = statement*;

    virtual ~sqlite_api() = default;

    virtual int open(const char* path, database_t* out_db)                        = 0;
    virtual int close(database_t db)                                              = 0;
    virtual int exec(database_t db, const char* sql)                              = 0;
    virtual int prepare(database_t db, const char* sql, statement_t* out_stmt)    = 0;
    virtual int finalize(statement_t stmt)                                        = 0;
    virtual int reset(statement_t stmt)                                           = 0;
    virtual int clear_bindings(statement_t stmt)                                  = 0;
    virtual int step(statement_t stmt)                                            = 0;
    virtual int bind_null(statement_t stmt, int position)                         = 0;
    virtual int bind_int(statement_t stmt, int position, int value)               = 0;
    virtual int bind_int64(statement_t stmt, int position, std::int64_t value)    = 0;
    virtual int bind_double(statement_t stmt, int position, double value)         = 0;
    virtual int bind_text(statement_t stmt, int position, std::string_view value) = 0;
    virtual int column_type(statement_t stmt, int position)                       = 0;
    virtual int column_int(statement_t stmt, int position)                        = 0;
    virtual std::int64_t column_int64(statement_t stmt, int position)             = 0;
    virtual double       column_double(statement_t stmt, int position)            = 0;
    virtual std::string  column_text(statement_t stmt, int position)              = 0;
    virtual std::string  errmsg(database_t db)                                    = 0;
    virtual std::string  errstr(int result_code)                                  = 0;
};

class mock_sqlite_api : public sqlite_api
{
public:
    MOCK_METHOD(int, open, (const char*, database_t*), (override));
    MOCK_METHOD(int, close, (database_t), (override));
    MOCK_METHOD(int, exec, (database_t, const char*), (override));
    MOCK_METHOD(int, prepare, (database_t, const char*, statement_t*), (override));
    MOCK_METHOD(int, finalize, (statement_t), (override));
    MOCK_METHOD(int, reset, (statement_t), (override));
    MOCK_METHOD(int, clear_bindings, (statement_t), (override));
    MOCK_METHOD(int, step, (statement_t), (override));
    MOCK_METHOD(int, bind_null, (statement_t, int), (override));
    MOCK_METHOD(int, bind_int, (statement_t, int, int), (override));
    MOCK_METHOD(int, bind_int64, (statement_t, int, std::int64_t), (override));
    MOCK_METHOD(int, bind_double, (statement_t, int, double), (override));
    MOCK_METHOD(int, bind_text, (statement_t, int, std::string_view), (override));
    MOCK_METHOD(int, column_type, (statement_t, int), (override));
    MOCK_METHOD(int, column_int, (statement_t, int), (override));
    MOCK_METHOD(std::int64_t, column_int64, (statement_t, int), (override));
    MOCK_METHOD(double, column_double, (statement_t, int), (override));
    MOCK_METHOD(std::string, column_text, (statement_t, int), (override));
    MOCK_METHOD(std::string, errmsg, (database_t), (override));
    MOCK_METHOD(std::string, errstr, (int), (override));
};

struct gmock_sqlite_policy
{
    using database_t  = sqlite_api::database_t;
    using statement_t = sqlite_api::statement_t;

    static constexpr int result_ok   = 0;
    static constexpr int result_row  = 100;
    static constexpr int result_done = 101;
    static constexpr int column_null = 5;

    static sqlite_api* instance;

    static int open(const char* path, database_t* out_db)
    {
        return instance->open(path, out_db);
    }
    static int close(database_t db) { return instance->close(db); }
    static int exec(database_t db, const char* sql) { return instance->exec(db, sql); }
    static int prepare(database_t db, const char* sql, statement_t* out_stmt)
    {
        return instance->prepare(db, sql, out_stmt);
    }
    static int finalize(statement_t stmt) { return instance->finalize(stmt); }
    static int reset(statement_t stmt) { return instance->reset(stmt); }
    static int clear_bindings(statement_t stmt) { return instance->clear_bindings(stmt); }
    static int step(statement_t stmt) { return instance->step(stmt); }
    static int bind_null(statement_t stmt, int position)
    {
        return instance->bind_null(stmt, position);
    }
    static int bind_int(statement_t stmt, int position, int value)
    {
        return instance->bind_int(stmt, position, value);
    }
    static int bind_int64(statement_t stmt, int position, std::int64_t value)
    {
        return instance->bind_int64(stmt, position, value);
    }
    static int bind_double(statement_t stmt, int position, double value)
    {
        return instance->bind_double(stmt, position, value);
    }
    static int bind_text(statement_t stmt, int position, std::string_view value)
    {
        return instance->bind_text(stmt, position, value);
    }
    static int column_type(statement_t stmt, int position)
    {
        return instance->column_type(stmt, position);
    }
    static int column_int(statement_t stmt, int position)
    {
        return instance->column_int(stmt, position);
    }
    static std::int64_t column_int64(statement_t stmt, int position)
    {
        return instance->column_int64(stmt, position);
    }
    static double column_double(statement_t stmt, int position)
    {
        return instance->column_double(stmt, position);
    }
    static std::string column_text(statement_t stmt, int position)
    {
        return instance->column_text(stmt, position);
    }
    static std::string errmsg(database_t db) { return instance->errmsg(db); }
    static std::string errstr(int result_code) { return instance->errstr(result_code); }
};

sqlite_api* gmock_sqlite_policy::instance = nullptr;

using mock_backend_t = profiler_hub::data_storage::database_backend<gmock_sqlite_policy>;

struct sqlite_mock_guard
{
    explicit sqlite_mock_guard(sqlite_api& mock)
    {
        gmock_sqlite_policy::instance = &mock;
    }
    ~sqlite_mock_guard() { gmock_sqlite_policy::instance = nullptr; }

    sqlite_mock_guard(const sqlite_mock_guard&)            = delete;
    sqlite_mock_guard& operator=(const sqlite_mock_guard&) = delete;
};

const embedded_schema_t*
newest_embedded_schema()
{
    const embedded_schema_t* newest = nullptr;
    for(const auto& schema : k_embedded_schemas)
    {
        const profiler_hub::version_t embedded{ schema.major,
                                                schema.minor,
                                                schema.patch };
        if(newest == nullptr ||
           profiler_hub::version_t{ newest->major, newest->minor, newest->patch } <
               embedded)
        {
            newest = &schema;
        }
    }
    return newest;
}

std::vector<std::string>
expected_schema_sql(const embedded_schema_t& schema, const std::string& uuid)
{
    const profiler_hub::version_t resolved{ schema.major, schema.minor, schema.patch };
    const char* parts[] = { schema.tables,     schema.indexes,       schema.views,
                            schema.data_views, schema.summary_views, schema.metadata };
    std::vector<std::string> expected;
    for(const char* part : parts)
    {
        std::string query = get_schema_query(part, uuid, resolved);
        if(!query.empty())
        {
            expected.push_back(std::move(query));
        }
    }
    return expected;
}

::testing::Matcher<const char*>
schema_statement_matcher(std::string_view version, const std::string& sql, bool is_tables)
{
    using ::testing::AllOf;
    using ::testing::HasSubstr;
    using ::testing::Not;
    using ::testing::StrEq;

    if(!is_tables || (version != "3.0.0" && version != "3.0.1" && version != "3.0.2" &&
                      version != "3.0.3" && version != "3.0.4" && version != "3.0.5"))
    {
        return StrEq(sql);
    }

    const auto needle = [](bool present, const char* text) {
        return present ? ::testing::Matcher<const char*>{ HasSubstr(text) }
                       : ::testing::Matcher<const char*>{ Not(HasSubstr(text)) };
    };

    bool allows_nic       = false;
    bool has_graph_launch = false;
    bool has_hip_event    = false;
    bool has_shader_array = false;
    if(version == "3.0.1")
    {
        allows_nic = true;
    }
    else if(version == "3.0.2" || version == "3.0.3")
    {
        allows_nic       = true;
        has_graph_launch = true;
    }
    else if(version == "3.0.4")
    {
        allows_nic       = true;
        has_graph_launch = true;
        has_hip_event    = true;
    }
    else if(version == "3.0.5")
    {
        allows_nic       = true;
        has_graph_launch = true;
        has_hip_event    = true;
        has_shader_array = true;
    }

    return AllOf(StrEq(sql),
                 needle(allows_nic, "'NIC'"),
                 needle(has_graph_launch, "rocpd_graph_launch"),
                 needle(has_hip_event, "rocpd_hip_event"),
                 needle(has_shader_array, "shader_array"));
}

void
expect_schema_exec(mock_sqlite_api&         sqlite,
                   const embedded_schema_t& schema,
                   const std::string&       uuid)
{
    using ::testing::_;
    using ::testing::Return;
    using ::testing::StrEq;

    const auto                    expected = expected_schema_sql(schema, uuid);
    const profiler_hub::version_t resolved{ schema.major, schema.minor, schema.patch };
    const std::string             schema_version_row =
        "(\"schema_version\", \"" + resolved.to_string() + "\")";

    for(size_t index = 0; index < expected.size(); ++index)
    {
        const bool                      is_tables   = index == 0;
        const bool                      is_metadata = index + 1 == expected.size();
        ::testing::Matcher<const char*> matcher =
            is_tables ? schema_statement_matcher(schema.version, expected[index], true)
                      : ::testing::Matcher<const char*>{ StrEq(expected[index]) };
        if(is_metadata)
        {
            matcher = ::testing::AllOf(std::move(matcher),
                                       ::testing::HasSubstr(schema_version_row));
        }

        EXPECT_CALL(sqlite, exec(_, matcher))
            .WillOnce(Return(gmock_sqlite_policy::result_ok));
    }
}

TEST(sqlite_backend_impl_test,
     storage_schema_version_executes_the_matching_embedded_schema)
{
    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema.db").string();
    std::filesystem::remove(db_path);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };

    for(const auto& schema : k_embedded_schemas)
    {
        SCOPED_TRACE(schema.version);
        const profiler_hub::version_t version{ schema.major, schema.minor, schema.patch };
        const profiler_hub::storage_t storage{ db_path, uuid, version };
        EXPECT_TRUE(storage.get_storage_version() == version);

        auto backend = mock_backend_t::create(
            db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
        {
            ::testing::InSequence sequence;
            expect_schema_exec(sqlite, schema, uuid);
            backend->initialize_schema(storage.get_storage_version());
        }
        ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
    }
}

TEST(sqlite_backend_impl_test,
     storage_latest_sentinel_executes_the_newest_embedded_schema)
{
    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema_latest.db")
            .string();
    std::filesystem::remove(db_path);

    const embedded_schema_t* newest = newest_embedded_schema();
    ASSERT_NE(newest, nullptr);

    const profiler_hub::storage_t storage{ db_path, uuid, profiler_hub::version_t{} };
    EXPECT_TRUE(storage.get_storage_version().is_latest());

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
    {
        ::testing::InSequence sequence;
        expect_schema_exec(sqlite, *newest, uuid);
        backend->initialize_schema(storage.get_storage_version());
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

}  // namespace
