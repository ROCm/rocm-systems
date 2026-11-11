// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "backends/embedded_schema.hpp"
#include "backends/sqlite_backend_impl.hpp"
#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
// windows.h defines HANDLE, BOOL, and WINAPI. tlhelp32.h uses them, so it
// stays in its own include block: clang-format sorts one block alphabetically
// and would put tlhelp32.h first.
#    include <windows.h>

#    include <tlhelp32.h>
#endif

// gtest finds PrintTo through argument-dependent lookup in profiler_hub, so this
// makes parameterized test names show "3.0.0" instead of raw bytes.
namespace profiler_hub
{
inline void
PrintTo(const version_t& version, std::ostream* os)
{
    *os << version.to_string();
}
}  // namespace profiler_hub

namespace
{

using rocpd::data_storage::schema::k_embedded_schemas;

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
        if(resolve_writable_schema(version) == nullptr)
        {
            EXPECT_THROW((profiler_hub::storage_t{ db_path, uuid, version }),
                         std::runtime_error);
            auto backend = mock_backend_t::create(
                db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
            EXPECT_THROW(backend->initialize_schema(version), std::runtime_error);
            continue;
        }

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
     storage_latest_sentinel_executes_the_newest_writable_schema)
{
    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema_latest.db")
            .string();
    std::filesystem::remove(db_path);

    const embedded_schema_t* writable =
        resolve_writable_schema(profiler_hub::version_t{});
    ASSERT_NE(writable, nullptr);
    EXPECT_STREQ(writable->version, "3.0.1");

    const profiler_hub::storage_t storage{ db_path, uuid, profiler_hub::version_t{} };
    const profiler_hub::version_t created{ writable->major,
                                           writable->minor,
                                           writable->patch };
    EXPECT_EQ(storage.get_storage_version(), created);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
    {
        ::testing::InSequence sequence;
        expect_schema_exec(sqlite, *writable, uuid);
        backend->initialize_schema(storage.get_storage_version());
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

TEST(sqlite_backend_impl_test, storage_3_0_0_executes_the_embedded_3_0_0_schema)
{
    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema_300.db")
            .string();
    std::filesystem::remove(db_path);

    const profiler_hub::version_t requested{ 3, 0, 0 };
    const embedded_schema_t*      schema = resolve_writable_schema(requested);
    ASSERT_NE(schema, nullptr);
    EXPECT_STREQ(schema->version, "3.0.0");

    const profiler_hub::storage_t storage{ db_path, uuid, requested };
    EXPECT_EQ(storage.get_storage_version(), requested);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
    {
        ::testing::InSequence sequence;
        expect_schema_exec(sqlite, *schema, uuid);
        backend->initialize_schema(storage.get_storage_version());
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

TEST(sqlite_backend_impl_test, storage_3_0_2_is_rejected)
{
    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema_302.db")
            .string();
    std::filesystem::remove(db_path);

    const profiler_hub::version_t requested{ 3, 0, 2 };
    EXPECT_EQ(resolve_writable_schema(requested), nullptr);
    EXPECT_THROW((profiler_hub::storage_t{ db_path, uuid, requested }),
                 std::runtime_error);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
    EXPECT_THROW(backend->initialize_schema(requested), std::runtime_error);
}

TEST(sqlite_backend_impl_test,
     write_statement_resets_and_clears_bindings_after_failed_step)
{
    using ::testing::_;
    using ::testing::Return;

    const std::string uuid = "writetest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_write_failed.db")
            .string();
    std::filesystem::remove(db_path);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);

    auto insert = backend->create_write_statement_executor<int32_t>(
        "INSERT INTO t(val) VALUES (?);");

    // Any non-ok, non-done code stands in for a real constraint violation; the
    // mock never touches a real database.
    constexpr int kConstraintViolation = 19;
    {
        ::testing::InSequence sequence;
        EXPECT_CALL(sqlite, bind_int(_, 1, 42))
            .WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_CALL(sqlite, step(_)).WillOnce(Return(kConstraintViolation));
        EXPECT_CALL(sqlite, reset(_)).WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_CALL(sqlite, clear_bindings(_))
            .WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_THROW(insert(42), std::runtime_error);
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

TEST(sqlite_backend_impl_test,
     write_statement_resets_and_clears_bindings_after_successful_step)
{
    using ::testing::_;
    using ::testing::Return;

    const std::string uuid = "writetest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_write_ok.db")
            .string();
    std::filesystem::remove(db_path);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);

    auto insert = backend->create_write_statement_executor<int32_t>(
        "INSERT INTO t(val) VALUES (?);");

    {
        ::testing::InSequence sequence;
        EXPECT_CALL(sqlite, bind_int(_, 1, 7))
            .WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_CALL(sqlite, step(_)).WillOnce(Return(gmock_sqlite_policy::result_done));
        EXPECT_CALL(sqlite, reset(_)).WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_CALL(sqlite, clear_bindings(_))
            .WillOnce(Return(gmock_sqlite_policy::result_ok));
        EXPECT_NO_THROW(insert(7));
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

TEST(sqlite_backend_impl_test, supported_writable_schemas_lists_writer_versions)
{
    EXPECT_EQ(supported_writable_schemas(), "3.0.0, 3.0.1");
}

TEST(sqlite_backend_impl_test, latest_schema_version_matches_newest_embedded_schema)
{
    const profiler_hub::version_t latest =
        rocpd::data_storage::schema::latest_schema_version();

    // The newest version must be an embedded row, and no embedded row may be newer.
    const embedded_schema_t* newest =
        rocpd::data_storage::schema::find_embedded_schema(latest.to_string());
    ASSERT_NE(newest, nullptr);
    EXPECT_EQ(latest.to_string(), newest->version);
    for(const auto& schema : k_embedded_schemas)
    {
        const profiler_hub::version_t embedded{ schema.major,
                                                schema.minor,
                                                schema.patch };
        EXPECT_FALSE(latest < embedded);
    }
}

class latest_schema_version_test
: public ::testing::TestWithParam<profiler_hub::version_t>
{};

TEST_P(latest_schema_version_test, equals_only_the_newest_versions_yml_entry)
{
    const profiler_hub::version_t version = GetParam();
    const profiler_hub::version_t latest =
        rocpd::data_storage::schema::latest_schema_version();

    const profiler_hub::version_t newest_in_versions_yml{ 3, 0, 5 };

    EXPECT_NE(latest, profiler_hub::version_t{});
    EXPECT_FALSE(latest < version);
    EXPECT_EQ(latest == version, version == newest_in_versions_yml);
}

INSTANTIATE_TEST_SUITE_P(schemas_3_0_0_through_3_0_5,
                         latest_schema_version_test,
                         ::testing::Values(profiler_hub::version_t{ 3, 0, 0 },
                                           profiler_hub::version_t{ 3, 0, 1 },
                                           profiler_hub::version_t{ 3, 0, 2 },
                                           profiler_hub::version_t{ 3, 0, 3 },
                                           profiler_hub::version_t{ 3, 0, 4 },
                                           profiler_hub::version_t{ 3, 0, 5 }));

class embedded_schema_catalog_test
: public ::testing::TestWithParam<profiler_hub::version_t>
{};

TEST_P(embedded_schema_catalog_test, version_is_embedded_from_versions_yml)
{
    const profiler_hub::version_t version = GetParam();
    const embedded_schema_t*      schema =
        rocpd::data_storage::schema::find_embedded_schema(version.to_string());

    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->version, version.to_string());
    EXPECT_EQ(schema->major, version.major);
    EXPECT_EQ(schema->minor, version.minor);
    EXPECT_EQ(schema->patch, version.patch);
    EXPECT_NE(schema->tables, nullptr);
    EXPECT_NE(schema->indexes, nullptr);
    EXPECT_NE(schema->views, nullptr);
    EXPECT_NE(schema->data_views, nullptr);
    EXPECT_NE(schema->summary_views, nullptr);
    EXPECT_NE(schema->metadata, nullptr);
}

INSTANTIATE_TEST_SUITE_P(schemas_3_0_0_through_3_0_5,
                         embedded_schema_catalog_test,
                         ::testing::Values(profiler_hub::version_t{ 3, 0, 0 },
                                           profiler_hub::version_t{ 3, 0, 1 },
                                           profiler_hub::version_t{ 3, 0, 2 },
                                           profiler_hub::version_t{ 3, 0, 3 },
                                           profiler_hub::version_t{ 3, 0, 4 },
                                           profiler_hub::version_t{ 3, 0, 5 }));

TEST(sqlite_backend_impl_test, rejected_schema_names_the_writable_schemas)
{
    const auto db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_schema_reject.db")
            .string();
    std::filesystem::remove(db_path);

    try
    {
        const profiler_hub::storage_t storage{ db_path,
                                               "schematest",
                                               profiler_hub::version_t{ 3, 0, 2 } };
        FAIL() << "storage construction accepted schema 3.0.2";
        (void) storage;
    } catch(const std::runtime_error& error)
    {
        const std::string message = error.what();
        EXPECT_NE(message.find("3.0.2"), std::string::npos);
        EXPECT_NE(message.find(supported_writable_schemas()), std::string::npos);
    }
}

class metadata_stamp_test : public ::testing::TestWithParam<profiler_hub::version_t>
{};

TEST_P(metadata_stamp_test, metadata_sql_contains_requested_schema_version)
{
    const profiler_hub::version_t version = GetParam();
    const embedded_schema_t*      schema  = resolve_writable_schema(version);
    ASSERT_NE(schema, nullptr);
    EXPECT_EQ(schema->version, version.to_string());

    const std::string uuid = "schematest";
    const auto        db_path =
        (std::filesystem::temp_directory_path() / "profiler_hub_mock_metadata_stamp.db")
            .string();
    std::filesystem::remove(db_path);

    ::testing::NiceMock<mock_sqlite_api> sqlite;
    const sqlite_mock_guard              guard{ sqlite };
    auto                                 backend =
        mock_backend_t::create(db_path, uuid, mock_backend_t::storage_mode_t::in_memory);
    {
        ::testing::InSequence sequence;
        expect_schema_exec(sqlite, *schema, uuid);
        backend->initialize_schema(version);
    }
    ASSERT_TRUE(::testing::Mock::VerifyAndClearExpectations(&sqlite));
}

INSTANTIATE_TEST_SUITE_P(writable_schemas,
                         metadata_stamp_test,
                         ::testing::Values(profiler_hub::version_t{ 3, 0, 0 },
                                           profiler_hub::version_t{ 3, 0, 1 }));

#ifdef _WIN32
// profiler-hub.dll, libprofiler-hub.dll, and an optional -<version> suffix.
// The unit-test executable also contains the schema headers, so match the DLL name.
bool
is_profiler_hub_dll(std::wstring_view module_name)
{
    std::wstring name(module_name);
    for(wchar_t& ch : name)
    {
        if(ch >= L'A' && ch <= L'Z')
        {
            ch = static_cast<wchar_t>(ch - L'A' + L'a');
        }
    }

    constexpr std::wstring_view k_suffix = L".dll";
    if(name.size() < k_suffix.size() ||
       name.compare(name.size() - k_suffix.size(), k_suffix.size(), k_suffix) != 0)
    {
        return false;
    }
    name.resize(name.size() - k_suffix.size());

    constexpr std::wstring_view k_lib_prefix = L"lib";
    if(name.compare(0, k_lib_prefix.size(), k_lib_prefix) == 0)
    {
        name.erase(0, k_lib_prefix.size());
    }

    constexpr std::wstring_view k_stem = L"profiler-hub";
    if(name.compare(0, k_stem.size(), k_stem) != 0)
    {
        return false;
    }
    if(name.size() == k_stem.size())
    {
        return true;
    }
    if(name[k_stem.size()] != L'-')
    {
        return false;
    }
    for(std::size_t index = k_stem.size() + 1; index < name.size(); ++index)
    {
        const wchar_t ch = name[index];
        if((ch < L'0' || ch > L'9') && ch != L'.')
        {
            return false;
        }
    }
    return name.size() > k_stem.size() + 1;
}

std::filesystem::path
loaded_profiler_hub_library_windows()
{
    const HANDLE snapshot =
        CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if(snapshot == INVALID_HANDLE_VALUE)
    {
        return {};
    }

    MODULEENTRY32W entry = {};
    entry.dwSize         = sizeof(entry);
    HMODULE module       = nullptr;
    if(Module32FirstW(snapshot, &entry) != FALSE)
    {
        do
        {
            if(is_profiler_hub_dll(entry.szModule))
            {
                module = entry.hModule;
                break;
            }
        } while(Module32NextW(snapshot, &entry) != FALSE);
    }
    CloseHandle(snapshot);
    if(module == nullptr)
    {
        return {};
    }

    std::wstring path(32768, L'\0');
    const DWORD  written =
        GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if(written == 0 || written >= path.size())
    {
        return {};
    }
    path.resize(written);
    return std::filesystem::path(path);
}
#endif

// The schema SQL is compiled into the profiler-hub shared library. Read that
// loaded file. Linux lists it in /proc/self/maps. Windows lists it in the
// process module list.
std::filesystem::path
loaded_profiler_hub_library()
{
#ifdef _WIN32
    return loaded_profiler_hub_library_windows();
#else
    std::ifstream maps("/proc/self/maps");
    std::string   line;
    while(std::getline(maps, line))
    {
        const auto path_at = line.rfind(' ');
        if(path_at == std::string::npos)
        {
            continue;
        }
        std::string path = line.substr(path_at + 1);
        if(path.find("libprofiler-hub.so") == std::string::npos)
        {
            continue;
        }
        const auto deleted = path.find(" (deleted)");
        if(deleted != std::string::npos)
        {
            path.resize(deleted);
        }
        return path;
    }
    return {};
#endif
}

TEST(sqlite_backend_impl_test, shared_library_contains_each_embedded_schema)
{
    const std::filesystem::path library_path = loaded_profiler_hub_library();
    ASSERT_FALSE(library_path.empty());

    std::ifstream library(library_path, std::ios::binary);
    ASSERT_TRUE(library.is_open()) << library_path.string();
    const std::string bytes((std::istreambuf_iterator<char>(library)),
                            std::istreambuf_iterator<char>());

    for(const auto& schema : k_embedded_schemas)
    {
        SCOPED_TRACE(schema.version);
        const std::string banner =
            std::string("-- RocPD schema version ") + schema.version;
        EXPECT_NE(bytes.find(banner), std::string::npos);
    }
}

}  // namespace
