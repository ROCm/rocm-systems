// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/storage.hpp"
#include "profiler-hub/version.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include "backends/sqlite_backend.hpp"
#include "schema/cache/schema_cache.hpp"
#include "schema/errors/validation_error.hpp"
#include "schema/pmc_event_input.hpp"
#include "schema/validation/schema_validator.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace
{

using namespace profiler_hub;
using namespace profiler_hub::schema;

const column_model*
column(const schema_model& model, const char* table, const char* name)
{
    const auto* found = model.find_table(table);
    if(found == nullptr) return nullptr;
    return found->find(name);
}

const validation_error*
find_error(const validation_result& result, std::string_view field)
{
    for(const auto& error : result.errors)
    {
        if(error.field == field) return &error;
    }
    return nullptr;
}

TEST(schema_cache_test, parses_on_first_access_and_reuses_the_model)
{
    schema_cache cache;

    const schema_model& first = cache.get(version_t{ 3, 0, 5 });
    EXPECT_EQ(cache.parse_count(), 1u);
    ASSERT_NE(first.find_table("rocpd_pmc_event"), nullptr);

    const schema_model& second = cache.get(version_t{ 3, 0, 5 });
    EXPECT_EQ(cache.parse_count(), 1u);
    EXPECT_EQ(&first, &second);
}

TEST(schema_cache_test, parses_each_version_once)
{
    schema_cache cache;

    const schema_model& v305  = cache.get(version_t{ 3, 0, 5 });
    const schema_model& again = cache.get(version_t{ 3, 0, 5 });
    const schema_model& v303  = cache.get(version_t{ 3, 0, 3 });

    EXPECT_EQ(&v305, &again);
    EXPECT_NE(&v305, &v303);
    EXPECT_EQ(cache.parse_count(), 2u);

    (void) cache.get(version_t{ 3, 0, 3 });
    EXPECT_EQ(cache.parse_count(), 2u);
}

TEST(schema_cache_test, pmc_event_columns_follow_the_schema_sql)
{
    schema_cache cache;

    const auto& v301 = cache.get(version_t{ 3, 0, 1 });
    EXPECT_EQ(column(v301, "rocpd_pmc_event", "sample_id"), nullptr);
    EXPECT_EQ(column(v301, "rocpd_pmc_event", "xcc"), nullptr);
    EXPECT_EQ(column(v301, "rocpd_graph_launch", "id"), nullptr);

    const auto& v303 = cache.get(version_t{ 3, 0, 3 });
    ASSERT_NE(column(v303, "rocpd_pmc_event", "sample_id"), nullptr);
    ASSERT_NE(column(v303, "rocpd_pmc_event", "xcc"), nullptr);
    ASSERT_NE(column(v303, "rocpd_pmc_event", "shader_engine"), nullptr);
    ASSERT_NE(column(v303, "rocpd_pmc_event", "instance"), nullptr);

    const auto& v305 = cache.get(version_t{ 3, 0, 5 });
    ASSERT_NE(column(v305, "rocpd_pmc_event", "sample_id"), nullptr);
    EXPECT_EQ(column(v305, "rocpd_pmc_event", "xcc"), nullptr);
    EXPECT_EQ(column(v305, "rocpd_pmc_event", "shader_engine"), nullptr);
    EXPECT_EQ(column(v305, "rocpd_pmc_event", "instance"), nullptr);

    const auto* pmc_id = column(v305, "rocpd_pmc_event", "pmc_id");
    const auto* id     = column(v305, "rocpd_pmc_event", "id");
    const auto* guid   = column(v305, "rocpd_pmc_event", "guid");
    ASSERT_NE(pmc_id, nullptr);
    ASSERT_NE(id, nullptr);
    ASSERT_NE(guid, nullptr);
    EXPECT_TRUE(pmc_id->required_from_user());
    EXPECT_FALSE(id->required_from_user());
    EXPECT_FALSE(guid->required_from_user());
    EXPECT_FALSE(column(v305, "rocpd_pmc_event", "sample_id")->required_from_user());
}

TEST(schema_cache_test, reads_check_constraints_from_the_sql)
{
    schema_cache cache;

    const auto* v300 =
        column(cache.get(version_t{ 3, 0, 0 }), "rocpd_info_agent", "type");
    const auto* v301 =
        column(cache.get(version_t{ 3, 0, 1 }), "rocpd_info_agent", "type");
    ASSERT_NE(v300, nullptr);
    ASSERT_NE(v301, nullptr);
    EXPECT_EQ(v300->allowed_values, (std::vector<std::string>{ "CPU", "GPU" }));
    EXPECT_EQ(v301->allowed_values, (std::vector<std::string>{ "CPU", "GPU", "NIC" }));
}

TEST(schema_cache_test, unknown_version_is_not_parsed)
{
    schema_cache cache;
    EXPECT_THROW((void) cache.get(version_t{ 9, 9, 9 }), std::invalid_argument);
    EXPECT_EQ(cache.parse_count(), 0u);
}

TEST(schema_validator_test, rejects_xcc_on_3_0_5)
{
    pmc_event_input_t input;
    input.pmc_id = 1;
    input.xcc    = 1;

    const auto result = schema_validator::validate(input, version_t{ 3, 0, 5 });

    EXPECT_FALSE(result.success());
    const auto* error = find_error(result, "xcc");
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(error->object, "PMCEvent");
    EXPECT_EQ(error->documentation, "docs/schema/3.0.5_usage.md");
    EXPECT_NE(error->reason.find("not supported"), std::string::npos);
}

TEST(schema_validator_test, accepts_sample_id_on_3_0_3)
{
    pmc_event_input_t input;
    // pmc_id is NOT NULL with no default, so a row that only sets sample_id is
    // incomplete.
    input.pmc_id    = 1;
    input.sample_id = 100;

    const auto result = schema_validator::validate(input, version_t{ 3, 0, 3 });

    EXPECT_TRUE(result.success());
    EXPECT_TRUE(result.errors.empty());
}

TEST(schema_validator_test, rejects_sample_id_on_3_0_1)
{
    pmc_event_input_t input;
    input.pmc_id    = 1;
    input.sample_id = 100;

    const auto result = schema_validator::validate(input, version_t{ 3, 0, 1 });

    EXPECT_FALSE(result.success());
    const auto* error = find_error(result, "sample_id");
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(error->documentation, documentation_link(version_t{ 3, 0, 1 }));
}

TEST(schema_validator_test, reports_missing_pmc_id)
{
    pmc_event_input_t input;
    input.value = 1.5;

    const auto result = schema_validator::validate(input, version_t{ 3, 0, 5 });

    EXPECT_FALSE(result.success());
    const auto* error = find_error(result, "pmc_id");
    ASSERT_NE(error, nullptr);
    EXPECT_NE(error->reason.find("required"), std::string::npos);
}

TEST(schema_validator_test, rejects_unknown_table)
{
    const auto result = schema_validator::validate_fields(
        "GraphLaunch", "rocpd_graph_launch", version_t{ 3, 0, 1 }, {});

    EXPECT_FALSE(result.success());
    ASSERT_EQ(result.errors.size(), 1u);
    EXPECT_TRUE(result.errors[0].field.empty());
    EXPECT_EQ(result.object, "GraphLaunch");
    EXPECT_NE(result.errors[0].reason.find("not supported"), std::string::npos);
    EXPECT_EQ(result.errors[0].documentation, "docs/schema/3.0.1_usage.md");
}

TEST(schema_validator_test, rejects_enum_value_missing_from_check)
{
    const std::vector<field_assignment> fields{
        { "nid", true, std::nullopt },
        { "pid", true, std::nullopt },
        { "type", true, std::string{ "NIC" } },
    };

    const auto result = schema_validator::validate_fields(
        "Agent", "rocpd_info_agent", version_t{ 3, 0, 0 }, fields);

    EXPECT_FALSE(result.success());
    const auto* error = find_error(result, "type");
    ASSERT_NE(error, nullptr);
    EXPECT_NE(error->reason.find("NIC"), std::string::npos);
    EXPECT_NE(error->reason.find("CPU"), std::string::npos);
    EXPECT_NE(error->reason.find("GPU"), std::string::npos);
    EXPECT_EQ(error->documentation, "docs/schema/3.0.0_usage.md");
}

TEST(schema_validator_test, accepts_enum_value_present_in_check)
{
    const std::vector<field_assignment> fields{
        { "nid", true, std::nullopt },
        { "pid", true, std::nullopt },
        { "type", true, std::string{ "NIC" } },
    };

    const auto result = schema_validator::validate_fields(
        "Agent", "rocpd_info_agent", version_t{ 3, 0, 1 }, fields);

    EXPECT_TRUE(result.success()) << result.format();
}

TEST(validation_error_test, documentation_link_uses_the_schema_version)
{
    EXPECT_EQ(documentation_link(version_t{ 3, 0, 0 }), "docs/schema/3.0.0_usage.md");
    EXPECT_EQ(documentation_link(version_t{ 3, 0, 3 }), "docs/schema/3.0.3_usage.md");
    EXPECT_EQ(documentation_link(version_t{ 3, 0, 5 }), "docs/schema/3.0.5_usage.md");
}

TEST(validation_error_test, formats_unsupported_field)
{
    pmc_event_input_t input;
    input.pmc_id = 1;
    input.xcc    = 1;

    const auto result = schema_validator::validate(input, version_t{ 3, 0, 5 });

    const std::string expected = "Validation Failed\n"
                                 "\n"
                                 "Schema:\n"
                                 "    3.0.5\n"
                                 "\n"
                                 "Object:\n"
                                 "    PMCEvent\n"
                                 "\n"
                                 "Field:\n"
                                 "    xcc\n"
                                 "\n"
                                 "Reason:\n"
                                 "    Field is not supported\n"
                                 "    by schema version 3.0.5\n"
                                 "\n"
                                 "Documentation:\n"
                                 "    docs/schema/3.0.5_usage.md\n";
    EXPECT_EQ(result.format(), expected);
}

class pmc_event_write_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const auto* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_db_path             = (std::filesystem::temp_directory_path() /
                     (std::string{ "pmc_event_write_" } + test_info->name() + ".db"))
                        .string();
        std::filesystem::remove(m_db_path);
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    [[nodiscard]] std::unique_ptr<writer_t> make_writer(version_t schema_version) const
    {
        return std::make_unique<writer_t>(
            std::make_unique<storage_t>(m_db_path, m_uuid, schema_version));
    }

    static void register_counter(writer_t& writer)
    {
        const writer_types::node_info_t node_info{ 1, 42, "machine-1" };
        writer.register_node_info(node_info);

        writer_types::process_info_t process_info;
        process_info.pid     = 100;
        process_info.node_id = 1;
        writer.register_process_info(process_info);

        writer_types::pmc_info_t pmc_info;
        pmc_info.unique_id.name = "occupancy";
        pmc_info.symbol         = "occupancy_symbol";
        pmc_info.node_id        = 1;
        pmc_info.process_id     = 100;
        writer.register_pmc_info(pmc_info);
    }

    [[nodiscard]] int64_t count_pmc_events() const
    {
        struct row_t
        {
            int64_t total = 0;
        };

        auto backend = data_storage::sqlite_backend::create(
            m_db_path, m_uuid, data_storage::sqlite_backend::storage_mode_t::on_disk);
        const std::string query = "SELECT COUNT(*) FROM rocpd_pmc_event_" + m_uuid;
        auto read = backend->create_read_statement_executor<row_t>(query, &row_t::total);
        const auto rows = read().to_vector();
        if(rows.empty()) return 0;
        return rows.front().total;
    }

    std::string m_db_path;
    std::string m_uuid = "pmcval000000";
};

TEST_F(pmc_event_write_test, skips_insert_when_validation_fails)
{
    auto writer = make_writer(version_t{ 3, 0, 5 });

    pmc_event_input_t input;
    input.pmc_id = 1;
    input.xcc    = 1;

    writer_types::pmc_event_data_t event;
    event.value = 1.5;
    const writer_types::pmc_info_unique_id_t pmc_id{ "occupancy", std::nullopt };

    // The counter is not registered. insert_pmc_event_data would throw, so a
    // successful return means the writer was not called.
    const auto result = schema_validator::write_pmc_event(
        *writer, input, version_t{ 3, 0, 5 }, event, pmc_id);

    EXPECT_FALSE(result.success());
    ASSERT_NE(find_error(result, "xcc"), nullptr);

    writer->flush_in_memory_data_to_disk();
    writer.reset();
    EXPECT_EQ(count_pmc_events(), 0);
}

TEST_F(pmc_event_write_test, inserts_when_validation_succeeds)
{
    auto writer = make_writer(version_t{ 3, 0, 3 });
    register_counter(*writer);

    pmc_event_input_t input;
    input.pmc_id    = 1;
    input.sample_id = 100;

    writer_types::pmc_event_data_t event;
    event.value            = 1.5;
    event.sample.timestamp = 1000;
    const writer_types::pmc_info_unique_id_t pmc_id{ "occupancy", std::nullopt };

    const auto result = schema_validator::write_pmc_event(
        *writer, input, version_t{ 3, 0, 3 }, event, pmc_id);

    EXPECT_TRUE(result.success()) << result.format();

    writer->flush_in_memory_data_to_disk();
    writer.reset();
    EXPECT_EQ(count_pmc_events(), 1);
}

}  // namespace
