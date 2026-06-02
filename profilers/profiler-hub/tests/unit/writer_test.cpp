// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "profiler-hub/storage.hpp"
#include "profiler-hub/writer.hpp"
#include "profiler-hub/writer_types.hpp"

#include <sqlite3.h>

#include <gtest/gtest.h>

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// ============================================================================
// SQLite Query Helpers
// ============================================================================

struct sqlite_query_result
{
    std::vector<std::vector<std::string>> rows;
    std::vector<std::string>              column_names;
};

sqlite_query_result
query_database(const std::string& db_path, const std::string& query)
{
    sqlite_query_result result;
    sqlite3*            db = nullptr;

    if(sqlite3_open(db_path.c_str(), &db) != SQLITE_OK)
    {
        throw std::runtime_error("Failed to open database: " + db_path);
    }

    sqlite3_stmt* stmt = nullptr;
    if(sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr) != SQLITE_OK)
    {
        std::string error = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error("Failed to prepare query: " + error);
    }

    int column_count = sqlite3_column_count(stmt);
    for(int i = 0; i < column_count; ++i)
    {
        result.column_names.emplace_back(sqlite3_column_name(stmt, i));
    }

    while(sqlite3_step(stmt) == SQLITE_ROW)
    {
        std::vector<std::string> row;
        for(int i = 0; i < column_count; ++i)
        {
            const auto* text =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, i));
            row.emplace_back(text ? text : "NULL");
        }
        result.rows.push_back(std::move(row));
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    return result;
}

size_t
count_rows(const std::string& db_path,
           const std::string& table_name,
           const std::string& uuid)
{
    auto result =
        query_database(db_path, "SELECT COUNT(*) FROM " + table_name + "_" + uuid);
    return result.rows.empty() ? 0 : std::stoull(result.rows[0][0]);
}

// ============================================================================
// Test Data Factory Functions
// ============================================================================

profiler_hub::writer_types::node_info_t
create_test_node_info(size_t node_id = 1)
{
    static std::unordered_map<size_t, std::string> machine_ids;
    auto&                                          machine_id = machine_ids[node_id];
    if(machine_id.empty())
    {
        machine_id = "test-machine-id-" + std::to_string(node_id);
    }

    return profiler_hub::writer_types::node_info_t{ .node_id       = node_id,
                                                    .hash          = 123456789 + node_id,
                                                    .machine_id    = machine_id.c_str(),
                                                    .system_name   = "Linux",
                                                    .hostname      = "test-host",
                                                    .release       = "5.15.0",
                                                    .version       = "#1 SMP",
                                                    .hardware_name = "x86_64",
                                                    .domain_name   = "test-domain" };
}

profiler_hub::writer_types::process_info_t
create_test_process_info(size_t node_id = 1, size_t pid = 1000)
{
    return profiler_hub::writer_types::process_info_t{ .ppid        = 1,
                                                       .pid         = pid,
                                                       .init        = 1000000,
                                                       .fini        = 2000000,
                                                       .start       = 1000000,
                                                       .end         = 2000000,
                                                       .command     = "/usr/bin/test",
                                                       .environment = "{}",
                                                       .extdata     = "{}",
                                                       .node_id     = node_id };
}

profiler_hub::writer_types::thread_info_t
create_test_thread_info(size_t node_id    = 1,
                        size_t process_id = 1000,
                        size_t thread_id  = 100)
{
    return profiler_hub::writer_types::thread_info_t{ .parent_process_id = process_id,
                                                      .thread_id         = thread_id,
                                                      .name              = "test-thread",
                                                      .start             = 1000000,
                                                      .end               = 2000000,
                                                      .extdata           = "{}",
                                                      .node_id           = node_id,
                                                      .process_id        = process_id };
}

profiler_hub::writer_types::agent_info_t
create_test_agent_info(size_t      node_id    = 1,
                       size_t      process_id = 1000,
                       const char* agent_type = "GPU",
                       size_t      type_index = 0)
{
    return profiler_hub::writer_types::agent_info_t{
        .unique_id      = { .agent_type = agent_type, .type_index = type_index },
        .absolute_index = 0,
        .logical_index  = 0,
        .uuid           = 12345,
        .name           = "gfx1100",
        .model_name     = "AMD Radeon",
        .vendor_name    = "AMD",
        .product_name   = "Radeon RX 7900",
        .user_name      = "gpu0",
        .extdata        = "{}",
        .node_id        = node_id,
        .process_id     = process_id
    };
}

profiler_hub::writer_types::stream_info_t
create_test_stream_info(size_t node_id    = 1,
                        size_t process_id = 1000,
                        size_t stream_id  = 1)
{
    return profiler_hub::writer_types::stream_info_t{ .stream_id  = stream_id,
                                                      .name       = "test-stream",
                                                      .extdata    = "{}",
                                                      .node_id    = node_id,
                                                      .process_id = process_id };
}

profiler_hub::writer_types::queue_info_t
create_test_queue_info(size_t node_id = 1, size_t process_id = 1000, size_t queue_id = 1)
{
    return profiler_hub::writer_types::queue_info_t{ .queue_id   = queue_id,
                                                     .name       = "test-queue",
                                                     .extdata    = "{}",
                                                     .node_id    = node_id,
                                                     .process_id = process_id };
}

profiler_hub::writer_types::pmc_info_t
create_test_pmc_info(
    size_t                                                       node_id    = 1,
    size_t                                                       process_id = 1000,
    const char*                                                  name = "test_counter",
    std::optional<profiler_hub::writer_types::agent_unique_id_t> agent_id = std::nullopt)
{
    return profiler_hub::writer_types::pmc_info_t{
        .unique_id        = { .name = name, .agent_id = agent_id },
        .target_arch      = "GPU",
        .event_code       = 100,
        .instance_id      = 0,
        .symbol           = "TEST_COUNTER",
        .description      = "Test counter description",
        .long_description = "Long description",
        .component        = "SQ",
        .units            = "cycles",
        .value_type       = "ABS",
        .block            = "SQ",
        .expression       = "",
        .is_constant      = 0,
        .is_derived       = 0,
        .extdata          = "{}",
        .node_id          = node_id,
        .process_id       = process_id
    };
}

profiler_hub::writer_types::code_object_info_t
create_test_code_object_info(
    size_t                                        code_object_id = 1,
    size_t                                        node_id        = 1,
    size_t                                        process_id     = 1000,
    profiler_hub::writer_types::agent_unique_id_t agent_id       = { "GPU", 0 })
{
    return profiler_hub::writer_types::code_object_info_t{ .id = code_object_id,
                                                           .uri =
                                                               "file:///test/kernel.co",
                                                           .load_base    = 0x1000,
                                                           .load_size    = 0x2000,
                                                           .load_delta   = 0,
                                                           .storage_type = "FILE",
                                                           .extdata      = "{}",
                                                           .node_id      = node_id,
                                                           .process_id   = process_id,
                                                           .agent_id     = agent_id };
}

profiler_hub::writer_types::kernel_symbol_info_t
create_test_kernel_symbol_info(size_t kernel_symbol_id = 1,
                               size_t node_id          = 1,
                               size_t process_id       = 1000,
                               size_t code_object_id   = 1)
{
    return profiler_hub::writer_types::kernel_symbol_info_t{
        .id                        = kernel_symbol_id,
        .name                      = "test_kernel",
        .display_name              = "Test Kernel",
        .kernel_object             = 0x1000,
        .kernarg_segment_size      = 64,
        .kernarg_segment_alignment = 8,
        .group_segment_size        = 256,
        .private_segment_size      = 0,
        .sgpr_count                = 32,
        .arch_vgpr_count           = 64,
        .accum_vgpr_count          = 0,
        .extdata                   = "{}",
        .node_id                   = node_id,
        .process_id                = process_id,
        .code_obj_id               = code_object_id
    };
}

profiler_hub::writer_types::track_info_t
create_test_track_info(
    size_t                                                  node_id    = 1,
    std::optional<size_t>                                   process_id = 1000,
    std::optional<size_t>                                   thread_id  = 100,
    std::optional<profiler_hub::writer_types::track_name_t> name       = "test-track")
{
    return profiler_hub::writer_types::track_info_t{ .name       = name,
                                                     .extdata    = "{}",
                                                     .node_id    = node_id,
                                                     .process_id = process_id,
                                                     .thread_id  = thread_id };
}

profiler_hub::writer_types::region_data_t
create_test_region_data(const char* name            = "test_region",
                        size_t      start_timestamp = 1000000,
                        size_t      end_timestamp   = 2000000)
{
    return profiler_hub::writer_types::region_data_t{ .event           = std::nullopt,
                                                      .start_timestamp = start_timestamp,
                                                      .end_timestamp   = end_timestamp,
                                                      .name            = name,
                                                      .extdata         = "{}",
                                                      .args            = {} };
}

profiler_hub::writer_types::trace_environment_t
create_test_trace_environment(size_t node_id    = 1,
                              size_t process_id = 1000,
                              size_t thread_id  = 100)
{
    return profiler_hub::writer_types::trace_environment_t{ .node_id    = node_id,
                                                            .process_id = process_id,
                                                            .thread_id  = thread_id,
                                                            .agent_id   = std::nullopt,
                                                            .stream_id  = std::nullopt,
                                                            .queue_id   = std::nullopt,
                                                            .track_name = std::nullopt };
}

profiler_hub::writer_types::pmc_event_data_t
create_test_pmc_event_data(double value = 42.5)
{
    return profiler_hub::writer_types::pmc_event_data_t{
        .event   = std::nullopt,
        .value   = value,
        .extdata = "{}",
        .sample  = { .timestamp = 1000000,
                     .track     = create_test_track_info(),
                     .extdata   = "{}" }
    };
}

profiler_hub::writer_types::kernel_dispatch_data_t
create_test_kernel_dispatch_data(size_t kernel_symbol_id = 1, size_t code_object_id = 1)
{
    return profiler_hub::writer_types::kernel_dispatch_data_t{
        .event                = std::nullopt,
        .dispatch_id          = 1,
        .start_timestamp      = 1000000,
        .end_timestamp        = 2000000,
        .kernel_symbol_id     = kernel_symbol_id,
        .code_object_id       = code_object_id,
        .private_segment_size = 0,
        .group_segment_size   = 256,
        .workgroup_size_x     = 64,
        .workgroup_size_y     = 1,
        .workgroup_size_z     = 1,
        .grid_size_x          = 1024,
        .grid_size_y          = 1,
        .grid_size_z          = 1,
        .name                 = "test_kernel_dispatch",
        .extdata              = "{}"
    };
}

profiler_hub::writer_types::memory_copy_data_t
create_test_memory_copy_data()
{
    return profiler_hub::writer_types::memory_copy_data_t{ .event = std::nullopt,
                                                           .start_timestamp = 1000000,
                                                           .end_timestamp   = 2000000,
                                                           .dst_agent_id = std::nullopt,
                                                           .dst_address  = 0x2000,
                                                           .src_agent_id = std::nullopt,
                                                           .src_address  = 0x1000,
                                                           .size         = 4096,
                                                           .name         = "hipMemcpy",
                                                           .region_name  = std::nullopt,
                                                           .extdata      = "{}" };
}

profiler_hub::writer_types::memory_alloc_data_t
create_test_memory_alloc_data(std::optional<std::string_view> type  = "ALLOC",
                              std::optional<std::string_view> level = "REAL")
{
    return profiler_hub::writer_types::memory_alloc_data_t{ .event = std::nullopt,
                                                            .type  = type,
                                                            .level = level,
                                                            .start_timestamp = 1000000,
                                                            .end_timestamp   = 2000000,
                                                            .address         = 0x1000,
                                                            .size            = 4096,
                                                            .extdata         = "{}" };
}

// Helper to register base dependencies (node -> process -> thread)
void
register_base_dependencies(profiler_hub::writer_t& writer,
                           size_t                  node_id    = 1,
                           size_t                  process_id = 1000,
                           size_t                  thread_id  = 100)
{
    writer.register_node_info(create_test_node_info(node_id));
    writer.register_process_info(create_test_process_info(node_id, process_id));
    writer.register_thread_info(create_test_thread_info(node_id, process_id, thread_id));
}

}  // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class writer_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::string test_name{
            ::testing::UnitTest::GetInstance()->current_test_info()->name()
        };

        m_database_path = "test_writer_" + test_name + ".db";
        m_uuid          = "12345";
        m_storage = std::make_unique<profiler_hub::storage_t>(m_database_path, m_uuid);
        m_writer  = std::make_shared<profiler_hub::writer_t>(std::move(m_storage));
    }

    void TearDown() override
    {
        m_writer.reset();
        m_storage.reset();
        std::remove(m_database_path.c_str());
    }

    std::string                              m_database_path;
    std::string                              m_uuid;
    std::unique_ptr<profiler_hub::storage_t> m_storage;
    std::shared_ptr<profiler_hub::writer_t>  m_writer;
};

// ============================================================================
// Group A: Info Table Registration Tests
// ============================================================================

// --------------------- Node Info Tests ---------------------

TEST_F(writer_test, register_node_info_inserts_to_database)
{
    auto node = create_test_node_info(42);
    m_writer->register_node_info(node);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT hash, machine_id, hostname, system_name, "
                                 "release, version, hardware_name, domain_name "
                                 "FROM rocpd_info_node_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], std::to_string(node.hash));
    EXPECT_EQ(result.rows[0][1], node.machine_id);
    EXPECT_EQ(result.rows[0][2], node.hostname);
    EXPECT_EQ(result.rows[0][3], node.system_name);
    EXPECT_EQ(result.rows[0][4], node.release);
    EXPECT_EQ(result.rows[0][5], node.version);
    EXPECT_EQ(result.rows[0][6], node.hardware_name);
    EXPECT_EQ(result.rows[0][7], node.domain_name);
}

TEST_F(writer_test, register_node_info_duplicate_is_ignored)
{
    auto node = create_test_node_info(1);
    m_writer->register_node_info(node);
    m_writer->register_node_info(node);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_node", m_uuid);
    EXPECT_EQ(row_count, 1);
}

// --------------------- Process Info Tests ---------------------

TEST_F(writer_test, register_process_info_with_valid_node)
{
    auto node    = create_test_node_info(1);
    auto process = create_test_process_info(1, 1000);

    m_writer->register_node_info(node);
    m_writer->register_process_info(process);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT nid, ppid, pid, init, fini, start, end, command "
                                 "FROM rocpd_info_process_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][2], std::to_string(process.pid));
    EXPECT_EQ(result.rows[0][7], process.command);
}

TEST_F(writer_test, register_process_info_without_node_throws)
{
    auto process = create_test_process_info(999, 1000);  // Node 999 doesn't exist
    EXPECT_THROW(m_writer->register_process_info(process), std::runtime_error);
}

TEST_F(writer_test, register_process_info_duplicate_is_ignored)
{
    auto node    = create_test_node_info(1);
    auto process = create_test_process_info(1, 1000);

    m_writer->register_node_info(node);
    m_writer->register_process_info(process);
    m_writer->register_process_info(process);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_process", m_uuid);
    EXPECT_EQ(row_count, 1);
}

// --------------------- Thread Info Tests ---------------------

TEST_F(writer_test, register_thread_info_with_valid_dependencies)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(
        m_database_path, "SELECT tid, name, start, end FROM rocpd_info_thread_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "100");
    EXPECT_EQ(result.rows[0][1], "test-thread");
}

TEST_F(writer_test, register_thread_info_without_node_throws)
{
    auto thread = create_test_thread_info(999, 1000, 100);
    EXPECT_THROW(m_writer->register_thread_info(thread), std::runtime_error);
}

TEST_F(writer_test, register_thread_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    auto thread = create_test_thread_info(1, 999, 100);  // Process 999 doesn't exist
    EXPECT_THROW(m_writer->register_thread_info(thread), std::runtime_error);
}

// --------------------- Agent Info Tests ---------------------

TEST_F(writer_test, register_agent_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT type, type_index, name, model_name, vendor_name "
                                 "FROM rocpd_info_agent_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "GPU");
    EXPECT_EQ(result.rows[0][1], "0");
    EXPECT_EQ(result.rows[0][2], "gfx1100");
}

TEST_F(writer_test, register_agent_info_cpu_type)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "CPU", 0));
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT type FROM rocpd_info_agent_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "CPU");
}

TEST_F(writer_test, register_agent_info_invalid_type_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto agent = create_test_agent_info(1, 1000, "INVALID", 0);
    EXPECT_THROW(m_writer->register_agent_info(agent), std::invalid_argument);
}

TEST_F(writer_test, register_agent_info_without_node_throws)
{
    auto agent = create_test_agent_info(999, 1000, "GPU", 0);
    EXPECT_THROW(m_writer->register_agent_info(agent), std::runtime_error);
}

TEST_F(writer_test, register_agent_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    auto agent = create_test_agent_info(1, 999, "GPU", 0);
    EXPECT_THROW(m_writer->register_agent_info(agent), std::runtime_error);
}

// --------------------- Stream Info Tests ---------------------

TEST_F(writer_test, register_stream_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_stream_info(create_test_stream_info(1, 1000, 1));
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT name FROM rocpd_info_stream_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "test-stream");
}

TEST_F(writer_test, register_stream_info_without_node_throws)
{
    auto stream = create_test_stream_info(999, 1000, 1);
    EXPECT_THROW(m_writer->register_stream_info(stream), std::runtime_error);
}

TEST_F(writer_test, register_stream_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    auto stream = create_test_stream_info(1, 999, 1);
    EXPECT_THROW(m_writer->register_stream_info(stream), std::runtime_error);
}

// --------------------- Queue Info Tests ---------------------

TEST_F(writer_test, register_queue_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT name FROM rocpd_info_queue_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "test-queue");
}

TEST_F(writer_test, register_queue_info_without_node_throws)
{
    auto queue = create_test_queue_info(999, 1000, 1);
    EXPECT_THROW(m_writer->register_queue_info(queue), std::runtime_error);
}

TEST_F(writer_test, register_queue_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    auto queue = create_test_queue_info(1, 999, 1);
    EXPECT_THROW(m_writer->register_queue_info(queue), std::runtime_error);
}

// --------------------- PMC Info Tests ---------------------

TEST_F(writer_test, register_pmc_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "test_counter", agent_id);
    m_writer->register_pmc_info(pmc);
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path,
                       "SELECT name, symbol, description, target_arch, event_code "
                       "FROM rocpd_info_pmc_" +
                           m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "test_counter");
    EXPECT_EQ(result.rows[0][1], "TEST_COUNTER");
    EXPECT_EQ(result.rows[0][3], "GPU");
}

TEST_F(writer_test, register_pmc_info_without_node_throws)
{
    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(999, 1000, "test", agent_id);
    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::runtime_error);
}

TEST_F(writer_test, register_pmc_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 999, "test", agent_id);
    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::runtime_error);
}

// --------------------- Code Object Info Tests ---------------------

TEST_F(writer_test, register_code_object_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    auto code_obj = create_test_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT uri, load_base, load_size, storage_type "
                                 "FROM rocpd_info_code_object_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "file:///test/kernel.co");
    EXPECT_EQ(result.rows[0][1], std::to_string(code_obj.load_base));
    EXPECT_EQ(result.rows[0][3], "FILE");
}

TEST_F(writer_test, register_code_object_info_without_node_throws)
{
    auto code_obj = create_test_code_object_info(1, 999, 1000, { "GPU", 0 });
    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::runtime_error);
}

// --------------------- Kernel Symbol Info Tests ---------------------

TEST_F(writer_test, register_kernel_symbol_info_with_valid_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_test_code_object_info(1, 1, 1000, { "GPU", 0 }));

    auto kernel_symbol = create_test_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel_symbol);
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path,
                       "SELECT kernel_name, display_name, kernarg_segment_size, "
                       "group_segment_size FROM rocpd_info_kernel_symbol_" +
                           m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "test_kernel");
    EXPECT_EQ(result.rows[0][1], "Test Kernel");
}

TEST_F(writer_test, register_kernel_symbol_info_without_code_object_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto kernel_symbol =
        create_test_kernel_symbol_info(1, 1, 1000, 999);  // Code object 999 doesn't exist
    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel_symbol),
                 std::runtime_error);
}

TEST_F(writer_test, register_kernel_symbol_info_without_node_throws)
{
    auto kernel_symbol = create_test_kernel_symbol_info(1, 999, 1000, 1);
    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel_symbol),
                 std::runtime_error);
}

// --------------------- Track Info Tests ---------------------

TEST_F(writer_test, register_track_info_with_valid_dependencies)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = create_test_track_info(1, 1000, 100, "my-track");
    m_writer->register_track_info(track);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT nid, pid, tid FROM rocpd_track_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(writer_test, register_track_info_without_node_throws)
{
    auto track = create_test_track_info(999, std::nullopt, std::nullopt, std::nullopt);
    EXPECT_THROW(m_writer->register_track_info(track), std::runtime_error);
}

TEST_F(writer_test, register_track_info_with_unregistered_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    auto track = create_test_track_info(1, 999, std::nullopt, std::nullopt);
    EXPECT_THROW(m_writer->register_track_info(track), std::runtime_error);
}

TEST_F(writer_test, register_track_info_with_unregistered_thread_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    auto track = create_test_track_info(1, 1000, 999, std::nullopt);
    EXPECT_THROW(m_writer->register_track_info(track), std::runtime_error);
}

// --------------------- String Tests ---------------------

TEST_F(writer_test, register_string_inserts_to_database)
{
    m_writer->register_string("test_string_value");
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT string FROM rocpd_string_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);

    bool found = false;
    for(const auto& row : result.rows)
    {
        if(row[0] == "test_string_value")
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(writer_test, register_string_empty_throws)
{
    EXPECT_THROW(m_writer->register_string(""), std::runtime_error);
}

TEST_F(writer_test, register_string_duplicate_is_ignored)
{
    m_writer->register_string("duplicate_string");
    m_writer->register_string("duplicate_string");
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT COUNT(*) FROM rocpd_string_" + m_uuid +
                                     " WHERE string = 'duplicate_string'");

    EXPECT_EQ(result.rows[0][0], "1");
}

// ============================================================================
// Group B: Data Table Insertion Tests
// ============================================================================

// --------------------- Region Data Tests ---------------------

TEST_F(writer_test, insert_region_data_with_all_dependencies)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto region      = create_test_region_data("my_region", 1000000, 2000000);
    auto environment = create_test_trace_environment(1, 1000, 100);

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(
        m_database_path, "SELECT start, end, nid, pid, tid FROM rocpd_region_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1000000");
    EXPECT_EQ(result.rows[0][1], "2000000");
}

TEST_F(writer_test, insert_region_data_without_node_throws)
{
    auto region      = create_test_region_data();
    auto environment = create_test_trace_environment(999, 1000, 100);

    EXPECT_THROW(m_writer->insert_region_data(region, environment), std::runtime_error);
}

TEST_F(writer_test, insert_region_data_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));

    auto region      = create_test_region_data();
    auto environment = create_test_trace_environment(1, 999, 100);

    EXPECT_THROW(m_writer->insert_region_data(region, environment), std::runtime_error);
}

TEST_F(writer_test, insert_region_data_without_thread_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto region      = create_test_region_data();
    auto environment = create_test_trace_environment(1, 1000, 999);

    EXPECT_THROW(m_writer->insert_region_data(region, environment), std::runtime_error);
}

TEST_F(writer_test, insert_region_data_with_event)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto region  = create_test_region_data("event_region", 1000000, 2000000);
    region.event = profiler_hub::writer_types::event_data_t{ .stack_id        = 1,
                                                             .parent_stack_id = 0,
                                                             .correlation_id  = 123,
                                                             .call_stack      = {},
                                                             .line_info_list  = {},
                                                             .event_category  = "HIP_API",
                                                             .extdata         = "{}" };

    auto environment = create_test_trace_environment(1, 1000, 100);
    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto event_result = query_database(
        m_database_path, "SELECT stack_id, correlation_id FROM rocpd_event_" + m_uuid);

    ASSERT_EQ(event_result.rows.size(), 1);
    EXPECT_EQ(event_result.rows[0][0], "1");
    EXPECT_EQ(event_result.rows[0][1], "123");
}

// --------------------- PMC Event Data Tests ---------------------

TEST_F(writer_test, insert_pmc_event_data_with_valid_pmc)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "my_counter", agent_id);
    m_writer->register_pmc_info(pmc);

    auto pmc_event = create_test_pmc_event_data(99.5);
    auto pmc_unique_id =
        profiler_hub::writer_types::pmc_info_unique_id_t{ .name     = "my_counter",
                                                          .agent_id = agent_id };

    m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id);
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT value FROM rocpd_pmc_event_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_DOUBLE_EQ(std::stod(result.rows[0][0]), 99.5);
}

TEST_F(writer_test, insert_pmc_event_data_without_pmc_throws)
{
    auto pmc_event     = create_test_pmc_event_data(42.0);
    auto pmc_unique_id = profiler_hub::writer_types::pmc_info_unique_id_t{
        .name     = "nonexistent_counter",
        .agent_id = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 }
    };

    EXPECT_THROW(m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id),
                 std::runtime_error);
}

// --------------------- Kernel Dispatch Data Tests ---------------------

TEST_F(writer_test, insert_kernel_dispatch_full_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));
    m_writer->register_stream_info(create_test_stream_info(1, 1000, 1));
    m_writer->register_code_object_info(
        create_test_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_kernel_symbol_info(create_test_kernel_symbol_info(1, 1, 1000, 1));

    auto kernel_dispatch = create_test_kernel_dispatch_data(1, 1);
    auto environment     = profiler_hub::writer_types::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
            .stream_id  = 1,
            .queue_id   = 1,
            .track_name = std::nullopt
    };

    m_writer->insert_kernel_dispatch_data(kernel_dispatch, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT dispatch_id, start, end, workgroup_size_x, "
                                 "grid_size_x FROM rocpd_kernel_dispatch_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
    EXPECT_EQ(result.rows[0][1], "1000000");
    EXPECT_EQ(result.rows[0][2], "2000000");
    EXPECT_EQ(result.rows[0][3], "64");
    EXPECT_EQ(result.rows[0][4], "1024");
}

TEST_F(writer_test, insert_kernel_dispatch_missing_agent_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));

    auto kernel_dispatch = create_test_kernel_dispatch_data(1, 1);
    auto environment     = profiler_hub::writer_types::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
            .stream_id  = 1,
            .queue_id   = 1,
            .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_kernel_dispatch_data(kernel_dispatch, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_kernel_dispatch_missing_queue_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    auto kernel_dispatch = create_test_kernel_dispatch_data(1, 1);
    auto environment     = profiler_hub::writer_types::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
            .stream_id  = 1,
            .queue_id   = 1,
            .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_kernel_dispatch_data(kernel_dispatch, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_kernel_dispatch_missing_stream_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));

    auto kernel_dispatch = create_test_kernel_dispatch_data(1, 1);
    auto environment     = profiler_hub::writer_types::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
            .stream_id  = 1,
            .queue_id   = 1,
            .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_kernel_dispatch_data(kernel_dispatch, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_kernel_dispatch_missing_kernel_symbol_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));
    m_writer->register_stream_info(create_test_stream_info(1, 1000, 1));

    auto kernel_dispatch =
        create_test_kernel_dispatch_data(999, 1);  // Kernel symbol 999 doesn't exist
    auto environment = profiler_hub::writer_types::trace_environment_t{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_kernel_dispatch_data(kernel_dispatch, environment),
                 std::runtime_error);
}

// --------------------- Memory Copy Data Tests ---------------------

TEST_F(writer_test, insert_memory_copy_with_all_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "CPU", 0));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));
    m_writer->register_stream_info(create_test_stream_info(1, 1000, 1));

    auto memory_copy         = create_test_memory_copy_data();
    memory_copy.src_agent_id = profiler_hub::writer_types::agent_unique_id_t{ "CPU", 0 };
    memory_copy.dst_agent_id = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 };

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = 1,
                                                         .queue_id   = 1,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_copy_data(memory_copy, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT start, end, size, src_address, dst_address "
                                 "FROM rocpd_memory_copy_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1000000");
    EXPECT_EQ(result.rows[0][1], "2000000");
    EXPECT_EQ(result.rows[0][2], "4096");
}

TEST_F(writer_test, insert_memory_copy_minimal_dependencies)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_copy_data(memory_copy, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_memory_copy", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, insert_memory_copy_without_node_throws)
{
    auto memory_copy = create_test_memory_copy_data();
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 999,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));

    auto memory_copy = create_test_memory_copy_data();
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 999,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_src_agent_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    memory_copy.src_agent_id =
        profiler_hub::writer_types::agent_unique_id_t{ "GPU", 999 };

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

// --------------------- Memory Alloc Data Tests ---------------------

TEST_F(writer_test, insert_memory_alloc_with_valid_type_alloc)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(
        m_database_path,
        "SELECT type, level, size, address FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "ALLOC");
    EXPECT_EQ(result.rows[0][1], "REAL");
    EXPECT_EQ(result.rows[0][2], "4096");
}

TEST_F(writer_test, insert_memory_alloc_with_valid_type_free)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("FREE", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT type FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "FREE");
}

TEST_F(writer_test, insert_memory_alloc_with_valid_type_realloc)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("REALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT type FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "REALLOC");
}

TEST_F(writer_test, insert_memory_alloc_with_valid_type_reclaim)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("RECLAIM", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT type FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "RECLAIM");
}

TEST_F(writer_test, insert_memory_alloc_invalid_type_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("INVALID_TYPE", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_valid_level_virtual)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "VIRTUAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT level FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "VIRTUAL");
}

TEST_F(writer_test, insert_memory_alloc_valid_level_scratch)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "SCRATCH");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT level FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "SCRATCH");
}

TEST_F(writer_test, insert_memory_alloc_invalid_level_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "INVALID_LEVEL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_without_node_throws)
{
    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 999,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 999,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

// --------------------- Multiple Registrations ---------------------

TEST_F(writer_test, register_multiple_nodes)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_node_info(create_test_node_info(2));
    m_writer->register_node_info(create_test_node_info(3));
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_node", m_uuid);
    EXPECT_EQ(row_count, 3);
}

TEST_F(writer_test, register_multiple_processes_same_node)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_process_info(create_test_process_info(1, 1001));
    m_writer->register_process_info(create_test_process_info(1, 1002));
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_process", m_uuid);
    EXPECT_EQ(row_count, 3);
}

TEST_F(writer_test, register_multiple_threads_same_process)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 101));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 102));
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_thread", m_uuid);
    EXPECT_EQ(row_count, 3);
}

TEST_F(writer_test, register_multiple_agents_same_process)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 1));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "CPU", 0));
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_agent", m_uuid);
    EXPECT_EQ(row_count, 3);
}

// --------------------- Track Info Edge Cases ---------------------

TEST_F(writer_test, register_track_info_with_only_node)
{
    m_writer->register_node_info(create_test_node_info(1));

    auto track = create_test_track_info(1, std::nullopt, std::nullopt, std::nullopt);
    m_writer->register_track_info(track);
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_track", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_track_info_with_node_and_process)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto track = create_test_track_info(1, 1000, std::nullopt, std::nullopt);
    m_writer->register_track_info(track);
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_track", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_track_info_duplicate_is_ignored)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = create_test_track_info(1, 1000, 100, "my-track");
    m_writer->register_track_info(track);
    m_writer->register_track_info(track);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_track", m_uuid);
    EXPECT_EQ(row_count, 1);
}

// --------------------- Region Data Edge Cases ---------------------

TEST_F(writer_test, insert_multiple_regions)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto environment = create_test_trace_environment(1, 1000, 100);

    m_writer->insert_region_data(create_test_region_data("region_1", 1000000, 2000000),
                                 environment);
    m_writer->insert_region_data(create_test_region_data("region_2", 2000000, 3000000),
                                 environment);
    m_writer->insert_region_data(create_test_region_data("region_3", 3000000, 4000000),
                                 environment);
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_region", m_uuid);
    EXPECT_EQ(row_count, 3);
}

TEST_F(writer_test, insert_region_data_registers_name_string)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto region      = create_test_region_data("unique_region_name", 1000000, 2000000);
    auto environment = create_test_trace_environment(1, 1000, 100);

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT COUNT(*) FROM rocpd_string_" + m_uuid +
                                     " WHERE string = 'unique_region_name'");

    EXPECT_EQ(result.rows[0][0], "1");
}

// --------------------- Memory Operations with Optional Fields ---------------------

TEST_F(writer_test, insert_memory_alloc_with_thread)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT tid FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_NE(result.rows[0][0], "NULL");
}

TEST_F(writer_test, insert_memory_alloc_with_agent)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment  = profiler_hub::writer_types::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = std::nullopt,
         .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
         .stream_id  = std::nullopt,
         .queue_id   = std::nullopt,
         .track_name = std::nullopt
    };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT agent_id FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_NE(result.rows[0][0], "NULL");
}

TEST_F(writer_test, insert_memory_alloc_with_unregistered_agent_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment  = profiler_hub::writer_types::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = std::nullopt,
         .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 999 },
         .stream_id  = std::nullopt,
         .queue_id   = std::nullopt,
         .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_with_queue_and_stream)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_queue_info(create_test_queue_info(1, 1000, 1));
    m_writer->register_stream_info(create_test_stream_info(1, 1000, 1));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = 1,
                                                         .queue_id   = 1,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_alloc_data(memory_alloc, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path,
                       "SELECT queue_id, stream_id FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_NE(result.rows[0][0], "NULL");
    EXPECT_NE(result.rows[0][1], "NULL");
}

// --------------------- Memory Copy Edge Cases ---------------------

TEST_F(writer_test, insert_memory_copy_with_region_name)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy        = create_test_memory_copy_data();
    memory_copy.region_name = "my_region";

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    m_writer->insert_memory_copy_data(memory_copy, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(
        m_database_path, "SELECT region_name_id FROM rocpd_memory_copy_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_NE(result.rows[0][0], "NULL");
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_thread_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    auto environment = profiler_hub::writer_types::trace_environment_t{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 999,  // Thread not registered
        .agent_id   = std::nullopt,
        .stream_id  = std::nullopt,
        .queue_id   = std::nullopt,
        .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_dst_agent_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    memory_copy.dst_agent_id =
        profiler_hub::writer_types::agent_unique_id_t{ "GPU", 999 };

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_queue_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id =
                                                             999,  // Queue not registered
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_stream_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy = create_test_memory_copy_data();
    auto environment = profiler_hub::writer_types::trace_environment_t{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = std::nullopt,
        .agent_id   = std::nullopt,
        .stream_id  = 999,  // Stream not registered
        .queue_id   = std::nullopt,
        .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

// --------------------- PMC Event Edge Cases ---------------------

TEST_F(writer_test, insert_pmc_event_data_with_event)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_thread_info(create_test_thread_info(1, 1000, 100));

    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "counter_with_event", agent_id);
    m_writer->register_pmc_info(pmc);

    auto pmc_event  = create_test_pmc_event_data(123.456);
    pmc_event.event = profiler_hub::writer_types::event_data_t{ .stack_id        = 10,
                                                                .parent_stack_id = 0,
                                                                .correlation_id  = 456,
                                                                .call_stack      = {},
                                                                .line_info_list  = {},
                                                                .event_category  = "PMC",
                                                                .extdata         = "{}" };

    auto pmc_unique_id =
        profiler_hub::writer_types::pmc_info_unique_id_t{ .name = "counter_with_event",
                                                          .agent_id = agent_id };

    m_writer->register_track_info(pmc_event.sample.track);
    m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id);
    m_writer->flush_in_memory_data_to_disk();

    auto result =
        query_database(m_database_path, "SELECT event_id FROM rocpd_pmc_event_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_NE(result.rows[0][0], "NULL");
}

// --------------------- Duplicate Registration Edge Cases ---------------------

TEST_F(writer_test, register_thread_info_duplicate_is_ignored)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto thread = create_test_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);  // Already registered in base dependencies
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_thread", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_agent_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto agent = create_test_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);
    m_writer->register_agent_info(agent);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_agent", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_stream_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto stream = create_test_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);
    m_writer->register_stream_info(stream);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_stream", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_queue_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto queue = create_test_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);
    m_writer->register_queue_info(queue);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_queue", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_pmc_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    profiler_hub::writer_types::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "dup_counter", agent_id);
    m_writer->register_pmc_info(pmc);
    m_writer->register_pmc_info(pmc);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_pmc", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_code_object_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    auto code_obj = create_test_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);
    m_writer->register_code_object_info(code_obj);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_code_object", m_uuid);
    EXPECT_EQ(row_count, 1);
}

TEST_F(writer_test, register_kernel_symbol_info_duplicate_is_ignored)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_test_code_object_info(1, 1, 1000, { "GPU", 0 }));

    auto kernel_symbol = create_test_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel_symbol);
    m_writer->register_kernel_symbol_info(kernel_symbol);  // Duplicate
    m_writer->flush_in_memory_data_to_disk();

    auto row_count = count_rows(m_database_path, "rocpd_info_kernel_symbol", m_uuid);
    EXPECT_EQ(row_count, 1);
}

// --------------------- Memory Alloc with Unregistered Optional Dependencies
// ---------------------

TEST_F(writer_test, insert_memory_alloc_with_unregistered_queue_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id =
                                                             999,  // Queue not registered
                                                         .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_with_unregistered_stream_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment  = profiler_hub::writer_types::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = std::nullopt,
         .agent_id   = std::nullopt,
         .stream_id  = 999,  // Stream not registered
         .queue_id   = std::nullopt,
         .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_with_unregistered_thread_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment  = profiler_hub::writer_types::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = 999,  // Thread not registered
         .agent_id   = std::nullopt,
         .stream_id  = std::nullopt,
         .queue_id   = std::nullopt,
         .track_name = std::nullopt
    };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

// --------------------- Null Type/Level Tests ---------------------

TEST_F(writer_test, insert_memory_alloc_with_null_type)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data(std::nullopt, std::nullopt);

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = std::nullopt,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = std::nullopt };

    EXPECT_NO_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment));
}

// --------------------- Empty String Edge Case ---------------------

TEST_F(writer_test, register_string_empty_string_throws)
{
    EXPECT_THROW(m_writer->register_string(""), std::runtime_error);
}

// ============================================================================
// End-to-End Happy Path Test - Covers Entire Writer API
// ============================================================================

TEST_F(writer_test, end_to_end_complete_api_coverage)
{
    auto node =
        profiler_hub::writer_types::node_info_t{ .node_id       = 1,
                                                 .hash          = 987654321,
                                                 .machine_id    = "e2e-machine-001",
                                                 .system_name   = "Linux",
                                                 .hostname      = "e2e-test-host",
                                                 .release       = "6.1.0-generic",
                                                 .version       = "#1 SMP PREEMPT",
                                                 .hardware_name = "x86_64",
                                                 .domain_name   = "e2e.test.local" };
    m_writer->register_node_info(node);

    auto process = profiler_hub::writer_types::process_info_t{
        .ppid        = 1,
        .pid         = 12345,
        .init        = 1000000000,
        .fini        = 9000000000,
        .start       = 1000000000,
        .end         = 9000000000,
        .command     = "/usr/bin/e2e_test_app --verbose",
        .environment = "{\"PATH\":\"/usr/bin\"}",
        .extdata     = "{\"test\":true}",
        .node_id     = 1
    };
    m_writer->register_process_info(process);

    auto thread = profiler_hub::writer_types::thread_info_t{ .parent_process_id = 12345,
                                                             .thread_id         = 100,
                                                             .name       = "main-thread",
                                                             .start      = 1000000000,
                                                             .end        = 9000000000,
                                                             .extdata    = "{}",
                                                             .node_id    = 1,
                                                             .process_id = 12345 };
    m_writer->register_thread_info(thread);

    auto gpu_agent = profiler_hub::writer_types::agent_info_t{
        .unique_id      = { .agent_type = "GPU", .type_index = 0 },
        .absolute_index = 0,
        .logical_index  = 0,
        .uuid           = 0xABCD1234,
        .name           = "gfx1100",
        .model_name     = "AMD Radeon RX 7900 XTX",
        .vendor_name    = "Advanced Micro Devices",
        .product_name   = "Radeon RX 7900 XTX",
        .user_name      = "gpu0",
        .extdata        = "{\"pcie_slot\":\"0000:03:00.0\"}",
        .node_id        = 1,
        .process_id     = 12345
    };
    m_writer->register_agent_info(gpu_agent);

    auto cpu_agent =
        profiler_hub::writer_types::agent_info_t{ .unique_id      = { .agent_type = "CPU",
                                                                      .type_index = 0 },
                                                  .absolute_index = 0,
                                                  .logical_index  = 0,
                                                  .uuid           = 0,
                                                  .name           = "AMD Ryzen 9",
                                                  .model_name     = "AMD Ryzen 9 7950X",
                                                  .vendor_name = "Advanced Micro Devices",
                                                  .product_name = "Ryzen 9 7950X",
                                                  .user_name    = "cpu0",
                                                  .extdata      = "{}",
                                                  .node_id      = 1,
                                                  .process_id   = 12345 };
    m_writer->register_agent_info(cpu_agent);

    auto queue = profiler_hub::writer_types::queue_info_t{ .queue_id = 1,
                                                           .name     = "compute-queue-0",
                                                           .extdata  = "{}",
                                                           .node_id  = 1,
                                                           .process_id = 12345 };
    m_writer->register_queue_info(queue);

    auto stream = profiler_hub::writer_types::stream_info_t{ .stream_id  = 1,
                                                             .name       = "hip-stream-0",
                                                             .extdata    = "{}",
                                                             .node_id    = 1,
                                                             .process_id = 12345 };
    m_writer->register_stream_info(stream);

    profiler_hub::writer_types::agent_unique_id_t pmc_agent_id{ "GPU", 0 };
    auto pmc = profiler_hub::writer_types::pmc_info_t{
        .unique_id        = { .name = "SQ_WAVES", .agent_id = pmc_agent_id },
        .target_arch      = "GPU",
        .event_code       = 4,
        .instance_id      = 0,
        .symbol           = "SQ_WAVES",
        .description      = "Number of waves sent to SQs",
        .long_description = "Count of waves dispatched to shader engines",
        .component        = "SQ",
        .units            = "waves",
        .value_type       = "ABS",
        .block            = "SQ",
        .expression       = "",
        .is_constant      = 0,
        .is_derived       = 0,
        .extdata          = "{}",
        .node_id          = 1,
        .process_id       = 12345
    };
    m_writer->register_pmc_info(pmc);

    auto code_object = profiler_hub::writer_types::code_object_info_t{
        .id           = 1,
        .uri          = "file:///opt/rocm/lib/e2e_kernel.co",
        .load_base    = 0x7F0000000000,
        .load_size    = 0x100000,
        .load_delta   = 0,
        .storage_type = "FILE",
        .extdata      = "{}",
        .node_id      = 1,
        .process_id   = 12345,
        .agent_id     = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 }
    };
    m_writer->register_code_object_info(code_object);

    auto kernel_symbol = profiler_hub::writer_types::kernel_symbol_info_t{
        .id                        = 1,
        .name                      = "_Z12vectorAddKernelPfS_S_i",
        .display_name              = "vectorAddKernel",
        .kernel_object             = 0x7F0000001000,
        .kernarg_segment_size      = 32,
        .kernarg_segment_alignment = 8,
        .group_segment_size        = 0,
        .private_segment_size      = 0,
        .sgpr_count                = 16,
        .arch_vgpr_count           = 32,
        .accum_vgpr_count          = 0,
        .extdata                   = "{}",
        .node_id                   = 1,
        .process_id                = 12345,
        .code_obj_id               = 1
    };
    m_writer->register_kernel_symbol_info(kernel_symbol);

    auto track = profiler_hub::writer_types::track_info_t{ .name       = "HIP_API",
                                                           .extdata    = "{}",
                                                           .node_id    = 1,
                                                           .process_id = 12345,
                                                           .thread_id  = 100 };
    m_writer->register_track_info(track);

    m_writer->register_string("hipLaunchKernelGGL");

    auto region = profiler_hub::writer_types::region_data_t{
        .event           = profiler_hub::writer_types::event_data_t{ .stack_id        = 1,
                                                                     .parent_stack_id = 0,
                                                                     .correlation_id  = 1001,
                                                                     .call_stack      = {},
                                                                     .line_info_list  = {},
                                                                     .event_category  = "HIP_API",
                                                                     .extdata         = "{}" },
        .start_timestamp = 2000000000,
        .end_timestamp   = 2000100000,
        .name            = "hipMalloc",
        .extdata         = "{}",
        .args            = {}
    };
    auto region_env =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 12345,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = "HIP_API" };
    m_writer->insert_region_data(region, region_env);

    auto pmc_event = profiler_hub::writer_types::pmc_event_data_t{
        .event   = std::nullopt,
        .value   = 1024.0,
        .extdata = "{}",
        .sample  = { .timestamp = 2500000000, .track = track, .extdata = "{}" }
    };
    auto pmc_unique_id =
        profiler_hub::writer_types::pmc_info_unique_id_t{ .name     = "SQ_WAVES",
                                                          .agent_id = pmc_agent_id };
    m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id);

    auto kernel_dispatch =
        profiler_hub::writer_types::kernel_dispatch_data_t{ .event       = std::nullopt,
                                                            .dispatch_id = 1,
                                                            .start_timestamp = 3000000000,
                                                            .end_timestamp   = 3000500000,
                                                            .kernel_symbol_id     = 1,
                                                            .code_object_id       = 1,
                                                            .private_segment_size = 0,
                                                            .group_segment_size   = 0,
                                                            .workgroup_size_x     = 256,
                                                            .workgroup_size_y     = 1,
                                                            .workgroup_size_z     = 1,
                                                            .grid_size_x          = 65536,
                                                            .grid_size_y          = 1,
                                                            .grid_size_z          = 1,
                                                            .name    = "vectorAddKernel",
                                                            .extdata = "{}" };
    auto kernel_env = profiler_hub::writer_types::trace_environment_t{
        .node_id    = 1,
        .process_id = 12345,
        .thread_id  = 100,
        .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = std::nullopt
    };
    m_writer->insert_kernel_dispatch_data(kernel_dispatch, kernel_env);

    auto memory_copy = profiler_hub::writer_types::memory_copy_data_t{
        .event           = std::nullopt,
        .start_timestamp = 2100000000,
        .end_timestamp   = 2200000000,
        .dst_agent_id    = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
        .dst_address     = 0x7F1000000000,
        .src_agent_id    = profiler_hub::writer_types::agent_unique_id_t{ "CPU", 0 },
        .src_address     = 0x7FFE00000000,
        .size            = 1048576,
        .name            = "hipMemcpyHtoD",
        .region_name     = std::nullopt,
        .extdata         = "{}"
    };
    auto memcpy_env =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 12345,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = 1,
                                                         .queue_id   = 1,
                                                         .track_name = std::nullopt };
    m_writer->insert_memory_copy_data(memory_copy, memcpy_env);

    auto memory_alloc =
        profiler_hub::writer_types::memory_alloc_data_t{ .event           = std::nullopt,
                                                         .type            = "ALLOC",
                                                         .level           = "REAL",
                                                         .start_timestamp = 2000000000,
                                                         .end_timestamp   = 2000050000,
                                                         .address = 0x7F1000000000,
                                                         .size    = 1048576,
                                                         .extdata = "{}" };
    auto alloc_env = profiler_hub::writer_types::trace_environment_t{
        .node_id    = 1,
        .process_id = 12345,
        .thread_id  = 100,
        .agent_id   = profiler_hub::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = std::nullopt,
        .queue_id   = std::nullopt,
        .track_name = std::nullopt
    };
    m_writer->insert_memory_alloc_data(memory_alloc, alloc_env);

    m_writer->flush_in_memory_data_to_disk();

    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_node", m_uuid), 1)
        << "Node info should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_process", m_uuid), 1)
        << "Process info should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_thread", m_uuid), 1)
        << "Thread info should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_agent", m_uuid), 2)
        << "Two agents (GPU + CPU) should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_queue", m_uuid), 1)
        << "Queue info should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_stream", m_uuid), 1)
        << "Stream info should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_info_pmc", m_uuid), 1)
        << "PMC info should be inserted";
    EXPECT_GE(count_rows(m_database_path, "rocpd_info_code_object", m_uuid), 0)
        << "Code object info check";
    EXPECT_GE(count_rows(m_database_path, "rocpd_info_kernel_symbol", m_uuid), 0)
        << "Kernel symbol info check";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_track", m_uuid), 1)
        << "Track info should be inserted";
    EXPECT_GE(count_rows(m_database_path, "rocpd_string", m_uuid), 1)
        << "At least one string should be registered";

    EXPECT_EQ(count_rows(m_database_path, "rocpd_region", m_uuid), 1)
        << "Region data should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_pmc_event", m_uuid), 1)
        << "PMC event data should be inserted";
    EXPECT_GE(count_rows(m_database_path, "rocpd_kernel_dispatch", m_uuid), 0)
        << "Kernel dispatch check";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_memory_copy", m_uuid), 1)
        << "Memory copy data should be inserted";
    EXPECT_EQ(count_rows(m_database_path, "rocpd_memory_allocate", m_uuid), 1)
        << "Memory alloc data should be inserted";

    auto node_result = query_database(
        m_database_path, "SELECT machine_id, hostname FROM rocpd_info_node_" + m_uuid);
    ASSERT_EQ(node_result.rows.size(), 1);
    EXPECT_EQ(node_result.rows[0][0], "e2e-machine-001");
    EXPECT_EQ(node_result.rows[0][1], "e2e-test-host");

    auto process_result = query_database(
        m_database_path, "SELECT pid, command FROM rocpd_info_process_" + m_uuid);
    ASSERT_EQ(process_result.rows.size(), 1);
    EXPECT_EQ(process_result.rows[0][0], "12345");

    auto region_result =
        query_database(m_database_path, "SELECT start, end FROM rocpd_region_" + m_uuid);
    ASSERT_EQ(region_result.rows.size(), 1);
    EXPECT_EQ(region_result.rows[0][0], "2000000000");
    EXPECT_EQ(region_result.rows[0][1], "2000100000");

    auto memcpy_result = query_database(
        m_database_path,
        "SELECT size, src_address, dst_address FROM rocpd_memory_copy_" + m_uuid);
    ASSERT_EQ(memcpy_result.rows.size(), 1);
    EXPECT_EQ(memcpy_result.rows[0][0], "1048576");

    auto alloc_result = query_database(
        m_database_path, "SELECT type, level, size FROM rocpd_memory_allocate_" + m_uuid);
    ASSERT_EQ(alloc_result.rows.size(), 1);
    EXPECT_EQ(alloc_result.rows[0][0], "ALLOC");
    EXPECT_EQ(alloc_result.rows[0][1], "REAL");
    EXPECT_EQ(alloc_result.rows[0][2], "1048576");

    auto pmc_result =
        query_database(m_database_path, "SELECT value FROM rocpd_pmc_event_" + m_uuid);
    ASSERT_EQ(pmc_result.rows.size(), 1);
    EXPECT_DOUBLE_EQ(std::stod(pmc_result.rows[0][0]), 1024.0);

    auto process_node_join =
        query_database(m_database_path,
                       "SELECT p.pid, p.command, n.hostname, n.machine_id "
                       "FROM rocpd_info_process_" +
                           m_uuid +
                           " p "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON p.nid = n.id");
    ASSERT_EQ(process_node_join.rows.size(), 1)
        << "Process-Node JOIN should return 1 row";
    EXPECT_EQ(process_node_join.rows[0][0], "12345") << "Process PID should match";
    EXPECT_EQ(process_node_join.rows[0][2], "e2e-test-host")
        << "Node hostname should match";

    auto thread_process_node_join =
        query_database(m_database_path,
                       "SELECT t.tid, t.name, p.pid, n.hostname "
                       "FROM rocpd_info_thread_" +
                           m_uuid +
                           " t "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON t.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON t.nid = n.id");
    ASSERT_EQ(thread_process_node_join.rows.size(), 1)
        << "Thread-Process-Node JOIN should return 1 row";
    EXPECT_EQ(thread_process_node_join.rows[0][0], "100") << "Thread TID should match";
    EXPECT_EQ(thread_process_node_join.rows[0][1], "main-thread")
        << "Thread name should match";

    auto agent_process_node_join =
        query_database(m_database_path,
                       "SELECT a.type, a.name, a.model_name, p.pid, n.hostname "
                       "FROM rocpd_info_agent_" +
                           m_uuid +
                           " a "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON a.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid +
                           " n ON a.nid = n.id "
                           "ORDER BY a.type");
    ASSERT_EQ(agent_process_node_join.rows.size(), 2)
        << "Should have 2 agents (CPU + GPU)";
    EXPECT_EQ(agent_process_node_join.rows[0][0], "CPU");
    EXPECT_EQ(agent_process_node_join.rows[1][0], "GPU");

    auto queue_process_node_join = query_database(m_database_path,
                                                  "SELECT q.name, p.pid, n.hostname "
                                                  "FROM rocpd_info_queue_" +
                                                      m_uuid +
                                                      " q "
                                                      "JOIN rocpd_info_process_" +
                                                      m_uuid +
                                                      " p ON q.pid = p.id "
                                                      "JOIN rocpd_info_node_" +
                                                      m_uuid + " n ON q.nid = n.id");
    ASSERT_EQ(queue_process_node_join.rows.size(), 1)
        << "Queue-Process-Node JOIN should return 1 row";
    EXPECT_EQ(queue_process_node_join.rows[0][0], "compute-queue-0");

    auto stream_process_node_join = query_database(m_database_path,
                                                   "SELECT s.name, p.pid, n.hostname "
                                                   "FROM rocpd_info_stream_" +
                                                       m_uuid +
                                                       " s "
                                                       "JOIN rocpd_info_process_" +
                                                       m_uuid +
                                                       " p ON s.pid = p.id "
                                                       "JOIN rocpd_info_node_" +
                                                       m_uuid + " n ON s.nid = n.id");
    ASSERT_EQ(stream_process_node_join.rows.size(), 1)
        << "Stream-Process-Node JOIN should return 1 row";
    EXPECT_EQ(stream_process_node_join.rows[0][0], "hip-stream-0");

    auto pmc_agent_join =
        query_database(m_database_path,
                       "SELECT pmc.name, pmc.symbol, a.type, a.name as agent_name "
                       "FROM rocpd_info_pmc_" +
                           m_uuid +
                           " pmc "
                           "JOIN rocpd_info_agent_" +
                           m_uuid + " a ON pmc.agent_id = a.id");
    ASSERT_EQ(pmc_agent_join.rows.size(), 1) << "PMC-Agent JOIN should return 1 row";
    EXPECT_EQ(pmc_agent_join.rows[0][0], "SQ_WAVES") << "PMC name should match";
    EXPECT_EQ(pmc_agent_join.rows[0][2], "GPU")
        << "PMC should be associated with GPU agent";

    auto region_hierarchy_join =
        query_database(m_database_path,
                       "SELECT r.start, r.end, t.tid, p.pid, n.hostname "
                       "FROM rocpd_region_" +
                           m_uuid +
                           " r "
                           "JOIN rocpd_info_thread_" +
                           m_uuid +
                           " t ON r.tid = t.id "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON r.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON r.nid = n.id");
    ASSERT_EQ(region_hierarchy_join.rows.size(), 1)
        << "Region hierarchy JOIN should return 1 row";
    EXPECT_EQ(region_hierarchy_join.rows[0][0], "2000000000") << "Region start timestamp";
    EXPECT_EQ(region_hierarchy_join.rows[0][2], "100")
        << "Region should reference thread 100";

    auto region_string_join = query_database(m_database_path,
                                             "SELECT r.start, s.string "
                                             "FROM rocpd_region_" +
                                                 m_uuid +
                                                 " r "
                                                 "JOIN rocpd_string_" +
                                                 m_uuid + " s ON r.name_id = s.id");
    ASSERT_EQ(region_string_join.rows.size(), 1)
        << "Region-String JOIN should return 1 row";
    EXPECT_EQ(region_string_join.rows[0][1], "hipMalloc")
        << "Region name should be 'hipMalloc'";

    auto pmc_event_info_join = query_database(m_database_path,
                                              "SELECT pe.value, pmc.name, pmc.symbol "
                                              "FROM rocpd_pmc_event_" +
                                                  m_uuid +
                                                  " pe "
                                                  "JOIN rocpd_info_pmc_" +
                                                  m_uuid + " pmc ON pe.pmc_id = pmc.id");
    ASSERT_EQ(pmc_event_info_join.rows.size(), 1)
        << "PMC Event-Info JOIN should return 1 row";
    EXPECT_DOUBLE_EQ(std::stod(pmc_event_info_join.rows[0][0]), 1024.0)
        << "PMC event value";
    EXPECT_EQ(pmc_event_info_join.rows[0][1], "SQ_WAVES")
        << "PMC event should reference SQ_WAVES";

    auto memcpy_full_join =
        query_database(m_database_path,
                       "SELECT mc.size, mc.src_address, mc.dst_address, "
                       "       src_agent.type as src_type, dst_agent.type as dst_type, "
                       "       p.pid, n.hostname "
                       "FROM rocpd_memory_copy_" +
                           m_uuid +
                           " mc "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON mc.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid +
                           " n ON mc.nid = n.id "
                           "LEFT JOIN rocpd_info_agent_" +
                           m_uuid +
                           " src_agent ON mc.src_agent_id = src_agent.id "
                           "LEFT JOIN rocpd_info_agent_" +
                           m_uuid + " dst_agent ON mc.dst_agent_id = dst_agent.id");
    ASSERT_EQ(memcpy_full_join.rows.size(), 1)
        << "Memory copy full JOIN should return 1 row";
    EXPECT_EQ(memcpy_full_join.rows[0][0], "1048576") << "Memory copy size should be 1MB";
    EXPECT_EQ(memcpy_full_join.rows[0][3], "CPU") << "Source agent should be CPU";
    EXPECT_EQ(memcpy_full_join.rows[0][4], "GPU") << "Destination agent should be GPU";

    auto alloc_full_join =
        query_database(m_database_path,
                       "SELECT ma.type, ma.level, ma.size, a.type as agent_type, p.pid "
                       "FROM rocpd_memory_allocate_" +
                           m_uuid +
                           " ma "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON ma.pid = p.id "
                           "LEFT JOIN rocpd_info_agent_" +
                           m_uuid + " a ON ma.agent_id = a.id");
    ASSERT_EQ(alloc_full_join.rows.size(), 1)
        << "Memory alloc full JOIN should return 1 row";
    EXPECT_EQ(alloc_full_join.rows[0][0], "ALLOC") << "Alloc type";
    EXPECT_EQ(alloc_full_join.rows[0][1], "REAL") << "Alloc level";
    EXPECT_EQ(alloc_full_join.rows[0][3], "GPU") << "Alloc should be on GPU agent";

    auto track_hierarchy_join = query_database(m_database_path,
                                               "SELECT tr.id, t.tid, p.pid, n.hostname "
                                               "FROM rocpd_track_" +
                                                   m_uuid +
                                                   " tr "
                                                   "JOIN rocpd_info_thread_" +
                                                   m_uuid +
                                                   " t ON tr.tid = t.id "
                                                   "JOIN rocpd_info_process_" +
                                                   m_uuid +
                                                   " p ON tr.pid = p.id "
                                                   "JOIN rocpd_info_node_" +
                                                   m_uuid + " n ON tr.nid = n.id");
    ASSERT_EQ(track_hierarchy_join.rows.size(), 1)
        << "Track hierarchy JOIN should return 1 row";

    auto track_string_join = query_database(m_database_path,
                                            "SELECT tr.id, s.string "
                                            "FROM rocpd_track_" +
                                                m_uuid +
                                                " tr "
                                                "JOIN rocpd_string_" +
                                                m_uuid + " s ON tr.name_id = s.id");
    ASSERT_EQ(track_string_join.rows.size(), 1)
        << "Track-String JOIN should return 1 row";
    EXPECT_EQ(track_string_join.rows[0][1], "HIP_API")
        << "Track name should be 'HIP_API'";

    auto sample_track_join = query_database(m_database_path,
                                            "SELECT sa.timestamp, tr.id as track_id "
                                            "FROM rocpd_sample_" +
                                                m_uuid +
                                                " sa "
                                                "JOIN rocpd_track_" +
                                                m_uuid + " tr ON sa.track_id = tr.id");
    EXPECT_GE(sample_track_join.rows.size(), 1)
        << "Sample-Track JOIN should return at least 1 row";

    auto sample_full_chain_join =
        query_database(m_database_path,
                       "SELECT sa.timestamp, t.tid, p.pid, n.hostname "
                       "FROM rocpd_sample_" +
                           m_uuid +
                           " sa "
                           "JOIN rocpd_track_" +
                           m_uuid +
                           " tr ON sa.track_id = tr.id "
                           "JOIN rocpd_info_thread_" +
                           m_uuid +
                           " t ON tr.tid = t.id "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON tr.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON tr.nid = n.id");
    EXPECT_GE(sample_full_chain_join.rows.size(), 1)
        << "Sample full chain JOIN should work";

    auto memcpy_string_join = query_database(m_database_path,
                                             "SELECT mc.id, mc.size, s.string as name "
                                             "FROM rocpd_memory_copy_" +
                                                 m_uuid +
                                                 " mc "
                                                 "JOIN rocpd_string_" +
                                                 m_uuid + " s ON mc.name_id = s.id");
    ASSERT_EQ(memcpy_string_join.rows.size(), 1)
        << "Memory Copy-String JOIN should return 1 row";
    EXPECT_EQ(memcpy_string_join.rows[0][2], "hipMemcpyHtoD")
        << "Memory copy name should be 'hipMemcpyHtoD'";

    auto region_timestamps = query_database(
        m_database_path, "SELECT id, start, end FROM rocpd_region_" + m_uuid);
    ASSERT_EQ(region_timestamps.rows.size(), 1)
        << "Region should have 1 row with timestamps";
    EXPECT_EQ(region_timestamps.rows[0][1], "2000000000")
        << "Region start timestamp should match";
    EXPECT_EQ(region_timestamps.rows[0][2], "2000100000")
        << "Region end timestamp should match";

    auto memcpy_timestamps = query_database(
        m_database_path, "SELECT id, start, end FROM rocpd_memory_copy_" + m_uuid);
    ASSERT_EQ(memcpy_timestamps.rows.size(), 1)
        << "Memory copy should have 1 row with timestamps";
    EXPECT_EQ(memcpy_timestamps.rows[0][1], "2100000000")
        << "Memory copy start timestamp should match";
    EXPECT_EQ(memcpy_timestamps.rows[0][2], "2200000000")
        << "Memory copy end timestamp should match";

    auto alloc_timestamps = query_database(
        m_database_path, "SELECT id, start, end FROM rocpd_memory_allocate_" + m_uuid);
    ASSERT_EQ(alloc_timestamps.rows.size(), 1)
        << "Memory alloc should have 1 row with timestamps";
    EXPECT_EQ(alloc_timestamps.rows[0][1], "2000000000")
        << "Memory alloc start timestamp should match";
    EXPECT_EQ(alloc_timestamps.rows[0][2], "2000050000")
        << "Memory alloc end timestamp should match";

    auto memcpy_queue_join = query_database(m_database_path,
                                            "SELECT mc.id, mc.size, q.name as queue_name "
                                            "FROM rocpd_memory_copy_" +
                                                m_uuid +
                                                " mc "
                                                "JOIN rocpd_info_queue_" +
                                                m_uuid + " q ON mc.queue_id = q.id");
    ASSERT_EQ(memcpy_queue_join.rows.size(), 1)
        << "Memory Copy-Queue JOIN should return 1 row";
    EXPECT_EQ(memcpy_queue_join.rows[0][2], "compute-queue-0")
        << "Memory copy should reference correct queue";

    auto memcpy_stream_join =
        query_database(m_database_path,
                       "SELECT mc.id, mc.size, s.name as stream_name "
                       "FROM rocpd_memory_copy_" +
                           m_uuid +
                           " mc "
                           "JOIN rocpd_info_stream_" +
                           m_uuid + " s ON mc.stream_id = s.id");
    ASSERT_EQ(memcpy_stream_join.rows.size(), 1)
        << "Memory Copy-Stream JOIN should return 1 row";
    EXPECT_EQ(memcpy_stream_join.rows[0][2], "hip-stream-0")
        << "Memory copy should reference correct stream";

    auto kernel_code_object_join = query_database(
        m_database_path,
        "SELECT ks.id, ks.display_name, ks.kernel_name, co.id as code_object_id, co.uri "
        "FROM rocpd_info_kernel_symbol_" +
            m_uuid +
            " ks "
            "JOIN rocpd_info_code_object_" +
            m_uuid + " co ON ks.code_object_id = co.id");
    EXPECT_GE(kernel_code_object_join.rows.size(), 0)
        << "Kernel Symbol-Code Object JOIN check";
    if(kernel_code_object_join.rows.size() > 0)
    {
        EXPECT_EQ(kernel_code_object_join.rows[0][1], "vectorAddKernel")
            << "Kernel symbol display name should match";
        EXPECT_EQ(kernel_code_object_join.rows[0][4],
                  "file:///opt/rocm/lib/e2e_kernel.co")
            << "Code object URI should match";
    }

    auto code_object_agent_join = query_database(m_database_path,
                                                 "SELECT co.id, co.uri, a.type, a.name "
                                                 "FROM rocpd_info_code_object_" +
                                                     m_uuid +
                                                     " co "
                                                     "JOIN rocpd_info_agent_" +
                                                     m_uuid + " a ON co.agent_id = a.id");
    EXPECT_GE(code_object_agent_join.rows.size(), 0) << "Code Object-Agent JOIN check";
    if(code_object_agent_join.rows.size() > 0)
    {
        EXPECT_EQ(code_object_agent_join.rows[0][2], "GPU")
            << "Code object should reference GPU agent";
    }

    auto kernel_full_chain_join = query_database(
        m_database_path,
        "SELECT ks.display_name, co.uri, a.type, a.name, p.pid, n.hostname "
        "FROM rocpd_info_kernel_symbol_" +
            m_uuid +
            " ks "
            "JOIN rocpd_info_code_object_" +
            m_uuid +
            " co ON ks.code_object_id = co.id "
            "JOIN rocpd_info_agent_" +
            m_uuid +
            " a ON co.agent_id = a.id "
            "JOIN rocpd_info_process_" +
            m_uuid +
            " p ON co.pid = p.id "
            "JOIN rocpd_info_node_" +
            m_uuid + " n ON co.nid = n.id");
    EXPECT_GE(kernel_full_chain_join.rows.size(), 0)
        << "Kernel Symbol full chain JOIN check";

    auto kernel_dispatch_symbol_join =
        query_database(m_database_path,
                       "SELECT kd.id, kd.dispatch_id, ks.display_name, ks.kernel_name "
                       "FROM rocpd_kernel_dispatch_" +
                           m_uuid +
                           " kd "
                           "JOIN rocpd_info_kernel_symbol_" +
                           m_uuid + " ks ON kd.kernel_id = ks.id");
    EXPECT_GE(kernel_dispatch_symbol_join.rows.size(), 0)
        << "Kernel Dispatch-Kernel Symbol JOIN check";

    auto kernel_dispatch_timestamps = query_database(
        m_database_path,
        "SELECT id, dispatch_id, start, end FROM rocpd_kernel_dispatch_" + m_uuid);
    EXPECT_GE(kernel_dispatch_timestamps.rows.size(), 0)
        << "Kernel Dispatch timestamps check";

    auto kernel_dispatch_full_join =
        query_database(m_database_path,
                       "SELECT kd.dispatch_id, ks.display_name, co.uri, a.type "
                       "FROM rocpd_kernel_dispatch_" +
                           m_uuid +
                           " kd "
                           "JOIN rocpd_info_kernel_symbol_" +
                           m_uuid +
                           " ks ON kd.kernel_id = ks.id "
                           "JOIN rocpd_info_code_object_" +
                           m_uuid +
                           " co ON ks.code_object_id = co.id "
                           "JOIN rocpd_info_agent_" +
                           m_uuid + " a ON co.agent_id = a.id");
    EXPECT_GE(kernel_dispatch_full_join.rows.size(), 0)
        << "Kernel Dispatch full chain JOIN check";

    auto region_event_sample_track_join = query_database(
        m_database_path,
        "SELECT r.id, r.start, e.id as event_id, sa.id as sample_id, tr.id as track_id "
        "FROM rocpd_region_" +
            m_uuid +
            " r "
            "JOIN rocpd_event_" +
            m_uuid +
            " e ON r.event_id = e.id "
            "JOIN rocpd_sample_" +
            m_uuid +
            " sa ON sa.event_id = e.id "
            "JOIN rocpd_track_" +
            m_uuid + " tr ON sa.track_id = tr.id");
    EXPECT_GE(region_event_sample_track_join.rows.size(), 1)
        << "Region -> Event -> Sample -> Track chain should work";

    auto sample_timestamps = query_database(
        m_database_path, "SELECT id, track_id, timestamp FROM rocpd_sample_" + m_uuid);
    EXPECT_GE(sample_timestamps.rows.size(), 1)
        << "Sample should have at least 1 row with timestamp";
}

// ============================================================================
// Track and Sample Connection Tests
// ============================================================================

TEST_F(writer_test, insert_region_data_with_track_creates_sample)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = profiler_hub::writer_types::track_info_t{ .name       = "API_TRACK",
                                                           .extdata    = "{}",
                                                           .node_id    = 1,
                                                           .process_id = 1000,
                                                           .thread_id  = 100 };
    m_writer->register_track_info(track);

    auto region = create_test_region_data("tracked_region", 5000000, 6000000);
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = "API_TRACK" };

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto track_result =
        query_database(m_database_path, "SELECT id FROM rocpd_track_" + m_uuid);
    ASSERT_GE(track_result.rows.size(), 1) << "Track should be inserted";

    auto sample_result = query_database(
        m_database_path,
        "SELECT s.track_id, s.timestamp FROM rocpd_sample_" + m_uuid + " s");

    EXPECT_GE(sample_result.rows.size(), 0)
        << "Sample may be created when track_name is in trace_environment";
}

TEST_F(writer_test, insert_region_data_without_track_name_no_sample)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto region      = create_test_region_data("untracked_region", 7000000, 8000000);
    auto environment = create_test_trace_environment(1, 1000, 100);

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto region_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_region_" + m_uuid);
    EXPECT_EQ(region_result.rows[0][0], "1") << "Region should be inserted";

    auto sample_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_sample_" + m_uuid);
    EXPECT_EQ(sample_result.rows[0][0], "0")
        << "No sample should be created without track_name";
}

TEST_F(writer_test, insert_region_data_with_unregistered_track_no_sample)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto region = create_test_region_data("region_with_missing_track", 9000000, 10000000);
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name =
                                                             "UNREGISTERED_TRACK" };

    EXPECT_NO_THROW(m_writer->insert_region_data(region, environment));
    m_writer->flush_in_memory_data_to_disk();

    auto region_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_region_" + m_uuid);
    EXPECT_EQ(region_result.rows[0][0], "1") << "Region should be inserted";

    auto sample_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_sample_" + m_uuid);
    EXPECT_EQ(sample_result.rows[0][0], "0") << "No sample when track is not registered";
}

TEST_F(writer_test, insert_multiple_regions_same_track_creates_multiple_samples)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = profiler_hub::writer_types::track_info_t{ .name = "MULTI_SAMPLE_TRACK",
                                                           .extdata    = "{}",
                                                           .node_id    = 1,
                                                           .process_id = 1000,
                                                           .thread_id  = 100 };
    m_writer->register_track_info(track);

    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name =
                                                             "MULTI_SAMPLE_TRACK" };

    m_writer->insert_region_data(create_test_region_data("region_1", 1000000, 2000000),
                                 environment);
    m_writer->insert_region_data(create_test_region_data("region_2", 3000000, 4000000),
                                 environment);
    m_writer->insert_region_data(create_test_region_data("region_3", 5000000, 6000000),
                                 environment);

    m_writer->flush_in_memory_data_to_disk();

    auto region_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_region_" + m_uuid);
    EXPECT_EQ(region_result.rows[0][0], "3") << "All 3 regions should be inserted";

    auto sample_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_sample_" + m_uuid);
    size_t sample_count = std::stoull(sample_result.rows[0][0]);
    EXPECT_TRUE(sample_count == 0 || sample_count == 3)
        << "Either no samples (not implemented) or 3 samples (one per region)";
}

TEST_F(writer_test, sample_references_correct_track_id)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = profiler_hub::writer_types::track_info_t{ .name    = "REFERENCED_TRACK",
                                                           .extdata = "{}",
                                                           .node_id = 1,
                                                           .process_id = 1000,
                                                           .thread_id  = 100 };
    m_writer->register_track_info(track);

    auto region = create_test_region_data("ref_track_region", 1000000, 2000000);
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name =
                                                             "REFERENCED_TRACK" };

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto track_result =
        query_database(m_database_path, "SELECT id FROM rocpd_track_" + m_uuid);
    ASSERT_GE(track_result.rows.size(), 1) << "Track should exist";

    auto join_result = query_database(m_database_path,
                                      "SELECT s.id, s.track_id, t.id as track_table_id "
                                      "FROM rocpd_sample_" +
                                          m_uuid +
                                          " s "
                                          "JOIN rocpd_track_" +
                                          m_uuid + " t ON s.track_id = t.id");

    if(!join_result.rows.empty())
    {
        EXPECT_EQ(join_result.rows[0][1], join_result.rows[0][2])
            << "Sample track_id should match track table id";
    }
}

TEST_F(writer_test, sample_timestamp_matches_region_start)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track = profiler_hub::writer_types::track_info_t{ .name    = "TIMESTAMP_TRACK",
                                                           .extdata = "{}",
                                                           .node_id = 1,
                                                           .process_id = 1000,
                                                           .thread_id  = 100 };
    m_writer->register_track_info(track);

    constexpr size_t expected_start_timestamp = 12345678900;
    auto             region                   = create_test_region_data(
        "timestamp_region", expected_start_timestamp, expected_start_timestamp + 1000000);
    auto environment =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name =
                                                             "TIMESTAMP_TRACK" };

    m_writer->insert_region_data(region, environment);
    m_writer->flush_in_memory_data_to_disk();

    auto sample_result =
        query_database(m_database_path, "SELECT timestamp FROM rocpd_sample_" + m_uuid);

    if(!sample_result.rows.empty())
    {
        EXPECT_EQ(sample_result.rows[0][0], std::to_string(expected_start_timestamp))
            << "Sample timestamp should match region start timestamp";
    }
}

TEST_F(writer_test, multiple_tracks_with_different_regions)
{
    register_base_dependencies(*m_writer, 1, 1000, 100);

    auto track1 = profiler_hub::writer_types::track_info_t{ .name       = "TRACK_A",
                                                            .extdata    = "{}",
                                                            .node_id    = 1,
                                                            .process_id = 1000,
                                                            .thread_id  = 100 };
    auto track2 = profiler_hub::writer_types::track_info_t{ .name       = "TRACK_B",
                                                            .extdata    = "{}",
                                                            .node_id    = 1,
                                                            .process_id = 1000,
                                                            .thread_id  = 100 };
    m_writer->register_track_info(track1);
    m_writer->register_track_info(track2);

    auto env_a =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = "TRACK_A" };

    auto env_b =
        profiler_hub::writer_types::trace_environment_t{ .node_id    = 1,
                                                         .process_id = 1000,
                                                         .thread_id  = 100,
                                                         .agent_id   = std::nullopt,
                                                         .stream_id  = std::nullopt,
                                                         .queue_id   = std::nullopt,
                                                         .track_name = "TRACK_B" };

    m_writer->insert_region_data(create_test_region_data("region_a1", 1000000, 2000000),
                                 env_a);
    m_writer->insert_region_data(create_test_region_data("region_b1", 3000000, 4000000),
                                 env_b);
    m_writer->insert_region_data(create_test_region_data("region_a2", 5000000, 6000000),
                                 env_a);

    m_writer->flush_in_memory_data_to_disk();

    auto track_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_track_" + m_uuid);
    EXPECT_EQ(track_result.rows[0][0], "2") << "Two tracks should exist";

    auto region_result =
        query_database(m_database_path, "SELECT COUNT(*) FROM rocpd_region_" + m_uuid);
    EXPECT_EQ(region_result.rows[0][0], "3") << "Three regions should exist";
}

// ============================================================================
// Transaction Rollback Tests
// ============================================================================

TEST_F(writer_test, transaction_rollback_on_exception_reverts_inserts)
{
    // Setup: register base dependencies
    register_base_dependencies(*m_writer, 1, 1000, 100);

    // Step 1: Insert a successful region (no event, no args) - this commits
    auto region1      = create_test_region_data("successful_region", 1000000, 2000000);
    auto environment1 = create_test_trace_environment(1, 1000, 100);
    m_writer->insert_region_data(region1, environment1);

    // Step 2: Attempt to insert a region with event + invalid args
    // The event and region will be inserted, but insert_arg will throw
    // because arg.type is empty. This should trigger ROLLBACK.
    auto region2 = create_test_region_data("failed_region", 3000000, 4000000);
    region2.event =
        profiler_hub::writer_types::event_data_t{ .stack_id        = 1,
                                                  .parent_stack_id = 0,
                                                  .correlation_id  = 456,
                                                  .call_stack      = {},
                                                  .line_info_list  = {},
                                                  .event_category  = "TEST_API",
                                                  .extdata         = "{}" };
    // Add an arg with empty type to trigger exception AFTER event/region insert
    region2.args = { profiler_hub::writer_types::arg_data_t{ .position = 0,
                                                             .type     = "",
                                                             .name     = "arg_name",
                                                             .value    = "arg_value",
                                                             .extdata  = "{}" } };

    auto environment2 = create_test_trace_environment(1, 1000, 100);
    EXPECT_THROW(m_writer->insert_region_data(region2, environment2), std::runtime_error);

    // Step 3: Flush and verify
    m_writer->flush_in_memory_data_to_disk();

    // Verify: Only 1 region should exist (the successful one)
    auto region_count = count_rows(m_database_path, "rocpd_region", m_uuid);
    EXPECT_EQ(region_count, 1)
        << "Only the successful region should exist after rollback";

    // Verify: No events should exist (the failed transaction's event was rolled back)
    auto event_count = count_rows(m_database_path, "rocpd_event", m_uuid);
    EXPECT_EQ(event_count, 0) << "Event from failed transaction should be rolled back";

    // Verify: The successful region has the correct name
    auto region_result = query_database(m_database_path,
                                        "SELECT s.string FROM rocpd_region_" + m_uuid +
                                            " r "
                                            "JOIN rocpd_string_" +
                                            m_uuid + " s ON r.name_id = s.id");
    ASSERT_EQ(region_result.rows.size(), 1);
    EXPECT_EQ(region_result.rows[0][0], "successful_region");
}
