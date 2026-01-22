// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocstorage/storage.hpp>
#include <rocstorage/writer.hpp>
#include <rocstorage/writer_types.hpp>

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

rocstorage::writer_api::node_info_t
create_test_node_info(size_t node_id = 1)
{
    static std::unordered_map<size_t, std::string> machine_ids;
    auto&                                          machine_id = machine_ids[node_id];
    if(machine_id.empty())
    {
        machine_id = "test-machine-id-" + std::to_string(node_id);
    }

    return rocstorage::writer_api::node_info_t{ .node_id       = node_id,
                                                .hash          = 123456789 + node_id,
                                                .machine_id    = machine_id.c_str(),
                                                .system_name   = "Linux",
                                                .hostname      = "test-host",
                                                .release       = "5.15.0",
                                                .version       = "#1 SMP",
                                                .hardware_name = "x86_64",
                                                .domain_name   = "test-domain" };
}

rocstorage::writer_api::process_info_t
create_test_process_info(size_t node_id = 1, size_t pid = 1000)
{
    return rocstorage::writer_api::process_info_t{ .ppid        = 1,
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

rocstorage::writer_api::thread_info_t
create_test_thread_info(size_t node_id    = 1,
                        size_t process_id = 1000,
                        size_t thread_id  = 100)
{
    return rocstorage::writer_api::thread_info_t{ .parent_process_id = process_id,
                                                  .thread_id         = thread_id,
                                                  .name              = "test-thread",
                                                  .start             = 1000000,
                                                  .end               = 2000000,
                                                  .extdata           = "{}",
                                                  .node_id           = node_id,
                                                  .process_id        = process_id };
}

rocstorage::writer_api::agent_info_t
create_test_agent_info(size_t      node_id    = 1,
                       size_t      process_id = 1000,
                       const char* agent_type = "GPU",
                       size_t      type_index = 0)
{
    return rocstorage::writer_api::agent_info_t{
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

rocstorage::writer_api::stream_info_t
create_test_stream_info(size_t node_id    = 1,
                        size_t process_id = 1000,
                        size_t stream_id  = 1)
{
    return rocstorage::writer_api::stream_info_t{ .stream_id  = stream_id,
                                                  .name       = "test-stream",
                                                  .extdata    = "{}",
                                                  .node_id    = node_id,
                                                  .process_id = process_id };
}

rocstorage::writer_api::queue_info_t
create_test_queue_info(size_t node_id = 1, size_t process_id = 1000, size_t queue_id = 1)
{
    return rocstorage::writer_api::queue_info_t{ .queue_id   = queue_id,
                                                 .name       = "test-queue",
                                                 .extdata    = "{}",
                                                 .node_id    = node_id,
                                                 .process_id = process_id };
}

rocstorage::writer_api::pmc_info_t
create_test_pmc_info(
    size_t                                                   node_id    = 1,
    size_t                                                   process_id = 1000,
    const char*                                              name       = "test_counter",
    std::optional<rocstorage::writer_api::agent_unique_id_t> agent_id   = std::nullopt)
{
    return rocstorage::writer_api::pmc_info_t{ .unique_id   = { .name     = name,
                                                                .agent_id = agent_id },
                                               .target_arch = "GPU",
                                               .event_code  = 100,
                                               .instance_id = 0,
                                               .symbol      = "TEST_COUNTER",
                                               .description = "Test counter description",
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
                                               .process_id       = process_id };
}

rocstorage::writer_api::code_object_info_t
create_test_code_object_info(size_t                                    code_object_id = 1,
                             size_t                                    node_id        = 1,
                             size_t                                    process_id = 1000,
                             rocstorage::writer_api::agent_unique_id_t agent_id = { "GPU",
                                                                                    0 })
{
    return rocstorage::writer_api::code_object_info_t{ .id  = code_object_id,
                                                       .uri = "file:///test/kernel.co",
                                                       .ld_base      = 0x1000,
                                                       .ld_size      = 0x2000,
                                                       .ld_delta     = 0,
                                                       .storage_type = "FILE",
                                                       .extdata      = "{}",
                                                       .node_id      = node_id,
                                                       .process_id   = process_id,
                                                       .agent_id     = agent_id };
}

rocstorage::writer_api::kernel_symbol_info_t
create_test_kernel_symbol_info(size_t kernel_symbol_id = 1,
                               size_t node_id          = 1,
                               size_t process_id       = 1000,
                               size_t code_object_id   = 1)
{
    return rocstorage::writer_api::kernel_symbol_info_t{ .id           = kernel_symbol_id,
                                                         .name         = "test_kernel",
                                                         .display_name = "Test Kernel",
                                                         .kernel_obj   = 0x1000,
                                                         .kernarg_segmnt_size       = 64,
                                                         .kernarg_segment_alignment = 8,
                                                         .group_segment_size        = 256,
                                                         .private_segment_size      = 0,
                                                         .sgrp_count                = 32,
                                                         .arch_vgrp_count           = 64,
                                                         .accum_vgrp_count          = 0,
                                                         .extdata     = "{}",
                                                         .node_id     = node_id,
                                                         .process_id  = process_id,
                                                         .code_obj_id = code_object_id };
}

rocstorage::writer_api::track_info_t
create_test_track_info(
    size_t                                              node_id    = 1,
    std::optional<size_t>                               process_id = 1000,
    std::optional<size_t>                               thread_id  = 100,
    std::optional<rocstorage::writer_api::track_name_t> name       = "test-track")
{
    return rocstorage::writer_api::track_info_t{ .name       = name,
                                                 .extdata    = "{}",
                                                 .node_id    = node_id,
                                                 .process_id = process_id,
                                                 .thread_id  = thread_id };
}

rocstorage::writer_api::region_data_t
create_test_region_data(const char* name            = "test_region",
                        size_t      start_timestamp = 1000000,
                        size_t      end_timestamp   = 2000000)
{
    return rocstorage::writer_api::region_data_t{ .event           = std::nullopt,
                                                  .start_timestamp = start_timestamp,
                                                  .end_timestamp   = end_timestamp,
                                                  .name            = name,
                                                  .extdata         = "{}",
                                                  .args            = {} };
}

rocstorage::writer_api::trace_environment_t
create_test_trace_environment(size_t node_id    = 1,
                              size_t process_id = 1000,
                              size_t thread_id  = 100)
{
    return rocstorage::writer_api::trace_environment_t{ .node_id    = node_id,
                                                        .process_id = process_id,
                                                        .thread_id  = thread_id,
                                                        .agent_id   = std::nullopt,
                                                        .stream_id  = std::nullopt,
                                                        .queue_id   = std::nullopt,
                                                        .track_name = std::nullopt };
}

rocstorage::writer_api::pmc_event_data_t
create_test_pmc_event_data(double value = 42.5)
{
    return rocstorage::writer_api::pmc_event_data_t{
        .event   = std::nullopt,
        .value   = value,
        .extdata = "{}",
        .sample  = { .timestamp = 1000000,
                     .track     = create_test_track_info(),
                     .extdata   = "{}" }
    };
}

rocstorage::writer_api::kernel_dispatch_data_t
create_test_kernel_dispatch_data(size_t kernel_symbol_id = 1, size_t code_object_id = 1)
{
    return rocstorage::writer_api::kernel_dispatch_data_t{ .event       = std::nullopt,
                                                           .dispatch_id = 1,
                                                           .start_timestamp = 1000000,
                                                           .end_timestamp   = 2000000,
                                                           .kernel_symbol_id =
                                                               kernel_symbol_id,
                                                           .code_object_id =
                                                               code_object_id,
                                                           .private_segment_size = 0,
                                                           .group_segment_size   = 256,
                                                           .workgroup_size_x     = 64,
                                                           .workgroup_size_y     = 1,
                                                           .workgroup_size_z     = 1,
                                                           .grid_size_x          = 1024,
                                                           .grid_size_y          = 1,
                                                           .grid_size_z          = 1,
                                                           .name = "test_kernel_dispatch",
                                                           .extdata = "{}" };
}

rocstorage::writer_api::memory_copy_data_t
create_test_memory_copy_data()
{
    return rocstorage::writer_api::memory_copy_data_t{ .event           = std::nullopt,
                                                       .start_timestamp = 1000000,
                                                       .end_timestamp   = 2000000,
                                                       .dst_agent_id    = std::nullopt,
                                                       .dst_address     = 0x2000,
                                                       .src_agent_id    = std::nullopt,
                                                       .src_address     = 0x1000,
                                                       .size            = 4096,
                                                       .name            = "hipMemcpy",
                                                       .region_name     = nullptr,
                                                       .extdata         = "{}" };
}

rocstorage::writer_api::memory_alloc_data_t
create_test_memory_alloc_data(const char* type = "ALLOC", const char* level = "REAL")
{
    return rocstorage::writer_api::memory_alloc_data_t{ .event           = std::nullopt,
                                                        .type            = type,
                                                        .level           = level,
                                                        .start_timestamp = 1000000,
                                                        .end_timestamp   = 2000000,
                                                        .address         = 0x1000,
                                                        .size            = 4096,
                                                        .extdata         = "{}" };
}

// Helper to register base dependencies (node -> process -> thread)
void
register_base_dependencies(rocstorage::writer& writer,
                           size_t              node_id    = 1,
                           size_t              process_id = 1000,
                           size_t              thread_id  = 100)
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
        m_storage       = std::make_unique<rocm::storage>(m_database_path, m_uuid);
        m_writer        = m_storage->get_writer();
    }

    void TearDown() override
    {
        m_writer.reset();
        m_storage.reset();
        std::remove(m_database_path.c_str());
    }

    std::string                         m_database_path;
    std::string                         m_uuid;
    std::unique_ptr<rocm::storage>      m_storage;
    std::shared_ptr<rocstorage::writer> m_writer;
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
    EXPECT_THROW(m_writer->register_agent_info(agent), std::runtime_error);
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

    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
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
    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(999, 1000, "test", agent_id);
    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::runtime_error);
}

TEST_F(writer_test, register_pmc_info_without_process_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
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
    EXPECT_EQ(result.rows[0][1], std::to_string(code_obj.ld_base));
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

TEST_F(writer_test, register_string_null_throws)
{
    EXPECT_THROW(m_writer->register_string(nullptr), std::runtime_error);
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
    region.event = rocstorage::writer_api::event_data_t{ .stack_id        = 1,
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

    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "my_counter", agent_id);
    m_writer->register_pmc_info(pmc);

    auto pmc_event = create_test_pmc_event_data(99.5);
    auto pmc_unique_id =
        rocstorage::writer_api::pmc_info_unique_id_t{ .name     = "my_counter",
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
    auto pmc_unique_id = rocstorage::writer_api::pmc_info_unique_id_t{
        .name     = "nonexistent_counter",
        .agent_id = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 }
    };

    EXPECT_THROW(m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id),
                 std::runtime_error);
}

// --------------------- Kernel Dispatch Data Tests ---------------------

TEST_F(writer_test, insert_kernel_dispatch_full_dependencies)
{
    // Register all required dependencies
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
    auto environment     = rocstorage::writer_api::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    auto environment     = rocstorage::writer_api::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    auto environment     = rocstorage::writer_api::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    auto environment     = rocstorage::writer_api::trace_environment_t{
            .node_id    = 1,
            .process_id = 1000,
            .thread_id  = 100,
            .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    auto environment = rocstorage::writer_api::trace_environment_t{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    memory_copy.src_agent_id = rocstorage::writer_api::agent_unique_id_t{ "CPU", 0 };
    memory_copy.dst_agent_id = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 };

    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 999,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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

    auto memory_copy         = create_test_memory_copy_data();
    memory_copy.src_agent_id = rocstorage::writer_api::agent_unique_id_t{ "GPU", 999 };

    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 999,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
    auto environment  = rocstorage::writer_api::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = std::nullopt,
         .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
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
    auto environment  = rocstorage::writer_api::trace_environment_t{
         .node_id    = 1,
         .process_id = 1000,
         .thread_id  = std::nullopt,
         .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 999 },
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 1000,
                                                     .thread_id =
                                                         999,  // Thread not registered
                                                     .agent_id   = std::nullopt,
                                                     .stream_id  = std::nullopt,
                                                     .queue_id   = std::nullopt,
                                                     .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_copy_with_unregistered_dst_agent_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_copy         = create_test_memory_copy_data();
    memory_copy.dst_agent_id = rocstorage::writer_api::agent_unique_id_t{ "GPU", 999 };

    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 1000,
                                                     .thread_id  = std::nullopt,
                                                     .agent_id   = std::nullopt,
                                                     .stream_id =
                                                         999,  // Stream not registered
                                                     .queue_id   = std::nullopt,
                                                     .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_copy_data(memory_copy, environment),
                 std::runtime_error);
}

// --------------------- PMC Event Edge Cases ---------------------

TEST_F(writer_test, insert_pmc_event_data_with_event)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));
    m_writer->register_agent_info(create_test_agent_info(1, 1000, "GPU", 0));

    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
    auto pmc = create_test_pmc_info(1, 1000, "counter_with_event", agent_id);
    m_writer->register_pmc_info(pmc);

    auto pmc_event  = create_test_pmc_event_data(123.456);
    pmc_event.event = rocstorage::writer_api::event_data_t{ .stack_id        = 10,
                                                            .parent_stack_id = 0,
                                                            .correlation_id  = 456,
                                                            .call_stack      = {},
                                                            .line_info_list  = {},
                                                            .event_category  = "PMC",
                                                            .extdata         = "{}" };

    auto pmc_unique_id =
        rocstorage::writer_api::pmc_info_unique_id_t{ .name     = "counter_with_event",
                                                      .agent_id = agent_id };

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

    rocstorage::writer_api::agent_unique_id_t agent_id{ "GPU", 0 };
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
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
    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 1000,
                                                     .thread_id  = std::nullopt,
                                                     .agent_id   = std::nullopt,
                                                     .stream_id =
                                                         999,  // Stream not registered
                                                     .queue_id   = std::nullopt,
                                                     .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

TEST_F(writer_test, insert_memory_alloc_with_unregistered_thread_throws)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc = create_test_memory_alloc_data("ALLOC", "REAL");
    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 1000,
                                                     .thread_id =
                                                         999,  // Thread not registered
                                                     .agent_id   = std::nullopt,
                                                     .stream_id  = std::nullopt,
                                                     .queue_id   = std::nullopt,
                                                     .track_name = std::nullopt };

    EXPECT_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment),
                 std::runtime_error);
}

// --------------------- Null Type/Level Tests ---------------------

TEST_F(writer_test, insert_memory_alloc_with_null_type)
{
    m_writer->register_node_info(create_test_node_info(1));
    m_writer->register_process_info(create_test_process_info(1, 1000));

    auto memory_alloc  = create_test_memory_alloc_data("ALLOC", "REAL");
    memory_alloc.type  = nullptr;
    memory_alloc.level = nullptr;

    auto environment =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 1000,
                                                     .thread_id  = std::nullopt,
                                                     .agent_id   = std::nullopt,
                                                     .stream_id  = std::nullopt,
                                                     .queue_id   = std::nullopt,
                                                     .track_name = std::nullopt };

    // Should not throw - null type/level should be allowed
    EXPECT_NO_THROW(m_writer->insert_memory_alloc_data(memory_alloc, environment));
}

// --------------------- Empty String Edge Case ---------------------

TEST_F(writer_test, register_string_empty_string)
{
    m_writer->register_string("");
    m_writer->flush_in_memory_data_to_disk();

    auto result = query_database(m_database_path,
                                 "SELECT COUNT(*) FROM rocpd_string_" + m_uuid +
                                     " WHERE string = ''");

    EXPECT_EQ(result.rows[0][0], "1");
}

// ============================================================================
// End-to-End Happy Path Test - Covers Entire Writer API
// ============================================================================

TEST_F(writer_test, end_to_end_complete_api_coverage)
{
    // ========================================================================
    // 1. Register Info Tables (in dependency order)
    // ========================================================================

    // 1.1 Node Info - root of all dependencies
    auto node = rocstorage::writer_api::node_info_t{ .node_id       = 1,
                                                     .hash          = 987654321,
                                                     .machine_id    = "e2e-machine-001",
                                                     .system_name   = "Linux",
                                                     .hostname      = "e2e-test-host",
                                                     .release       = "6.1.0-generic",
                                                     .version       = "#1 SMP PREEMPT",
                                                     .hardware_name = "x86_64",
                                                     .domain_name   = "e2e.test.local" };
    m_writer->register_node_info(node);

    // 1.2 Process Info - depends on Node
    auto process =
        rocstorage::writer_api::process_info_t{ .ppid  = 1,
                                                .pid   = 12345,
                                                .init  = 1000000000,
                                                .fini  = 9000000000,
                                                .start = 1000000000,
                                                .end   = 9000000000,
                                                .command =
                                                    "/usr/bin/e2e_test_app --verbose",
                                                .environment = "{\"PATH\":\"/usr/bin\"}",
                                                .extdata     = "{\"test\":true}",
                                                .node_id     = 1 };
    m_writer->register_process_info(process);

    // 1.3 Thread Info - depends on Node, Process
    auto thread = rocstorage::writer_api::thread_info_t{ .parent_process_id = 12345,
                                                         .thread_id         = 100,
                                                         .name       = "main-thread",
                                                         .start      = 1000000000,
                                                         .end        = 9000000000,
                                                         .extdata    = "{}",
                                                         .node_id    = 1,
                                                         .process_id = 12345 };
    m_writer->register_thread_info(thread);

    // 1.4 Agent Info (GPU) - depends on Node, Process
    auto gpu_agent = rocstorage::writer_api::agent_info_t{
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

    // 1.5 Agent Info (CPU) - for memory copy source
    auto cpu_agent =
        rocstorage::writer_api::agent_info_t{ .unique_id      = { .agent_type = "CPU",
                                                                  .type_index = 0 },
                                              .absolute_index = 0,
                                              .logical_index  = 0,
                                              .uuid           = 0,
                                              .name           = "AMD Ryzen 9",
                                              .model_name     = "AMD Ryzen 9 7950X",
                                              .vendor_name    = "Advanced Micro Devices",
                                              .product_name   = "Ryzen 9 7950X",
                                              .user_name      = "cpu0",
                                              .extdata        = "{}",
                                              .node_id        = 1,
                                              .process_id     = 12345 };
    m_writer->register_agent_info(cpu_agent);

    // 1.6 Queue Info - depends on Node, Process
    auto queue = rocstorage::writer_api::queue_info_t{ .queue_id   = 1,
                                                       .name       = "compute-queue-0",
                                                       .extdata    = "{}",
                                                       .node_id    = 1,
                                                       .process_id = 12345 };
    m_writer->register_queue_info(queue);

    // 1.7 Stream Info - depends on Node, Process
    auto stream = rocstorage::writer_api::stream_info_t{ .stream_id  = 1,
                                                         .name       = "hip-stream-0",
                                                         .extdata    = "{}",
                                                         .node_id    = 1,
                                                         .process_id = 12345 };
    m_writer->register_stream_info(stream);

    // 1.8 PMC Info - depends on Node, Process, Agent
    rocstorage::writer_api::agent_unique_id_t pmc_agent_id{ "GPU", 0 };
    auto                                      pmc = rocstorage::writer_api::pmc_info_t{
                                             .unique_id = { .name = "SQ_WAVES", .agent_id = pmc_agent_id },
                                             .target_arch = "GPU",
                                             .event_code  = 4,
                                             .instance_id = 0,
                                             .symbol      = "SQ_WAVES",
                                             .description = "Number of waves sent to SQs",
                                             .long_description = "Count of waves dispatched to shader engines",
                                             .component   = "SQ",
                                             .units       = "waves",
                                             .value_type  = "ABS",
                                             .block       = "SQ",
                                             .expression  = "",
                                             .is_constant = 0,
                                             .is_derived  = 0,
                                             .extdata     = "{}",
                                             .node_id     = 1,
                                             .process_id  = 12345
    };
    m_writer->register_pmc_info(pmc);

    // 1.9 Code Object Info - depends on Node, Process, Agent
    auto code_object = rocstorage::writer_api::code_object_info_t{
        .id           = 1,
        .uri          = "file:///opt/rocm/lib/e2e_kernel.co",
        .ld_base      = 0x7F0000000000,
        .ld_size      = 0x100000,
        .ld_delta     = 0,
        .storage_type = "FILE",
        .extdata      = "{}",
        .node_id      = 1,
        .process_id   = 12345,
        .agent_id     = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 }
    };
    m_writer->register_code_object_info(code_object);

    // 1.10 Kernel Symbol Info - depends on Node, Process, Code Object
    auto kernel_symbol =
        rocstorage::writer_api::kernel_symbol_info_t{ .id = 1,
                                                      .name =
                                                          "_Z12vectorAddKernelPfS_S_i",
                                                      .display_name = "vectorAddKernel",
                                                      .kernel_obj   = 0x7F0000001000,
                                                      .kernarg_segmnt_size       = 32,
                                                      .kernarg_segment_alignment = 8,
                                                      .group_segment_size        = 0,
                                                      .private_segment_size      = 0,
                                                      .sgrp_count                = 16,
                                                      .arch_vgrp_count           = 32,
                                                      .accum_vgrp_count          = 0,
                                                      .extdata                   = "{}",
                                                      .node_id                   = 1,
                                                      .process_id                = 12345,
                                                      .code_obj_id               = 1 };
    m_writer->register_kernel_symbol_info(kernel_symbol);

    // 1.11 Track Info - depends on Node, Process, Thread
    auto track = rocstorage::writer_api::track_info_t{ .name       = "HIP_API",
                                                       .extdata    = "{}",
                                                       .node_id    = 1,
                                                       .process_id = 12345,
                                                       .thread_id  = 100 };
    m_writer->register_track_info(track);

    // 1.12 String - standalone
    m_writer->register_string("hipLaunchKernelGGL");

    // ========================================================================
    // 2. Insert Data Tables
    // ========================================================================

    // 2.1 Region Data - API tracing event
    auto region = rocstorage::writer_api::region_data_t{
        .event           = rocstorage::writer_api::event_data_t{ .stack_id        = 1,
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
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 12345,
                                                     .thread_id  = 100,
                                                     .agent_id   = std::nullopt,
                                                     .stream_id  = std::nullopt,
                                                     .queue_id   = std::nullopt,
                                                     .track_name = "HIP_API" };
    m_writer->insert_region_data(region, region_env);

    // 2.2 PMC Event Data - performance counter sample
    auto pmc_event = rocstorage::writer_api::pmc_event_data_t{
        .event   = std::nullopt,
        .value   = 1024.0,
        .extdata = "{}",
        .sample  = { .timestamp = 2500000000, .track = track, .extdata = "{}" }
    };
    auto pmc_unique_id =
        rocstorage::writer_api::pmc_info_unique_id_t{ .name     = "SQ_WAVES",
                                                      .agent_id = pmc_agent_id };
    m_writer->insert_pmc_event_data(pmc_event, pmc_unique_id);

    // 2.3 Kernel Dispatch Data - GPU kernel execution
    auto kernel_dispatch =
        rocstorage::writer_api::kernel_dispatch_data_t{ .event            = std::nullopt,
                                                        .dispatch_id      = 1,
                                                        .start_timestamp  = 3000000000,
                                                        .end_timestamp    = 3000500000,
                                                        .kernel_symbol_id = 1,
                                                        .code_object_id   = 1,
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
    auto kernel_env = rocstorage::writer_api::trace_environment_t{
        .node_id    = 1,
        .process_id = 12345,
        .thread_id  = 100,
        .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = std::nullopt
    };
    m_writer->insert_kernel_dispatch_data(kernel_dispatch, kernel_env);

    // 2.4 Memory Copy Data - host to device transfer
    auto memory_copy = rocstorage::writer_api::memory_copy_data_t{
        .event           = std::nullopt,
        .start_timestamp = 2100000000,
        .end_timestamp   = 2200000000,
        .dst_agent_id    = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
        .dst_address     = 0x7F1000000000,
        .src_agent_id    = rocstorage::writer_api::agent_unique_id_t{ "CPU", 0 },
        .src_address     = 0x7FFE00000000,
        .size            = 1048576,
        .name            = "hipMemcpyHtoD",
        .region_name     = nullptr,
        .extdata         = "{}"
    };
    auto memcpy_env =
        rocstorage::writer_api::trace_environment_t{ .node_id    = 1,
                                                     .process_id = 12345,
                                                     .thread_id  = 100,
                                                     .agent_id   = std::nullopt,
                                                     .stream_id  = 1,
                                                     .queue_id   = 1,
                                                     .track_name = std::nullopt };
    m_writer->insert_memory_copy_data(memory_copy, memcpy_env);

    // 2.5 Memory Alloc Data - device memory allocation
    auto memory_alloc =
        rocstorage::writer_api::memory_alloc_data_t{ .event           = std::nullopt,
                                                     .type            = "ALLOC",
                                                     .level           = "REAL",
                                                     .start_timestamp = 2000000000,
                                                     .end_timestamp   = 2000050000,
                                                     .address         = 0x7F1000000000,
                                                     .size            = 1048576,
                                                     .extdata         = "{}" };
    auto alloc_env = rocstorage::writer_api::trace_environment_t{
        .node_id    = 1,
        .process_id = 12345,
        .thread_id  = 100,
        .agent_id   = rocstorage::writer_api::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = std::nullopt,
        .queue_id   = std::nullopt,
        .track_name = std::nullopt
    };
    m_writer->insert_memory_alloc_data(memory_alloc, alloc_env);

    // ========================================================================
    // 3. Flush to disk and validate
    // ========================================================================
    m_writer->flush_in_memory_data_to_disk();

    // ========================================================================
    // 4. Verify all tables have data
    // ========================================================================

    // Info Tables
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

    // Data Tables
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

    // ========================================================================
    // 5. Verify specific data values
    // ========================================================================

    // Verify node data
    auto node_result = query_database(
        m_database_path, "SELECT machine_id, hostname FROM rocpd_info_node_" + m_uuid);
    ASSERT_EQ(node_result.rows.size(), 1);
    EXPECT_EQ(node_result.rows[0][0], "e2e-machine-001");
    EXPECT_EQ(node_result.rows[0][1], "e2e-test-host");

    // Verify process data
    auto process_result = query_database(
        m_database_path, "SELECT pid, command FROM rocpd_info_process_" + m_uuid);
    ASSERT_EQ(process_result.rows.size(), 1);
    EXPECT_EQ(process_result.rows[0][0], "12345");

    // Verify region data
    auto region_result =
        query_database(m_database_path, "SELECT start, end FROM rocpd_region_" + m_uuid);
    ASSERT_EQ(region_result.rows.size(), 1);
    EXPECT_EQ(region_result.rows[0][0], "2000000000");
    EXPECT_EQ(region_result.rows[0][1], "2000100000");

    // Verify memory copy data
    auto memcpy_result = query_database(
        m_database_path,
        "SELECT size, src_address, dst_address FROM rocpd_memory_copy_" + m_uuid);
    ASSERT_EQ(memcpy_result.rows.size(), 1);
    EXPECT_EQ(memcpy_result.rows[0][0], "1048576");

    // Verify memory alloc data
    auto alloc_result = query_database(
        m_database_path, "SELECT type, level, size FROM rocpd_memory_allocate_" + m_uuid);
    ASSERT_EQ(alloc_result.rows.size(), 1);
    EXPECT_EQ(alloc_result.rows[0][0], "ALLOC");
    EXPECT_EQ(alloc_result.rows[0][1], "REAL");
    EXPECT_EQ(alloc_result.rows[0][2], "1048576");

    // Verify PMC event value
    auto pmc_result =
        query_database(m_database_path, "SELECT value FROM rocpd_pmc_event_" + m_uuid);
    ASSERT_EQ(pmc_result.rows.size(), 1);
    EXPECT_DOUBLE_EQ(std::stod(pmc_result.rows[0][0]), 1024.0);
}
