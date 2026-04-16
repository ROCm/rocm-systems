// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @brief V3 Schema Value Insertion Tests
 *
 * These tests verify that actual values are correctly inserted into V3 tables
 * and that foreign key relationships are properly maintained.
 * Tests cover all 20 tables defined in schema/3.0.0/rocpd_tables.sql
 */

#include <chrono>
#include <filesystem>
#include <gtest/gtest.h>
#include <rocpdsna/storage.hpp>
#include <rocpdsna/writer.hpp>
#include <rocpdsna/writer_types.hpp>
#include <sqlite3.h>
#include <string>
#include <thread>
#include <vector>

#define V3_VERSION                                                                       \
    rocpdsna::version_t { 3, 0, 0 }

namespace
{

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
        throw std::runtime_error("Failed to prepare query: " + error +
                                 " Query: " + query);
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

}  // anonymous namespace

// ============================================================================
// V3 Value Insertion Test Fixture
// ============================================================================

class SchemaV3ValueInsertionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = "test_v3_value_insertion.db";
        m_uuid    = "v3val";
        std::filesystem::remove(m_db_path);

        auto storage =
            std::make_unique<rocpdsna::storage_t>(m_db_path, m_uuid, V3_VERSION);
        m_writer = std::make_unique<rocpdsna::writer_t>(std::move(storage));
    }

    void TearDown() override
    {
        m_writer.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::filesystem::remove(m_db_path);
    }

    void flush_and_wait()
    {
        m_writer->flush_in_memory_data_to_disk();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // ========================================================================
    // Test Data Factories
    // ========================================================================

    rocpdsna::writer_types::node_info_t create_node_info(size_t node_id = 1)
    {
        static std::unordered_map<size_t, std::string> machine_ids;
        auto&                                          machine_id = machine_ids[node_id];
        if(machine_id.empty())
        {
            machine_id = "test-machine-" + std::to_string(node_id);
        }
        return { .node_id       = node_id,
                 .hash          = 100000 + node_id,
                 .machine_id    = machine_id.c_str(),
                 .system_name   = "Linux",
                 .hostname      = "test-host",
                 .release       = "3.0",
                 .version       = "#1",
                 .hardware_name = "x86_64",
                 .domain_name   = "test.local",
                 .name          = "TestNode" };
    }

    rocpdsna::writer_types::process_info_t create_process_info(size_t node_id = 1,
                                                               size_t pid     = 1000)
    {
        return { .ppid        = 1,
                 .pid         = pid,
                 .init        = 1000000,
                 .fini        = 9000000,
                 .start       = 1000000,
                 .end         = 9000000,
                 .command     = "ls",
                 .environment = R"({"PATH":"/usr/bin"})",
                 .extdata     = R"({"custom":"data"})",
                 .node_id     = node_id,
                 .name        = "TestProcess" };
    }

    rocpdsna::writer_types::thread_info_t create_thread_info(size_t node_id    = 1,
                                                             size_t process_id = 1000,
                                                             size_t thread_id  = 100)
    {
        return { .parent_process_id = process_id,
                 .thread_id         = thread_id,
                 .name              = "TestThread",
                 .start             = 1100000,
                 .end               = 8900000,
                 .extdata           = R"({"priority":"high","affinity":3})",
                 .node_id           = node_id,
                 .process_id        = process_id };
    }

    rocpdsna::writer_types::agent_info_t create_agent_info(size_t      node_id    = 1,
                                                           size_t      process_id = 1000,
                                                           const char* agent_type = "GPU",
                                                           size_t      type_index = 0)
    {
        return { .unique_id      = { .agent_type = agent_type, .type_index = type_index },
                 .absolute_index = 0,
                 .logical_index  = 0,
                 .uuid           = 99999,
                 .name           = "gfx1100",
                 .model_name     = "AMD Radeon RX 7900",
                 .vendor_name    = "AMD",
                 .product_name   = "Radeon RX 7900 XTX",
                 .user_name      = "gpu0",
                 .extdata        = R"({"pci_bus":"0000:03:00.0","mem_gb":24})",
                 .node_id        = node_id,
                 .process_id     = process_id };
    }

    rocpdsna::writer_types::queue_info_t create_queue_info(size_t node_id    = 1,
                                                           size_t process_id = 1000,
                                                           size_t queue_id   = 1)
    {
        return { .queue_id   = queue_id,
                 .name       = "TestQueue",
                 .extdata    = R"({"capacity":128,"type":"compute"})",
                 .node_id    = node_id,
                 .process_id = process_id };
    }

    rocpdsna::writer_types::stream_info_t create_stream_info(size_t node_id    = 1,
                                                             size_t process_id = 1000,
                                                             size_t stream_id  = 1)
    {
        return { .stream_id  = stream_id,
                 .name       = "TestStream",
                 .extdata    = R"({"flags":"default","priority":0})",
                 .node_id    = node_id,
                 .process_id = process_id };
    }

    rocpdsna::writer_types::pmc_info_t create_pmc_info(
        size_t                                    node_id    = 1,
        size_t                                    process_id = 1000,
        const char*                               name       = "SQ_WAVES",
        rocpdsna::writer_types::agent_unique_id_t agent_id =
            rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 })
    {
        return { .unique_id        = { .name = name, .agent_id = agent_id },
                 .target_arch      = "GPU",
                 .event_code       = 42,
                 .instance_id      = 0,
                 .symbol           = "SQ_WAVES",
                 .description      = "Number of waves",
                 .long_description = "Total number of waves",
                 .component        = "SQ",
                 .units            = "waves",
                 .value_type       = "ACCUM",
                 .block            = "SQ",
                 .expression       = "",
                 .is_constant      = 0,
                 .is_derived       = 0,
                 .extdata          = R"({"shader_engine":0,"cu_mask":"0xff"})",
                 .node_id          = node_id,
                 .process_id       = process_id,
                 .qualifier        = "ACCUM" };
    }

    rocpdsna::writer_types::code_object_info_t create_code_object_info(
        size_t                                    code_object_id = 1,
        size_t                                    node_id        = 1,
        size_t                                    process_id     = 1000,
        rocpdsna::writer_types::agent_unique_id_t agent_id       = { "GPU", 0 })
    {
        return { .id           = code_object_id,
                 .uri          = "www.amd.com",
                 .load_base    = 0x7f0000000000,
                 .load_size    = 0x100000,
                 .load_delta   = 0,
                 .storage_type = "FILE",
                 .extdata      = R"({"isa":"gfx1100","format":"elf"})",
                 .node_id      = node_id,
                 .process_id   = process_id,
                 .agent_id     = agent_id };
    }

    rocpdsna::writer_types::kernel_symbol_info_t create_kernel_symbol_info(
        size_t kernel_id      = 1,
        size_t node_id        = 1,
        size_t process_id     = 1000,
        size_t code_object_id = 1)
    {
        return { .id                        = kernel_id,
                 .name                      = "vector_add",
                 .display_name              = "VectorAdd Kernel",
                 .kernel_object             = 0x1234,
                 .kernarg_segment_size      = 64,
                 .kernarg_segment_alignment = 8,
                 .group_segment_size        = 256,
                 .private_segment_size      = 0,
                 .sgpr_count                = 32,
                 .arch_vgpr_count           = 64,
                 .accum_vgpr_count          = 0,
                 .extdata                   = R"({"lds_size":256,"waves_per_eu":4})",
                 .node_id                   = node_id,
                 .process_id                = process_id,
                 .code_obj_id               = code_object_id };
    }

    rocpdsna::writer_types::track_info_t create_track_info(
        size_t                          node_id    = 1,
        std::optional<size_t>           process_id = 1000,
        std::optional<size_t>           thread_id  = 100,
        std::optional<std::string_view> name       = "TestTrack")
    {
        return { .name       = name,
                 .extdata    = R"({"source":"profiler","kind":"gpu"})",
                 .node_id    = node_id,
                 .process_id = process_id,
                 .thread_id  = thread_id };
    }

    rocpdsna::writer_types::region_data_t create_region_data(
        const char* name  = "test_region",
        size_t      start = 1000000,
        size_t      end   = 2000000)
    {
        return { .event =
                     rocpdsna::writer_types::event_data_t{
                         .stack_id        = 1,
                         .parent_stack_id = 0,
                         .correlation_id  = 1,
                         .call_stack      = {},
                         .line_info_list  = {},
                         .event_category  = "GENERAL",
                         .extdata         = R"({"origin":"api","depth":0})" },
                 .start_timestamp = start,
                 .end_timestamp   = end,
                 .name            = name,
                 .extdata         = R"({"scope":"global","async":false})",
                 .args            = {} };
    }

    rocpdsna::writer_types::kernel_dispatch_data_t create_kernel_dispatch_data(
        size_t dispatch_id = 1,
        size_t kernel_id   = 1,
        size_t code_obj_id = 1,
        size_t start       = 2000000,
        size_t end         = 3000000)
    {
        return { .event =
                     rocpdsna::writer_types::event_data_t{
                         .stack_id        = 1,
                         .parent_stack_id = 0,
                         .correlation_id  = 1,
                         .call_stack      = {},
                         .line_info_list  = {},
                         .event_category  = "GENERAL",
                         .extdata         = R"({"origin":"api","depth":0})" },
                 .dispatch_id          = dispatch_id,
                 .start_timestamp      = start,
                 .end_timestamp        = end,
                 .kernel_symbol_id     = kernel_id,
                 .code_object_id       = code_obj_id,
                 .private_segment_size = 0,
                 .group_segment_size   = 256,
                 .workgroup_size_x     = 64,
                 .workgroup_size_y     = 1,
                 .workgroup_size_z     = 1,
                 .grid_size_x          = 1024,
                 .grid_size_y          = 1,
                 .grid_size_z          = 1,
                 .name                 = "vector_add",
                 .extdata              = R"({"queue_idx":0,"signal_handle":42})" };
    }

    rocpdsna::writer_types::memory_copy_data_t create_memory_copy_data(
        size_t start = 3000000,
        size_t end   = 3500000)
    {
        return { .event =
                     rocpdsna::writer_types::event_data_t{
                         .stack_id        = 1,
                         .parent_stack_id = 0,
                         .correlation_id  = 1,
                         .call_stack      = {},
                         .line_info_list  = {},
                         .event_category  = "GENERAL",
                         .extdata         = R"({"origin":"api","depth":0})" },
                 .start_timestamp = start,
                 .end_timestamp   = end,
                 .dst_agent_id    = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
                 .dst_address     = 0x7f0000100000,
                 .src_agent_id    = rocpdsna::writer_types::agent_unique_id_t{ "CPU", 0 },
                 .src_address     = 0x7f0000200000,
                 .size            = 1024 * 1024,
                 .name            = "hipMemcpy",
                 .region_name     = "default_region",
                 .extdata         = R"({"direction":"HtoD","pinned":true})" };
    }

    rocpdsna::writer_types::memory_alloc_data_t create_memory_alloc_data(
        size_t start = 500000,
        size_t end   = 600000)
    {
        return { .event =
                     rocpdsna::writer_types::event_data_t{
                         .stack_id        = 1,
                         .parent_stack_id = 0,
                         .correlation_id  = 1,
                         .call_stack      = {},
                         .line_info_list  = {},
                         .event_category  = "GENERAL",
                         .extdata         = R"({"origin":"api","depth":0})" },
                 .type            = "ALLOC",
                 .level           = "REAL",
                 .start_timestamp = start,
                 .end_timestamp   = end,
                 .address         = 0x7f0000100000,
                 .size            = 1024 * 1024,
                 .extdata         = R"({"pool":"default","managed":false})" };
    }

    rocpdsna::writer_types::trace_environment_t create_trace_env()
    {
        return { .node_id    = 1,
                 .process_id = 1000,
                 .thread_id  = 100,
                 .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
                 .stream_id  = 1,
                 .queue_id   = 1,
                 .track_name = "TestTrack" };
    }

    void register_base_entities()
    {
        m_writer->register_node_info(create_node_info(1));
        m_writer->register_process_info(create_process_info(1, 1000));
        m_writer->register_thread_info(create_thread_info(1, 1000, 100));
        m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
        m_writer->register_queue_info(create_queue_info(1, 1000, 1));
        m_writer->register_stream_info(create_stream_info(1, 1000, 1));
        m_writer->register_track_info(create_track_info(1, 1000, 100, "TestTrack"));
    }

    std::string                         m_db_path;
    std::string                         m_uuid;
    std::unique_ptr<rocpdsna::writer_t> m_writer;
};

// ============================================================================
// 1. rocpd_metadata - Populated on schema creation
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, metadata_table_populated)
{
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT tag, value FROM rocpd_metadata_" + m_uuid + " ORDER BY tag");

    ASSERT_GE(result.rows.size(), 3u);

    bool found_schema = false;
    bool found_uuid   = false;
    bool found_guid   = false;
    for(const auto& row : result.rows)
    {
        if(row[0] == "schema_version")
        {
#if USE_SCHEMA_FROM_ROCPROFILER_SDK_ROCPD
            EXPECT_EQ(row[1], "3.0.0");
#else
            EXPECT_EQ(row[1], "3");
#endif
            found_schema = true;
        }
        else if(row[0] == "uuid")
        {
            EXPECT_TRUE(row[1].find(m_uuid) != std::string::npos)
                << "Expected uuid value to contain '" << m_uuid << "', got: " << row[1];
            found_uuid = true;
        }
        else if(row[0] == "guid")
        {
            found_guid = true;
        }
    }
    EXPECT_TRUE(found_schema);
    EXPECT_TRUE(found_uuid);
    EXPECT_TRUE(found_guid);
}

// ============================================================================
// 2. rocpd_string - String deduplication via UNIQUE
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, string_table_deduplication)
{
    m_writer->register_string("hello");
    m_writer->register_string("world");
    m_writer->register_string("hello");
    flush_and_wait();

    auto cnt = count_rows(m_db_path, "rocpd_string", m_uuid);
    auto result =
        query_database(m_db_path,
                       "SELECT string FROM rocpd_string_" + m_uuid +
                           " WHERE string IN ('hello','world') ORDER BY string");

    EXPECT_GE(cnt, 2u);
    ASSERT_GE(result.rows.size(), 2u);
    EXPECT_EQ(result.rows[0][0], "hello");
    EXPECT_EQ(result.rows[1][0], "world");
}

// ============================================================================
// 3. rocpd_info_node - All V3 columns (no "name" column in V3)
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_node_values_inserted)
{
    m_writer->register_node_info(create_node_info(1));
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT hash, machine_id, system_name, hostname, release, version, "
        "hardware_name, domain_name FROM rocpd_info_node_" +
            m_uuid);

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "100001");
    EXPECT_EQ(result.rows[0][1], "test-machine-1");
    EXPECT_EQ(result.rows[0][2], "Linux");
    EXPECT_EQ(result.rows[0][3], "test-host");
    EXPECT_EQ(result.rows[0][4], "3.0");
    EXPECT_EQ(result.rows[0][5], "#1");
    EXPECT_EQ(result.rows[0][6], "x86_64");
    EXPECT_EQ(result.rows[0][7], "test.local");
}

// ============================================================================
// 4. rocpd_info_process - FK to node, no "name" column in V3
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_process_values_and_fk)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT p.pid, p.ppid, p.command, p.init, p.fini, n.machine_id, "
                       "p.extdata "
                       "FROM rocpd_info_process_" +
                           m_uuid +
                           " p "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON p.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "1000");
    EXPECT_EQ(result.rows[0][1], "1");
    EXPECT_EQ(result.rows[0][2], "ls");
    EXPECT_EQ(result.rows[0][3], "1000000");
    EXPECT_EQ(result.rows[0][4], "9000000");
    EXPECT_EQ(result.rows[0][5], "test-machine-1");
    EXPECT_EQ(result.rows[0][6], R"({"custom":"data"})");
}

// ============================================================================
// 5. rocpd_info_thread - FK to node and process
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_thread_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_thread_info(create_thread_info(1, 1000, 100));
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT t.tid, t.name, t.start, t.end, p.pid, n.machine_id, "
                       "t.extdata "
                       "FROM rocpd_info_thread_" +
                           m_uuid +
                           " t "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON t.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON t.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "100");
    EXPECT_EQ(result.rows[0][1], "TestThread");
    EXPECT_EQ(result.rows[0][2], "1100000");
    EXPECT_EQ(result.rows[0][3], "8900000");
    EXPECT_EQ(result.rows[0][4], "1000");
    EXPECT_EQ(result.rows[0][5], "test-machine-1");
    EXPECT_EQ(result.rows[0][6], R"({"priority":"high","affinity":3})");
}

// ============================================================================
// 6. rocpd_info_agent - V3 has user_name, no generic_name
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_agent_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT a.type, a.type_index, a.name, a.model_name, a.vendor_name, "
        "a.product_name, a.user_name, p.pid, a.extdata "
        "FROM rocpd_info_agent_" +
            m_uuid +
            " a "
            "JOIN rocpd_info_process_" +
            m_uuid + " p ON a.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "GPU");
    EXPECT_EQ(result.rows[0][1], "0");
    EXPECT_EQ(result.rows[0][2], "gfx1100");
    EXPECT_EQ(result.rows[0][3], "AMD Radeon RX 7900");
    EXPECT_EQ(result.rows[0][4], "AMD");
    EXPECT_EQ(result.rows[0][5], "Radeon RX 7900 XTX");
    EXPECT_EQ(result.rows[0][6], "gpu0");
    EXPECT_EQ(result.rows[0][7], "1000");
    EXPECT_EQ(result.rows[0][8], R"({"pci_bus":"0000:03:00.0","mem_gb":24})");
}

// ============================================================================
// 7. rocpd_info_queue - FK to node and process
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_queue_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_queue_info(create_queue_info(1, 1000, 1));
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT q.name, p.pid, q.extdata FROM rocpd_info_queue_" + m_uuid +
                           " q "
                           "JOIN rocpd_info_process_" +
                           m_uuid + " p ON q.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "TestQueue");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[0][2], R"({"capacity":128,"type":"compute"})");
}

// ============================================================================
// 8. rocpd_info_stream - FK to node and process
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_stream_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_stream_info(create_stream_info(1, 1000, 1));
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT s.name, p.pid, s.extdata FROM rocpd_info_stream_" + m_uuid +
            " s "
            "JOIN rocpd_info_process_" +
            m_uuid + " p ON s.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "TestStream");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[0][2], R"({"flags":"default","priority":0})");
}

// ============================================================================
// 9. rocpd_info_pmc - V3 has no qualifier column; FK to agent
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_pmc_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_pmc_info(create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 }));
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT pm.name, pm.symbol, pm.description, pm.units, "
                                 "pm.value_type, pm.block, a.type, pm.extdata "
                                 "FROM rocpd_info_pmc_" +
                                     m_uuid +
                                     " pm "
                                     "JOIN rocpd_info_agent_" +
                                     m_uuid + " a ON pm.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "SQ_WAVES");
    EXPECT_EQ(result.rows[0][1], "SQ_WAVES");
    EXPECT_EQ(result.rows[0][2], "Number of waves");
    EXPECT_EQ(result.rows[0][3], "waves");
    EXPECT_EQ(result.rows[0][4], "ACCUM");
    EXPECT_EQ(result.rows[0][5], "SQ");
    EXPECT_EQ(result.rows[0][6], "GPU");
    EXPECT_EQ(result.rows[0][7], R"({"shader_engine":0,"cu_mask":"0xff"})");
}

// ============================================================================
// 10. rocpd_info_code_object - FK to agent
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_code_object_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT co.uri, co.storage_type, a.type, co.extdata "
                                 "FROM rocpd_info_code_object_" +
                                     m_uuid +
                                     " co "
                                     "JOIN rocpd_info_agent_" +
                                     m_uuid + " a ON co.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "www.amd.com");
    EXPECT_EQ(result.rows[0][1], "FILE");
    EXPECT_EQ(result.rows[0][2], "GPU");
    EXPECT_EQ(result.rows[0][3], R"({"isa":"gfx1100","format":"elf"})");
}

// ============================================================================
// 11. rocpd_info_kernel_symbol - FK to code_object, node, process
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, info_kernel_symbol_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_kernel_symbol_info(create_kernel_symbol_info(1, 1, 1000, 1));
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT ks.kernel_name, ks.display_name, ks.sgpr_count, ks.arch_vgpr_count, "
        "co.uri, n.hostname, p.pid, ks.extdata "
        "FROM rocpd_info_kernel_symbol_" +
            m_uuid +
            " ks "
            "JOIN rocpd_info_code_object_" +
            m_uuid +
            " co ON ks.code_object_id = co.id "
            "JOIN rocpd_info_node_" +
            m_uuid +
            " n ON ks.nid = n.id "
            "JOIN rocpd_info_process_" +
            m_uuid + " p ON ks.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "vector_add");
    EXPECT_EQ(result.rows[0][1], "VectorAdd Kernel");
    EXPECT_EQ(result.rows[0][2], "32");
    EXPECT_EQ(result.rows[0][3], "64");
    EXPECT_EQ(result.rows[0][4], "www.amd.com");
    EXPECT_EQ(result.rows[0][5], "test-host");
    EXPECT_EQ(result.rows[0][6], "1000");
    EXPECT_EQ(result.rows[0][7], R"({"lds_size":256,"waves_per_eu":4})");
}

// ============================================================================
// 12. rocpd_track - FK to node, process, thread, string
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, track_values_and_fks)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_thread_info(create_thread_info(1, 1000, 100));
    m_writer->register_track_info(create_track_info(1, 1000, 100, "MyTrack"));
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT tr.nid, s.string, n.machine_id, tr.extdata "
                                 "FROM rocpd_track_" +
                                     m_uuid +
                                     " tr "
                                     "JOIN rocpd_string_" +
                                     m_uuid +
                                     " s ON tr.name_id = s.id "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid + " n ON tr.nid = n.id");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][1], "MyTrack");
    EXPECT_EQ(result.rows[0][2], "test-machine-1");
    EXPECT_EQ(result.rows[0][3], R"({"source":"profiler","kind":"gpu"})");
}

// ============================================================================
// 13. rocpd_event - V3 has call_stack and line_info JSONB columns
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, event_table_via_region)
{
    register_base_entities();

    auto region = create_region_data("test_api_call", 1000000, 2000000);
    m_writer->insert_region_data(region, create_trace_env());
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT stack_id, correlation_id, extdata FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "1");
    EXPECT_EQ(result.rows[0][1], "1");
    EXPECT_EQ(result.rows[0][2], R"({"origin":"api","depth":0})");
}

// ============================================================================
// 14. rocpd_arg - FK to event, populated via region with args
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, arg_table_populated_via_region)
{
    register_base_entities();

    rocpdsna::writer_types::region_data_t region{
        .event =
            rocpdsna::writer_types::event_data_t{ .stack_id        = 1,
                                                  .parent_stack_id = 0,
                                                  .correlation_id  = 42,
                                                  .call_stack      = {},
                                                  .line_info_list  = {},
                                                  .event_category  = "GENERAL",
                                                  .extdata =
                                                      R"({"origin":"api","depth":0})" },
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "hipMalloc",
        .extdata         = R"({"scope":"device","async":false})",
        .args = { { .position = 0, .type = "void**", .name = "ptr", .value = "0x7f00" },
                  { .position = 1, .type = "size_t", .name = "size", .value = "1024" } }
    };

    m_writer->insert_region_data(region, create_trace_env());
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT a.position, a.type, a.name, a.value "
                                 "FROM rocpd_arg_" +
                                     m_uuid + " a ORDER BY a.position");

    ASSERT_GE(result.rows.size(), 2u);
    EXPECT_EQ(result.rows[0][0], "0");
    EXPECT_EQ(result.rows[0][1], "void**");
    EXPECT_EQ(result.rows[0][2], "ptr");
    EXPECT_EQ(result.rows[0][3], "0x7f00");
    EXPECT_EQ(result.rows[1][0], "1");
    EXPECT_EQ(result.rows[1][1], "size_t");
    EXPECT_EQ(result.rows[1][2], "size");
    EXPECT_EQ(result.rows[1][3], "1024");
}

// ============================================================================
// 15. rocpd_pmc_event - FK to pmc and event
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, pmc_event_table_populated)
{
    register_base_entities();

    m_writer->register_pmc_info(create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 }));

    rocpdsna::writer_types::pmc_event_data_t pmc_event{
        .event =
            rocpdsna::writer_types::event_data_t{ .stack_id        = 1,
                                                  .parent_stack_id = 0,
                                                  .correlation_id  = 1,
                                                  .call_stack      = {},
                                                  .line_info_list  = {},
                                                  .event_category  = "GENERAL",
                                                  .extdata =
                                                      R"({"origin":"pmc","depth":0})" },
        .value   = 42.5,
        .extdata = R"({"unit":"count","normalized":false})",
        .sample  = { .timestamp = 5000000,
                     .track =
                         rocpdsna::writer_types::track_info_t{
                             .name       = "TestTrack",
                             .extdata    = R"({"source":"profiler","kind":"gpu"})",
                             .node_id    = 1,
                             .process_id = 1000,
                             .thread_id  = 100 },
                     .extdata = R"({"interval_ms":10})" }
    };

    rocpdsna::writer_types::pmc_info_unique_id_t pmc_uid{
        .name     = "SQ_WAVES",
        .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 }
    };

    m_writer->insert_pmc_event_data(pmc_event, pmc_uid);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT pe.value, pm.name, pe.extdata "
                                 "FROM rocpd_pmc_event_" +
                                     m_uuid +
                                     " pe "
                                     "JOIN rocpd_info_pmc_" +
                                     m_uuid + " pm ON pe.pmc_id = pm.id");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "42.5");
    EXPECT_EQ(result.rows[0][1], "SQ_WAVES");
    EXPECT_EQ(result.rows[0][2], R"({"unit":"count","normalized":false})");
}

// ============================================================================
// 16. rocpd_region - V3 has direct start/end BIGINT, nid/pid/tid, name_id FK
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, region_values_with_direct_timestamps)
{
    register_base_entities();

    auto region = create_region_data("test_function", 1000000, 2000000);
    m_writer->insert_region_data(region, create_trace_env());
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT r.start, r.end, s.string "
                                 "FROM rocpd_region_" +
                                     m_uuid +
                                     " r "
                                     "JOIN rocpd_string_" +
                                     m_uuid + " s ON r.name_id = s.id");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "1000000");
    EXPECT_EQ(result.rows[0][1], "2000000");
    EXPECT_EQ(result.rows[0][2], "test_function");
}

// ============================================================================
// 17. rocpd_sample - FK to track, direct timestamp BIGINT
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, sample_table_populated_via_region)
{
    register_base_entities();

    auto region = create_region_data("test_fn", 1000000, 2000000);
    m_writer->insert_region_data(region, create_trace_env());
    flush_and_wait();

    auto cnt = count_rows(m_db_path, "rocpd_sample", m_uuid);
    EXPECT_GE(cnt, 1u);

    auto result = query_database(
        m_db_path, "SELECT sa.timestamp FROM rocpd_sample_" + m_uuid + " sa");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_NE(result.rows[0][0], "NULL");
}

// ============================================================================
// 18. rocpd_kernel_dispatch - V3 has direct
// nid/pid/tid/agent_id/queue_id/stream_id/start/end
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, kernel_dispatch_full_chain)
{
    register_base_entities();

    m_writer->register_agent_info(create_agent_info(1, 1000, "CPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_kernel_symbol_info(create_kernel_symbol_info(1, 1, 1000, 1));

    auto dispatch = create_kernel_dispatch_data(1, 1, 1, 2000000, 3000000);
    m_writer->insert_kernel_dispatch_data(dispatch, create_trace_env());
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT kd.dispatch_id, kd.workgroup_size_x, kd.grid_size_x, "
                       "kd.start, kd.end, ks.kernel_name "
                       "FROM rocpd_kernel_dispatch_" +
                           m_uuid +
                           " kd "
                           "JOIN rocpd_info_kernel_symbol_" +
                           m_uuid + " ks ON kd.kernel_id = ks.id");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "1");
    EXPECT_EQ(result.rows[0][1], "64");
    EXPECT_EQ(result.rows[0][2], "1024");
    EXPECT_EQ(result.rows[0][3], "2000000");
    EXPECT_EQ(result.rows[0][4], "3000000");
    EXPECT_EQ(result.rows[0][5], "vector_add");
}

// ============================================================================
// 19. rocpd_memory_copy - V3 has direct nid/pid/tid/start/end, agent FKs
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, memory_copy_values_and_fks)
{
    register_base_entities();

    m_writer->register_agent_info(create_agent_info(1, 1000, "CPU", 0));

    auto copy         = create_memory_copy_data(3000000, 3500000);
    copy.src_agent_id = rocpdsna::writer_types::agent_unique_id_t{ "CPU", 0 };
    copy.dst_agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 };

    m_writer->insert_memory_copy_data(copy, create_trace_env());
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT mc.size, mc.start, mc.end, "
                                 "src_a.type as src_type, dst_a.type as dst_type "
                                 "FROM rocpd_memory_copy_" +
                                     m_uuid +
                                     " mc "
                                     "LEFT JOIN rocpd_info_agent_" +
                                     m_uuid +
                                     " src_a ON mc.src_agent_id = src_a.id "
                                     "LEFT JOIN rocpd_info_agent_" +
                                     m_uuid + " dst_a ON mc.dst_agent_id = dst_a.id");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "1048576");
    EXPECT_EQ(result.rows[0][1], "3000000");
    EXPECT_EQ(result.rows[0][2], "3500000");
    EXPECT_EQ(result.rows[0][3], "CPU");
    EXPECT_EQ(result.rows[0][4], "GPU");
}

// ============================================================================
// 20. rocpd_memory_allocate - V3 has direct nid/pid/tid/agent_id/start/end
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, memory_allocate_values)
{
    register_base_entities();

    auto alloc = create_memory_alloc_data(500000, 600000);
    m_writer->insert_memory_alloc_data(alloc, create_trace_env());
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT ma.type, ma.level, ma.size, ma.start, ma.end "
                                 "FROM rocpd_memory_allocate_" +
                                     m_uuid + " ma");

    ASSERT_GE(result.rows.size(), 1u);
    EXPECT_EQ(result.rows[0][0], "ALLOC");
    EXPECT_EQ(result.rows[0][1], "REAL");
    EXPECT_EQ(result.rows[0][2], "1048576");
    EXPECT_EQ(result.rows[0][3], "500000");
    EXPECT_EQ(result.rows[0][4], "600000");
}

// ============================================================================
// FK Integrity Tests - Missing dependencies throw
// ============================================================================

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_missing_node_throws)
{
    auto process = create_process_info(999, 1000);
    EXPECT_THROW(m_writer->register_process_info(process), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto thread = create_thread_info(1, 999, 100);
    EXPECT_THROW(m_writer->register_thread_info(thread), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_thread_missing_node_throws)
{
    auto thread = create_thread_info(999, 1000, 100);
    EXPECT_THROW(m_writer->register_thread_info(thread), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_agent_missing_node_throws)
{
    auto agent = create_agent_info(999, 1000, "GPU", 0);
    EXPECT_THROW(m_writer->register_agent_info(agent), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_agent_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto agent = create_agent_info(1, 999, "GPU", 0);
    EXPECT_THROW(m_writer->register_agent_info(agent), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_queue_missing_node_throws)
{
    auto queue = create_queue_info(999, 1000, 1);
    EXPECT_THROW(m_writer->register_queue_info(queue), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_queue_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto queue = create_queue_info(1, 999, 1);
    EXPECT_THROW(m_writer->register_queue_info(queue), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_stream_missing_node_throws)
{
    auto stream = create_stream_info(999, 1000, 1);
    EXPECT_THROW(m_writer->register_stream_info(stream), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_stream_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto stream = create_stream_info(1, 999, 1);
    EXPECT_THROW(m_writer->register_stream_info(stream), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_pmc_missing_node_throws)
{
    auto pmc = create_pmc_info(999, 1000, "SQ_WAVES");
    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_pmc_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto pmc = create_pmc_info(1, 999, "SQ_WAVES");
    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_code_object_missing_node_throws)
{
    auto code_obj = create_code_object_info(1, 999, 1000, { "GPU", 0 });
    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_code_object_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto code_obj = create_code_object_info(1, 1, 999, { "GPU", 0 });
    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_code_object_missing_agent_throws)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 99 });
    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_kernel_symbol_missing_node_throws)
{
    auto kernel = create_kernel_symbol_info(1, 999, 1000, 1);
    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_kernel_symbol_missing_process_throws)
{
    m_writer->register_node_info(create_node_info(1));
    auto kernel = create_kernel_symbol_info(1, 1, 999, 1);
    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

TEST_F(SchemaV3ValueInsertionTest, fk_integrity_kernel_symbol_missing_code_object_throws)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    auto kernel = create_kernel_symbol_info(1, 1, 1000, 999);
    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

// ============================================================================
// Duplicate Registration Tests
// ============================================================================

TEST_F(SchemaV3ValueInsertionTest, duplicate_node_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_node_info(create_node_info(1));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_node", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_process_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_process_info(create_process_info(1, 1000));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_process", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_thread_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_thread_info(create_thread_info(1, 1000, 100));
    m_writer->register_thread_info(create_thread_info(1, 1000, 100));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_thread", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_agent_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_agent", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_queue_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_queue_info(create_queue_info(1, 1000, 1));
    m_writer->register_queue_info(create_queue_info(1, 1000, 1));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_queue", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_stream_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_stream_info(create_stream_info(1, 1000, 1));
    m_writer->register_stream_info(create_stream_info(1, 1000, 1));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_stream", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_pmc_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);
    m_writer->register_pmc_info(pmc);
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_pmc", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_code_object_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_code_object", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_kernel_symbol_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_agent_info(create_agent_info(1, 1000, "GPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_kernel_symbol_info(create_kernel_symbol_info(1, 1, 1000, 1));
    m_writer->register_kernel_symbol_info(create_kernel_symbol_info(1, 1, 1000, 1));
    flush_and_wait();
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_kernel_symbol", m_uuid), 1u);
}

TEST_F(SchemaV3ValueInsertionTest, duplicate_track_registration_ignored)
{
    m_writer->register_node_info(create_node_info(1));
    m_writer->register_process_info(create_process_info(1, 1000));
    m_writer->register_thread_info(create_thread_info(1, 1000, 100));
    m_writer->register_track_info(create_track_info(1, 1000, 100, "MyTrack"));
    m_writer->register_track_info(create_track_info(1, 1000, 100, "MyTrack"));
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT COUNT(*) FROM rocpd_track_" + m_uuid +
                                     " tr "
                                     "JOIN rocpd_string_" +
                                     m_uuid +
                                     " s ON tr.name_id = s.id "
                                     "WHERE s.string = 'MyTrack'");
    EXPECT_EQ(result.rows[0][0], "1");
}

// ============================================================================
// V3-specific: V4-only tables should NOT exist
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, v4_only_tables_absent)
{
    flush_and_wait();

    auto check_absent = [&](const std::string& table) {
        sqlite3* db = nullptr;
        sqlite3_open(m_db_path.c_str(), &db);
        std::string query =
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?";
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table.c_str(), -1, SQLITE_STATIC);
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return exists;
    };

    EXPECT_FALSE(check_absent("rocpd_timestamp_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_call_stack_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_line_info_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_info_category_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_info_source_code_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_info_pc_" + m_uuid));
    EXPECT_FALSE(check_absent("rocpd_info_address_range_" + m_uuid));
}

// ============================================================================
// End-to-end: All V3 tables populated in single flow
// ============================================================================
TEST_F(SchemaV3ValueInsertionTest, end_to_end_all_tables)
{
    register_base_entities();

    m_writer->register_agent_info(create_agent_info(1, 1000, "CPU", 0));
    m_writer->register_code_object_info(
        create_code_object_info(1, 1, 1000, { "GPU", 0 }));
    m_writer->register_kernel_symbol_info(create_kernel_symbol_info(1, 1, 1000, 1));
    m_writer->register_pmc_info(create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 }));

    auto env = create_trace_env();

    auto region = create_region_data("test_fn", 1000000, 2000000);
    m_writer->insert_region_data(region, env);

    auto dispatch = create_kernel_dispatch_data(1, 1, 1, 2000000, 3000000);
    m_writer->insert_kernel_dispatch_data(dispatch, env);

    auto copy = create_memory_copy_data(3000000, 3500000);
    m_writer->insert_memory_copy_data(copy, env);

    auto alloc = create_memory_alloc_data(500000, 600000);
    m_writer->insert_memory_alloc_data(alloc, env);

    rocpdsna::writer_types::pmc_event_data_t pmc_event{
        .event =
            rocpdsna::writer_types::event_data_t{ .stack_id        = 1,
                                                  .parent_stack_id = 0,
                                                  .correlation_id  = 1,
                                                  .call_stack      = {},
                                                  .line_info_list  = {},
                                                  .event_category  = "GENERAL",
                                                  .extdata =
                                                      R"({"origin":"pmc","depth":0})" },
        .value   = 100.0,
        .extdata = R"({"unit":"count","normalized":false})",
        .sample  = { .timestamp = 5000000,
                     .track =
                         rocpdsna::writer_types::track_info_t{
                             .name       = "TestTrack",
                             .extdata    = R"({"source":"profiler","kind":"gpu"})",
                             .node_id    = 1,
                             .process_id = 1000,
                             .thread_id  = 100 },
                     .extdata = R"({"interval_ms":10})" }
    };

    m_writer->insert_pmc_event_data(
        pmc_event,
        { .name     = "SQ_WAVES",
          .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });

    flush_and_wait();

    EXPECT_GE(count_rows(m_db_path, "rocpd_metadata", m_uuid), 3u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_string", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_node", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_process", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_thread", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_agent", m_uuid), 2u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_queue", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_stream", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_pmc", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_code_object", m_uuid), 1u);
    EXPECT_EQ(count_rows(m_db_path, "rocpd_info_kernel_symbol", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_track", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_event", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_region", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_sample", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_kernel_dispatch", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_memory_copy", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_memory_allocate", m_uuid), 1u);
    EXPECT_GE(count_rows(m_db_path, "rocpd_pmc_event", m_uuid), 1u);
}
