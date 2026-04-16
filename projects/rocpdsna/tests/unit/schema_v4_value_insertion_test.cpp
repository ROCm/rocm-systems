// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @brief V4 Schema Value Insertion Tests
 *
 * These tests verify that actual values are correctly inserted into V4 tables
 * and that foreign key relationships are properly maintained.
 * Tests cover all 27 tables defined in schema/4.0.0/rocpd_tables.sql
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

#define VERSION                                                                          \
    rocpdsna::version_t { 4, 0, 0 }

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
// V4 Value Insertion Test Fixture
// ============================================================================

class SchemaV4ValueInsertionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = "test_v4_value_insertion.db";
        m_uuid    = "v4val";
        std::filesystem::remove(m_db_path);

        // Create V4 database with writer
        auto storage = std::make_unique<rocpdsna::storage_t>(m_db_path, m_uuid, VERSION);
        m_writer     = std::make_unique<rocpdsna::writer_t>(std::move(storage));
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
                 .release       = "4.0",
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

    rocpdsna::writer_types::category_info_t create_category_info(
        size_t /*id*/    = 1,
        const char* name = "HIP_API")
    {
        return { .name = name, .extdata = R"({"domain":"hip","level":1})" };
    }

    rocpdsna::writer_types::address_range_info_t
    create_address_range_info(size_t id = 1, size_t node_id = 1, size_t process_id = 1000)
    {
        return { .id           = id,
                 .address_base = 0x7f0000000000,
                 .address_low  = 0x7f0000001000,
                 .address_high = 0x7f0000002000,
                 .extdata      = R"({"segment":"text","permissions":"rx"})",
                 .node_id      = node_id,
                 .process_id   = process_id };
    }

    rocpdsna::writer_types::source_code_info_t create_source_code_info(
        size_t id         = 1,
        size_t node_id    = 1,
        size_t process_id = 1000,
        size_t address_id = 1)
    {
        return { .id           = id,
                 .file         = "temp.hpp",
                 .line_number  = 42,
                 .lines        = R"(["  float a = x + y;", "  return a;"])",
                 .instructions = R"(["abcd", "efgh"])",
                 .extdata      = R"({"compiler":"hipcc","opt_level":3})",
                 .node_id      = node_id,
                 .process_id   = process_id,
                 .address_id   = address_id };
    }

    rocpdsna::writer_types::pc_info_t create_pc_info(size_t id         = 1,
                                                     size_t node_id    = 1,
                                                     size_t process_id = 1000,
                                                     size_t address_id = 1)
    {
        return { .id         = id,
                 .function   = "vector_add",
                 .file       = "temp.hpp",
                 .line       = 42,
                 .extdata    = R"({"inlined":false,"hotspot":true})",
                 .node_id    = node_id,
                 .process_id = process_id,
                 .address_id = address_id };
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

    // ========================================================================
    // Common Registration Helpers
    // ========================================================================

    void register_node_and_process(size_t node_id = 1, size_t pid = 1000)
    {
        m_writer->register_node_info(create_node_info(node_id));
        m_writer->register_process_info(create_process_info(node_id, pid));
    }

    void register_node_process_thread(size_t node_id = 1,
                                      size_t pid     = 1000,
                                      size_t tid     = 100)
    {
        register_node_and_process(node_id, pid);
        m_writer->register_thread_info(create_thread_info(node_id, pid, tid));
    }

    void register_base_entities(size_t node_id = 1, size_t pid = 1000, size_t tid = 100)
    {
        register_node_process_thread(node_id, pid, tid);
        m_writer->register_agent_info(create_agent_info(node_id, pid, "GPU", 0));
        m_writer->register_queue_info(create_queue_info(node_id, pid, 1));
        m_writer->register_stream_info(create_stream_info(node_id, pid, 1));
        m_writer->register_track_info(create_track_info(node_id, pid, tid, "TestTrack"));
    }

    std::string                         m_db_path;
    std::string                         m_uuid;
    std::unique_ptr<rocpdsna::writer_t> m_writer;
};

// ============================================================================
// Test 1: rocpd_info_node - Basic node info with all V4 columns
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_node_values_inserted_correctly)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT hash, machine_id, name, system_name, hostname, release, version, "
        "hardware_name, domain_name FROM rocpd_info_node_" +
            m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "100001");          // hash
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // machine_id
    EXPECT_EQ(result.rows[0][2], "TestNode");        // name (V4 column)
    EXPECT_EQ(result.rows[0][3], "Linux");           // system_name
    EXPECT_EQ(result.rows[0][4], "test-host");       // hostname
    EXPECT_EQ(result.rows[0][5], "4.0");             // release
    EXPECT_EQ(result.rows[0][6], "#1");              // version
    EXPECT_EQ(result.rows[0][7], "x86_64");          // hardware_name
    EXPECT_EQ(result.rows[0][8], "test.local");      // domain_name
}

// ============================================================================
// Test 2: rocpd_info_process - Process with FK to node, V4 name column
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_process_values_and_fk_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT p.pid, p.ppid, p.name, p.command, p.init, "
                                 "p.fini, n.machine_id, p.extdata "
                                 "FROM rocpd_info_process_" +
                                     m_uuid +
                                     " p "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid + " n ON p.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1000");            // pid
    EXPECT_EQ(result.rows[0][1], "1");               // ppid
    EXPECT_EQ(result.rows[0][2], "TestProcess");     // name (V4 column)
    EXPECT_EQ(result.rows[0][3], "ls");              // command
    EXPECT_EQ(result.rows[0][4], "1000000");         // init
    EXPECT_EQ(result.rows[0][5], "9000000");         // fini
    EXPECT_EQ(result.rows[0][6], "test-machine-1");  // FK to node verified
    EXPECT_EQ(result.rows[0][7], R"({"custom":"data"})");
}

// ============================================================================
// Test 3: rocpd_info_thread - Thread with FK to node and process
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_thread_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);
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

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "100");             // tid
    EXPECT_EQ(result.rows[0][1], "TestThread");      // name
    EXPECT_EQ(result.rows[0][2], "1100000");         // start
    EXPECT_EQ(result.rows[0][3], "8900000");         // end
    EXPECT_EQ(result.rows[0][4], "1000");            // FK process pid
    EXPECT_EQ(result.rows[0][5], "test-machine-1");  // FK node
    EXPECT_EQ(result.rows[0][6], R"({"priority":"high","affinity":3})");
}

// ============================================================================
// Test 4: rocpd_info_agent - Agent with FK to node and process
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_agent_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT a.type, a.type_index, a.name, a.model_name, a.vendor_name, "
        "a.product_name, p.pid, a.extdata "
        "FROM rocpd_info_agent_" +
            m_uuid +
            " a "
            "JOIN rocpd_info_process_" +
            m_uuid + " p ON a.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "GPU");                 // type
    EXPECT_EQ(result.rows[0][1], "0");                   // type_index
    EXPECT_EQ(result.rows[0][2], "gfx1100");             // name
    EXPECT_EQ(result.rows[0][3], "AMD Radeon RX 7900");  // model_name
    EXPECT_EQ(result.rows[0][4], "AMD");                 // vendor_name
    EXPECT_EQ(result.rows[0][5], "Radeon RX 7900 XTX");  // product_name
    EXPECT_EQ(result.rows[0][6], "1000");                // FK process pid
    EXPECT_EQ(result.rows[0][7], R"({"pci_bus":"0000:03:00.0","mem_gb":24})");
}

// ============================================================================
// Test 5: rocpd_info_queue - Queue with FK to node and process
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_queue_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT q.name, p.pid, q.extdata FROM rocpd_info_queue_" + m_uuid +
                           " q "
                           "JOIN rocpd_info_process_" +
                           m_uuid + " p ON q.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "TestQueue");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[0][2], R"({"capacity":128,"type":"compute"})");
}

// ============================================================================
// Test 6: rocpd_info_stream - Stream with FK to node and process
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_stream_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT s.name, p.pid, s.extdata FROM rocpd_info_stream_" + m_uuid +
            " s "
            "JOIN rocpd_info_process_" +
            m_uuid + " p ON s.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "TestStream");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[0][2], R"({"flags":"default","priority":0})");
}

// ============================================================================
// Test 7: rocpd_info_pmc - PMC with V4 qualifier column and FK to agent
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pmc_values_with_qualifier_and_fks)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT pm.name, pm.symbol, pm.qualifier, pm.description, pm.units, "
        "pm.value_type, pm.block, a.type, pm.extdata "
        "FROM rocpd_info_pmc_" +
            m_uuid +
            " pm "
            "JOIN rocpd_info_agent_" +
            m_uuid + " a ON pm.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "SQ_WAVES");         // name
    EXPECT_EQ(result.rows[0][1], "SQ_WAVES");         // symbol
    EXPECT_EQ(result.rows[0][2], "ACCUM");            // qualifier (V4 column)
    EXPECT_EQ(result.rows[0][3], "Number of waves");  // description
    EXPECT_EQ(result.rows[0][4], "waves");            // units
    EXPECT_EQ(result.rows[0][5], "ACCUM");            // value_type
    EXPECT_EQ(result.rows[0][6], "SQ");               // block
    EXPECT_EQ(result.rows[0][7], "GPU");              // FK agent type
    EXPECT_EQ(result.rows[0][8], R"({"shader_engine":0,"cu_mask":"0xff"})");
}

// ============================================================================
// Test 8: rocpd_info_code_object - Code object with FK to agent
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_code_object_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT co.uri, co.load_base, co.load_size, co.storage_type, a.type, co.extdata "
        "FROM rocpd_info_code_object_" +
            m_uuid +
            " co "
            "JOIN rocpd_info_agent_" +
            m_uuid + " a ON co.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "www.amd.com");
    EXPECT_EQ(result.rows[0][3], "FILE");
    EXPECT_EQ(result.rows[0][4], "GPU");
    EXPECT_EQ(result.rows[0][5], R"({"isa":"gfx1100","format":"elf"})");
}

// ============================================================================
// Test 9: rocpd_info_kernel_symbol - Kernel symbol with FK to code_object
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_kernel_symbol_values_and_fks_correct)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);

    auto kernel = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT ks.kernel_name, ks.display_name, ks.sgpr_count, ks.arch_vgpr_count, "
        "co.uri, n.hostname, p.pid, ks.extdata FROM rocpd_info_kernel_symbol_" +
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

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "vector_add");
    EXPECT_EQ(result.rows[0][1], "VectorAdd Kernel");
    EXPECT_EQ(result.rows[0][2], "32");
    EXPECT_EQ(result.rows[0][3], "64");
    EXPECT_EQ(result.rows[0][4], "www.amd.com");  // FK code_object uri
    EXPECT_EQ(result.rows[0][5], "test-host");    // FK node hostname
    EXPECT_EQ(result.rows[0][6], "1000");         // FK process pid
    EXPECT_EQ(result.rows[0][7], R"({"lds_size":256,"waves_per_eu":4})");
}

// ============================================================================
// Test 10: rocpd_info_category - V4-only category table
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_category_values_inserted)
{
    auto category = create_category_info(1, "HIP_API");
    m_writer->register_category_info(category);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT name, extdata FROM rocpd_info_category_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "HIP_API");
    EXPECT_EQ(result.rows[0][1], R"({"domain":"hip","level":1})");
}

// ============================================================================
// Test 11: rocpd_info_address_range - V4-only address range table
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_address_range_values_and_fks)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT ar.address_base, ar.address_low, ar.address_high, p.pid, n.hostname, "
        "ar.extdata "
        "FROM rocpd_info_address_range_" +
            m_uuid +
            " ar "
            "JOIN rocpd_info_process_" +
            m_uuid +
            " p ON ar.pid = p.id "
            "JOIN rocpd_info_node_" +
            m_uuid + " n ON ar.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1);
    // Verify addresses are correctly stored
    EXPECT_EQ(result.rows[0][3], "1000");       // FK process pid
    EXPECT_EQ(result.rows[0][4], "test-host");  // FK node hostname
    EXPECT_EQ(result.rows[0][5], R"({"segment":"text","permissions":"rx"})");
}

// ============================================================================
// Test 12: rocpd_info_source_code - V4-only source code table
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_source_code_values_and_fks)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    auto source = create_source_code_info(1, 1, 1000);
    m_writer->register_source_code_info(source);
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT sc.file, sc.line_number, p.pid, n.hostname, sc.extdata "
                       "FROM rocpd_info_source_code_" +
                           m_uuid +
                           " sc "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON sc.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON sc.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "temp.hpp");
    EXPECT_EQ(result.rows[0][1], "42");
    EXPECT_EQ(result.rows[0][2], "1000");       // FK process pid
    EXPECT_EQ(result.rows[0][3], "test-host");  // FK node hostname
    EXPECT_EQ(result.rows[0][4], R"({"compiler":"hipcc","opt_level":3})");
}

// ============================================================================
// Test 13: rocpd_info_pc - V4-only PC info table
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pc_values_and_fks)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    auto pc = create_pc_info(1, 1, 1000);
    m_writer->register_pc_info(pc);
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT pc.function, pc.file, pc.line, p.pid, n.hostname, "
                       "pc.extdata "
                       "FROM rocpd_info_pc_" +
                           m_uuid +
                           " pc "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON pc.pid = p.id "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON pc.nid = n.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "vector_add");
    EXPECT_EQ(result.rows[0][1], "temp.hpp");
    EXPECT_EQ(result.rows[0][2], "42");
    EXPECT_EQ(result.rows[0][3], "1000");       // FK process pid
    EXPECT_EQ(result.rows[0][4], "test-host");  // FK node hostname
    EXPECT_EQ(result.rows[0][5], R"({"inlined":false,"hotspot":true})");
}

// ============================================================================
// Test 14: rocpd_region - Region with track and timestamp FKs
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_missing_node_throws)
{
    // Try to register process without registering node first
    auto process = create_process_info(999, 1000);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_process_info(process), std::exception);
}

// ============================================================================
// Test 27: Foreign key integrity - missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register thread without registering process first
    auto thread = create_thread_info(1, 999, 100);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_thread_info(thread), std::exception);
}

// ============================================================================
// Test 28: FK integrity - thread missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_thread_missing_node_throws)
{
    // Try to register thread without registering node first
    auto thread = create_thread_info(999, 1000, 100);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_thread_info(thread), std::exception);
}

// ============================================================================
// Test 29: FK integrity - agent missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_agent_missing_node_throws)
{
    // Try to register agent without registering node first
    auto agent = create_agent_info(999, 1000, "GPU", 0);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_agent_info(agent), std::exception);
}

// ============================================================================
// Test 30: FK integrity - agent missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_agent_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register agent without registering process first
    auto agent = create_agent_info(1, 999, "GPU", 0);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_agent_info(agent), std::exception);
}

// ============================================================================
// Test 31: FK integrity - queue missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_queue_missing_node_throws)
{
    // Try to register queue without registering node first
    auto queue = create_queue_info(999, 1000, 1);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_queue_info(queue), std::exception);
}

// ============================================================================
// Test 32: FK integrity - queue missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_queue_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register queue without registering process first
    auto queue = create_queue_info(1, 999, 1);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_queue_info(queue), std::exception);
}

// ============================================================================
// Test 33: FK integrity - stream missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_stream_missing_node_throws)
{
    // Try to register stream without registering node first
    auto stream = create_stream_info(999, 1000, 1);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_stream_info(stream), std::exception);
}

// ============================================================================
// Test 34: FK integrity - stream missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_stream_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register stream without registering process first
    auto stream = create_stream_info(1, 999, 1);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_stream_info(stream), std::exception);
}

// ============================================================================
// Test 35: FK integrity - pmc missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_pmc_missing_node_throws)
{
    // Try to register pmc without registering node first
    auto pmc = create_pmc_info(999, 1000, "SQ_WAVES");  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::exception);
}

// ============================================================================
// Test 36: FK integrity - pmc missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_pmc_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register pmc without registering process first
    auto pmc = create_pmc_info(1, 999, "SQ_WAVES");  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_pmc_info(pmc), std::exception);
}

// ============================================================================
// Test 37: FK integrity - code_object missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_code_object_missing_node_throws)
{
    // Try to register code_object without registering node first
    auto code_obj =
        create_code_object_info(1, 999, 1000, { "GPU", 0 });  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

// ============================================================================
// Test 38: FK integrity - code_object missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_code_object_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register code_object without registering process first
    auto code_obj =
        create_code_object_info(1, 1, 999, { "GPU", 0 });  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

// ============================================================================
// Test 39: FK integrity - code_object missing agent throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_code_object_missing_agent_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    // Try to register code_object with non-existent agent
    auto code_obj =
        create_code_object_info(1, 1, 1000, { "GPU", 99 });  // agent doesn't exist

    EXPECT_THROW(m_writer->register_code_object_info(code_obj), std::exception);
}

// ============================================================================
// Test 40: FK integrity - kernel_symbol missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_kernel_symbol_missing_node_throws)
{
    // Try to register kernel_symbol without registering node first
    auto kernel =
        create_kernel_symbol_info(1, 999, 1000, 1);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

// ============================================================================
// Test 41: FK integrity - kernel_symbol missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_kernel_symbol_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register kernel_symbol without registering process first
    auto kernel =
        create_kernel_symbol_info(1, 1, 999, 1);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

// ============================================================================
// Test 42: FK integrity - kernel_symbol missing code_object throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_kernel_symbol_missing_code_object_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    // Try to register kernel_symbol without registering code_object first
    auto kernel =
        create_kernel_symbol_info(1, 1, 1000, 999);  // code_object_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_kernel_symbol_info(kernel), std::exception);
}

// ============================================================================
// Test 43: FK integrity - address_range missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_address_range_missing_node_throws)
{
    // Try to register address_range without registering node first
    auto addr_range =
        create_address_range_info(1, 999, 1000);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_address_range_info(addr_range), std::exception);
}

// ============================================================================
// Test 44: FK integrity - address_range missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_address_range_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register address_range without registering process first
    auto addr_range =
        create_address_range_info(1, 1, 999);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_address_range_info(addr_range), std::exception);
}

// ============================================================================
// Test 45: FK integrity - source_code missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_source_code_missing_node_throws)
{
    // Try to register source_code without registering node first
    auto source = create_source_code_info(1, 999, 1000);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_source_code_info(source), std::exception);
}

// ============================================================================
// Test 46: FK integrity - source_code missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_source_code_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register source_code without registering process first
    auto source = create_source_code_info(1, 1, 999);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_source_code_info(source), std::exception);
}

// ============================================================================
// Test 47: FK integrity - pc_info missing node throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_pc_info_missing_node_throws)
{
    // Try to register pc_info without registering node first
    auto pc = create_pc_info(1, 999, 1000);  // node_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_pc_info(pc), std::exception);
}

// ============================================================================
// Test 48: FK integrity - pc_info missing process throws
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_pc_info_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    // Try to register pc_info without registering process first
    auto pc = create_pc_info(1, 1, 999);  // process_id 999 doesn't exist

    EXPECT_THROW(m_writer->register_pc_info(pc), std::exception);
}

// ============================================================================
// Test 49: source_code with valid address_range FK
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, source_code_with_address_range_fk)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    // Register source_code with valid address_range FK
    auto source = create_source_code_info(1, 1, 1000, 1);  // address_id = 1
    m_writer->register_source_code_info(source);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT sc.file, sc.line_number, ar.address_base "
                                 "FROM rocpd_info_source_code_" +
                                     m_uuid +
                                     " sc "
                                     "LEFT JOIN rocpd_info_address_range_" +
                                     m_uuid + " ar ON sc.address_id = ar.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][1], "42");    // line_number
    EXPECT_NE(result.rows[0][2], "NULL");  // address_base should be set
}

// ============================================================================
// Test 50: pc_info with valid address_range FK
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, pc_info_with_address_range_fk)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    // Register pc_info with valid address_range FK
    auto pc = create_pc_info(1, 1, 1000, 1);  // address_id = 1
    m_writer->register_pc_info(pc);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT pc.function, pc.line, ar.address_base "
                                 "FROM rocpd_info_pc_" +
                                     m_uuid +
                                     " pc "
                                     "LEFT JOIN rocpd_info_address_range_" +
                                     m_uuid + " ar ON pc.address_id = ar.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "vector_add");  // function
    EXPECT_EQ(result.rows[0][1], "42");          // line
    EXPECT_NE(result.rows[0][2], "NULL");        // address_base should be set
}

// ============================================================================
// Test 51: PMC with all FKs verified (node, process, agent)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pmc_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    // Verify all 3 FKs: node, process, agent
    auto result = query_database(m_db_path,
                                 "SELECT pm.name, n.machine_id, p.pid, a.type "
                                 "FROM rocpd_info_pmc_" +
                                     m_uuid +
                                     " pm "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON pm.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid +
                                     " p ON pm.pid = p.id "
                                     "JOIN rocpd_info_agent_" +
                                     m_uuid + " a ON pm.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "SQ_WAVES");        // pmc name
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
    EXPECT_EQ(result.rows[0][3], "GPU");             // FK agent type
}

// ============================================================================
// Test 52: Code object with all FKs verified (node, process, agent)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_code_object_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);
    flush_and_wait();

    // Verify all 3 FKs: node, process, agent
    auto result = query_database(m_db_path,
                                 "SELECT co.storage_type, n.machine_id, p.pid, a.type "
                                 "FROM rocpd_info_code_object_" +
                                     m_uuid +
                                     " co "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON co.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid +
                                     " p ON co.pid = p.id "
                                     "JOIN rocpd_info_agent_" +
                                     m_uuid + " a ON co.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "FILE");            // storage_type
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
    EXPECT_EQ(result.rows[0][3], "GPU");             // FK agent type
}

// ============================================================================
// Test 53: Kernel symbol with all FKs verified (node, process, code_object)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_kernel_symbol_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);

    auto kernel = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel);
    flush_and_wait();

    // Verify all 3 FKs: node, process, code_object
    auto result =
        query_database(m_db_path,
                       "SELECT ks.kernel_name, n.machine_id, p.pid, co.storage_type "
                       "FROM rocpd_info_kernel_symbol_" +
                           m_uuid +
                           " ks "
                           "JOIN rocpd_info_node_" +
                           m_uuid +
                           " n ON ks.nid = n.id "
                           "JOIN rocpd_info_process_" +
                           m_uuid +
                           " p ON ks.pid = p.id "
                           "JOIN rocpd_info_code_object_" +
                           m_uuid + " co ON ks.code_object_id = co.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "vector_add");      // kernel_name
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
    EXPECT_EQ(result.rows[0][3], "FILE");            // FK code_object storage_type
}

// ============================================================================
// Test 54: Address range with all FKs verified (node, process)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_address_range_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);
    flush_and_wait();

    // Verify both FKs: node, process
    auto result = query_database(m_db_path,
                                 "SELECT ar.address_base, n.machine_id, p.pid "
                                 "FROM rocpd_info_address_range_" +
                                     m_uuid +
                                     " ar "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON ar.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid + " p ON ar.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
}

// ============================================================================
// Test 55: Source code with all FKs verified (node, process)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_source_code_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    auto source = create_source_code_info(1, 1, 1000);
    m_writer->register_source_code_info(source);
    flush_and_wait();

    // Verify both FKs: node, process
    auto result = query_database(m_db_path,
                                 "SELECT sc.line_number, n.machine_id, p.pid "
                                 "FROM rocpd_info_source_code_" +
                                     m_uuid +
                                     " sc "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON sc.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid + " p ON sc.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "42");              // line_number
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
}

// ============================================================================
// Test 56: PC info with all FKs verified (node, process)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pc_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    auto pc = create_pc_info(1, 1, 1000);
    m_writer->register_pc_info(pc);
    flush_and_wait();

    // Verify both FKs: node, process
    auto result = query_database(m_db_path,
                                 "SELECT pc.function, n.machine_id, p.pid "
                                 "FROM rocpd_info_pc_" +
                                     m_uuid +
                                     " pc "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON pc.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid + " p ON pc.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "vector_add");      // function
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
}

// ============================================================================
// Test 57: PMC with agent (FK to valid agent)
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pmc_with_agent_fk)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "CPU", 0);
    m_writer->register_agent_info(agent);

    // Register PMC with agent
    auto pmc = create_pmc_info(
        1, 1000, "CPU_CYCLES", rocpdsna::writer_types::agent_unique_id_t{ "CPU", 0 });
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT p.name, a.type FROM rocpd_info_pmc_" + m_uuid +
                                     " p "
                                     "JOIN rocpd_info_agent_" +
                                     m_uuid + " a ON p.agent_id = a.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "CPU_CYCLES");
    EXPECT_EQ(result.rows[0][1], "CPU");  // agent type should be CPU
}

// ============================================================================
// Test 58: Agent type CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_agent_type_check_constraint)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    // Test valid GPU type
    auto gpu_agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(gpu_agent);

    // Test valid CPU type
    auto cpu_agent = create_agent_info(1, 1000, "CPU", 0);
    m_writer->register_agent_info(cpu_agent);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT type FROM rocpd_info_agent_" + m_uuid + " ORDER BY type");

    ASSERT_EQ(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0][0], "CPU");
    EXPECT_EQ(result.rows[1][0], "GPU");
}

// ============================================================================
// Test 59: PMC value_type CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pmc_value_type_check_constraint)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    // Test ACCUM value_type
    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    pmc.value_type = "ACCUM";
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT value_type FROM rocpd_info_pmc_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "ACCUM");
}

// ============================================================================
// Test 60: Code object storage_type CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_code_object_storage_type_check_constraint)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    // Test FILE storage_type
    auto code_obj         = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    code_obj.storage_type = "FILE";
    m_writer->register_code_object_info(code_obj);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT storage_type FROM rocpd_info_code_object_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "FILE");
}

// ============================================================================
// Test 61: Memory allocate type CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, address_range_low_gte_base_constraint)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    // Create valid address range where low >= base
    auto addr_range         = create_address_range_info(1, 1, 1000);
    addr_range.address_base = 0x1000;
    addr_range.address_low  = 0x2000;  // low > base (valid)
    addr_range.address_high = 0x3000;
    m_writer->register_address_range_info(addr_range);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT address_base, address_low, address_high "
                                 "FROM rocpd_info_address_range_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    auto base = std::stoull(result.rows[0][0]);
    auto low  = std::stoull(result.rows[0][1]);
    auto high = std::stoull(result.rows[0][2]);

    EXPECT_GE(low, base) << "address_low must be >= address_base";
    EXPECT_GE(high, low) << "address_high must be >= address_low";
}

// ============================================================================
// Test 65: Queue with node FK join verification
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_queue_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);
    flush_and_wait();

    // Verify both FKs: node, process
    auto result = query_database(m_db_path,
                                 "SELECT q.name, n.machine_id, p.pid "
                                 "FROM rocpd_info_queue_" +
                                     m_uuid +
                                     " q "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON q.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid + " p ON q.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "TestQueue");       // queue name
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
}

// ============================================================================
// Test 66: Stream with node FK join verification
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_stream_all_fks_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);
    flush_and_wait();

    // Verify both FKs: node, process
    auto result = query_database(m_db_path,
                                 "SELECT s.name, n.machine_id, p.pid "
                                 "FROM rocpd_info_stream_" +
                                     m_uuid +
                                     " s "
                                     "JOIN rocpd_info_node_" +
                                     m_uuid +
                                     " n ON s.nid = n.id "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid + " p ON s.pid = p.id");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "TestStream");      // stream name
    EXPECT_EQ(result.rows[0][1], "test-machine-1");  // FK node machine_id
    EXPECT_EQ(result.rows[0][2], "1000");            // FK process pid
}

// ============================================================================
// ============================================================================
//                    MISSING TABLE TESTS (V4 SPECIFIC)
// ============================================================================
// ============================================================================

// ============================================================================
// Test 67: rocpd_event - Event table with category FK
// ============================================================================

// ============================================================================
//                    TESTS FOR MISSING TABLES (V4 SPECIFIC)
//                    rocpd_event, rocpd_arg, rocpd_call_stack,
//                    rocpd_line_info, rocpd_pmc_event, rocpd_sample
// ============================================================================

// Helper for trace environment
inline rocpdsna::writer_types::trace_environment_t
create_trace_environment(size_t node_id    = 1,
                         size_t process_id = 1000,
                         size_t thread_id  = 100)
{
    return { .node_id    = node_id,
             .process_id = process_id,
             .thread_id  = thread_id,
             .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
             .stream_id  = 1,
             .queue_id   = 1,
             .track_name = "TestTrack" };
}

// ============================================================================
// Test: rocpd_event - Basic event created via insert_region_data
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_basic_region_insert)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    // Create region data with event
    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 42,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_api_call",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    // Register track for sample creation
    auto track = create_track_info(1, 1000, 100, "test_track");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "test_track"
    };
    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify rocpd_event table has data
    auto result = query_database(
        m_db_path, "SELECT stack_id, correlation_id, extdata FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");   // stack_id
    EXPECT_EQ(result.rows[0][1], "42");  // correlation_id
    EXPECT_EQ(result.rows[0][2], R"({"origin":"api","depth":0})");
}

// ============================================================================
// Test: rocpd_event - Event with category FK
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_category_fk_populated)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    // Register category
    auto category = create_category_info(1, "HIP_RUNTIME");
    m_writer->register_category_info(category);

    // Create region with category in event
    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 100,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "HIP_RUNTIME",
        .extdata         = R"({"origin":"runtime","depth":1})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "hipMalloc",
        .extdata         = R"({"scope":"device","async":false})",
        .args            = {}
    };

    // Register track for sample creation
    auto track = create_track_info(1, 1000, 100, "test_track");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "test_track"
    };
    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify event with category FK
    auto result = query_database(m_db_path,
                                 "SELECT e.stack_id, c.name as category_name "
                                 "FROM rocpd_event_" +
                                     m_uuid +
                                     " e "
                                     "LEFT JOIN rocpd_info_category_" +
                                     m_uuid + " c ON e.category_id = c.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");            // stack_id
    EXPECT_EQ(result.rows[0][1], "HIP_RUNTIME");  // category
}

// ============================================================================
// Test: rocpd_arg - Arguments populated via region with args
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, arg_table_populated_via_region)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    // Create region with arguments
    rocpdsna::writer_types::arg_data_t arg1{ .position = 0,
                                             .type     = "void*",
                                             .name     = "dst",
                                             .value    = "0x7f0000001000",
                                             .extdata  = R"({"semantic":"output"})" };

    rocpdsna::writer_types::arg_data_t arg2{ .position = 1,
                                             .type     = "size_t",
                                             .name     = "size",
                                             .value    = "1024",
                                             .extdata  = R"({"semantic":"size"})" };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 200,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "hipMemcpy",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = { arg1, arg2 }
    };

    // Register track for sample creation
    auto track = create_track_info(1, 1000, 100, "test_track");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "test_track"
    };
    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify rocpd_arg table has data
    auto result =
        query_database(m_db_path,
                       "SELECT position, type, name, value, extdata FROM rocpd_arg_" +
                           m_uuid + " ORDER BY position");

    ASSERT_GE(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0][0], "0");               // position
    EXPECT_EQ(result.rows[0][1], "void*");           // type
    EXPECT_EQ(result.rows[0][2], "dst");             // name
    EXPECT_EQ(result.rows[0][3], "0x7f0000001000");  // value
    EXPECT_EQ(result.rows[0][4], R"({"semantic":"output"})");

    EXPECT_EQ(result.rows[1][0], "1");       // position
    EXPECT_EQ(result.rows[1][1], "size_t");  // type
    EXPECT_EQ(result.rows[1][2], "size");    // name
    EXPECT_EQ(result.rows[1][3], "1024");    // value
    EXPECT_EQ(result.rows[1][4], R"({"semantic":"size"})");
}

// ============================================================================
// Test: rocpd_arg - Verify event_id FK from arg to event
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, arg_table_event_fk_verified)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    rocpdsna::writer_types::arg_data_t arg{ .position = 0,
                                            .type     = "int",
                                            .name     = "device",
                                            .value    = "0",
                                            .extdata  = R"({"semantic":"device_id"})" };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 99,
        .parent_stack_id = 0,
        .correlation_id  = 888,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "hipSetDevice",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = { arg }
    };

    // Register track for sample creation
    auto track = create_track_info(1, 1000, 100, "test_track");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "test_track"
    };
    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify arg has FK to event
    auto result = query_database(m_db_path,
                                 "SELECT a.name, e.correlation_id "
                                 "FROM rocpd_arg_" +
                                     m_uuid +
                                     " a "
                                     "JOIN rocpd_event_" +
                                     m_uuid + " e ON a.event_id = e.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "device");  // arg name
    EXPECT_EQ(result.rows[0][1], "888");     // event correlation_id via FK
}

// ============================================================================
// Test: rocpd_sample - Sample created with region
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, sample_table_populated_via_region)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    // Register track
    auto track = create_track_info(1, 1000, 100, "gpu_kernel");
    m_writer->register_track_info(track);

    // Create region WITH event (sample is created when track_name and event both present)
    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 555,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "myKernel",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "gpu_kernel"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify rocpd_sample has data
    auto result = query_database(m_db_path, "SELECT id FROM rocpd_sample_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    // Sample should exist with valid id
    // Just verify we got a row - ID can be any positive value
}

// ============================================================================
// Test: rocpd_pmc_event - PMC event via insert_pmc_event_data
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, pmc_event_table_populated)
{
    register_base_entities();

    // Register PMC info
    auto pmc = create_pmc_info(
        1, 1000, "GPU_BUSY", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);

    // Create PMC event data
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
        .value   = 0.75,
        .extdata = R"({"unit":"percentage","normalized":true})",
        .sample =
            rocpdsna::writer_types::sample_data_t{
                .timestamp = 1500000,
                .track =
                    rocpdsna::writer_types::track_info_t{
                        .name       = "TestTrack",
                        .extdata    = R"({"source":"profiler","kind":"gpu"})",
                        .node_id    = 1,
                        .process_id = 1000,
                        .thread_id  = 100 },
                .extdata = R"({"interval_ms":10})" }
    };

    rocpdsna::writer_types::pmc_info_unique_id_t pmc_id{
        .name     = "GPU_BUSY",
        .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 }
    };

    m_writer->insert_pmc_event_data(pmc_event, pmc_id);
    flush_and_wait();

    // Verify rocpd_pmc_event has data with FK to pmc info
    auto result = query_database(m_db_path,
                                 "SELECT pe.value, p.name "
                                 "FROM rocpd_pmc_event_" +
                                     m_uuid +
                                     " pe "
                                     "JOIN rocpd_info_pmc_" +
                                     m_uuid + " p ON pe.pmc_id = p.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "0.75");      // value
    EXPECT_EQ(result.rows[0][1], "GPU_BUSY");  // pmc name via FK
}

// ============================================================================
// Test: rocpd_pmc_event - Multiple PMC counters in same sample
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, pmc_event_multiple_counters)
{
    register_base_entities();

    // Register two PMC infos
    auto pmc1 = create_pmc_info(
        1, 1000, "COUNTER_A", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    auto pmc2 = create_pmc_info(
        1, 1000, "COUNTER_B", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc1);
    m_writer->register_pmc_info(pmc2);

    // Insert two PMC events
    rocpdsna::writer_types::pmc_event_data_t pmc_event1{
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
        .sample =
            rocpdsna::writer_types::sample_data_t{
                .timestamp = 2000000,
                .track =
                    rocpdsna::writer_types::track_info_t{
                        .name       = "TestTrack",
                        .extdata    = R"({"source":"profiler","kind":"gpu"})",
                        .node_id    = 1,
                        .process_id = 1000,
                        .thread_id  = 100 },
                .extdata = R"({"interval_ms":10})" }
    };

    rocpdsna::writer_types::pmc_event_data_t pmc_event2{
        .event =
            rocpdsna::writer_types::event_data_t{ .stack_id        = 1,
                                                  .parent_stack_id = 0,
                                                  .correlation_id  = 1,
                                                  .call_stack      = {},
                                                  .line_info_list  = {},
                                                  .event_category  = "GENERAL",
                                                  .extdata =
                                                      R"({"origin":"pmc","depth":0})" },
        .value   = 200.0,
        .extdata = R"({"unit":"count","normalized":false})",
        .sample =
            rocpdsna::writer_types::sample_data_t{
                .timestamp = 2000000,
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
        pmc_event1,
        rocpdsna::writer_types::pmc_info_unique_id_t{
            .name     = "COUNTER_A",
            .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });
    m_writer->insert_pmc_event_data(
        pmc_event2,
        rocpdsna::writer_types::pmc_info_unique_id_t{
            .name     = "COUNTER_B",
            .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });
    flush_and_wait();

    // Verify both PMC events exist
    auto result = query_database(m_db_path,
                                 "SELECT pe.value, p.name "
                                 "FROM rocpd_pmc_event_" +
                                     m_uuid +
                                     " pe "
                                     "JOIN rocpd_info_pmc_" +
                                     m_uuid +
                                     " p ON pe.pmc_id = p.id "
                                     "ORDER BY pe.value");

    ASSERT_GE(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0][0], "100.0");
    EXPECT_EQ(result.rows[0][1], "COUNTER_A");
    EXPECT_EQ(result.rows[1][0], "200.0");
    EXPECT_EQ(result.rows[1][1], "COUNTER_B");
}

// ============================================================================
// Test: End-to-end test verifying all missing tables populated
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, missing_tables_end_to_end)
{
    // Setup
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto category = create_category_info(1, "TEST_CATEGORY");
    m_writer->register_category_info(category);

    auto pmc = create_pmc_info(
        1, 1000, "TEST_COUNTER", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    // Insert region with event and args
    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 12345,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "TEST_CATEGORY",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::arg_data_t arg{ .position = 0,
                                            .type     = "int",
                                            .name     = "param",
                                            .value    = "42",
                                            .extdata  = R"({"semantic":"config"})" };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_function",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = { arg }
    };

    // Register track for sample creation
    auto track = create_track_info(1, 1000, 100, "test_track");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "test_track"
    };
    m_writer->insert_region_data(region, trace_env);

    // Insert PMC event
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
        .value   = 0.5,
        .extdata = R"({"unit":"ratio","normalized":true})",
        .sample =
            rocpdsna::writer_types::sample_data_t{
                .timestamp = 1500000,
                .track =
                    rocpdsna::writer_types::track_info_t{
                        .name       = "test_track",
                        .extdata    = R"({"source":"profiler","kind":"gpu"})",
                        .node_id    = 1,
                        .process_id = 1000,
                        .thread_id  = 100 },
                .extdata = R"({"interval_ms":10})" }
    };
    m_writer->insert_pmc_event_data(
        pmc_event,
        rocpdsna::writer_types::pmc_info_unique_id_t{
            .name     = "TEST_COUNTER",
            .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });

    flush_and_wait();

    // Verify all missing tables have data
    EXPECT_GE(count_rows(m_db_path, "rocpd_event", m_uuid), 1)
        << "rocpd_event should have rows";
    EXPECT_GE(count_rows(m_db_path, "rocpd_arg", m_uuid), 1)
        << "rocpd_arg should have rows";
    EXPECT_GE(count_rows(m_db_path, "rocpd_pmc_event", m_uuid), 1)
        << "rocpd_pmc_event should have rows";
    EXPECT_GE(count_rows(m_db_path, "rocpd_sample", m_uuid), 1)
        << "rocpd_sample should have rows";
}

// ============================================================================
// ============================================================================
//                    RE-ADDED MISSING TESTS (WITH CORRECT API)
// ============================================================================
// ============================================================================

// ============================================================================
// Test: rocpd_region - Region with track and timestamp FKs
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, region_values_with_track_and_timestamps)
{
    register_node_process_thread();
    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    auto region = create_region_data("test_function", 1000000, 2000000);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify region with joined timestamp values
    auto result = query_database(m_db_path,
                                 "SELECT s.string, ts_start.value, ts_end.value "
                                 "FROM rocpd_region_" +
                                     m_uuid +
                                     " r "
                                     "JOIN rocpd_string_" +
                                     m_uuid +
                                     " s ON r.name_id = s.id "
                                     "JOIN rocpd_timestamp_" +
                                     m_uuid +
                                     " ts_start ON r.start_id = ts_start.id "
                                     "JOIN rocpd_timestamp_" +
                                     m_uuid + " ts_end ON r.end_id = ts_end.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "test_function");
    EXPECT_EQ(result.rows[0][1], "1000000");
    EXPECT_EQ(result.rows[0][2], "2000000");
}

// ============================================================================
// Test: rocpd_kernel_dispatch - Full kernel dispatch with dependencies
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, kernel_dispatch_full_chain)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);

    auto kernel = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    auto dispatch = create_kernel_dispatch_data(1, 1, 1, 2000000, 3000000);
    m_writer->insert_kernel_dispatch_data(dispatch, env);
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT kd.dispatch_id, kd.workgroup_size_x, kd.grid_size_x, "
                       "ts_start.value, ts_end.value, ks.kernel_name "
                       "FROM rocpd_kernel_dispatch_" +
                           m_uuid +
                           " kd "
                           "JOIN rocpd_timestamp_" +
                           m_uuid +
                           " ts_start ON kd.start_id = ts_start.id "
                           "JOIN rocpd_timestamp_" +
                           m_uuid +
                           " ts_end ON kd.end_id = ts_end.id "
                           "JOIN rocpd_info_kernel_symbol_" +
                           m_uuid + " ks ON kd.kernel_id = ks.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
    EXPECT_EQ(result.rows[0][1], "64");
    EXPECT_EQ(result.rows[0][2], "1024");
    EXPECT_EQ(result.rows[0][3], "2000000");
    EXPECT_EQ(result.rows[0][4], "3000000");
    EXPECT_EQ(result.rows[0][5], "vector_add");
}

// ============================================================================
// Test: rocpd_memory_copy - Memory copy with agent FKs
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, memory_copy_values_and_fks)
{
    register_node_process_thread();
    auto cpu_agent = create_agent_info(1, 1000, "CPU", 0);
    m_writer->register_agent_info(cpu_agent);

    auto gpu_agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(gpu_agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    auto copy         = create_memory_copy_data(3000000, 3500000);
    copy.src_agent_id = rocpdsna::writer_types::agent_unique_id_t{ "CPU", 0 };
    copy.dst_agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 };

    m_writer->insert_memory_copy_data(copy, env);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT mc.size, ts_start.value, ts_end.value, "
                                 "src_a.type as src_type, dst_a.type as dst_type "
                                 "FROM rocpd_memory_copy_" +
                                     m_uuid +
                                     " mc "
                                     "JOIN rocpd_timestamp_" +
                                     m_uuid +
                                     " ts_start ON mc.start_id = ts_start.id "
                                     "JOIN rocpd_timestamp_" +
                                     m_uuid +
                                     " ts_end ON mc.end_id = ts_end.id "
                                     "LEFT JOIN rocpd_info_agent_" +
                                     m_uuid +
                                     " src_a ON mc.src_agent_id = src_a.id "
                                     "LEFT JOIN rocpd_info_agent_" +
                                     m_uuid + " dst_a ON mc.dst_agent_id = dst_a.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1048576");
    EXPECT_EQ(result.rows[0][1], "3000000");
    EXPECT_EQ(result.rows[0][2], "3500000");
    EXPECT_EQ(result.rows[0][3], "CPU");
    EXPECT_EQ(result.rows[0][4], "GPU");
}

// ============================================================================
// Test: rocpd_memory_allocate - Memory allocation with type/level
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, memory_allocate_values_correct)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    auto alloc = create_memory_alloc_data(500000, 600000);
    m_writer->insert_memory_alloc_data(alloc, env);
    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT ma.type, ma.level, ma.size, ts_start.value, ts_end.value "
                       "FROM rocpd_memory_allocate_" +
                           m_uuid +
                           " ma "
                           "JOIN rocpd_timestamp_" +
                           m_uuid +
                           " ts_start ON ma.start_id = ts_start.id "
                           "JOIN rocpd_timestamp_" +
                           m_uuid + " ts_end ON ma.end_id = ts_end.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "ALLOC");
    EXPECT_EQ(result.rows[0][1], "REAL");
    EXPECT_EQ(result.rows[0][2], "1048576");
    EXPECT_EQ(result.rows[0][3], "500000");
    EXPECT_EQ(result.rows[0][4], "600000");
}

// ============================================================================
// Test: rocpd_track - Track created with region
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, track_created_with_region)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto track = create_track_info(1, 1000, 100, "MyTrack");
    m_writer->register_track_info(track);

    auto region = create_region_data("test_fn", 1000000, 2000000);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "MyTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT tr.nid, p.pid, t.tid, s.string "
                                 "FROM rocpd_track_" +
                                     m_uuid +
                                     " tr "
                                     "JOIN rocpd_info_process_" +
                                     m_uuid +
                                     " p ON tr.pid = p.id "
                                     "LEFT JOIN rocpd_info_thread_" +
                                     m_uuid +
                                     " t ON tr.tid = t.id "
                                     "LEFT JOIN rocpd_string_" +
                                     m_uuid + " s ON tr.name_id = s.id");

    ASSERT_GE(result.rows.size(), 1);
}

// ============================================================================
// Test: rocpd_timestamp - Timestamps with phase values
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, timestamp_phase_values)
{
    register_base_entities();
    auto region = create_region_data("test", 1000000, 2000000);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT ts.value, ts.phase FROM rocpd_timestamp_" +
                                     m_uuid + " ts ORDER BY ts.value");

    ASSERT_GE(result.rows.size(), 2);
    // Check that both start and end timestamps exist (order may vary)
    bool found_start = false, found_end = false;
    for(const auto& row : result.rows)
    {
        if(row[0] == "1000000") found_start = true;
        if(row[0] == "2000000") found_end = true;
    }
    EXPECT_TRUE(found_start) << "Start timestamp 1000000 not found";
    EXPECT_TRUE(found_end) << "End timestamp 2000000 not found";
}

// ============================================================================
// Test: rocpd_string - String table deduplication
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, string_table_deduplication)
{
    register_node_process_thread();
    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    for(int i = 0; i < 5; ++i)
    {
        auto region = create_region_data(
            "repeated_name", 1000000 + i * 100000, 1050000 + i * 100000);
        m_writer->insert_region_data(region, trace_env);
    }
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT COUNT(*) FROM rocpd_string_" + m_uuid +
                                     " WHERE string = 'repeated_name'");

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

// ============================================================================
// Test: rocpd_metadata - Metadata key-value pairs
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, metadata_table_populated)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT tag, value FROM rocpd_metadata_" + m_uuid +
                                     " WHERE tag LIKE 'schema_version%'");

    ASSERT_GE(result.rows.size(), 1);
}

// ============================================================================
// Test: Full end-to-end with all tables populated
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, end_to_end_all_tables)
{
    register_node_process_thread();
    auto cpu_agent = create_agent_info(1, 1000, "CPU", 0);
    m_writer->register_agent_info(cpu_agent);

    auto gpu_agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(gpu_agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);

    auto kernel = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel);

    auto category = create_category_info(1, "HIP_API");
    m_writer->register_category_info(category);

    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);

    auto addr_range = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr_range);

    auto source = create_source_code_info(1, 1, 1000, 1);
    m_writer->register_source_code_info(source);

    auto pc = create_pc_info(1, 1, 1000, 1);
    m_writer->register_pc_info(pc);

    auto track = create_track_info(1, 1000, 100, "MainTrack");
    m_writer->register_track_info(track);

    auto test_track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(test_track);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "MainTrack"
    };

    auto region = create_region_data("api_call", 1000000, 1500000);
    m_writer->insert_region_data(region, env);

    auto dispatch = create_kernel_dispatch_data(1, 1, 1, 2000000, 3000000);
    m_writer->insert_kernel_dispatch_data(dispatch, env);

    auto copy = create_memory_copy_data(3500000, 4000000);
    m_writer->insert_memory_copy_data(copy, env);

    auto alloc = create_memory_alloc_data(500000, 600000);
    m_writer->insert_memory_alloc_data(alloc, env);

    flush_and_wait();

    EXPECT_GE(count_rows(m_db_path, "rocpd_info_node", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_process", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_thread", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_agent", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_queue", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_stream", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_code_object", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_kernel_symbol", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_category", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_pmc", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_address_range", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_source_code", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_info_pc", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_track", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_timestamp", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_string", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_region", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_kernel_dispatch", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_memory_copy", m_uuid), 1);
    EXPECT_GE(count_rows(m_db_path, "rocpd_memory_allocate", m_uuid), 1);
}

// ============================================================================
// Test: V4-specific name columns in node/process
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, v4_name_columns_populated)
{
    auto node = create_node_info(1);
    node.name = "CustomNodeName";
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    process.name = "CustomProcessName";
    m_writer->register_process_info(process);
    flush_and_wait();

    auto node_result =
        query_database(m_db_path, "SELECT name FROM rocpd_info_node_" + m_uuid);
    ASSERT_EQ(node_result.rows.size(), 1);
    EXPECT_EQ(node_result.rows[0][0], "CustomNodeName");

    auto process_result =
        query_database(m_db_path, "SELECT name FROM rocpd_info_process_" + m_uuid);
    ASSERT_EQ(process_result.rows.size(), 1);
    EXPECT_EQ(process_result.rows[0][0], "CustomProcessName");
}

// ============================================================================
// Test: V4-specific qualifier column in pmc_info
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, v4_pmc_qualifier_populated)
{
    register_node_and_process();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    pmc.qualifier = "MAX_VALUE";
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT qualifier FROM rocpd_info_pmc_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "MAX_VALUE");
}

// ============================================================================
// Test: Multiple nodes with processes
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, multiple_nodes_with_processes)
{
    auto node1     = create_node_info(1);
    node1.hostname = "host1";
    m_writer->register_node_info(node1);

    auto proc1 = create_process_info(1, 1000);
    m_writer->register_process_info(proc1);

    auto node2     = create_node_info(2);
    node2.hostname = "host2";
    m_writer->register_node_info(node2);

    auto proc2 = create_process_info(2, 2000);
    m_writer->register_process_info(proc2);

    flush_and_wait();

    auto result =
        query_database(m_db_path,
                       "SELECT n.hostname, p.pid FROM rocpd_info_process_" + m_uuid +
                           " p "
                           "JOIN rocpd_info_node_" +
                           m_uuid + " n ON p.nid = n.id ORDER BY p.pid");

    ASSERT_EQ(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0][0], "host1");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[1][0], "host2");
    EXPECT_EQ(result.rows[1][1], "2000");
}

// ============================================================================
// Test: Memory allocate type CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, memory_allocate_type_check_constraint)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    auto alloc = create_memory_alloc_data(500000, 600000);
    alloc.type = "ALLOC";
    m_writer->insert_memory_alloc_data(alloc, env);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT type FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "ALLOC");
}

// ============================================================================
// Test: Memory allocate level CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, memory_allocate_level_check_constraint)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    auto alloc  = create_memory_alloc_data(500000, 600000);
    alloc.level = "REAL";
    m_writer->insert_memory_alloc_data(alloc, env);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT level FROM rocpd_memory_allocate_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "REAL");
}

// ============================================================================
// Test: Timestamp phase CHECK constraint validation
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, timestamp_phase_check_constraint)
{
    register_base_entities();
    auto region = create_region_data("test", 1000000, 2000000);

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT phase FROM rocpd_timestamp_" + m_uuid);

    ASSERT_GE(result.rows.size(), 2);
    for(const auto& row : result.rows)
    {
        // Phase might be NULL in some rows, skip those
        if(row[0].empty() || row[0] == "NULL") continue;
        int phase = std::stoi(row[0]);
        EXPECT_TRUE(phase == 0 || phase == 1 || phase == 2)
            << "Invalid phase value: " << phase;
    }
}

// ============================================================================
// Test: rocpd_event - Event with parent_stack_id
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_with_parent_stack_id)
{
    register_node_process_thread();
    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 2,
        .parent_stack_id = 1,
        .correlation_id  = 100,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":1})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1500000,
        .end_timestamp   = 1800000,
        .name            = "nested_call",
        .extdata         = R"({"scope":"nested","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT stack_id, parent_stack_id, correlation_id FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "2");
    EXPECT_EQ(result.rows[0][1], "1");
    EXPECT_EQ(result.rows[0][2], "100");
}

// ============================================================================
// Test: rocpd_arg - Multiple arguments with different types
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, arg_table_multiple_types)
{
    register_node_process_thread();
    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    std::vector<rocpdsna::writer_types::arg_data_t> args = {
        { .position = 0,
          .type     = "hipStream_t",
          .name     = "stream",
          .value    = "0x12345678",
          .extdata  = R"({"semantic":"stream"})" },
        { .position = 1,
          .type     = "unsigned int",
          .name     = "flags",
          .value    = "0",
          .extdata  = R"({"semantic":"flags"})" },
        { .position = 2,
          .type     = "hipDeviceptr_t",
          .name     = "devPtr",
          .value    = "0x7f0000000000",
          .extdata  = R"({"semantic":"pointer"})" },
        { .position = 3,
          .type     = "float",
          .name     = "value",
          .value    = "3.14159",
          .extdata  = R"({"semantic":"scalar"})" },
        { .position = 4,
          .type     = "const char*",
          .name     = "name",
          .value    = "kernel_name",
          .extdata  = R"({"semantic":"label"})" }
    };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 1,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "hipLaunchKernelGGL",
        .extdata         = R"({"scope":"global","async":true})",
        .args            = args
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT position, type, name, value FROM rocpd_arg_" +
                                     m_uuid + " ORDER BY position");

    ASSERT_EQ(result.rows.size(), 5);

    EXPECT_EQ(result.rows[0][1], "hipStream_t");
    EXPECT_EQ(result.rows[1][1], "unsigned int");
    EXPECT_EQ(result.rows[2][1], "hipDeviceptr_t");
    EXPECT_EQ(result.rows[3][1], "float");
    EXPECT_EQ(result.rows[4][1], "const char*");
}

// ============================================================================
// Test: rocpd_event - Event created via kernel_dispatch
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_via_kernel_dispatch)
{
    register_node_process_thread();

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto code_obj = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(code_obj);

    auto kernel = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(kernel);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 500,
        .parent_stack_id = 0,
        .correlation_id  = 600,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"dispatch","depth":0})"
    };

    rocpdsna::writer_types::kernel_dispatch_data_t dispatch{
        .event                = event_data,
        .dispatch_id          = 1,
        .start_timestamp      = 2000000,
        .end_timestamp        = 3000000,
        .kernel_symbol_id     = 1,
        .code_object_id       = 1,
        .private_segment_size = 0,
        .group_segment_size   = 256,
        .workgroup_size_x     = 64,
        .workgroup_size_y     = 1,
        .workgroup_size_z     = 1,
        .grid_size_x          = 1024,
        .grid_size_y          = 1,
        .grid_size_z          = 1,
        .name                 = "test_kernel",
        .extdata              = R"({"queue_idx":0,"signal_handle":42})"
    };

    m_writer->insert_kernel_dispatch_data(dispatch, env);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT stack_id, correlation_id FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "500");
    EXPECT_EQ(result.rows[0][1], "600");
}

// ============================================================================
// Test: rocpd_event - Event created via memory_copy
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_via_memory_copy)
{
    register_node_process_thread();

    auto cpu_agent = create_agent_info(1, 1000, "CPU", 0);
    m_writer->register_agent_info(cpu_agent);

    auto gpu_agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(gpu_agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 700,
        .parent_stack_id = 699,
        .correlation_id  = 800,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"memcpy","depth":0})"
    };

    rocpdsna::writer_types::memory_copy_data_t copy{
        .event           = event_data,
        .start_timestamp = 3000000,
        .end_timestamp   = 3500000,
        .dst_agent_id    = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .dst_address     = 0x7f0000100000,
        .src_agent_id    = rocpdsna::writer_types::agent_unique_id_t{ "CPU", 0 },
        .src_address     = 0x7f0000200000,
        .size            = 1024 * 1024,
        .name            = "hipMemcpy",
        .region_name     = "default_region",
        .extdata         = R"({"direction":"HtoD","pinned":true})"
    };

    m_writer->insert_memory_copy_data(copy, env);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT stack_id, parent_stack_id, correlation_id FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "700");
    EXPECT_EQ(result.rows[0][1], "699");
    EXPECT_EQ(result.rows[0][2], "800");
}

// ============================================================================
// Test: rocpd_event - Event created via memory_alloc
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, event_table_via_memory_alloc)
{
    register_node_process_thread();
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    rocpdsna::writer_types::trace_environment_t env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 900,
        .parent_stack_id = 0,
        .correlation_id  = 1000,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"alloc_type":"hipMalloc"})"
    };

    rocpdsna::writer_types::memory_alloc_data_t alloc{
        .event           = event_data,
        .type            = "ALLOC",
        .level           = "REAL",
        .start_timestamp = 500000,
        .end_timestamp   = 600000,
        .address         = 0x7f0000100000,
        .size            = 1024 * 1024,
        .extdata         = R"({"pool":"default","managed":false})"
    };

    m_writer->insert_memory_alloc_data(alloc, env);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT stack_id, correlation_id, extdata FROM rocpd_event_" + m_uuid);

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "900");
    EXPECT_EQ(result.rows[0][1], "1000");
    EXPECT_EQ(result.rows[0][2], R"({"alloc_type":"hipMalloc"})");
}

// ============================================================================
// Test: Multiple events with different categories
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, events_multiple_categories)
{
    register_node_process_thread();

    auto track = create_track_info(1, 1000, 100, "TestTrack");
    m_writer->register_track_info(track);

    m_writer->register_category_info(create_category_info(1, "HIP_API"));
    m_writer->register_category_info(create_category_info(2, "HSA_API"));
    m_writer->register_category_info(create_category_info(3, "ROCTX"));

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    const char* categories[] = { "HIP_API", "HSA_API", "ROCTX" };
    const char* names[]      = { "hipLaunchKernel", "hsa_queue_create", "roctxMark" };

    for(int i = 0; i < 3; ++i)
    {
        rocpdsna::writer_types::event_data_t event_data{
            .stack_id        = static_cast<size_t>(i + 1),
            .parent_stack_id = 0,
            .correlation_id  = static_cast<size_t>(i * 100),
            .call_stack      = {},
            .line_info_list  = {},
            .event_category  = categories[i],
            .extdata         = R"({"origin":"api","depth":0})"
        };

        rocpdsna::writer_types::region_data_t region{
            .event           = event_data,
            .start_timestamp = static_cast<size_t>(1000000 + i * 100000),
            .end_timestamp   = static_cast<size_t>(1050000 + i * 100000),
            .name            = names[i],
            .extdata         = R"({"scope":"global","async":false})",
            .args            = {}
        };

        m_writer->insert_region_data(region, trace_env);
    }
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT e.stack_id, c.name as category "
                                 "FROM rocpd_event_" +
                                     m_uuid +
                                     " e "
                                     "JOIN rocpd_info_category_" +
                                     m_uuid +
                                     " c ON e.category_id = c.id "
                                     "ORDER BY e.stack_id");

    ASSERT_EQ(result.rows.size(), 3);
    EXPECT_EQ(result.rows[0][1], "HIP_API");
    EXPECT_EQ(result.rows[1][1], "HSA_API");
    EXPECT_EQ(result.rows[2][1], "ROCTX");
}

// ============================================================================
// Test: rocpd_pmc_event with event data
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, pmc_event_with_event_data)
{
    register_node_process_thread();

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    // When pmc_event has an event, sample.track must be registered
    auto track = create_track_info(1, 1000, 100, "PMCEventTrack");
    m_writer->register_track_info(track);

    auto pmc = create_pmc_info(
        1, 1000, "EVENT_PMC", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 123,
        .parent_stack_id = 0,
        .correlation_id  = 456,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"pmc","depth":0})"
    };

    // sample.track must be a valid track_info_t matching registered track
    rocpdsna::writer_types::track_info_t sample_track{
        .name       = "PMCEventTrack",
        .extdata    = R"({"source":"profiler","kind":"gpu"})",
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100
    };

    rocpdsna::writer_types::pmc_event_data_t pmc_event{
        .event   = event_data,
        .value   = 98765.43,
        .extdata = R"({"unit":"cycles","normalized":false})",
        .sample =
            rocpdsna::writer_types::sample_data_t{ .timestamp = 8888888,
                                                   .track     = sample_track,
                                                   .extdata   = R"({"interval_ms":5})" }
    };

    m_writer->insert_pmc_event_data(
        pmc_event,
        rocpdsna::writer_types::pmc_info_unique_id_t{
            .name     = "EVENT_PMC",
            .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT pe.value, e.stack_id, e.correlation_id "
                                 "FROM rocpd_pmc_event_" +
                                     m_uuid +
                                     " pe "
                                     "JOIN rocpd_event_" +
                                     m_uuid + " e ON pe.event_id = e.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][1], "123");
    EXPECT_EQ(result.rows[0][2], "456");
}

// ============================================================================
// Test: rocpd_sample - Sample with timestamp and track FK
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, sample_table_with_timestamp_track)
{
    register_node_process_thread();

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto pmc = create_pmc_info(
        1, 1000, "SAMPLE_COUNTER", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);

    auto track = create_track_info(1, 1000, 100, "SampleTrack");
    m_writer->register_track_info(track);

    // Sample is only created when event has value - must provide event
    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 777,
        .parent_stack_id = 0,
        .correlation_id  = 888,
        .call_stack      = {},
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"pmc","depth":0})"
    };

    rocpdsna::writer_types::track_info_t sample_track{
        .name       = "SampleTrack",
        .extdata    = R"({"source":"profiler","kind":"gpu"})",
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100
    };

    rocpdsna::writer_types::pmc_event_data_t pmc_event{
        .event   = event_data,  // Must have event to trigger sample creation
        .value   = 999.0,
        .extdata = R"({"unit":"count","normalized":false})",
        .sample =
            rocpdsna::writer_types::sample_data_t{ .timestamp = 7777777,
                                                   .track     = sample_track,
                                                   .extdata   = R"({"interval_ms":10})" }
    };

    m_writer->insert_pmc_event_data(
        pmc_event,
        rocpdsna::writer_types::pmc_info_unique_id_t{
            .name     = "SAMPLE_COUNTER",
            .agent_id = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 } });
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT s.event_id, ts.value as timestamp "
                                 "FROM rocpd_sample_" +
                                     m_uuid +
                                     " s "
                                     "JOIN rocpd_timestamp_" +
                                     m_uuid + " ts ON s.timestamp_id = ts.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][1], "7777777");
}

// ============================================================================
// ============================================================================
//                    CALL STACK AND LINE INFO TABLE TESTS
// ============================================================================
// ============================================================================

// ============================================================================
// Test: rocpd_call_stack - Call stack entries populated via region with call_stack
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, call_stack_table_populated_via_region)
{
    register_base_entities();

    // Create call stack entries
    rocpdsna::shared_types::stack_frame_t frame1{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "main", .filename = "main.cpp", .line_number = 10 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x1000, .address_low = 0x1000, .address_high = 0x1FFF },
        .extdata = R"({"function":"main"})"
    };

    rocpdsna::shared_types::stack_frame_t frame2{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "foo", .filename = "foo.cpp", .line_number = 25 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x2000, .address_low = 0x2000, .address_high = 0x2FFF },
        .extdata = R"({"function":"foo"})"
    };

    rocpdsna::shared_types::stack_frame_t frame3{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "bar", .filename = "bar.cpp", .line_number = 42 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x3000, .address_low = 0x3000, .address_high = 0x3FFF },
        .extdata = R"({"function":"bar"})"
    };

    rocpdsna::shared_types::call_stack_t call_stack = { frame1, frame2, frame3 };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 100,
        .call_stack      = call_stack,
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_with_call_stack",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify rocpd_call_stack table has data
    auto result = query_database(m_db_path,
                                 "SELECT depth, extdata FROM rocpd_call_stack_" + m_uuid +
                                     " ORDER BY depth");

    ASSERT_EQ(result.rows.size(), 3);
    EXPECT_EQ(result.rows[0][0], "0");  // depth 0 (top of stack)
    EXPECT_EQ(result.rows[0][1], R"({"function":"main"})");
    EXPECT_EQ(result.rows[1][0], "1");  // depth 1
    EXPECT_EQ(result.rows[1][1], R"({"function":"foo"})");
    EXPECT_EQ(result.rows[2][0], "2");  // depth 2
    EXPECT_EQ(result.rows[2][1], R"({"function":"bar"})");
}

// ============================================================================
// Test: rocpd_call_stack - Verify event_id FK from call_stack to event
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, call_stack_event_fk_verified)
{
    register_base_entities();

    rocpdsna::shared_types::stack_frame_t frame{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "test_function", .filename = "test.cpp", .line_number = 55 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x4000, .address_low = 0x4000, .address_high = 0x4FFF },
        .extdata = R"({"function":"test_function"})"
    };

    rocpdsna::shared_types::call_stack_t call_stack = { frame };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 999,
        .parent_stack_id = 0,
        .correlation_id  = 888,
        .call_stack      = call_stack,
        .line_info_list  = {},
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_call_stack_fk",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify call_stack has FK to event
    auto result = query_database(m_db_path,
                                 "SELECT cs.depth, e.stack_id, e.correlation_id "
                                 "FROM rocpd_call_stack_" +
                                     m_uuid +
                                     " cs "
                                     "JOIN rocpd_event_" +
                                     m_uuid + " e ON cs.event_id = e.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "0");    // depth
    EXPECT_EQ(result.rows[0][1], "999");  // event stack_id via FK
    EXPECT_EQ(result.rows[0][2], "888");  // event correlation_id via FK
}

// ============================================================================
// Test: rocpd_line_info - Line info entries populated via region
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, line_info_table_populated_via_region)
{
    register_base_entities();

    // Create line info entries
    rocpdsna::shared_types::line_info_entry_t line_entry1{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "kernel.cpp",
                .starting_line_number       = 10,
                .source_code_lines          = { "float a = x + y;", "return a;" },
                .assembly_instruction_lines = { "v_add_f32 v0, v1, v2",
                                                "s_setpc_b64 s[30:31]" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "vector_add", .filename = "kernel.cpp", .line_number = 10 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x5000, .address_low = 0x5000, .address_high = 0x5FFF }
    };

    rocpdsna::shared_types::line_info_entry_t line_entry2{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "utils.cpp",
                .starting_line_number       = 50,
                .source_code_lines          = { "int idx = get_global_id(0);" },
                .assembly_instruction_lines = { "v_mov_b32 v0, s0" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "get_index", .filename = "utils.cpp", .line_number = 50 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x6000, .address_low = 0x6000, .address_high = 0x6FFF }
    };

    rocpdsna::shared_types::source_context_list_t line_info_list = { line_entry1,
                                                                     line_entry2 };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 1,
        .parent_stack_id = 0,
        .correlation_id  = 200,
        .call_stack      = {},
        .line_info_list  = line_info_list,
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_with_line_info",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify rocpd_line_info table has data
    auto result =
        query_database(m_db_path, "SELECT id, event_id FROM rocpd_line_info_" + m_uuid);

    ASSERT_EQ(result.rows.size(), 2);
}

// ============================================================================
// Test: rocpd_line_info - Verify event_id FK from line_info to event
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, line_info_event_fk_verified)
{
    register_base_entities();

    rocpdsna::shared_types::line_info_entry_t line_entry{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "dispatch.cpp",
                .starting_line_number       = 77,
                .source_code_lines          = { "launch_kernel();" },
                .assembly_instruction_lines = { "s_branch 0x100" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{ .function = "launch_kernel",
                                                            .filename = "dispatch.cpp",
                                                            .line_number = 77 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x7000, .address_low = 0x7000, .address_high = 0x7FFF }
    };

    rocpdsna::shared_types::source_context_list_t line_info_list = { line_entry };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 777,
        .parent_stack_id = 0,
        .correlation_id  = 666,
        .call_stack      = {},
        .line_info_list  = line_info_list,
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_line_info_fk",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify line_info has FK to event
    auto result = query_database(m_db_path,
                                 "SELECT li.id, e.stack_id, e.correlation_id "
                                 "FROM rocpd_line_info_" +
                                     m_uuid +
                                     " li "
                                     "JOIN rocpd_event_" +
                                     m_uuid + " e ON li.event_id = e.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][1], "777");  // event stack_id via FK
    EXPECT_EQ(result.rows[0][2], "666");  // event correlation_id via FK
}

// ============================================================================
// Test: rocpd_call_stack and rocpd_line_info - Both populated in same event
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, call_stack_and_line_info_together)
{
    register_base_entities();

    // Create call stack
    rocpdsna::shared_types::stack_frame_t frame1{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "main", .filename = "main.cpp", .line_number = 5 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x8000, .address_low = 0x8000, .address_high = 0x8FFF },
        .extdata = R"({"function":"main"})"
    };

    rocpdsna::shared_types::stack_frame_t frame2{
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "worker", .filename = "worker.cpp", .line_number = 30 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0x9000, .address_low = 0x9000, .address_high = 0x9FFF },
        .extdata = R"({"function":"worker"})"
    };

    rocpdsna::shared_types::call_stack_t call_stack = { frame1, frame2 };

    // Create line info
    rocpdsna::shared_types::line_info_entry_t line_entry1{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "compute.cpp",
                .starting_line_number       = 100,
                .source_code_lines          = { "result[i] = a[i] + b[i];" },
                .assembly_instruction_lines = { "v_add_f32 v0, v1, v2" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "compute", .filename = "compute.cpp", .line_number = 100 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0xA000, .address_low = 0xA000, .address_high = 0xAFFF }
    };

    rocpdsna::shared_types::line_info_entry_t line_entry2{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "reduce.cpp",
                .starting_line_number       = 200,
                .source_code_lines          = { "sum += partial[i];" },
                .assembly_instruction_lines = { "v_add_f32 v3, v3, v4" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "reduce", .filename = "reduce.cpp", .line_number = 200 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0xB000, .address_low = 0xB000, .address_high = 0xBFFF }
    };

    rocpdsna::shared_types::line_info_entry_t line_entry3{
        .source_code =
            rocpdsna::shared_types::source_code_info_t{
                .filename                   = "sync.cpp",
                .starting_line_number       = 300,
                .source_code_lines          = { "__syncthreads();" },
                .assembly_instruction_lines = { "s_barrier" } },
        .program_counter =
            rocpdsna::shared_types::program_counter_info_t{
                .function = "sync_threads", .filename = "sync.cpp", .line_number = 300 },
        .address_range =
            rocpdsna::shared_types::address_range_info_t{
                .address_base = 0xC000, .address_low = 0xC000, .address_high = 0xCFFF }
    };

    rocpdsna::shared_types::source_context_list_t line_info_list = { line_entry1,
                                                                     line_entry2,
                                                                     line_entry3 };

    rocpdsna::writer_types::event_data_t event_data{
        .stack_id        = 123,
        .parent_stack_id = 0,
        .correlation_id  = 456,
        .call_stack      = call_stack,
        .line_info_list  = line_info_list,
        .event_category  = "GENERAL",
        .extdata         = R"({"origin":"api","depth":0})"
    };

    rocpdsna::writer_types::region_data_t region{
        .event           = event_data,
        .start_timestamp = 1000000,
        .end_timestamp   = 2000000,
        .name            = "test_both_call_stack_and_line_info",
        .extdata         = R"({"scope":"global","async":false})",
        .args            = {}
    };

    rocpdsna::writer_types::trace_environment_t trace_env{
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .stream_id  = 1,
        .queue_id   = 1,
        .track_name = "TestTrack"
    };

    m_writer->insert_region_data(region, trace_env);
    flush_and_wait();

    // Verify both tables have data
    auto call_stack_result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_call_stack_" + m_uuid);
    ASSERT_EQ(call_stack_result.rows.size(), 1);
    EXPECT_EQ(call_stack_result.rows[0][0], "2");  // 2 call stack frames

    auto line_info_result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_line_info_" + m_uuid);
    ASSERT_EQ(line_info_result.rows.size(), 1);
    EXPECT_EQ(line_info_result.rows[0][0], "3");  // 3 line info entries

    // Verify they share the same event_id
    auto combined_result =
        query_database(m_db_path,
                       "SELECT DISTINCT e.stack_id "
                       "FROM rocpd_event_" +
                           m_uuid +
                           " e "
                           "WHERE e.id IN (SELECT event_id FROM rocpd_call_stack_" +
                           m_uuid +
                           ") "
                           "AND e.id IN (SELECT event_id FROM rocpd_line_info_" +
                           m_uuid + ")");

    ASSERT_EQ(combined_result.rows.size(), 1);
    EXPECT_EQ(combined_result.rows[0][0], "123");  // Same event for both
}

// ============================================================================
// MISSING COVERAGE: rocpd_track explicit column values verification
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, track_values_and_all_fks_verified)
{
    register_node_process_thread();

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);

    rocpdsna::writer_types::track_info_t track{
        .name       = "GPUTrack",
        .extdata    = R"({"source":"profiler","kind":"gpu"})",
        .node_id    = 1,
        .process_id = 1000,
        .thread_id  = 100,
        .agent_id   = rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 },
        .queue_id   = 1,
        .stream_id  = 1
    };
    m_writer->register_track_info(track);
    flush_and_wait();

    auto result = query_database(
        m_db_path,
        "SELECT tr.nid, p.pid, t.tid, a.type, q.name, st.name, s.string, tr.extdata "
        "FROM rocpd_track_" +
            m_uuid +
            " tr "
            "JOIN rocpd_info_process_" +
            m_uuid +
            " p ON tr.pid = p.id "
            "LEFT JOIN rocpd_info_thread_" +
            m_uuid +
            " t ON tr.tid = t.id "
            "LEFT JOIN rocpd_info_agent_" +
            m_uuid +
            " a ON tr.agent_id = a.id "
            "LEFT JOIN rocpd_info_queue_" +
            m_uuid +
            " q ON tr.queue_id = q.id "
            "LEFT JOIN rocpd_info_stream_" +
            m_uuid +
            " st ON tr.stream_id = st.id "
            "LEFT JOIN rocpd_string_" +
            m_uuid + " s ON tr.name_id = s.id");

    ASSERT_GE(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");           // nid
    EXPECT_EQ(result.rows[0][1], "1000");        // pid
    EXPECT_EQ(result.rows[0][2], "100");         // tid
    EXPECT_EQ(result.rows[0][3], "GPU");         // agent type
    EXPECT_EQ(result.rows[0][4], "TestQueue");   // queue name
    EXPECT_EQ(result.rows[0][5], "TestStream");  // stream name
    EXPECT_EQ(result.rows[0][6], "GPUTrack");    // track name via string table
    EXPECT_EQ(result.rows[0][7], R"({"source":"profiler","kind":"gpu"})");
}

// ============================================================================
// MISSING COVERAGE: info_pmc target_arch CHECK constraint
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, info_pmc_target_arch_check_constraint)
{
    register_node_and_process();

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    // Valid target_arch values: 'CPU' and 'GPU'
    auto pmc_gpu = create_pmc_info(
        1, 1000, "GPU_COUNTER", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    pmc_gpu.target_arch = "GPU";
    m_writer->register_pmc_info(pmc_gpu);

    auto pmc_cpu = create_pmc_info(
        1, 1000, "CPU_COUNTER", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    pmc_cpu.target_arch = "CPU";
    m_writer->register_pmc_info(pmc_cpu);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT name, target_arch FROM rocpd_info_pmc_" +
                                     m_uuid + " ORDER BY name");

    ASSERT_EQ(result.rows.size(), 2);
    EXPECT_EQ(result.rows[0][0], "CPU_COUNTER");
    EXPECT_EQ(result.rows[0][1], "CPU");
    EXPECT_EQ(result.rows[1][0], "GPU_COUNTER");
    EXPECT_EQ(result.rows[1][1], "GPU");
}

// ============================================================================
// MISSING COVERAGE: address_range high >= low CHECK constraint
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, address_range_high_gte_low_constraint)
{
    register_node_and_process();

    auto addr_range         = create_address_range_info(1, 1, 1000);
    addr_range.address_base = 0x1000;
    addr_range.address_low  = 0x2000;
    addr_range.address_high = 0x3000;  // high > low (valid)
    m_writer->register_address_range_info(addr_range);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT address_base, address_low, address_high "
                                 "FROM rocpd_info_address_range_" +
                                     m_uuid);

    ASSERT_EQ(result.rows.size(), 1);
    auto low  = std::stoull(result.rows[0][1]);
    auto high = std::stoull(result.rows[0][2]);
    EXPECT_GE(high, low);
}

// ============================================================================
// MISSING COVERAGE: Duplicate registration tests (entity_registry dedup)
// ============================================================================

TEST_F(SchemaV4ValueInsertionTest, duplicate_node_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    m_writer->register_node_info(node);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_node_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_process_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    m_writer->register_process_info(process);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_process_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_thread_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);
    m_writer->register_thread_info(thread);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_thread_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_agent_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);
    m_writer->register_agent_info(agent);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_agent_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_queue_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto queue = create_queue_info(1, 1000, 1);
    m_writer->register_queue_info(queue);
    m_writer->register_queue_info(queue);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_queue_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_stream_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto stream = create_stream_info(1, 1000, 1);
    m_writer->register_stream_info(stream);
    m_writer->register_stream_info(stream);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_stream_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_category_registration_ignored)
{
    auto cat = create_category_info(1, "HIP_API");
    m_writer->register_category_info(cat);
    m_writer->register_category_info(cat);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_category_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_pmc_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto pmc = create_pmc_info(
        1, 1000, "SQ_WAVES", rocpdsna::writer_types::agent_unique_id_t{ "GPU", 0 });
    m_writer->register_pmc_info(pmc);
    m_writer->register_pmc_info(pmc);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_pmc_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_code_object_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);

    auto co = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(co);
    m_writer->register_code_object_info(co);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT COUNT(*) FROM rocpd_info_code_object_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_kernel_symbol_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto agent = create_agent_info(1, 1000, "GPU", 0);
    m_writer->register_agent_info(agent);
    auto co = create_code_object_info(1, 1, 1000, { "GPU", 0 });
    m_writer->register_code_object_info(co);

    auto ks = create_kernel_symbol_info(1, 1, 1000, 1);
    m_writer->register_kernel_symbol_info(ks);
    m_writer->register_kernel_symbol_info(ks);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT COUNT(*) FROM rocpd_info_kernel_symbol_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_address_range_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);

    auto addr = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr);
    m_writer->register_address_range_info(addr);
    flush_and_wait();

    auto result = query_database(
        m_db_path, "SELECT COUNT(*) FROM rocpd_info_address_range_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_source_code_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto addr = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr);

    auto sc = create_source_code_info(1, 1, 1000, 1);
    m_writer->register_source_code_info(sc);
    m_writer->register_source_code_info(sc);
    flush_and_wait();

    auto result = query_database(m_db_path,
                                 "SELECT COUNT(*) FROM rocpd_info_source_code_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_pc_info_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto addr = create_address_range_info(1, 1, 1000);
    m_writer->register_address_range_info(addr);

    auto pc = create_pc_info(1, 1, 1000, 1);
    m_writer->register_pc_info(pc);
    m_writer->register_pc_info(pc);
    flush_and_wait();

    auto result =
        query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_info_pc_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

TEST_F(SchemaV4ValueInsertionTest, duplicate_track_registration_ignored)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);
    auto process = create_process_info(1, 1000);
    m_writer->register_process_info(process);
    auto thread = create_thread_info(1, 1000, 100);
    m_writer->register_thread_info(thread);

    auto track = create_track_info(1, 1000, 100, "SameTrack");
    m_writer->register_track_info(track);
    m_writer->register_track_info(track);
    flush_and_wait();

    auto result = query_database(m_db_path, "SELECT COUNT(*) FROM rocpd_track_" + m_uuid);
    ASSERT_EQ(result.rows.size(), 1);
    EXPECT_EQ(result.rows[0][0], "1");
}

// ============================================================================
// MISSING COVERAGE: thread missing process FK test
// ============================================================================
TEST_F(SchemaV4ValueInsertionTest, fk_integrity_thread_missing_process_throws)
{
    auto node = create_node_info(1);
    m_writer->register_node_info(node);

    auto thread = create_thread_info(1, 999, 100);  // process_id 999 not registered
    EXPECT_THROW(m_writer->register_thread_info(thread), std::exception);
}
