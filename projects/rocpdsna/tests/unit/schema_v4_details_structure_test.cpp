// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @brief Comprehensive V4 schema validation - one test per table
 * Each test validates: all columns, keys, V4-specific features, and absence of V3-only
 * features
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
class SchemaV4ComprehensiveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_db_path = "test_v4_comprehensive.db";
        m_uuid    = "v4comp";
        std::filesystem::remove(m_db_path);

        // Create V4 database
        auto storage = std::make_unique<rocpdsna::storage_t>(m_db_path, m_uuid, VERSION);
        auto writer  = std::make_unique<rocpdsna::writer_t>(std::move(storage));

        // Write minimal data to initialize schema
        rocpdsna::writer_types::node_info_t node;
        node.hash          = 1;
        node.machine_id    = "test";
        node.hostname      = "test";
        node.system_name   = "test";
        node.release       = "1.0";
        node.version       = "1.0";
        node.hardware_name = "test";
        node.domain_name   = "test";
        writer->register_node_info(node);
        writer->flush_in_memory_data_to_disk();
        writer.reset();

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void TearDown() override { std::filesystem::remove(m_db_path); }

    sqlite3* open_db()
    {
        sqlite3* db = nullptr;
        int      rc = sqlite3_open(m_db_path.c_str(), &db);
        return (rc == SQLITE_OK) ? db : nullptr;
    }

    bool table_exists(sqlite3* db, const std::string& table_name)
    {
        std::string query =
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table_name.c_str(), -1, SQLITE_STATIC);
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return exists;
    }

    bool view_exists(sqlite3* db, const std::string& table_name)
    {
        std::string query = "SELECT name FROM sqlite_master WHERE type='view' AND name=?";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, table_name.c_str(), -1, SQLITE_STATIC);
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return exists;
    }

    bool column_exists(sqlite3*           db,
                       const std::string& table_name,
                       const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        bool found = false;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                found = true;
                break;
            }
        }
        sqlite3_finalize(stmt);
        return found;
    }

    // PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
    bool column_is_notnull(sqlite3*           db,
                           const std::string& table_name,
                           const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        bool notnull = false;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                notnull = (sqlite3_column_int(stmt, 3) != 0);
                break;
            }
        }
        sqlite3_finalize(stmt);
        return notnull;
    }

    bool column_is_pk(sqlite3*           db,
                      const std::string& table_name,
                      const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        bool pk = false;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                pk = (sqlite3_column_int(stmt, 5) != 0);
                break;
            }
        }
        sqlite3_finalize(stmt);
        return pk;
    }

    std::string column_default(sqlite3*           db,
                               const std::string& table_name,
                               const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        std::string dflt;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                const char* val =
                    reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
                if(val) dflt = val;
                break;
            }
        }
        sqlite3_finalize(stmt);
        return dflt;
    }

    std::string column_type(sqlite3*           db,
                            const std::string& table_name,
                            const std::string& column_name)
    {
        std::string   query = "PRAGMA table_info(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        std::string type_str;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if(name && column_name == name)
            {
                const char* t =
                    reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
                if(t) type_str = t;
                break;
            }
        }
        sqlite3_finalize(stmt);
        return type_str;
    }

    // PRAGMA foreign_key_list: id, seq, table, from, to, on_update, on_delete, match
    struct fk_info
    {
        std::string from_col;
        std::string ref_table;
        std::string ref_col;
    };

    std::vector<fk_info> get_foreign_keys(sqlite3* db, const std::string& table_name)
    {
        std::string   query = "PRAGMA foreign_key_list(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        std::vector<fk_info> fks;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            fk_info     fk;
            const char* ref_table =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            const char* from_col =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            const char* ref_col =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
            if(ref_table) fk.ref_table = ref_table;
            if(from_col) fk.from_col = from_col;
            if(ref_col) fk.ref_col = ref_col;
            fks.push_back(fk);
        }
        sqlite3_finalize(stmt);
        return fks;
    }

    bool has_fk(sqlite3*           db,
                const std::string& table_name,
                const std::string& from_col,
                const std::string& ref_table_prefix)
    {
        auto fks = get_foreign_keys(db, table_name);
        for(const auto& fk : fks)
        {
            if(fk.from_col == from_col &&
               fk.ref_table.find(ref_table_prefix) != std::string::npos)
                return true;
        }
        return false;
    }

    // PRAGMA index_list + PRAGMA index_info for UNIQUE constraints
    bool column_is_unique(sqlite3*           db,
                          const std::string& table_name,
                          const std::string& column_name)
    {
        std::string   query = "PRAGMA index_list(" + table_name + ")";
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, nullptr);
        std::vector<std::pair<std::string, bool>> indexes;
        while(sqlite3_step(stmt) == SQLITE_ROW)
        {
            const char* idx_name =
                reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            bool unique = (sqlite3_column_int(stmt, 2) != 0);
            if(idx_name) indexes.emplace_back(idx_name, unique);
        }
        sqlite3_finalize(stmt);

        for(const auto& [idx_name, unique] : indexes)
        {
            if(!unique) continue;
            std::string   idx_query = "PRAGMA index_info(" + idx_name + ")";
            sqlite3_stmt* idx_stmt;
            sqlite3_prepare_v2(db, idx_query.c_str(), -1, &idx_stmt, nullptr);
            while(sqlite3_step(idx_stmt) == SQLITE_ROW)
            {
                const char* col_name =
                    reinterpret_cast<const char*>(sqlite3_column_text(idx_stmt, 2));
                if(col_name && column_name == col_name)
                {
                    sqlite3_finalize(idx_stmt);
                    return true;
                }
            }
            sqlite3_finalize(idx_stmt);
        }
        return false;
    }

    std::string m_db_path;
    std::string m_uuid;
};

// Test 1: rocpd_metadata table
TEST_F(SchemaV4ComprehensiveTest, metadata_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_metadata_" + m_uuid;
    // if(!table_exists(db, table_name))
    // {
    //     table_name = "rocpd_metadata";
    // }
    ASSERT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "tag"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));

    sqlite3_close(db);
}

// Test 2: rocpd_string table
TEST_F(SchemaV4ComprehensiveTest, string_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_string_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "string"));

    sqlite3_close(db);
}

// Test 3: rocpd_info_node table
TEST_F(SchemaV4ComprehensiveTest, info_node_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_node_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "hash"));
    EXPECT_TRUE(column_exists(db, table_name, "machine_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "system_name"));
    EXPECT_TRUE(column_exists(db, table_name, "hostname"));
    EXPECT_TRUE(column_exists(db, table_name, "release"));
    EXPECT_TRUE(column_exists(db, table_name, "version"));
    EXPECT_TRUE(column_exists(db, table_name, "hardware_name"));
    EXPECT_TRUE(column_exists(db, table_name, "domain_name"));

    sqlite3_close(db);
}

// Test 4: rocpd_info_process table
TEST_F(SchemaV4ComprehensiveTest, info_process_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_process_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "init"));
    EXPECT_TRUE(column_exists(db, table_name, "fini"));
    EXPECT_TRUE(column_exists(db, table_name, "start"));
    EXPECT_TRUE(column_exists(db, table_name, "end"));
    EXPECT_TRUE(column_exists(db, table_name, "command"));
    EXPECT_TRUE(column_exists(db, table_name, "environment"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 5: rocpd_info_thread table
TEST_F(SchemaV4ComprehensiveTest, info_thread_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_thread_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "tid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "start"));
    EXPECT_TRUE(column_exists(db, table_name, "end"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 6: rocpd_info_category table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_category_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_category_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 7: rocpd_info_agent table
TEST_F(SchemaV4ComprehensiveTest, info_agent_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_agent_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "absolute_index"));
    EXPECT_TRUE(column_exists(db, table_name, "logical_index"));
    EXPECT_TRUE(column_exists(db, table_name, "type_index"));
    EXPECT_TRUE(column_exists(db, table_name, "uuid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "generic_name"));  // V4: replaces user_name
    EXPECT_TRUE(column_exists(db, table_name, "model_name"));
    EXPECT_TRUE(column_exists(db, table_name, "vendor_name"));
    EXPECT_TRUE(column_exists(db, table_name, "product_name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had "user_name", V4 doesn't
    EXPECT_FALSE(column_exists(db, table_name, "user_name"));

    sqlite3_close(db);
}

// Test 8: rocpd_info_queue table
TEST_F(SchemaV4ComprehensiveTest, info_queue_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_queue_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 9: rocpd_info_stream table
TEST_F(SchemaV4ComprehensiveTest, info_stream_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_stream_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 10: rocpd_info_pmc table
TEST_F(SchemaV4ComprehensiveTest, info_pmc_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_pmc_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "target_arch"));
    EXPECT_TRUE(column_exists(db, table_name, "event_code"));
    EXPECT_TRUE(column_exists(db, table_name, "instance_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "symbol"));
    EXPECT_TRUE(column_exists(db, table_name, "qualifier"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "description"));
    EXPECT_TRUE(column_exists(db, table_name, "long_description"));
    EXPECT_TRUE(column_exists(db, table_name, "component"));
    EXPECT_TRUE(column_exists(db, table_name, "units"));
    EXPECT_TRUE(column_exists(db, table_name, "value_type"));
    EXPECT_TRUE(column_exists(db, table_name, "block"));
    EXPECT_TRUE(column_exists(db, table_name, "expression"));
    EXPECT_TRUE(column_exists(db, table_name, "is_constant"));
    EXPECT_TRUE(column_exists(db, table_name, "is_derived"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 11: rocpd_info_code_object table
TEST_F(SchemaV4ComprehensiveTest, info_code_object_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_code_object_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "uri"));
    EXPECT_TRUE(column_exists(db, table_name, "load_base"));
    EXPECT_TRUE(column_exists(db, table_name, "load_size"));
    EXPECT_TRUE(column_exists(db, table_name, "load_delta"));
    EXPECT_TRUE(column_exists(db, table_name, "storage_type"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 12: rocpd_info_kernel_symbol table
TEST_F(SchemaV4ComprehensiveTest, info_kernel_symbol_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_kernel_symbol_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "code_object_id"));
    EXPECT_TRUE(column_exists(db, table_name, "kernel_name"));
    EXPECT_TRUE(column_exists(db, table_name, "display_name"));
    EXPECT_TRUE(column_exists(db, table_name, "kernel_object"));
    EXPECT_TRUE(column_exists(db, table_name, "kernarg_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "kernarg_segment_alignment"));
    EXPECT_TRUE(column_exists(db, table_name, "group_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "private_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "sgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "arch_vgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "accum_vgpr_count"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 13: rocpd_info_address_range table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_address_range_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_address_range_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "address_base"));
    EXPECT_TRUE(column_exists(db, table_name, "address_low"));
    EXPECT_TRUE(column_exists(db, table_name, "address_high"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 14: rocpd_info_source_code table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_source_code_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_source_code_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "address_id"));
    EXPECT_TRUE(column_exists(db, table_name, "file"));
    EXPECT_TRUE(column_exists(db, table_name, "line_number"));
    EXPECT_TRUE(column_exists(db, table_name, "lines"));
    EXPECT_TRUE(column_exists(db, table_name, "instructions"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 15: rocpd_info_pc table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, info_pc_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_info_pc_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "function"));
    EXPECT_TRUE(column_exists(db, table_name, "address_id"));
    EXPECT_TRUE(column_exists(db, table_name, "file"));
    EXPECT_TRUE(column_exists(db, table_name, "line"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 16: rocpd_track table (V4 enhanced)
TEST_F(SchemaV4ComprehensiveTest, track_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_track_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (enhanced from V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "nid"));
    EXPECT_TRUE(column_exists(db, table_name, "ppid"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "pid"));
    EXPECT_TRUE(column_exists(db, table_name, "tid"));
    EXPECT_TRUE(column_exists(db, table_name, "agent_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "queue_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "stream_id"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 17: rocpd_timestamp table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, timestamp_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_timestamp_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 for timestamp normalization)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "phase"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));

    sqlite3_close(db);
}

// Test 18: rocpd_event table
TEST_F(SchemaV4ComprehensiveTest, event_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_event_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "category_id"));
    EXPECT_TRUE(column_exists(db, table_name, "stack_id"));
    EXPECT_TRUE(column_exists(db, table_name, "parent_stack_id"));
    EXPECT_TRUE(column_exists(db, table_name, "correlation_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had embedded call_stack and line_info JSONB, V4 doesn't
    EXPECT_FALSE(column_exists(db, table_name, "call_stack"));
    EXPECT_FALSE(column_exists(db, table_name, "line_info"));

    sqlite3_close(db);
}

// Test 19: rocpd_arg table
TEST_F(SchemaV4ComprehensiveTest, arg_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_arg_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "position"));
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "name"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 20: rocpd_line_info table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, line_info_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_line_info_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 - normalized from V3's embedded JSONB)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "source_code_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 21: rocpd_call_stack table (V4 new)
TEST_F(SchemaV4ComprehensiveTest, call_stack_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_call_stack_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (new in V4 - normalized from V3's embedded JSONB)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "depth"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 22: rocpd_pmc_event table
TEST_F(SchemaV4ComprehensiveTest, pmc_event_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_pmc_event_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns (same as V3)
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "pmc_id"));
    EXPECT_TRUE(column_exists(db, table_name, "value"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    sqlite3_close(db);
}

// Test 23: rocpd_region table
TEST_F(SchemaV4ComprehensiveTest, region_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_region_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized with foreign keys
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(
        column_exists(db, table_name, "start_id"));        // V4: foreign key to timestamp
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));  // V4: foreign key to timestamp
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and nid/pid/tid
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));

    sqlite3_close(db);
}

// Test 24: rocpd_sample table
TEST_F(SchemaV4ComprehensiveTest, sample_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_sample_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));       // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "timestamp_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamp
    EXPECT_FALSE(column_exists(db, table_name, "timestamp"));

    sqlite3_close(db);
}

// Test 25: rocpd_kernel_dispatch table
TEST_F(SchemaV4ComprehensiveTest, kernel_dispatch_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_kernel_dispatch_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "kernel_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dispatch_id"));
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "private_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "group_segment_size"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_x"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_y"));
    EXPECT_TRUE(column_exists(db, table_name, "workgroup_size_z"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_x"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_y"));
    EXPECT_TRUE(column_exists(db, table_name, "grid_size_z"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "agent_id"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}

// Test 26: rocpd_memory_copy table
TEST_F(SchemaV4ComprehensiveTest, memory_copy_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_memory_copy_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dst_agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "dst_address"));
    EXPECT_TRUE(column_exists(db, table_name, "src_agent_id"));
    EXPECT_TRUE(column_exists(db, table_name, "src_address"));
    EXPECT_TRUE(column_exists(db, table_name, "size"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}

// Test 27: rocpd_memory_allocate table
TEST_F(SchemaV4ComprehensiveTest, memory_allocate_table_complete)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);

    std::string table_name = "rocpd_memory_allocate_" + m_uuid;
    EXPECT_TRUE(table_exists(db, table_name));

    // V4 columns - normalized
    EXPECT_TRUE(column_exists(db, table_name, "id"));
    EXPECT_TRUE(column_exists(db, table_name, "guid"));
    EXPECT_TRUE(column_exists(db, table_name, "track_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "type"));
    EXPECT_TRUE(column_exists(db, table_name, "level"));
    EXPECT_TRUE(column_exists(db, table_name, "start_id"));  // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "end_id"));    // V4: foreign key
    EXPECT_TRUE(column_exists(db, table_name, "name_id"));   // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "address"));
    EXPECT_TRUE(column_exists(db, table_name, "size"));
    EXPECT_TRUE(column_exists(db, table_name, "region_name_id"));  // V4 added
    EXPECT_TRUE(column_exists(db, table_name, "event_id"));
    EXPECT_TRUE(column_exists(db, table_name, "extdata"));

    // V3 had direct timestamps and direct foreign keys
    EXPECT_FALSE(column_exists(db, table_name, "start"));
    EXPECT_FALSE(column_exists(db, table_name, "end"));
    EXPECT_FALSE(column_exists(db, table_name, "nid"));
    EXPECT_FALSE(column_exists(db, table_name, "pid"));
    EXPECT_FALSE(column_exists(db, table_name, "tid"));
    EXPECT_FALSE(column_exists(db, table_name, "agent_id"));
    EXPECT_FALSE(column_exists(db, table_name, "queue_id"));
    EXPECT_FALSE(column_exists(db, table_name, "stream_id"));

    sqlite3_close(db);
}

// ============================================================================
// CONSTRAINT VALIDATION TESTS
// Each test validates PK, FK, UK, NOT NULL, and DEFAULT for one table
// ============================================================================

TEST_F(SchemaV4ComprehensiveTest, metadata_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_metadata_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "tag"));
    EXPECT_TRUE(column_is_notnull(db, t, "value"));

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, string_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_string_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "string"));
    EXPECT_TRUE(column_is_unique(db, t, "string"));

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_node_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_node_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "hash"));
    EXPECT_TRUE(column_is_notnull(db, t, "machine_id"));
    EXPECT_TRUE(column_is_unique(db, t, "hash"));
    EXPECT_TRUE(column_is_unique(db, t, "machine_id"));

    EXPECT_FALSE(column_is_notnull(db, t, "name"));
    EXPECT_FALSE(column_is_notnull(db, t, "system_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "hostname"));
    EXPECT_FALSE(column_is_notnull(db, t, "release"));
    EXPECT_FALSE(column_is_notnull(db, t, "version"));
    EXPECT_FALSE(column_is_notnull(db, t, "hardware_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "domain_name"));

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_process_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_process_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "environment"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "ppid"));
    EXPECT_FALSE(column_is_notnull(db, t, "name"));
    EXPECT_FALSE(column_is_notnull(db, t, "init"));
    EXPECT_FALSE(column_is_notnull(db, t, "fini"));
    EXPECT_FALSE(column_is_notnull(db, t, "start"));
    EXPECT_FALSE(column_is_notnull(db, t, "end"));
    EXPECT_FALSE(column_is_notnull(db, t, "command"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_EQ(column_default(db, t, "environment"), "\"{}\"");
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_thread_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_thread_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "tid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "ppid"));
    EXPECT_FALSE(column_is_notnull(db, t, "name"));
    EXPECT_FALSE(column_is_notnull(db, t, "start"));
    EXPECT_FALSE(column_is_notnull(db, t, "end"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_category_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_category_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "name"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_agent_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_agent_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "type"));
    EXPECT_FALSE(column_is_notnull(db, t, "absolute_index"));
    EXPECT_FALSE(column_is_notnull(db, t, "logical_index"));
    EXPECT_FALSE(column_is_notnull(db, t, "type_index"));
    EXPECT_FALSE(column_is_notnull(db, t, "uuid"));
    EXPECT_FALSE(column_is_notnull(db, t, "name"));
    EXPECT_FALSE(column_is_notnull(db, t, "generic_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "model_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "vendor_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "product_name"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_queue_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_queue_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));
    EXPECT_FALSE(column_is_notnull(db, t, "name"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_stream_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_stream_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));
    EXPECT_FALSE(column_is_notnull(db, t, "name"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_pmc_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_pmc_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "name"));
    EXPECT_TRUE(column_is_notnull(db, t, "symbol"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "agent_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "target_arch"));
    EXPECT_FALSE(column_is_notnull(db, t, "event_code"));
    EXPECT_FALSE(column_is_notnull(db, t, "instance_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "qualifier"));
    EXPECT_FALSE(column_is_notnull(db, t, "description"));
    EXPECT_FALSE(column_is_notnull(db, t, "component"));
    EXPECT_FALSE(column_is_notnull(db, t, "block"));
    EXPECT_FALSE(column_is_notnull(db, t, "expression"));
    EXPECT_FALSE(column_is_notnull(db, t, "is_constant"));
    EXPECT_FALSE(column_is_notnull(db, t, "is_derived"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "agent_id", "rocpd_info_agent"));
    EXPECT_EQ(column_default(db, t, "long_description"), "\"\"");
    EXPECT_EQ(column_default(db, t, "units"), "\"\"");
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_code_object_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_code_object_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "agent_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "uri"));
    EXPECT_FALSE(column_is_notnull(db, t, "load_base"));
    EXPECT_FALSE(column_is_notnull(db, t, "load_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "load_delta"));
    EXPECT_FALSE(column_is_notnull(db, t, "storage_type"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "agent_id", "rocpd_info_agent"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_kernel_symbol_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_kernel_symbol_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "code_object_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "kernel_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "display_name"));
    EXPECT_FALSE(column_is_notnull(db, t, "kernel_object"));
    EXPECT_FALSE(column_is_notnull(db, t, "kernarg_segment_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "kernarg_segment_alignment"));
    EXPECT_FALSE(column_is_notnull(db, t, "group_segment_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "private_segment_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "sgpr_count"));
    EXPECT_FALSE(column_is_notnull(db, t, "arch_vgpr_count"));
    EXPECT_FALSE(column_is_notnull(db, t, "accum_vgpr_count"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "code_object_id", "rocpd_info_code_object"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_address_range_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_address_range_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "address_base"));
    EXPECT_FALSE(column_is_notnull(db, t, "address_low"));
    EXPECT_FALSE(column_is_notnull(db, t, "address_high"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_source_code_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_source_code_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "lines"));
    EXPECT_TRUE(column_is_notnull(db, t, "instructions"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "address_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "file"));
    EXPECT_FALSE(column_is_notnull(db, t, "line_number"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "address_id", "rocpd_info_address_range"));
    EXPECT_EQ(column_default(db, t, "lines"), "\"[]\"");
    EXPECT_EQ(column_default(db, t, "instructions"), "\"[]\"");
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, info_pc_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_info_pc_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pid"));
    EXPECT_TRUE(column_is_notnull(db, t, "function"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "address_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "file"));
    EXPECT_FALSE(column_is_notnull(db, t, "line"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "address_id", "rocpd_info_address_range"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, track_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_track_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "nid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "ppid"));
    EXPECT_FALSE(column_is_notnull(db, t, "pid"));
    EXPECT_FALSE(column_is_notnull(db, t, "tid"));
    EXPECT_FALSE(column_is_notnull(db, t, "agent_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "queue_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "stream_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "name_id"));

    EXPECT_TRUE(has_fk(db, t, "nid", "rocpd_info_node"));
    EXPECT_TRUE(has_fk(db, t, "pid", "rocpd_info_process"));
    EXPECT_TRUE(has_fk(db, t, "tid", "rocpd_info_thread"));
    EXPECT_TRUE(has_fk(db, t, "agent_id", "rocpd_info_agent"));
    EXPECT_TRUE(has_fk(db, t, "queue_id", "rocpd_info_queue"));
    EXPECT_TRUE(has_fk(db, t, "stream_id", "rocpd_info_stream"));
    EXPECT_TRUE(has_fk(db, t, "name_id", "rocpd_string"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, timestamp_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_timestamp_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "value"));

    EXPECT_FALSE(column_is_notnull(db, t, "phase"));
    EXPECT_FALSE(column_is_notnull(db, t, "track_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, event_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_event_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "category_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "stack_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "parent_stack_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "correlation_id"));

    EXPECT_TRUE(has_fk(db, t, "category_id", "rocpd_info_category"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, arg_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_arg_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "event_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "position"));
    EXPECT_TRUE(column_is_notnull(db, t, "type"));
    EXPECT_TRUE(column_is_notnull(db, t, "name"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "value"));

    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, line_info_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_line_info_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "event_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "source_code_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "pc_id"));

    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_TRUE(has_fk(db, t, "source_code_id", "rocpd_info_source_code"));
    EXPECT_TRUE(has_fk(db, t, "pc_id", "rocpd_info_pc"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, call_stack_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_call_stack_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "event_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "depth"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "pc_id"));

    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_TRUE(has_fk(db, t, "pc_id", "rocpd_info_pc"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, pmc_event_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_pmc_event_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "pmc_id"));

    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "value"));

    EXPECT_TRUE(has_fk(db, t, "pmc_id", "rocpd_info_pmc"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "value"), "0.0");
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, region_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_region_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "track_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "name_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "start_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "end_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));
    EXPECT_TRUE(has_fk(db, t, "name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "start_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "end_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, sample_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_sample_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "track_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "name_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "timestamp_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));
    EXPECT_TRUE(has_fk(db, t, "name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "timestamp_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, kernel_dispatch_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_kernel_dispatch_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "track_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "kernel_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "dispatch_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "start_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "end_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "workgroup_size_x"));
    EXPECT_TRUE(column_is_notnull(db, t, "workgroup_size_y"));
    EXPECT_TRUE(column_is_notnull(db, t, "workgroup_size_z"));
    EXPECT_TRUE(column_is_notnull(db, t, "grid_size_x"));
    EXPECT_TRUE(column_is_notnull(db, t, "grid_size_y"));
    EXPECT_TRUE(column_is_notnull(db, t, "grid_size_z"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "private_segment_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "group_segment_size"));
    EXPECT_FALSE(column_is_notnull(db, t, "region_name_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));
    EXPECT_TRUE(has_fk(db, t, "kernel_id", "rocpd_info_kernel_symbol"));
    EXPECT_TRUE(has_fk(db, t, "start_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "end_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "region_name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, memory_copy_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_memory_copy_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "track_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "start_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "end_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "name_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "size"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "dst_agent_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "dst_address"));
    EXPECT_FALSE(column_is_notnull(db, t, "src_agent_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "src_address"));
    EXPECT_FALSE(column_is_notnull(db, t, "region_name_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));
    EXPECT_TRUE(has_fk(db, t, "start_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "end_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "dst_agent_id", "rocpd_info_agent"));
    EXPECT_TRUE(has_fk(db, t, "src_agent_id", "rocpd_info_agent"));
    EXPECT_TRUE(has_fk(db, t, "region_name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}

TEST_F(SchemaV4ComprehensiveTest, memory_allocate_constraints)
{
    sqlite3* db = open_db();
    ASSERT_NE(db, nullptr);
    std::string t = "rocpd_memory_allocate_" + m_uuid;

    EXPECT_TRUE(column_is_pk(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "id"));
    EXPECT_TRUE(column_is_notnull(db, t, "guid"));
    EXPECT_TRUE(column_is_notnull(db, t, "track_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "start_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "end_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "name_id"));
    EXPECT_TRUE(column_is_notnull(db, t, "size"));
    EXPECT_TRUE(column_is_notnull(db, t, "extdata"));

    EXPECT_FALSE(column_is_notnull(db, t, "type"));
    EXPECT_FALSE(column_is_notnull(db, t, "level"));
    EXPECT_FALSE(column_is_notnull(db, t, "address"));
    EXPECT_FALSE(column_is_notnull(db, t, "region_name_id"));
    EXPECT_FALSE(column_is_notnull(db, t, "event_id"));

    EXPECT_TRUE(has_fk(db, t, "track_id", "rocpd_track"));
    EXPECT_TRUE(has_fk(db, t, "start_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "end_id", "rocpd_timestamp"));
    EXPECT_TRUE(has_fk(db, t, "name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "region_name_id", "rocpd_string"));
    EXPECT_TRUE(has_fk(db, t, "event_id", "rocpd_event"));
    EXPECT_EQ(column_default(db, t, "extdata"), "\"{}\"");

    sqlite3_close(db);
}