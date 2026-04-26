// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Tests that all vtable buffers share a single writer connection with the
// sqlite_backend. With 5 separate writer connections to a WAL-mode db, only
// one writer at a time may hold the write lock; concurrent flushes serialize
// via busy_timeout and each connection allocates its own page cache. The
// shared-connection refactor eliminates both issues.

#include "data_storage/backends/sqlite_backend.hpp"
#include "data_storage/vtable/kernel_dispatch_buffer.hpp"
#include "data_storage/vtable/memory_alloc_buffer.hpp"
#include "data_storage/vtable/memory_copy_buffer.hpp"
#include "data_storage/vtable/pmc_event_buffer.hpp"
#include "data_storage/vtable/region_buffer.hpp"

#include "rocpdsna/storage.hpp"
#include "rocpdsna/writer.hpp"
#include "rocpdsna/writer_types.hpp"

#include <gtest/gtest.h>

#include <spdlog/fmt/bundled/core.h>
#include <sqlite3.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

namespace
{

using namespace rocpdsna::writer_types;
using namespace rocpdsna::data_storage;

constexpr size_t k_node_id = 1;
constexpr size_t k_pid     = 1000;
constexpr size_t k_tid     = 1001;
constexpr size_t k_rows    = 50000;

// Bring the writer up to a state where every per-table buffer has been
// instantiated by the vtable layer (CREATE VIRTUAL TABLE during
// initialize_schema).
void
register_full_setup(rocpdsna::writer_t& writer, agent_unique_id_t& gpu_agent_out)
{
    writer.register_node_info(node_info_t{ .node_id       = k_node_id,
                                           .hash          = 0xDEADBEEFULL,
                                           .machine_id    = "test-machine",
                                           .system_name   = "Linux",
                                           .hostname      = "test-host",
                                           .release       = "6.0.0",
                                           .version       = "v1",
                                           .hardware_name = "x86_64",
                                           .domain_name   = "local" });

    writer.register_process_info(process_info_t{ .ppid        = 0,
                                                 .pid         = k_pid,
                                                 .init        = 0,
                                                 .fini        = 0,
                                                 .start       = 0,
                                                 .end         = 0,
                                                 .command     = "/bin/test",
                                                 .environment = "{}",
                                                 .extdata     = "{}",
                                                 .node_id     = k_node_id });

    writer.register_thread_info(thread_info_t{ .parent_process_id = k_pid,
                                               .thread_id         = k_tid,
                                               .name              = "main",
                                               .start             = 0,
                                               .end               = 0,
                                               .extdata           = "{}",
                                               .node_id           = k_node_id,
                                               .process_id        = k_pid });

    const agent_unique_id_t gpu_agent{ "GPU", 0 };
    writer.register_agent_info(agent_info_t{ .unique_id      = gpu_agent,
                                             .absolute_index = 0,
                                             .logical_index  = 0,
                                             .uuid           = 0xABCDULL,
                                             .name           = "gfx90a",
                                             .model_name     = "MI200",
                                             .vendor_name    = "AMD",
                                             .product_name   = "MI210",
                                             .user_name      = "",
                                             .extdata        = "{}",
                                             .node_id        = k_node_id,
                                             .process_id     = k_pid });

    writer.register_queue_info(queue_info_t{ .queue_id   = 1,
                                             .name       = "hsa_queue_0",
                                             .extdata    = "{}",
                                             .node_id    = k_node_id,
                                             .process_id = k_pid });

    writer.register_stream_info(stream_info_t{ .stream_id  = 1,
                                               .name       = "hip_stream_0",
                                               .extdata    = "{}",
                                               .node_id    = k_node_id,
                                               .process_id = k_pid });

    writer.register_code_object_info(code_object_info_t{ .id  = 1,
                                                         .uri = "file:///kernels.hsaco",
                                                         .load_base    = 0x10000,
                                                         .load_size    = 0x1000,
                                                         .load_delta   = 0,
                                                         .storage_type = "FILE",
                                                         .extdata      = "{}",
                                                         .node_id      = k_node_id,
                                                         .process_id   = k_pid,
                                                         .agent_id     = gpu_agent });

    writer.register_kernel_symbol_info(
        kernel_symbol_info_t{ .id                        = 1,
                              .name                      = "vectorAdd",
                              .display_name              = "vectorAdd",
                              .kernel_object             = 0x1234,
                              .kernarg_segment_size      = 256,
                              .kernarg_segment_alignment = 8,
                              .group_segment_size        = 65536,
                              .private_segment_size      = 0,
                              .sgpr_count                = 32,
                              .arch_vgpr_count           = 64,
                              .accum_vgpr_count          = 0,
                              .extdata                   = "{}",
                              .node_id                   = k_node_id,
                              .process_id                = k_pid,
                              .code_obj_id               = 1 });

    writer.register_track_info(track_info_t{ .name       = "gpu_kernel",
                                             .extdata    = "{}",
                                             .node_id    = k_node_id,
                                             .process_id = k_pid,
                                             .thread_id  = k_tid });

    gpu_agent_out = gpu_agent;
}

void
push_kernel_dispatch_rows(vtable::kernel_dispatch_buffer* buf, size_t n)
{
    buf->reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        buf->push(
            vtable::kernel_dispatch_row{ .id          = static_cast<int64_t>(i + 1),
                                         .nid         = 1,
                                         .pid         = 1,
                                         .tid         = 1,
                                         .agent_id    = 1,
                                         .kernel_id   = 1,
                                         .dispatch_id = static_cast<int64_t>(i),
                                         .queue_id    = 1,
                                         .stream_id   = 1,
                                         .start       = static_cast<int64_t>(i * 1000),
                                         .end = static_cast<int64_t>(i * 1000 + 500),
                                         .private_segment_size = 0,
                                         .group_segment_size   = 65536,
                                         .workgroup_size_x     = 256,
                                         .workgroup_size_y     = 1,
                                         .workgroup_size_z     = 1,
                                         .grid_size_x          = 1024,
                                         .grid_size_y          = 1,
                                         .grid_size_z          = 1,
                                         .region_name_id       = std::nullopt,
                                         .event_id             = std::nullopt,
                                         .extdata              = "{}" });
    }
}

void
push_memory_copy_rows(vtable::memory_copy_buffer* buf, size_t n)
{
    buf->reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        buf->push(
            vtable::memory_copy_row{ .id           = static_cast<int64_t>(i + 1),
                                     .nid          = 1,
                                     .pid          = 1,
                                     .tid          = 1,
                                     .start        = static_cast<int64_t>(i * 1000),
                                     .end          = static_cast<int64_t>(i * 1000 + 500),
                                     .name_id      = 1,
                                     .dst_agent_id = std::nullopt,
                                     .dst_address  = static_cast<int64_t>(0x10000 + i),
                                     .src_agent_id = std::nullopt,
                                     .src_address  = static_cast<int64_t>(0x20000 + i),
                                     .size         = 4096,
                                     .queue_id     = std::nullopt,
                                     .stream_id    = std::nullopt,
                                     .region_name_id = std::nullopt,
                                     .event_id       = std::nullopt,
                                     .extdata        = "{}" });
    }
}

void
push_memory_alloc_rows(vtable::memory_alloc_buffer* buf, size_t n)
{
    buf->reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        buf->push(vtable::memory_alloc_row{ .id       = static_cast<int64_t>(i + 1),
                                            .nid      = 1,
                                            .pid      = 1,
                                            .tid      = 1,
                                            .agent_id = 1,
                                            .type     = std::string_view{ "ALLOC" },
                                            .level    = std::string_view{ "REAL" },
                                            .start    = static_cast<int64_t>(i * 1000),
                                            .end = static_cast<int64_t>(i * 1000 + 500),
                                            .address  = static_cast<int64_t>(0x30000 + i),
                                            .size     = 4096,
                                            .queue_id = std::nullopt,
                                            .stream_id = std::nullopt,
                                            .event_id  = std::nullopt,
                                            .extdata   = "{}" });
    }
}

void
push_region_rows(vtable::region_buffer* buf, size_t n)
{
    buf->reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        buf->push(vtable::region_row{ .id       = static_cast<int64_t>(i + 1),
                                      .nid      = 1,
                                      .pid      = 1,
                                      .tid      = 1,
                                      .start    = static_cast<int64_t>(i * 1000),
                                      .end      = static_cast<int64_t>(i * 1000 + 500),
                                      .name_id  = 1,
                                      .event_id = std::nullopt,
                                      .extdata  = "{}" });
    }
}

void
push_pmc_event_rows(vtable::pmc_event_buffer* buf, size_t n)
{
    buf->reserve(n);
    for(size_t i = 0; i < n; ++i)
    {
        buf->push(vtable::pmc_event_row{ .id       = static_cast<int64_t>(i + 1),
                                         .event_id = std::nullopt,
                                         .pmc_id   = 1,
                                         .value    = static_cast<double>(i % 100),
                                         .extdata  = "{}" });
    }
}

class concurrent_flush_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        const std::string test_name{
            ::testing::UnitTest::GetInstance()->current_test_info()->name()
        };
        m_database_path = "test_concurrent_" + test_name + ".db";
        std::remove(m_database_path.c_str());

        m_storage = std::make_unique<rocpdsna::storage_t>(m_database_path,
                                                          std::string{ "concflush" });
        m_writer  = std::make_unique<rocpdsna::writer_t>(std::move(m_storage));
        register_full_setup(*m_writer, m_gpu_agent);

        const std::string uuid = "concflush";
        m_kd                   = vtable::kernel_dispatch_buffer::get_active_instance(
            "rocpd_kernel_dispatch_" + uuid);
        m_mc =
            vtable::memory_copy_buffer::get_active_instance("rocpd_memory_copy_" + uuid);
        m_ma = vtable::memory_alloc_buffer::get_active_instance("rocpd_memory_allocate_" +
                                                                uuid);
        m_rg = vtable::region_buffer::get_active_instance("rocpd_region_" + uuid);
        m_pm = vtable::pmc_event_buffer::get_active_instance("rocpd_pmc_event_" + uuid);

        ASSERT_NE(m_kd, nullptr);
        ASSERT_NE(m_mc, nullptr);
        ASSERT_NE(m_ma, nullptr);
        ASSERT_NE(m_rg, nullptr);
        ASSERT_NE(m_pm, nullptr);
    }

    void TearDown() override
    {
        m_writer.reset();
        m_storage.reset();
        std::remove(m_database_path.c_str());
        std::remove((m_database_path + "-wal").c_str());
        std::remove((m_database_path + "-shm").c_str());
    }

    std::string                          m_database_path;
    std::unique_ptr<rocpdsna::storage_t> m_storage;
    std::unique_ptr<rocpdsna::writer_t>  m_writer;
    agent_unique_id_t                    m_gpu_agent;

    vtable::kernel_dispatch_buffer* m_kd = nullptr;
    vtable::memory_copy_buffer*     m_mc = nullptr;
    vtable::memory_alloc_buffer*    m_ma = nullptr;
    vtable::region_buffer*          m_rg = nullptr;
    vtable::pmc_event_buffer*       m_pm = nullptr;
};

// ---------------------------------------------------------------------------
// 1. All buffers must share one writer connection with the backend.
//    Pre-refactor: each buffer opens its own sqlite3 -> all 5 writer_connection()
//    pointers differ. Post-refactor: one shared writer connection.
// ---------------------------------------------------------------------------
TEST_F(concurrent_flush_test, all_buffers_share_writer_connection)
{
    // Force each buffer to lazily realise its writer connection by flushing
    // a single pushed row. Empty flush is a no-op.
    push_kernel_dispatch_rows(m_kd, 1);
    push_memory_copy_rows(m_mc, 1);
    push_memory_alloc_rows(m_ma, 1);
    push_region_rows(m_rg, 1);
    push_pmc_event_rows(m_pm, 1);

    ASSERT_EQ(m_kd->flush(), SQLITE_OK);
    ASSERT_EQ(m_mc->flush(), SQLITE_OK);
    ASSERT_EQ(m_ma->flush(), SQLITE_OK);
    ASSERT_EQ(m_rg->flush(), SQLITE_OK);
    ASSERT_EQ(m_pm->flush(), SQLITE_OK);

    sqlite3* kd_conn = m_kd->writer_connection();
    sqlite3* mc_conn = m_mc->writer_connection();
    sqlite3* ma_conn = m_ma->writer_connection();
    sqlite3* rg_conn = m_rg->writer_connection();
    sqlite3* pm_conn = m_pm->writer_connection();

    ASSERT_NE(kd_conn, nullptr);
    EXPECT_EQ(kd_conn, mc_conn);
    EXPECT_EQ(kd_conn, ma_conn);
    EXPECT_EQ(kd_conn, rg_conn);
    EXPECT_EQ(kd_conn, pm_conn);
}

// ---------------------------------------------------------------------------
// 1b. Destructor must drain pending vtable buffers BEFORE closing the
//     SQLite connection. If sqlite3_close happens first, vtable xDisconnect
//     destroys each buffer; the buffer's flush() then issues
//     BEGIN/INSERT/COMMIT on a connection that is mid-close, which fails
//     and silently drops the rows.
// ---------------------------------------------------------------------------
TEST_F(concurrent_flush_test, destructor_flushes_buffers_before_closing_connection)
{
    push_kernel_dispatch_rows(m_kd, 1);
    push_memory_copy_rows(m_mc, 1);
    push_memory_alloc_rows(m_ma, 1);
    push_region_rows(m_rg, 1);
    push_pmc_event_rows(m_pm, 1);

    // Drop in-process pointers; they will dangle once storage is destroyed.
    m_kd = nullptr;
    m_mc = nullptr;
    m_ma = nullptr;
    m_rg = nullptr;
    m_pm = nullptr;

    // Trigger destruction. NO explicit flush -- the dtor must do it.
    m_writer.reset();
    m_storage.reset();

    // Reopen with a fresh sqlite3 handle and verify each table has the row.
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(m_database_path.c_str(), &db), SQLITE_OK);

    auto count_table = [&](const std::string& sql) -> int64_t {
        sqlite3_stmt* stmt = nullptr;
        EXPECT_EQ(sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr), SQLITE_OK);
        int64_t value = -1;
        if(sqlite3_step(stmt) == SQLITE_ROW) value = sqlite3_column_int64(stmt, 0);
        sqlite3_finalize(stmt);
        return value;
    };

    EXPECT_EQ(count_table("SELECT COUNT(*) FROM rocpd_kernel_dispatch_concflush"), 1)
        << "destructor must flush kernel_dispatch buffer before close";
    EXPECT_EQ(count_table("SELECT COUNT(*) FROM rocpd_memory_copy_concflush"), 1)
        << "destructor must flush memory_copy buffer before close";
    EXPECT_EQ(count_table("SELECT COUNT(*) FROM rocpd_memory_allocate_concflush"), 1)
        << "destructor must flush memory_alloc buffer before close";
    EXPECT_EQ(count_table("SELECT COUNT(*) FROM rocpd_region_concflush"), 1)
        << "destructor must flush region buffer before close";
    EXPECT_EQ(count_table("SELECT COUNT(*) FROM rocpd_pmc_event_concflush"), 1)
        << "destructor must flush pmc_event buffer before close";

    sqlite3_close(db);
}

// ---------------------------------------------------------------------------
// 2. Sequential drain of all 5 buffers must succeed and the wall time is
//    recorded for comparison against the pre-refactor baseline. This mirrors
//    the production drain order in sqlite_backend::flush(): the shared
//    writer connection only allows one transaction at a time, so flushes
//    are intrinsically sequential (this also removes the busy_timeout
//    contention path that existed when each buffer owned its own writer).
// ---------------------------------------------------------------------------
TEST_F(concurrent_flush_test, sequential_drain_succeeds_and_records_timing)
{
    push_kernel_dispatch_rows(m_kd, k_rows);
    push_memory_copy_rows(m_mc, k_rows);
    push_memory_alloc_rows(m_ma, k_rows);
    push_region_rows(m_rg, k_rows);
    push_pmc_event_rows(m_pm, k_rows);

    const auto t0 = std::chrono::steady_clock::now();

    EXPECT_EQ(m_kd->flush(), SQLITE_OK);
    EXPECT_EQ(m_mc->flush(), SQLITE_OK);
    EXPECT_EQ(m_ma->flush(), SQLITE_OK);
    EXPECT_EQ(m_rg->flush(), SQLITE_OK);
    EXPECT_EQ(m_pm->flush(), SQLITE_OK);

    const auto   t1      = std::chrono::steady_clock::now();
    const double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::fprintf(stderr,
                 "[shared_conn_drain] 5 buffers x %zu rows: wall_ms=%.2f\n",
                 k_rows,
                 wall_ms);
}

}  // namespace
