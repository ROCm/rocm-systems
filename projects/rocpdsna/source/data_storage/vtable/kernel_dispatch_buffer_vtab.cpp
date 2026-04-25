// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "kernel_dispatch_buffer_vtab.hpp"

#include "debug.hpp"

#include <sqlite3.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <vector>

namespace rocpdsna::data_storage::vtable
{

namespace
{

// Column order matches the writer's INSERT for rocpd_kernel_dispatch.
// 21 int64 columns + 1 text column (extdata).
constexpr std::array<const char*, 22> k_column_names = {
    "id",
    "nid",
    "pid",
    "tid",
    "agent_id",
    "kernel_id",
    "dispatch_id",
    "queue_id",
    "stream_id",
    "start",
    "end",
    "private_segment_size",
    "group_segment_size",
    "workgroup_size_x",
    "workgroup_size_y",
    "workgroup_size_z",
    "grid_size_x",
    "grid_size_y",
    "grid_size_z",
    "region_name_id",
    "event_id",
    "extdata",
};

constexpr size_t k_int_column_count  = 21;  // indices 0..20
constexpr size_t k_text_column_index = 21;
constexpr size_t k_flush_threshold   = 100000;

// SQLite uses int64 sentinels for "missing" so we keep a parallel null-bitmap
// per int column.
struct int_column_t
{
    std::vector<int64_t> values;
    std::vector<uint8_t> is_null;
};

struct text_column_t
{
    std::vector<std::string> values;
    std::vector<uint8_t>     is_null;
};

struct kernel_dispatch_vtab : public sqlite3_vtab
{
    sqlite3*      db          = nullptr;  // owning connection
    sqlite3*      writer_conn = nullptr;  // bulk-write conn
    std::string   real_table_name;
    std::string   db_path;
    std::string   insert_sql;
    sqlite3_stmt* insert_stmt = nullptr;
    std::array<int_column_t, k_int_column_count> int_cols{};
    text_column_t                                text_col{};
    size_t                                       row_count = 0;
};

struct kernel_dispatch_cursor : public sqlite3_vtab_cursor
{
    // POC: cursor always reports EOF, no rows returned.
    sqlite3_int64 rowid = 0;
};

int
prepare_insert_stmt(kernel_dispatch_vtab* vtab)
{
    if(vtab->insert_stmt != nullptr) return SQLITE_OK;

    // Open a dedicated bulk-writer connection on the same database file.
    // We can't run BEGIN/COMMIT on the outer connection from inside xUpdate
    // (the outer connection is busy stepping the vtable INSERT), so the
    // bulk path uses its own connection.
    if(vtab->writer_conn == nullptr)
    {
        if(vtab->db_path.empty() || vtab->db_path == ":memory:")
        {
            LOG_ERROR("vtable: bulk writer requires on-disk db, got '{}'", vtab->db_path);
            return SQLITE_ERROR;
        }
        int rc = sqlite3_open(vtab->db_path.c_str(), &vtab->writer_conn);
        if(rc != SQLITE_OK)
        {
            LOG_ERROR("vtable: failed to open bulk writer conn for '{}': {}",
                      vtab->db_path,
                      sqlite3_errmsg(vtab->writer_conn));
            return rc;
        }
        // Match the writer connection PRAGMAs.
        sqlite3_exec(
            vtab->writer_conn, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);
        sqlite3_exec(
            vtab->writer_conn, "PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr);
        sqlite3_exec(
            vtab->writer_conn, "PRAGMA foreign_keys=OFF", nullptr, nullptr, nullptr);
        sqlite3_exec(
            vtab->writer_conn, "PRAGMA cache_size=-65536", nullptr, nullptr, nullptr);
        sqlite3_exec(
            vtab->writer_conn, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr);
        sqlite3_busy_timeout(vtab->writer_conn, 5000);
    }

    int rc = sqlite3_prepare_v2(
        vtab->writer_conn, vtab->insert_sql.c_str(), -1, &vtab->insert_stmt, nullptr);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("vtable: failed to prepare insert: {} sql={}",
                  sqlite3_errmsg(vtab->writer_conn),
                  vtab->insert_sql);
        return rc;
    }
    return SQLITE_OK;
}

int
flush_buffer(kernel_dispatch_vtab* vtab)
{
    if(vtab->row_count == 0) return SQLITE_OK;

    int rc = prepare_insert_stmt(vtab);
    if(rc != SQLITE_OK) return rc;

    // Bulk transaction on the dedicated writer connection.
    char* err = nullptr;
    rc        = sqlite3_exec(vtab->writer_conn, "BEGIN", nullptr, nullptr, &err);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("vtable: BEGIN failed: {}", err != nullptr ? err : "?");
        sqlite3_free(err);
        return rc;
    }

    sqlite3_stmt* stmt = vtab->insert_stmt;
    for(size_t r = 0; r < vtab->row_count; ++r)
    {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        for(size_t c = 0; c < k_int_column_count; ++c)
        {
            int pos = static_cast<int>(c) + 1;
            if(vtab->int_cols[c].is_null[r] != 0u)
            {
                sqlite3_bind_null(stmt, pos);
            }
            else
            {
                sqlite3_bind_int64(stmt, pos, vtab->int_cols[c].values[r]);
            }
        }
        const int text_pos = static_cast<int>(k_text_column_index) + 1;
        if(vtab->text_col.is_null[r] != 0u)
        {
            sqlite3_bind_null(stmt, text_pos);
        }
        else
        {
            const std::string& s = vtab->text_col.values[r];
            sqlite3_bind_text(
                stmt, text_pos, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
        }

        rc = sqlite3_step(stmt);
        if(rc != SQLITE_DONE)
        {
            LOG_ERROR("vtable: step failed at row {}: {}",
                      r,
                      sqlite3_errmsg(vtab->writer_conn));
            sqlite3_exec(vtab->writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
            return rc;
        }
    }

    rc = sqlite3_exec(vtab->writer_conn, "COMMIT", nullptr, nullptr, &err);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("vtable: COMMIT failed: {}", err != nullptr ? err : "?");
        sqlite3_free(err);
        sqlite3_exec(vtab->writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    for(auto& col : vtab->int_cols)
    {
        col.values.clear();
        col.is_null.clear();
    }
    vtab->text_col.values.clear();
    vtab->text_col.is_null.clear();
    vtab->row_count = 0;
    return SQLITE_OK;
}

int
xCreateOrConnect(sqlite3* db,
                 void* /*aux*/,
                 int                argc,
                 const char* const* argv,
                 sqlite3_vtab**     out_vtab,
                 char**             pzErr)
{
    // argv[0]=module, argv[1]=db, argv[2]=table, argv[3..]=args.
    if(argc < 4)
    {
        *pzErr = sqlite3_mprintf(
            "kernel_dispatch_buffer requires one argument: real table name");
        return SQLITE_ERROR;
    }
    std::string real = argv[3];
    // Strip surrounding quotes if any.
    if(real.size() >= 2 && (real.front() == '\'' || real.front() == '"') &&
       real.front() == real.back())
    {
        real = real.substr(1, real.size() - 2);
    }

    std::string create_sql =
        "CREATE TABLE x("
        "id INTEGER, nid INTEGER, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start INTEGER, \"end\" INTEGER, private_segment_size INTEGER, "
        "group_segment_size INTEGER, workgroup_size_x INTEGER, workgroup_size_y INTEGER, "
        "workgroup_size_z INTEGER, grid_size_x INTEGER, grid_size_y INTEGER, "
        "grid_size_z INTEGER, region_name_id INTEGER, event_id INTEGER, extdata TEXT"
        ")";
    int rc = sqlite3_declare_vtab(db, create_sql.c_str());
    if(rc != SQLITE_OK)
    {
        *pzErr = sqlite3_mprintf("declare_vtab failed: %s", sqlite3_errmsg(db));
        return rc;
    }

    auto* vtab = new(std::nothrow) kernel_dispatch_vtab();
    if(vtab == nullptr) return SQLITE_NOMEM;
    std::memset(static_cast<sqlite3_vtab*>(vtab), 0, sizeof(sqlite3_vtab));
    vtab->db              = db;
    vtab->real_table_name = real;
    const char* path      = sqlite3_db_filename(db, "main");
    vtab->db_path         = (path != nullptr) ? std::string(path) : std::string{};

    std::string columns;
    std::string placeholders;
    for(size_t i = 0; i < k_column_names.size(); ++i)
    {
        if(i != 0)
        {
            columns += ',';
            placeholders += ',';
        }
        columns += '"';
        columns += k_column_names[i];
        columns += '"';
        placeholders += '?';
    }
    vtab->insert_sql =
        "INSERT INTO \"" + real + "\" (" + columns + ") VALUES (" + placeholders + ")";

    *out_vtab = vtab;
    return SQLITE_OK;
}

int
xCreate(sqlite3*           db,
        void*              aux,
        int                argc,
        const char* const* argv,
        sqlite3_vtab**     out_vtab,
        char**             pzErr)
{
    return xCreateOrConnect(db, aux, argc, argv, out_vtab, pzErr);
}

int
xConnect(sqlite3*           db,
         void*              aux,
         int                argc,
         const char* const* argv,
         sqlite3_vtab**     out_vtab,
         char**             pzErr)
{
    return xCreateOrConnect(db, aux, argc, argv, out_vtab, pzErr);
}

int
xDisconnect(sqlite3_vtab* p)
{
    auto* vtab = static_cast<kernel_dispatch_vtab*>(p);
    if(vtab != nullptr)
    {
        flush_buffer(vtab);
        if(vtab->insert_stmt != nullptr) sqlite3_finalize(vtab->insert_stmt);
        if(vtab->writer_conn != nullptr) sqlite3_close(vtab->writer_conn);
        delete vtab;
    }
    return SQLITE_OK;
}

int
xDestroy(sqlite3_vtab* p)
{
    return xDisconnect(p);
}

int
xBestIndex(sqlite3_vtab* /*p*/, sqlite3_index_info* info)
{
    info->estimatedCost = 1e9;  // discourage scans
    info->estimatedRows = 0;
    return SQLITE_OK;
}

int
xOpen(sqlite3_vtab* /*p*/, sqlite3_vtab_cursor** out)
{
    auto* cur = new(std::nothrow) kernel_dispatch_cursor();
    if(cur == nullptr) return SQLITE_NOMEM;
    std::memset(static_cast<sqlite3_vtab_cursor*>(cur), 0, sizeof(sqlite3_vtab_cursor));
    *out = cur;
    return SQLITE_OK;
}

int
xClose(sqlite3_vtab_cursor* c)
{
    delete static_cast<kernel_dispatch_cursor*>(c);
    return SQLITE_OK;
}

int
xFilter(sqlite3_vtab_cursor* /*c*/,
        int /*idx*/,
        const char* /*idxStr*/,
        int /*argc*/,
        sqlite3_value** /*argv*/)
{
    return SQLITE_OK;
}

int
xNext(sqlite3_vtab_cursor* c)
{
    static_cast<kernel_dispatch_cursor*>(c)->rowid++;
    return SQLITE_OK;
}

int
xEof(sqlite3_vtab_cursor* /*c*/)
{
    return 1;  // POC: never returns rows from buffer.
}

int
xColumn(sqlite3_vtab_cursor* /*c*/, sqlite3_context* ctx, int /*col*/)
{
    sqlite3_result_null(ctx);
    return SQLITE_OK;
}

int
xRowid(sqlite3_vtab_cursor* c, sqlite3_int64* out)
{
    *out = static_cast<kernel_dispatch_cursor*>(c)->rowid;
    return SQLITE_OK;
}

int
xUpdate(sqlite3_vtab* p, int argc, sqlite3_value** argv, sqlite3_int64* out_rowid)
{
    auto* vtab = static_cast<kernel_dispatch_vtab*>(p);

    // argc == 1: DELETE; argc > 1 with argv[0] != NULL: UPDATE; argv[0] == NULL: INSERT.
    if(argc == 1)
    {
        return SQLITE_READONLY;
    }
    if(sqlite3_value_type(argv[0]) != SQLITE_NULL)
    {
        return SQLITE_READONLY;
    }

    // INSERT: argv[1] is the rowid (or NULL), argv[2..] are the column values
    // in declared order.
    const int expected = 2 + static_cast<int>(k_column_names.size());
    if(argc != expected)
    {
        LOG_ERROR("vtable xUpdate: argc {} != expected {}", argc, expected);
        return SQLITE_ERROR;
    }

    for(size_t c = 0; c < k_int_column_count; ++c)
    {
        sqlite3_value* v   = argv[2 + static_cast<int>(c)];
        auto&          col = vtab->int_cols[c];
        if(sqlite3_value_type(v) == SQLITE_NULL)
        {
            col.values.push_back(0);
            col.is_null.push_back(1);
        }
        else
        {
            col.values.push_back(sqlite3_value_int64(v));
            col.is_null.push_back(0);
        }
    }
    {
        sqlite3_value* v = argv[2 + static_cast<int>(k_text_column_index)];
        if(sqlite3_value_type(v) == SQLITE_NULL)
        {
            vtab->text_col.values.emplace_back();
            vtab->text_col.is_null.push_back(1);
        }
        else
        {
            const auto* txt = reinterpret_cast<const char*>(sqlite3_value_text(v));
            const auto  n   = static_cast<size_t>(sqlite3_value_bytes(v));
            vtab->text_col.values.emplace_back(txt != nullptr ? std::string(txt, n)
                                                              : std::string{});
            vtab->text_col.is_null.push_back(0);
        }
    }
    vtab->row_count++;
    if(out_rowid != nullptr) *out_rowid = static_cast<sqlite3_int64>(vtab->row_count);

    // POC: don't flush from inside xUpdate. The outer connection holds locks
    // while sqlite3_step is in progress on the vtable INSERT, so a nested
    // BEGIN on a second connection self-deadlocks. Flush happens at
    // xDisconnect / xDestroy instead. Threshold retained for documentation.
    (void) k_flush_threshold;
    return SQLITE_OK;
}

const sqlite3_module k_module = {
    /* iVersion    */ 1,
    /* xCreate     */ xCreate,
    /* xConnect    */ xConnect,
    /* xBestIndex  */ xBestIndex,
    /* xDisconnect */ xDisconnect,
    /* xDestroy    */ xDestroy,
    /* xOpen       */ xOpen,
    /* xClose      */ xClose,
    /* xFilter     */ xFilter,
    /* xNext       */ xNext,
    /* xEof        */ xEof,
    /* xColumn     */ xColumn,
    /* xRowid      */ xRowid,
    /* xUpdate     */ xUpdate,
    /* xBegin      */ nullptr,
    /* xSync       */ nullptr,
    /* xCommit     */ nullptr,
    /* xRollback   */ nullptr,
    /* xFindFunction */ nullptr,
    /* xRename     */ nullptr,
    /* xSavepoint  */ nullptr,
    /* xRelease    */ nullptr,
    /* xRollbackTo */ nullptr,
    /* xShadowName */ nullptr,
};

}  // namespace

int
register_kernel_dispatch_buffer_module(sqlite3* db)
{
    int rc = sqlite3_create_module(db, "kernel_dispatch_buffer", &k_module, nullptr);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("Failed to register kernel_dispatch_buffer module: {}",
                  sqlite3_errmsg(db));
    }
    return rc;
}

}  // namespace rocpdsna::data_storage::vtable
