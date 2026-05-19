// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "event_buffer_vtab.hpp"

#include "event_buffer.hpp"

#include "debug.hpp"

#include <sqlite3.h>

#include <cstring>
#include <memory>
#include <new>
#include <string>

namespace profiler_hub::data_storage::vtable
{

namespace
{

struct event_vtab : public sqlite3_vtab
{
    sqlite3*                      db = nullptr;
    std::string                   real_table_name;
    std::unique_ptr<event_buffer> buffer;
};

struct event_cursor : public sqlite3_vtab_cursor
{
    sqlite3_int64 rowid = 0;
};

int
xCreateOrConnect(sqlite3* db,
                 void* /*aux*/,
                 int                argc,
                 const char* const* argv,
                 sqlite3_vtab**     out_vtab,
                 char**             pzErr)
{
    if(argc < 4)
    {
        *pzErr = sqlite3_mprintf("event_buffer requires one argument: real table name");
        return SQLITE_ERROR;
    }
    std::string real = argv[3];
    if(real.size() >= 2 && (real.front() == '\'' || real.front() == '"') &&
       real.front() == real.back())
    {
        real = real.substr(1, real.size() - 2);
    }

    std::string create_sql = "CREATE TABLE x("
                             "id INTEGER, category_id INTEGER, stack_id INTEGER, "
                             "parent_stack_id INTEGER, correlation_id INTEGER, "
                             "call_stack TEXT, line_info TEXT, extdata TEXT"
                             ")";
    int         rc         = sqlite3_declare_vtab(db, create_sql.c_str());
    if(rc != SQLITE_OK)
    {
        *pzErr = sqlite3_mprintf("declare_vtab failed: %s", sqlite3_errmsg(db));
        return rc;
    }

    auto* vtab = new(std::nothrow) event_vtab();
    if(vtab == nullptr) return SQLITE_NOMEM;
    std::memset(static_cast<sqlite3_vtab*>(vtab), 0, sizeof(sqlite3_vtab));
    vtab->db              = db;
    vtab->real_table_name = real;

    vtab->buffer = std::make_unique<event_buffer>(vtab->real_table_name, db);
    event_buffer::register_instance(vtab->real_table_name, vtab->buffer.get());

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
    auto* vtab = static_cast<event_vtab*>(p);
    if(vtab != nullptr)
    {
        event_buffer::unregister_instance(vtab->real_table_name);
        vtab->buffer.reset();
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
    info->estimatedCost = 1e9;
    info->estimatedRows = 0;
    return SQLITE_OK;
}

int
xOpen(sqlite3_vtab* /*p*/, sqlite3_vtab_cursor** out)
{
    auto* cur = new(std::nothrow) event_cursor();
    if(cur == nullptr) return SQLITE_NOMEM;
    std::memset(static_cast<sqlite3_vtab_cursor*>(cur), 0, sizeof(sqlite3_vtab_cursor));
    *out = cur;
    return SQLITE_OK;
}

int
xClose(sqlite3_vtab_cursor* c)
{
    delete static_cast<event_cursor*>(c);
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
    static_cast<event_cursor*>(c)->rowid++;
    return SQLITE_OK;
}

int
xEof(sqlite3_vtab_cursor* /*c*/)
{
    return 1;
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
    *out = static_cast<event_cursor*>(c)->rowid;
    return SQLITE_OK;
}

int
xUpdate(sqlite3_vtab* p, int argc, sqlite3_value** argv, sqlite3_int64* out_rowid)
{
    auto* vtab = static_cast<event_vtab*>(p);

    if(argc == 1)
    {
        return SQLITE_READONLY;
    }
    if(sqlite3_value_type(argv[0]) != SQLITE_NULL)
    {
        return SQLITE_READONLY;
    }

    constexpr int expected_argc = 2 + static_cast<int>(event_buffer::k_total_columns);
    if(argc != expected_argc)
    {
        LOG_ERROR("vtable xUpdate: argc {} != expected {}", argc, expected_argc);
        return SQLITE_ERROR;
    }

    vtab->buffer->push_from_values(argv);
    if(out_rowid != nullptr)
        *out_rowid = static_cast<sqlite3_int64>(vtab->buffer->row_count());
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
register_event_buffer_module(sqlite3* db)
{
    int rc = sqlite3_create_module(db, "event_buffer", &k_module, nullptr);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("Failed to register event_buffer module: {}", sqlite3_errmsg(db));
    }
    return rc;
}

}  // namespace profiler_hub::data_storage::vtable
