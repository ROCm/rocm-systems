// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sqlite3.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace profiler_hub::test
{

inline void
exec_sql(sqlite3* db, const char* sql)
{
    char* error = nullptr;
    if(sqlite3_exec(db, sql, nullptr, nullptr, &error) != SQLITE_OK)
    {
        const std::string message = error != nullptr ? error : "sqlite3_exec failed";
        sqlite3_free(error);
        throw std::runtime_error{ message };
    }
}

/** Writes a minimal rocmProfileData schema 2 file: one process, two threads, one
 *  kernel, one copy linked to the inner API call, and one GPU counter. */
inline void
write_schema2(const std::filesystem::path& path)
{
    sqlite3* db = nullptr;
    if(sqlite3_open(path.string().c_str(), &db) != SQLITE_OK)
    {
        const std::string message = sqlite3_errmsg(db);
        sqlite3_close(db);
        throw std::runtime_error{ message };
    }
    try
    {
        exec_sql(db, R"sql(
CREATE TABLE rocpd_metadata(tag TEXT, value TEXT);
INSERT INTO rocpd_metadata(tag, value) VALUES
    ('schema_version', '2.0.0'),
    ('schema_version_major', '2'),
    ('schema_version_minor', '0'),
    ('schema_version_patch', '0');

CREATE TABLE rocpd_string(id INTEGER, string TEXT);
INSERT INTO rocpd_string(id, string) VALUES
    (1, 'hipLaunchKernel'),
    (2, 'vector_add'),
    (3, 'Copy'),
    (4, 'CopyHostToDevice'),
    (5, 'memcpy'),
    (6, 'grid=1'),
    (8, 'Copy');

CREATE TABLE rocpd_api(
    id INTEGER, start INTEGER, "end" INTEGER, apiName_id INTEGER, args_id INTEGER,
    pid INTEGER, tid INTEGER);
INSERT INTO rocpd_api(id, start, "end", apiName_id, args_id, pid, tid) VALUES
    (1, 100, 500, 1, 6, 10, 20),
    (2, 150, 400, 5, NULL, 10, 20),
    (3, 200, 250, 1, NULL, 10, 21);

CREATE TABLE rocpd_op(
    id INTEGER, start INTEGER, "end" INTEGER, description_id INTEGER, opType_id INTEGER,
    gpuId INTEGER, queueId INTEGER);
INSERT INTO rocpd_op(id, start, "end", description_id, opType_id, gpuId, queueId) VALUES
    (10, 160, 300, 2, 1, 1, 2),
    (11, 180, 220, NULL, 4, 1, 2);

CREATE TABLE rocpd_api_ops(api_id INTEGER, op_id INTEGER);
INSERT INTO rocpd_api_ops(api_id, op_id) VALUES (2, 10), (2, 11);

CREATE TABLE rocpd_kernelapi(
    api_ptr_id INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER,
    workgroupX INTEGER, workgroupY INTEGER, workgroupZ INTEGER,
    groupSegmentSize INTEGER, privateSegmentSize INTEGER, kernelName_id INTEGER);
INSERT INTO rocpd_kernelapi VALUES (2, 8, 4, 1, 64, 1, 1, 256, 0, 2);

CREATE TABLE rocpd_copyapi(api_ptr_id INTEGER, size INTEGER, dst TEXT, src TEXT);
INSERT INTO rocpd_copyapi VALUES (2, 4096, '0x1000', 'nope');

CREATE TABLE rocpd_monitor(
    id INTEGER, deviceId INTEGER, monitorType TEXT, start INTEGER, "end" INTEGER,
    value TEXT);
INSERT INTO rocpd_monitor(id, deviceId, monitorType, start, "end", value) VALUES
    (1, 0, 'IGNORED', 50, 60, '9'),
    (2, 1, 'SCLK', 100, 110, '1.5'),
    (3, 1, 'SCLK', 300, 310, '2.5');
)sql");
    } catch(...)
    {
        sqlite3_close(db);
        throw;
    }
    sqlite3_close(db);
}

}  // namespace profiler_hub::test
