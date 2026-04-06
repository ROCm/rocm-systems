// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocpdsna
{

struct version_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
};

/***
 * @brief Controls how the writer stores data during a profiling session.
 *
 * in_memory: All writes go to a SQLite in-memory database. Data is not
 *   visible on disk until flush_in_memory_data_to_disk() is called.
 *   Lower per-insert overhead; higher peak memory usage.
 *
 * on_disk: The database is opened directly on disk in WAL mode. Writes
 *   go to the WAL file via normal I/O unless wal_mmap_size > 0 is passed
 *   to the storage_t constructor, in which case the WAL file is
 *   pre-allocated and memory-mapped for faster sequential appends.
 *   flush_in_memory_data_to_disk() is a no-op in this mode.
 *   Supports concurrent readers via IPC.
 */
enum class write_mode_t
{
    in_memory,
    on_disk
};
}  // namespace rocpdsna
