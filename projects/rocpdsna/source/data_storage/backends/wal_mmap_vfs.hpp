// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>

namespace rocpdsna::data_storage
{

// Registers (once per unique size) a custom SQLite VFS that intercepts WAL
// file opens and services them via a pre-allocated mmap region instead of
// read()/write() system calls.  All other file types (main database, shm)
// pass through to the default unix VFS unchanged.
//
// Returns the VFS name to pass as the last argument to sqlite3_open_v2().
// The returned pointer is valid for the lifetime of the process.
//
// If mmap_size == 0 this function must not be called; callers should pass
// nullptr to sqlite3_open_v2() to select the default VFS instead.
const char* register_wal_mmap_vfs(size_t mmap_size);

}  // namespace rocpdsna::data_storage
