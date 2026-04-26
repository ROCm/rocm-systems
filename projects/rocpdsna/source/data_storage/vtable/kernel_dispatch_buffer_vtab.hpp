// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// SQLite virtual table that buffers INSERTs for the rocpd_kernel_dispatch
// table in per-column vectors and flushes them in bulk transactions. Trades
// durability between flushes for write throughput.
//
// Constraints:
//   - INSERT only; UPDATE/DELETE return SQLITE_READONLY.
//   - SELECT against the vtable returns no rows. Buffered rows are NOT
//     visible until flushed to the real table.
//   - Schema is hard-coded to the kernel_dispatch column layout
//     (21 int64 + 1 text).

#pragma once

#include <sqlite3.h>

namespace rocpdsna::data_storage::vtable
{

// Registers the "kernel_dispatch_buffer" module on the given connection.
// Idempotent per connection (sqlite3_create_module errors on second call,
// which is logged but treated as success).
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE kernel_dispatch_buf
//       USING kernel_dispatch_buffer('rocpd_kernel_dispatch_<uuid>');
int
register_kernel_dispatch_buffer_module(sqlite3* db);

}  // namespace rocpdsna::data_storage::vtable
