// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace rocpdsna::data_storage::vtable
{

// Registers the "memory_alloc_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE memory_alloc_buf
//       USING memory_alloc_buffer('rocpd_memory_allocate_<uuid>');
int
register_memory_alloc_buffer_module(sqlite3* db);

}  // namespace rocpdsna::data_storage::vtable
