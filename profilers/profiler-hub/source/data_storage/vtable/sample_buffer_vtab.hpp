// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace profiler_hub::data_storage::vtable
{

// Registers the "sample_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE sample_buf
//       USING sample_buffer('rocpd_sample_<uuid>');
int
register_sample_buffer_module(sqlite3* db);

}  // namespace profiler_hub::data_storage::vtable
