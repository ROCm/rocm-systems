// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace profiler_hub::data_storage::vtable
{

// Registers the "arg_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE arg_buf
//       USING arg_buffer('rocpd_arg_<uuid>');
int
register_arg_buffer_module(sqlite3* db);

}  // namespace profiler_hub::data_storage::vtable
