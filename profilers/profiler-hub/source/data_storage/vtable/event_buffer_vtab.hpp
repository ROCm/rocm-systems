// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace profiler_hub::data_storage::vtable
{

// Registers the "event_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE event_buf
//       USING event_buffer('rocpd_event_<uuid>');
int
register_event_buffer_module(sqlite3* db);

}  // namespace profiler_hub::data_storage::vtable
