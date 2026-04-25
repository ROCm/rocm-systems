// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace rocpdsna::data_storage::vtable
{

// Registers the "region_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE region_buf
//       USING region_buffer('rocpd_region_<uuid>');
int
register_region_buffer_module(sqlite3* db);

}  // namespace rocpdsna::data_storage::vtable
