// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <sqlite3.h>

namespace rocpdsna::data_storage::vtable
{

// Registers the "pmc_event_buffer" module on the given connection.
//
// Usage from SQL:
//   CREATE VIRTUAL TABLE pmc_event_buf
//       USING pmc_event_buffer('rocpd_pmc_event_<uuid>');
int
register_pmc_event_buffer_module(sqlite3* db);

}  // namespace rocpdsna::data_storage::vtable
