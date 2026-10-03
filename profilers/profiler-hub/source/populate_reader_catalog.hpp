// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace profiler_hub
{

class reader_catalog_t;

namespace common
{
class connection_pool;
class thread_pool;
}  // namespace common

void
populate_reader_catalog(common::thread_pool&     workers,
                        common::connection_pool& connections,
                        reader_catalog_t&        catalog);

}  // namespace profiler_hub
