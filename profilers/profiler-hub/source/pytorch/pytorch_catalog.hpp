// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "pytorch_store.hpp"

#include <cstddef>

namespace profiler_hub::pytorch
{

// Category ids (below 256) and name ids (from here) share one string table, as rocpd
// strings do, so the two ranges must not meet.
inline constexpr std::size_t k_catalog_name_id_base = 256;

// Holds the tracks and the metadata around them in the same catalog the SQLite path
// fills, so ph_ctx builds its C structures from it the same way. The store's strings
// move into the catalog, which keeps them for the life of the ctx.
void
fill_catalog(reader_catalog_t& catalog, pytorch_store_t& store);

}  // namespace profiler_hub::pytorch
