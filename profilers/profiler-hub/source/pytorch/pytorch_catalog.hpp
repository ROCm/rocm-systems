// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "kineto_loader.hpp"
#include "pytorch_store.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace profiler_hub
{
struct reader_catalog_t;
}

namespace profiler_hub::pytorch
{
// Category ids (below 256) and name ids (from here) share one string table, as rocpd
// strings do, so the two ranges must not meet.
inline constexpr std::size_t k_catalog_name_id_base = 256;

std::shared_ptr<reader_catalog_t>
make_catalog(const loaded_trace_t& trace);

// Holds the derived tracks and the metadata around them in the same catalog the SQLite
// path fills, so ph_ctx builds its C structures from it the same way.
void
fill_catalog(reader_catalog_t&                 catalog,
             const std::vector<store_track_t>& tracks,
             const loaded_trace_t&             trace);
}  // namespace profiler_hub::pytorch
