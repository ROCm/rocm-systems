// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "pytorch_store.hpp"

#include <cstddef>

namespace profiler_hub::pytorch
{

// Holds the tracks and the metadata around them in the same catalog the SQLite path
// fills, so ph_ctx builds its C structures from it the same way. The store's strings
// move into the catalog, which keeps them for the life of the ctx. Category ids and name
// ids share one string table, as rocpd strings do: categories take the ids below the
// returned one, names take it and the ids after it.
[[nodiscard]] std::size_t
fill_catalog(reader_catalog_t& catalog, pytorch_store_t& store);

}  // namespace profiler_hub::pytorch
