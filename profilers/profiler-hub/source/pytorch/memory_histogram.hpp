// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader.hpp"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstddef>

namespace profiler_hub
{
class cached_track_reader;
}  // namespace profiler_hub

namespace profiler_hub::pytorch
{

// The overview histogram of a source without SQL, computed from the events the reader
// already holds, on the same grid and with the same overlap rule as histogram.cpp.
reader_types::histogram_t
memory_histogram(const reader_catalog_t& catalog,
                 cached_track_reader&    reader,
                 std::size_t             max_buckets);

}  // namespace profiler_hub::pytorch
