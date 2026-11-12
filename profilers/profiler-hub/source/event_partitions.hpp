// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"
#include "profiler-hub/cpp/reader_types.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace profiler_hub
{

/** Smallest and largest row id of one event table; @c last is inclusive. */
struct table_id_span
{
    reader_types::event_type_t type;
    size_t                     first;
    size_t                     last;
};

/** Half-open range [begin, end) of row ids of one event table. */
struct id_range
{
    reader_types::event_type_t type;
    size_t                     begin;
    size_t                     end;
};

/**
 * Cuts the id spans of the event tables into about @p parts ranges in total, each table
 * getting a share proportional to its span. The ranges of one table are contiguous and in
 * id order, together they cover the span exactly, and the tables keep the order of @p
 * spans. Spans with last < first are ignored.
 */
[[nodiscard]] std::vector<id_range>
plan_id_ranges(std::span<const table_id_span> spans, size_t parts);

/**
 * Joins the events read for @p ranges (outputs[i] belongs to ranges[i]) into one list
 * ordered by start time. The outputs of one table must be adjacent and are emptied.
 * Events of equal start keep the order of their table in @p ranges.
 */
[[nodiscard]] std::vector<ph_event_t>
merge_event_partitions(std::span<const id_range>             ranges,
                       std::vector<std::vector<ph_event_t>>& outputs);

}  // namespace profiler_hub
