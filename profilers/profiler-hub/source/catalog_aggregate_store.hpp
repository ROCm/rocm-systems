// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace profiler_hub
{
struct reader_catalog_t;

// Schema 2 grouped scans (processes, threads, agents, queues, counters,
// tracks). A hit fills those lists and sets derived_aggregates_ready.
// Strings and the node are not part of this file.
[[nodiscard]] bool load_catalog_aggregates(reader_catalog_t& catalog);

// Writes the lists produced by a fresh schema 2 build. A failure leaves the
// open on the SQL result; the next open tries again.
void save_catalog_aggregates(const reader_catalog_t& catalog);

// Schema 3 track list and stream-to-hardware links. Those come from grouped
// scans of the event tables, not from a track table. Info-table rows
// (nodes, processes, threads, agents, and the rest) stay in SQLite. A hit
// fills the track list, its lookup maps, memory-activity series, and the
// links, and sets derived_aggregates_ready.
[[nodiscard]] bool load_derived_tracks(reader_catalog_t& catalog);

// Writes the schema 3 track list after a fresh scan. A failure leaves the
// open on the SQL result; the next open tries again.
void save_derived_tracks(const reader_catalog_t& catalog);

}  // namespace profiler_hub
