// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "reader_catalog.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub
{

/** @brief A copied time window. Absent on resolved_event_query when the
 *         caller left ph_time_filter_t::is_valid at zero. */
struct time_window
{
    uint64_t start{};
    uint64_t end{};
};

/** @brief Filter copied from ph_event_filter_t, with search text resolved to
 *         string-table ids and kernel-symbol ids. Optional ids are absent
 *         when the caller's ph_optional_uint64_t::is_valid is zero. */
struct resolved_event_query
{
    std::optional<time_window> time;
    std::vector<uint32_t>      tracks;
    std::vector<uint32_t>      event_types;
    std::optional<uint64_t>    node_id;
    std::optional<uint64_t>    agent_id;
    std::optional<uint64_t>    process_id;
    std::optional<uint64_t>    thread_id;
    bool                       include_category{};
    bool                       has_string_filter{};
    std::vector<std::string>   search_strings;
    std::vector<size_t>        string_ids;
    std::vector<size_t>        kernel_symbol_ids;
};

/** @brief Case-insensitive match. substring selects a contained term. partial
 *         selects any term; otherwise every term must match. */
[[nodiscard]] bool text_matches_terms(std::string_view                     text,
                                      const std::vector<std::string>&      terms,
                                      bool                                 substring,
                                      bool                                 partial);

/** @brief Fills string_ids and kernel_symbol_ids from the catalog. An empty
 *         search_strings list leaves both id lists empty and does not mark
 *         the query as string-filtered. */
void resolve_event_query_strings(const reader_catalog_t& catalog,
                                 resolved_event_query&   query,
                                 bool                    substring,
                                 bool                    partial);

}  // namespace profiler_hub
