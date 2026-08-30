// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_hsa_dbi_report_registry_lifecycle.h
/// @brief Shared bounded-lifetime mechanics for mode-owned report registries.

#pragma once

#include "hsa/hsa_api_trace_minimal.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace rocjitsu::consan_hook::detail {

struct AutoReportRegionSearch {
  CoreApiTable *core = nullptr;
  size_t requested_size = 0;
  hsa_region_t region{};
  bool found = false;
  bool fine_grained = false;
};

inline hsa_status_t HSA_API select_auto_report_region(hsa_region_t region, void *data) {
  auto *search = static_cast<AutoReportRegionSearch *>(data);
  hsa_region_segment_t segment{};
  hsa_status_t status =
      search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
  if (status != HSA_STATUS_SUCCESS || segment != HSA_REGION_SEGMENT_GLOBAL)
    return status;
  bool alloc_allowed = false;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                                &alloc_allowed);
  if (status != HSA_STATUS_SUCCESS || !alloc_allowed)
    return status;
  size_t max_size = 0;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_size);
  if (status != HSA_STATUS_SUCCESS || max_size < search->requested_size)
    return status;
  uint32_t flags = 0;
  status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  if (status != HSA_STATUS_SUCCESS)
    return status;
  const bool fine_grained = (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
  const bool coarse_grained = (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
  if (fine_grained) {
    search->region = region;
    search->found = true;
    search->fine_grained = true;
    return HSA_STATUS_INFO_BREAK;
  }
  if (!search->found && coarse_grained) {
    search->region = region;
    search->found = true;
    search->fine_grained = false;
  }
  return HSA_STATUS_SUCCESS;
}

template <typename Entry, size_t Capacity>
[[nodiscard]] auto find_auto_report_entry(std::array<Entry, Capacity> &entries, size_t entry_count,
                                          uint64_t reader, uint64_t generation) {
  return std::find_if(entries.begin(), entries.begin() + entry_count,
                      [reader, generation](const Entry &candidate) {
                        return candidate.reader == reader && candidate.generation == generation;
                      });
}

template <typename Entry, size_t Capacity>
void erase_auto_report_entry(std::array<Entry, Capacity> &entries, size_t &entry_count,
                             size_t index) {
  for (size_t next = index + 1u; next < entry_count; ++next)
    entries[next - 1u] = std::move(entries[next]);
  entries[--entry_count] = {};
}

template <typename Entry, size_t Capacity>
void bind_auto_report_entry(std::array<Entry, Capacity> &entries, size_t entry_count,
                            uint64_t reader, uint64_t generation, uint64_t executable) {
  const auto entry = find_auto_report_entry(entries, entry_count, reader, generation);
  if (entry == entries.begin() + entry_count)
    return;
  entry->executable = executable;
  entry->executable_bound = true;
}

template <typename Entry, size_t Capacity, typename Release>
void discard_auto_report_entry(std::array<Entry, Capacity> &entries, size_t &entry_count,
                               uint64_t reader, uint64_t generation, Release &&release) {
  const auto entry = find_auto_report_entry(entries, entry_count, reader, generation);
  if (entry == entries.begin() + entry_count || !release(*entry))
    return;
  erase_auto_report_entry(entries, entry_count, static_cast<size_t>(entry - entries.begin()));
}

template <typename Entry, size_t Capacity, typename Summary, typename Summarize, typename Release,
          typename Accumulate>
void retire_auto_report_entries(std::array<Entry, Capacity> &entries, size_t &entry_count,
                                uint64_t executable, Summary &retired_summary,
                                Summarize &&summarize, Release &&release, Accumulate &&accumulate) {
  size_t index = 0;
  while (index < entry_count) {
    Entry &entry = entries[index];
    if (!entry.executable_bound || entry.executable != executable) {
      ++index;
      continue;
    }
    const Summary entry_summary = summarize(entry);
    if (!release(entry)) {
      ++index;
      continue;
    }
    accumulate(retired_summary, entry_summary);
    erase_auto_report_entry(entries, entry_count, index);
  }
}

} // namespace rocjitsu::consan_hook::detail
