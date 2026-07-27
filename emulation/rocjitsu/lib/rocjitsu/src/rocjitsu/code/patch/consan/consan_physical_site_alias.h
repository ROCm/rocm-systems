// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::consan_detail {

/// Invalidate a direct kernel_descriptor_file_offset hint after an alias merge.
inline constexpr auto invalidate_physical_site_single_owner_hint =
    [](auto &retained, const auto &) { retained.kernel_descriptor_file_offset.reset(); };

/// Retain the first candidate for each physical file offset and fold
/// semantically equivalent aliases into it.
///
/// Callers own the candidate-specific semantic comparison and alias merge.
/// If a candidate carries a single-owner hint, merge_alias must invalidate it:
/// one canonical physical site can execute through every aliasing owner. Use
/// invalidate_physical_site_single_owner_hint when that hint is stored directly
/// as kernel_descriptor_file_offset. On failure, candidates is unchanged.
template <typename Candidate, typename FileOffset, typename ContainerName, typename SemanticsEqual,
          typename MergeAlias>
[[nodiscard]] bool canonicalize_physical_site_aliases(
    std::vector<Candidate> &candidates, std::vector<std::string> &errors,
    std::string_view site_description, FileOffset file_offset, ContainerName container_name,
    SemanticsEqual semantics_equal, MergeAlias merge_alias) {
  std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset;
  canonical_index_by_file_offset.reserve(candidates.size());
  std::vector<uint64_t> file_offsets;
  file_offsets.reserve(candidates.size());
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const Candidate &candidate = candidates[candidate_index];
    const uint64_t offset = file_offset(candidate);
    file_offsets.push_back(offset);
    const auto [entry, inserted] = canonical_index_by_file_offset.emplace(offset, candidate_index);
    if (inserted)
      continue;

    const Candidate &retained = candidates[entry->second];
    if (!semantics_equal(retained, candidate)) {
      errors.emplace_back(std::string(site_description) + " at file offset " +
                          std::to_string(offset) + " was decoded inconsistently through aliases '" +
                          std::string(container_name(retained)) + "' and '" +
                          std::string(container_name(candidate)) + "'");
      return false;
    }
  }

  canonical_index_by_file_offset.clear();
  std::vector<Candidate> canonical;
  canonical.reserve(candidates.size());
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    Candidate &candidate = candidates[candidate_index];
    const uint64_t offset = file_offsets[candidate_index];
    const auto [entry, inserted] = canonical_index_by_file_offset.emplace(offset, canonical.size());
    if (inserted) {
      canonical.push_back(std::move(candidate));
      continue;
    }

    Candidate &retained = canonical[entry->second];
    merge_alias(retained, candidate);
  }
  candidates = std::move(canonical);
  return true;
}

template <typename Candidate, typename FileOffset, typename ContainerName, typename SemanticsEqual>
[[nodiscard]] bool
canonicalize_physical_site_aliases(std::vector<Candidate> &candidates,
                                   std::vector<std::string> &errors,
                                   std::string_view site_description, FileOffset file_offset,
                                   ContainerName container_name, SemanticsEqual semantics_equal) {
  return canonicalize_physical_site_aliases(
      candidates, errors, site_description, std::move(file_offset), std::move(container_name),
      std::move(semantics_equal), [](Candidate &, const Candidate &) {});
}

} // namespace rocjitsu::consan_detail
