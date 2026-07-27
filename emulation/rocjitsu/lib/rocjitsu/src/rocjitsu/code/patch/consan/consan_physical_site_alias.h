// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::consan_detail {

/// Invalidate a direct kernel_descriptor_file_offset hint after an alias merge.
inline constexpr auto invalidate_physical_site_single_owner_hint =
    [](auto &retained, const auto &) { retained.kernel_descriptor_file_offset.reset(); };

template <typename Candidate, typename ContainerName>
void append_physical_site_alias_conflict(std::vector<std::string> &errors,
                                         std::string_view site_description, uint64_t file_offset,
                                         const Candidate &retained, const Candidate &candidate,
                                         ContainerName &container_name) {
  errors.emplace_back(
      std::string(site_description) + " at file offset " + std::to_string(file_offset) +
      " was decoded inconsistently through aliases '" + std::string(container_name(retained)) +
      "' and '" + std::string(container_name(candidate)) + "'");
}

/// Incrementally retain the first candidate for each physical file offset.
///
/// Conflicts fail fast and leave every previously accepted candidate intact.
/// Once insert returns false, later inserts also return false without changing
/// state or appending diagnostics. take returns no inventory after a conflict
/// and permanently closes the canonicalizer after a successful extraction.
template <typename Candidate, typename FileOffset, typename ContainerName, typename SemanticsEqual,
          typename MergeAlias>
class PhysicalSiteAliasCanonicalizer {
public:
  PhysicalSiteAliasCanonicalizer(std::vector<std::string> &errors,
                                 std::string_view site_description, FileOffset file_offset,
                                 ContainerName container_name, SemanticsEqual semantics_equal,
                                 MergeAlias merge_alias, size_t expected_candidate_count = 0)
      : errors_(errors), site_description_(site_description), file_offset_(std::move(file_offset)),
        container_name_(std::move(container_name)), semantics_equal_(std::move(semantics_equal)),
        merge_alias_(std::move(merge_alias)) {
    canonical_index_by_file_offset_.reserve(expected_candidate_count);
    canonical_.reserve(expected_candidate_count);
  }

  [[nodiscard]] bool insert(Candidate candidate) {
    if (state_ != State::Accepting)
      return false;
    const uint64_t file_offset = file_offset_(candidate);
    const auto [entry, inserted] =
        canonical_index_by_file_offset_.emplace(file_offset, canonical_.size());
    if (inserted) {
      canonical_.push_back(std::move(candidate));
      return true;
    }

    Candidate &retained = canonical_[entry->second];
    if (!semantics_equal_(retained, candidate)) {
      append_physical_site_alias_conflict(errors_, site_description_, file_offset, retained,
                                          candidate, container_name_);
      state_ = State::Failed;
      return false;
    }
    merge_alias_(retained, candidate);
    return true;
  }

  [[nodiscard]] bool failed() const { return state_ == State::Failed; }
  [[nodiscard]] const std::vector<Candidate> &candidates() const { return canonical_; }

  /// Extract the completed inventory exactly once.
  ///
  /// A failed or already-extracted canonicalizer returns nullopt. In every
  /// case, the object is left closed and internally self-consistent.
  [[nodiscard]] std::optional<std::vector<Candidate>> take() && {
    canonical_index_by_file_offset_.clear();
    if (state_ != State::Accepting) {
      canonical_.clear();
      return std::nullopt;
    }
    state_ = State::Taken;
    return std::exchange(canonical_, {});
  }

private:
  enum class State : uint8_t {
    Accepting,
    Failed,
    Taken,
  };

  /// Insert after the batch API's earlier pass established semantic
  /// equivalence and the candidate's projected file offset.
  void insert_prevalidated(Candidate candidate, uint64_t file_offset) {
    assert(state_ == State::Accepting);
    assert(file_offset == file_offset_(candidate));
    const auto [entry, inserted] =
        canonical_index_by_file_offset_.emplace(file_offset, canonical_.size());
    if (inserted) {
      canonical_.push_back(std::move(candidate));
      return;
    }
    merge_alias_(canonical_[entry->second], candidate);
  }

  template <typename OtherCandidate, typename OtherFileOffset, typename OtherContainerName,
            typename OtherSemanticsEqual, typename OtherMergeAlias>
  friend bool canonicalize_physical_site_aliases(std::vector<OtherCandidate> &,
                                                 std::vector<std::string> &, std::string_view,
                                                 OtherFileOffset, OtherContainerName,
                                                 OtherSemanticsEqual, OtherMergeAlias);

  std::vector<std::string> &errors_;
  std::string site_description_;
  [[no_unique_address]] FileOffset file_offset_;
  [[no_unique_address]] ContainerName container_name_;
  [[no_unique_address]] SemanticsEqual semantics_equal_;
  [[no_unique_address]] MergeAlias merge_alias_;
  std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset_;
  std::vector<Candidate> canonical_;
  State state_ = State::Accepting;
};

template <typename Candidate, typename FileOffset, typename ContainerName, typename SemanticsEqual,
          typename MergeAlias>
[[nodiscard]] auto make_physical_site_alias_canonicalizer(
    std::vector<std::string> &errors, std::string_view site_description, FileOffset file_offset,
    ContainerName container_name, SemanticsEqual semantics_equal, MergeAlias merge_alias,
    size_t expected_candidate_count = 0) {
  return PhysicalSiteAliasCanonicalizer<Candidate, std::decay_t<FileOffset>,
                                        std::decay_t<ContainerName>, std::decay_t<SemanticsEqual>,
                                        std::decay_t<MergeAlias>>(
      errors, site_description, std::move(file_offset), std::move(container_name),
      std::move(semantics_equal), std::move(merge_alias), expected_candidate_count);
}

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
  std::vector<uint64_t> file_offsets;
  file_offsets.reserve(candidates.size());
  {
    std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset;
    canonical_index_by_file_offset.reserve(candidates.size());
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
      const Candidate &candidate = candidates[candidate_index];
      const uint64_t offset = file_offset(candidate);
      file_offsets.push_back(offset);
      const auto [entry, inserted] =
          canonical_index_by_file_offset.emplace(offset, candidate_index);
      if (inserted)
        continue;

      const Candidate &retained = candidates[entry->second];
      if (!semantics_equal(retained, candidate)) {
        append_physical_site_alias_conflict(errors, site_description, offset, retained, candidate,
                                            container_name);
        return false;
      }
    }
  }

  auto canonicalizer = make_physical_site_alias_canonicalizer<Candidate>(
      errors, site_description, std::move(file_offset), std::move(container_name),
      std::move(semantics_equal), std::move(merge_alias), candidates.size());
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    canonicalizer.insert_prevalidated(std::move(candidates[candidate_index]),
                                      file_offsets[candidate_index]);
  }
  std::optional<std::vector<Candidate>> canonical = std::move(canonicalizer).take();
  if (!canonical)
    return false;
  candidates = std::move(*canonical);
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
