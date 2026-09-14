// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_XCD_SHARD_H_
#define ROCJITSU_VM_AMDGPU_XCD_SHARD_H_

/// @file xcd_shard.h
/// @brief Round-robin split of an ordinal range across a participating set.

#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief One member's share of an ordinal range split round-robin.
///
/// @details A partitioning primitive over abstract chunk ordinals: given a
/// participating set of @c stride members, the member at @c rank owns exactly
/// the ordinals congruent to its rank modulo the stride. It knows nothing about
/// what a chunk is and imposes no policy about who participates.
///
/// The permutation is part of the contract, not just the even split: kernel
/// libraries ship a chunk-index swizzle for cache locality (the
/// `chunk % participants` remap) that assumes consecutive ordinals are
/// distributed this way.
///
/// Rank is the index *within the participating set*. It is deliberately neither
/// a SoC-global XCD id nor an engine partition id; a caller that partitions a
/// subset of a device maps rank back to whatever hardware it stands for.
///
/// The chunk is likewise the caller's unit, and choosing it is dispatch policy
/// that does not belong to this type. For a clustered dispatch the two obvious
/// choices disagree: workgroups 0 and 1 of a 2-wide cluster must share a
/// destination, while an ordinal mapping over workgroups would separate them.
///
/// Callers walk their share by shard-local index: iterate @c i over
/// `[0, owned_chunks(total))` and map each through nth_owned_chunk(). Do not
/// step a grid-wide ordinal by the stride instead — on a maximal range that
/// addition wraps back to a low ordinal, whereas the shard-local count cannot
/// run past the end.
///
/// Any stride >= 1 is valid and need not be a power of two; the topologies under
/// configs/ use eight. Both fields are plain counts rather than, say, a log2
/// stride: the arithmetic below is a division and a multiply, not a mask,
/// precisely so that a non-power-of-two stride works.
///
/// A default-constructed shard is unsharded and owns every ordinal, which is
/// what a caller that is not splitting anything gets.
class XcdShard {
public:
  XcdShard() = default;

  /// @brief Construct a shard for one member of a participating set.
  /// @param rank This member's index within the set; must be < @p stride.
  /// @param stride Number of participating members; must be >= 1.
  XcdShard(uint32_t rank, uint32_t stride) : rank_(rank), stride_(stride) {
    assert(stride_ >= 1 && "shard stride is a count of participants and must be at least one");
    assert(rank_ < stride_ && "shard rank is a set index and must be below the stride");
  }

  [[nodiscard]] uint32_t rank() const { return rank_; }
  [[nodiscard]] uint32_t stride() const { return stride_; }

  /// @returns True when nothing is being split, so this shard owns every
  /// ordinal. This asks about the shard, not about coverage of a particular
  /// range: a shard of a one-chunk range may own that whole range and still
  /// answer false. A shard with no participants is not unsharded either — it is
  /// invalid, and reports so rather than claiming the range.
  [[nodiscard]] bool is_unsharded() const { return stride_ == 1; }

  /// @brief Count the chunks this shard owns.
  /// @param total_chunks Chunk count of the whole range.
  /// @returns Number of chunks owned by this shard, possibly zero when the range
  /// has fewer chunks than participating members.
  [[nodiscard]] uint32_t owned_chunks(uint32_t total_chunks) const {
    // A zero stride violates the constructor's contract, which the assertions
    // catch in a debug build. They compile out of a release build, so own
    // nothing rather than divide by zero. Failing closed keeps the invalid shard
    // identifiable through rank()/stride(); rewriting it to the unsharded shard
    // would instead hand a caller the entire range on the strength of a bug.
    if (stride_ == 0 || rank_ >= total_chunks)
      return 0;
    // Quotient/remainder rather than the usual (n + stride - 1) / stride: with a
    // maximal grid that addition wraps in 32 bits and silently reports zero
    // chunks, dropping almost the entire grid.
    const uint32_t remaining = total_chunks - rank_;
    return remaining / stride_ + (remaining % stride_ != 0 ? 1u : 0u);
  }

  /// @brief Map a shard-local chunk index to its grid-wide chunk ordinal.
  /// @param shard_chunk_index Zero-based index within this shard's own chunks,
  /// which is to say below owned_chunks() for the range being walked.
  /// @returns The grid-wide chunk ordinal.
  [[nodiscard]] uint32_t nth_owned_chunk(uint32_t shard_chunk_index) const {
    return rank_ + shard_chunk_index * stride_;
  }

private:
  uint32_t rank_ = 0;
  uint32_t stride_ = 1;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_XCD_SHARD_H_
