// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/memory_transaction.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Contiguous bytes supplied by active lanes of one memory instruction.
struct CoalescedLaneRun {
  uint64_t address = 0;
  uint64_t lane_mask = 0;
  uint32_t first_lane = 0;
  uint32_t size = 0;
  bool lane_ordered = true;
};

/// @brief Gather a transaction, clipping lanes split by its boundaries.
inline void gather_lane_run(uint64_t address, uint64_t lane_mask, uint32_t size,
                            const uint64_t *addrs, const uint8_t *lanes, uint32_t stride,
                            uint8_t *bytes) {
  const bool whole_lanes = std::popcount(lane_mask) * stride == size;
  for (uint64_t remaining = lane_mask; remaining; remaining &= remaining - 1) {
    const uint32_t lane = std::countr_zero(remaining);
    if (whole_lanes) {
      std::memcpy(bytes + (addrs[lane] - address), lanes + lane * stride, stride);
      continue;
    }
    const uint32_t lane_offset = address > addrs[lane] ? address - addrs[lane] : 0;
    const uint32_t run_offset = addrs[lane] > address ? addrs[lane] - address : 0;
    const uint32_t count = std::min(stride - lane_offset, size - run_offset);
    std::memcpy(bytes + run_offset, lanes + lane * stride + lane_offset, count);
  }
}

/// @brief Scatter a transaction without overwriting another fragment of a lane.
inline void scatter_lane_run(uint64_t address, uint64_t lane_mask, uint32_t size,
                             const uint64_t *addrs, const uint8_t *bytes, uint32_t stride,
                             uint8_t *lanes) {
  const bool whole_lanes = std::popcount(lane_mask) * stride == size;
  for (uint64_t remaining = lane_mask; remaining; remaining &= remaining - 1) {
    const uint32_t lane = std::countr_zero(remaining);
    if (whole_lanes) {
      std::memcpy(lanes + lane * stride, bytes + (addrs[lane] - address), stride);
      continue;
    }
    const uint32_t lane_offset = address > addrs[lane] ? address - addrs[lane] : 0;
    const uint32_t run_offset = addrs[lane] > address ? addrs[lane] - address : 0;
    const uint32_t count = std::min(stride - lane_offset, size - run_offset);
    std::memcpy(lanes + lane * stride + lane_offset, bytes + run_offset, count);
  }
}

namespace detail {

/// Keep sorting and gather/scatter storage off the ordinary contiguous path.
template <typename F>
RJ_NOINLINE bool for_each_reordered_lane_run(const uint64_t *addrs, uint64_t lane_mask,
                                             uint32_t wf_size, uint32_t stride,
                                             MemoryTransactionPolicy transactions, F &fn) {
  struct Lane {
    uint64_t address;
    uint32_t index;
  };
  std::array<Lane, 64> lanes;
  uint32_t count = 0;
  bool address_ordered = true;
  for (uint64_t remaining = lane_mask; remaining; remaining &= remaining - 1) {
    const uint32_t lane = std::countr_zero(remaining);
    address_ordered &= count == 0 || lanes[count - 1].address <= addrs[lane];
    lanes[count++] = {addrs[lane], lane};
  }
  if (!address_ordered)
    std::ranges::sort(lanes.begin(), lanes.begin() + count,
                      [](const Lane &a, const Lane &b) { return a.address < b.address; });

  // Preserve the existing lane order for overlapping operands. In particular,
  // sorting must not change which lane supplies the final byte of a store.
  bool overlap = false;
  bool split_operands = false;
  const uint64_t transaction_mask = ~uint64_t{transactions.bytes() - 1};
  for (uint32_t i = 0; i < count; ++i) {
    if (lanes[i].address > std::numeric_limits<uint64_t>::max() - (stride - 1) ||
        (i && lanes[i].address - lanes[i - 1].address < stride)) {
      overlap = true;
      break;
    }
    split_operands |= (lanes[i].address & transaction_mask) !=
                      ((lanes[i].address + stride - 1) & transaction_mask);
  }
  if (overlap) {
    for (uint64_t remaining = lane_mask; remaining;) {
      const uint32_t first = std::countr_zero(remaining);
      uint32_t last = first;
      uint64_t mask = uint64_t{1} << first;
      remaining &= ~mask;
      while (last + 1 < wf_size && (remaining & (uint64_t{1} << (last + 1))) &&
             addrs[last] <= std::numeric_limits<uint64_t>::max() - stride &&
             addrs[last + 1] == addrs[last] + stride) {
        ++last;
        mask |= uint64_t{1} << last;
        remaining &= ~(uint64_t{1} << last);
      }
      if (!fn(CoalescedLaneRun{.address = addrs[first],
                               .lane_mask = mask,
                               .first_lane = first,
                               .size = (last - first + 1) * stride,
                               .lane_ordered = true}))
        return false;
    }
    return true;
  }

  if (split_operands) {
    // A lane may contribute to two transactions. Partition contiguous bytes
    // first, then gather/scatter the intersecting lane fragments. Keep this
    // uncommon path separate from naturally aligned operands and fast runs.
    std::vector<CoalescedLaneRun> runs;
    runs.reserve(count * 2);
    for (uint32_t first = 0; first < count;) {
      uint32_t last = first;
      while (last + 1 < count && lanes[last + 1].address - lanes[last].address == stride)
        ++last;
      uint64_t address = lanes[first].address;
      uint64_t remaining = uint64_t{last - first + 1} * stride;
      uint32_t begin = first;
      while (remaining) {
        const uint32_t chunk = transactions.chunk_size(address, remaining);
        while (address - lanes[begin].address >= stride)
          ++begin;
        uint32_t end = begin;
        while (end < last && lanes[end + 1].address - address < chunk)
          ++end;
        uint64_t mask = 0;
        bool ordered = address == lanes[begin].address && chunk == (end - begin + 1) * stride;
        for (uint32_t i = begin; i <= end; ++i) {
          mask |= uint64_t{1} << lanes[i].index;
          ordered &= lanes[i].index == lanes[begin].index + i - begin;
        }
        runs.push_back({.address = address,
                        .lane_mask = mask,
                        .first_lane = lanes[begin].index,
                        .size = chunk,
                        .lane_ordered = ordered});
        address += chunk;
        remaining -= chunk;
      }
      first = last + 1;
    }
    std::ranges::sort(runs, [](const CoalescedLaneRun &a, const CoalescedLaneRun &b) {
      const auto first_a = std::countr_zero(a.lane_mask);
      const auto first_b = std::countr_zero(b.lane_mask);
      return first_a != first_b ? first_a < first_b : a.address < b.address;
    });
    for (const auto &run : runs) {
      if (!fn(run))
        return false;
    }
    return true;
  }

  // Combine only bytes actually requested, and never join reordered accesses
  // across a transaction boundary. Gaps and clipped elements stay gaps.
  auto visit_runs = [&](auto &&emit) {
    for (uint32_t first = 0; first < count;) {
      uint32_t last = first;
      uint64_t mask = uint64_t{1} << lanes[first].index;
      bool ordered = true;
      while (last + 1 < count &&
             lanes[last].address <= std::numeric_limits<uint64_t>::max() - stride &&
             lanes[last + 1].address == lanes[last].address + stride &&
             (lanes[first].address & transaction_mask) ==
                 ((lanes[last + 1].address + stride - 1) & transaction_mask)) {
        ordered &= lanes[last + 1].index == lanes[last].index + 1;
        mask |= uint64_t{1} << lanes[++last].index;
      }
      const CoalescedLaneRun run{.address = lanes[first].address,
                                 .lane_mask = mask,
                                 .first_lane = lanes[first].index,
                                 .size = (last - first + 1) * stride,
                                 .lane_ordered = ordered};
      if (!emit(run))
        return false;
      first = last + 1;
    }
    return true;
  };
  if (address_ordered)
    return visit_runs(fn);
  std::array<CoalescedLaneRun, 64> runs;
  uint32_t run_count = 0;
  visit_runs([&](const CoalescedLaneRun &run) {
    runs[run_count++] = run;
    return true;
  });
  // Keep independent transactions in first-participating-lane order. Only the
  // lanes within a contiguous transaction need gathering or scattering.
  std::ranges::sort(runs.begin(), runs.begin() + run_count,
                    [](const CoalescedLaneRun &a, const CoalescedLaneRun &b) {
                      return std::countr_zero(a.lane_mask) < std::countr_zero(b.lane_mask);
                    });
  for (uint32_t i = 0; i < run_count; ++i) {
    if (!fn(runs[i]))
      return false;
  }
  return true;
}

} // namespace detail

/// @brief Visit contiguous lane runs independently of the lanes' address order.
/// @details Lane-ordered runs may span transaction boundaries; their caller must
/// split them using the policy. Reordered multi-lane runs fit one transaction.
/// Overlapping or wrapping operands retain the original lane sequence;
/// they do not receive the contiguous, non-overlapping snapshot guarantee.
/// The callback returns false to stop after an unavailable or failed transfer.
template <typename F>
bool for_each_coalesced_lane_run(const uint64_t *addrs, uint64_t lane_mask, uint32_t wf_size,
                                 uint32_t stride, MemoryTransactionPolicy transactions, F &&fn) {
  assert(wf_size <= 64 && stride != 0);
  if (!lane_mask)
    return true;
  const uint32_t first = std::countr_zero(lane_mask);
  uint32_t last = first;
  uint64_t remaining = lane_mask & ~(uint64_t{1} << first);
  while (last + 1 < wf_size && (remaining & (uint64_t{1} << (last + 1))) &&
         addrs[last] <= std::numeric_limits<uint64_t>::max() - stride &&
         addrs[last + 1] == addrs[last] + stride) {
    ++last;
    remaining &= ~(uint64_t{1} << last);
  }
  if (!remaining)
    return fn(CoalescedLaneRun{.address = addrs[first],
                               .lane_mask = lane_mask,
                               .first_lane = first,
                               .size = (last - first + 1) * stride,
                               .lane_ordered = true});
  // Ordinary strided instructions also stay inline: validating monotonic,
  // non-overlapping operands needs no lane/run arrays or gather buffers. A
  // contiguous address across an inactive lane still needs the reordered path.
  bool direct_runs = true;
  uint32_t previous = first;
  for (uint64_t check = lane_mask & ~(uint64_t{1} << first); check; check &= check - 1) {
    const uint32_t lane = std::countr_zero(check);
    if (addrs[previous] > std::numeric_limits<uint64_t>::max() - stride ||
        addrs[lane] < addrs[previous] + stride ||
        (lane != previous + 1 && addrs[lane] == addrs[previous] + stride)) {
      direct_runs = false;
      break;
    }
    previous = lane;
  }
  if (direct_runs) {
    for (uint64_t pending = lane_mask; pending;) {
      const uint32_t begin = std::countr_zero(pending);
      uint32_t end = begin;
      uint64_t mask = uint64_t{1} << begin;
      pending &= ~mask;
      while (end + 1 < wf_size && (pending & (uint64_t{1} << (end + 1))) &&
             addrs[end + 1] == addrs[end] + stride) {
        ++end;
        mask |= uint64_t{1} << end;
        pending &= ~(uint64_t{1} << end);
      }
      if (!fn(CoalescedLaneRun{.address = addrs[begin],
                               .lane_mask = mask,
                               .first_lane = begin,
                               .size = (end - begin + 1) * stride,
                               .lane_ordered = true}))
        return false;
    }
    return true;
  }
  return detail::for_each_reordered_lane_run(addrs, lane_mask, wf_size, stride, transactions, fn);
}

} // namespace rocjitsu::amdgpu
