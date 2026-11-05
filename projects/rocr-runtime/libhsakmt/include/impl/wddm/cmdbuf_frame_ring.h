/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

namespace wsl {
namespace thunk {

// PM4 command-buffer frame ring scheduling, in terms of the submission ordinal.
//
// ComputeQueue translates AQL packets into a command buffer laid out as
// GetAqlFrameNum() equally sized frames. One submission owns exactly one frame,
// but SwitchAql2PM4() merges a run of consecutive dispatches into a single
// submission, so submissions and AQL packets do not advance together: a frame
// may hold one packet or a whole merge run.
//
// Both the frame a submission writes and the fence value it signals therefore
// count submissions, never AQL packets. Submission P writes frame
// (P - 1) % frame_num and signals fence value P, and it signals it only once
// every packet merged into it has retired. Consecutive submissions signal
// consecutive values, so "the GPU is finished with frame f" is exactly
// decidable: submission P must not write its frame until the queue's monitored
// fence has reached P - frame_num, the ordinal of that frame's previous
// occupant.
//
// Selecting the frame by the AQL write index instead collapses the ring.
// InitCmdbufInfo() clamps the merge limit to the frame count and SwitchAql2PM4()
// merges until the write index reaches a multiple of that limit, so at the clamp
// every run begins on a multiple of frame_num and `write_index % frame_num`
// names the same frame every time, however many frames were allocated. Each
// submission then serialises behind the previous one rather than overlapping it.
//
// All three rules answer "what will the next submission do?" given how many have
// already been issued, which is what ComputeQueue keeps in sync_point.
struct CmdbufFrameRing {
  // The fence value the next submission signals.
  static constexpr uint64_t NextFenceValue(uint64_t submitted) { return submitted + 1; }

  // The frame the next submission writes.
  static constexpr uint64_t NextFrameIndex(uint64_t submitted, uint32_t frame_num) {
    return submitted % frame_num;
  }

  // The fence value that must be reached before the next submission may
  // overwrite its frame, or 0 while the ring has not wrapped and the frame has
  // no previous occupant to wait for. Nothing is subtracted until it has
  // wrapped, so this cannot underflow.
  static constexpr uint64_t NextFrameReuseFence(uint64_t submitted, uint32_t frame_num) {
    return (submitted < frame_num) ? 0 : (submitted + 1 - frame_num);
  }
};

}  // namespace thunk
}  // namespace wsl
