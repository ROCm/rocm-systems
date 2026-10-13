/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// CmdbufFrameRing: the PM4 frame and fence schedule, replayed on the CPU.
//
// The scheduling is pure arithmetic over the submission ordinal, so the cases
// that matter - a run of one packet, a merged run, the first wrap, and the fence
// value a reused frame has to wait for - are driven here with no GPU, no Windows
// SDK and no DXG interop headers. The header under test depends on nothing but
// <cstdint> for exactly this reason.

#include "impl/wddm/cmdbuf_frame_ring.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using wsl::thunk::CmdbufFrameRing;

int failures = 0;

void Failed(const char* expr, int line) {
  std::fprintf(stderr, "cmdbuf_frame_ring_test:%d: FAILED: %s\n", line, expr);
  ++failures;
}

#define CHECK(expr) ((expr) ? (void)0 : Failed(#expr, __LINE__))

// Where a queue stands once a schedule of merge runs has been replayed through
// the ring. `run_lengths[i]` is how many AQL packets SwitchAql2PM4() merged into
// the (i+1)-th submission. Steps in the order ComputeQueue does: gate on the
// frame's previous occupant, claim the frame, translate the run into it, then
// submit and advance sync_point.
struct Replay {
  uint64_t submitted = 0;    // == sync_point, and the last fence value signalled
  uint64_t packets = 0;      // AQL packets translated so far
  uint64_t last_frame = 0;   // frame the last submission wrote
  uint64_t last_fence = 0;   // fence value the last submission signalled
  uint64_t last_wait = 0;    // fence value the last submission had to wait for
  uint64_t frames_used = 0;  // bit per frame touched, so a collapsed ring shows
};

Replay ReplayOrdinal(const std::vector<uint64_t>& run_lengths, uint32_t frame_num) {
  Replay r;
  for (uint64_t run : run_lengths) {
    r.last_wait = CmdbufFrameRing::NextFrameReuseFence(r.submitted, frame_num);
    r.last_frame = CmdbufFrameRing::NextFrameIndex(r.submitted, frame_num);
    r.frames_used |= uint64_t{1} << r.last_frame;
    r.packets += run;
    r.last_fence = CmdbufFrameRing::NextFenceValue(r.submitted);
    r.submitted = r.last_fence;
  }
  return r;
}

// The rule this replaces: the frame selected by the AQL write index. Kept here
// to pin the regression, and nowhere near the shipped header.
uint64_t FramesUsedByPacketIndex(const std::vector<uint64_t>& run_lengths, uint32_t frame_num) {
  uint64_t packets = 0;
  uint64_t used = 0;
  for (uint64_t run : run_lengths) {
    used |= uint64_t{1} << (packets % frame_num);
    packets += run;
  }
  return used;
}

// One AQL packet per submission: the only case where counting submissions and
// counting packets agree.
void OnePacketPerSubmission() {
  const Replay r = ReplayOrdinal({1, 1, 1, 1}, 4);

  CHECK(r.submitted == 4);
  CHECK(r.packets == 4);
  CHECK(r.last_frame == 3);
  CHECK(r.last_fence == 4);
  CHECK(r.last_wait == 0);       // a ring that has not wrapped reuses nothing
  CHECK(r.frames_used == 0xfu);  // and every frame was used exactly once
}

// A merge run consumes one frame however many packets it holds, so the frames
// and the fence values are unchanged while the AQL write index runs ahead.
void MergedRunsConsumeOneFrameEach() {
  const Replay r = ReplayOrdinal({3, 1, 2, 4}, 4);

  CHECK(r.submitted == 4);
  CHECK(r.packets == 10);
  CHECK(r.last_frame == 3);
  CHECK(r.last_fence == 4);
  CHECK(r.frames_used == 0xfu);
}

// A reused frame waits for the fence value its previous occupant signals, and
// that submission signals it only after every packet merged into it retired.
void ReuseWaitsForThePreviousOccupant() {
  const std::vector<uint64_t> runs = {3, 1, 2, 4, 1};

  const Replay first = ReplayOrdinal({3}, 4);
  CHECK(first.last_frame == 0);
  CHECK(first.last_fence == 1);
  CHECK(first.packets == 3);  // fence value 1 covers all three merged packets

  const Replay fifth = ReplayOrdinal(runs, 4);
  CHECK(fifth.last_frame == 0);                // wrapped back onto frame 0
  CHECK(fifth.last_wait == first.last_fence);  // so it waits for submission 1

  // Specifically not the value the AQL write index would have implied: ten
  // packets in, an index-relative wait names something submission 1 never
  // signals.
  const Replay fourth = ReplayOrdinal({3, 1, 2, 4}, 4);
  CHECK(fifth.last_wait != fourth.packets - 4 + 1);
}

// The wait trails the submission count by exactly the ring depth, indefinitely.
void Wraparound() {
  const Replay r = ReplayOrdinal({1, 2, 1, 3, 1, 1, 2}, 3);

  CHECK(r.submitted == 7);
  CHECK(r.last_frame == 0);  // submission 7 of a 3-frame ring is back on frame 0
  CHECK(r.last_fence == 7);
  CHECK(r.last_wait == 4);  // waiting for submission 4, the previous occupant
  CHECK(r.frames_used == 0x7u);

  // Far from the origin it is still plain modular arithmetic, with no drift.
  CHECK(CmdbufFrameRing::NextFrameIndex(1000000001ULL, 3) == 2);
  CHECK(CmdbufFrameRing::NextFrameReuseFence(1000000001ULL, 3) == 999999999ULL);
}

// Nothing is subtracted before the ring wraps, so no wait can underflow into a
// value the GPU will never signal.
void NoUnderflowBeforeTheFirstWrap() {
  CHECK(CmdbufFrameRing::NextFrameReuseFence(0, 4) == 0);
  CHECK(CmdbufFrameRing::NextFrameReuseFence(3, 4) == 0);
  CHECK(CmdbufFrameRing::NextFrameReuseFence(4, 4) == 1);

  // A one-frame ring serialises on the immediately preceding submission.
  CHECK(CmdbufFrameRing::NextFrameReuseFence(0, 1) == 0);
  CHECK(CmdbufFrameRing::NextFrameReuseFence(1, 1) == 1);
  CHECK(CmdbufFrameRing::NextFrameIndex(7, 1) == 0);
}

// The regression this indexing exists for. InitCmdbufInfo() clamps the merge
// limit to the frame count, so a run can advance the AQL write index by exactly
// frame_num; a frame chosen by that index is then the same frame every time and
// the ring is one frame deep however many were allocated.
void FullMergeRunsStillUseEveryFrame() {
  const std::vector<uint64_t> at_the_limit = {4, 4, 4, 4};

  CHECK(ReplayOrdinal(at_the_limit, 4).frames_used == 0xfu);
  CHECK(FramesUsedByPacketIndex(at_the_limit, 4) == 0x1u);
}

}  // namespace

int main() {
  OnePacketPerSubmission();
  MergedRunsConsumeOneFrameEach();
  ReuseWaitsForThePreviousOccupant();
  Wraparound();
  NoUnderflowBeforeTheFirstWrap();
  FullMergeRunsStillUseEveryFrame();

  if (failures != 0) {
    std::fprintf(stderr, "cmdbuf_frame_ring_test: %d check(s) failed\n", failures);
    return 1;
  }

  std::printf("cmdbuf_frame_ring_test: all checks passed\n");
  return 0;
}
