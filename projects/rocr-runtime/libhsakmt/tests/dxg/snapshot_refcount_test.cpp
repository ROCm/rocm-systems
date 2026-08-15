/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* State machine behind hsaKmtAcquireSystemProperties() /
 * hsaKmtReleaseSystemProperties(). Pure counter transitions: no DXCore
 * adapter, no open thunk and no GPU, so this runs anywhere.
 */

#include "snapshot_refcount.hpp"

#include "unit_test_harness.h"

using wsl::thunk::SnapshotRefcount;
using ReleaseAction = SnapshotRefcount::ReleaseAction;

TEST_CASE(a_cold_refcount_holds_nothing) {
  SnapshotRefcount refs;
  CHECK_EQ(refs.count(), 0u);
}

/* The first acquire built the snapshot and published it. */
TEST_CASE(the_first_acquire_takes_one_reference) {
  SnapshotRefcount refs;
  refs.OnSnapshotPublished();
  CHECK_EQ(refs.count(), 1u);
}

/* A second component - rocprofiler-sdk alongside ROCr - finds the snapshot
 * already built and takes a reference on it instead of rebuilding.
 */
TEST_CASE(a_second_acquire_adds_a_reference) {
  SnapshotRefcount refs;
  refs.OnSnapshotPublished();
  refs.AddReference();
  CHECK_EQ(refs.count(), 2u);
}

/* The release path end to end. With both components holding it, the first
 * release keeps the snapshot alive for the one still reading it and only the
 * last one drops it.
 */
TEST_CASE(two_acquires_need_two_releases_to_drop_the_snapshot) {
  SnapshotRefcount refs;
  refs.OnSnapshotPublished();
  refs.AddReference();

  CHECK(refs.Release() == ReleaseAction::kKeepSnapshot);
  CHECK(refs.Release() == ReleaseAction::kDropSnapshot);
  CHECK_EQ(refs.count(), 0u);
}

/* An unmatched release must be rejected, not wrap the unsigned counter round
 * to UINT32_MAX and leave a snapshot nobody can ever drop.
 */
TEST_CASE(an_unmatched_release_is_rejected_without_underflowing) {
  SnapshotRefcount refs;

  CHECK(refs.Release() == ReleaseAction::kUnbalanced);
  CHECK_EQ(refs.count(), 0u);

  /* and again, in case the first one corrupted the count */
  CHECK(refs.Release() == ReleaseAction::kUnbalanced);
  CHECK_EQ(refs.count(), 0u);
}

/* topology_drop_snapshot() and topology_discard_partial_snapshot() run on the
 * error path of an acquire, which can fail either before or after publishing.
 * So the reset has to hold whatever the count was and leave nothing behind for
 * a later release to find.
 */
TEST_CASE(an_error_unwind_resets_the_count_from_any_depth) {
  SnapshotRefcount refs;
  refs.OnSnapshotPublished();
  refs.AddReference();
  refs.AddReference();
  CHECK_EQ(refs.count(), 3u);

  refs.Clear();  // topology_drop_snapshot()
  CHECK_EQ(refs.count(), 0u);
  CHECK(refs.Release() == ReleaseAction::kUnbalanced);
}

/* A fork child inherits the parent's count along with its address space, and
 * the atfork handler zeroes it so the child cannot release references it never
 * took and tear down WDDMDevices belonging to the parent. Its own next acquire
 * then has to look like a cold one: build, publish, own exactly one reference.
 */
TEST_CASE(a_fork_child_can_rebuild_from_a_cleared_count) {
  SnapshotRefcount refs;
  refs.OnSnapshotPublished();
  refs.AddReference();
  refs.Clear();  // topology_clear_snapshot_refs()

  refs.OnSnapshotPublished();
  CHECK_EQ(refs.count(), 1u);
  CHECK(refs.Release() == ReleaseAction::kDropSnapshot);
}

int main() { return unittest::RunAllTests(); }
