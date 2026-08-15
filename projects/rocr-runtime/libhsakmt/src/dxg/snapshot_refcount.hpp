/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

namespace wsl {
namespace thunk {

/* Reference count for the shared KMT topology snapshot.
 *
 * More than one component in a process consumes the topology - ROCr, and on
 * WSL rocprofiler-sdk as well - so hsaKmtAcquireSystemProperties() hands out
 * references to one snapshot and only the last hsaKmtReleaseSystemProperties()
 * tears it down.
 *
 * The transitions live here, apart from the WDDM objects they guard, so they
 * can be exercised without a GPU, a DXCore adapter or an open thunk. The
 * caller still owns the actual teardown; every method below only reports what
 * the caller must do next. Serialization is the caller's job too: in the thunk
 * every one of these runs under hsakmtRuntime::hsakmt_mutex.
 *
 * librocdxg and libhsa-runtime64 are both built from rocr-runtime and are
 * expected to ship together. A runtime out of some other package is refused
 * where it is new enough to ask: DxgAbiCheck() answers with this build's
 * sizeof(HsaNodeProperties), 396 against ROCm 7.2.x's 376. One too old to ask
 * never reaches that check, and what then keeps it from tearing down a
 * snapshot another consumer still holds is this count, which rejects a release
 * it never handed out.
 */
class SnapshotRefcount {
 public:
  enum class ReleaseAction {
    /* No reference to give back. An unmatched release must be rejected
     * rather than allowed to tear down a snapshot somebody else still owns.
     */
    kUnbalanced,
    /* A reference went away but others remain; the snapshot stays. */
    kKeepSnapshot,
    /* That was the last reference; the caller drops the snapshot. */
    kDropSnapshot,
  };

  uint32_t count() const { return refs_; }

  /* A caller found a snapshot that is already built and took a reference on
   * it.
   */
  void AddReference() { ++refs_; }

  /* A caller built the snapshot and published it. The count is assigned, not
   * incremented: nothing can hold a reference to a snapshot that did not
   * exist a moment ago, and a build that failed part way never gets here.
   */
  void OnSnapshotPublished() { refs_ = 1; }

  ReleaseAction Release() {
    if (refs_ == 0) return ReleaseAction::kUnbalanced;
    if (--refs_ == 0) return ReleaseAction::kDropSnapshot;
    return ReleaseAction::kKeepSnapshot;
  }

  /* The snapshot is gone or was never ours: the last release dropped it, a
   * failed build unwound it, or a fork child disowned what it inherited.
   * Whatever the reason, no caller holds a reference to it any more.
   */
  void Clear() { refs_ = 0; }

 private:
  uint32_t refs_ = 0;
};

}  // namespace thunk
}  // namespace wsl
