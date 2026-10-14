/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Lazy, collective creation of the DDA AllToAll fabric buffers (push staging buffer, Direct flag
 * buffer): the agreement protocol, independent of HIP so host tests can drive it with fakes.
 * The HIP side (allocation, export, import, peer tables) is DdaA2aLazyOps in
 * dda_alltoall_fabric_lazy_host.h.
 * See LICENSE.txt for license information.
 ************************************************************************/

#ifndef DDA_ALLTOALL_LAZY_H_
#define DDA_ALLTOALL_LAZY_H_

#include "nccl.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dda_a2a_lazy {

// Parts (bit mask).
constexpr uint32_t kPush = 1u;    // push staging buffer ([flags | data])
constexpr uint32_t kDirect = 2u;  // Direct flag buffer

// ncclComm::ddaA2aLazy
constexpr int kUntried = 0;  // fabric path up, buffers not created yet
constexpr int kDone = 1;     // agreement ran (the agreed parts exist, possibly none)
constexpr int kOff = -1;     // no fabric path on this comm, or the agreement could not run

// One rank's contribution: the parts it created and exported, the push data size, and the
// opaque export descriptors of its buffers (Blob: fixed size, trivially copyable).
template <class Blob>
struct Rec {
  uint32_t ok = 0;
  uint32_t pad = 0;
  int64_t pushBytes = 0;
  Blob blob{};
};

struct Result {
  uint32_t parts = 0;      // identical on every rank
  int64_t pushBytes = 0;   // identical on every rank: min over ranks (0 without kPush)
};

// Smallest push data size offered when the configured one does not fit.
constexpr int64_t kPushMinBytes = 64ll << 20;
// Device memory a rank must still have free after creating the push buffer. The comm keeps taking
// p2p for in-place AllToAll and for sizes outside the push window, and p2p allocates its connection
// buffers on its first such call (about 0.9 GiB per rank for an 8-rank AllToAll on gfx1250). A p2p
// connection that cannot be allocated on one rank leaves the peers waiting in bootstrap until
// ncclCommAbort, so the push buffer must not take the memory p2p still needs: without this
// headroom the rank offers no buffer, and the agreement turns push off on every rank.
constexpr int64_t kPushMinFreeBytes = 2ll << 30;

// Push data sizes a rank tries to allocate, largest first: the configured size, then halves (4 KiB
// multiples) while >= minBytes and >= nRanks * 4 KiB, keeping only those whose allocation
// (flagBytes + size) leaves at least minFree of freeBytes. Empty: the rank offers no buffer.
inline std::vector<int64_t> pushSizes(int64_t want, int nRanks, int64_t flagBytes, int64_t freeBytes,
                                      int64_t minBytes = kPushMinBytes, int64_t minFree = kPushMinFreeBytes) {
  std::vector<int64_t> v;
  for (int64_t b = want; b > 0; b = b / 2 / 4096 * 4096) {
    if (flagBytes + b + minFree <= freeBytes) v.push_back(b);
    if (b / 2 < minBytes || b / 2 < (int64_t)nRanks * 4096) break;
  }
  return v;
}

// Ops:
//   using Blob = ...;  int rank() const;  int nRanks() const;
//   ncclResult_t allGather(void* buf, size_t bytesPerRank);    // bootstrap allgather
//   void setup(uint32_t want, int64_t pushBytes, Rec<Blob>* mine);   // local create + export:
//                                       // sets mine->ok (subset of want), mine->pushBytes, blob
//   uint32_t import(const Rec<Blob>* all, uint32_t parts, int64_t pushBytes);  // map peers of
//                                       // `parts`, build device peer tables; returns parts mapped
//   void release(uint32_t parts);       // undo setup + import of `parts` (any subset, idempotent)
//
// Every rank makes the same bootstrap calls (two allgathers, or one when no part survives the
// first, which every rank sees identically) whatever fails locally, so a local failure can never
// leave another rank waiting in bootstrap: it turns into "part disabled on every rank".
// Parts and push size are decided from the gathered records alone, so they are rank-uniform.
// A bootstrap failure is returned as is (the comm's bootstrap is broken: report, do not guess).
template <class Ops>
ncclResult_t agree(Ops& ops, uint32_t want, int64_t wantPushBytes, Result* out) {
  using R = Rec<typename Ops::Blob>;
  *out = Result{};
  const int n = ops.nRanks();
  const int me = ops.rank();
  R mine{};
  ops.setup(want, wantPushBytes, &mine);
  if (mine.ok & ~want) ops.release(mine.ok & ~want);
  mine.ok &= want;
  if ((mine.ok & kPush) && mine.pushBytes <= 0) ops.release(kPush);
  if (!(mine.ok & kPush) || mine.pushBytes <= 0) {
    mine.ok &= ~kPush;
    mine.pushBytes = 0;
  }
  std::vector<R> all(n);
  all[me] = mine;
  ncclResult_t r = ops.allGather(all.data(), sizeof(R));
  if (r != ncclSuccess) {
    ops.release(mine.ok);
    return r;
  }
  uint32_t parts = kPush | kDirect;
  int64_t pushBytes = 0;
  for (int i = 0; i < n; ++i) {
    parts &= all[i].ok;
    if (all[i].ok & kPush) pushBytes = (pushBytes == 0 || all[i].pushBytes < pushBytes) ? all[i].pushBytes : pushBytes;
  }
  if (!(parts & kPush)) pushBytes = 0;
  if (parts == 0) {
    ops.release(mine.ok);
    return ncclSuccess;
  }
  std::vector<uint32_t> mapped(n, 0);
  mapped[me] = ops.import(all.data(), parts, pushBytes) & parts;
  r = ops.allGather(mapped.data(), sizeof(uint32_t));
  if (r != ncclSuccess) {
    ops.release(mine.ok);
    return r;
  }
  for (int i = 0; i < n; ++i) parts &= mapped[i];
  ops.release(mine.ok & ~parts);
  out->parts = parts;
  out->pushBytes = (parts & kPush) ? pushBytes : 0;
  return ncclSuccess;
}

}  // namespace dda_a2a_lazy

#endif  // DDA_ALLTOALL_LAZY_H_
