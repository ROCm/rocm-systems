/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef LL128_RECV_WAIT_H_
#define LL128_RECV_WAIT_H_

#include <hip/hip_runtime.h>
#include <stdint.h>

// Windowed LL128 receive wait, used by Primitives<ProtoLL128>::recvWaitLoad when
// RCCL_LL128_RECV_WINDOW_ENABLE is set. One peer's slice is Rows rows; each row is
// one 16-byte load per lane, and the flag lanes hold a line's flag in that row's
// odd word.
//
// The pieces take the row load, the per-row ballot and the spin test as callables
// instead of reading Primitives, so the unit tests can drive them on the host
// against a simulated wire.

// One bit per row.
template <int Rows>
__host__ __device__ __forceinline__ constexpr uint32_t ll128AllRows() {
  static_assert(Rows >= 1 && Rows <= 32, "reload tracking holds one bit per row");
  return Rows == 32 ? ~0u : (1u << Rows) - 1;
}

// Stores rowBallot(r), row r's warp vote, in ok[r] and returns true when every row
// voted all of warpMask. Lanes not holding a flag must vote yes.
template <int Rows, typename RowBallot>
__host__ __device__ __forceinline__ bool ll128VoteRows(uint64_t (&ok)[Rows], uint64_t warpMask, RowBallot&& rowBallot) {
  uint64_t allOk = warpMask;
#pragma unroll
  for (int r = 0; r < Rows; r++) {
    ok[r] = rowBallot(r);
    allOk &= ok[r];
  }
  return allOk == warpMask;
}

// Rows whose every flag was valid at the vote that filled ok.
template <int Rows>
__host__ __device__ __forceinline__ uint32_t ll128ValidRows(const uint64_t (&ok)[Rows], uint64_t warpMask) {
  uint32_t rows = 0;
#pragma unroll
  for (int r = 0; r < Rows; r++) {
    if (ok[r] == warpMask) rows |= 1u << r;
  }
  return rows;
}

// Sets [lo, hi) to span the lowest to the highest row not in skip, so skipped rows
// inside it are reloaded too. Returns false, leaving lo and hi unset, when every
// row is skipped.
template <int Rows>
__host__ __device__ __forceinline__ bool ll128ReloadWindow(uint32_t skip, int& lo, int& hi) {
  uint32_t pending = ll128AllRows<Rows>() & ~skip;
  if (pending == 0) return false;
  lo = __builtin_ctz(pending);
  hi = 32 - __builtin_clz(pending);
  return true;
}

// Calls loadRow(r) for every row in skip's reload window, lowest first.
template <int Rows, typename LoadRow>
__host__ __device__ __forceinline__ void ll128LoadRows(uint32_t skip, LoadRow&& loadRow) {
  int lo, hi;
  if (!ll128ReloadWindow<Rows>(skip, lo, hi)) return;
#pragma unroll
  for (int r = 0; r < Rows; r++) {
    if (r >= lo && r < hi) loadRow(r);
  }
}

// Spins until every flag of the slice is valid, leaving the slice in the registers
// loadRow fills. Each spin reloads only the window of rows not all valid at the
// previous vote. keepWaiting(needReload) returns whether the warp spins again; it
// is where the caller checks abort.
//
// Confirm = false: a row's data is taken from the load that returned its valid
// flags. Confirm = true: every row is reloaded once after a vote saw it valid.
template <int Rows, bool Confirm, typename LoadRow, typename RowBallot, typename KeepWaiting>
__host__ __device__ __forceinline__ void ll128RecvWaitRows(uint64_t warpMask, LoadRow&& loadRow, RowBallot&& rowBallot,
                                                           KeepWaiting&& keepWaiting) {
  constexpr uint32_t AllRows = ll128AllRows<Rows>();
  uint64_t ok[Rows];

#pragma unroll
  for (int r = 0; r < Rows; r++) loadRow(r);
  if (ll128VoteRows<Rows>(ok, warpMask, rowBallot)) {
    if (Confirm) {
#pragma unroll
      for (int r = 0; r < Rows; r++) loadRow(r);
    }
    return;
  }
  uint32_t valid = ll128ValidRows<Rows>(ok, warpMask);
  uint32_t done = Confirm ? 0 : valid;
  bool needReload;
  do {
    ll128LoadRows<Rows>(done, loadRow);
    if (Confirm) done |= valid;
    needReload = !ll128VoteRows<Rows>(ok, warpMask, rowBallot);
    valid = needReload ? ll128ValidRows<Rows>(ok, warpMask) : AllRows;
    if (!Confirm) done = valid;
  } while (keepWaiting(needReload));
  if (Confirm) ll128LoadRows<Rows>(done, loadRow);
}

#endif  // LL128_RECV_WAIT_H_
