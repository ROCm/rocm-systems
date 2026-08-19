/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Put-size limits and segment math for the GIN Anvil-SDMA device backend.
// The single source of truth for the two per-put byte caps that
// ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA> segments every SDMA copy
// against, plus the pure segmentation arithmetic itself. Kept as a standalone,
// dependency-free header (no ncclDevComm, no GPU intrinsics, no getenv) so the
// same code is (a) called from the SDMA backend Put on device and (b) unit-tested
// on the host with GoogleTest (define GIN_SDMA_HOST_ONLY to drop the device
// attributes and compile it as plain host C++; see
// projects/rccl/test/gin/gin_sdma_policy_test.cpp).

#ifndef _NCCL_DEVICE_GIN_ANVIL_SDMA_PUT_POLICY_H_
#define _NCCL_DEVICE_GIN_ANVIL_SDMA_PUT_POLICY_H_

#include <cstddef>

// The device backend gets __host__ __device__; plain host builds (and the host
// unit test, which defines GIN_SDMA_HOST_ONLY) get no attribute so the header
// compiles as ordinary C++ with no device codegen.
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(GIN_SDMA_HOST_ONLY)
#define GIN_SDMA_HD __host__ __device__
#else
#define GIN_SDMA_HD
#endif

namespace gin_sdma {

// Max bytes per single SDMA copy on the Anvil-SDMA backend. The SDMA linear-copy
// descriptor count field is 30 bits and 1-based (HW encodes count = bytes - 1,
// see rocr-runtime amd_blit_sdma.cpp / sdma_registers.h), so the largest single
// packet is (2^30 - 1) + 1 = 2^30 = exactly 1 GiB. A put of >1 GiB silently
// truncates: a 2 GiB put encodes count = (2^31-1) & 0x3FFFFFFF = 2^30-1 and the
// HW copies only 1 GiB, corrupting the transfer. Every SDMA copy must be split
// into segments of at most this size.
//
// Set to exactly 1 GiB: this is the hardware maximum, so the segmentation clamp
// (seg <= kGinPutMaxBytes) guarantees count = seg-1 <= 2^30-1 = 0x3FFFFFFF, which
// fills the 30-bit field exactly with no truncation. Zero margin by design; do
// NOT raise above 2^30. 1 GiB is a multiple of 32 B, satisfying the copy
// descriptor's 32 B length alignment.
static constexpr size_t kGinPutMaxBytes = 1024ull * 1024 * 1024;  // 1 GiB (2^30, HW max)

// Max bytes per single SDMA copy that the Anvil-SDMA backend copies *reliably*
// on MI355X + ROCm 7.13 (NCCL_GIN_TYPE=5). This is SMALLER than kGinPutMaxBytes:
// the 30-bit count field bounds correctness at 1 GiB, but a single copy
// descriptor at/above 256 MiB (2^28) on the fused COPY_LINEAR_WAIT_SIGNAL_MI4
// path stalls the SDMA engine, so the fused copy never lands AND its SignalInc
// never fires -> every rank spins forever in waitSignal (a HANG, not a data
// miscompare). Measured on 8x MI355X (2026-08-07, alltoall_perf -D 3): a single
// 256 MiB/peer put (AllToAll @ 2 GiB total) HANGS; capping each copy to 128 MiB
// (2 copies/peer) completes with identical bandwidth (busbw ~423 GB/s, unchanged
// vs the 1 GiB total case). 128 MiB is proven safe with zero measured perf loss;
// do not raise without re-measuring. The backend Put segments every SDMA copy to
// this size, so it protects ALL callers of gin.put() on the Anvil-SDMA backend.
static constexpr size_t kGinSdmaSafeCopyBytes = 128ull * 1024 * 1024;  // 128 MiB (reliable single-copy max)

// The single per-copy segmentation limit the backend Put clamps to: the smaller
// of the two constants above (correctness bound AND reliability bound). Compile-
// time constant so the device loop and the host unit test share one value.
static constexpr size_t kGinPutSegBytes =
    kGinSdmaSafeCopyBytes < kGinPutMaxBytes ? kGinSdmaSafeCopyBytes : kGinPutMaxBytes;

// ------------------------------ segment math ------------------------------
//
// Pure arithmetic for splitting one peer transfer into <=kGinPutSegBytes copies.
// The Anvil-SDMA backend Put drives its ::sdma_anvil::put() loop entirely from
// these two functions, so the same code that runs on device is what the host
// unit test exercises. Every SDMA copy must satisfy: no segment exceeds the cap,
// the segments tile [0,bytes) contiguously and sum back to bytes, and exactly
// one segment is flagged final (it alone carries the caller's SignalInc).

// One SDMA-copy segment of a chunked transfer.
struct PutSegment {
  size_t offset;   // byte offset of this segment within the transfer
  size_t bytes;    // segment length in bytes (always <= maxSeg)
  bool   isFinal;  // last segment: carries the caller's remote action (signal)
};

// Number of copy segments a `bytes`-length transfer splits into when each copy
// is capped at `maxSeg`. A zero-length transfer still issues exactly one (empty)
// copy so its trailing signal fires; otherwise it is ceil(bytes/maxSeg).
// maxSeg == 0 is a caller error and yields 0 (callers pass a nonzero cap).
GIN_SDMA_HD inline size_t ginPutSegmentCount(size_t bytes, size_t maxSeg) {
  if (maxSeg == 0) return 0;
  if (bytes == 0) return 1;
  return (bytes + maxSeg - 1) / maxSeg;
}

// The i-th (0-based, i < ginPutSegmentCount) segment of a `bytes`-length transfer
// capped at `maxSeg`. Segments tile [0,bytes) contiguously; the last one is
// flagged isFinal so the backend carries the remote action on it alone.
GIN_SDMA_HD inline PutSegment ginPutSegmentAt(size_t bytes, size_t maxSeg, size_t i) {
  PutSegment s{0, 0, false};
  s.offset = i * maxSeg;
  const size_t rem = bytes - s.offset;
  s.bytes = rem > maxSeg ? maxSeg : rem;
  s.isFinal = (s.offset + s.bytes >= bytes);
  return s;
}

}  // namespace gin_sdma

#endif  // _NCCL_DEVICE_GIN_ANVIL_SDMA_PUT_POLICY_H_
