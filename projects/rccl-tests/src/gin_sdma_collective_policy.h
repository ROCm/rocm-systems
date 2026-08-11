/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared design constants for the GIN-SDMA collective path (deviceImpl == 3).
// Currently the single source of truth for the two put-size limits that
// common.h::ginPutChunked segments every GIN-tier put against. Kept as a
// standalone, dependency-free header (no ncclDevComm, no GPU intrinsics, no
// getenv) so the same values can be shared by the device kernels and validated
// on the host.

#ifndef GIN_SDMA_COLLECTIVE_POLICY_H_
#define GIN_SDMA_COLLECTIVE_POLICY_H_

#include <cstddef>

namespace gin_sdma {

// Max bytes per single gin.put() on the Anvil-SDMA backend. The SDMA linear-copy
// descriptor count field is 30 bits and 1-based (HW encodes count = bytes - 1,
// see rocr-runtime amd_blit_sdma.cpp / sdma_registers.h), so the largest single
// packet is (2^30 - 1) + 1 = 2^30 = exactly 1 GiB. A put of >1 GiB silently
// truncates: a 2 GiB put encodes count = (2^31-1) & 0x3FFFFFFF = 2^30-1 and the
// HW copies only 1 GiB, corrupting the transfer. Every GIN-tier put must be
// split into segments of at most this size (see common.h::ginPutChunked).
//
// Set to exactly 1 GiB: this is the hardware maximum, so the segmentation clamp
// (seg <= kGinPutMaxBytes) guarantees count = seg-1 <= 2^30-1 = 0x3FFFFFFF, which
// fills the 30-bit field exactly with no truncation. Zero margin by design; do
// NOT raise above 2^30. 1 GiB is a multiple of 32 B, satisfying the copy
// descriptor's 32 B length alignment.
static constexpr size_t kGinPutMaxBytes = 1024ull * 1024 * 1024;  // 1 GiB (2^30, HW max)

// Max bytes per single gin.put() that the Anvil-SDMA backend copies *reliably*
// on MI355X + ROCm 7.13 (NCCL_GIN_TYPE=5). This is SMALLER than kGinPutMaxBytes:
// the 30-bit count field bounds correctness at 1 GiB, but a single copy
// descriptor at/above 256 MiB (2^28) on the fused COPY_LINEAR_WAIT_SIGNAL_MI4
// path stalls the SDMA engine, so the fused copy never lands AND its SignalInc
// never fires -> every rank spins forever in waitSignal (a HANG, not a data
// miscompare). Measured on 8x MI355X (2026-08-07, alltoall_perf -D 3): a single
// 256 MiB/peer put (AllToAll @ 2 GiB total) HANGS; capping each put to 128 MiB
// (2 puts/peer) completes with identical bandwidth (busbw ~423 GB/s, unchanged
// vs the 1 GiB total case). 128 MiB is proven safe with zero measured perf loss;
// do not raise without re-measuring. ginPutChunked segments every GIN-tier put
// to this size, so it protects ALL GIN-SDMA collectives that route through it.
static constexpr size_t kGinSdmaSafeCopyBytes = 128ull * 1024 * 1024;  // 128 MiB (reliable single-copy max)

}  // namespace gin_sdma

#endif  // GIN_SDMA_COLLECTIVE_POLICY_H_
