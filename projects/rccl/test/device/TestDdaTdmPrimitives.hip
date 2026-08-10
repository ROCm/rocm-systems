// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Standalone checks for the DDA Simple-protocol TDM primitives in
// algorithms/CollCommonTdm.h. The primitives are exercised directly rather than
// through the fabric kernels so this runs on a single device: peer buffers are
// just separate local allocations, and the rank-to-rank barrier is replaced by
// separate kernel launches.
//
// Each case runs twice, once with 256B-aligned buffers (the TDM path) and once
// with buffers offset by 16 bytes (the vector fallback), so both sides of the
// alignment branch are covered.
//
// Build:
//   hipcc -x hip --offload-arch=gfx1250 -O2 -std=c++17 -DUSE_ROCM \
//     -I<rccl>/src/include -I<rccl>/src/device -I<tdm_fallback_dir> \
//     TestDdaTdmPrimitives.hip -o TestDdaTdmPrimitives

#include <hip/hip_runtime.h>

#include "algorithms/CollCommonTdm.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace meta::comms;

#define HIPCHECK(cmd)                                                                                                  \
  do {                                                                                                                 \
    hipError_t e = (cmd);                                                                                              \
    if (e != hipSuccess) {                                                                                             \
      printf("FATAL %s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e));                                            \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)

constexpr int kNRanks = 4;
constexpr size_t kBytesPerRank = 32 * 1024;

// ---------------------------------------------------------------------------
// Kernels: one phase each, so a kernel boundary stands in for the GPU barrier.
// ---------------------------------------------------------------------------

__launch_bounds__(kTdmThreadsPerBlock) __global__
  void kCopy(const uint8_t* src, uint8_t* dst, size_t nbytes) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  tdmCopyRange(src, dst, nbytes, tdmWindow(lds, 0), tdmWindow(lds, 1), tdmWarpTile());
}

template <int NRANKS_CT>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void kReduce(uint8_t* const* peers, uint8_t* dst, int nRanks, size_t srcOff, size_t dstOff, size_t nbytes) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  tdmReduceRange<float, NRANKS_CT, false>(peers, dst, nullptr, nRanks, srcOff, dstOff, nbytes, tdmWindow(lds, 0),
                                          tdmWindow(lds, 1), tdmWarpTile());
}

template <int NRANKS_CT>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void kAllGather(uint8_t* const* peers, uint8_t* dst, int selfRank, int nRanks, size_t perRankBytes) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  tdmAllGather<NRANKS_CT>(peers, dst, selfRank, nRanks, perRankBytes, /*enableOffset=*/false, tdmWindow(lds, 0),
                          tdmWindow(lds, 1), tdmWarpTile());
}

template <int NRANKS_CT>
__launch_bounds__(kTdmThreadsPerBlock) __global__
  void kAllToAll(uint8_t* const* peers, uint8_t* dst, int selfRank, int nRanks, size_t perRankBytes) {
  __shared__ __align__(kTdmRowBytes) uint8_t lds[kTdmLdsBytes];
  tdmAllToAll<NRANKS_CT>(peers, dst, selfRank, nRanks, perRankBytes, tdmWindow(lds, 0), tdmWindow(lds, 1),
                         tdmWarpTile());
}

// ---------------------------------------------------------------------------

static int gFailures = 0;

static void report(const char* name, bool ok, const char* detail = "") {
  printf("  %-46s %s%s%s\n", name, ok ? "PASS" : "FAIL", *detail ? " : " : "", detail);
  if (!ok) {
    gFailures++;
  }
}

// A distinct value per (rank, element) so a mis-tiled copy shows up as a
// mismatch rather than coincidentally matching.
static float pattern(int rank, size_t i) {
  return (float)(rank * 1000000 + (int)(i % 65536));
}

static void runSuite(const char* label, size_t skewBytes) {
  printf("%s (skew=%zu bytes -> %s path)\n", label, skewBytes, skewBytes ? "vector fallback" : "TDM");

  const size_t nElems = kBytesPerRank / sizeof(float);
  const size_t allocBytes = kBytesPerRank * kNRanks + skewBytes + kTdmRowBytes;

  std::vector<uint8_t*> peerBase(kNRanks), peer(kNRanks);
  for (int r = 0; r < kNRanks; ++r) {
    HIPCHECK(hipMalloc(&peerBase[r], allocBytes));
    HIPCHECK(hipMemset(peerBase[r], 0, allocBytes));
    peer[r] = peerBase[r] + skewBytes;
  }
  uint8_t *dstBase = nullptr, *srcBase = nullptr;
  HIPCHECK(hipMalloc(&dstBase, allocBytes));
  HIPCHECK(hipMalloc(&srcBase, allocBytes));
  HIPCHECK(hipMemset(dstBase, 0, allocBytes));
  uint8_t* dst = dstBase + skewBytes;
  uint8_t* src = srcBase + skewBytes;

  uint8_t** peersDev = nullptr;
  HIPCHECK(hipMalloc(&peersDev, kNRanks * sizeof(uint8_t*)));
  HIPCHECK(hipMemcpy(peersDev, peer.data(), kNRanks * sizeof(uint8_t*), hipMemcpyHostToDevice));

  std::vector<float> host(nElems * kNRanks), out(nElems * kNRanks);

  // --- copy of one rank's worth of bytes ---------------------------------
  for (size_t i = 0; i < nElems; ++i) {
    host[i] = pattern(7, i);
  }
  HIPCHECK(hipMemcpy(src, host.data(), kBytesPerRank, hipMemcpyHostToDevice));
  {
    auto gb = getTdmGridAndBlockDims(kBytesPerRank, 256);
    kCopy<<<gb.first, gb.second>>>(src, dst, kBytesPerRank);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(out.data(), dst, kBytesPerRank, hipMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < nElems && ok; ++i) {
      ok = out[i] == pattern(7, i);
    }
    report("tdmCopyRange", ok);
  }

  // --- seed every peer with its own pattern ------------------------------
  for (int r = 0; r < kNRanks; ++r) {
    for (size_t i = 0; i < nElems * kNRanks; ++i) {
      host[i] = pattern(r, i);
    }
    HIPCHECK(hipMemcpy(peer[r], host.data(), kBytesPerRank * kNRanks, hipMemcpyHostToDevice));
  }

  // --- reduce-scatter shape: fold shard `self` across all peers ----------
  for (int self = 0; self < kNRanks; ++self) {
    const size_t srcOff = (size_t)self * kBytesPerRank;
    HIPCHECK(hipMemset(dst, 0, kBytesPerRank));
    auto gb = getTdmGridAndBlockDims(kBytesPerRank, 256);
    kReduce<kNRanks><<<gb.first, gb.second>>>(peersDev, dst, kNRanks, srcOff, 0, kBytesPerRank);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(out.data(), dst, kBytesPerRank, hipMemcpyDeviceToHost));
    bool ok = true;
    size_t badIdx = 0;
    float got = 0, want = 0;
    for (size_t i = 0; i < nElems; ++i) {
      float expect = 0;
      for (int r = 0; r < kNRanks; ++r) {
        expect += pattern(r, self * nElems + i);
      }
      if (out[i] != expect) {
        ok = false;
        badIdx = i;
        got = out[i];
        want = expect;
        break;
      }
    }
    char buf[160] = "";
    if (!ok) {
      snprintf(buf, sizeof(buf), "shard %d elem %zu got %.1f want %.1f", self, badIdx, got, want);
    }
    char name[80];
    snprintf(name, sizeof(name), "tdmReduceRange (NRANKS_CT=4, shard %d)", self);
    report(name, ok, buf);
  }

  // Runtime-rank-count instantiation takes a different unroll path.
  {
    HIPCHECK(hipMemset(dst, 0, kBytesPerRank));
    auto gb = getTdmGridAndBlockDims(kBytesPerRank, 256);
    kReduce<0><<<gb.first, gb.second>>>(peersDev, dst, kNRanks, kBytesPerRank, 0, kBytesPerRank);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(out.data(), dst, kBytesPerRank, hipMemcpyDeviceToHost));
    bool ok = true;
    for (size_t i = 0; i < nElems && ok; ++i) {
      float expect = 0;
      for (int r = 0; r < kNRanks; ++r) {
        expect += pattern(r, nElems + i);
      }
      ok = out[i] == expect;
    }
    report("tdmReduceRange (NRANKS_CT=0, runtime)", ok);
  }

  // --- all-gather: peer r's base shard lands at slot r -------------------
  for (int self = 0; self < kNRanks; ++self) {
    HIPCHECK(hipMemset(dst, 0, kBytesPerRank * kNRanks));
    auto gb = getTdmGridAndBlockDims(kBytesPerRank, 256);
    kAllGather<kNRanks><<<gb.first, gb.second>>>(peersDev, dst, self, kNRanks, kBytesPerRank);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(out.data(), dst, kBytesPerRank * kNRanks, hipMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < kNRanks && ok; ++r) {
      for (size_t i = 0; i < nElems && ok; ++i) {
        ok = out[r * nElems + i] == pattern(r, i);
      }
    }
    char name[80];
    snprintf(name, sizeof(name), "tdmAllGather (self=%d)", self);
    report(name, ok);
  }

  // --- alltoall: this rank's shard pulled out of every peer --------------
  for (int self = 0; self < kNRanks; ++self) {
    HIPCHECK(hipMemset(dst, 0, kBytesPerRank * kNRanks));
    auto gb = getTdmGridAndBlockDims(kBytesPerRank, 256);
    kAllToAll<kNRanks><<<gb.first, gb.second>>>(peersDev, dst, self, kNRanks, kBytesPerRank);
    HIPCHECK(hipDeviceSynchronize());
    HIPCHECK(hipMemcpy(out.data(), dst, kBytesPerRank * kNRanks, hipMemcpyDeviceToHost));
    bool ok = true;
    for (int r = 0; r < kNRanks && ok; ++r) {
      for (size_t i = 0; i < nElems && ok; ++i) {
        ok = out[r * nElems + i] == pattern(r, self * nElems + i);
      }
    }
    char name[80];
    snprintf(name, sizeof(name), "tdmAllToAll (self=%d)", self);
    report(name, ok);
  }

  for (int r = 0; r < kNRanks; ++r) {
    HIPCHECK(hipFree(peerBase[r]));
  }
  HIPCHECK(hipFree(dstBase));
  HIPCHECK(hipFree(srcBase));
  HIPCHECK(hipFree(peersDev));
}

int main() {
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  printf("device: %s  TDM supported: %s\n", prop.gcnArchName, tdmSimpleSupported(0) ? "yes" : "no");
  printf("window=%u B  issueWarps=%u  block=%u  lds=%u B (%u blocks/CU of %u B)\n\n", kTdmWindowBytes, kTdmIssueWarps,
         kTdmThreadsPerBlock, kTdmLdsBytes, kTdmMaxLdsPerBlockBytes / kTdmLdsBytes, kTdmMaxLdsPerBlockBytes);

  runSuite("aligned", 0);
  printf("\n");
  runSuite("misaligned", 16);

  printf("\n%s (%d failure%s)\n", gFailures ? "FAILED" : "ALL PASS", gFailures, gFailures == 1 ? "" : "s");
  return gFailures ? 1 : 0;
}
