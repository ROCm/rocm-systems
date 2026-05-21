/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * combined_inline_test.cpp
 *
 * Minimal but meaningful test that exercises BOTH bucket A and bucket B
 * APIs in the same kernel via the INLINE path (<nccl_device.h>).
 *
 * Pattern:
 *   each lane in a warp ->
 *     1. uses bucket B (ncclCoopAny + ncclCoopWarp) to discover its
 *        own thread_rank within the warp (== lane id);
 *     2. uses bucket A (ncclGetPeerPointer with the team overload) to
 *        compute the symmetric-window address belonging to peer == its
 *        own thread_rank.
 *
 * That's the canonical swizzle pattern in real distributed kernels:
 *   "tell me who I am; tell me where peer-N's data lives".
 * It's a single launch, one kernel body, but both APIs are wired
 * end-to-end together. The host re-derives every expected address with
 * pure arithmetic for verification.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>
#include <nccl_device.h>          /* bucket A + bucket B inline bodies */

/* Pure-host re-derivation of the kernel-side arithmetic, identical to
 * the bucket A test:
 *   i        = lsaRank + (peer - tm.rank) * tm.stride
 *   delta4G  = i * stride4G
 *   shifted  = base with its high 32 bits incremented by delta4G
 *   result   = shifted + offset
 */
static uintptr_t host_expected(uintptr_t base,
                               int lsaRank, uint32_t stride4G,
                               int tm_rank, int tm_stride,
                               int peer, size_t offset)
{
  int      i       = lsaRank + (peer - tm_rank) * tm_stride;
  uint32_t delta4G = (uint32_t)((int32_t)i * (int32_t)stride4G);
  uint32_t lo      = (uint32_t)(base & 0xFFFFFFFFu);
  uint32_t hi      = (uint32_t)(base >> 32) + delta4G;
  uintptr_t shift  = ((uintptr_t)hi << 32) | lo;
  return shift + offset;
}

/* Single kernel that touches both API surfaces:
 *   - ncclCoopAny(ncclCoopWarp{})   -> bucket B
 *   - ncclGetPeerPointer(...)        -> bucket A
 * launched with exactly warpSize threads so threadIdx.x == lane. */
__global__ void k_combined(char* base, uint32_t stride4G, size_t offset,
                           int tm_nRanks, int tm_rank, int tm_stride,
                           void** out)
{
  /* Bucket B: discover this thread's identity within the warp. */
  ncclCoopAny coop(ncclCoopWarp{});
  const int my_rank = coop.thread_rank();         /* == lane id */

  /* Bucket A: compute peer-pointer using the coop-supplied rank. */
  ncclWindow_vidmem w{};
  w.winHost     = nullptr;
  w.lsaFlatBase = base;
  w.lsaRank     = 0;
  w.worldRank   = 0;
  w.stride4G    = stride4G;
  w.mcOffset4K  = 0;

  ncclTeam tm{ tm_nRanks, tm_rank, tm_stride };
  out[threadIdx.x] = ncclGetPeerPointer(&w, offset, tm, my_rank);
}

#define HIP_CHECK(stmt) do {                                          \
    hipError_t _e = (stmt);                                           \
    if (_e != hipSuccess) {                                           \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",        \
                   (int)_e, hipGetErrorName(_e),                      \
                   __FILE__, __LINE__, hipGetErrorString(_e));        \
      std::exit(2);                                                   \
    }                                                                 \
  } while (0)

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev <= 0) { std::fprintf(stderr, "No HIP devices.\n"); return 2; }
  std::printf("[combined-inline] devices=%d\n", nDev);

  /* Synthetic, never-dereferenced window parameters. The high half of
   * base makes the add4G shift observable. */
  const uintptr_t base    = (uintptr_t)0x100000000ull;
  const uint32_t  stride4G = 2;
  const size_t    offset   = 128;
  const int       tm_nRanks = 8;
  const int       tm_rank   = 0;
  const int       tm_stride = 1;

  int failures = 0;

  for (int d = 0; d < nDev; ++d) {
    HIP_CHECK(hipSetDevice(d));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    const int warpSizeHost = prop.warpSize;
    std::printf("[gpu%d:%s warpSize=%d]\n", d, prop.name, warpSizeHost);

    void** d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(void*) * warpSizeHost));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(void*) * warpSizeHost));

    k_combined<<<1, warpSizeHost>>>(
        (char*)base, stride4G, offset,
        tm_nRanks, tm_rank, tm_stride,
        d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<void*> got(warpSizeHost, nullptr);
    HIP_CHECK(hipMemcpy(got.data(), d_out, sizeof(void*) * warpSizeHost,
                        hipMemcpyDeviceToHost));

    int bad = 0;
    for (int t = 0; t < warpSizeHost; ++t) {
      uintptr_t exp = host_expected(base, /*lsaRank=*/0, stride4G,
                                    tm_rank, tm_stride,
                                    /*peer==coop.thread_rank==*/t,
                                    offset);
      uintptr_t obs = (uintptr_t)got[t];
      if (exp != obs) {
        if (bad == 0)
          std::printf("  lane=%d got=0x%016lx expect=0x%016lx [FAIL]\n",
                      t, (unsigned long)obs, (unsigned long)exp);
        bad++;
      }
    }
    std::printf("  combined  warpSize=%d bad=%d %s\n",
                warpSizeHost, bad, bad ? "[FAIL]" : "[OK]");
    if (bad) failures++;
    HIP_CHECK(hipFree(d_out));
  }

  std::printf("[combined-inline] failures=%d %s\n",
              failures, failures ? "[FAIL]" : "[OK]");
  return failures ? 1 : 0;
}
