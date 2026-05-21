/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * combined_bitcode_test.cpp
 *
 * Sister of combined_inline_test.cpp using the BITCODE path.
 *
 * Same minimal pattern: each lane uses bucket B (ncclCoopAny / Warp) to
 * discover its rank, then uses bucket A (ncclGetPeerPointerTeam) with
 * that rank to compute the peer address. Both calls route through
 * extern "C" __device__ thunks in <nccl_device_wrapper.h>; the bodies
 * are supplied at link time by librccl_device.bc.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>
#include <nccl_device_wrapper.h>  /* bucket A + bucket B C-ABI decls */

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

__global__ void k_combined(char* base, uint32_t stride4G, size_t offset,
                           int tm_nRanks, int tm_rank, int tm_stride,
                           void** out)
{
  /* Bucket B: call the bitcode thunks. */
  ncclCoopAny coop;
  ncclCoopAnyInitWarp(&coop);
  const int my_rank = ncclCoopThreadRank(&coop);

  /* Bucket A: call the bitcode thunk. */
  ncclWindow_vidmem w{};
  w.winHost     = nullptr;
  w.lsaFlatBase = base;
  w.lsaRank     = 0;
  w.worldRank   = 0;
  w.stride4G    = stride4G;
  w.mcOffset4K  = 0;

  ncclTeam tm{ tm_nRanks, tm_rank, tm_stride };
  out[threadIdx.x] = ncclGetPeerPointerTeam(&w, offset, tm, my_rank);
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
  std::printf("[combined-bitcode] devices=%d\n", nDev);

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
                                    /*peer=*/t, offset);
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

  std::printf("[combined-bitcode] failures=%d %s\n",
              failures, failures ? "[FAIL]" : "[OK]");
  return failures ? 1 : 0;
}
