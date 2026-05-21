/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * coop_bitcode_test.cpp
 *
 * Sister of coop_inline_test.cpp using the BITCODE path.
 *
 *   - <nccl_device_wrapper.h> forward-declares the bucket B thunks
 *     (ncclCoopAnyInit{Thread,Warp,Cta}, ncclCoopSize, ncclCoopThreadRank)
 *     as `extern "C" __device__`. Their bodies live in librccl_device.bc
 *     and are supplied at link time via -Xoffload-linker.
 *
 *   - The wrapper transitively pulls in <nccl_device.h>, so the type
 *     ncclCoopAny itself is available here as a plain struct — we just
 *     never call any of its inline methods directly. All dispatch goes
 *     through the C-ABI thunks, which means the test exercises the full
 *     bitcode flow:  init -> vtable populated -> indirect call into
 *     bitcode trampoline -> per-Impl method.
 *
 * Test cases are intentionally identical to coop_inline_test.cpp so the
 * two paths can be compared at the functional level (each kernel must
 * pass independently for the two implementations to agree).
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>
/* RCCL_ENABLE_NCCL_COOP_ANY: same gate as the inline test. The wrapper
 * header's bucket B forward-declarations are #if-gated on this, and the
 * bitcode itself was built with it on. */
#include <nccl_device_wrapper.h>

struct CoopProbe { int size; int thread_rank; };

__global__ void k_coop_thread(CoopProbe* out) {
  /* Storage for the coop. ncclCoopAny is a 24-byte POD-like struct
   * (16-byte aligned storage + 8-byte vtable ptr); default-construct
   * leaves it uninitialised, and the bitcode init thunk does
   * placement-new + vtable assignment. */
  ncclCoopAny coop;
  ncclCoopAnyInitThread(&coop);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&coop);
  out[tid].thread_rank = ncclCoopThreadRank(&coop);
}

__global__ void k_coop_warp(CoopProbe* out) {
  ncclCoopAny coop;
  ncclCoopAnyInitWarp(&coop);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&coop);
  out[tid].thread_rank = ncclCoopThreadRank(&coop);
}

__global__ void k_coop_cta(CoopProbe* out) {
  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&coop);
  out[tid].thread_rank = ncclCoopThreadRank(&coop);
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

template <typename Launch, typename Expect>
static int run_case(const char* tag, int nThreads,
                    Launch&& launch, Expect&& expectFn) {
  CoopProbe* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(CoopProbe) * nThreads));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(CoopProbe) * nThreads));
  launch(d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<CoopProbe> h_out(nThreads);
  HIP_CHECK(hipMemcpy(h_out.data(), d_out,
                      sizeof(CoopProbe) * nThreads,
                      hipMemcpyDeviceToHost));

  int bad = 0;
  for (int t = 0; t < nThreads; ++t) {
    auto e = expectFn(t);
    if (h_out[t].size != e.first || h_out[t].thread_rank != e.second) {
      if (bad == 0) {
        std::printf("    %s tid=%d got{size=%d,rank=%d} "
                    "expect{size=%d,rank=%d}\n",
                    tag, t,
                    h_out[t].size, h_out[t].thread_rank,
                    e.first, e.second);
      }
      bad++;
    }
  }
  HIP_CHECK(hipFree(d_out));
  std::printf("  %-16s nThreads=%-4d bad=%d %s\n",
              tag, nThreads, bad, bad ? "[FAIL]" : "[OK]");
  return bad;
}

int main() {
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  if (nDev <= 0) { std::fprintf(stderr, "No HIP devices.\n"); return 2; }
  std::printf("[coop-bitcode] devices=%d\n", nDev);

  int failures = 0;

  for (int d = 0; d < nDev; ++d) {
    HIP_CHECK(hipSetDevice(d));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    const int warpSizeHost = prop.warpSize;
    std::printf("[gpu%d:%s warpSize=%d]\n", d, prop.name, warpSizeHost);

    failures += run_case(
        "Thread", 32,
        [](CoopProbe* o){ k_coop_thread<<<1, 32>>>(o); },
        [](int){ return std::pair<int,int>{1, 0}; });

    failures += run_case(
        "Warp", warpSizeHost,
        [&](CoopProbe* o){ k_coop_warp<<<1, warpSizeHost>>>(o); },
        [&](int t){
          return std::pair<int,int>{warpSizeHost, t % warpSizeHost};
        });

    const int ctaN = 128;
    failures += run_case(
        "Cta", ctaN,
        [&](CoopProbe* o){ k_coop_cta<<<1, ctaN>>>(o); },
        [&](int t){ return std::pair<int,int>{ctaN, t}; });
  }

  std::printf("[coop-bitcode] failures=%d %s\n",
              failures, failures ? "[FAIL]" : "[OK]");
  return failures ? 1 : 0;
}
