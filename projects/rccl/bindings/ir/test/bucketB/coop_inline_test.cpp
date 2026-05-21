/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * coop_inline_test.cpp
 *
 * Multi-GPU test for the bucket B (RCCL_ENABLE_NCCL_COOP_ANY) APIs
 * using the INLINE path: <nccl_device.h> brings in coop.h, which
 * defines ncclCoopAny + the per-Impl tile/cta types as inline device
 * code. Construction, vtable wiring, and dispatch all happen inside the
 * consumer's translation unit; no bitcode is involved.
 *
 * Three kernels exercise the three most-used coop shapes:
 *
 *   k_coop_thread   — ncclCoopThread (== ncclCoopTile<1>)
 *                     expected: size==1, thread_rank==0 for every thread.
 *
 *   k_coop_warp     — ncclCoopWarp   (== ncclCoopTile<WARP_SIZE>)
 *                     expected: size==warpSize, thread_rank == lane id.
 *
 *   k_coop_cta      — ncclCoopCta    (whole CTA)
 *                     expected: size==blockDim, thread_rank == flat tid.
 *
 * For every visible GPU we run all three kernels and check each thread's
 * (size, thread_rank) pair against the host-side expectation. Sister
 * file coop_bitcode_test.cpp runs the same logical tests but routed
 * through the C-ABI thunks supplied by librccl_device.bc.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nccl.h>
/* RCCL_ENABLE_NCCL_COOP_ANY must be set on the consumer command line
 * (the runner script passes -DRCCL_ENABLE_NCCL_COOP_ANY=1). Without it
 * coop.h does not define ncclCoopAny and this TU does not compile. */
#include <nccl_device.h>

struct CoopProbe { int size; int thread_rank; };

__global__ void k_coop_thread(CoopProbe* out) {
  ncclCoopAny coop(ncclCoopThread{});
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = coop.size();
  out[tid].thread_rank = coop.thread_rank();
}

__global__ void k_coop_warp(CoopProbe* out) {
  ncclCoopAny coop(ncclCoopWarp{});
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = coop.size();
  out[tid].thread_rank = coop.thread_rank();
}

__global__ void k_coop_cta(CoopProbe* out) {
  ncclCoopAny coop(ncclCoopCta{});
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = coop.size();
  out[tid].thread_rank = coop.thread_rank();
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

/* Run one kernel launch and verify each thread's (size, rank) against
 * an expected-value function. Returns number of mismatched threads. */
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
  std::printf("[coop-inline] devices=%d\n", nDev);

  int failures = 0;

  for (int d = 0; d < nDev; ++d) {
    HIP_CHECK(hipSetDevice(d));
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    const int warpSizeHost = prop.warpSize;
    std::printf("[gpu%d:%s warpSize=%d]\n", d, prop.name, warpSizeHost);

    /* ncclCoopThread:  size==1, rank==0 everywhere. */
    failures += run_case(
        "Thread", 32,
        [](CoopProbe* o){ k_coop_thread<<<1, 32>>>(o); },
        [](int){ return std::pair<int,int>{1, 0}; });

    /* ncclCoopWarp:    size==warpSize, rank == lane id == tid % warpSize.
     * One block of warpSize threads -> threadIdx.x == lane. */
    failures += run_case(
        "Warp", warpSizeHost,
        [&](CoopProbe* o){ k_coop_warp<<<1, warpSizeHost>>>(o); },
        [&](int t){
          return std::pair<int,int>{warpSizeHost, t % warpSizeHost};
        });

    /* ncclCoopCta:    size==blockDim, rank == flat tid within block. */
    const int ctaN = 128;
    failures += run_case(
        "Cta", ctaN,
        [&](CoopProbe* o){ k_coop_cta<<<1, ctaN>>>(o); },
        [&](int t){ return std::pair<int,int>{ctaN, t}; });
  }

  std::printf("[coop-inline] failures=%d %s\n",
              failures, failures ? "[FAIL]" : "[OK]");
  return failures ? 1 : 0;
}
