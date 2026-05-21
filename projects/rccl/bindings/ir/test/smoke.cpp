/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 *
 * Smoke test for the librccl_device.bc bitcode artifact.
 *
 * Validates end-to-end that a downstream HIP application can:
 *   1. Pre-link librccl_device.bc into its own device translation unit
 *      via `-Xclang -mlink-bitcode-file=<path>/librccl_device.bc`.
 *   2. Call a public `extern "C" __device__` thunk from a kernel.
 *   3. Get back a result that matches a host-side recomputation of the
 *      same arithmetic.
 *
 * Test cases:
 *   [A] Bucket A   ncclGetPeerPointerTeam        — always present.
 *   [B] Bucket B   ncclCoopAnyInit{Thread,Warp,Cta}/Size/ThreadRank
 *                  — only compiled when SMOKE_ENABLE_COOP is defined,
 *                    which run_smoke.sh enables iff librccl_device.bc
 *                    was built with -DRCCL_ENABLE_NCCL_COOP_ANY=ON.
 *
 * The bucket B cases exercise the ncclCoopAny vtable: InitX writes the
 * vtable pointer into the storage struct, then Size / ThreadRank do an
 * indirect call back into the bitcode. So we're not just verifying that
 * exported thunks resolve and run, we're verifying that an entire
 * polymorphic-dispatch path stays intact through clang -> opt ->
 * llvm-link -> lld -> AMDGPU LTO.
 ************************************************************************/
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* Mirror of struct ncclWindow_vidmem from
 * src/include/nccl_device/impl/core__types.h. Layout must match the
 * one the bitcode was compiled against; this struct is the public ABI
 * the bitcode reads through. */
struct ncclWindow_vidmem {
  void*    winHost;
  char*    lsaFlatBase;
  int      lsaRank;
  int      worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
};
using ncclWindow_t = ncclWindow_vidmem*;

/* Mirror of struct ncclTeam from src/include/nccl_device/core.h */
struct ncclTeam {
  int nRanks, rank, stride;
};

/* Bucket A entry point we want to exercise. The extern "C" declaration
 * here is what the offload linker resolves against the matching
 * definition in librccl_device.bc. */
extern "C" __device__ void* ncclGetPeerPointerTeam(
    ncclWindow_t w, size_t offset, ncclTeam tm, int peer);

#ifdef SMOKE_ENABLE_COOP
/* Opaque mirror of ncclCoopAny from src/include/nccl_device/coop.h.
 *
 *   struct ncclCoopAny {
 *     struct Storage { alignas(alignof(void*)) char space[16]; };
 *     Storage       storage;
 *     VTable const* vtable;
 *   };
 *
 * Size = 16 + 8 = 24 B with 8 B alignment on a 64-bit target. We model
 * it here as a plain 24-byte aligned blob so the test stays a pure
 * C-ABI consumer of the bitcode (no inclusion of coop.h, no template
 * machinery, no need to define RCCL_ENABLE_NCCL_COOP_ANY in this TU). */
struct alignas(8) ncclCoopAnyOpaque {
  unsigned char bytes[24];
};

extern "C" __device__ void ncclCoopAnyInitThread(ncclCoopAnyOpaque* coop);
extern "C" __device__ void ncclCoopAnyInitWarp(ncclCoopAnyOpaque* coop);
extern "C" __device__ void ncclCoopAnyInitCta(ncclCoopAnyOpaque* coop);
extern "C" __device__ int  ncclCoopSize(const ncclCoopAnyOpaque* coop);
extern "C" __device__ int  ncclCoopThreadRank(const ncclCoopAnyOpaque* coop);
#endif  /* SMOKE_ENABLE_COOP */

/* Pure-host recomputation of the same arithmetic the bitcode performs
 * (see ncclGetPeerPointer overload in impl/core__funcs.h, plus
 * nccl::utility::add4G in utility.h):
 *
 *   i = lsaRank + (peer - tm.rank) * tm.stride
 *   delta4G = i * stride4G
 *   base_int = (uintptr_t)lsaFlatBase
 *   base_int = (base_int & 0xFFFFFFFF) | ((uint64_t)((uint32_t)(base_int >> 32) + delta4G) << 32)
 *   return  base_int + offset
 */
static uintptr_t host_expected(uintptr_t base, int lsaRank,
                               uint32_t stride4G, ncclTeam tm,
                               int peer, size_t offset) {
  int i = lsaRank + (peer - tm.rank) * tm.stride;
  uint32_t delta4G = (uint32_t)((int32_t)i * (int32_t)stride4G);
  uint32_t lo = (uint32_t)(base & 0xFFFFFFFFu);
  uint32_t hi = (uint32_t)(base >> 32) + delta4G;
  uintptr_t shifted = ((uintptr_t)hi << 32) | lo;
  return shifted + offset;
}

__global__ void k_call_thunk(
    const ncclWindow_vidmem* w, size_t offset,
    ncclTeam tm, int peer, void** out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *out = ncclGetPeerPointerTeam(const_cast<ncclWindow_t>(w),
                                  offset, tm, peer);
  }
}

#ifdef SMOKE_ENABLE_COOP
/* Per-thread output for the coop tests: size and thread_rank as
 * reported by the bitcode-provided wrapper API. */
struct CoopProbe {
  int size;
  int thread_rank;
};

__global__ void k_coop_thread(CoopProbe* out) {
  ncclCoopAnyOpaque c;
  ncclCoopAnyInitThread(&c);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&c);
  out[tid].thread_rank = ncclCoopThreadRank(&c);
}

__global__ void k_coop_warp(CoopProbe* out) {
  ncclCoopAnyOpaque c;
  ncclCoopAnyInitWarp(&c);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&c);
  out[tid].thread_rank = ncclCoopThreadRank(&c);
}

__global__ void k_coop_cta(CoopProbe* out) {
  ncclCoopAnyOpaque c;
  ncclCoopAnyInitCta(&c);
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid].size        = ncclCoopSize(&c);
  out[tid].thread_rank = ncclCoopThreadRank(&c);
}
#endif  /* SMOKE_ENABLE_COOP */

#define HIP_CHECK(stmt) do {                                        \
    hipError_t _e = (stmt);                                         \
    if (_e != hipSuccess) {                                         \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",      \
                   (int)_e, hipGetErrorName(_e),                    \
                   __FILE__, __LINE__, hipGetErrorString(_e));      \
      std::exit(2);                                                 \
    }                                                               \
  } while (0)

int main() {
  /* Fabricate a window whose lsaFlatBase has interesting upper bits so
   * add4G's high-half-add is observable. 0x0000_0001_0000_0000 sits
   * exactly at the 4 GiB boundary. */
  ncclWindow_vidmem w_host = {};
  w_host.winHost     = nullptr;
  w_host.lsaFlatBase = (char*)(uintptr_t)0x100000000ull;
  w_host.lsaRank     = 0;
  w_host.worldRank   = 0;
  w_host.stride4G    = 2;
  w_host.mcOffset4K  = 0;

  const ncclTeam tm  = { /*nRanks=*/4, /*rank=*/0, /*stride=*/1 };
  const int      peer   = 1;
  const size_t   offset = 128;

  /* Push the window to device memory. */
  ncclWindow_vidmem* w_dev = nullptr;
  HIP_CHECK(hipMalloc(&w_dev, sizeof(ncclWindow_vidmem)));
  HIP_CHECK(hipMemcpy(w_dev, &w_host, sizeof(ncclWindow_vidmem),
                      hipMemcpyHostToDevice));

  /* Output slot for the kernel result. */
  void**  out_dev = nullptr;
  void*   out_host = nullptr;
  HIP_CHECK(hipMalloc(&out_dev, sizeof(void*)));
  HIP_CHECK(hipMemset(out_dev, 0, sizeof(void*)));

  k_call_thunk<<<1, 1>>>(w_dev, offset, tm, peer, out_dev);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(&out_host, out_dev, sizeof(void*),
                      hipMemcpyDeviceToHost));

  const uintptr_t got    = (uintptr_t)out_host;
  const uintptr_t expect = host_expected((uintptr_t)w_host.lsaFlatBase,
                                          w_host.lsaRank, w_host.stride4G,
                                          tm, peer, offset);

  std::printf("ncclGetPeerPointerTeam: got=0x%016lx expect=0x%016lx ",
              (unsigned long)got, (unsigned long)expect);

  HIP_CHECK(hipFree(out_dev));
  HIP_CHECK(hipFree(w_dev));

  int failures = 0;
  if (got != expect) {
    std::printf("[FAIL]\n");
    failures++;
  } else {
    std::printf("[OK]\n");
  }

#ifdef SMOKE_ENABLE_COOP
  /* -------- Bucket B: ncclCoopAny vtable dispatch -------- */
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  const int warpSizeHost = prop.warpSize;

  auto run_coop = [&](const char* name,
                      auto&& launcher /* (CoopProbe*) -> void */,
                      int nThreads,
                      auto expectFn /* (int tid)->std::pair<int,int> */)
  {
    CoopProbe* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, sizeof(CoopProbe) * nThreads));
    HIP_CHECK(hipMemset(d_out, 0, sizeof(CoopProbe) * nThreads));
    launcher(d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    CoopProbe* h_out = new CoopProbe[nThreads];
    HIP_CHECK(hipMemcpy(h_out, d_out, sizeof(CoopProbe) * nThreads,
                        hipMemcpyDeviceToHost));

    int bad = 0;
    int first_bad_tid = -1;
    CoopProbe first_bad{};
    std::pair<int,int> first_bad_expect{0,0};
    for (int t = 0; t < nThreads; ++t) {
      auto e = expectFn(t);
      if (h_out[t].size != e.first || h_out[t].thread_rank != e.second) {
        if (bad == 0) {
          first_bad_tid = t;
          first_bad = h_out[t];
          first_bad_expect = e;
        }
        bad++;
      }
    }
    std::printf("%s: nThreads=%d bad=%d", name, nThreads, bad);
    if (bad) {
      std::printf("  first_bad tid=%d got{size=%d,rank=%d} "
                  "expect{size=%d,rank=%d}",
                  first_bad_tid, first_bad.size, first_bad.thread_rank,
                  first_bad_expect.first, first_bad_expect.second);
      std::printf(" [FAIL]\n");
      failures++;
    } else {
      std::printf(" [OK]\n");
    }
    delete[] h_out;
    HIP_CHECK(hipFree(d_out));
  };

  /* ncclCoopThread (ncclCoopTile<1>):
   *   size()        = 1
   *   thread_rank() = lane() % 1 = 0  for every thread */
  run_coop("ncclCoopAny[Thread]",
           [](CoopProbe* o){ k_coop_thread<<<1, 32>>>(o); },
           32,
           [](int /*t*/){ return std::pair<int,int>{1, 0}; });

  /* ncclCoopWarp (ncclCoopTile<WARP_SIZE>):
   *   size()        = WARP_SIZE  (64 on gfx9)
   *   thread_rank() = lane() % WARP_SIZE = lane()
   * Launch one warp's worth of threads in one block so threadIdx.x ==
   * lane. */
  run_coop("ncclCoopAny[Warp]",
           [&](CoopProbe* o){
             k_coop_warp<<<1, warpSizeHost>>>(o);
           },
           warpSizeHost,
           [&](int t){ return std::pair<int,int>{warpSizeHost,
                                                  t % warpSizeHost}; });

  /* ncclCoopCta (whole CTA):
   *   size()        = nThreads in the block
   *   thread_rank() = flattened threadIdx within the block (here = tid) */
  const int ctaN = 128;
  run_coop("ncclCoopAny[Cta]",
           [&](CoopProbe* o){ k_coop_cta<<<1, ctaN>>>(o); },
           ctaN,
           [&](int t){ return std::pair<int,int>{ctaN, t}; });
#else
  std::printf("(bucket B skipped: SMOKE_ENABLE_COOP not defined; "
              "bitcode built without RCCL_ENABLE_NCCL_COOP_ANY)\n");
#endif  /* SMOKE_ENABLE_COOP */

  return failures ? 1 : 0;
}
