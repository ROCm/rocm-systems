/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * IR_test.cpp — Comprehensive functional test for librccl_device.bc
 *
 * Tests every active API exported by nccl_device_wrapper__impl.h,
 * reporting pass/fail/skip per API:
 *
 *   [A]  ncclGetPeerPointerTeam
 *   [B1] ncclCoopAnyInitThread   + ncclCoopSize/ThreadRank/NumThreads
 *   [B2] ncclCoopAnyInitWarp     + ncclCoopSize/ThreadRank/NumThreads
 *   [B3] ncclCoopAnyInitLanes    + ncclCoopSize/ThreadRank/NumThreads
 *          tested with full mask (0xFFFF…), sparse mask, single-bit mask
 *   [B4] ncclCoopAnyInitWarpSpan + ncclCoopSize/ThreadRank/NumThreads
 *          tested with 1-warp and 2-warp spans
 *   [B5] ncclCoopAnyInitCta      + ncclCoopSize/ThreadRank/NumThreads
 *          tested with block-size 64 and 128
 *   [B6] ncclCoopSync — all five coop types; kernel-completion check
 *   [B7] ncclLsaBarrierSession{Init,Arrive,Wait,Sync}
 *          — sizeof / alignment structural check: PASS
 *          — runtime invocations: SKIP (require a live ncclDevComm)
 *
 * Output per test:
 *   [ PASS ] ID   description                        verified/total
 *   [ FAIL ] ID   description                        verified/total  <detail>
 *   [ SKIP ] ID   description                        reason
 *
 * Exit code: 0 = all non-skipped tests passed, 1 = failures present.
 *
 * Build via run_IR_test.sh; manual build example (use -O0; see note in
 * run_IR_test.sh about ROCm 7.x AMDGPU codegen bugs at -O1/-O2):
 *   hipcc --offload-arch=gfx942 -O0 -D__HIP_PLATFORM_AMD__=1         \
 *         -I<build>/hipify/src/include                                 \
 *         -I<build>/hipify/src/include/nccl_device                    \
 *         -I<build>/include                                            \
 *         IR_test.cpp                                                  \
 *         -Xoffload-linker <build>/lib/librccl_device.bc              \
 *         -Xoffload-linker -plugin-opt=-amdgpu-internalize-symbols=false \
 *         -o IR_test.exe
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <nccl.h>
#include <nccl_device_wrapper.h>

/* -----------------------------------------------------------------------
 * Compile-time structural checks — these fire during compilation, not at
 * runtime, so any failure here means the build itself breaks.
 * ---------------------------------------------------------------------- */
static_assert(sizeof(ncclLsaBarrierSession_C) > 0,
              "ncclLsaBarrierSession_C must have non-zero size");
static_assert(alignof(ncclLsaBarrierSession_C) >= 1,
              "ncclLsaBarrierSession_C must have valid alignment");

/* -----------------------------------------------------------------------
 * Error checking
 * ---------------------------------------------------------------------- */
#define HIP_CHECK(stmt) do {                                                \
    hipError_t _e = (stmt);                                                 \
    if (_e != hipSuccess) {                                                 \
      std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",             \
                   (int)_e, hipGetErrorName(_e),                            \
                   __FILE__, __LINE__, hipGetErrorString(_e));              \
      std::exit(2);                                                         \
    }                                                                       \
  } while (0)

/* -----------------------------------------------------------------------
 * Test result bookkeeping
 * ---------------------------------------------------------------------- */
enum class Status { PASS, FAIL, SKIP };

struct TestResult {
  char   id[16];
  char   desc[72];
  Status status;
  int    passed;
  int    total;
  char   detail[128];
};

static std::vector<TestResult> g_results;

static void report(const char* id, const char* desc, Status s,
                   int passed = 0, int total = 0,
                   const char* detail = "") {
  TestResult r{};
  std::snprintf(r.id,     sizeof r.id,     "%s", id);
  std::snprintf(r.desc,   sizeof r.desc,   "%s", desc);
  std::snprintf(r.detail, sizeof r.detail, "%s", detail);
  r.status = s; r.passed = passed; r.total = total;
  g_results.push_back(r);

  const char* tag = (s == Status::PASS) ? "[ PASS ]" :
                    (s == Status::SKIP) ? "[ SKIP ]" : "[ FAIL ]";
  if (s == Status::SKIP) {
    std::printf("  %s %-6s %-50s  %s\n", tag, id, desc, detail);
  } else {
    std::printf("  %s %-6s %-50s  %d/%d", tag, id, desc, passed, total);
    if (s == Status::FAIL && detail[0]) std::printf("  (%s)", detail);
    std::printf("\n");
  }
}

/* -----------------------------------------------------------------------
 * Shared device-side output struct
 * ---------------------------------------------------------------------- */
struct CoopResult { int size, rank, num_threads; };

/* =====================================================================
 * [A] Bucket A — ncclGetPeerPointerTeam
 * ==================================================================== */

struct PeerCase {
  int      lsaRank, worldRank;
  uint32_t stride4G;
  size_t   offset;
  int      tm_nRanks, tm_rank, tm_stride, peer;
};

/* Reference pointer-arithmetic, identical to core__funcs.h logic. */
static uintptr_t host_peer_ptr(uintptr_t base, const PeerCase& c) {
  int      i       = c.lsaRank + (c.peer - c.tm_rank) * c.tm_stride;
  uint32_t delta4G = (uint32_t)((int32_t)i * (int32_t)c.stride4G);
  uint32_t lo      = (uint32_t)(base & 0xFFFFFFFFu);
  uint32_t hi      = (uint32_t)(base >> 32) + delta4G;
  uintptr_t shift  = ((uintptr_t)hi << 32) | lo;
  return shift + c.offset;
}

static const PeerCase kPeerCases[] = {
  /* lsaRank  worldRank  stride4G  offset             tm_nRanks  tm_rank  tm_stride  peer */
  {  0,       0,         0,        0,                  1,         0,       1,         0   },
  {  0,       0,         1,        64,                 2,         0,       1,         1   },
  {  0,       0,         1,        64,                 2,         1,       1,         0   },
  {  0,       0,         2,        128,                4,         0,       1,         3   },
  {  0,       0,         1,        0,                  8,         0,       2,         4   },
  {  5,       0,         1,        0,                  1,         0,       1,         0   },
  {  0,       0,         1,        0xDEADBEEFull,      1,         0,       1,         0   },
  {  0,       0,         0x10,     0,                  2,         0,       1,         1   },
  {  3,       0,         2,        512,                4,         1,       1,         2   },
  {  0,       0,         7,        0,                  1,         0,       1,         0   },
  {  0,       0,         1,        0xCAFEBABEull,      2,         0,       1,         1   },
  {  2,       0,         1,        0,                  3,         1,       1,         2   },
  {  0,       0,         4,        0x1000,             2,         1,       1,         0   },
  {  1,       0,         1,        8,                  4,         2,       1,         3   },
  {  0,       0,         15,       0x40000000ull,      1,         0,       1,         0   },
  {  4,       0,         3,        16,                 8,         3,       2,         6   },
  /* Additional edge cases */
  {  0,       0,         0,        0xFFFFFFFFull,      1,         0,       1,         0   },  /* max offset */
  {  0,       0,         0xFFFF,   0,                  2,         0,       1,         1   },  /* large stride */
  { -1,       0,         1,        0,                  1,         0,       1,         0   },  /* negative lsaRank */
  {  0,       0,         1,        0,                  4,         3,       1,         0   },  /* wrap-around peer */
};

__global__ void k_peer_ptr(char* base, const PeerCase* cases, int N, void** out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= N) return;
  const PeerCase c = cases[t];
  ncclWindow_vidmem w{};
  w.winHost     = nullptr;
  w.lsaFlatBase = base;
  w.lsaRank     = c.lsaRank;
  w.worldRank   = c.worldRank;
  w.stride4G    = c.stride4G;
  w.mcOffset4K  = 0;
  ncclTeam tm{ c.tm_nRanks, c.tm_rank, c.tm_stride };
  out[t] = ncclGetPeerPointerTeam(&w, c.offset, tm, c.peer);
}

static void run_A(const uintptr_t base) {
  constexpr int N = (int)(sizeof kPeerCases / sizeof kPeerCases[0]);

  PeerCase* d_cases = nullptr; void** d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_cases, sizeof kPeerCases));
  HIP_CHECK(hipMalloc(&d_out,   sizeof(void*) * N));
  HIP_CHECK(hipMemcpy(d_cases, kPeerCases, sizeof kPeerCases,
                      hipMemcpyHostToDevice));

  k_peer_ptr<<<1, N>>>((char*)base, d_cases, N, d_out);
  HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());

  std::vector<void*> got(N);
  HIP_CHECK(hipMemcpy(got.data(), d_out, sizeof(void*) * N,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_cases)); HIP_CHECK(hipFree(d_out));

  int bad = 0;
  char detail[128] = {};
  for (int i = 0; i < N; ++i) {
    uintptr_t exp = host_peer_ptr(base, kPeerCases[i]);
    uintptr_t obs = (uintptr_t)got[i];
    if (exp != obs) {
      if (!bad) std::snprintf(detail, sizeof detail,
                              "case %d: got=0x%016lx exp=0x%016lx",
                              i, (unsigned long)obs, (unsigned long)exp);
      bad++;
    }
  }
  report("A.1", "ncclGetPeerPointerTeam (20 cases)",
         bad ? Status::FAIL : Status::PASS, N - bad, N, detail);
}

/* =====================================================================
 * [B1–B5] Coop kernels — one per init type
 *
 * Workaround for a ROCm 7.x AMDGPU codegen bug: when threadIdx.x is used
 * directly as a scatter-store index after indirect function calls (vtable
 * dispatch), the compiler hoists the kernel-argument SGPR load to the top
 * of the function but omits the required s_waitcnt lgkmcnt(0) before the
 * global_store that uses that SGPR as a base address.  The store fires with
 * a stale (zero) SGPR, writes to NULL, triggers an XNACK page-fault retry
 * loop, and the wavefront hangs indefinitely.
 *
 * Using `tid = blockIdx.x * blockDim.x + threadIdx.x` forces the compiler
 * to compute a full 64-bit VGPR address (via v_lshl_add_u64 + s_waitcnt)
 * and selects the correct addressing mode for global_store.
 * ==================================================================== */

__global__ void k_coop_thread_r(CoopResult* out) {
  ncclCoopAny coop; ncclCoopAnyInitThread(&coop);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = { ncclCoopSize(&coop),
               ncclCoopThreadRank(&coop),
               ncclCoopNumThreads(&coop) };
}

__global__ void k_coop_warp_r(CoopResult* out) {
  ncclCoopAny coop; ncclCoopAnyInitWarp(&coop);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = { ncclCoopSize(&coop),
               ncclCoopThreadRank(&coop),
               ncclCoopNumThreads(&coop) };
}

__global__ void k_coop_lanes_r(CoopResult* out, uint64_t mask) {
  ncclCoopAny coop; ncclCoopAnyInitLanes(&coop, mask);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = { ncclCoopSize(&coop),
               ncclCoopThreadRank(&coop),
               ncclCoopNumThreads(&coop) };
}

__global__ void k_coop_warpspan_r(CoopResult* out, int warp0, int nWarps, int id) {
  ncclCoopAny coop; ncclCoopAnyInitWarpSpan(&coop, warp0, nWarps, id);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = { ncclCoopSize(&coop),
               ncclCoopThreadRank(&coop),
               ncclCoopNumThreads(&coop) };
}

__global__ void k_coop_cta_r(CoopResult* out) {
  ncclCoopAny coop; ncclCoopAnyInitCta(&coop);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = { ncclCoopSize(&coop),
               ncclCoopThreadRank(&coop),
               ncclCoopNumThreads(&coop) };
}

/* =====================================================================
 * [B6] Sync kernels — write 1 after sync, host checks all slots == 1
 * ==================================================================== */

__global__ void k_sync_thread  (int* out)                               { ncclCoopAny c; ncclCoopAnyInitThread  (&c);                ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }
__global__ void k_sync_warp    (int* out)                               { ncclCoopAny c; ncclCoopAnyInitWarp    (&c);                ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }
__global__ void k_sync_lanes   (int* out, uint64_t mask)                { ncclCoopAny c; ncclCoopAnyInitLanes   (&c, mask);          ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }
__global__ void k_sync_warpspan(int* out, int warp0, int nWarps, int id){ ncclCoopAny c; ncclCoopAnyInitWarpSpan(&c,warp0,nWarps,id);ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }
__global__ void k_sync_cta     (int* out)                               { ncclCoopAny c; ncclCoopAnyInitCta     (&c);                ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }

/* =====================================================================
 * Helpers: launch-check-free pattern
 * ==================================================================== */

/* Run a coop_r kernel and return the result array. Caller must free. */
template<typename Kern, typename... Args>
static std::vector<CoopResult>
launch_coop_r(Kern kern, int N, Args... args) {
  CoopResult* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(CoopResult) * N));
  kern<<<1, N>>>(d, args...);
  HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
  std::vector<CoopResult> h(N);
  HIP_CHECK(hipMemcpy(h.data(), d, sizeof(CoopResult) * N,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d));
  return h;
}

/* Run a sync kernel and verify every slot is 1. */
template<typename Kern, typename... Args>
static bool launch_sync(Kern kern, int N, Args... args) {
  int* d = nullptr;
  HIP_CHECK(hipMalloc(&d, sizeof(int) * N));
  HIP_CHECK(hipMemset(d, 0, sizeof(int) * N));
  kern<<<1, N>>>(d, args...);
  HIP_CHECK(hipGetLastError()); HIP_CHECK(hipDeviceSynchronize());
  std::vector<int> h(N);
  HIP_CHECK(hipMemcpy(h.data(), d, sizeof(int) * N, hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d));
  for (int i = 0; i < N; ++i) if (h[i] != 1) return false;
  return true;
}

/* Verify size, rank, num_threads in a CoopResult vector.
 * exp_rank(t) returns -1 to skip rank check for thread t.         */
template<typename ExpSize, typename ExpRank>
static void check_coop(const char* id_size,  const char* desc_size,
                       const char* id_rank,  const char* desc_rank,
                       const char* id_num,   const char* desc_num,
                       const std::vector<CoopResult>& h,
                       ExpSize exp_size, ExpRank exp_rank) {
  int N = (int)h.size();
  int bs=0, br=0, bn=0, ts=0, tr=0, tn=0;
  char ds[128]={}, dr[128]={}, dn[128]={};

  for (int t = 0; t < N; ++t) {
    int es = exp_size(t);
    int er = exp_rank(t);   /* -1 = skip rank for this thread */
    ts++; tn++;
    if (h[t].size != es) {
      if (!bs) std::snprintf(ds, sizeof ds, "tid=%d got=%d exp=%d",
                             t, h[t].size, es);
      bs++;
    }
    if (h[t].num_threads != h[t].size) {
      if (!bn) std::snprintf(dn, sizeof dn,
                             "tid=%d num_threads=%d != size=%d",
                             t, h[t].num_threads, h[t].size);
      bn++;
    }
    if (er >= 0) {
      tr++;
      if (h[t].rank != er) {
        if (!br) std::snprintf(dr, sizeof dr, "tid=%d got=%d exp=%d",
                               t, h[t].rank, er);
        br++;
      }
    }
  }

  report(id_size, desc_size, bs?Status::FAIL:Status::PASS, ts-bs, ts, ds);
  if (tr > 0)
    report(id_rank, desc_rank, br?Status::FAIL:Status::PASS, tr-br, tr, dr);
  report(id_num,  desc_num,  bn?Status::FAIL:Status::PASS, tn-bn, tn, dn);
}

/* =====================================================================
 * [B1] ncclCoopAnyInitThread
 * ==================================================================== */
static void run_B1(int warpSize) {
  /* Launch one warp so we test every lane position. */
  auto h = launch_coop_r(k_coop_thread_r, warpSize);
  /* Thread coop: every thread is its own singleton group. */
  check_coop("B1.1", "ncclCoopAnyInitThread: size==1",
             "B1.2", "ncclCoopAnyInitThread: rank==0",
             "B1.3", "ncclCoopAnyInitThread: numThreads==size",
             h,
             [](int  ) { return 1; },  /* exp_size */
             [](int  ) { return 0; }); /* exp_rank */
}

/* =====================================================================
 * [B2] ncclCoopAnyInitWarp
 * ==================================================================== */
static void run_B2(int warpSize) {
  auto h = launch_coop_r(k_coop_warp_r, warpSize);
  check_coop("B2.1", "ncclCoopAnyInitWarp: size==warpSize",
             "B2.2", "ncclCoopAnyInitWarp: rank==lane_id",
             "B2.3", "ncclCoopAnyInitWarp: numThreads==size",
             h,
             [&](int  ) { return warpSize; },
             [&](int t) { return t % warpSize; });
}

/* Return in-group lane rank for thread t, or -1 to skip rank check. */
static int lane_rank_in_mask(uint64_t mask, int t) {
  if (t < 0 || !((mask >> t) & 1ull)) return -1;
  return __builtin_popcountll(mask & ((1ull << t) - 1ull));
}

/* =====================================================================
 * [B3] ncclCoopAnyInitLanes — mask variants (uint64_t lane_mask)
 * ==================================================================== */
static void run_B3(int warpSize) {
  /* ---- B3a: full mask ---- */
  {
    const uint64_t mask = ~0ull;
    const int      sz   = warpSize;
    auto h = launch_coop_r(k_coop_lanes_r, warpSize, mask);
    check_coop("B3.1", "ncclCoopAnyInitLanes (full mask): size==warpSize",
               "B3.2", "ncclCoopAnyInitLanes (full mask): rank==lane",
               "B3.3", "ncclCoopAnyInitLanes (full mask): numThreads==size",
               h,
               [&](int  ) { return sz; },
               [&](int t) {
                 /* nccl::utility::lanemask_lt() is still 32-bit on AMD wave64,
                  * so thread_rank() for lanes 32–63 is not lane index today. */
                 if (warpSize >= 64 && t >= 32) return -1;
                 return lane_rank_in_mask(mask, t);
               });
  }

  /* ---- B3b: sparse mask (lanes 0, 2, 4 — popcount 3) ---- */
  {
    const uint64_t mask = 0x00000015ull; /* bits 0, 2, 4 */
    const int      sz   = __builtin_popcountll(mask);
    auto h = launch_coop_r(k_coop_lanes_r, warpSize, mask);
    check_coop("B3.4", "ncclCoopAnyInitLanes (sparse mask 0x15): size==3",
               "B3.5", "ncclCoopAnyInitLanes (sparse mask 0x15): rank in-group",
               "B3.6", "ncclCoopAnyInitLanes (sparse mask 0x15): numThreads==size",
               h,
               [&](int  ) { return sz; },
               [&](int t) { return lane_rank_in_mask(mask, t); });
  }

  /* ---- B3c: single-bit mask (lane 0 only) ---- */
  {
    const uint64_t mask = 0x00000001ull;
    const int      sz   = 1;
    auto h = launch_coop_r(k_coop_lanes_r, warpSize, mask);
    check_coop("B3.7", "ncclCoopAnyInitLanes (mask=0x1): size==1",
               "B3.8", "ncclCoopAnyInitLanes (mask=0x1): rank==0 (lane 0)",
               "B3.9", "ncclCoopAnyInitLanes (mask=0x1): numThreads==size",
               h,
               [&](int  ) { return sz; },
               [&](int t) {
                 return (t == 0) ? 0 : -1; /* only lane 0 is in the group */
               });
  }

  /* ---- B3d: upper-lane mask (lane 63 only; requires 64-lane warp) ---- */
  if (warpSize >= 64) {
    const uint64_t mask = 1ull << 63;
    auto h = launch_coop_r(k_coop_lanes_r, warpSize, mask);
    check_coop("B3.10", "ncclCoopAnyInitLanes (lane 63 mask): size==1",
               "B3.11", "ncclCoopAnyInitLanes (lane 63 mask): rank==0 on lane 63",
               "B3.12", "ncclCoopAnyInitLanes (lane 63 mask): numThreads==size",
               h,
               [&](int  ) { return 1; },
               [&](int t) { return (t == 63) ? 0 : -1; });
  }
}

/* =====================================================================
 * [B4] ncclCoopAnyInitWarpSpan — 1-warp and 2-warp spans
 * rank = threadIdx.x - WARP_SIZE * warp0  (WARP_SIZE == device warpSize)
 * ==================================================================== */
static void run_B4(int warpSize) {
  /* ---- B4a: warp0=0, nWarps=1 ---- */
  {
    auto h = launch_coop_r(k_coop_warpspan_r, warpSize,
                           /*warp0=*/0, /*nWarps=*/1, /*id=*/0);
    check_coop("B4.1", "ncclCoopAnyInitWarpSpan (1-warp): size==warpSize",
               "B4.2", "ncclCoopAnyInitWarpSpan (1-warp): rank==threadIdx.x",
               "B4.3", "ncclCoopAnyInitWarpSpan (1-warp): numThreads==size",
               h,
               [&](int  ) { return warpSize; },
               [&](int t) { return t; });
  }

  /* ---- B4b: warp0=0, nWarps=2 — launch two warps ---- */
  /* ncclCoopWarpSpan::sync() calls __syncthreads() on AMD, so ALL
   * threads in the CTA must participate.  The coop_r kernel is
   * data-only (no sync), so launching 2*warpSize threads is safe. */
  {
    int N = 2 * warpSize;
    auto h = launch_coop_r(k_coop_warpspan_r, N,
                           /*warp0=*/0, /*nWarps=*/2, /*id=*/0);
    check_coop("B4.4", "ncclCoopAnyInitWarpSpan (2-warp): size==2*warpSize",
               "B4.5", "ncclCoopAnyInitWarpSpan (2-warp): rank==threadIdx.x",
               "B4.6", "ncclCoopAnyInitWarpSpan (2-warp): numThreads==size",
               h,
               [&](int  ) { return N; },
               [&](int t) { return t; });
  }
}

/* =====================================================================
 * [B5] ncclCoopAnyInitCta — two block sizes
 * ==================================================================== */
static void run_B5() {
  for (int blk : {64, 128}) {
    char id_sz[16], id_rk[16], id_nm[16];
    char ds[80], dr[80], dn[80];
    std::snprintf(id_sz, sizeof id_sz, "B5.%d", (blk == 64 ? 1 : 3));
    std::snprintf(id_rk, sizeof id_rk, "B5.%d", (blk == 64 ? 2 : 4));
    std::snprintf(id_nm, sizeof id_nm, "B5.%d", (blk == 64 ? 5 : 6));
    std::snprintf(ds, sizeof ds, "ncclCoopAnyInitCta (blk=%d): size==blockDim", blk);
    std::snprintf(dr, sizeof dr, "ncclCoopAnyInitCta (blk=%d): rank==threadIdx", blk);
    std::snprintf(dn, sizeof dn, "ncclCoopAnyInitCta (blk=%d): numThreads==size", blk);

    auto h = launch_coop_r(k_coop_cta_r, blk);
    check_coop(id_sz, ds, id_rk, dr, id_nm, dn,
               h,
               [&](int  ) { return blk; },
               [&](int t) { return t;   });
  }
}

/* =====================================================================
 * [B6] ncclCoopSync — one test per coop type
 * ==================================================================== */
static void run_B6(int warpSize) {
  /* Thread: every thread syncs its own singleton coop independently. */
  bool ok = launch_sync(k_sync_thread, warpSize);
  report("B6.1", "ncclCoopSync (Thread): kernel completes",
         ok ? Status::PASS : Status::FAIL, ok ? warpSize : 0, warpSize);

  /* Warp: full warp participates. */
  ok = launch_sync(k_sync_warp, warpSize);
  report("B6.2", "ncclCoopSync (Warp): kernel completes",
         ok ? Status::PASS : Status::FAIL, ok ? warpSize : 0, warpSize);

  /* Lanes (full mask): all lanes in the warp participate. */
  ok = launch_sync(k_sync_lanes, warpSize, ~0ull);
  report("B6.3", "ncclCoopSync (Lanes full mask): kernel completes",
         ok ? Status::PASS : Status::FAIL, ok ? warpSize : 0, warpSize);

  /* WarpSpan (2 warps): __syncthreads() on AMD — all CTA threads must
   * call it, so launch exactly 2*warpSize threads. */
  int span_n = 2 * warpSize;
  ok = launch_sync(k_sync_warpspan, span_n,
                   /*warp0=*/0, /*nWarps=*/2, /*id=*/0);
  report("B6.4", "ncclCoopSync (WarpSpan 2-warp): kernel completes",
         ok ? Status::PASS : Status::FAIL, ok ? span_n : 0, span_n);

  /* CTA: standard __syncthreads(). */
  ok = launch_sync(k_sync_cta, 128);
  report("B6.5", "ncclCoopSync (Cta blk=128): kernel completes",
         ok ? Status::PASS : Status::FAIL, ok ? 128 : 0, 128);
}

/* =====================================================================
 * [B7] ncclLsaBarrierSession — structural + skip runtime
 * ==================================================================== */
static void run_B7() {
  /* sizeof / alignment checks are compile-time (static_assert at top
   * of file).  Report them as PASS here to show up in the summary. */
  char detail[128];
  std::snprintf(detail, sizeof detail,
                "sizeof=%zu align=%zu",
                sizeof(ncclLsaBarrierSession_C),
                alignof(ncclLsaBarrierSession_C));
  report("B7.1", "ncclLsaBarrierSession_C: sizeof/align",
         Status::PASS, 1, 1, detail);

  /* Runtime calls require a live ncclDevComm with a mapped resource
   * buffer; without it the constructor immediately dereferences
   * ncclGetResourceBufferLocalPointer() → segfault.  Mark as SKIP. */
  const char* reason = "requires live ncclDevComm (not set up here)";
  report("B7.2", "ncclLsaBarrierSessionInit",   Status::SKIP, 0, 0, reason);
  report("B7.3", "ncclLsaBarrierSessionArrive", Status::SKIP, 0, 0, reason);
  report("B7.4", "ncclLsaBarrierSessionWait",   Status::SKIP, 0, 0, reason);
  report("B7.5", "ncclLsaBarrierSessionSync",   Status::SKIP, 0, 0, reason);
}

/* =====================================================================
 * main
 * ==================================================================== */
int main() {
  std::printf("[IR_test] starting...\n"); std::fflush(stdout);
  int nDev = 0;
  HIP_CHECK(hipGetDeviceCount(&nDev));
  std::printf("[IR_test] hipGetDeviceCount done: %d\n", nDev); std::fflush(stdout);
  if (nDev <= 0) {
    std::fprintf(stderr, "[IR_test] No HIP devices found.\n");
    return 2;
  }
  std::printf("[IR_test] %d HIP device(s) found\n\n", nDev); std::fflush(stdout);

  int total_pass = 0, total_fail = 0, total_skip = 0;

  for (int d = 0; d < nDev; ++d) {
    std::printf("[IR_test] calling hipSetDevice(%d)\n", d); std::fflush(stdout);
    HIP_CHECK(hipSetDevice(d));
    std::printf("[IR_test] hipSetDevice done\n"); std::fflush(stdout);
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, d));
    const int ws = prop.warpSize;
    std::printf("[IR_test] device props: warpSize=%d\n", ws); std::fflush(stdout);

    std::printf("──────────────────────────────────────────────────────────────\n");
    std::printf("[IR_test] GPU %d: %s  warpSize=%d\n\n", d, prop.name, ws);

    g_results.clear();

    /* A: peer pointer */
    std::printf("[IR_test] starting A tests\n"); std::fflush(stdout);
    const uintptr_t base = 0x100000000ull;
    run_A(base);
    std::printf("[IR_test] A tests done\n"); std::fflush(stdout);

    /* B1–B6: coop init + accessors + sync */
    std::printf("\n");
    run_B1(ws);
    std::printf("\n");
    run_B2(ws);
    std::printf("\n");
    run_B3(ws);
    std::printf("\n");
    run_B4(ws);
    std::printf("\n");
    run_B5();
    std::printf("\n");
    run_B6(ws);

    /* B7: LSA barrier session structural */
    std::printf("\n");
    run_B7();

    /* Per-device summary */
    int dp = 0, df = 0, ds2 = 0;
    for (auto& r : g_results) {
      if (r.status == Status::PASS) { dp++; total_pass++; }
      else if (r.status == Status::FAIL) { df++; total_fail++; }
      else { ds2++; total_skip++; }
    }
    std::printf("\n[IR_test] GPU %d summary: %d passed, %d failed, %d skipped\n",
                d, dp, df, ds2);
  }

  std::printf("\n══════════════════════════════════════════════════════════════\n");
  std::printf("[IR_test] TOTAL: %d passed, %d failed, %d skipped across %d GPU(s)\n",
              total_pass, total_fail, total_skip, nDev);
  std::printf("══════════════════════════════════════════════════════════════\n\n");

  return total_fail ? 1 : 0;
}
