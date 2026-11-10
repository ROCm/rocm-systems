/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * IR_test.cpp — Comprehensive functional test for librccl_device.bc
 *
 * GoogleTest port of the original standalone driver. Tests every active API
 * exported by nccl_device_wrapper__impl.h, with one GTest case per API group
 * so the suite can be filtered with --gtest_filter and consumed by the RCCL
 * test_runner / CI exactly like the other rccl-UnitTests* binaries:
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
 *   [C0] bucket-C export/link coverage — takes the device address of all four
 *          ncclGin/composite barrier thunks and checks each resolves from the
 *          bitcode (covers the functions without needing live resources)
 *   [C1] ncclGinBarrierSession{Init,Sync}
 *          — sizeof / alignment structural check: PASS
 *          — runtime invocations: SKIP (require a live ncclDevComm + GIN net)
 *   [C2] ncclBarrierSession{Init,Sync}  (composite inner-LSA + outer-GIN)
 *          — sizeof / alignment structural check: PASS
 *          — runtime invocations: SKIP (require a live ncclDevComm + GIN net)
 *   [D0] ReduceCopy typed thunks — link coverage for all 64 exports
 *          (ncclLsa{ReduceSum,Copy,ReduceSumCopy}_<T>, ncclLocalReduceSumCopy_<T>
 *          x 10 types; ncclMultimem{ReduceSum,Copy,ReduceSumCopy}_<T> x 8 types)
 *   [D1] ncclLocalReduceSumCopy_<T> — functional, typed over all 10 types:
 *          count sweep, misaligned bases, fan-in/fan-out shapes,
 *          cta/warp/thread coop, out-of-bounds guard checks.
 *          The Lsa* thunks are exercised functionally by the multi-rank
 *          IR_reducecopy_mpi_test.cpp.
 *
 * Build is NOT a normal CMake/GTest build: librccl_device.bc must already
 * exist (cmake -DEMIT_LLVM_IR=ON) and the translation unit must be compiled
 * by hipcc at -O0 with the bitcode routed to the AMDGPU device-side LTO link.
 * The pytest harness in test/ir-device/ drives the build + run in CI (it links
 * GTest and emits the bitcode on demand); see test/ir-device/tests/conftest.py.
 ************************************************************************/

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

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

/* [C] GIN + composite barrier session structs — enabling bucket [C] makes
 * these compile; the static_asserts guard against a regression that would
 * silently strip the wrapper structs back out of the build. */
static_assert(sizeof(ncclGinBarrierSession_C) > 0,
              "ncclGinBarrierSession_C must have non-zero size");
static_assert(alignof(ncclGinBarrierSession_C) >= 1,
              "ncclGinBarrierSession_C must have valid alignment");
static_assert(sizeof(ncclBarrierSession_C) > 0,
              "ncclBarrierSession_C must have non-zero size");
static_assert(alignof(ncclBarrierSession_C) >= 1,
              "ncclBarrierSession_C must have valid alignment");

/* -----------------------------------------------------------------------
 * Error checking — fail the current GTest assertion rather than exit(),
 * so a HIP error in one case does not abort the whole binary.
 * ---------------------------------------------------------------------- */
#define HIP_ASSERT(stmt)                                                     \
  do {                                                                       \
    hipError_t _e = (stmt);                                                  \
    ASSERT_EQ(_e, hipSuccess)                                                \
        << "HIP error " << (int)_e << " (" << hipGetErrorName(_e) << ") at " \
        << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(_e);     \
  } while (0)

/* -----------------------------------------------------------------------
 * Shared device-side output struct
 * ---------------------------------------------------------------------- */
struct CoopResult { int size, rank, num_threads; };

/* =====================================================================
 * Test fixture — selects a device once and caches its warp size.
 * ==================================================================== */
class IRDeviceTest : public ::testing::Test {
 protected:
  static int warpSize_;

  static void SetUpTestSuite() {
    int nDev = 0;
    ASSERT_EQ(hipGetDeviceCount(&nDev), hipSuccess);
    ASSERT_GT(nDev, 0) << "No HIP devices found";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    warpSize_ = prop.warpSize;
    std::printf("[IR_test] device 0: %s  warpSize=%d\n", prop.name, warpSize_);
  }
};
int IRDeviceTest::warpSize_ = 0;

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

TEST_F(IRDeviceTest, A_GetPeerPointerTeam) {
  constexpr int N = (int)(sizeof kPeerCases / sizeof kPeerCases[0]);
  const uintptr_t base = 0x100000000ull;

  PeerCase* d_cases = nullptr; void** d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_cases, sizeof kPeerCases));
  HIP_ASSERT(hipMalloc(&d_out,   sizeof(void*) * N));
  HIP_ASSERT(hipMemcpy(d_cases, kPeerCases, sizeof kPeerCases,
                       hipMemcpyHostToDevice));

  k_peer_ptr<<<1, N>>>((char*)base, d_cases, N, d_out);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<void*> got(N);
  HIP_ASSERT(hipMemcpy(got.data(), d_out, sizeof(void*) * N,
                       hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d_cases));
  HIP_ASSERT(hipFree(d_out));

  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE("peer case " + std::to_string(i));
    EXPECT_EQ((uintptr_t)got[i], host_peer_ptr(base, kPeerCases[i]));
  }
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
  /* Data-only kernel (size/rank accessors, no sync) -> no barrier scratch needed. */
  ncclCoopAny coop; ncclCoopAnyInitWarpSpan(&coop, warp0, nWarps, id, /*barrierLds=*/nullptr);
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
__global__ void k_sync_warpspan(int* out, int warp0, int nWarps, int id){
  /* Kernel owns the named-barrier LDS and hands it to the bitcode thunk, so the
   * backend sizes it into THIS kernel's group segment (see coop.h rationale). */
  __shared__ ncclCoopNamedBarrierSlot bar[ncclCoopNamedBarrierSlots];
  ncclCoopNamedBarrierInit(bar);
  ncclCoopAny c; ncclCoopAnyInitWarpSpan(&c, warp0, nWarps, id, bar);
  ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1;
}
__global__ void k_sync_cta     (int* out)                               { ncclCoopAny c; ncclCoopAnyInitCta     (&c);                ncclCoopSync(&c); int t=blockIdx.x*blockDim.x+threadIdx.x; out[t]=1; }

/* =====================================================================
 * Helpers: launch-check-free pattern
 * ==================================================================== */

/* Run a coop_r kernel and return the result array. */
template<typename Kern, typename... Args>
static std::vector<CoopResult>
launch_coop_r(Kern kern, int N, Args... args) {
  CoopResult* d = nullptr;
  std::vector<CoopResult> h(N);
  if (hipMalloc(&d, sizeof(CoopResult) * N) != hipSuccess) return h;
  kern<<<1, N>>>(d, args...);
  (void)hipGetLastError(); (void)hipDeviceSynchronize();
  (void)hipMemcpy(h.data(), d, sizeof(CoopResult) * N, hipMemcpyDeviceToHost);
  (void)hipFree(d);
  return h;
}

/* Run a sync kernel and verify every slot is 1. */
template<typename Kern, typename... Args>
static bool launch_sync(Kern kern, int N, Args... args) {
  int* d = nullptr;
  if (hipMalloc(&d, sizeof(int) * N) != hipSuccess) return false;
  (void)hipMemset(d, 0, sizeof(int) * N);
  kern<<<1, N>>>(d, args...);
  (void)hipGetLastError(); (void)hipDeviceSynchronize();
  std::vector<int> h(N);
  (void)hipMemcpy(h.data(), d, sizeof(int) * N, hipMemcpyDeviceToHost);
  (void)hipFree(d);
  for (int i = 0; i < N; ++i) if (h[i] != 1) return false;
  return true;
}

/* Verify size, num_threads (and optionally rank) in a CoopResult vector.
 * exp_rank(t) returns -1 to skip the rank check for thread t. */
template<typename ExpSize, typename ExpRank>
static void check_coop(const std::vector<CoopResult>& h,
                       ExpSize exp_size, ExpRank exp_rank) {
  const int N = (int)h.size();
  for (int t = 0; t < N; ++t) {
    SCOPED_TRACE("tid=" + std::to_string(t));
    EXPECT_EQ(h[t].size, exp_size(t));
    EXPECT_EQ(h[t].num_threads, h[t].size);
    const int er = exp_rank(t);
    if (er >= 0) EXPECT_EQ(h[t].rank, er);
  }
}

/* Return in-group lane rank for thread t, or -1 to skip rank check. */
static int lane_rank_in_mask(uint64_t mask, int t) {
  if (t < 0 || !((mask >> t) & 1ull)) return -1;
  return __builtin_popcountll(mask & ((1ull << t) - 1ull));
}

/* =====================================================================
 * [B1] ncclCoopAnyInitThread
 * ==================================================================== */
TEST_F(IRDeviceTest, B1_CoopInitThread) {
  /* Launch one warp so we test every lane position.
   * Thread coop: every thread is its own singleton group. */
  auto h = launch_coop_r(k_coop_thread_r, warpSize_);
  check_coop(h,
             [](int)  { return 1; },   /* exp_size */
             [](int)  { return 0; });  /* exp_rank */
}

/* =====================================================================
 * [B2] ncclCoopAnyInitWarp
 * ==================================================================== */
TEST_F(IRDeviceTest, B2_CoopInitWarp) {
  auto h = launch_coop_r(k_coop_warp_r, warpSize_);
  const int ws = warpSize_;
  check_coop(h,
             [ws](int)  { return ws; },
             [ws](int t){ return t % ws; });
}

/* =====================================================================
 * [B3] ncclCoopAnyInitLanes — mask variants (uint64_t lane_mask)
 * ==================================================================== */
TEST_F(IRDeviceTest, B3a_CoopInitLanes_FullMask) {
  const uint64_t mask = ~0ull;
  const int      sz   = warpSize_;
  const int      ws   = warpSize_;
  auto h = launch_coop_r(k_coop_lanes_r, warpSize_, mask);
  check_coop(h,
             [sz](int)  { return sz; },
             [ws, mask](int t) {
               /* nccl::utility::lanemask_lt() is still 32-bit on AMD wave64,
                * so thread_rank() for lanes 32–63 is not lane index today. */
               if (ws >= 64 && t >= 32) return -1;
               return lane_rank_in_mask(mask, t);
             });
}

TEST_F(IRDeviceTest, B3b_CoopInitLanes_SparseMask) {
  const uint64_t mask = 0x00000015ull; /* bits 0, 2, 4 */
  const int      sz   = __builtin_popcountll(mask);
  auto h = launch_coop_r(k_coop_lanes_r, warpSize_, mask);
  check_coop(h,
             [sz](int)  { return sz; },
             [mask](int t) { return lane_rank_in_mask(mask, t); });
}

TEST_F(IRDeviceTest, B3c_CoopInitLanes_SingleBitMask) {
  const uint64_t mask = 0x00000001ull;
  auto h = launch_coop_r(k_coop_lanes_r, warpSize_, mask);
  check_coop(h,
             [](int)  { return 1; },
             [](int t){ return (t == 0) ? 0 : -1; });
}

TEST_F(IRDeviceTest, B3d_CoopInitLanes_UpperLaneMask) {
  if (warpSize_ < 64) GTEST_SKIP() << "requires a 64-lane warp";
  const uint64_t mask = 1ull << 63;
  auto h = launch_coop_r(k_coop_lanes_r, warpSize_, mask);
  check_coop(h,
             [](int)  { return 1; },
             [](int t){ return (t == 63) ? 0 : -1; });
}

TEST_F(IRDeviceTest, B3e_CoopInitLanes_IgnoresNonexistentLanes) {
  const uint64_t mask = 0xffffffff00000000ull;
  const int      sz   = warpSize_ > 32 ? 32 : 0;
  auto h = launch_coop_r(k_coop_lanes_r, warpSize_, mask);
  check_coop(h,
             [sz](int) { return sz; },
             [](int)   { return -1; });
}

/* ncclCoopTile::laneMask() is not reached by the ncclCoopAny tests above.
 * AMD shims discard the mask today; this pins the value the tile returns. */
__global__ void k_coop_tile4_lane_mask(uint64_t* out) {
  ncclCoopTile<4> tile;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = static_cast<uint64_t>(tile.laneMask());
}

TEST_F(IRDeviceTest, B3f_CoopTileLaneMask_AlignedGroupOfFour) {
  const int n = warpSize_;
  uint64_t* d = nullptr;
  HIP_ASSERT(hipMalloc(&d, sizeof(uint64_t) * n));
  k_coop_tile4_lane_mask<<<1, n>>>(d);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<uint64_t> got(n);
  HIP_ASSERT(hipMemcpy(got.data(), d, sizeof(uint64_t) * n, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d));

  const uint64_t window = ~uint64_t(0) >> (64 - n);
  for (int lane = 0; lane < n; ++lane) {
    const uint64_t expect = (window >> (n - 4)) << (lane & -4);
    EXPECT_EQ(got[lane], expect) << "lane " << lane;
  }
}

/* =====================================================================
 * [B4] ncclCoopAnyInitWarpSpan — 1-warp and 2-warp spans
 * rank = threadIdx.x - WARP_SIZE * warp0  (WARP_SIZE == device warpSize)
 * ==================================================================== */
TEST_F(IRDeviceTest, B4a_CoopInitWarpSpan_OneWarp) {
  const int ws = warpSize_;
  auto h = launch_coop_r(k_coop_warpspan_r, warpSize_,
                         /*warp0=*/0, /*nWarps=*/1, /*id=*/0);
  check_coop(h,
             [ws](int)  { return ws; },
             [](int t)  { return t; });
}

TEST_F(IRDeviceTest, B4b_CoopInitWarpSpan_TwoWarp) {
  /* ncclCoopWarpSpan::sync() calls __syncthreads() on AMD, so ALL threads
   * in the CTA must participate. The coop_r kernel is data-only (no sync),
   * so launching 2*warpSize threads is safe. */
  const int N = 2 * warpSize_;
  auto h = launch_coop_r(k_coop_warpspan_r, N,
                         /*warp0=*/0, /*nWarps=*/2, /*id=*/0);
  check_coop(h,
             [N](int)  { return N; },
             [](int t) { return t; });
}

/* =====================================================================
 * [B5] ncclCoopAnyInitCta — two block sizes
 * ==================================================================== */
TEST_F(IRDeviceTest, B5_CoopInitCta) {
  for (int blk : {64, 128}) {
    SCOPED_TRACE("blockDim=" + std::to_string(blk));
    auto h = launch_coop_r(k_coop_cta_r, blk);
    check_coop(h,
               [blk](int) { return blk; },
               [](int t)  { return t;   });
  }
}

/* =====================================================================
 * [B6] ncclCoopSync — one test per coop type
 * ==================================================================== */
TEST_F(IRDeviceTest, B6_CoopSync) {
  /* Thread: every thread syncs its own singleton coop independently. */
  EXPECT_TRUE(launch_sync(k_sync_thread, warpSize_))
      << "ncclCoopSync (Thread) kernel did not complete";

  /* Warp: full warp participates. */
  EXPECT_TRUE(launch_sync(k_sync_warp, warpSize_))
      << "ncclCoopSync (Warp) kernel did not complete";

  /* Lanes (full mask): all lanes in the warp participate. */
  EXPECT_TRUE(launch_sync(k_sync_lanes, warpSize_, ~0ull))
      << "ncclCoopSync (Lanes full mask) kernel did not complete";

  /* WarpSpan (2 warps): __syncthreads() on AMD — all CTA threads must
   * call it, so launch exactly 2*warpSize threads. */
  EXPECT_TRUE(launch_sync(k_sync_warpspan, 2 * warpSize_,
                          /*warp0=*/0, /*nWarps=*/2, /*id=*/0))
      << "ncclCoopSync (WarpSpan 2-warp) kernel did not complete";

  /* CTA: standard __syncthreads(). */
  EXPECT_TRUE(launch_sync(k_sync_cta, 128))
      << "ncclCoopSync (Cta blk=128) kernel did not complete";
}

/* =====================================================================
 * [B7] ncclLsaBarrierSession — structural + skip runtime
 * ==================================================================== */
TEST_F(IRDeviceTest, B7a_LsaBarrierSessionStructural) {
  /* sizeof / alignment checks are compile-time (static_assert at top of
   * file); re-assert at runtime so the case shows up in the summary. */
  EXPECT_GT(sizeof(ncclLsaBarrierSession_C), 0u);
  EXPECT_GE(alignof(ncclLsaBarrierSession_C), 1u);
}

TEST_F(IRDeviceTest, B7b_LsaBarrierSessionRuntime) {
  /* Runtime calls require a live ncclDevComm with a mapped resource buffer;
   * without it the constructor immediately dereferences
   * ncclGetResourceBufferLocalPointer() → segfault. */
  GTEST_SKIP() << "ncclLsaBarrierSession{Init,Arrive,Wait,Sync} require a "
                  "live ncclDevComm (not set up here)";
}

/* =====================================================================
 * [C] Bucket-C symbol link/export coverage
 *
 * The structural cases (C1a/C2a) only touch the session *structs*, and the
 * runtime cases (C1b/C2b) are SKIPPED because they need a live ncclDevComm +
 * GIN net. To still cover the four exported *functions* themselves, this
 * kernel takes the device address of each thunk and stores it. Storing the
 * pointer value (not a folded != nullptr) forces the consumer -> bitcode
 * relocation for every symbol, so a missing/renamed/internalized export makes
 * the device link fail or yields a null address here — without invoking the
 * functions (which would require live resources).
 * ==================================================================== */
__global__ void k_bucketC_symbol_addrs(uint64_t* out) {
  out[0] = (uint64_t)(uintptr_t)(void*)&ncclGinBarrierSessionInit;
  out[1] = (uint64_t)(uintptr_t)(void*)&ncclGinBarrierSessionSync;
  out[2] = (uint64_t)(uintptr_t)(void*)&ncclBarrierSessionInit;
  out[3] = (uint64_t)(uintptr_t)(void*)&ncclBarrierSessionSync;
}

TEST_F(IRDeviceTest, C0_BucketCSymbolsLinkable) {
  constexpr int N = 4;
  uint64_t* d = nullptr;
  HIP_ASSERT(hipMalloc(&d, sizeof(uint64_t) * N));
  HIP_ASSERT(hipMemset(d, 0, sizeof(uint64_t) * N));
  k_bucketC_symbol_addrs<<<1, 1>>>(d);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  uint64_t got[N] = {};
  HIP_ASSERT(hipMemcpy(got, d, sizeof got, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d));

  const char* names[N] = {
      "ncclGinBarrierSessionInit", "ncclGinBarrierSessionSync",
      "ncclBarrierSessionInit",    "ncclBarrierSessionSync"};
  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE(names[i]);
    EXPECT_NE(got[i], 0ull) << names[i] << " did not resolve from the bitcode";
  }
}

/* =====================================================================
 * [C1] ncclGinBarrierSession — structural + skip runtime
 * ==================================================================== */
TEST_F(IRDeviceTest, C1a_GinBarrierSessionStructural) {
  /* Mirrors B7a: the sizeof / alignof static_asserts at the top of the file
   * fire at compile time; re-assert at runtime so the case appears in the
   * summary and proves bucket [C]'s ncclGinBarrierSession_C is instantiated. */
  EXPECT_GT(sizeof(ncclGinBarrierSession_C), 0u);
  EXPECT_GE(alignof(ncclGinBarrierSession_C), 1u);
}

TEST_F(IRDeviceTest, C1b_GinBarrierSessionRuntime) {
  /* ncclGinBarrierSessionInit takes an ncclGin_C, ncclTeam and
   * ncclGinBarrierHandle bound to live GPU-Initiated-Networking resources;
   * ncclGinBarrierSessionSync then issues network fences against them. Both
   * require a live ncclDevComm with an established GIN connection, which is
   * not available in this standalone device-API test. */
  GTEST_SKIP() << "ncclGinBarrierSession{Init,Sync} require a live ncclDevComm "
                  "+ GIN network resources (not set up here)";
}

/* =====================================================================
 * [C2] ncclBarrierSession (composite inner-LSA + outer-GIN) — structural
 *      + skip runtime
 * ==================================================================== */
TEST_F(IRDeviceTest, C2a_BarrierSessionStructural) {
  EXPECT_GT(sizeof(ncclBarrierSession_C), 0u);
  EXPECT_GE(alignof(ncclBarrierSession_C), 1u);
}

TEST_F(IRDeviceTest, C2b_BarrierSessionRuntime) {
  /* The composite barrier wires together an inner LSA barrier and an outer
   * GIN barrier (inner/outer teams, an ncclLsaBarrierHandle, an
   * ncclGinBarrierHandle and an ncclGin_C), so it needs everything B7b and
   * C1b need at once: a live ncclDevComm plus GIN network resources. */
  GTEST_SKIP() << "ncclBarrierSession{Init,Sync} require a live ncclDevComm "
                  "+ GIN network resources (not set up here)";
}

/* =====================================================================
 * [D] ReduceCopy typed C-ABI thunks (upstream NCCL bbb1375fa382,
 *     "bindings/ir: expose typed ReduceCopy APIs").
 *
 * The expected thunk inventory is spelled out HERE, independently of
 * bindings/ir/util.h, so that dropping a type from util.h, a declaration
 * from nccl_device_wrapper.h, or a name from the opt internalize list in
 * CMakeLists.txt / Makefile all surface as a failure instead of silently
 * shrinking what is tested.
 *
 * FP8 (F8E4M3/F8E5M2) is intentionally absent: RCCL's device ReduceSum/Copy
 * has no fp8 support, so the HIP bitcode exports no FP8 thunks (test/ir-device
 * checks that absence at the symbol level).
 * ==================================================================== */
#define IR_RC_ALL_TYPES(X) \
  X(I8, int8_t) X(U8, uint8_t) X(I32, int32_t) X(U32, uint32_t) \
  X(I64, int64_t) X(U64, uint64_t) X(F16, half) X(F32, float) \
  X(F64, double) X(BF16, __hip_bfloat16)

#define IR_RC_MULTIMEM_TYPES(X) \
  X(I32, int32_t) X(U32, uint32_t) X(I64, int64_t) X(U64, uint64_t) \
  X(F16, half) X(F32, float) X(F64, double) X(BF16, __hip_bfloat16)

/* ---------------------------------------------------------------------
 * [D0] Link coverage for every ReduceCopy thunk (64 symbols).
 *
 * Same technique as C0: store each thunk's device address so the consumer
 * -> bitcode relocation is forced for every symbol. A missing, renamed or
 * internalized export fails the device link or yields a null address. The
 * Multimem* thunks cannot run on AMD (no multimem hardware; they assert at
 * runtime), so link coverage is the only coverage they get.
 * ------------------------------------------------------------------- */
#define IR_RC_ADDR(op, S) (uint64_t)(uintptr_t)(void*)&op##_##S
#define IR_RC_NAME(op, S) #op "_" #S

#define IR_RC_ALL_ADDRS(S, T)                       \
  IR_RC_ADDR(ncclLsaReduceSum, S),                  \
  IR_RC_ADDR(ncclLsaCopy, S),                       \
  IR_RC_ADDR(ncclLsaReduceSumCopy, S),              \
  IR_RC_ADDR(ncclLocalReduceSumCopy, S),
#define IR_RC_MM_ADDRS(S, T)                        \
  IR_RC_ADDR(ncclMultimemReduceSum, S),             \
  IR_RC_ADDR(ncclMultimemCopy, S),                  \
  IR_RC_ADDR(ncclMultimemReduceSumCopy, S),
#define IR_RC_ALL_NAMES(S, T)                       \
  IR_RC_NAME(ncclLsaReduceSum, S),                  \
  IR_RC_NAME(ncclLsaCopy, S),                       \
  IR_RC_NAME(ncclLsaReduceSumCopy, S),              \
  IR_RC_NAME(ncclLocalReduceSumCopy, S),
#define IR_RC_MM_NAMES(S, T)                        \
  IR_RC_NAME(ncclMultimemReduceSum, S),             \
  IR_RC_NAME(ncclMultimemCopy, S),                  \
  IR_RC_NAME(ncclMultimemReduceSumCopy, S),

static const char* const kReduceCopyThunkNames[] = {
  IR_RC_ALL_TYPES(IR_RC_ALL_NAMES)
  IR_RC_MULTIMEM_TYPES(IR_RC_MM_NAMES)
};
static constexpr int kNumReduceCopyThunks =
    (int)(sizeof(kReduceCopyThunkNames) / sizeof(kReduceCopyThunkNames[0]));
static_assert(kNumReduceCopyThunks == 4 * 10 + 3 * 8,
              "ReduceCopy thunk inventory: 4 ops x 10 types + 3 multimem ops x 8 types");

__global__ void k_reducecopy_symbol_addrs(uint64_t* out) {
  const uint64_t addrs[] = {
    IR_RC_ALL_TYPES(IR_RC_ALL_ADDRS)
    IR_RC_MULTIMEM_TYPES(IR_RC_MM_ADDRS)
  };
  for (int i = 0; i < kNumReduceCopyThunks; ++i) out[i] = addrs[i];
}

TEST_F(IRDeviceTest, D0_ReduceCopySymbolsLinkable) {
  const int N = kNumReduceCopyThunks;
  uint64_t* d = nullptr;
  HIP_ASSERT(hipMalloc(&d, sizeof(uint64_t) * N));
  HIP_ASSERT(hipMemset(d, 0, sizeof(uint64_t) * N));
  k_reducecopy_symbol_addrs<<<1, 1>>>(d);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint64_t> got(N);
  HIP_ASSERT(hipMemcpy(got.data(), d, sizeof(uint64_t) * N, hipMemcpyDeviceToHost));
  HIP_ASSERT(hipFree(d));

  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE(kReduceCopyThunkNames[i]);
    EXPECT_NE(got[i], 0ull) << kReduceCopyThunkNames[i]
                            << " did not resolve from the bitcode";
    for (int j = 0; j < i; ++j)
      EXPECT_NE(got[i], got[j]) << kReduceCopyThunkNames[i] << " aliases "
                                << kReduceCopyThunkNames[j];
  }
}

/* ---------------------------------------------------------------------
 * [D1] ncclLocalReduceSumCopy_<T> — functional, all 10 types.
 *
 * The Local variant takes plain device pointers (nSrc strided sources ->
 * nDst strided destinations, each destination receiving the full sum), so
 * it runs in this single-process binary with no communicator. Exercises:
 *   - per-type dispatch: each _<T> thunk must instantiate the template for
 *     exactly that element type (a width/float-vs-int mixup corrupts values
 *     or the guard bytes);
 *   - the 16B / 4B / scalar vectorization tiers and scalar tail
 *     (count sweep), plus a misaligned base pointer;
 *   - nSrc/nDst fan-in/fan-out shapes, including pure copy (1 -> 1);
 *   - cta / warp / thread ncclCoopAny flavors passed by value through the
 *     C ABI;
 *   - no out-of-bounds writes (sentinel-filled gaps between and after the
 *     destination slices must be untouched).
 * Values are small integers so every type, including F16/BF16, sums exactly.
 * ------------------------------------------------------------------- */
namespace {

template <typename T> struct RcHost {
  static T from(int v) { return static_cast<T>(v); }
  static double to(T v) { return static_cast<double>(v); }
};
template <> struct RcHost<half> {
  static half from(int v) { return __float2half(static_cast<float>(v)); }
  static double to(half v) { return static_cast<double>(__half2float(v)); }
};
template <> struct RcHost<__hip_bfloat16> {
  static __hip_bfloat16 from(int v) { return __float2bfloat16(static_cast<float>(v)); }
  static double to(__hip_bfloat16 v) { return static_cast<double>(__bfloat162float(v)); }
};

/* Seed value for source s, element i. Signed types get negative values too
 * so sign handling is exercised; magnitudes stay small enough that an 8-way
 * sum is exact in int8 and in bf16 (8-bit mantissa). */
template <typename T> int rcSeed(int s, size_t i) {
  int v = (s + 1) + static_cast<int>(i % 7);
  return std::is_signed<T>::value || !std::is_integral<T>::value ? v - 8 : v;
}

enum RcCoopKind { kRcCta = 0, kRcWarp = 1, kRcThread = 2 };

constexpr unsigned char kRcSentinel = 0xA5;

/* Measured on gfx950 (ROCm 7.2): integer thunks fit in 16 KiB, F16 needs
 * 16-20 KiB. 32 KiB leaves ~1.6x headroom, so a large stack-usage regression
 * in the thunks still fails this suite. */
constexpr size_t kRcDefaultStackBytes = 32768;

}  // namespace

#define IR_RC_DEFINE_LOCAL_TRAITS(S, TY)                                      \
  struct RcLocal_##S {                                                        \
    using T = TY;                                                             \
    static constexpr const char* name = #S;                                   \
    static __device__ void call(ncclCoopAny c, int nSrc, T* src, size_t sd,   \
                                int nDst, T* dst, size_t dd, size_t n) {      \
      ncclLocalReduceSumCopy_##S(c, nSrc, src, sd, nDst, dst, dd, n);         \
    }                                                                         \
  };
IR_RC_ALL_TYPES(IR_RC_DEFINE_LOCAL_TRAITS)

template <typename Traits>
__global__ void k_local_reduce_sum_copy(int coopKind, int nSrc,
                                        typename Traits::T* src, size_t srcDispl,
                                        int nDst, typename Traits::T* dst,
                                        size_t dstDispl, size_t count) {
  ncclCoopAny coop;
  if (coopKind == kRcCta)       ncclCoopAnyInitCta(&coop);
  else if (coopKind == kRcWarp) ncclCoopAnyInitWarp(&coop);
  else                          ncclCoopAnyInitThread(&coop);
  Traits::call(coop, nSrc, src, srcDispl, nDst, dst, dstDispl, count);
}

/* Per-thread stack the ReduceCopy thunks need.
 *
 * Kernels that call these bitcode thunks are compiled with a *dynamic* stack:
 * the thunks dispatch through ncclCoopAny's function-pointer table, so the
 * backend cannot bound the call-graph stack statically. HIP then caps the
 * per-thread stack at hipLimitStackSize, whose default (1 KiB) is too small
 * for the ReduceCopy thunks — at the default they fault with
 * HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION even for count == 0. Callers of
 * the thunks (Triton/MLIR consumers of librccl_device.bc included) must raise
 * the limit; the tests do the same. Override with IR_RC_STACK_BYTES. */
static size_t rcStackBytes() {
  if (const char* e = std::getenv("IR_RC_STACK_BYTES")) return std::strtoull(e, nullptr, 0);
  return kRcDefaultStackBytes;
}

template <typename Traits>
class IRReduceCopyLocalTest : public IRDeviceTest {
 protected:
  using T = typename Traits::T;

  static void SetUpTestSuite() {
    IRDeviceTest::SetUpTestSuite();
    ASSERT_EQ(hipDeviceSetLimit(hipLimitStackSize, rcStackBytes()), hipSuccess);
  }

  /* One launch + full verification. `misalign` shifts both base pointers by
   * one element so the 16B fast path cannot be taken from element 0. */
  void runCase(int coopKind, int nThreads, int nSrc, int nDst, size_t count,
               bool misalign) {
    SCOPED_TRACE(std::string(Traits::name) + " coop=" + std::to_string(coopKind) +
                 " nSrc=" + std::to_string(nSrc) + " nDst=" + std::to_string(nDst) +
                 " count=" + std::to_string(count) +
                 " misalign=" + std::to_string(misalign));
    const size_t shift = misalign ? 1 : 0;
    /* Odd padding so consecutive slices start at different alignments. */
    const size_t srcDispl = count + 5;
    const size_t dstDispl = count + 3;
    const size_t srcElts = shift + nSrc * srcDispl;
    const size_t dstElts = shift + nDst * dstDispl + 16;  // trailing guard

    std::vector<T> hSrc(srcElts, RcHost<T>::from(0));
    for (int s = 0; s < nSrc; ++s)
      for (size_t i = 0; i < count; ++i)
        hSrc[shift + s * srcDispl + i] = RcHost<T>::from(rcSeed<T>(s, i));

    T* dSrc = nullptr;
    T* dDst = nullptr;
    HIP_ASSERT(hipMalloc(&dSrc, srcElts * sizeof(T)));
    HIP_ASSERT(hipMalloc(&dDst, dstElts * sizeof(T)));
    HIP_ASSERT(hipMemcpy(dSrc, hSrc.data(), srcElts * sizeof(T), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemset(dDst, kRcSentinel, dstElts * sizeof(T)));

    k_local_reduce_sum_copy<Traits><<<1, nThreads>>>(
        coopKind, nSrc, dSrc + shift, srcDispl, nDst, dDst + shift, dstDispl, count);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());

    std::vector<T> hDst(dstElts);
    HIP_ASSERT(hipMemcpy(hDst.data(), dDst, dstElts * sizeof(T), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipFree(dSrc));
    HIP_ASSERT(hipFree(dDst));

    /* Payload: every destination holds the full sum. */
    size_t bad = 0;
    for (int d = 0; d < nDst && bad < 8; ++d) {
      for (size_t i = 0; i < count && bad < 8; ++i) {
        int sum = 0;
        for (int s = 0; s < nSrc; ++s) sum += rcSeed<T>(s, i);
        const double want = RcHost<T>::to(RcHost<T>::from(sum));
        const double got = RcHost<T>::to(hDst[shift + d * dstDispl + i]);
        if (got != want) {
          ADD_FAILURE() << "dst=" << d << " elt=" << i << " got=" << got
                        << " want=" << want;
          ++bad;
        }
      }
    }
    /* Guards: every byte outside the written slices keeps the sentinel. */
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(hDst.data());
    for (size_t e = 0; e < dstElts && bad < 8; ++e) {
      bool written = false;
      if (e >= shift) {
        const size_t rel = e - shift;
        written = rel / dstDispl < (size_t)nDst && rel % dstDispl < count;
      }
      if (written) continue;
      for (size_t b = 0; b < sizeof(T); ++b) {
        if (bytes[e * sizeof(T) + b] != kRcSentinel) {
          ADD_FAILURE() << "out-of-bounds write at element " << e;
          ++bad;
          break;
        }
      }
    }
  }
};

/* Spelled out (not generated) so the list is an independent oracle. */
using IRReduceCopyLocalTypes =
    ::testing::Types<RcLocal_I8, RcLocal_U8, RcLocal_I32, RcLocal_U32,
                     RcLocal_I64, RcLocal_U64, RcLocal_F16, RcLocal_F32,
                     RcLocal_F64, RcLocal_BF16>;

struct IRReduceCopyLocalNames {
  template <typename Traits> static std::string GetName(int) { return Traits::name; }
};
TYPED_TEST_SUITE(IRReduceCopyLocalTest, IRReduceCopyLocalTypes, IRReduceCopyLocalNames);

/* Counts straddling the 16B / 4B / scalar tiers and unroll boundaries for
 * 1..8-byte elements, plus a multi-MB case for the bulk loop. */
static const size_t kRcCounts[] = {0, 1, 3, 15, 16, 17, 63, 64, 65, 1023,
                                   1024, 1025, 4099, 65543, (1u << 20) + 3};

TYPED_TEST(IRReduceCopyLocalTest, D1a_Cta_CountSweep) {
  for (size_t count : kRcCounts) {
    this->runCase(kRcCta, 256, /*nSrc=*/4, /*nDst=*/3, count, /*misalign=*/false);
    if (::testing::Test::HasFailure()) return;
  }
}

TYPED_TEST(IRReduceCopyLocalTest, D1b_Cta_Misaligned) {
  for (size_t count : {size_t{1}, size_t{17}, size_t{1025}, size_t{65543}}) {
    this->runCase(kRcCta, 256, /*nSrc=*/3, /*nDst=*/2, count, /*misalign=*/true);
    if (::testing::Test::HasFailure()) return;
  }
}

TYPED_TEST(IRReduceCopyLocalTest, D1c_Cta_FanInFanOutShapes) {
  /* 1->1 is a pure copy; 8->1 a pure reduce; 1->4 a pure broadcast. */
  const int shapes[][2] = {{1, 1}, {2, 1}, {8, 1}, {1, 4}, {5, 5}};
  for (auto& sh : shapes) {
    this->runCase(kRcCta, 256, sh[0], sh[1], /*count=*/4099, /*misalign=*/false);
    if (::testing::Test::HasFailure()) return;
  }
}

TYPED_TEST(IRReduceCopyLocalTest, D1d_WarpAndThreadCoop) {
  /* Launch exactly one warp: with warp coop the single warp owns the whole
   * range; with thread coop every lane redundantly writes identical values,
   * which must still produce the exact result. */
  const int nThreads = IRDeviceTest::warpSize_;
  for (size_t count : {size_t{1}, size_t{65}, size_t{1025}}) {
    this->runCase(kRcWarp, nThreads, /*nSrc=*/4, /*nDst=*/2, count, /*misalign=*/false);
    if (::testing::Test::HasFailure()) return;
    this->runCase(kRcThread, nThreads, /*nSrc=*/4, /*nDst=*/2, count, /*misalign=*/false);
    if (::testing::Test::HasFailure()) return;
  }
}
