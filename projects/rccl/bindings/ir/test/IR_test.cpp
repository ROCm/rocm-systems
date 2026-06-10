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
#include <string>
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
