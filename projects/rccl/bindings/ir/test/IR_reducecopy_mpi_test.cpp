/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * IR_reducecopy_mpi_test.cpp — multi-rank functional test for the typed LSA
 * ReduceCopy thunks exported by librccl_device.bc (upstream NCCL
 * bbb1375fa382, "bindings/ir: expose typed ReduceCopy APIs"):
 *
 *   ncclLsaReduceSum_<T>      N symmetric sources (LSA team) -> 1 local dst
 *   ncclLsaCopy_<T>           1 local source -> N symmetric dsts (LSA team)
 *   ncclLsaReduceSumCopy_<T>  N symmetric sources -> N symmetric dsts
 *
 * for all 10 HIP types (I8 U8 I32 U32 I64 U64 F16 F32 F64 BF16).
 *
 * The kernels call ONLY the exported C thunks for the operation under test —
 * no nccl_device templates, no device-side barriers. Cross-rank ordering is
 * provided by the host (hipDeviceSynchronize + MPI_Barrier around every
 * launch), so a failure here points at the thunk/bitcode, not at a barrier.
 *
 * For Copy and ReduceSumCopy a single, rotating "origin" rank issues the op
 * and every rank verifies its own destination: this proves the fan-out to
 * each peer individually (with all ranks writing, one rank's correct write
 * could mask another's missing one). Destination windows are sentinel-filled
 * so out-of-bounds writes (e.g. a thunk bound to the wrong element width) are
 * caught, and a non-zero window offset checks offset handling.
 *
 * The local ncclLocalReduceSumCopy_<T> thunk and link coverage for all 64
 * thunks (incl. Multimem*, which cannot run on AMD) live in IR_test.cpp.
 *
 * Run (single node, >=2 GPUs, symmetric memory):
 *   NCCL_CUMEM_ENABLE=1 mpirun -np 2 ./IR_reducecopy_mpi_test.exe
 *
 * Cases GTEST_SKIP (rather than fail) when a prerequisite is absent; the skip
 * decision is agreed across ranks so no rank is left waiting in a collective.
 ************************************************************************/

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device_wrapper.h>

extern int ncclCuMemRuntimeSupported();

/* -----------------------------------------------------------------------
 * Error-checking macros
 * --------------------------------------------------------------------- */
#define HIP_ASSERT(stmt)                                                     \
  do {                                                                       \
    hipError_t _e = (stmt);                                                  \
    ASSERT_EQ(_e, hipSuccess)                                                \
        << "HIP error " << (int)_e << " (" << hipGetErrorName(_e) << ") at " \
        << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(_e);     \
  } while (0)

#define NCCL_ASSERT(stmt)                                                    \
  do {                                                                       \
    ncclResult_t _r = (stmt);                                                \
    ASSERT_EQ(_r, ncclSuccess)                                               \
        << "NCCL error " << (int)_r << " (" << ncclGetErrorString(_r)        \
        << ") at " << __FILE__ << ":" << __LINE__;                           \
  } while (0)

namespace {

/* Per-thread stack for kernels calling the ReduceCopy thunks. They dispatch
 * through ncclCoopAny's function-pointer table, so the kernel gets a dynamic
 * stack and HIP caps it at hipLimitStackSize; the 1 KiB default faults.
 * See kRcDefaultStackBytes in IR_test.cpp for the measured requirement. */
constexpr size_t kRcDefaultStackBytes = 32768;
size_t rcStackBytes() {
  if (const char* e = std::getenv("IR_RC_STACK_BYTES")) return std::strtoull(e, nullptr, 0);
  return kRcDefaultStackBytes;
}

constexpr int kThreads = 256;
constexpr unsigned char kSentinel = 0xA5;
/* Window offset (in elements) used by every case: odd, so the symmetric
 * pointer is never 16B-aligned for any element size > 1 byte. */
constexpr size_t kOffElts = 3;
constexpr size_t kGuardElts = 16;
/* Count sweep: scalar-only, around the 16B/4B vector tiers, and a bulk case. */
const size_t kCounts[] = {1, 17, 1025, 65543};
constexpr size_t kMaxCount = 65543;

/* Host <-> T conversions for exact small-integer payloads. */
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

/* Seed for rank r, element i. Signed and float types include negatives; with
 * <=8 ranks every sum stays exact in int8 and bf16. */
template <typename T> int rcSeed(int r, size_t i) {
  int v = (r + 1) + static_cast<int>(i % 7);
  return std::is_signed<T>::value || !std::is_integral<T>::value ? v - 8 : v;
}

/* -----------------------------------------------------------------------
 * Shared communicator: created once per process (comm + devComm init are
 * expensive and every typed case can reuse them).
 * --------------------------------------------------------------------- */
struct RcWorld {
  int worldRank = -1, worldSize = -1;
  ncclComm_t comm = nullptr;
  ncclDevComm devComm{};
  bool devCommValid = false;
  std::string skipReason;  // non-empty => every case skips (agreed by all ranks)
};
RcWorld g;

std::string localSkipReason(int nodeSize, int nDev) {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || std::strtoll(cumem, nullptr, 0) == 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE=1)";
  if (!ncclCuMemRuntimeSupported())
    return "Symmetric memory not supported on this platform (cuMem requires dma-buf/P2PDMA)";
  if (g.worldSize < 2) return "Requires >=2 ranks";
  if (nodeSize != g.worldSize)
    return "Requires a single-node run (all ranks in one LSA team)";
  if (nDev < nodeSize)
    return "Requires one GPU per rank (" + std::to_string(nDev) + " GPUs for " +
           std::to_string(nodeSize) + " ranks)";
  return "";
}

/* Agree on a skip across ranks: if any rank must skip, all do. */
std::string agreeSkip(const std::string& local) {
  int mine = local.empty() ? 0 : 1, any = 0;
  MPI_Allreduce(&mine, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (!any) return "";
  return local.empty() ? "Skipped because another rank cannot run this test" : local;
}

class RcEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &g.worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &g.worldSize);

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, nodeSize = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int nDev = 0;
    if (hipGetDeviceCount(&nDev) != hipSuccess) nDev = 0;
    g.skipReason = agreeSkip(localSkipReason(nodeSize, nDev));
    if (!g.skipReason.empty()) return;

    std::string err;
    if (hipSetDevice(localRank) != hipSuccess) err = "hipSetDevice failed";
    if (err.empty() && hipDeviceSetLimit(hipLimitStackSize, rcStackBytes()) != hipSuccess)
      err = "hipDeviceSetLimit(hipLimitStackSize) failed";

    ncclUniqueId id{};
    if (g.worldRank == 0 && ncclGetUniqueId(&id) != ncclSuccess) err = "ncclGetUniqueId failed";
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);
    if (err.empty() && ncclCommInitRank(&g.comm, g.worldSize, id, g.worldRank) != ncclSuccess)
      err = "ncclCommInitRank failed";
    if (err.empty()) {
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      ncclResult_t r = ncclDevCommCreate(g.comm, &reqs, &g.devComm);
      if (r == ncclSuccess) g.devCommValid = true;
      else err = std::string("ncclDevCommCreate failed: ") + ncclGetErrorString(r);
    }
    /* Setup errors are environment problems (cuMem, topology), not thunk
     * bugs: surface them as an agreed skip with the reason. */
    g.skipReason = agreeSkip(err);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  void TearDown() override {
    if (g.devCommValid) (void)ncclDevCommDestroy(g.comm, &g.devComm);
    if (g.comm) (void)ncclCommDestroy(g.comm);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  }
};

const ::testing::Environment* const kRcEnv =
    ::testing::AddGlobalTestEnvironment(new RcEnvironment);

/* A symmetric window plus its local mapping; collective alloc/free. */
struct SymBuf {
  void* ptr = nullptr;
  ncclWindow_t win = nullptr;
  size_t bytes = 0;
  ncclResult_t alloc(size_t b) {
    bytes = b;
    ncclResult_t r = ncclMemAlloc(&ptr, bytes);
    if (r != ncclSuccess) return r;
    return ncclCommWindowRegister(g.comm, ptr, bytes, &win, NCCL_WIN_COLL_SYMMETRIC);
  }
  ~SymBuf() {
    if (win) (void)ncclCommWindowDeregister(g.comm, win);
    if (ptr) (void)ncclMemFree(ptr);
  }
};

}  // namespace

/* =====================================================================
 * Per-type traits: forward to exactly one exported thunk per op.
 * ==================================================================== */
#define IR_RC_ALL_TYPES(X) \
  X(I8, int8_t) X(U8, uint8_t) X(I32, int32_t) X(U32, uint32_t) \
  X(I64, int64_t) X(U64, uint64_t) X(F16, half) X(F32, float) \
  X(F64, double) X(BF16, __hip_bfloat16)

#define IR_RC_DEFINE_LSA_TRAITS(S, TY)                                         \
  struct RcLsa_##S {                                                           \
    using T = TY;                                                              \
    static constexpr const char* name = #S;                                    \
    static __device__ void reduceSum(ncclCoopAny c, ncclWindow_t w, size_t o,  \
                                     T* dst, size_t n, ncclTeam t) {           \
      ncclLsaReduceSum_##S(c, w, o, dst, n, t);                                \
    }                                                                          \
    static __device__ void copy(ncclCoopAny c, T* src, ncclWindow_t w,         \
                                size_t o, size_t n, ncclTeam t) {              \
      ncclLsaCopy_##S(c, src, w, o, n, t);                                     \
    }                                                                          \
    static __device__ void reduceSumCopy(ncclCoopAny c, ncclWindow_t sw,       \
                                         size_t so, ncclWindow_t dw, size_t dofs, \
                                         size_t n, ncclTeam t) {               \
      ncclLsaReduceSumCopy_##S(c, sw, so, dw, dofs, n, t);                     \
    }                                                                          \
  };
IR_RC_ALL_TYPES(IR_RC_DEFINE_LSA_TRAITS)

template <typename Tr>
__global__ void k_lsa_reduce_sum(ncclWindow_t srcWin, size_t srcOff,
                                 typename Tr::T* dst, size_t count, ncclTeam team) {
  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);
  Tr::reduceSum(coop, srcWin, srcOff, dst, count, team);
}

template <typename Tr>
__global__ void k_lsa_copy(typename Tr::T* src, ncclWindow_t dstWin, size_t dstOff,
                           size_t count, ncclTeam team) {
  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);
  Tr::copy(coop, src, dstWin, dstOff, count, team);
}

template <typename Tr>
__global__ void k_lsa_reduce_sum_copy(ncclWindow_t srcWin, size_t srcOff,
                                      ncclWindow_t dstWin, size_t dstOff,
                                      size_t count, ncclTeam team) {
  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);
  Tr::reduceSumCopy(coop, srcWin, srcOff, dstWin, dstOff, count, team);
}

/* =====================================================================
 * Typed fixture
 * ==================================================================== */
template <typename Tr>
class IRReduceCopyMPITest : public ::testing::Test {
 protected:
  using T = typename Tr::T;
  static constexpr size_t kWinElts = kOffElts + kMaxCount + kGuardElts;

  void SetUp() override {
    if (!g.skipReason.empty()) GTEST_SKIP() << g.skipReason;
    team_.nRanks = g.devComm.lsaSize;
    team_.rank = g.devComm.lsaRank;
    team_.stride = 1;
    ASSERT_EQ(team_.nRanks, g.worldSize) << "single-node run must form one LSA team";
  }

  void TearDown() override {
    /* Keep ranks in lockstep between cases even if one failed early. */
    if (g.skipReason.empty()) MPI_Barrier(MPI_COMM_WORLD);
  }

  /* Seed this rank's symmetric source window: [off, off+kMaxCount) gets
   * rcSeed(lsaRank, i); everything else is zero. */
  void seedSrcWindow(SymBuf& src) {
    std::vector<T> h(kWinElts, RcHost<T>::from(0));
    for (size_t i = 0; i < kMaxCount; ++i)
      h[kOffElts + i] = RcHost<T>::from(rcSeed<T>(team_.rank, i));
    HIP_ASSERT(hipMemcpy(src.ptr, h.data(), kWinElts * sizeof(T), hipMemcpyHostToDevice));
  }

  int expectedSum(size_t i) const {
    int sum = 0;
    for (int p = 0; p < team_.nRanks; ++p) sum += rcSeed<T>(p, i);
    return sum;
  }

  /* Verify a buffer laid out as [off][count payload][rest]: payload must match
   * want(i), every other byte must still hold the sentinel. Non-fatal so the
   * rank keeps participating in collectives. */
  template <typename Want>
  void verify(const std::vector<T>& h, size_t off, size_t count, Want want,
              const char* what) {
    int bad = 0;
    for (size_t i = 0; i < count && bad < 8; ++i) {
      const double w = RcHost<T>::to(RcHost<T>::from(want(i)));
      const double got = RcHost<T>::to(h[off + i]);
      if (got != w) {
        ADD_FAILURE() << what << " rank=" << g.worldRank << " elt=" << i
                      << " got=" << got << " want=" << w;
        ++bad;
      }
    }
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (size_t e = 0; e < h.size() && bad < 8; ++e) {
      if (e >= off && e < off + count) continue;
      for (size_t k = 0; k < sizeof(T); ++k)
        if (b[e * sizeof(T) + k] != kSentinel) {
          ADD_FAILURE() << what << " rank=" << g.worldRank
                        << " out-of-bounds write at element " << e;
          ++bad;
          break;
        }
    }
  }

  ncclTeam team_{};
};

struct IRReduceCopyMPINames {
  template <typename Tr> static std::string GetName(int) { return Tr::name; }
};
using IRReduceCopyMPITypes =
    ::testing::Types<RcLsa_I8, RcLsa_U8, RcLsa_I32, RcLsa_U32, RcLsa_I64,
                     RcLsa_U64, RcLsa_F16, RcLsa_F32, RcLsa_F64, RcLsa_BF16>;
TYPED_TEST_SUITE(IRReduceCopyMPITest, IRReduceCopyMPITypes, IRReduceCopyMPINames);

/* [E1] ncclLsaReduceSum_<T>: every rank reduces all peers' symmetric source
 * (at a non-zero window offset) into its own local destination. */
TYPED_TEST(IRReduceCopyMPITest, E1_LsaReduceSum) {
  using T = typename TestFixture::T;
  SymBuf src;
  NCCL_ASSERT(src.alloc(this->kWinElts * sizeof(T)));
  this->seedSrcWindow(src);

  const size_t dstElts = kMaxCount + kGuardElts;
  T* dDst = nullptr;
  HIP_ASSERT(hipMalloc(&dDst, dstElts * sizeof(T)));
  std::vector<T> h(dstElts);

  for (size_t count : kCounts) {
    SCOPED_TRACE(std::string(TypeParam::name) + " count=" + std::to_string(count));
    HIP_ASSERT(hipMemset(dDst, kSentinel, dstElts * sizeof(T)));
    HIP_ASSERT(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);  // all sources seeded before anyone reads
    k_lsa_reduce_sum<TypeParam><<<1, kThreads>>>(src.win, kOffElts * sizeof(T), dDst,
                                                 count, this->team_);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    HIP_ASSERT(hipMemcpy(h.data(), dDst, dstElts * sizeof(T), hipMemcpyDeviceToHost));
    this->verify(h, 0, count, [&](size_t i) { return this->expectedSum(i); },
                 "LsaReduceSum");
  }
  HIP_ASSERT(hipFree(dDst));
}

/* [E2] ncclLsaCopy_<T>: one rotating origin copies its local source into
 * every peer's symmetric destination; every rank checks its own window. */
TYPED_TEST(IRReduceCopyMPITest, E2_LsaCopy) {
  using T = typename TestFixture::T;
  SymBuf dst;
  NCCL_ASSERT(dst.alloc(this->kWinElts * sizeof(T)));

  T* dSrc = nullptr;
  HIP_ASSERT(hipMalloc(&dSrc, kMaxCount * sizeof(T)));
  std::vector<T> hSrc(kMaxCount);
  for (size_t i = 0; i < kMaxCount; ++i)
    hSrc[i] = RcHost<T>::from(rcSeed<T>(this->team_.rank, i));
  HIP_ASSERT(hipMemcpy(dSrc, hSrc.data(), kMaxCount * sizeof(T), hipMemcpyHostToDevice));
  std::vector<T> h(this->kWinElts);

  int iter = 0;
  for (size_t count : kCounts) {
    const int origin = iter++ % this->team_.nRanks;
    SCOPED_TRACE(std::string(TypeParam::name) + " count=" + std::to_string(count) +
                 " origin=" + std::to_string(origin));
    HIP_ASSERT(hipMemset(dst.ptr, kSentinel, dst.bytes));
    HIP_ASSERT(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);  // all destinations reset before the write
    if (this->team_.rank == origin) {
      k_lsa_copy<TypeParam><<<1, kThreads>>>(dSrc, dst.win, kOffElts * sizeof(T), count,
                                             this->team_);
      HIP_ASSERT(hipGetLastError());
      HIP_ASSERT(hipDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    HIP_ASSERT(hipMemcpy(h.data(), dst.ptr, dst.bytes, hipMemcpyDeviceToHost));
    this->verify(h, kOffElts, count, [&](size_t i) { return rcSeed<T>(origin, i); },
                 "LsaCopy");
  }
  HIP_ASSERT(hipFree(dSrc));
}

/* [E3] ncclLsaReduceSumCopy_<T>: one rotating origin reduces all peers'
 * sources and writes the sum to every peer's destination window. */
TYPED_TEST(IRReduceCopyMPITest, E3_LsaReduceSumCopy) {
  using T = typename TestFixture::T;
  SymBuf src, dst;
  NCCL_ASSERT(src.alloc(this->kWinElts * sizeof(T)));
  NCCL_ASSERT(dst.alloc(this->kWinElts * sizeof(T)));
  this->seedSrcWindow(src);
  std::vector<T> h(this->kWinElts);

  int iter = 0;
  for (size_t count : kCounts) {
    const int origin = iter++ % this->team_.nRanks;
    SCOPED_TRACE(std::string(TypeParam::name) + " count=" + std::to_string(count) +
                 " origin=" + std::to_string(origin));
    HIP_ASSERT(hipMemset(dst.ptr, kSentinel, dst.bytes));
    HIP_ASSERT(hipDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    if (this->team_.rank == origin) {
      k_lsa_reduce_sum_copy<TypeParam><<<1, kThreads>>>(
          src.win, kOffElts * sizeof(T), dst.win, kOffElts * sizeof(T), count, this->team_);
      HIP_ASSERT(hipGetLastError());
      HIP_ASSERT(hipDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    HIP_ASSERT(hipMemcpy(h.data(), dst.ptr, dst.bytes, hipMemcpyDeviceToHost));
    this->verify(h, kOffElts, count, [&](size_t i) { return this->expectedSum(i); },
                 "LsaReduceSumCopy");
  }
}
