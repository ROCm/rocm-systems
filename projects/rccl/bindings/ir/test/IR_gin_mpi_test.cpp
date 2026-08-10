/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * IR_gin_mpi_test.cpp — multi-rank functional test for the bucket [C]
 * GIN / composite barrier thunks exported by librccl_device.bc.
 *
 * Unlike IR_test.cpp (single process, structural + link coverage only), this
 * binary stands up a real 2-rank ncclComm with an established GIN connection
 * and drives the four bitcode thunks end-to-end on device:
 *
 *   ncclGinBarrierSessionInit / ncclGinBarrierSessionSync   (rail GIN barrier)
 *   ncclBarrierSessionInit     / ncclBarrierSessionSync     (composite barrier)
 *
 * It is a separate binary because a GIN barrier is inherently collective: it
 * signals peer ranks over the network, so it needs MPI (>=2 ranks on >=2
 * GPUs), the host RCCL library (ncclCommInitRank / ncclDevCommCreate) AND the
 * device bitcode linked in for the thunks — none of which the single-process
 * IR_test harness provides.
 *
 * Modeled on test/transport/GinDeviceMPITests.cpp::Barrier_TwoRanks, but the
 * kernels call the exported C thunks from the bitcode instead of the C++
 * ncclGinBarrierSession<> / ncclBarrierSession<> templates directly — that is
 * what makes this a test of librccl_device.bc rather than of the headers.
 *
 * Run (single node, 2 GPUs, IB GIN proxy):
 *   NCCL_CUMEM_ENABLE=1 NCCL_GIN_TYPE=2 NCCL_GIN_ENABLE=1 \
 *   RCCL_ENABLE_INTRANET=1 \
 *   mpirun -np 2 ./IR_gin_mpi_test.exe
 *
 * Every case GTEST_SKIPs (rather than fails) when a prerequisite is absent so
 * the suite is safe on machines without GIN-capable hardware.
 ************************************************************************/

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device_wrapper.h>

/* -----------------------------------------------------------------------
 * Error-checking macros — fail the current assertion rather than abort the
 * whole MPI job, so one rank's HIP/NCCL error is still reported cleanly.
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

/* -----------------------------------------------------------------------
 * GIN prerequisite checks — mirrored from GinDeviceMPITests.cpp so this
 * standalone binary skips for the same reasons the in-tree MPI suite does.
 * --------------------------------------------------------------------- */
std::string ginEnvDisabledReason() {
  if (const char* e = std::getenv("NCCL_GIN_ENABLE"); e && std::strcmp(e, "0") == 0)
    return "GIN explicitly disabled by environment (NCCL_GIN_ENABLE=0)";
  return "";
}

std::string ginTypeReason() {
  const char* ginType = std::getenv("NCCL_GIN_TYPE");
  if (!ginType) return "GIN type not set (required NCCL_GIN_TYPE=2)";
  if (std::atoi(ginType) != 2)
    return std::string("Invalid GIN type: ") + ginType + " (required NCCL_GIN_TYPE=2)";
  return "";
}

std::string cuMemReason() {
  const char* cumem = std::getenv("NCCL_CUMEM_ENABLE");
  if (!cumem || std::strcmp(cumem, "1") != 0)
    return "Symmetric memory required (NCCL_CUMEM_ENABLE=1)";
  return "";
}

/* Single-node runs need intranet mode, else the topology pruner removes the
 * NET node and GIN has no path to bind. */
std::string intranetReason() {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int nodeSize = 0, worldSize = 0;
  MPI_Comm_size(nodeComm, &nodeSize);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  MPI_Comm_free(&nodeComm);
  if (nodeSize != worldSize) return "";  /* genuinely multi-node */
  const char* intra = std::getenv("RCCL_ENABLE_INTRANET");
  if (!intra || std::strcmp(intra, "1") != 0)
    return "Intranet mode required for single-node run (RCCL_ENABLE_INTRANET=1)";
  return "";
}

std::string ginProxyTestSkipReason() {
  for (auto check : {ginEnvDisabledReason, ginTypeReason, cuMemReason, intranetReason}) {
    if (auto reason = check(); !reason.empty()) return reason;
  }
  return "";
}

/* Node-local rank, used to bind each MPI rank to its own GPU. */
int localRankOf(MPI_Comm world) {
  MPI_Comm nodeComm;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
  int localRank = 0;
  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_free(&nodeComm);
  return localRank;
}

}  // namespace

/* =====================================================================
 * Device kernels — call the bitcode thunks (not the templates).
 *
 * ncclGin_C_init / ncclTeam* are NCCL_DEVICE_INLINE helpers from the staged
 * headers (they inline into the kernel); only the four nccl*BarrierSession*
 * symbols are external thunks resolved from librccl_device.bc.
 * ==================================================================== */

/* [C1] GIN-only rail barrier: Init once, then Sync `iters` times. Each Sync
 * sends a SignalInc to the peer and waits for the local cell to reach the new
 * epoch, so completion after many rounds proves both ranks stayed in lockstep
 * (a broken epoch would deadlock at iter 1). barrierIndex 0 < railGinBarrierCount. */
__global__ void k_ir_gin_barrier(int iters, struct ncclDevComm devComm) {
  /* ncclGin_C / the *_C session structs have no default ctor: the net is
   * constructed via its parameterized ctor, and the session is left as raw
   * aligned storage that the Init thunk placement-news into. */
  ncclGin_C net(devComm, NCCL_GIN_BACKEND_MASK_ALL, /*contextIndex=*/0,
                NCCL_GIN_RESOURCE_SHARING_GPU);

  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);

  __attribute__((aligned(alignof(ncclGinBarrierSession_C))))
      char sbuf[sizeof(ncclGinBarrierSession_C)];
  ncclGinBarrierSession_C* session =
      reinterpret_cast<ncclGinBarrierSession_C*>(sbuf);

  ncclGinBarrierSessionInit(session, coop, net, ncclTeamRail(devComm),
                            devComm.railGinBarrier, /*index=*/0);
  for (int i = 0; i < iters; ++i) {
    ncclGinBarrierSessionSync(session, coop, cuda::memory_order_relaxed,
                              ncclGinFenceLevel::Relaxed);
  }
}

/* [C2] Composite barrier: inner LSA (lsa team / lsaBarrier) + outer GIN (rail
 * team / railGinBarrier), matching the ncclTeamTagWorld decomposition. */
__global__ void k_ir_composite_barrier(int iters, struct ncclDevComm devComm) {
  ncclGin_C net(devComm, NCCL_GIN_BACKEND_MASK_ALL, /*contextIndex=*/0,
                NCCL_GIN_RESOURCE_SHARING_GPU);

  ncclCoopAny coop;
  ncclCoopAnyInitCta(&coop);

  __attribute__((aligned(alignof(ncclBarrierSession_C))))
      char sbuf[sizeof(ncclBarrierSession_C)];
  ncclBarrierSession_C* session =
      reinterpret_cast<ncclBarrierSession_C*>(sbuf);

  ncclBarrierSessionInit(session, coop,
                         /*innerTeam=*/ncclTeamLsa(devComm),
                         /*outerTeam=*/ncclTeamRail(devComm),
                         net,
                         /*innerBarHandle=*/devComm.lsaBarrier,
                         /*outerBarHandle=*/devComm.railGinBarrier,
                         /*index=*/0,
                         /*multimem=*/false,
                         /*innerMmHandle=*/ncclMultimemHandle{});
  for (int i = 0; i < iters; ++i) {
    ncclBarrierSessionSync(session, coop, cuda::memory_order_relaxed,
                           ncclGinFenceLevel::Relaxed);
  }
}

/* =====================================================================
 * Fixture — builds a 2-rank comm + GIN device comm once per test, with full
 * skip-gating. Each rank is bound to its own node-local GPU.
 * ==================================================================== */
class IRGinBarrierMPITest : public ::testing::Test {
 protected:
  ncclComm_t  comm_   = nullptr;
  hipStream_t stream_ = nullptr;
  ncclDevComm devComm_{};
  bool        devCommValid_ = false;
  int         worldRank_ = -1, worldSize_ = -1;

  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);

    if (auto reason = ginProxyTestSkipReason(); !reason.empty())
      GTEST_SKIP() << reason;
    if (worldSize_ != 2)
      GTEST_SKIP() << "Requires exactly 2 ranks (got " << worldSize_ << ")";

    int nDev = 0;
    ASSERT_EQ(hipGetDeviceCount(&nDev), hipSuccess);
    const int local = localRankOf(MPI_COMM_WORLD);
    if (nDev < worldSize_)
      GTEST_SKIP() << "Requires >=" << worldSize_ << " GPUs (got " << nDev << ")";
    ASSERT_EQ(hipSetDevice(local), hipSuccess);

    ncclUniqueId id{};
    if (worldRank_ == 0) ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    MPI_Bcast(&id, sizeof id, MPI_BYTE, 0, MPI_COMM_WORLD);

    ASSERT_EQ(ncclCommInitRank(&comm_, worldSize_, id, worldRank_), ncclSuccess);
    ASSERT_EQ(hipStreamCreate(&stream_), hipSuccess);
  }

  /* Try to build the GIN device comm. Returns the ncclResult_t so the caller
   * can GTEST_SKIP (vs FAIL) when device-comm creation is blocked by an
   * environment prerequisite rather than by the code under test. */
  ncclResult_t makeGinDevComm() {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.ginConnectionType  = NCCL_GIN_CONNECTION_FULL;
    /* ginForceEnable is required: ncclDevCommCreate only activates GIN when
     * nNodes>1 / ginForceEnable / ginSignalCount / ginCounterCount is set, and
     * the gate intentionally ignores *BarrierCount. Without it a single-node
     * barrier-only request gets NULL ginHandles[0] and faults on device. */
    reqs.ginForceEnable     = true;
    reqs.lsaBarrierCount    = 1;   /* composite inner barrier */
    reqs.railGinBarrierCount = 1;  /* GIN + composite outer barrier (index 0) */

    ncclResult_t r = ncclDevCommCreate(comm_, &reqs, &devComm_);
    if (r == ncclSuccess) {
      devCommValid_ = true;
      MPI_Barrier(MPI_COMM_WORLD);  /* all ranks set up before any launch */
    }
    return r;
  }

  void TearDown() override {
    if (devCommValid_) (void)ncclDevCommDestroy(comm_, &devComm_);
    if (stream_)       (void)hipStreamDestroy(stream_);
    if (comm_)         (void)ncclCommDestroy(comm_);
    if (worldSize_ == 2) MPI_Barrier(MPI_COMM_WORLD);
  }
};

/* Large enough that a stuck epoch deadlocks at iter 1 instead of silently
 * passing on iter 0; small enough to stay fast. */
static constexpr int kIters = 16;

/* =====================================================================
 * [C1] ncclGinBarrierSession{Init,Sync} — functional rail GIN barrier
 * ==================================================================== */
TEST_F(IRGinBarrierMPITest, C1_GinBarrierSession_TwoRanks) {
  /* ncclDevCommCreate needs symmetric-memory (cuMem/VMM) support, which the
   * runtime disables on hosts with Linux kernel < 6.8 or a HIP version below
   * the cuMem threshold (ROCm 7.0.2.x backport / HIP >= 7.12). Skip (not fail)
   * when a GIN device comm cannot be created on this host. */
  if (ncclResult_t r = makeGinDevComm(); r != ncclSuccess)
    GTEST_SKIP() << "ncclDevCommCreate failed (" << ncclGetErrorString(r)
                 << "): GIN device comm requires symmetric-memory (cuMem) "
                    "support — Linux kernel >= 6.8 and HIP >= 7.12 / ROCm "
                    "7.0.2.x. Not available on this host.";

  k_ir_gin_barrier<<<1, 32, 0, stream_>>>(kIters, devComm_);
  HIP_ASSERT(hipGetLastError());
  /* Kernel completion IS the assertion: Init + kIters Syncs across both ranks
   * over the GIN rail barrier all succeeded in lockstep. */
  HIP_ASSERT(hipStreamSynchronize(stream_));
  MPI_Barrier(MPI_COMM_WORLD);
}

/* =====================================================================
 * [C2] ncclBarrierSession{Init,Sync} — functional composite (LSA+GIN) barrier
 * ==================================================================== */
TEST_F(IRGinBarrierMPITest, C2_CompositeBarrierSession_TwoRanks) {
  /* Same symmetric-memory (cuMem) prerequisite as C1; skip when unavailable. */
  if (ncclResult_t r = makeGinDevComm(); r != ncclSuccess)
    GTEST_SKIP() << "ncclDevCommCreate failed (" << ncclGetErrorString(r)
                 << "): GIN device comm requires symmetric-memory (cuMem) "
                    "support — Linux kernel >= 6.8 and HIP >= 7.12 / ROCm "
                    "7.0.2.x. Not available on this host.";

  k_ir_composite_barrier<<<1, 32, 0, stream_>>>(kIters, devComm_);
  HIP_ASSERT(hipGetLastError());
  HIP_ASSERT(hipStreamSynchronize(stream_));
  MPI_Barrier(MPI_COMM_WORLD);
}

/* =====================================================================
 * MPI bring-up via a GoogleTest global environment.
 *
 * Keeps this binary in the same format as IR_test.cpp (links gtest_main, no
 * custom main): gtest_main calls RUN_ALL_TESTS, and this environment's
 * SetUp/TearDown run MPI_Init/MPI_Finalize around the whole suite.
 * ==================================================================== */
namespace {
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) MPI_Init(nullptr, nullptr);
  }
  void TearDown() override {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
  }
};

const ::testing::Environment* const kMpiEnv =
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);
}  // namespace
