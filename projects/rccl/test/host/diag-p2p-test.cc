/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/diagnostics/p2p.cc, #include-d via DIAG_P2P_CC_PATH to reach its file-static helpers.
 * ncclCalloc is DiagCalloc around that include: it fails the Nth calloc of a size or pads a zeroed guard element.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/diagnostics_p2p_device_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/signature-drift.h"
#include "fakes/transport_p2p_fakes.h"

#include "alloc.h"
#include "bootstrap.h"
#include "comm.h"
#include "graph.h"
#include "graph/topo.h"
#include "transport.h"

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::CaptureStdout;
using RcclUnitTesting::LogHas;
using RcclUnitTesting::ScopedDebugLogging;

static std::size_t g_diagCallocFailBytes = 0;
static int g_diagCallocFailNth = 0;
static bool g_diagCallocPad = false;
static std::vector<void*> g_diagHostLive;
template <typename T>
static ncclResult_t DiagCalloc(const char* file, int line, const char* fn, T** ptr, std::size_t nelem) {
  if (g_diagCallocFailBytes != 0 && nelem * sizeof(T) == g_diagCallocFailBytes && --g_diagCallocFailNth == 0) {
    g_diagCallocFailBytes = 0;
    return ncclSystemError;
  }
  const ncclResult_t ret =
      ncclCallocDebug(ptr, g_diagCallocPad && nelem != 0 ? nelem + 1 : nelem, file, line, fn, true);
  if (ret == ncclSuccess && *ptr != nullptr) {
    g_diagHostLive.push_back(*ptr);
  }
  return ret;
}

static std::vector<void*> g_diagHostFrees;
static void DiagFree(void* ptr) {
  g_diagHostFrees.push_back(ptr);
  auto it = std::find(g_diagHostLive.begin(), g_diagHostLive.end(), ptr);
  if (it != g_diagHostLive.end()) {
    g_diagHostLive.erase(it);
  } else if (ptr != nullptr) {
    ADD_FAILURE() << "free of untracked host pointer " << ptr;
  }
  std::free(ptr);
}

// Device memory is host heap; frees release only what the alloc seam made.
struct DiagDeviceFree {
  void* ptr;
  struct ncclMemManager* manager;
};
static std::vector<void*> g_diagDeviceLive;
static std::vector<DiagDeviceFree> g_diagDeviceFrees;

static ncclResult_t DefaultDiagCudaCalloc(void** ptr, std::size_t bytes, struct ncclMemManager*, ncclMemType_t) {
  *ptr = std::calloc(1, bytes);
  if (*ptr == nullptr) {
    return ncclSystemError;
  }
  g_diagDeviceLive.push_back(*ptr);
  return ncclSuccess;
}
static std::function<ncclResult_t(void**, std::size_t, struct ncclMemManager*, ncclMemType_t)> g_diagCudaCalloc =
    DefaultDiagCudaCalloc;

static ncclResult_t DefaultDiagCudaFree(void* ptr, struct ncclMemManager* manager) {
  g_diagDeviceFrees.push_back({ptr, manager});
  auto it = std::find(g_diagDeviceLive.begin(), g_diagDeviceLive.end(), ptr);
  if (it != g_diagDeviceLive.end()) {
    g_diagDeviceLive.erase(it);
    std::free(ptr);
  }
  return ncclSuccess;
}
static std::function<ncclResult_t(void*, struct ncclMemManager*)> g_diagCudaFree = DefaultDiagCudaFree;

static ncclResult_t DefaultDiagCuMemFreeAddr(void*, struct ncclMemManager*, int) {
  return ncclSuccess;
}
static std::function<ncclResult_t(void*, struct ncclMemManager*, int)> g_diagCuMemFreeAddr = DefaultDiagCuMemFreeAddr;
ASSERT_HOOK_MATCHES_PROD(g_diagCuMemFreeAddr, ncclCuMemFreeAddr);
static ncclResult_t DiagCuMemFreeAddr(void* ptr, struct ncclMemManager* manager, int numSegments = 1) {
  return g_diagCuMemFreeAddr(ptr, manager, numSegments);
}

#undef ncclCalloc
#define ncclCalloc(...) DiagCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)
#undef ncclCudaCalloc
#define ncclCudaCalloc(ptr, nelem, manager, memType) \
  g_diagCudaCalloc(reinterpret_cast<void**>(ptr), (nelem) * sizeof(**(ptr)), manager, memType)
#define ncclCudaFree(ptr, manager) g_diagCudaFree(ptr, manager)
#define ncclCuMemFreeAddr(...) DiagCuMemFreeAddr(__VA_ARGS__)
#define free(ptr) DiagFree(ptr)

#include DIAG_P2P_CC_PATH

#undef ncclCalloc
#undef ncclCudaCalloc
#undef ncclCudaFree
#undef ncclCuMemFreeAddr
#undef free

namespace {

constexpr uint64_t kHostHash = 0x40570000ULL;
constexpr uint64_t kPidHashBase = 0x91d00000ULL;
constexpr uintptr_t kMemManagerBits = 0x3e3;
constexpr uintptr_t kBootstrapBits = 0xb007;

void FailCallocOf(std::size_t bytes, int nth) {
  g_diagCallocFailBytes = bytes;
  g_diagCallocFailNth = nth;
}

// One DIAG_PRINT line: "<host>:<pid> <body>\n".
std::string DiagLine(const std::string& body) {
  diagLogInit();
  return std::string(diagLogHost) + ":" + std::to_string(diagLogPid) + " " + body + "\n";
}

std::string DiagP2pOkLine(int edges) {
  return DiagLine("NCCL DIAG [OK]   p2p: all " + std::to_string(edges) + " directed GPU P2P edges verified");
}

std::string DiagP2pCleanupLine(int rank, int result) {
  return DiagLine("NCCL DIAG [INFO] p2p: resource cleanup failed rank=" + std::to_string(rank) +
                  " result=" + std::to_string(result) + "; some temporary resources may remain");
}

ncclDiagP2pEdgeInfo Edge(int pathType, int handleType) {
  ncclDiagP2pEdgeInfo edge{};
  edge.pathType = pathType;
  edge.handleType = handleType;
  return edge;
}

// World rank 4 at slot 1 of a four-rank group; its outbound peers are the other slots.
constexpr int kGroupWorld = 6;
constexpr int kGroupRank = 4;
constexpr int kGroupN = 4;
constexpr int kGroupSelf = 1;
constexpr int kGroupRanks[kGroupN] = {1, 4, 5, 2};
constexpr int kGroupOutPeers[] = {0, 2, 3};
constexpr int kGroupOutPeerCount = static_cast<int>(std::size(kGroupOutPeers));

constexpr int OwnEdge(int slot) {
  return kGroupSelf * kGroupN + slot;
}

constexpr int InEdge(int slot) {
  return slot * kGroupN + kGroupSelf;
}

class DiagP2pMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (void* p : g_diagDeviceLive) {
      std::free(p);
    }
    g_diagDeviceLive.clear();
    g_diagDeviceFrees.clear();
    g_diagCudaCalloc = DefaultDiagCudaCalloc;
    g_diagCudaFree = DefaultDiagCudaFree;
    g_diagCuMemFreeAddr = DefaultDiagCuMemFreeAddr;
    g_diagCallocFailBytes = 0;
    g_diagCallocFailNth = 0;
    g_diagCallocPad = false;
    g_diagHostLive.clear();
    g_diagHostFrees.clear();
    ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
    ResetDevRuntimeMicroFakes();
    ResetDiagnosticsP2pDeviceFakes();
    ResetHipFakes();
    ResetNcclFakes();
    ResetTransportP2pFakes();
  }

  // Rank r: own process, cudaDev r, nvmlDev 10 + r.
  void BuildComm(int nRanks, int rank, std::vector<int> localRanks) {
    comm_ = std::make_unique<ncclComm>();
    peers_.assign(nRanks, ncclPeerInfo{});
    for (int r = 0; r < nRanks; r++) {
      peers_[r].rank = r;
      peers_[r].cudaDev = r;
      peers_[r].nvmlDev = 10 + r;
      peers_[r].hostHash = kHostHash;
      peers_[r].pidHash = kPidHashBase + r;
    }
    localRanks_ = std::move(localRanks);
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->cudaDev = rank;
    comm_->memManager = reinterpret_cast<ncclMemManager*>(kMemManagerBits);
    comm_->bootstrap = reinterpret_cast<void*>(kBootstrapBits);
    comm_->peerInfo = peers_.data();
    comm_->localRankToRank = localRanks_.data();
    comm_->localRanks = static_cast<int>(localRanks_.size());
    for (int slot = 0; slot < comm_->localRanks; slot++) {
      if (localRanks_[slot] == rank) {
        comm_->localRank = slot;
      }
    }
  }

  void BuildGroupComm() {
    BuildComm(kGroupWorld, kGroupRank, std::vector<int>(std::begin(kGroupRanks), std::end(kGroupRanks)));
  }

  void BuildLeaderComm() {
    BuildComm(4, 0, {0, 3, 1, 2});
  }

  // GPU node i holds rank gpuRanks[i]; paths[i][j] is node i to node j.
  void BuildTopo(const std::vector<int>& gpuRanks, const std::vector<std::vector<int>>& paths) {
    topo_ = std::make_unique<ncclTopoSystem>();
    const int n = static_cast<int>(gpuRanks.size());
    links_.assign(n, std::vector<ncclTopoLinkList>(n));
    topo_->nodes[GPU].count = n;
    for (int i = 0; i < n; i++) {
      topo_->nodes[GPU].nodes[i].gpu.rank = gpuRanks[i];
      topo_->nodes[GPU].nodes[i].paths[GPU] = links_[i].data();
      for (int j = 0; j < n; j++) {
        links_[i][j].type = paths[i][j];
      }
    }
    comm_->topo = topo_.get();
  }

  std::unique_ptr<ncclComm> comm_;
  std::vector<ncclPeerInfo> peers_;
  std::vector<int> localRanks_;
  std::unique_ptr<ncclTopoSystem> topo_;
  std::vector<std::vector<ncclTopoLinkList>> links_;
};

TEST_F(DiagP2pMicrotest, Patterns_PackTagAndBothRanks) {
  EXPECT_EQ(ncclDiagP2pWritePattern(3, 5), (1ULL << 62) | (3ULL << 31) | 5ULL);
  EXPECT_EQ(ncclDiagP2pReadPattern(5, 3), (2ULL << 62) | (5ULL << 31) | 3ULL);
  EXPECT_EQ(ncclDiagP2pWritePattern(-1, -1), (1ULL << 62) | (0x7fffffffULL << 31) | 0x7fffffffULL);
  EXPECT_EQ(ncclDiagP2pReadPattern(-1, 0), (2ULL << 62) | (0x7fffffffULL << 31));
  EXPECT_EQ(ncclDiagP2pWritePattern(0, -1), (1ULL << 62) | 0x7fffffffULL);
}

TEST_F(DiagP2pMicrotest, RankToSlot_FindsSlotWithinBoundOnly) {
  const int ranks[] = {7, 3, 9};
  EXPECT_EQ(ncclDiagP2pRankToSlot(ranks, 3, 7), 0);
  EXPECT_EQ(ncclDiagP2pRankToSlot(ranks, 3, 3), 1);
  EXPECT_EQ(ncclDiagP2pRankToSlot(ranks, 3, 9), 2);
  EXPECT_EQ(ncclDiagP2pRankToSlot(ranks, 3, 4), -1);
  EXPECT_EQ(ncclDiagP2pRankToSlot(ranks, 2, 9), -1);
}

TEST_F(DiagP2pMicrotest, SameProcess_RequiresHostAndPidHashMatch) {
  BuildComm(3, 0, {0, 1, 2});
  peers_[1].pidHash = peers_[0].pidHash;
  peers_[2].pidHash = peers_[0].pidHash;
  peers_[2].hostHash = kHostHash + 1;
  EXPECT_TRUE(ncclDiagP2pSameProcess(comm_.get(), 0, 1));
  EXPECT_FALSE(ncclDiagP2pSameProcess(comm_.get(), 0, 2));
  peers_[1].pidHash = kPidHashBase + 7;
  EXPECT_FALSE(ncclDiagP2pSameProcess(comm_.get(), 0, 1));
}

TEST_F(DiagP2pMicrotest, HandleType_SameProcessIsDirectWithoutQueryingCuMem) {
  BuildComm(2, 0, {0, 1});
  peers_[1].pidHash = peers_[0].pidHash;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  EXPECT_EQ(ncclDiagP2pHandleType(comm_.get(), 0, 1), ncclDiagP2pHandleDirect);
  EXPECT_EQ(cuMem.calls, 0);
}

TEST_F(DiagP2pMicrotest, HandleType_CrossProcessWithoutCuMemIsLegacyIpc) {
  BuildComm(2, 0, {0, 1});
  ScopedHook cuMem(g_cuMemEnable, [] { return 0; });
  EXPECT_EQ(ncclDiagP2pHandleType(comm_.get(), 0, 1), ncclDiagP2pHandleLegacyIpc);
  EXPECT_EQ(cuMem.calls, 1);
}

// ROCm gap: CUDART_VERSION unset hides POSIX_FD; a fix flips this pin.
TEST_F(DiagP2pMicrotest, HandleType_CuMemPosixFdReportsOtherOnRocm) {
  BuildComm(2, 0, {0, 1});
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;
  EXPECT_EQ(ncclDiagP2pHandleType(comm_.get(), 0, 1), ncclDiagP2pHandleCuMemOther);
}

TEST_F(DiagP2pMicrotest, HandleName_MapsEveryHandleAndDefaultsToNone) {
  EXPECT_STREQ(ncclDiagP2pHandleName(ncclDiagP2pHandleDirect), "DIRECT");
  EXPECT_STREQ(ncclDiagP2pHandleName(ncclDiagP2pHandleLegacyIpc), "LEGACY_CUDA_IPC");
  EXPECT_STREQ(ncclDiagP2pHandleName(ncclDiagP2pHandleCuMemPosixFd), "CUMEM_POSIX_FD");
  EXPECT_STREQ(ncclDiagP2pHandleName(ncclDiagP2pHandleCuMemFabric), "CUMEM_FABRIC");
  EXPECT_STREQ(ncclDiagP2pHandleName(ncclDiagP2pHandleCuMemOther), "CUMEM_OTHER");
  EXPECT_STREQ(ncclDiagP2pHandleName(0), "NONE");
  EXPECT_STREQ(ncclDiagP2pHandleName(6), "NONE");
}

TEST_F(DiagP2pMicrotest, ReasonName_MapsEveryReasonAndDefaultsToNone) {
  const char* const kNames[] = {"none",          "indirect",   "noDescriptor", "import", "writeLaunch",
                                "writeMismatch", "readLaunch", "readMismatch", "topo",   "localCuda"};
  static_assert(std::size(kNames) == ncclDiagP2pReasonLocalCuda + 1, "kNames must cover every reason");
  for (int reason = 0; reason <= ncclDiagP2pReasonLocalCuda; reason++) {
    EXPECT_STREQ(ncclDiagP2pReasonName(reason), kNames[reason]) << "reason " << reason;
  }
  EXPECT_STREQ(ncclDiagP2pReasonName(ncclDiagP2pReasonLocalCuda + 1), "none");
}

TEST_F(DiagP2pMicrotest, PathName_IndexesPathTableAndRejectsOutOfRange) {
  const char* const kNames[] = {"LOC", "XGMI", "NVB", "C2C", "PIX", "PXB", "P2C", "PXN", "PHB", "SYS", "NET", "DIS"};
  static_assert(std::size(kNames) == PATH_DIS + 1, "kNames must cover every PATH_* type");
  for (int path = PATH_LOC; path <= PATH_DIS; path++) {
    EXPECT_STREQ(ncclDiagP2pPathName(path), kNames[path]) << "path " << path;
  }
  EXPECT_STREQ(ncclDiagP2pPathName(PATH_LOC - 1), "UNK");
  EXPECT_STREQ(ncclDiagP2pPathName(PATH_DIS + 1), "UNK");
}

TEST_F(DiagP2pMicrotest, IsFabricEdge_FabricHandleOrNetPath) {
  ncclDiagP2pEdgeInfo fabricOverNvl = Edge(PATH_NVL, ncclDiagP2pHandleCuMemFabric);
  ncclDiagP2pEdgeInfo legacyOverNet = Edge(PATH_NET, ncclDiagP2pHandleLegacyIpc);
  ncclDiagP2pEdgeInfo legacyOverSys = Edge(PATH_SYS, ncclDiagP2pHandleLegacyIpc);
  EXPECT_TRUE(ncclDiagP2pIsFabricEdge(&fabricOverNvl));
  EXPECT_TRUE(ncclDiagP2pIsFabricEdge(&legacyOverNet));
  EXPECT_FALSE(ncclDiagP2pIsFabricEdge(&legacyOverSys));
}

constexpr char kImexAdvice[] =
    "check the IMEX domain with 'nvidia-imex-ctl -H -N' (nodes READY, connectivity C) and verify access to "
    "/dev/nvidia-caps-imex-channels/channel*";
constexpr char kNvlinkAdvice[] =
    "check the single-node NVLink topology and peer-access state with 'nvidia-smi topo -m' and "
    "'nvidia-smi topo -p2p n'";
constexpr char kPcieAdvice[] =
    "check the affected pair with 'nvidia-smi topo -p2p p', then check Linux bare-metal IOMMU mode and PCIe "
    "ACS settings";
constexpr char kGenericAdvice[] =
    "inspect the affected GPU pair with 'nvidia-smi topo -m' and the applicable 'nvidia-smi topo -p2p' check";

TEST_F(DiagP2pMicrotest, EdgeAdvice_SelectsByFabricThenPathClass) {
  const struct {
    int path;
    int handle;
    const char* advice;
  } kCases[] = {
      {PATH_NVL, ncclDiagP2pHandleCuMemFabric, kImexAdvice}, {PATH_NET, ncclDiagP2pHandleDirect, kImexAdvice},
      {PATH_NVL, ncclDiagP2pHandleDirect, kNvlinkAdvice},    {PATH_NVB, ncclDiagP2pHandleDirect, kNvlinkAdvice},
      {PATH_PIX, ncclDiagP2pHandleDirect, kPcieAdvice},      {PATH_PXB, ncclDiagP2pHandleDirect, kPcieAdvice},
      {PATH_PHB, ncclDiagP2pHandleDirect, kPcieAdvice},      {PATH_SYS, ncclDiagP2pHandleDirect, kPcieAdvice},
      {PATH_LOC, ncclDiagP2pHandleDirect, kGenericAdvice},   {PATH_C2C, ncclDiagP2pHandleDirect, kGenericAdvice},
      {PATH_P2C, ncclDiagP2pHandleDirect, kGenericAdvice},   {PATH_PXN, ncclDiagP2pHandleDirect, kGenericAdvice},
      {PATH_DIS, ncclDiagP2pHandleDirect, kGenericAdvice},
  };
  for (const auto& c : kCases) {
    ncclDiagP2pEdgeInfo edge = Edge(c.path, c.handle);
    EXPECT_STREQ(ncclDiagP2pEdgeAdvice(&edge), c.advice) << "path " << c.path << " handle " << c.handle;
  }
}

TEST_F(DiagP2pMicrotest, ImportAdvice_FabricDefersToEdgeAdviceElseByHandle) {
  const struct {
    int path;
    int handle;
    const char* advice;
  } kCases[] = {
      {PATH_NET, ncclDiagP2pHandleLegacyIpc, kImexAdvice},
      {PATH_NVL, ncclDiagP2pHandleCuMemFabric, kImexAdvice},
      {PATH_NVL, ncclDiagP2pHandleDirect,
       "inspect preceding CUDA peer-access or virtual-memory mapping errors on the source rank"},
      {PATH_NVL, ncclDiagP2pHandleLegacyIpc,
       "check CUDA IPC support, GPU visibility, and process or container isolation"},
      {PATH_NVL, ncclDiagP2pHandleCuMemPosixFd,
       "check cuMem POSIX-FD sharing support and process or container permissions"},
      {PATH_NVL, ncclDiagP2pHandleCuMemOther,
       "check CUDA virtual-memory handle support and permissions between the processes"},
  };
  for (const auto& c : kCases) {
    ncclDiagP2pEdgeInfo edge = Edge(c.path, c.handle);
    EXPECT_STREQ(ncclDiagP2pImportAdvice(&edge), c.advice) << "path " << c.path << " handle " << c.handle;
  }
}

TEST_F(DiagP2pMicrotest, PathType_ReadsSrcRowAtDstColumn) {
  BuildComm(6, 4, {4, 1, 5});
  BuildTopo({4, 1, 5},
            {{PATH_LOC, PATH_NVL, PATH_PIX}, {PATH_SYS, PATH_LOC, PATH_PXB}, {PATH_PHB, PATH_NVB, PATH_LOC}});
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 4, 5), PATH_PIX);
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 5, 4), PATH_PHB);
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 1, 5), PATH_PXB);
}

TEST_F(DiagP2pMicrotest, PathType_DisconnectedWhenTopoMissingRankAbsentOrTypeOutOfRange) {
  BuildComm(6, 4, {4, 1});
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 4, 1), PATH_DIS);
  BuildTopo({4, 1}, {{PATH_LOC, -1}, {PATH_DIS + 1, PATH_LOC}});
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 3, 1), PATH_DIS);
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 4, 3), PATH_DIS);
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 4, 1), PATH_DIS);
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 1, 4), PATH_DIS);
  links_[0][1].type = PATH_NET;
  EXPECT_EQ(ncclDiagP2pPathType(comm_.get(), 4, 1), PATH_NET);
}

TEST_F(DiagP2pMicrotest, SetReason_FirstReasonWins) {
  ncclDiagP2pEdgeResult result{};
  ncclDiagP2pSetReason(&result, ncclDiagP2pReasonImport);
  EXPECT_EQ(result.reason, ncclDiagP2pReasonImport);
  ncclDiagP2pSetReason(&result, ncclDiagP2pReasonReadMismatch);
  EXPECT_EQ(result.reason, ncclDiagP2pReasonImport);
}

TEST_F(DiagP2pMicrotest, SetLocalReason_TouchesOnlyTestedEdgesInOwnRow) {
  constexpr int kN = 3;
  ncclDiagP2pEdgeResult results[kN * kN] = {};
  for (auto& r : results) {
    r.tested = 1;
  }
  results[1 * kN + 0].tested = 0;
  results[1 * kN + 2].reason = ncclDiagP2pReasonImport;
  ncclDiagP2pSetLocalReason(1, kN, results, ncclDiagP2pReasonWriteLaunch);
  const int kWant[kN * kN] = {0, 0, 0, 0, ncclDiagP2pReasonWriteLaunch, ncclDiagP2pReasonImport, 0, 0, 0};
  for (int i = 0; i < kN * kN; i++) {
    EXPECT_EQ(results[i].reason, kWant[i]) << "entry " << i;
  }
}

TEST_F(DiagP2pMicrotest, CudaSuccess_FailureWarnsAndClearsStickyError) {
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });
  EXPECT_TRUE(ncclDiagP2pCudaSuccess(hipSuccess, "phaseA"));
  EXPECT_EQ(lastError.calls, 0);
  bool ok = true;
  const std::string log = CaptureLog([&] { ok = ncclDiagP2pCudaSuccess(hipErrorInvalidValue, "phaseB"); });
  EXPECT_FALSE(ok);
  EXPECT_EQ(lastError.calls, 1);
  EXPECT_TRUE(LogHas(log, " Diagnostics P2P phaseB CUDA failure: [hip_fake] stub error\n")) << log;
}

TEST_F(DiagP2pMicrotest, NcclSuccess_FailureWarnsWithResult) {
  EXPECT_TRUE(ncclDiagP2pNcclSuccess(ncclSuccess, "phaseA"));
  bool ok = true;
  const std::string log = CaptureLog([&] { ok = ncclDiagP2pNcclSuccess(ncclSystemError, "phaseB"); });
  EXPECT_FALSE(ok);
  EXPECT_TRUE(LogHas(log, " Diagnostics P2P phaseB returned 2\n")) << log;
}

TEST_F(DiagP2pMicrotest, LogEdge_InfoLineCarriesBothPeersAndOptionalExtra) {
  BuildComm(6, 4, {4, 1});
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  ncclDiagP2pEdgeInfo edge = Edge(PATH_PXB, ncclDiagP2pHandleLegacyIpc);
  edge.read = 1;
  const std::string log = CaptureLog([&] {
    ncclDiagP2pLogEdge(comm_.get(), "write", 4, 1, &edge, nullptr);
    ncclDiagP2pLogEdge(comm_.get(), "import", 1, 4, &edge, "import=1");
  });
  EXPECT_TRUE(LogHas(log,
                     " Diagnostics P2P write srcRank=4 srcCudaDev=4 srcNvmlDev=14 dstRank=1 dstCudaDev=1 "
                     "dstNvmlDev=11 path=PXB handle=LEGACY_CUDA_IPC topoRead=1\n"))
      << log;
  EXPECT_TRUE(LogHas(log,
                     " Diagnostics P2P import srcRank=1 srcCudaDev=1 srcNvmlDev=11 dstRank=4 dstCudaDev=4 "
                     "dstNvmlDev=14 path=PXB handle=LEGACY_CUDA_IPC topoRead=1 import=1\n"))
      << log;
}

TEST_F(DiagP2pMicrotest, FormatPeerFields_WritesBothPeersAndTruncatesToSize) {
  BuildComm(6, 4, {4, 1});
  ncclDiagP2pEdgeInfo edge = Edge(PATH_SYS, ncclDiagP2pHandleCuMemOther);
  char buf[256];
  ncclDiagP2pFormatPeerFields(comm_.get(), 1, 4, &edge, buf, sizeof(buf));
  EXPECT_STREQ(buf,
               "srcRank=1 srcCudaDev=1 srcNvmlDev=11 dstRank=4 dstCudaDev=4 dstNvmlDev=14 path=SYS "
               "handle=CUMEM_OTHER");
  char small[12];
  ncclDiagP2pFormatPeerFields(comm_.get(), 1, 4, &edge, small, sizeof(small));
  EXPECT_STREQ(small, "srcRank=1 s");
}

TEST_F(DiagP2pMicrotest, BuildGroupSummary_CountsTestedPassedAndIndirect) {
  constexpr int kN = 3;
  ncclDiagP2pEdgeResult results[kN * kN] = {};
  results[0 * kN + 1] = {1, ncclDiagP2pReasonNone, 0, 0, 0};
  results[0 * kN + 2] = {1, ncclDiagP2pReasonImport, 0, 0, 0};
  results[1 * kN + 0] = {0, ncclDiagP2pReasonIndirect, 0, 0, 0};
  results[1 * kN + 2] = {1, ncclDiagP2pReasonNone, 0, 0, 0};
  results[2 * kN + 0] = {1, ncclDiagP2pReasonIndirect, 0, 0, 0};
  results[2 * kN + 2] = {0, ncclDiagP2pReasonNone, 0, 0, 0};
  ncclDiagP2pSummary summary = {100, 200, 300};
  ncclDiagP2pBuildGroupSummary(kN, results, &summary);
  EXPECT_EQ(summary.tested, 4u);
  EXPECT_EQ(summary.passed, 2u);
  EXPECT_EQ(summary.skipped, 2u);
}

TEST_F(DiagP2pMicrotest, ReportSummary_PrintsOnlyOnRankZeroWithCountsSummedOverRanks) {
  BuildComm(3, 1, {0, 1, 2});
  ncclDiagP2pSummary summaries[3] = {{2, 2, 0}, {3, 3, 0}, {1, 1, 0}};
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }), "");
  comm_->rank = 0;
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }),
            DiagLine("NCCL DIAG [OK]   p2p: all 6 directed GPU P2P edges verified"));
  summaries[2] = {1, 1, 2};
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }),
            DiagLine("NCCL DIAG [OK]   p2p: all 6 directed GPU P2P edges verified (skipped indirect=2)"));
  summaries[1] = {3, 1, 1};
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }),
            DiagLine("NCCL DIAG [INFO] p2p: 4/6 directed GPU P2P edges verified (skipped indirect=3)"));
  summaries[1] = {3, 1, 0};
  summaries[2] = {1, 1, 0};
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }),
            DiagLine("NCCL DIAG [INFO] p2p: 4/6 directed GPU P2P edges verified"));
}

TEST_F(DiagP2pMicrotest, ReportSummary_SilentWhenNothingTested) {
  BuildComm(2, 0, {0, 1});
  ncclDiagP2pSummary summaries[2] = {{0, 0, 3}, {0, 0, 1}};
  EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReportSummary(comm_.get(), summaries); }), "");
}

constexpr char kFields41[] =
    "srcRank=4 srcCudaDev=4 srcNvmlDev=14 dstRank=1 dstCudaDev=1 dstNvmlDev=11 path=PIX handle=LEGACY_CUDA_IPC";

TEST_F(DiagP2pMicrotest, Report_EachReasonHasItsOwnLine) {
  BuildComm(6, 4, {4, 1});
  const ncclDiagP2pEdgeInfo edge = Edge(PATH_PIX, ncclDiagP2pHandleLegacyIpc);
  const std::string fields = kFields41;
  const struct {
    int reason;
    std::string line;
  } kCases[] = {
      {ncclDiagP2pReasonNoDescriptor,
       "NCCL DIAG [INFO] p2p: destination buffer unavailable " + fields +
           " reason=noDescriptor; inspect earlier allocation, export, or initialization errors on the destination "
           "rank, then " + kPcieAdvice},
      {ncclDiagP2pReasonLocalCuda,
       "NCCL DIAG [INFO] p2p: local CUDA setup failed " + fields +
           " reason=localCuda; inspect preceding device, stream, allocation, or initialization errors on the source "
           "rank"},
      {ncclDiagP2pReasonImport,
       "NCCL DIAG [INFO] p2p: peer-memory import failed " + fields +
           " reason=import; check CUDA IPC support, GPU visibility, and process or container isolation"},
      {ncclDiagP2pReasonWriteMismatch,
       "NCCL DIAG [INFO] p2p: write mismatch " + fields +
           " expected=0x4000000200000001 got=0x0000000000000abc verify=0x0000000000000def; " + kPcieAdvice},
      {ncclDiagP2pReasonReadMismatch,
       "NCCL DIAG [INFO] p2p: read mismatch " + fields + " expected=0x8000000080000004 got=0x0000000000000123; " +
           kPcieAdvice},
      {ncclDiagP2pReasonTopo, "NCCL DIAG [INFO] p2p: topology check failed " + fields +
                                  " reason=topo; inspect preceding topology records, then " + kPcieAdvice},
      {ncclDiagP2pReasonWriteLaunch, "NCCL DIAG [INFO] p2p: launch/check failed " + fields +
                                         " reason=writeLaunch; inspect preceding CUDA or NCCL warnings, then " +
                                         kPcieAdvice},
      {ncclDiagP2pReasonReadLaunch, "NCCL DIAG [INFO] p2p: launch/check failed " + fields +
                                        " reason=readLaunch; inspect preceding CUDA or NCCL warnings, then " +
                                        kPcieAdvice},
  };
  for (const auto& c : kCases) {
    const ncclDiagP2pEdgeResult result = {1, c.reason, 0xabc, 0xdef, 0x123};
    EXPECT_EQ(CaptureStdout([&] { ncclDiagP2pReport(comm_.get(), 4, 1, &edge, &result); }), DiagLine(c.line));
  }
}

TEST_F(DiagP2pMicrotest, ReportGroupFailures_ReportsTestedFailuresBySlotRanksInOrder) {
  BuildComm(6, 4, {4, 1, 5});
  constexpr int kN = 3;
  ncclDiagP2pEdgeInfo edges[kN * kN];
  for (auto& e : edges) {
    e = Edge(PATH_SYS, ncclDiagP2pHandleCuMemOther);
  }
  edges[0 * kN + 1] = Edge(PATH_PIX, ncclDiagP2pHandleLegacyIpc);
  ncclDiagP2pEdgeResult results[kN * kN] = {};
  results[0 * kN + 1] = {1, ncclDiagP2pReasonImport, 0, 0, 0};
  results[0 * kN + 2] = {1, ncclDiagP2pReasonNone, 0, 0, 0};
  results[1 * kN + 0] = {0, ncclDiagP2pReasonIndirect, 0, 0, 0};
  results[2 * kN + 1] = {1, ncclDiagP2pReasonTopo, 0, 0, 0};
  const std::string importLine = std::string("NCCL DIAG [INFO] p2p: peer-memory import failed ") + kFields41 +
                                 " reason=import; check CUDA IPC support, GPU visibility, and process or container "
                                 "isolation";
  const std::string topoLine = std::string(
                                   "NCCL DIAG [INFO] p2p: topology check failed srcRank=5 srcCudaDev=5 srcNvmlDev=15 "
                                   "dstRank=1 dstCudaDev=1 dstNvmlDev=11 path=SYS handle=CUMEM_OTHER reason=topo; "
                                   "inspect preceding topology records, then ") +
                               kPcieAdvice;
  EXPECT_EQ(
      CaptureStdout([&] { ncclDiagP2pReportGroupFailures(comm_.get(), localRanks_.data(), kN, edges, results); }),
      DiagLine(importLine) + DiagLine(topoLine));
}

TEST_F(DiagP2pMicrotest, BuildRankSet_LocalRanksOrMnnvlClique) {
  BuildComm(6, 1, {4, 1, 5});
  int* ranks = nullptr;
  int rank = -7;
  int nRanks = -7;
  EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclSuccess);
  EXPECT_EQ(ranks, localRanks_.data());
  EXPECT_EQ(rank, 1);
  EXPECT_EQ(nRanks, 3);
  int cliqueRanks[] = {0, 2, 1, 3};
  comm_->MNNVL = 1;
  comm_->clique.ranks = cliqueRanks;
  comm_->clique.size = 4;
  comm_->cliqueRank = 2;
  EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclSuccess);
  EXPECT_EQ(ranks, cliqueRanks);
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(nRanks, 4);
}

void SetUuids(std::vector<ncclPeerInfo>* peers, const std::vector<uint8_t>& tags) {
  for (size_t r = 0; r < tags.size(); r++) {
    std::memset((*peers)[r].fabricInfo.clusterUuid, 0, sizeof((*peers)[r].fabricInfo.clusterUuid));
    (*peers)[r].fabricInfo.clusterUuid[15] = tags[r];
  }
}

TEST_F(DiagP2pMicrotest, BuildRankSet_CrossCliqueCollectsSameUuidInRankOrder) {
  BuildComm(6, 5, {4, 5});
  comm_->p2pCrossClique = true;
  comm_->MNNVL = 1;
  SetUuids(&peers_, {7, 9, 7, 9, 7, 9});
  comm_->nvlDomainSize = 3;
  int* ranks = nullptr;
  int rank = -7;
  int nRanks = -7;
  ASSERT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclSuccess);
  EXPECT_EQ(std::vector<int>(ranks, ranks + 3), (std::vector<int>{1, 3, 5}));
  EXPECT_EQ(rank, 2);
  EXPECT_EQ(nRanks, 3);
  std::free(ranks);
}

TEST_F(DiagP2pMicrotest, BuildRankSet_CrossCliqueRejectsDomainSizeMismatch) {
  BuildComm(4, 2, {0, 1, 2, 3});
  comm_->p2pCrossClique = true;
  SetUuids(&peers_, {3, 8, 8, 8});
  int* ranks = nullptr;
  int rank = -7;
  int nRanks = -7;
  for (int domainSize : {0, -1}) {
    comm_->nvlDomainSize = domainSize;
    EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclInternalError);
    EXPECT_EQ(ranks, nullptr);
    EXPECT_EQ(rank, -7);
    EXPECT_EQ(nRanks, -7);
  }
  g_diagCallocPad = true;
  for (int domainSize : {2, 4}) {
    comm_->nvlDomainSize = domainSize;
    EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclInternalError) << domainSize;
    EXPECT_EQ(nRanks, domainSize);
    ASSERT_NE(ranks, nullptr) << domainSize;
    EXPECT_EQ(ranks[domainSize], 0);
    std::free(ranks);
    ranks = nullptr;
  }
  comm_->nvlDomainSize = 3;
  EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclSuccess);
  EXPECT_EQ(rank, 1);
  std::free(ranks);
}

TEST_F(DiagP2pMicrotest, BuildRankSet_CrossCliqueCallocFailurePropagates) {
  BuildComm(3, 1, {0, 1, 2});
  comm_->p2pCrossClique = true;
  comm_->nvlDomainSize = 3;
  FailCallocOf(3 * sizeof(int), 1);
  int* ranks = nullptr;
  int rank = -7;
  int nRanks = -7;
  EXPECT_EQ(ncclDiagP2pBuildRankSet(comm_.get(), &ranks, &rank, &nRanks), ncclSystemError);
  EXPECT_EQ(ranks, nullptr);
}

::testing::AssertionResult VerifyNcclDiagP2pEdgeInfo(const ncclDiagP2pEdgeInfo& actual,
                                                     const ncclDiagP2pEdgeInfo& expected) {
  const struct {
    const char* name;
    int actual;
    int expected;
  } fields[] = {
      {"p2p", actual.p2p, expected.p2p},
      {"read", actual.read, expected.read},
      {"pathType", actual.pathType, expected.pathType},
      {"sameProcess", actual.sameProcess, expected.sameProcess},
      {"handleType", actual.handleType, expected.handleType},
  };
  std::string mismatches;
  for (const auto& field : fields) {
    if (field.actual != field.expected) {
      mismatches += std::string(" ") + field.name + "=" + std::to_string(field.actual) + " (expected " +
                    std::to_string(field.expected) + ")";
    }
  }
  if (mismatches.empty()) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "edge info fields differ:" << mismatches;
}

const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x5157);

hipError_t HonestMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind, hipStream_t) {
  std::memcpy(dst, src, bytes);
  return hipSuccess;
}

TEST_F(DiagP2pMicrotest, DiscoverLocalEdges_ClassifiesEachPeerInOwnRowOnly) {
  BuildComm(6, 4, {1, 4, 5, 2, 0});
  constexpr int kN = 5;
  constexpr int kSelf = 1;
  std::vector<std::vector<int>> paths(kN, std::vector<int>(kN, PATH_DIS));
  paths[kSelf] = {PATH_NVL, PATH_NET, PATH_PIX, PATH_SYS, PATH_PHB};
  BuildTopo({1, 4, 5, 2, 0}, paths);
  peers_[1].pidHash = peers_[4].pidHash;
  peers_[4].cudaDev = 3;
  std::vector<ncclDiagP2pEdgeInfo> edges(kN * kN);
  std::memset(edges.data(), 0x7f, edges.size() * sizeof(edges[0]));
  const std::vector<ncclDiagP2pEdgeInfo> poison = edges;
  std::vector<ncclDiagP2pEdgeResult> results(kN * kN);
  std::vector<int> dsts;
  ScopedHook topo(g_ncclTopoCheckP2p, [&](int rank1, int rank2, int* p2p, int* read, int* inter, int*, int*) {
    EXPECT_EQ(rank1, 4);
    dsts.push_back(rank2);
    *p2p = rank2 != 0;
    *read = rank2 != 5;
    *inter = rank2 == 5 ? 2 : (rank2 == 0 ? 3 : -1);
    return rank2 == 2 ? ncclSystemError : ncclSuccess;
  });
  int outPeers[kN] = {99, 99, 99, 99, 99};
  int outPeerCount = 99;
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  const std::string log = CaptureLog([&] {
    ncclDiagP2pDiscoverLocalEdges(comm_.get(), localRanks_.data(), kSelf, kN, edges.data(), results.data(), outPeers,
                                  &outPeerCount);
  });
  EXPECT_EQ(dsts, (std::vector<int>{1, 5, 2, 0}));
  EXPECT_EQ(outPeerCount, 1);
  EXPECT_EQ(outPeers[0], 0);
  const ncclDiagP2pEdgeInfo* row = edges.data() + kSelf * kN;
  EXPECT_TRUE(VerifyNcclDiagP2pEdgeInfo(row[0], {1, 1, PATH_NVL, 1, ncclDiagP2pHandleDirect}));
  EXPECT_TRUE(VerifyNcclDiagP2pEdgeInfo(row[1], {0, 0, PATH_LOC, 1, ncclDiagP2pHandleDirect}));
  EXPECT_TRUE(VerifyNcclDiagP2pEdgeInfo(row[2], {0, 0, PATH_PIX, 0, ncclDiagP2pHandleLegacyIpc}));
  EXPECT_TRUE(VerifyNcclDiagP2pEdgeInfo(row[3], {0, 0, PATH_SYS, 0, ncclDiagP2pHandleLegacyIpc}));
  EXPECT_TRUE(VerifyNcclDiagP2pEdgeInfo(row[4], {0, 1, PATH_PHB, 0, ncclDiagP2pHandleLegacyIpc}));
  const int kTested[kN] = {1, 0, 0, 1, 0};
  const int kReason[kN] = {0, 0, ncclDiagP2pReasonIndirect, ncclDiagP2pReasonTopo, 0};
  for (int i = 0; i < kN * kN; i++) {
    const bool own = i / kN == kSelf;
    EXPECT_EQ(results[i].tested, own ? kTested[i % kN] : 0) << i;
    EXPECT_EQ(results[i].reason, own ? kReason[i % kN] : 0) << i;
    if (!own) {
      EXPECT_EQ(std::memcmp(&edges[i], &poison[i], sizeof(edges[i])), 0) << i;
    }
  }
  EXPECT_TRUE(LogHas(log,
                     " Diagnostics P2P skip srcRank=4 srcCudaDev=3 srcNvmlDev=14 dstRank=5 dstCudaDev=5 dstNvmlDev=15 "
                     "path=PIX handle=LEGACY_CUDA_IPC topoRead=0 reason=indirect\n"))
      << log;
  EXPECT_TRUE(LogHas(log, " Diagnostics P2P topo check failed srcRank=4 dstRank=2 result=2\n")) << log;
}

struct DiagP2pInboundScene {
  ncclDiagP2pEdgeInfo edges[kGroupN * kGroupN] = {};
  int inPeers[kGroupN] = {99, 99, 99, 99};
  int inPeerCount = 99;
  bool needsLocalHandle = true;
  void Run(ncclComm* comm, const int* ranks) {
    ncclDiagP2pBuildInboundPeers(comm, ranks, kGroupSelf, kGroupN, edges, inPeers, &inPeerCount, &needsLocalHandle);
  }
};

TEST_F(DiagP2pMicrotest, BuildInboundPeers_ListsSourcesInOwnColumn) {
  BuildGroupComm();
  DiagP2pInboundScene s;
  s.edges[InEdge(0)] = {1, 0, PATH_NVL, 1, ncclDiagP2pHandleDirect};
  s.edges[InEdge(3)] = {1, 0, PATH_NVL, 0, ncclDiagP2pHandleLegacyIpc};
  s.edges[OwnEdge(0)] = {1, 0, PATH_NVL, 1, ncclDiagP2pHandleDirect};
  s.Run(comm_.get(), localRanks_.data());
  EXPECT_EQ(s.inPeerCount, 2);
  EXPECT_EQ(s.inPeers[0], 0);
  EXPECT_EQ(s.inPeers[1], 3);
  EXPECT_FALSE(s.needsLocalHandle);
}

TEST_F(DiagP2pMicrotest, BuildInboundPeers_NeedsHandleOnlyForCuMemSameProcessOtherDevice) {
  BuildGroupComm();
  peers_[kGroupRank].cudaDev = 3;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  const auto run = [&](int sameProcess, bool expected) {
    DiagP2pInboundScene s;
    s.edges[InEdge(3)] = {1, 0, PATH_NVL, sameProcess, ncclDiagP2pHandleDirect};
    s.needsLocalHandle = !expected;
    s.Run(comm_.get(), localRanks_.data());
    EXPECT_EQ(s.inPeerCount, 1);
    return s.needsLocalHandle;
  };
  EXPECT_TRUE(run(1, true));
  EXPECT_FALSE(run(0, false));
  peers_[2].cudaDev = 3;
  EXPECT_FALSE(run(1, false));
}

// ROCm gap: CUDART_VERSION unset leaks the retained handle; a fix flips this.
TEST_F(DiagP2pMicrotest, ReleaseLocalHandle_IsInternalErrorWithoutReleasingOnRocm) {
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ncclDiagP2pMemDesc desc{};
  EXPECT_EQ(ncclDiagP2pReleaseLocalHandle(&desc), ncclInternalError);
  EXPECT_EQ(release.calls, 0);
}

struct DiagP2pMapScene {
  ncclDiagP2pMemDesc desc{};
  ncclDiagP2pMapping mapping{};
  ncclDiagP2pSlot peerSlots[2] = {};
  DiagP2pMapScene() {
    desc.valid = 1;
    desc.bytes = sizeof(peerSlots);
    desc.directPtr = reinterpret_cast<uintptr_t>(peerSlots);
    mapping.peerAccessDev = -3;
  }
};

TEST_F(DiagP2pMicrotest, MapSameProcess_SameDeviceUsesDirectPointerWithoutPeerAccess) {
  BuildComm(6, 4, {1, 4});
  comm_->cudaDev = 3;
  peers_[1].cudaDev = 3;
  DiagP2pMapScene s;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipSuccess; });
  EXPECT_EQ(ncclDiagP2pMapSameProcess(comm_.get(), 1, &s.desc, &s.mapping), ncclSuccess);
  EXPECT_EQ(s.mapping.ptr, static_cast<void*>(s.peerSlots));
  EXPECT_EQ(s.mapping.active, 1);
  EXPECT_EQ(s.mapping.peerAccessEnabled, 0);
  EXPECT_EQ(s.mapping.sameProcessCuMem, 0);
  EXPECT_EQ(enable.calls, 0);
  EXPECT_EQ(cuMem.calls, 0);
}

// ROCm gap: CUDART_VERSION unset fails every cuMem same-process import.
TEST_F(DiagP2pMicrotest, MapSameProcess_CuMemOtherDeviceIsInternalErrorOnRocm) {
  BuildComm(6, 4, {1, 4});
  DiagP2pMapScene s;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipSuccess; });
  EXPECT_EQ(ncclDiagP2pMapSameProcess(comm_.get(), 1, &s.desc, &s.mapping), ncclInternalError);
  EXPECT_EQ(s.mapping.ptr, nullptr);
  EXPECT_EQ(s.mapping.active, 0);
  EXPECT_EQ(s.mapping.sameProcessCuMem, 0);
  EXPECT_EQ(enable.calls, 0);
}

TEST_F(DiagP2pMicrotest, MapSameProcess_LegacyEnablesPeerAccessToDestinationDevice) {
  BuildComm(6, 4, {1, 4});
  peers_[1].cudaDev = 3;
  DiagP2pMapScene s;
  int enabledDev = -1;
  unsigned enabledFlags = 99;
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [&](int dev, unsigned flags) {
    enabledDev = dev;
    enabledFlags = flags;
    return hipSuccess;
  });
  EXPECT_EQ(ncclDiagP2pMapSameProcess(comm_.get(), 1, &s.desc, &s.mapping), ncclSuccess);
  EXPECT_EQ(enabledDev, 3);
  EXPECT_EQ(enabledFlags, 0u);
  EXPECT_EQ(s.mapping.ptr, static_cast<void*>(s.peerSlots));
  EXPECT_EQ(s.mapping.active, 1);
  EXPECT_EQ(s.mapping.peerAccessEnabled, 1);
  EXPECT_EQ(s.mapping.peerAccessDev, 3);
}

TEST_F(DiagP2pMicrotest, MapSameProcess_LegacyAlreadyEnabledMapsWithoutOwningPeerAccess) {
  BuildComm(6, 4, {1, 4});
  DiagP2pMapScene s;
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipErrorPeerAccessAlreadyEnabled; });
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });
  EXPECT_EQ(ncclDiagP2pMapSameProcess(comm_.get(), 1, &s.desc, &s.mapping), ncclSuccess);
  EXPECT_EQ(lastError.calls, 1);
  EXPECT_EQ(s.mapping.ptr, static_cast<void*>(s.peerSlots));
  EXPECT_EQ(s.mapping.active, 1);
  EXPECT_EQ(s.mapping.peerAccessEnabled, 0);
  EXPECT_EQ(s.mapping.peerAccessDev, -3);
}

TEST_F(DiagP2pMicrotest, MapSameProcess_LegacyEnableFailureLeavesMappingInactive) {
  BuildComm(6, 4, {1, 4});
  peers_[1].cudaDev = 3;
  DiagP2pMapScene s;
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipErrorInvalidDevice; });
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  ncclResult_t ret = ncclSuccess;
  const std::string log = CaptureLog([&] { ret = ncclDiagP2pMapSameProcess(comm_.get(), 1, &s.desc, &s.mapping); });
  EXPECT_EQ(ret, ncclUnhandledCudaError);
  EXPECT_EQ(lastError.calls, 1);
  EXPECT_EQ(s.mapping.ptr, nullptr);
  EXPECT_EQ(s.mapping.active, 0);
  EXPECT_EQ(s.mapping.peerAccessEnabled, 0);
  EXPECT_TRUE(LogHas(log, " Diagnostics: failed to enable peer access to dev 3: [hip_fake] stub error\n")) << log;
}

TEST_F(DiagP2pMicrotest, FreeMapping_NullMappingIsNoOp) {
  BuildComm(2, 0, {0, 1});
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), nullptr), ncclSuccess);
}

TEST_F(DiagP2pMicrotest, FreeMapping_ReleasesByMappingKindThenClears) {
  BuildComm(2, 0, {0, 1});
  int buffer = 0;
  std::vector<void*> cuMemFreed;
  std::vector<void*> ipcClosed;
  ScopedHook cuMemFree(g_diagCuMemFreeAddr, [&](void* p, ncclMemManager* manager, int numSegments) {
    EXPECT_EQ(manager, nullptr);
    EXPECT_EQ(numSegments, 1);
    cuMemFreed.push_back(p);
    return ncclSuccess;
  });
  ScopedHook ipcClose(g_hipIpcCloseMemHandle, [&](void* p) {
    ipcClosed.push_back(p);
    return hipSuccess;
  });
  const auto run = [&](int sameCuMem, int crossCuMem, int legacy, int tracked) {
    ncclDiagP2pMapping m = {&buffer, 1, sameCuMem, crossCuMem, legacy, tracked, 0, 0};
    EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &m), ncclSuccess);
    EXPECT_EQ(m.ptr, nullptr);
    EXPECT_EQ(m.active, 0);
  };
  run(1, 1, 1, 1);
  EXPECT_EQ(cuMemFreed, std::vector<void*>{&buffer});
  run(0, 1, 1, 1);
  run(0, 1, 0, 0);
  ASSERT_EQ(g_diagDeviceFrees.size(), 2u);
  EXPECT_EQ(g_diagDeviceFrees[0].ptr, &buffer);
  EXPECT_EQ(g_diagDeviceFrees[0].manager, comm_->memManager);
  EXPECT_EQ(g_diagDeviceFrees[1].ptr, &buffer);
  EXPECT_EQ(g_diagDeviceFrees[1].manager, nullptr);
  EXPECT_TRUE(ipcClosed.empty());
  run(0, 0, 1, 0);
  EXPECT_EQ(ipcClosed, std::vector<void*>{&buffer});
  run(0, 0, 0, 0);
  EXPECT_EQ(cuMemFreed.size(), 1u);
  EXPECT_EQ(g_diagDeviceFrees.size(), 2u);
  EXPECT_EQ(ipcClosed.size(), 1u);
}

TEST_F(DiagP2pMicrotest, FreeMapping_InactiveOrNullPointerFreesNothing) {
  BuildComm(2, 0, {0, 1});
  int buffer = 0;
  ScopedHook ipcClose(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });
  ncclDiagP2pMapping inactive = {&buffer, 0, 0, 0, 1, 0, 0, 0};
  ncclDiagP2pMapping nullPtr = {nullptr, 1, 0, 0, 1, 0, 0, 0};
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &inactive), ncclSuccess);
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &nullPtr), ncclSuccess);
  EXPECT_EQ(ipcClose.calls, 0);
  EXPECT_EQ(inactive.ptr, nullptr);
  EXPECT_EQ(nullPtr.active, 0);
}

TEST_F(DiagP2pMicrotest, FreeMapping_DisablesPeerAccessItEnabled) {
  BuildComm(2, 0, {0, 1});
  std::vector<int> disabled;
  hipError_t disableErr = hipSuccess;
  ScopedHook disable(g_hipDeviceDisablePeerAccess, [&](int dev) {
    disabled.push_back(dev);
    return disableErr;
  });
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });
  ncclDiagP2pMapping m = {nullptr, 0, 0, 0, 0, 0, 1, 6};
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &m), ncclSuccess);
  EXPECT_EQ(m.peerAccessEnabled, 0);
  EXPECT_EQ(lastError.calls, 0);
  disableErr = hipErrorPeerAccessNotEnabled;
  m.peerAccessEnabled = 1;
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &m), ncclSuccess);
  EXPECT_EQ(lastError.calls, 1);
  disableErr = hipErrorInvalidDevice;
  m.peerAccessEnabled = 1;
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &m), ncclUnhandledCudaError);
  EXPECT_EQ(lastError.calls, 2);
  EXPECT_EQ(disabled, (std::vector<int>{6, 6, 6}));
  m.peerAccessEnabled = 0;
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &m), ncclSuccess);
  EXPECT_EQ(disable.calls, 3);
}

TEST_F(DiagP2pMicrotest, FreeMapping_FirstFailureWinsAndCleanupContinues) {
  BuildComm(2, 0, {0, 1});
  int buffer = 0;
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });
  ScopedHook disable(g_hipDeviceDisablePeerAccess, [](int) { return hipErrorInvalidDevice; });
  ScopedHook cuMemFree(g_diagCuMemFreeAddr, [](void*, ncclMemManager*, int) { return ncclSystemError; });
  ncclDiagP2pMapping cuMem = {&buffer, 1, 1, 0, 0, 0, 1, 2};
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &cuMem), ncclSystemError);
  EXPECT_EQ(disable.calls, 1);
  EXPECT_EQ(cuMem.peerAccessEnabled, 0);
  ScopedHook ipcClose(g_hipIpcCloseMemHandle, [](void*) { return hipErrorInvalidValue; });
  ncclDiagP2pMapping legacy = {&buffer, 1, 0, 0, 1, 0, 0, 0};
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &legacy), ncclUnhandledCudaError);
  EXPECT_EQ(lastError.calls, 2);
  EXPECT_EQ(legacy.ptr, nullptr);
  ScopedHook cudaFree(g_diagCudaFree, [](void*, ncclMemManager*) { return ncclInvalidUsage; });
  ncclDiagP2pMapping crossCuMem = {&buffer, 1, 0, 1, 0, 1, 1, 3};
  EXPECT_EQ(ncclDiagP2pFreeMapping(comm_.get(), &crossCuMem), ncclInvalidUsage);
  EXPECT_EQ(cudaFree.calls, 1);
  EXPECT_EQ(disable.calls, 2);
}

struct DiagP2pImportScene {
  ncclDiagP2pEdgeInfo edges[kGroupN * kGroupN] = {};
  ncclDiagP2pMemDesc memDescs[kGroupN] = {};
  ncclDiagP2pMapping mappings[kGroupN] = {};
  ncclDiagP2pEdgeResult results[kGroupN * kGroupN] = {};
  ncclDiagP2pSlot peerSlots[kGroupN][kGroupN] = {};
  DiagP2pImportScene() {
    edges[OwnEdge(0)] = {1, 0, PATH_NVL, 1, ncclDiagP2pHandleDirect};
    edges[OwnEdge(2)] = {1, 0, PATH_PIX, 0, ncclDiagP2pHandleLegacyIpc};
    edges[OwnEdge(3)] = {1, 0, PATH_SYS, 0, ncclDiagP2pHandleLegacyIpc};
    for (int slot : kGroupOutPeers) {
      memDescs[slot] = {1, sizeof(peerSlots[slot]), reinterpret_cast<uintptr_t>(peerSlots[slot]), {}};
      results[OwnEdge(slot)].tested = 1;
    }
  }
  int Reason(int slot) const {
    return results[OwnEdge(slot)].reason;
  }
  void Run(ncclComm* comm, const int* ranks, bool cudaUsable, hipStream_t stream) {
    ncclDiagP2pImportMappings(comm, ranks, kGroupSelf, kGroupN, kGroupOutPeerCount, kGroupOutPeers, edges, memDescs,
                              mappings, results, cudaUsable, stream);
  }
};

constexpr char kImportFields[] = " Diagnostics P2P import srcRank=4 srcCudaDev=4 srcNvmlDev=14 dstRank=";

TEST_F(DiagP2pMicrotest, ImportMappings_RoutesByProcessAndRecordsMappingKind) {
  BuildGroupComm();
  peers_[1].cudaDev = 4;
  DiagP2pImportScene s;
  s.memDescs[3].valid = 0;
  std::vector<int> peers;
  ScopedHook importHook(g_ncclP2pImportShareableBuffer, [&](ncclComm* comm, int peer, size_t size, ncclIpcDesc* desc,
                                                            void** ptr, void* owner, ncclMemType_t type) {
    EXPECT_EQ(comm, comm_.get());
    EXPECT_EQ(size, sizeof(s.peerSlots[2]));
    EXPECT_EQ(desc, &s.memDescs[2].ipcDesc);
    EXPECT_EQ(owner, static_cast<void*>(s.peerSlots[2]));
    EXPECT_EQ(type, ncclMemScratch);
    peers.push_back(peer);
    *ptr = s.peerSlots[3];
    return ncclSuccess;
  });
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  const std::string log = CaptureLog([&] { s.Run(comm_.get(), localRanks_.data(), true, kStream); });
  EXPECT_EQ(peers, std::vector<int>{5});
  EXPECT_EQ(s.Reason(0), ncclDiagP2pReasonNone);
  EXPECT_EQ(s.Reason(2), ncclDiagP2pReasonNone);
  EXPECT_EQ(s.Reason(3), ncclDiagP2pReasonNoDescriptor);
  EXPECT_EQ(s.mappings[0].ptr, static_cast<void*>(s.peerSlots[0]));
  EXPECT_EQ(s.mappings[0].active, 1);
  EXPECT_EQ(s.mappings[0].legacyIpc, 0);
  EXPECT_EQ(s.mappings[2].ptr, static_cast<void*>(s.peerSlots[3]));
  EXPECT_EQ(s.mappings[2].active, 1);
  EXPECT_EQ(s.mappings[2].legacyIpc, 1);
  EXPECT_EQ(s.mappings[2].crossProcessCuMem, 0);
  EXPECT_EQ(s.mappings[3].active, 0);
  const std::string tail = " path=SYS handle=LEGACY_CUDA_IPC topoRead=0 import=0 reason=noDescriptor\n";
  EXPECT_TRUE(LogHas(log, (std::string(kImportFields) +
                           "1 dstCudaDev=4 dstNvmlDev=11 path=XGMI handle=DIRECT topoRead=0 import=1\n")
                              .c_str()))
      << log;
  EXPECT_TRUE(LogHas(log, (std::string(kImportFields) +
                           "5 dstCudaDev=5 dstNvmlDev=15 path=PIX handle=LEGACY_CUDA_IPC topoRead=0 import=1\n")
                              .c_str()))
      << log;
  EXPECT_TRUE(LogHas(log, (std::string(kImportFields) + "2 dstCudaDev=2 dstNvmlDev=12" + tail).c_str())) << log;
}

TEST_F(DiagP2pMicrotest, ImportMappings_CuMemTracksOnlySuccessfulImportsAndFlagsPartialFailure) {
  BuildGroupComm();
  peers_[1].cudaDev = 4;
  DiagP2pImportScene s;
  s.mappings[3].tracked = 1;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook importHook(g_ncclP2pImportShareableBuffer,
                        [&](ncclComm*, int peer, size_t, ncclIpcDesc*, void** ptr, void*, ncclMemType_t) {
                          *ptr = s.peerSlots[1];
                          return peer == 2 ? ncclSystemError : ncclSuccess;
                        });
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  const std::string log = CaptureLog([&] { s.Run(comm_.get(), localRanks_.data(), true, kStream); });
  EXPECT_EQ(s.Reason(2), ncclDiagP2pReasonNone);
  EXPECT_EQ(s.Reason(3), ncclDiagP2pReasonImport);
  for (int slot : {2, 3}) {
    EXPECT_EQ(s.mappings[slot].active, 1) << slot;
    EXPECT_EQ(s.mappings[slot].crossProcessCuMem, 1) << slot;
    EXPECT_EQ(s.mappings[slot].legacyIpc, 0) << slot;
  }
  EXPECT_EQ(s.mappings[2].tracked, 1);
  EXPECT_EQ(s.mappings[3].tracked, 0);
  EXPECT_TRUE(LogHas(log, (std::string(kImportFields) +
                           "2 dstCudaDev=2 dstNvmlDev=12 path=SYS handle=LEGACY_CUDA_IPC topoRead=0 import=0 "
                           "reason=import\n")
                              .c_str()))
      << log;
}

TEST_F(DiagP2pMicrotest, ImportMappings_FailedOrEmptyImportIsImportReason) {
  BuildGroupComm();
  DiagP2pImportScene s;
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipErrorInvalidDevice; });
  ScopedHook importHook(g_ncclP2pImportShareableBuffer,
                        [&](ncclComm*, int peer, size_t, ncclIpcDesc*, void**, void*, ncclMemType_t) {
                          return peer == 2 ? ncclSystemError : ncclSuccess;
                        });
  s.Run(comm_.get(), localRanks_.data(), true, kStream);
  for (int slot : kGroupOutPeers) {
    EXPECT_EQ(s.Reason(slot), ncclDiagP2pReasonImport) << slot;
    EXPECT_EQ(s.mappings[slot].active, 0) << slot;
  }
  EXPECT_EQ(importHook.calls, 2);
}

TEST_F(DiagP2pMicrotest, ImportMappings_LocalCudaFailureSkipsEveryImport) {
  BuildGroupComm();
  ScopedHook importHook(g_ncclP2pImportShareableBuffer,
                        [](ncclComm*, int, size_t, ncclIpcDesc*, void**, void*, ncclMemType_t) { return ncclSuccess; });
  for (const bool usable : {false, true}) {
    DiagP2pImportScene s;
    s.results[OwnEdge(3)].reason = ncclDiagP2pReasonTopo;
    s.memDescs[2].valid = 0;
    ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
    const std::string log =
        CaptureLog([&] { s.Run(comm_.get(), localRanks_.data(), usable, usable ? nullptr : kStream); });
    EXPECT_EQ(s.Reason(0), ncclDiagP2pReasonLocalCuda);
    EXPECT_EQ(s.Reason(2), ncclDiagP2pReasonLocalCuda);
    EXPECT_EQ(s.Reason(3), ncclDiagP2pReasonTopo);
    EXPECT_TRUE(LogHas(log, (std::string(kImportFields) +
                             "5 dstCudaDev=5 dstNvmlDev=15 path=PIX handle=LEGACY_CUDA_IPC topoRead=0 import=0 "
                             "reason=localCuda\n")
                                .c_str()))
        << log;
  }
  EXPECT_EQ(importHook.calls, 0);
}

struct DiagP2pRemoteOpsScene {
  ncclDiagP2pEdgeResult results[kGroupN * kGroupN] = {};
  ncclDiagP2pMapping mappings[kGroupN] = {};
  ncclDiagP2pSlot slotsA[kGroupN] = {};
  ncclDiagP2pSlot slotsB[kGroupN] = {};
  ncclDiagP2pRemoteOp* opsHost = nullptr;
  ncclDiagP2pRemoteOp* opsDev = nullptr;
  int opCount = -1;
  DiagP2pRemoteOpsScene() {
    results[OwnEdge(2)].reason = ncclDiagP2pReasonImport;
    mappings[0].ptr = slotsA;
    mappings[3].ptr = slotsB;
  }
  ~DiagP2pRemoteOpsScene() {
    std::free(opsHost);
  }
  bool Run(ncclComm* comm, const int* ranks) {
    return ncclDiagP2pPrepareRemoteOps(comm, ranks, kGroupSelf, kGroupN, kGroupOutPeerCount, kGroupOutPeers, results,
                                       mappings, &opsHost, &opsDev, &opCount, kStream);
  }
};

TEST_F(DiagP2pMicrotest, PrepareRemoteOps_BuildsOneOpPerImportedPeerAndCopiesToDevice) {
  BuildGroupComm();
  DiagP2pRemoteOpsScene s;
  hipStream_t copyStream = nullptr;
  ScopedHook copy(g_hipMemcpyAsync,
                  [&](void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
                    EXPECT_EQ(kind, hipMemcpyHostToDevice);
                    copyStream = stream;
                    return HonestMemcpyAsync(dst, src, bytes, kind, stream);
                  });
  ScopedHook devAlloc(g_diagCudaCalloc,
                      [&](void** ptr, std::size_t bytes, ncclMemManager* manager, ncclMemType_t memType) {
                        EXPECT_EQ(bytes, 2 * sizeof(ncclDiagP2pRemoteOp));
                        EXPECT_EQ(manager, comm_->memManager);
                        EXPECT_EQ(memType, ncclMemScratch);
                        return DefaultDiagCudaCalloc(ptr, bytes, manager, memType);
                      });
  ASSERT_TRUE(s.Run(comm_.get(), localRanks_.data()));
  ASSERT_EQ(s.opCount, 2);
  EXPECT_EQ(s.opsHost[0].remoteSlots, s.slotsA);
  EXPECT_EQ(s.opsHost[1].remoteSlots, s.slotsB);
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(s.opsHost[i].srcRank, 4);
    EXPECT_EQ(s.opsHost[i].srcSlot, kGroupSelf);
  }
  EXPECT_EQ(s.opsHost[0].dstRank, 1);
  EXPECT_EQ(s.opsHost[1].dstRank, 2);
  ASSERT_EQ(g_hipMemcpyAsyncArgs.size(), 1u);
  EXPECT_EQ(g_hipMemcpyAsyncArgs[0].dst, s.opsDev);
  EXPECT_EQ(g_hipMemcpyAsyncArgs[0].src, s.opsHost);
  EXPECT_EQ(g_hipMemcpyAsyncArgs[0].bytes, 2 * sizeof(ncclDiagP2pRemoteOp));
  EXPECT_EQ(copyStream, kStream);
  EXPECT_EQ(std::memcmp(s.opsDev, s.opsHost, 2 * sizeof(ncclDiagP2pRemoteOp)), 0);
  EXPECT_EQ(devAlloc.calls, 1);
}

TEST_F(DiagP2pMicrotest, PrepareRemoteOps_NothingImportedAllocatesNothing) {
  BuildGroupComm();
  DiagP2pRemoteOpsScene s;
  for (int slot : kGroupOutPeers) {
    s.results[OwnEdge(slot)].reason = ncclDiagP2pReasonImport;
  }
  EXPECT_TRUE(s.Run(comm_.get(), localRanks_.data()));
  EXPECT_EQ(s.opCount, 0);
  EXPECT_EQ(s.opsHost, nullptr);
  EXPECT_EQ(s.opsDev, nullptr);
  EXPECT_TRUE(g_diagDeviceLive.empty());
  EXPECT_EQ(g_hipMemcpyAsyncCalls, 0);
}

TEST_F(DiagP2pMicrotest, PrepareRemoteOps_EachFailureReturnsFalseWithItsWarning) {
  BuildGroupComm();
  ScopedHook copy(g_hipMemcpyAsync, HonestMemcpyAsync);
  {
    DiagP2pRemoteOpsScene s;
    FailCallocOf(2 * sizeof(ncclDiagP2pRemoteOp), 1);
    bool ok = true;
    const std::string log = CaptureLog([&] { ok = s.Run(comm_.get(), localRanks_.data()); });
    EXPECT_FALSE(ok);
    EXPECT_TRUE(g_diagDeviceLive.empty());
    EXPECT_TRUE(LogHas(log, " Diagnostics P2P allocate remote op descriptors returned 2\n")) << log;
  }
  {
    DiagP2pRemoteOpsScene s;
    ScopedHook devAlloc(g_diagCudaCalloc,
                        [](void**, std::size_t, ncclMemManager*, ncclMemType_t) { return ncclSystemError; });
    bool ok = true;
    const std::string log = CaptureLog([&] { ok = s.Run(comm_.get(), localRanks_.data()); });
    EXPECT_FALSE(ok);
    EXPECT_EQ(copy.calls, 0);
    EXPECT_NE(s.opsHost, nullptr);
    EXPECT_EQ(s.opCount, 0);
    EXPECT_TRUE(LogHas(log, " Diagnostics P2P allocate remote op descriptors on device returned 2\n")) << log;
  }
  {
    DiagP2pRemoteOpsScene s;
    ScopedHook failCopy(g_hipMemcpyAsync,
                        [](void*, const void*, size_t, hipMemcpyKind, hipStream_t) { return hipErrorInvalidValue; });
    bool ok = true;
    const std::string log = CaptureLog([&] { ok = s.Run(comm_.get(), localRanks_.data()); });
    EXPECT_FALSE(ok);
    EXPECT_EQ(s.opCount, 2);
    EXPECT_EQ(failCopy.calls, 1);
    EXPECT_TRUE(LogHas(log, " Diagnostics P2P copy remote op descriptors CUDA failure: [hip_fake] stub error\n"))
        << log;
  }
}

template <typename Hook>
using HookOf = ScopedHook<rccl_test_host::FnSigOf_t<Hook>>;

constexpr int kSavedDevice = 7;

// Declared in call order: a stage's ordinal is the number of collectives reached.
enum class DiagP2pStage { kNone, kSetup, kEdges, kDescs, kWrote, kObs, kResults, kSummary, kReported, kFreed };
enum class DiagP2pLaunch { kNone, kInit, kWrite, kVerify, kRead };
// Stream syncs in happy-path order.
enum class DiagP2pSync { kNone, kInit, kWrite, kVerify, kRead, kCleanup };

bool IsDeviceBuffer(const void* ptr) {
  return std::count(g_diagDeviceLive.begin(), g_diagDeviceLive.end(), ptr) == 1;
}

int CountOf(const std::string& text, const std::string& needle) {
  int count = 0;
  for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + 1)) {
    count++;
  }
  return count;
}

// One real rank; the scene plays every peer's side of each collective.
struct DiagP2pRunScene {
  DiagP2pRunScene(ncclComm* runComm, std::vector<int> runRanks)
      : comm(runComm),
        ranks(std::move(runRanks)),
        n(static_cast<int>(ranks.size())),
        self(static_cast<int>(std::find(ranks.begin(), ranks.end(), comm->rank) - ranks.begin())),
        peerSlots(n, std::vector<ncclDiagP2pSlot>(n)),
        peerSetup(comm->nRanks, ncclSuccess),
        zeroObs(n, false) {
    for (int p = 0; p < n; p++) {
      for (int q = 0; q < n; q++) {
        peerSlots[p][q].readPattern = ncclDiagP2pReadPattern(ranks[p], ranks[q]);
      }
    }
  }

  // Every ncclDiagP2pRun exit path must release all it allocated; helper tests leave buffers to TearDown instead.
  ~DiagP2pRunScene() {
    FailCallocOf(0, 0);
    EXPECT_TRUE(g_diagDeviceLive.empty());
    EXPECT_TRUE(g_diagHostLive.empty());
  }

  ncclResult_t Run() {
    ncclResult_t ret = ncclNumResults;
    out = CaptureStdout([&] { ret = ncclDiagP2pRun(comm); });
    return ret;
  }

  std::vector<int> Reasons() const {
    std::vector<int> reasons;
    for (const auto& r : myResults) {
      reasons.push_back(r.reason);
    }
    return reasons;
  }

  // Every outbound edge carries `reason`; the self edge stays clean.
  std::vector<int> RowOf(int reason) const {
    std::vector<int> row(n, reason);
    row[self] = ncclDiagP2pReasonNone;
    return row;
  }

  ncclResult_t Collective(DiagP2pStage stage) {
    stages.push_back(stage);
    return stage == failStage ? ncclRemoteError : ncclSuccess;
  }

  ncclResult_t Gather(void* buf, int bytes) {
    if (bytes == sizeof(ncclResult_t)) {
      auto* results = static_cast<ncclResult_t*>(buf);
      mySetup = results[comm->rank];
      for (int r = 0; r < comm->nRanks; r++) {
        if (r != comm->rank) {
          results[r] = peerSetup[r];
        }
      }
      return Collective(DiagP2pStage::kSetup);
    }
    EXPECT_EQ(bytes, static_cast<int>(sizeof(ncclDiagP2pSummary)));
    mySummary = static_cast<ncclDiagP2pSummary*>(buf)[comm->rank];
    return Collective(DiagP2pStage::kSummary);
  }

  static DiagP2pStage NextIntraGather(DiagP2pStage last) {
    switch (last) {
    case DiagP2pStage::kSetup:
      return DiagP2pStage::kEdges;
    case DiagP2pStage::kEdges:
      return DiagP2pStage::kDescs;
    case DiagP2pStage::kWrote:
      return DiagP2pStage::kObs;
    case DiagP2pStage::kObs:
      return DiagP2pStage::kResults;
    default:
      ADD_FAILURE() << "unexpected intra-node gather";
      return DiagP2pStage::kNone;
    }
  }

  template <typename T>
  T* Mine(DiagP2pStage want, DiagP2pStage stage, void* buf, int bytes, int perRank, std::vector<T>* mine) {
    if (stage != want) {
      return nullptr;
    }
    if (bytes != static_cast<int>(perRank * sizeof(T))) {
      ADD_FAILURE() << "unexpected gather size " << bytes;
      return nullptr;
    }
    T* all = static_cast<T*>(buf);
    mine->assign(all + self * perRank, all + (self + 1) * perRank);
    return all;
  }

  ncclResult_t IntraGather(int* gatherRanks, int gatherSelf, int gatherN, void* buf, int bytes) {
    rankSet = gatherRanks;
    EXPECT_EQ(std::vector<int>(gatherRanks, gatherRanks + gatherN), ranks);
    EXPECT_EQ(gatherSelf, self);
    EXPECT_FALSE(stages.empty());
    if (stages.empty()) {
      return ncclInternalError;
    }
    const DiagP2pStage stage = NextIntraGather(stages.back());
    std::vector<ncclDiagP2pMemDesc> desc;
    auto* edges = Mine(DiagP2pStage::kEdges, stage, buf, bytes, n, &myEdges);
    auto* descs = Mine(DiagP2pStage::kDescs, stage, buf, bytes, 1, &desc);
    auto* obs = Mine(DiagP2pStage::kObs, stage, buf, bytes, n, &myObs);
    auto* results = Mine(DiagP2pStage::kResults, stage, buf, bytes, n, &myResults);
    if (descs != nullptr) {
      myDesc = desc[0];
    }
    for (int p = 0; p < n; p++) {
      if (descs != nullptr && p != self) {
        descs[p] = {1, n * sizeof(ncclDiagP2pSlot), reinterpret_cast<uintptr_t>(peerSlots[p].data()), {}};
        std::memcpy(&descs[p].ipcDesc, &descs[p].directPtr, sizeof(descs[p].directPtr));
      }
      if (p == self) {
        continue;
      }
      for (int q = 0; q < n; q++) {
        const uint64_t seen = zeroObs[p] ? 0 : peerSlots[p][q].writeValue;
        if (edges != nullptr) {
          edges[p * n + q] = {p != q && (inboundP2p || q != self), 0, PATH_DIS,
                              ncclDiagP2pSameProcess(comm, ranks[p], ranks[q]), ncclDiagP2pHandleLegacyIpc};
        } else if (obs != nullptr) {
          obs[p * n + q] = {p == writeFlipPeer ? seen ^ 2 : seen, p == verifyFlipPeer ? seen ^ 1 : seen};
        } else if (results != nullptr) {
          results[p * n + q] = {p != q, ncclDiagP2pReasonNone, 0, 0, 0};
        }
      }
    }
    return Collective(stage);
  }

  // Peers write into this rank's slots before the barrier releases it.
  ncclResult_t IntraBarrier(void* bootstrap, int* barrierRanks, int barrierSelf, int barrierN, int tag) {
    EXPECT_EQ(bootstrap, comm->bootstrap);
    EXPECT_EQ(std::vector<int>(barrierRanks, barrierRanks + barrierN), ranks);
    EXPECT_EQ(barrierSelf, self);
    if (tag != kDiagP2pBarrierWrote) {
      EXPECT_EQ(tag, kDiagP2pBarrierImportsFreed);
      closedAtFreed = ipcClose.calls;
      slotsLiveAtFreed = IsDeviceBuffer(reinterpret_cast<void*>(myDesc.directPtr));
      return Collective(DiagP2pStage::kFreed);
    }
    auto* slots = reinterpret_cast<ncclDiagP2pSlot*>(myDesc.directPtr);
    if (slots != nullptr) {
      mySlots.assign(slots, slots + n);
      for (int p = 0; p < n; p++) {
        if (p != self) {
          slots[p].writeValue = ncclDiagP2pWritePattern(ranks[p], comm->rank);
        }
      }
    }
    return Collective(DiagP2pStage::kWrote);
  }

  ncclComm* const comm;
  const std::vector<int> ranks;
  const int n;
  const int self;
  std::vector<std::vector<ncclDiagP2pSlot>> peerSlots;
  std::vector<ncclResult_t> peerSetup;
  std::vector<bool> zeroObs;

  DiagP2pStage failStage = DiagP2pStage::kNone;
  DiagP2pLaunch failLaunch = DiagP2pLaunch::kNone;
  DiagP2pSync failSync = DiagP2pSync::kNone;
  hipMemcpyKind failCopyKind = hipMemcpyDefault;
  size_t failCopyBytes = 0;
  size_t failDeviceAllocBytes = 0;
  int failDeviceFreeNth = 0;
  bool failShareableAlloc = false;
  bool failGetDevice = false;
  int failSetDeviceCall = 0;
  bool failStreamCreate = false;
  bool failStreamDestroy = false;
  bool failIpcClose = false;
  int savedDevice = kSavedDevice;
  int outboundP2p = 1;
  bool inboundP2p = true;
  uint64_t writeCorruption = 0;
  uint64_t verifyCorruption = 0;
  int writeFlipPeer = -1;
  int verifyFlipPeer = -1;

  std::vector<DiagP2pStage> stages;
  std::string out;
  ncclResult_t mySetup = ncclNumResults;
  ncclDiagP2pMemDesc myDesc{};
  std::vector<ncclDiagP2pEdgeInfo> myEdges;
  std::vector<ncclDiagP2pWriteObs> myObs;
  std::vector<ncclDiagP2pEdgeResult> myResults;
  std::vector<ncclDiagP2pSlot> mySlots;
  ncclDiagP2pSummary mySummary{};
  std::vector<int> setDevices;
  const int* rankSet = nullptr;
  int directMap = -1;
  int syncs = 0;
  int deviceFrees = 0;
  int closedAtFreed = -1;
  bool slotsLiveAtFreed = false;

  HookOf<decltype(g_devrBootstrapAllGather)> gather{
      g_devrBootstrapAllGather, [this](void* bootstrap, void* buf, int bytes) {
        EXPECT_EQ(bootstrap, comm->bootstrap);
        return Gather(buf, bytes);
      }};
  HookOf<decltype(g_devrBootstrapIntraNodeAllGather)> intraGather{
      g_devrBootstrapIntraNodeAllGather,
      [this](void* bootstrap, int* r, int s, int c, void* buf, int bytes) {
        EXPECT_EQ(bootstrap, comm->bootstrap);
        return IntraGather(r, s, c, buf, bytes);
      }};
  HookOf<decltype(g_devrBootstrapIntraNodeBarrier)> intraBarrier{
      g_devrBootstrapIntraNodeBarrier,
      [this](void* b, int* r, int s, int c, int tag) { return IntraBarrier(b, r, s, c, tag); }};
  HookOf<decltype(g_devrBootstrapBarrier)> barrier{
      g_devrBootstrapBarrier, [this](void* bootstrap, int rank, int nRanks, int tag) {
        EXPECT_EQ(bootstrap, comm->bootstrap);
        EXPECT_EQ(rank, comm->rank);
        EXPECT_EQ(nRanks, comm->nRanks);
        EXPECT_EQ(tag, kDiagP2pBarrierReported);
        return Collective(DiagP2pStage::kReported);
      }};
  HookOf<decltype(g_ncclTopoCheckP2p)> topo{
      g_ncclTopoCheckP2p, [this](int, int, int* p2p, int* read, int* inter, int*, int*) {
        *p2p = outboundP2p;
        *read = 0;
        *inter = -1;
        return ncclSuccess;
      }};
  HookOf<decltype(g_ncclP2pAllocateShareableBuffer)> alloc{
      g_ncclP2pAllocateShareableBuffer,
      [this](size_t size, int map, ncclIpcDesc*, void** ptr, int peer, ncclMemManager* manager, ncclMemType_t type) {
        EXPECT_EQ(peer, -1);
        EXPECT_EQ(manager, comm->memManager);
        EXPECT_EQ(type, ncclMemScratch);
        directMap = map;
        return failShareableAlloc ? ncclSystemError : g_diagCudaCalloc(ptr, size, manager, type);
      }};
  HookOf<decltype(g_diagCudaCalloc)> deviceAlloc{
      g_diagCudaCalloc, [this](void** ptr, std::size_t bytes, ncclMemManager* manager, ncclMemType_t type) {
        EXPECT_EQ(manager, comm->memManager);
        EXPECT_EQ(type, ncclMemScratch);
        if (failDeviceAllocBytes != 0 && bytes == failDeviceAllocBytes) {
          return ncclSystemError;
        }
        return DefaultDiagCudaCalloc(ptr, bytes, manager, type);
      }};
  HookOf<decltype(g_diagCudaFree)> deviceFree{
      g_diagCudaFree, [this](void* ptr, ncclMemManager* manager) {
        const ncclResult_t ret = DefaultDiagCudaFree(ptr, manager);
        return ++deviceFrees == failDeviceFreeNth ? ncclSystemError : ret;
      }};
  HookOf<decltype(g_hipGetDevice)> getDevice{
      g_hipGetDevice, [this](int* dev) {
        *dev = savedDevice;
        return failGetDevice ? hipErrorNoDevice : hipSuccess;
      }};
  HookOf<decltype(g_hipSetDevice)> setDevice{
      g_hipSetDevice, [this](int dev) {
        setDevices.push_back(dev);
        return static_cast<int>(setDevices.size()) == failSetDeviceCall ? hipErrorInvalidDevice : hipSuccess;
      }};
  HookOf<decltype(g_hipStreamCreateWithFlags)> streamCreate{
      g_hipStreamCreateWithFlags, [this](hipStream_t* stream, unsigned flags) {
        EXPECT_EQ(flags, hipStreamNonBlocking);
        *stream = failStreamCreate ? nullptr : kStream;
        return failStreamCreate ? hipErrorOutOfMemory : hipSuccess;
      }};
  HookOf<decltype(g_hipStreamDestroy)> streamDestroy{
      g_hipStreamDestroy, [this](hipStream_t stream) {
        EXPECT_EQ(stream, kStream);
        return failStreamDestroy ? hipErrorInvalidHandle : hipSuccess;
      }};
  HookOf<decltype(g_hipStreamSynchronize)> sync{
      g_hipStreamSynchronize, [this](hipStream_t stream) {
        EXPECT_EQ(stream, kStream);
        return ++syncs == static_cast<int>(failSync) ? hipErrorLaunchFailure : hipSuccess;
      }};
  HookOf<decltype(g_hipMemcpyAsync)> copy{
      g_hipMemcpyAsync, [this](void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
        EXPECT_EQ(stream, kStream);
        if (kind == failCopyKind && bytes == failCopyBytes) {
          return hipErrorInvalidValue;
        }
        return HonestMemcpyAsync(dst, src, bytes, kind, stream);
      }};
  HookOf<decltype(g_hipIpcCloseMemHandle)> ipcClose{
      g_hipIpcCloseMemHandle, [this](void*) { return failIpcClose ? hipErrorInvalidValue : hipSuccess; }};
  HookOf<decltype(g_ncclDiagP2pInitSlots)> initLaunch{
      g_ncclDiagP2pInitSlots, [this](ncclDiagP2pSlot* slots, const int* slotRanks, int count, int dst, hipStream_t s) {
        EXPECT_EQ(s, kStream);
        EXPECT_TRUE(IsDeviceBuffer(slots));
        EXPECT_TRUE(IsDeviceBuffer(slotRanks));
        return failLaunch == DiagP2pLaunch::kInit ? ncclUnhandledCudaError
                                                  : EmulateDiagP2pInitSlots(slots, slotRanks, count, dst, s);
      }};
  HookOf<decltype(g_ncclDiagP2pRemoteWrite)> writeLaunch{
      g_ncclDiagP2pRemoteWrite, [this](const ncclDiagP2pRemoteOp* ops, int count, hipStream_t s) {
        EXPECT_EQ(s, kStream);
        EXPECT_TRUE(IsDeviceBuffer(ops));
        EXPECT_GT(count, 0);
        if (failLaunch == DiagP2pLaunch::kWrite || count <= 0) {
          return ncclUnhandledCudaError;
        }
        const ncclResult_t ret = EmulateDiagP2pRemoteWrite(ops, count, s);
        ops[0].remoteSlots[ops[0].srcSlot].writeValue ^= writeCorruption;
        return ret;
      }};
  HookOf<decltype(g_ncclDiagP2pVerifyWrites)> verifyLaunch{
      g_ncclDiagP2pVerifyWrites, [this](ncclDiagP2pSlot* slots, int count, hipStream_t s) {
        EXPECT_EQ(s, kStream);
        EXPECT_TRUE(IsDeviceBuffer(slots));
        if (failLaunch == DiagP2pLaunch::kVerify) {
          return ncclUnhandledCudaError;
        }
        const ncclResult_t ret = EmulateDiagP2pVerifyWrites(slots, count, s);
        for (int i = 0; i < count; i++) {
          slots[i].verifyValue ^= verifyCorruption;
        }
        return ret;
      }};
  HookOf<decltype(g_ncclDiagP2pRemoteRead)> readLaunch{
      g_ncclDiagP2pRemoteRead, [this](const ncclDiagP2pRemoteOp* ops, int count, uint64_t* readback, hipStream_t s) {
        EXPECT_EQ(s, kStream);
        EXPECT_TRUE(IsDeviceBuffer(ops));
        EXPECT_TRUE(IsDeviceBuffer(readback));
        return failLaunch == DiagP2pLaunch::kRead ? ncclUnhandledCudaError
                                                  : EmulateDiagP2pRemoteRead(ops, count, readback, s);
      }};
};

TEST_F(DiagP2pMicrotest, Run_HappyPathOnRankZero_PrintsOkAndReleasesEverything) {
  BuildLeaderComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.out, DiagP2pOkLine(12));
  EXPECT_EQ(run.stages, (std::vector<DiagP2pStage>{DiagP2pStage::kSetup, DiagP2pStage::kEdges, DiagP2pStage::kDescs,
                                                   DiagP2pStage::kWrote, DiagP2pStage::kObs, DiagP2pStage::kResults,
                                                   DiagP2pStage::kSummary, DiagP2pStage::kReported,
                                                   DiagP2pStage::kFreed}));
  EXPECT_EQ(run.mySetup, ncclSuccess);
  EXPECT_EQ(run.mySummary.tested, 12u);
  EXPECT_EQ(run.mySummary.passed, 12u);
  EXPECT_EQ(run.mySummary.skipped, 0u);
  EXPECT_EQ(run.setDevices, (std::vector<int>{0, kSavedDevice}));
  EXPECT_EQ(run.streamDestroy.calls, 1);
  EXPECT_EQ(run.ipcClose.calls, run.n - 1);
  EXPECT_EQ(run.closedAtFreed, run.n - 1);
  EXPECT_TRUE(run.slotsLiveAtFreed);
  EXPECT_EQ(run.directMap, 0);
  ASSERT_EQ(g_diagDeviceFrees.size(), 4u);
  for (const auto& f : g_diagDeviceFrees) {
    EXPECT_EQ(f.manager, comm_->memManager);
  }
  EXPECT_EQ(std::count(g_diagHostFrees.begin(), g_diagHostFrees.end(), run.rankSet), 0);
}

TEST_F(DiagP2pMicrotest, Run_CrossClique_FreesTheRankSetItBuilt) {
  BuildComm(4, 2, {0, 1, 2, 3});
  comm_->p2pCrossClique = true;
  comm_->nvlDomainSize = 4;
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_NE(run.rankSet, localRanks_.data());
  EXPECT_EQ(std::count(g_diagHostFrees.begin(), g_diagHostFrees.end(), run.rankSet), 1);
}

TEST_F(DiagP2pMicrotest, Run_HappyPathOnOtherSlot_ContributesExactRowsAndLogsEachEdge) {
  BuildGroupComm();
  comm_->cudaDev = 3;
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.savedDevice = 3;
  ScopedDebugLogging debug(NCCL_LOG_INFO, NCCL_INIT);
  const std::string log = CaptureLog([&] { EXPECT_EQ(run.Run(), ncclSuccess); });
  EXPECT_EQ(run.out, "");
  EXPECT_TRUE(run.setDevices.empty());
  EXPECT_EQ(run.myDesc.valid, 1);
  EXPECT_EQ(run.myDesc.bytes, kGroupN * sizeof(ncclDiagP2pSlot));
  EXPECT_EQ(run.mySummary.tested, 0u);
  EXPECT_EQ(run.ipcClose.calls, kGroupN - 1);
  const std::string fields = "srcRank=4 srcCudaDev=4 srcNvmlDev=14 dstRank=1 dstCudaDev=1 dstNvmlDev=11 path=DIS "
                             "handle=LEGACY_CUDA_IPC topoRead=0\n";
  EXPECT_EQ(CountOf(log, " Diagnostics P2P write "), 3) << log;
  EXPECT_EQ(CountOf(log, " Diagnostics P2P read "), 3) << log;
  EXPECT_TRUE(LogHas(log, (" Diagnostics P2P write " + fields).c_str())) << log;
  EXPECT_TRUE(LogHas(log, (" Diagnostics P2P read " + fields).c_str())) << log;
  ASSERT_EQ(run.myResults.size(), static_cast<size_t>(kGroupN));
  ASSERT_EQ(run.myEdges.size(), static_cast<size_t>(kGroupN));
  ASSERT_EQ(run.myObs.size(), static_cast<size_t>(kGroupN));
  ASSERT_EQ(run.mySlots.size(), static_cast<size_t>(kGroupN));
  for (int q = 0; q < kGroupN; q++) {
    const ncclDiagP2pEdgeResult& r = run.myResults[q];
    const bool peer = q != kGroupSelf;
    const uint64_t wrote = peer ? ncclDiagP2pWritePattern(kGroupRank, kGroupRanks[q]) : 0;
    const uint64_t seen = peer ? ncclDiagP2pWritePattern(kGroupRanks[q], kGroupRank) : 0;
    EXPECT_EQ(run.myEdges[q].p2p, peer) << q;
    EXPECT_EQ(r.tested, peer) << q;
    EXPECT_EQ(r.reason, ncclDiagP2pReasonNone) << q;
    EXPECT_EQ(r.writeGot, wrote) << q;
    EXPECT_EQ(r.verifyGot, wrote) << q;
    EXPECT_EQ(r.readGot, peer ? ncclDiagP2pReadPattern(kGroupRanks[q], kGroupRank) : 0) << q;
    EXPECT_EQ(run.peerSlots[q][kGroupSelf].writeValue, wrote) << q;
    EXPECT_EQ(run.myObs[q].writeValue, seen) << q;
    EXPECT_EQ(run.myObs[q].verifyValue, seen) << q;
    EXPECT_EQ(run.mySlots[q].readPattern, ncclDiagP2pReadPattern(kGroupRank, kGroupRanks[q])) << q;
    EXPECT_EQ(run.mySlots[q].writeValue, 0u) << q;
  }
}

TEST_F(DiagP2pMicrotest, Run_GroupLeaderOnNonZeroRank_StoresSummaryAtOwnRankSilently) {
  BuildComm(4, 3, {3, 0, 1, 2});
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.out, "");
  EXPECT_EQ(run.mySummary.tested, 12u);
  EXPECT_EQ(run.mySummary.passed, 12u);
}

TEST_F(DiagP2pMicrotest, Run_PeerSetupFailures_ReportedThenLastReturnedBeforeEdgeWork) {
  {
    BuildLeaderComm();
    DiagP2pRunScene run(comm_.get(), localRanks_);
    run.peerSetup[2] = ncclSystemError;
    run.peerSetup[3] = ncclInvalidUsage;
    EXPECT_EQ(run.Run(), ncclInvalidUsage);
    EXPECT_EQ(run.stages, std::vector<DiagP2pStage>{DiagP2pStage::kSetup});
    EXPECT_EQ(run.out, DiagLine("NCCL DIAG [INFO] p2p: setup failed on rank 2 result=2") +
                           DiagLine("NCCL DIAG [INFO] p2p: setup failed on rank 3 result=5"));
    EXPECT_EQ(run.streamDestroy.calls, 1);
  }
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.peerSetup[0] = ncclSystemError;
  EXPECT_EQ(run.Run(), ncclSystemError);
  EXPECT_EQ(run.out, "");
}

TEST_F(DiagP2pMicrotest, Run_LocalSetupFailure_IsSharedThroughAgreementGather) {
  BuildLeaderComm();
  comm_->p2pCrossClique = true;
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclInternalError);
  EXPECT_EQ(run.mySetup, ncclInternalError);
  EXPECT_EQ(run.stages, std::vector<DiagP2pStage>{DiagP2pStage::kSetup});
  EXPECT_EQ(run.out, DiagLine("NCCL DIAG [INFO] p2p: setup failed on rank 0 result=3"));
}

// Pins a hang: this failure skips the gather the other ranks wait in.
TEST_F(DiagP2pMicrotest, Run_SetupResultsCallocFailure_ReturnsWithoutAgreementGather) {
  BuildGroupComm();
  comm_->cudaDev = 3;
  DiagP2pRunScene run(comm_.get(), localRanks_);
  FailCallocOf(kGroupWorld * sizeof(ncclResult_t), 1);
  EXPECT_EQ(run.Run(), ncclSystemError);
  EXPECT_TRUE(run.stages.empty());
  EXPECT_EQ(run.streamDestroy.calls, 1);
  EXPECT_EQ(run.setDevices, (std::vector<int>{3, kSavedDevice}));
}

TEST_F(DiagP2pMicrotest, Run_CollectiveFailure_AbortsAndSkipsImportsFreedBarrier) {
  BuildGroupComm();
  for (DiagP2pStage stage : {DiagP2pStage::kSetup, DiagP2pStage::kEdges, DiagP2pStage::kDescs, DiagP2pStage::kWrote,
                             DiagP2pStage::kObs, DiagP2pStage::kResults, DiagP2pStage::kSummary,
                             DiagP2pStage::kReported}) {
    SCOPED_TRACE(static_cast<int>(stage));
    DiagP2pRunScene run(comm_.get(), localRanks_);
    run.failStage = stage;
    EXPECT_EQ(run.Run(), ncclRemoteError);
    EXPECT_EQ(run.stages.back(), stage);
    EXPECT_EQ(run.stages.size(), static_cast<size_t>(stage));
    EXPECT_EQ(run.ipcClose.calls, stage >= DiagP2pStage::kWrote ? kGroupN - 1 : 0);
    EXPECT_EQ(run.streamDestroy.calls, 1);
    EXPECT_EQ(run.setDevices, (std::vector<int>{kGroupRank, kSavedDevice}));
  }
}

struct DiagP2pFault {
  std::string name;
  std::function<void(DiagP2pRunScene*)> arm;
};

// Each fault runs in a fresh scene and must still mark every outbound edge `reason` and return `result`.
void ExpectEachFault(ncclComm* comm, const std::vector<int>& ranks, int reason, const std::vector<DiagP2pFault>& faults,
                     const std::function<void(const DiagP2pRunScene&)>& check = nullptr,
                     ncclResult_t result = ncclSuccess) {
  for (const auto& f : faults) {
    SCOPED_TRACE(f.name);
    DiagP2pRunScene run(comm, ranks);
    f.arm(&run);
    EXPECT_EQ(run.Run(), result);
    EXPECT_EQ(run.Reasons(), run.RowOf(reason));
    if (check) {
      check(run);
    }
  }
}

TEST_F(DiagP2pMicrotest, Run_LocalCudaSetupFaults_MarkOutboundEdgesLocalCuda) {
  BuildGroupComm();
  ExpectEachFault(
      comm_.get(), localRanks_, ncclDiagP2pReasonLocalCuda,
      {
          {"getDevice", [](DiagP2pRunScene* r) { r->failGetDevice = true; }},
          {"setDevice", [](DiagP2pRunScene* r) { r->failSetDeviceCall = 1; }},
          {"streamCreate", [](DiagP2pRunScene* r) { r->failStreamCreate = true; }},
          {"rankMapAlloc", [](DiagP2pRunScene* r) { r->failDeviceAllocBytes = r->n * sizeof(int); }},
          {"rankMapCopy",
           [](DiagP2pRunScene* r) {
             r->failCopyKind = hipMemcpyHostToDevice;
             r->failCopyBytes = r->n * sizeof(int);
           }},
          {"initLaunch", [](DiagP2pRunScene* r) { r->failLaunch = DiagP2pLaunch::kInit; }},
          {"initSync", [](DiagP2pRunScene* r) { r->failSync = DiagP2pSync::kInit; }},
      });
}

TEST_F(DiagP2pMicrotest, Run_LocalCudaSetupFault_SkipsStreamAndRestoresOnlyKnownDevice) {
  BuildGroupComm();
  {
    DiagP2pRunScene run(comm_.get(), localRanks_);
    run.failGetDevice = true;
    EXPECT_EQ(run.Run(), ncclSuccess);
    EXPECT_TRUE(run.setDevices.empty());
    EXPECT_EQ(run.streamCreate.calls, 0);
    EXPECT_EQ(run.myDesc.valid, 0);
    EXPECT_EQ(run.myDesc.bytes, kGroupN * sizeof(ncclDiagP2pSlot));
  }
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.failSetDeviceCall = 1;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.setDevices, (std::vector<int>{kGroupRank, kSavedDevice}));
  EXPECT_EQ(run.streamCreate.calls, 0);
  EXPECT_EQ(run.streamDestroy.calls, 0);
}

TEST_F(DiagP2pMicrotest, Run_ShareableAllocFailure_KeepsOutboundEdgesButPublishesNoBuffer) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.failShareableAlloc = true;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.Reasons(), run.RowOf(ncclDiagP2pReasonNone));
  EXPECT_EQ(run.myDesc.valid, 0);
  EXPECT_EQ(run.verifyLaunch.calls, 0);
  EXPECT_EQ(run.readLaunch.calls, 1);
}

TEST_F(DiagP2pMicrotest, Run_WritePhaseFaults_MarkWriteLaunch) {
  BuildGroupComm();
  ExpectEachFault(
      comm_.get(), localRanks_, ncclDiagP2pReasonWriteLaunch,
      {
          {"writeLaunch", [](DiagP2pRunScene* r) { r->failLaunch = DiagP2pLaunch::kWrite; }},
          {"writeSync", [](DiagP2pRunScene* r) { r->failSync = DiagP2pSync::kWrite; }},
          {"remoteOpsAlloc",
           [](DiagP2pRunScene* r) { r->failDeviceAllocBytes = (r->n - 1) * sizeof(ncclDiagP2pRemoteOp); }},
      });
}

// A local verify failure zeroes this rank's view; peers then blame their writes.
TEST_F(DiagP2pMicrotest, Run_VerifyFaults_ZeroOwnObservationsAndSkipRead) {
  BuildGroupComm();
  ExpectEachFault(
      comm_.get(), localRanks_, ncclDiagP2pReasonReadLaunch,
      {
          {"hostSlotsCalloc", [](DiagP2pRunScene* r) { FailCallocOf(r->n * sizeof(ncclDiagP2pSlot), 1); }},
          {"verifyLaunch", [](DiagP2pRunScene* r) { r->failLaunch = DiagP2pLaunch::kVerify; }},
          {"verifyCopy",
           [](DiagP2pRunScene* r) {
             r->failCopyKind = hipMemcpyDeviceToHost;
             r->failCopyBytes = r->n * sizeof(ncclDiagP2pSlot);
           }},
          {"verifySync", [](DiagP2pRunScene* r) { r->failSync = DiagP2pSync::kVerify; }},
      },
      [](const DiagP2pRunScene& run) {
        for (const auto& o : run.myObs) {
          EXPECT_EQ(o.writeValue, 0u);
          EXPECT_EQ(o.verifyValue, 0u);
        }
        EXPECT_EQ(run.readLaunch.calls, 0);
      });
}

TEST_F(DiagP2pMicrotest, Run_ReadPhaseFaults_MarkReadLaunch) {
  BuildGroupComm();
  ExpectEachFault(
      comm_.get(), localRanks_, ncclDiagP2pReasonReadLaunch,
      {
          {"readbackAlloc", [](DiagP2pRunScene* r) { r->failDeviceAllocBytes = (r->n - 1) * sizeof(uint64_t); }},
          // Ordinal 2: setupResults (kGroupWorld ncclResult_t) is also 24 bytes and is calloc'd first.
          {"readbackCalloc", [](DiagP2pRunScene* r) { FailCallocOf((r->n - 1) * sizeof(uint64_t), 2); }},
          {"readLaunch", [](DiagP2pRunScene* r) { r->failLaunch = DiagP2pLaunch::kRead; }},
          {"readCopy",
           [](DiagP2pRunScene* r) {
             r->failCopyKind = hipMemcpyDeviceToHost;
             r->failCopyBytes = (r->n - 1) * sizeof(uint64_t);
           }},
          {"readSync", [](DiagP2pRunScene* r) { r->failSync = DiagP2pSync::kRead; }},
      });
}

TEST_F(DiagP2pMicrotest, Run_WriteAndReadMismatches_ReportObservedValues) {
  BuildLeaderComm();
  // Shift every cudaDev off its rank so a cudaDev/rank swap in the report changes the text.
  for (auto& peer : peers_) {
    peer.cudaDev = (peer.rank + 1) % 4;
  }
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.writeCorruption = 0x10;
  run.peerSlots[3][0].readPattern = 0xbad;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.Reasons(), (std::vector<int>{0, ncclDiagP2pReasonWriteMismatch, 0, ncclDiagP2pReasonReadMismatch}));
  const std::string head = "srcRank=0 srcCudaDev=1 srcNvmlDev=10 dstRank=";
  const std::string tail = " path=DIS handle=LEGACY_CUDA_IPC";
  EXPECT_EQ(run.out,
            DiagLine("NCCL DIAG [INFO] p2p: 10/12 directed GPU P2P edges verified") +
                DiagLine("NCCL DIAG [INFO] p2p: write mismatch " + head + "3 dstCudaDev=0 dstNvmlDev=13" + tail +
                         " expected=0x4000000000000003 got=0x4000000000000013 verify=0x4000000000000013; " +
                         kGenericAdvice) +
                DiagLine("NCCL DIAG [INFO] p2p: read mismatch " + head + "2 dstCudaDev=3 dstNvmlDev=12" + tail +
                         " expected=0x8000000100000000 got=0x0000000000000bad; " + kGenericAdvice));
}

// Pins misattribution: a destination-local failure reads as a write mismatch.
TEST_F(DiagP2pMicrotest, Run_DestinationLocalFailure_IsBlamedOnSourceWrite) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.zeroObs[2] = true;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.peerSlots[2][kGroupSelf].writeValue, ncclDiagP2pWritePattern(kGroupRank, kGroupRanks[2]));
  EXPECT_EQ(run.Reasons(), (std::vector<int>{0, 0, ncclDiagP2pReasonWriteMismatch, 0}));
  ASSERT_EQ(run.myResults.size(), static_cast<size_t>(kGroupN));
  EXPECT_EQ(run.myResults[2].writeGot, 0u);
  EXPECT_EQ(run.myResults[2].verifyGot, 0u);
}

TEST_F(DiagP2pMicrotest, Run_CleanupFailures_PrintedAndReturnedOnlyWithoutRunError) {
  BuildLeaderComm();
  const auto printed = [](int result) {
    return [result](const DiagP2pRunScene& run) {
      EXPECT_EQ(run.out, DiagP2pOkLine(12) + DiagP2pCleanupLine(0, result));
      EXPECT_EQ(run.stages.back(), DiagP2pStage::kFreed);
      EXPECT_EQ(run.deviceFrees, 4);
    };
  };
  const int none = ncclDiagP2pReasonNone;
  ExpectEachFault(
      comm_.get(), localRanks_, none,
      {
          {"streamDestroy", [](DiagP2pRunScene* r) { r->failStreamDestroy = true; }},
          {"cleanupSync", [](DiagP2pRunScene* r) { r->failSync = DiagP2pSync::kCleanup; }},
          {"ipcClose", [](DiagP2pRunScene* r) { r->failIpcClose = true; }},
          {"restoreDevice", [](DiagP2pRunScene* r) { r->failSetDeviceCall = 2; }},
      },
      printed(ncclUnhandledCudaError), ncclUnhandledCudaError);
  ExpectEachFault(
      comm_.get(), localRanks_, none,
      {{"importsFreedBarrier", [](DiagP2pRunScene* r) { r->failStage = DiagP2pStage::kFreed; }}},
      printed(ncclRemoteError), ncclRemoteError);
  // Later stream and device faults must not overwrite the first cleanup error.
  std::vector<DiagP2pFault> deviceFrees;
  for (int nth = 1; nth <= 4; nth++) {
    deviceFrees.push_back({"deviceFree" + std::to_string(nth), [nth](DiagP2pRunScene* r) {
                             r->failDeviceFreeNth = nth;
                             r->failStreamDestroy = true;
                             r->failSetDeviceCall = 2;
                           }});
  }
  ExpectEachFault(comm_.get(), localRanks_, none, deviceFrees, printed(ncclSystemError), ncclSystemError);
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.failStreamDestroy = true;
  run.failStage = DiagP2pStage::kReported;
  EXPECT_EQ(run.Run(), ncclRemoteError);
  EXPECT_EQ(run.out, DiagP2pOkLine(12) + DiagP2pCleanupLine(0, ncclUnhandledCudaError));
}

// ROCm gaps: cuMem same-process import and handle release fail; fix flips this.
TEST_F(DiagP2pMicrotest, Run_CuMemSameProcessPeer_FailsImportAndCleanupOnRocm) {
  BuildLeaderComm();
  peers_[3].pidHash = peers_[0].pidHash;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclInternalError);
  EXPECT_EQ(run.directMap, 1);
  EXPECT_EQ(run.Reasons(), (std::vector<int>{0, ncclDiagP2pReasonImport, 0, 0}));
  EXPECT_EQ(run.out,
            DiagLine("NCCL DIAG [INFO] p2p: 11/12 directed GPU P2P edges verified") +
                DiagLine("NCCL DIAG [INFO] p2p: peer-memory import failed srcRank=0 srcCudaDev=0 srcNvmlDev=10 "
                         "dstRank=3 dstCudaDev=3 dstNvmlDev=13 path=DIS handle=DIRECT reason=import; inspect "
                         "preceding CUDA peer-access or virtual-memory mapping errors on the source rank") +
                DiagP2pCleanupLine(0, ncclInternalError));
}

TEST_F(DiagP2pMicrotest, Run_LegacySameProcessPeer_EnablesPeerAccessThenDisablesIt) {
  BuildLeaderComm();
  comm_->cudaDev = 5;
  peers_[2].cudaDev = 6;
  peers_[2].pidHash = peers_[0].pidHash;
  ScopedHook enable(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipSuccess; });
  std::vector<int> disabled;
  ScopedHook disable(g_hipDeviceDisablePeerAccess, [&](int dev) {
    disabled.push_back(dev);
    return hipSuccess;
  });
  DiagP2pRunScene run(comm_.get(), localRanks_);
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.Reasons(), run.RowOf(ncclDiagP2pReasonNone));
  EXPECT_EQ(disabled, std::vector<int>{6});
  EXPECT_EQ(run.ipcClose.calls, run.n - 2);
  EXPECT_EQ(run.out,
            DiagLine("NCCL DIAG [INFO] p2p: temporarily enabled context-wide CUDA peer access rank=0 cudaDev=5; "
                     "avoid concurrent CUDA use on this context until diagnostics completes") +
                DiagP2pOkLine(12));
}

TEST_F(DiagP2pMicrotest, Run_RankMapCopyFailure_WarnsAsSlotInitFailure) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.failCopyKind = hipMemcpyHostToDevice;
  run.failCopyBytes = kGroupN * sizeof(int);
  const std::string log = CaptureLog([&] { EXPECT_EQ(run.Run(), ncclSuccess); });
  EXPECT_TRUE(LogHas(log, " Diagnostics P2P copy P2P rank map CUDA failure: [hip_fake] stub error\n")) << log;
  EXPECT_TRUE(LogHas(log, " Diagnostics P2P initialize local slots returned 1\n")) << log;
  EXPECT_EQ(run.initLaunch.calls, 0);
}

TEST_F(DiagP2pMicrotest, Run_NoInboundPeers_PublishesEmptyDescriptorWithoutSlots) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.inboundP2p = false;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.Reasons(), run.RowOf(ncclDiagP2pReasonNone));
  EXPECT_EQ(run.myDesc.valid, 0);
  EXPECT_EQ(run.myDesc.bytes, 0u);
  EXPECT_EQ(run.alloc.calls, 0);
  EXPECT_EQ(run.verifyLaunch.calls, 0);
  for (const auto& o : run.myObs) {
    EXPECT_EQ(o.writeValue, 0u);
    EXPECT_EQ(o.verifyValue, 0u);
  }
}

TEST_F(DiagP2pMicrotest, Run_NoOutboundOrNoImports_LaunchesNothingRemote) {
  BuildGroupComm();
  {
    DiagP2pRunScene run(comm_.get(), localRanks_);
    run.outboundP2p = 0;
    EXPECT_EQ(run.Run(), ncclSuccess);
    for (const auto& r : run.myResults) {
      EXPECT_EQ(r.tested, 0);
    }
    EXPECT_EQ(run.writeLaunch.calls, 0);
    EXPECT_EQ(run.readLaunch.calls, 0);
  }
  DiagP2pRunScene run(comm_.get(), localRanks_);
  ScopedHook importHook(g_ncclP2pImportShareableBuffer, [](ncclComm*, int, size_t, ncclIpcDesc*, void**, void*,
                                                           ncclMemType_t) { return ncclSystemError; });
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.Reasons(), run.RowOf(ncclDiagP2pReasonImport));
  EXPECT_EQ(run.writeLaunch.calls, 0);
  EXPECT_EQ(run.readLaunch.calls, 0);
}

TEST_F(DiagP2pMicrotest, Run_EitherObservedValueWrong_IsWriteMismatch) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.verifyFlipPeer = 2;
  run.writeFlipPeer = 3;
  run.verifyCorruption = 4;
  EXPECT_EQ(run.Run(), ncclSuccess);
  ASSERT_EQ(run.myObs.size(), static_cast<size_t>(kGroupN));
  ASSERT_EQ(run.myResults.size(), static_cast<size_t>(kGroupN));
  for (int q = 0; q < kGroupN; q++) {
    const uint64_t seen = q == kGroupSelf ? 0 : ncclDiagP2pWritePattern(kGroupRanks[q], kGroupRank);
    EXPECT_EQ(run.myObs[q].writeValue, seen) << q;
    EXPECT_EQ(run.myObs[q].verifyValue, q == kGroupSelf ? 0 : seen ^ 4) << q;
  }
  const int kMismatch = ncclDiagP2pReasonWriteMismatch;
  EXPECT_EQ(run.Reasons(), (std::vector<int>{0, 0, kMismatch, kMismatch}));
  const uint64_t to2 = ncclDiagP2pWritePattern(kGroupRank, kGroupRanks[2]);
  const uint64_t to3 = ncclDiagP2pWritePattern(kGroupRank, kGroupRanks[3]);
  EXPECT_EQ(run.myResults[2].writeGot, to2);
  EXPECT_EQ(run.myResults[2].verifyGot, to2 ^ 1);
  EXPECT_EQ(run.myResults[3].writeGot, to3 ^ 2);
  EXPECT_EQ(run.myResults[3].verifyGot, to3);
}

TEST_F(DiagP2pMicrotest, Run_WriteSyncVsLaunchFailure_OnlySyncStopsLaterPhases) {
  BuildGroupComm();
  for (const bool syncFails : {true, false}) {
    SCOPED_TRACE(syncFails);
    DiagP2pRunScene run(comm_.get(), localRanks_);
    run.failSync = syncFails ? DiagP2pSync::kWrite : DiagP2pSync::kNone;
    run.failLaunch = syncFails ? DiagP2pLaunch::kNone : DiagP2pLaunch::kWrite;
    EXPECT_EQ(run.Run(), ncclSuccess);
    EXPECT_EQ(run.verifyLaunch.calls, syncFails ? 0 : 1);
    EXPECT_EQ(run.readLaunch.calls, syncFails ? 0 : 1);
    const int writeSync = static_cast<int>(DiagP2pSync::kWrite);
    EXPECT_EQ(run.syncs, syncFails ? writeSync + 1 : static_cast<int>(DiagP2pSync::kCleanup));
  }
}

TEST_F(DiagP2pMicrotest, Run_InitLaunchFailure_SyncsOnlyWritePhaseAndCleanup) {
  BuildGroupComm();
  DiagP2pRunScene run(comm_.get(), localRanks_);
  run.failLaunch = DiagP2pLaunch::kInit;
  EXPECT_EQ(run.Run(), ncclSuccess);
  EXPECT_EQ(run.syncs, 2);
}

}  // namespace
