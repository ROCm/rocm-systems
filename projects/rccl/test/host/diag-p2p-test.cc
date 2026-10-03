/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/diagnostics/p2p.cc, #include-d via DIAG_P2P_CC_PATH to reach its file-static helpers.
 * ncclCalloc is DiagCalloc around that include: it fails the Nth calloc of a size or pads a zeroed guard element.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
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
template <typename T>
static ncclResult_t DiagCalloc(const char* file, int line, const char* fn, T** ptr, std::size_t nelem) {
  if (g_diagCallocFailBytes != 0 && nelem * sizeof(T) == g_diagCallocFailBytes && --g_diagCallocFailNth == 0) {
    g_diagCallocFailBytes = 0;
    return ncclSystemError;
  }
  return ncclCallocDebug(ptr, g_diagCallocPad && nelem != 0 ? nelem + 1 : nelem, file, line, fn, true);
}

#undef ncclCalloc
#define ncclCalloc(...) DiagCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

#include DIAG_P2P_CC_PATH

#undef ncclCalloc

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

ncclDiagP2pEdgeInfo Edge(int pathType, int handleType) {
  ncclDiagP2pEdgeInfo edge{};
  edge.pathType = pathType;
  edge.handleType = handleType;
  return edge;
}

class DiagP2pMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    g_diagCallocFailBytes = 0;
    g_diagCallocFailNth = 0;
    g_diagCallocPad = false;
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

}  // namespace
