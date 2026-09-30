/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the remediation text of src/diagnostics/p2p.cc (NCCL_RUN_DIAGNOSTICS).

#include <algorithm>
#include <cctype>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

// graph/topo.cc owns topoPathTypeStr and is not part of this binary; the report under test only reads it,
// so this TU supplies its own copy under a private name.
#define topoPathTypeStr diagnosticsP2pTestPathNames

#include DIAG_P2P_CC_PATH

const char* diagnosticsP2pTestPathNames[] = {"LOC", "XGMI", "NVB", "C2C", "PIX", "PXB",
                                             "P2C", "PXN",  "PHB", "SYS", "NET", "DIS"};

#undef topoPathTypeStr

namespace {

constexpr int kPathTypes[] = {PATH_LOC, PATH_NVL, PATH_NVB, PATH_C2C, PATH_PIX, PATH_PXB,
                              PATH_P2C, PATH_PXN, PATH_PHB, PATH_SYS, PATH_NET, PATH_DIS};
constexpr int kHandleTypes[] = {0,
                                ncclDiagP2pHandleDirect,
                                ncclDiagP2pHandleLegacyIpc,
                                ncclDiagP2pHandleCuMemPosixFd,
                                ncclDiagP2pHandleCuMemFabric,
                                ncclDiagP2pHandleCuMemOther};
constexpr const char* kNvidiaTerms[] = {"nvidia", "nvlink", "imex", "cuda", "cumem"};

ncclDiagP2pEdgeInfo MakeEdge(int pathType, int handleType) {
  ncclDiagP2pEdgeInfo edge = {};
  edge.p2p = 1;
  edge.pathType = pathType;
  edge.handleType = handleType;
  return edge;
}

std::string Lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

// The free text of a report line: every key=value token dropped, since those keep their NCCL names.
std::string FreeText(const std::string& line) {
  std::istringstream words(line);
  std::string word, text;
  while (words >> word) {
    if (word.find('=') != std::string::npos) continue;
    text += word + " ";
  }
  return text;
}

void ExpectNoNvidiaTerms(const std::string& text, const std::string& context) {
  const std::string lower = Lower(text);
  for (const char* term : kNvidiaTerms) {
    EXPECT_EQ(lower.find(term), std::string::npos) << "'" << term << "' in " << context << ": " << text;
  }
}

std::string Context(int pathType, int handleType) {
  return "path=" + std::to_string(pathType) + " handle=" + std::to_string(handleType);
}

TEST(DiagnosticsP2pMicrotest, EdgeAdviceNamesAmdSmiForEveryEdge) {
  for (int pathType : kPathTypes) {
    for (int handleType : kHandleTypes) {
      const ncclDiagP2pEdgeInfo edge = MakeEdge(pathType, handleType);
      const std::string advice = ncclDiagP2pEdgeAdvice(&edge);
      ExpectNoNvidiaTerms(advice, Context(pathType, handleType));
      EXPECT_NE(advice.find("amd-smi"), std::string::npos) << Context(pathType, handleType) << ": " << advice;
    }
  }
}

TEST(DiagnosticsP2pMicrotest, XgmiEdgeAdviceChecksLinkStatusAndType) {
  for (int pathType : {PATH_NVL, PATH_NVB}) {
    const ncclDiagP2pEdgeInfo edge = MakeEdge(pathType, ncclDiagP2pHandleLegacyIpc);
    const std::string advice = ncclDiagP2pEdgeAdvice(&edge);
    EXPECT_NE(advice.find("amd-smi xgmi -l"), std::string::npos) << advice;
    EXPECT_NE(advice.find("amd-smi topology -t"), std::string::npos) << advice;
  }
}

TEST(DiagnosticsP2pMicrotest, PcieEdgeAdviceChecksAccessDmaAndHost) {
  for (int pathType : {PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS}) {
    const ncclDiagP2pEdgeInfo edge = MakeEdge(pathType, ncclDiagP2pHandleLegacyIpc);
    const std::string advice = ncclDiagP2pEdgeAdvice(&edge);
    EXPECT_NE(advice.find("amd-smi topology -a"), std::string::npos) << advice;
    EXPECT_NE(advice.find("amd-smi topology -d"), std::string::npos) << advice;
    EXPECT_NE(advice.find("IOMMU"), std::string::npos) << advice;
    EXPECT_NE(advice.find("ACS"), std::string::npos) << advice;
  }
}

TEST(DiagnosticsP2pMicrotest, FabricEdgeAdviceIsGenericPairCheck) {
  const ncclDiagP2pEdgeInfo fabricHandle = MakeEdge(PATH_NVL, ncclDiagP2pHandleCuMemFabric);
  const ncclDiagP2pEdgeInfo netPath = MakeEdge(PATH_NET, ncclDiagP2pHandleLegacyIpc);
  for (const ncclDiagP2pEdgeInfo* edge : {&fabricHandle, &netPath}) {
    const std::string advice = ncclDiagP2pEdgeAdvice(edge);
    EXPECT_NE(advice.find("amd-smi topology -t"), std::string::npos) << advice;
    EXPECT_NE(advice.find("amd-smi topology -a"), std::string::npos) << advice;
    EXPECT_EQ(advice.find("xgmi -l"), std::string::npos) << advice;
  }
}

TEST(DiagnosticsP2pMicrotest, ImportAdviceNamesHipForEveryHandle) {
  for (int pathType : kPathTypes) {
    for (int handleType : kHandleTypes) {
      const ncclDiagP2pEdgeInfo edge = MakeEdge(pathType, handleType);
      ExpectNoNvidiaTerms(ncclDiagP2pImportAdvice(&edge), Context(pathType, handleType));
    }
  }
  const ncclDiagP2pEdgeInfo direct = MakeEdge(PATH_NVL, ncclDiagP2pHandleDirect);
  const ncclDiagP2pEdgeInfo legacyIpc = MakeEdge(PATH_NVL, ncclDiagP2pHandleLegacyIpc);
  const ncclDiagP2pEdgeInfo posixFd = MakeEdge(PATH_NVL, ncclDiagP2pHandleCuMemPosixFd);
  EXPECT_NE(std::string(ncclDiagP2pImportAdvice(&direct)).find("HIP peer-access"), std::string::npos);
  EXPECT_NE(std::string(ncclDiagP2pImportAdvice(&legacyIpc)).find("HIP IPC"), std::string::npos);
  EXPECT_NE(std::string(ncclDiagP2pImportAdvice(&posixFd)).find("HIP virtual-memory"), std::string::npos);
}

TEST(DiagnosticsP2pMicrotest, FabricImportAdviceFallsBackToEdgeAdvice) {
  const ncclDiagP2pEdgeInfo edge = MakeEdge(PATH_NVL, ncclDiagP2pHandleCuMemFabric);
  EXPECT_STREQ(ncclDiagP2pImportAdvice(&edge), ncclDiagP2pEdgeAdvice(&edge));
}

struct ReportCase {
  int reason;
  const char* kind;
};

constexpr ReportCase kReportCases[] = {
    {ncclDiagP2pReasonNoDescriptor, "p2p: destination buffer unavailable"},
    {ncclDiagP2pReasonLocalCuda, "p2p: local HIP setup failed"},
    {ncclDiagP2pReasonImport, "p2p: peer-memory import failed"},
    {ncclDiagP2pReasonWriteMismatch, "p2p: write mismatch"},
    {ncclDiagP2pReasonReadMismatch, "p2p: read mismatch"},
    {ncclDiagP2pReasonTopo, "p2p: topology check failed"},
    {ncclDiagP2pReasonWriteLaunch, "p2p: launch/check failed"},
    {ncclDiagP2pReasonReadLaunch, "p2p: launch/check failed"},
};

class DiagnosticsP2pReportMicrotest : public ::testing::TestWithParam<ReportCase> {
 protected:
  void SetUp() override {
    comm_ = std::make_unique<ncclComm>();
    peers_ = std::make_unique<ncclPeerInfo[]>(2);
    for (int rank = 0; rank < 2; rank++) {
      peers_[rank].cudaDev = rank + 4;
      peers_[rank].nvmlDev = rank + 4;
    }
    comm_->peerInfo = peers_.get();
  }

  std::string Report(const ncclDiagP2pEdgeInfo& edge) {
    ncclDiagP2pEdgeResult result = {};
    result.tested = 1;
    result.reason = GetParam().reason;
    testing::internal::CaptureStdout();
    ncclDiagP2pReport(comm_.get(), 0, 1, &edge, &result);
    return testing::internal::GetCapturedStdout();
  }

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclPeerInfo[]> peers_;
};

TEST_P(DiagnosticsP2pReportMicrotest, FreeTextNamesAmdToolsAndKeysKeepNcclNames) {
  for (int pathType : {PATH_NVL, PATH_PIX, PATH_SYS}) {
    for (int handleType : {ncclDiagP2pHandleDirect, ncclDiagP2pHandleLegacyIpc}) {
      const ncclDiagP2pEdgeInfo edge = MakeEdge(pathType, handleType);
      const std::string line = Report(edge);
      const std::string context = std::string(GetParam().kind) + " " + Context(pathType, handleType);
      ASSERT_NE(line.find(std::string("NCCL DIAG [INFO] ") + GetParam().kind), std::string::npos)
          << context << ": " << line;
      ExpectNoNvidiaTerms(FreeText(line), context);
      EXPECT_NE(line.find("srcRank=0 srcCudaDev=4 srcNvmlDev=4 dstRank=1 dstCudaDev=5 dstNvmlDev=5"),
                std::string::npos)
          << line;
      EXPECT_NE(line.find(std::string("handle=") + ncclDiagP2pHandleName(handleType)), std::string::npos) << line;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DiagnosticsP2pReport, DiagnosticsP2pReportMicrotest, ::testing::ValuesIn(kReportCases),
                         [](const ::testing::TestParamInfo<ReportCase>& info) {
                           return std::string(ncclDiagP2pReasonName(info.param.reason));
                         });

TEST(DiagnosticsP2pMicrotest, XgmiReportLineSuggestsXgmiLinkStatus) {
  auto comm = std::make_unique<ncclComm>();
  auto peers = std::make_unique<ncclPeerInfo[]>(2);
  comm->peerInfo = peers.get();
  const ncclDiagP2pEdgeInfo edge = MakeEdge(PATH_NVL, ncclDiagP2pHandleLegacyIpc);
  ncclDiagP2pEdgeResult result = {};
  result.tested = 1;
  result.reason = ncclDiagP2pReasonNoDescriptor;
  testing::internal::CaptureStdout();
  ncclDiagP2pReport(comm.get(), 0, 1, &edge, &result);
  const std::string line = testing::internal::GetCapturedStdout();
  EXPECT_NE(line.find("handle=LEGACY_CUDA_IPC reason=noDescriptor"), std::string::npos) << line;
  EXPECT_NE(line.find("'amd-smi xgmi -l'"), std::string::npos) << line;
}

}  // namespace
