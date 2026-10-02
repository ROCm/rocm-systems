/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the AMD SMI arm of src/ras/diagnostics_gpu.cc. Each case runs one check end to end:
// every simulated node registers its ranks and answers the amd_smi_diag* queries (fakes/amdsmi_fakes.cc), the
// node's CollectLocal records are concatenated the way the RAS overlay gathers them, and Summarize turns the
// result into the captured report lines.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ScopedHook.h"
#include "comm.h"
#include "fakes/amdsmi_fakes.h"
#include "fakes/nccl_fakes.h"      // g_loadParam, used by param_redirect.h
#include "fakes/param_redirect.h"  // NCCL_DIAGNOSTICS_ECC_THRESHOLD is read on every call
#include "ras/ras_internal.h"

#include RAS_DIAGNOSTICS_GPU_CC_PATH

// Owned by src/misc/rocmwrap.cc, which rccl-UnitTestsMicro does not link; each simulated node sets it.
int ncclCudaDriverVersionCache = -1;

namespace {

constexpr uint64_t kCommHash = UINT64_C(0x5fa31c27a9e0d1b4);
constexpr int kRanks = 4;
constexpr int kRanksPerNode = 2;
constexpr int kNodes = kRanks / kRanksPerNode;
constexpr char kModel[] = "AMD Instinct MI355X";
constexpr int kDriverVersion = 70253211;
constexpr int kXgmiLinks = 7;

int64_t BusIdOfRank(int rank) { return int64_t(0x05 + 0x20 * rank) << 12; }

int RankOfBusId(int64_t busId) {
  for (int rank = 0; rank < kRanks; rank++) {
    if (BusIdOfRank(rank) == busId) return rank;
  }
  return -1;
}

// What AMD SMI reports for the GPU of one rank; an empty model makes the model query fail.
struct FakeGpu {
  std::string model = kModel;
  amdsmiDiagEccCounts ecc{};
  amdsmiDiagXgmiLinks links{kXgmiLinks, 0};
};

// What one node reports; ranks 2n and 2n+1 run on node n.
struct FakeNode {
  bool amdSmiAvailable = true;
  uint32_t nGpus = 8;
  int driverVersion = kDriverVersion;
};

// One rank of the communicator under test, registered the way ncclCommInit leaves it in ncclComms.
struct FakeRank {
  explicit FakeRank(int rank) : comm(std::make_unique<ncclComm>()), peer(std::make_unique<ncclPeerInfo[]>(1)) {
    comm->commHash = kCommHash;
    comm->peerInfo = peer.get();
    comm->peerInfoValid = true;
    comm->rank = rank;
    comm->nRanks = kRanks;
    comm->cudaDev = rank % kRanksPerNode;
    comm->nvmlDev = rank % kRanksPerNode;
    comm->busId = BusIdOfRank(rank);
    comm->localRank = rank % kRanksPerNode;
    comm->localRanks = kRanksPerNode;
    peer[0].hostHash = UINT64_C(0x99aabbccddeeff00);
    peer[0].pidHash = UINT64_C(0x0123456789abcdef);
  }

  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclPeerInfo[]> peer;
};

using CollectFn = ncclResult_t (*)(const rasDiagnosticsContext*, rasDiagnosticsLocalData*);
using SummarizeFn = ncclResult_t (*)(const rasDiagnosticsContext*, const rasDiagnosticsReporter*, const char*, int);

ncclResult_t CaptureLine(void* target, const char* line) {
  static_cast<std::vector<std::string>*>(target)->emplace_back(line);
  return ncclSuccess;
}

void SetRegisteredComms(const std::vector<FakeRank>& ranks) {
  std::lock_guard<std::mutex> lock(ncclCommsMutex);
  std::free(ncclComms);
  ncclComms = nullptr;
  nNcclComms = 0;
  if (ranks.empty()) return;
  ncclComms = static_cast<ncclComm**>(std::calloc(ranks.size(), sizeof(*ncclComms)));
  ASSERT_NE(nullptr, ncclComms);
  nNcclComms = static_cast<int>(ranks.size());
  for (size_t i = 0; i < ranks.size(); i++) ncclComms[i] = ranks[i].comm.get();
}

class RasDiagnosticsGpuMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetAmdSmiFakes();
    SetRegisteredComms({});
  }
  void TearDown() override {
    ResetAmdSmiFakes();
    SetRegisteredComms({});
    ncclCudaDriverVersionCache = -1;
  }

  // Answers the AMD SMI queries of node `nodeIdx`; a bus ID that is not on that node is a test bug.
  void InstallNodeFakes(int nodeIdx) {
    const FakeNode& node = nodes[nodeIdx];
    ncclCudaDriverVersionCache = node.driverVersion;
    if (!node.amdSmiAvailable) {
      ResetAmdSmiFakes();
      return;
    }
    auto gpuOf = [this, nodeIdx](int64_t busId) -> FakeGpu* {
      int rank = RankOfBusId(busId);
      EXPECT_EQ(nodeIdx, rank / kRanksPerNode) << "query for bus ID 0x" << std::hex << busId;
      return rank < 0 ? nullptr : &gpus[rank];
    };
    g_amdSmiDiagGpuCount = [node](uint32_t* count) {
      *count = node.nGpus;
      return ncclSuccess;
    };
    g_amdSmiDiagGpuModel = [gpuOf](int64_t busId, char* model, size_t len) {
      FakeGpu* gpu = gpuOf(busId);
      if (gpu == nullptr || gpu->model.empty()) return ncclSystemError;
      snprintf(model, len, "%s", gpu->model.c_str());
      return ncclSuccess;
    };
    g_amdSmiDiagEccCounts = [gpuOf](int64_t busId, amdsmiDiagEccCounts* counts) {
      FakeGpu* gpu = gpuOf(busId);
      if (gpu == nullptr) return ncclSystemError;
      *counts = gpu->ecc;
      return ncclSuccess;
    };
    g_amdSmiDiagXgmiLinks = [gpuOf](int64_t busId, amdsmiDiagXgmiLinks* links) {
      FakeGpu* gpu = gpuOf(busId);
      if (gpu == nullptr) return ncclSystemError;
      *links = gpu->links;
      return ncclSuccess;
    };
  }

  // Collects the check on every node, then summarizes all records together and returns the report lines.
  std::vector<std::string> Run(CollectFn collect, SummarizeFn summarize) {
    rasDiagnosticsContext ctx{};
    std::string gathered;
    for (int nodeIdx = 0; nodeIdx < kNodes; nodeIdx++) {
      std::vector<FakeRank> ranks;
      for (int i = 0; i < kRanksPerNode; i++) ranks.emplace_back(nodeIdx * kRanksPerNode + i);
      SetRegisteredComms(ranks);
      InstallNodeFakes(nodeIdx);
      rasDiagnosticsLocalData data{};
      EXPECT_EQ(ncclSuccess, collect(&ctx, &data));
      EXPECT_EQ(kRanksPerNode, data.nRecords);
      if (data.records != nullptr) gathered.append(data.records, data.recordsBytes);
      free(data.records);
      SetRegisteredComms({});
    }
    ResetAmdSmiFakes();

    std::vector<std::string> lines;
    rasDiagnosticsReporter reporter{CaptureLine, nullptr, &lines};
    EXPECT_EQ(ncclSuccess, summarize(&ctx, &reporter, gathered.data(), static_cast<int>(gathered.size())));
    for (const std::string& line : lines) {
      EXPECT_EQ(std::string::npos, line.find("NVML")) << line;
      EXPECT_EQ(std::string::npos, line.find("NVLink")) << line;
      EXPECT_EQ(std::string::npos, line.find("CUDA")) << line;
    }
    return lines;
  }

  std::vector<std::string> RunGpuInventory() {
    return Run(rasDiagnosticsGpuModelCollectLocal, rasDiagnosticsGpuModelSummarize);
  }
  std::vector<std::string> RunDriverVersion() {
    return Run(rasDiagnosticsCudaDriverVersionCollectLocal, rasDiagnosticsCudaDriverVersionSummarize);
  }
  std::vector<std::string> RunEcc() { return Run(rasDiagnosticsEccCollectLocal, rasDiagnosticsEccSummarize); }
  std::vector<std::string> RunXgmi() { return Run(rasDiagnosticsNvLinkCollectLocal, rasDiagnosticsNvLinkSummarize); }

  FakeNode nodes[kNodes];
  FakeGpu gpus[kRanks];
};

using Lines = std::vector<std::string>;

TEST_F(RasDiagnosticsGpuMicrotest, HealthyNodesReportOkForEveryCheck) {
  EXPECT_EQ(Lines{"[OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunGpuInventory());
  EXPECT_EQ(Lines{"[OK]   HIP driver version: 70253211 consistent across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunDriverVersion());
  EXPECT_EQ(Lines{"[OK]   ECC: no uncorrected volatile errors across 4 ranks in comm 0x5fa31c27a9e0d1b4"}, RunEcc());
  EXPECT_EQ(Lines{"[OK]   XGMI: 7 links per GPU, all active across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunXgmi());
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuCountMismatchAcrossNodes) {
  nodes[1].nGpus = 7;
  EXPECT_EQ(Lines{"[INFO] GPU inventory: count mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {2,3} "
                  "differ from rank 0 (8)"},
            RunGpuInventory());
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelMismatchOnOneRank) {
  gpus[3].model = "AMD Instinct MI300X";
  EXPECT_EQ(Lines{"[INFO] GPU inventory: model mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {3} "
                  "differ from rank 0 (AMD Instinct MI355X)"},
            RunGpuInventory());
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelQueryFailsOnOneRank) {
  gpus[1].model = "";
  EXPECT_EQ(Lines{"[INFO] GPU inventory: model mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {1} "
                  "differ from rank 0 (AMD Instinct MI355X)"},
            RunGpuInventory());
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionMismatchAcrossNodes) {
  nodes[1].driverVersion = 70051831;
  EXPECT_EQ(Lines{"[INFO] HIP driver version: mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {2,3} "
                  "differ from rank 0 (70253211)"},
            RunDriverVersion());
}

TEST_F(RasDiagnosticsGpuMicrotest, EccUncorrectableErrorsOnOneRank) {
  gpus[2].ecc.uncorrectable = 3;
  gpus[2].ecc.correctable = 40;
  EXPECT_EQ(Lines{"[INFO] ECC: uncorrected volatile errors on rank(s) {2} (worst=3) across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunEcc());
}

TEST_F(RasDiagnosticsGpuMicrotest, EccDeferredErrorsCountAsUncorrected) {
  gpus[1].ecc.deferred = 2;
  gpus[3].ecc.uncorrectable = 1;
  gpus[3].ecc.deferred = 4;
  EXPECT_EQ(Lines{"[INFO] ECC: uncorrected volatile errors on rank(s) {1,3} (worst=5) across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunEcc());
}

TEST_F(RasDiagnosticsGpuMicrotest, EccCorrectableErrorsIgnoredWithoutThreshold) {
  gpus[0].ecc.correctable = 1000;
  EXPECT_EQ(Lines{"[OK]   ECC: no uncorrected volatile errors across 4 ranks in comm 0x5fa31c27a9e0d1b4"}, RunEcc());
}

TEST_F(RasDiagnosticsGpuMicrotest, EccCorrectableErrorsAtOrAboveThreshold) {
  ScopedHook loadParam(g_loadParam, [](const char* env, int64_t deftVal) -> int64_t {
    return std::string(env) == "DIAGNOSTICS_ECC_THRESHOLD" ? 10 : deftVal;
  });
  gpus[0].ecc.correctable = 9;
  gpus[1].ecc.correctable = 10;
  gpus[3].ecc.correctable = 12;
  EXPECT_EQ(Lines{"[INFO] ECC: corrected volatile errors at or above threshold 10 on rank(s) {1,3} (worst=12) "
                  "across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunEcc());
  EXPECT_GE(loadParam.calls, 1);
}

TEST_F(RasDiagnosticsGpuMicrotest, XgmiInactiveLinkOnOneRank) {
  gpus[2].links.nDown = 1;
  EXPECT_EQ(Lines{"[INFO] XGMI: inactive link(s) on rank(s) {2} across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunXgmi());
}

TEST_F(RasDiagnosticsGpuMicrotest, XgmiLinkCountMismatchOnOneRank) {
  gpus[1].links.nLinks = 6;
  EXPECT_EQ(Lines{"[INFO] XGMI: link-count mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {1} differ "
                  "from rank 0 (7)"},
            RunXgmi());
}

TEST_F(RasDiagnosticsGpuMicrotest, NoXgmiLinesWithoutXgmiLinks) {
  for (FakeGpu& gpu : gpus) gpu.links = {0, 0};
  EXPECT_EQ(Lines{}, RunXgmi());
}

TEST_F(RasDiagnosticsGpuMicrotest, AmdSmiUnavailableOnEveryNode) {
  for (FakeNode& node : nodes) node.amdSmiAvailable = false;
  EXPECT_EQ(Lines{"[INFO] GPU inventory: unavailable via AMD SMI across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunGpuInventory());
  EXPECT_EQ(Lines{"[OK]   HIP driver version: 70253211 consistent across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunDriverVersion());
  EXPECT_EQ(Lines{"[INFO] ECC: unavailable via AMD SMI across 4 ranks in comm 0x5fa31c27a9e0d1b4"}, RunEcc());
  EXPECT_EQ(Lines{}, RunXgmi());
}

TEST_F(RasDiagnosticsGpuMicrotest, AmdSmiUnavailableOnOneNode) {
  nodes[1].amdSmiAvailable = false;
  EXPECT_EQ(Lines{"[INFO] ECC: no uncorrected volatile errors across 2 of 4 ranks in comm 0x5fa31c27a9e0d1b4 (ECC "
                  "counters unavailable via AMD SMI on 2 ranks)"},
            RunEcc());
  EXPECT_EQ(Lines{"[INFO] XGMI: link-count mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {2,3} differ "
                  "from rank 0 (7)"},
            RunXgmi());
}

}  // namespace
