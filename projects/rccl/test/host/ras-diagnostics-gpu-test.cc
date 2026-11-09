/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_gpu.cc, which builds its AMD SMI arm here. The collect and
// summarize cases call one function each. The end-to-end cases (Run) collect a check on every simulated node, whose
// amd_smi_diag* answers come from fakes/amdsmi_fakes.cc, concatenate the records the way the RAS overlay gathers
// them, and summarize the result into report lines.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comm.h"
#include "fmt/format.h"
#include "fakes/amdsmi_fakes.h"
#include "fakes/ras_diagnostics_test_support.h"
#include "cudawrap.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "nvmlwrap.h"
#include "ras/diagnostics_checks_common.h"

namespace {

int64_t g_eccThreshold = 0;
ncclResult_t g_driverResult = ncclSuccess;
int g_driverVersion = 0;
std::vector<std::string> g_paramEnvs;
std::vector<int64_t> g_paramDefaults;

}  // namespace

ncclResult_t DiagnosticsGpuTestCudaDriverVersion(int* version);

#define ncclCudaDriverVersion DiagnosticsGpuTestCudaDriverVersion

#include RAS_DIAGNOSTICS_GPU_CC_PATH

#undef ncclCudaDriverVersion

// Owned by src/misc/nvmlwrap.cc, which this binary does not link.
int ncclNvmlDeviceCount = 0;
ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices]{};
ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices]{};

ncclResult_t DiagnosticsGpuTestCudaDriverVersion(int* version) {
  *version = 0x5EED;
  if (g_driverResult != ncclSuccess) return g_driverResult;
  *version = g_driverVersion;
  return ncclSuccess;
}

namespace {

using ras_test::OwnedComm;
using ras_test::ReporterState;
using ras_test::MakeReporter;
using ras_test::InstallNcclComms;

bool Contains(const std::vector<std::string>& lines, const std::string& text) {
  for (const std::string& line : lines) {
    if (line.find(text) != std::string::npos) return true;
  }
  return false;
}

rasDiagnosticsRankHeader MakeRank(uint64_t commHash, int rank, int nRanks) {
  rasDiagnosticsRankHeader header{};
  header.commId = {commHash, commHash + 1, commHash + 2};
  header.commRank = rank;
  header.commNRanks = nRanks;
  return header;
}

template <typename T>
std::vector<char> BuildRecords(const std::vector<std::pair<rasDiagnosticsRankHeader, T>>& records) {
  const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(T));
  std::vector<char> data(records.size() * stride, 0);
  size_t offset = 0;
  for (const auto& record : records) {
    memcpy(data.data() + offset, &record.first, sizeof(record.first));
    memcpy(data.data() + offset + sizeof(record.first), &record.second, sizeof(record.second));
    offset += stride;
  }
  return data;
}

rasDiagnosticsGpuModelData GpuModel(int count, const char* model) {
  rasDiagnosticsGpuModelData data{};
  data.nGpus = static_cast<uint8_t>(count);
  snprintf(data.model, sizeof(data.model), "%s", model);
  return data;
}

rasDiagnosticsCudaDriverVersionData DriverVersion(uint32_t version) {
  return rasDiagnosticsCudaDriverVersionData{version};
}

rasDiagnosticsNvidiaDriverVersionData NvidiaDriverVersion(const char* version) {
  rasDiagnosticsNvidiaDriverVersionData data{};
  snprintf(data.version, sizeof(data.version), "%s", version);
  return data;
}

rasDiagnosticsEccData EccData(bool available, uint64_t correctedSram = 0, uint64_t uncorrectedSram = 0,
                              uint64_t correctedDram = 0, uint64_t uncorrectedDram = 0) {
  return rasDiagnosticsEccData{correctedSram, uncorrectedSram, correctedDram, uncorrectedDram,
                               static_cast<uint8_t>(available)};
}

rasDiagnosticsNvLinkData NvLinkData(int links, int inactive, uint32_t minSpeed = 50000, uint32_t maxSpeed = 50000) {
  return rasDiagnosticsNvLinkData{static_cast<uint8_t>(links), static_cast<uint8_t>(inactive),
                                 links > inactive ? minSpeed : 0, links > inactive ? maxSpeed : 0};
}

void ResetState() {
  ras_test::ResetNcclComms();
  ResetAmdSmiFakes();
  g_eccThreshold = 0;
  g_paramEnvs.clear();
  g_paramDefaults.clear();
  g_driverResult = ncclSuccess;
  g_driverVersion = 0;
}

constexpr uint64_t kCommHash = UINT64_C(0x5fa31c27a9e0d1b4);
constexpr int kRanks = 4;
constexpr int kRanksPerNode = 2;
constexpr int kNodes = kRanks / kRanksPerNode;
constexpr char kModel[] = "AMD Instinct MI355X";
constexpr int kDriverVersion = 70253211;
constexpr char kGpuDriverVersion[] = "6.19.14.31400000";
constexpr int kXgmiLinks = 7;

int64_t BusIdOfRank(int rank) { return int64_t(0x05 + 0x20 * rank) << 12; }

int RankOfBusId(int64_t busId) {
  for (int rank = 0; rank < kRanks; rank++) {
    if (BusIdOfRank(rank) == busId) return rank;
  }
  return -1;
}

// What AMD SMI reports for the GPU of one rank; an empty model makes the model query fail, unknownToAmdSmi every
// query.
struct FakeGpu {
  bool unknownToAmdSmi = false;
  std::string model = kModel;
  amdsmiDiagEccCounts ecc{};
  amdsmiDiagXgmiLinks links{kXgmiLinks, 0};
};

// What one node reports; ranks 2n and 2n+1 run on node n.
struct FakeNode {
  bool amdSmiAvailable = true;
  uint32_t nGpus = 8;
  int driverVersion = kDriverVersion;
  std::string gpuDriverVersion = kGpuDriverVersion;  // amdgpu; empty makes the query fail
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

using Lines = std::vector<std::string>;

class RasDiagnosticsGpuMicrotest : public ::testing::Test {
 protected:

  rasDiagnosticsContext ctx{};
  ReporterState state;
  rasDiagnosticsReporter reporter{};

  void SetUp() override {
    ResetState();
    reporter = MakeReporter(&state);
    g_loadParam = [](const char* env, int64_t defaultValue) {
      g_paramEnvs.emplace_back(env);
      g_paramDefaults.push_back(defaultValue);
      return strcmp(env, "DIAGNOSTICS_ECC_THRESHOLD") == 0 ? g_eccThreshold : defaultValue;
    };
  }
  void TearDown() override {
    ResetState();
    ResetNcclFakes();
  }

  template <typename Payload>
  void CollectPayload(rasDiagnosticsCollectLocalFn collect, Payload* payload) {
    rasDiagnosticsLocalData data{};
    const auto result = collect(&ctx, &data);
    ncclUniquePtr<char> recordsOwner(data.records);
    ASSERT_EQ(ncclSuccess, result);
    ASSERT_EQ(1, data.nRecords);
    ASSERT_EQ(rasDiagnosticsLocalRecordStride(sizeof(Payload)), static_cast<size_t>(data.recordStride));
    ASSERT_EQ(data.recordStride, data.recordsBytes);
    ASSERT_NE(nullptr, data.records);
    memcpy(payload, data.records + sizeof(rasDiagnosticsRankHeader), sizeof(*payload));
  }

  template <typename Payload>
  void ExpectIncomplete(rasDiagnosticsSummarizeFn summarize, const char* checkName, uint64_t commHash, const Payload& payload) {
    state.lines.clear();
    auto records = BuildRecords<Payload>({{MakeRank(commHash, 0, 2), payload}});
    ASSERT_EQ(ncclSuccess, summarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
    ASSERT_EQ(1u, state.lines.size());
    EXPECT_EQ(fmt::format("[INFO] {}: diagnostics incomplete, gathered 1/2 ranks in comm "
                          "0x{:x}/0x{:x}/0x{:x}",
                          checkName, commHash, commHash + 1, commHash + 2), state.lines[0]);
  }

  template <typename Payload>
  void ExpectReporterFailure(
    rasDiagnosticsSummarizeFn summarize, std::initializer_list<std::pair<rasDiagnosticsRankHeader, Payload>> input) {
    state.lines.clear();
    state.calls = 0;
    state.failCall = 1;
    state.emitResult = ncclRemoteError;
    auto records = BuildRecords<Payload>(input);

    EXPECT_EQ(ncclRemoteError, summarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
    EXPECT_EQ(1, state.calls);

    state.emitResult = ncclSuccess;
    state.failCall = 0;
  }

  // Answers the AMD SMI queries of node `nodeIdx`; a bus ID that is not on that node is a test bug.
  void InstallNodeFakes(int nodeIdx) {
    const FakeNode& node = nodes[nodeIdx];
    g_driverVersion = node.driverVersion;
    if (!node.amdSmiAvailable) {
      ResetAmdSmiFakes();
      return;
    }
    // Like AMD SMI, the fakes know the physical GPUs only: a partition alias (function .1-.7) is not found.
    auto gpuOf = [this, nodeIdx](int64_t busId) -> FakeGpu* {
      if (busId & 0xf) {
        aliasQueries++;
        return nullptr;
      }
      int rank = RankOfBusId(busId);
      if (rank >= 0 && gpus[rank].unknownToAmdSmi) return nullptr;
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
    g_amdSmiDiagDriverVersion = [gpuOf, node](int64_t busId, char* version, size_t len) {
      FakeGpu* gpu = gpuOf(busId);
      if (gpu == nullptr || node.gpuDriverVersion.empty()) return ncclSystemError;
      snprintf(version, len, "%s", node.gpuDriverVersion.c_str());
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
  Lines Run(rasDiagnosticsCollectLocalFn collect, rasDiagnosticsSummarizeFn summarize) {
    std::string gathered;
    for (int nodeIdx = 0; nodeIdx < kNodes; nodeIdx++) {
      std::vector<FakeRank> ranks;
      std::vector<ncclComm*> comms;
      for (int i = 0; i < kRanksPerNode; i++) {
        ranks.emplace_back(nodeIdx * kRanksPerNode + i);
        ranks.back().comm->busId |= partitionFunction;
        comms.push_back(ranks.back().comm.get());
      }
      InstallNcclComms(comms);
      InstallNodeFakes(nodeIdx);
      rasDiagnosticsLocalData data{};
      EXPECT_EQ(ncclSuccess, collect(&ctx, &data));
      EXPECT_EQ(kRanksPerNode, data.nRecords);
      if (data.records != nullptr) gathered.append(data.records, data.recordsBytes);
      free(data.records);
      ras_test::ResetNcclComms();
    }
    ResetAmdSmiFakes();

    state.lines.clear();
    EXPECT_EQ(ncclSuccess, summarize(&ctx, &reporter, gathered.data(), static_cast<int>(gathered.size())));
    for (const std::string& line : state.lines) {
      EXPECT_EQ(std::string::npos, line.find("NVML")) << line;
      EXPECT_EQ(std::string::npos, line.find("NVLink")) << line;
      EXPECT_EQ(std::string::npos, line.find("CUDA")) << line;
    }
    return state.lines;
  }

  Lines RunGpuInventory() { return Run(rasDiagnosticsGpuModelCollectLocal, rasDiagnosticsGpuModelSummarize); }
  Lines RunDriverVersion() {
    return Run(rasDiagnosticsCudaDriverVersionCollectLocal, rasDiagnosticsCudaDriverVersionSummarize);
  }
  Lines RunGpuDriverVersion() {
    return Run(rasDiagnosticsNvidiaDriverVersionCollectLocal, rasDiagnosticsNvidiaDriverVersionSummarize);
  }
  Lines RunEcc() { return Run(rasDiagnosticsEccCollectLocal, rasDiagnosticsEccSummarize); }
  Lines RunXgmi() { return Run(rasDiagnosticsNvLinkCollectLocal, rasDiagnosticsNvLinkSummarize); }

  FakeNode nodes[kNodes];
  FakeGpu gpus[kRanks];
  // Function of the bus ID HIP gives every rank: 0 for a whole GPU, 1-7 for a partition alias (CPX, DPX).
  int partitionFunction = 0;
  int aliasQueries = 0;
};

}  // namespace

TEST_F(RasDiagnosticsGpuMicrotest, SummariesValidateArguments) {
  const std::array<rasDiagnosticsSummarizeFn, 5> summaries = {rasDiagnosticsGpuModelSummarize,
                                             rasDiagnosticsCudaDriverVersionSummarize,
                                             rasDiagnosticsNvidiaDriverVersionSummarize,
                                             rasDiagnosticsEccSummarize,
                                             rasDiagnosticsNvLinkSummarize};
  rasDiagnosticsReporter invalid{};
  const char byte = 0;

  for (rasDiagnosticsSummarizeFn summarize : summaries) {
    EXPECT_EQ(ncclInternalError, summarize(&ctx, nullptr, &byte, 1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &invalid, &byte, 1));
    EXPECT_EQ(ncclSuccess, summarize(&ctx, &reporter, nullptr, 0));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, nullptr, 1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, &byte, -1));
    EXPECT_EQ(ncclInternalError, summarize(&ctx, &reporter, &byte, 1));
  }
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionCollectHandlesSuccessFailureAndZero) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  owned.comm->nvmlDev = 0;
  InstallNcclComms({owned.comm.get()});
  rasDiagnosticsCudaDriverVersionData data{};
  g_driverVersion = 70002000;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsCudaDriverVersionCollectLocal, &data));
  EXPECT_EQ(70002000u, data.version);

  g_driverVersion = 0;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsCudaDriverVersionCollectLocal, &data));
  EXPECT_EQ(0u, data.version);  // Successful zero means no driver installed, not a failed query.

  g_driverResult = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsCudaDriverVersionCollectLocal, &data));
  EXPECT_EQ(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN, data.version);
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuDriverVersionCollectClearsFailedOutputAndTerminatesFullBuffers) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  InstallNcclComms({owned.comm.get()});
  rasDiagnosticsNvidiaDriverVersionData data{};
  std::string version = "6.14.14";
  ncclResult_t versionResult = ncclSuccess;
  std::vector<size_t> bufferSizes;
  g_amdSmiDiagDriverVersion = [&](int64_t, char* out, size_t len) {
    bufferSizes.push_back(len);
    if (versionResult != ncclSuccess) {
      // An unsuccessful query must not leave partial output in the collected POD.
      memset(out, '!', len);
      return versionResult;
    }
    version.copy(out, len);
    if (version.size() < len) out[version.size()] = '\0';
    return ncclSuccess;
  };

  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvidiaDriverVersionCollectLocal, &data));
  EXPECT_STREQ("6.14.14", data.version);
  for (size_t i = version.size(); i < sizeof(data.version); ++i) EXPECT_EQ('\0', data.version[i]);

  versionResult = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvidiaDriverVersionCollectLocal, &data));
  for (char byte : data.version) EXPECT_EQ('\0', byte);

  versionResult = ncclSuccess;
  version.assign(sizeof(data.version), 'V');
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvidiaDriverVersionCollectLocal, &data));
  EXPECT_EQ(std::string(sizeof(data.version) - 1, 'V'), std::string(data.version, sizeof(data.version) - 1));
  EXPECT_EQ('\0', data.version[sizeof(data.version) - 1]);
  EXPECT_EQ(std::vector<size_t>(3, sizeof(data.version)), bufferSizes);
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuDriverVersionCollectRejectsInvalidArguments) {
  int queries = 0;
  g_amdSmiDiagDriverVersion = [&](int64_t, char*, size_t) {
    ++queries;
    return ncclSuccess;
  };
  rasDiagnosticsLocalData data{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNvidiaDriverVersionCollectLocal(nullptr, &data));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsNvidiaDriverVersionCollectLocal(&ctx, nullptr));
  EXPECT_EQ(0, queries);
}

TEST_F(RasDiagnosticsGpuMicrotest, CollectLocalBuildsOneRecordForEachCheck) {
  OwnedComm comm(0x100, 0x101, 0x102, 0);
  comm.comm->nRanks = 1;
  comm.comm->nvmlDev = 0;
  InstallNcclComms({comm.comm.get()});
  g_amdSmiDiagGpuCount = [](uint32_t* count) {
    *count = 1;
    return ncclSuccess;
  };
  g_amdSmiDiagGpuModel = [](int64_t, char* model, size_t len) {
    snprintf(model, len, "%s", "MI300X");
    return ncclSuccess;
  };
  g_amdSmiDiagDriverVersion = [](int64_t, char* version, size_t len) {
    snprintf(version, len, "%s", "6.14.14");
    return ncclSuccess;
  };
  g_amdSmiDiagEccCounts = [](int64_t, amdsmiDiagEccCounts* counts) {
    *counts = {};
    return ncclSuccess;
  };
  g_amdSmiDiagXgmiLinks = [](int64_t, amdsmiDiagXgmiLinks* links) {
    *links = {};
    return ncclSuccess;
  };
  g_driverVersion = 70002000;

  const std::array<rasDiagnosticsCollectLocalFn, 5> collectors = {rasDiagnosticsGpuModelCollectLocal,
                                                                  rasDiagnosticsCudaDriverVersionCollectLocal,
                                                                  rasDiagnosticsEccCollectLocal,
                                                                  rasDiagnosticsNvLinkCollectLocal,
                                                                  rasDiagnosticsNvidiaDriverVersionCollectLocal};
  const std::array<size_t, 5> payloadSizes = {sizeof(rasDiagnosticsGpuModelData),
                                              sizeof(rasDiagnosticsCudaDriverVersionData),
                                              sizeof(rasDiagnosticsEccData),
                                              sizeof(rasDiagnosticsNvLinkData),
                                              sizeof(rasDiagnosticsNvidiaDriverVersionData)};
  for (size_t index = 0; index < collectors.size(); index++) {
    rasDiagnosticsLocalData data{};
    const auto result = collectors[index](&ctx, &data);
    ncclUniquePtr<char> recordsOwner(data.records);
    ASSERT_EQ(ncclSuccess, result);
    EXPECT_EQ(1, data.nRecords);
    EXPECT_EQ(rasDiagnosticsLocalRecordStride(payloadSizes[index]), static_cast<size_t>(data.recordStride));
    EXPECT_GT(data.recordsBytes, static_cast<int>(sizeof(rasDiagnosticsRankHeader)));
    const auto* rank = reinterpret_cast<const rasDiagnosticsRankHeader*>(data.records);
    EXPECT_EQ(0x100u, rank->commId.commHash);
    EXPECT_EQ(0, rank->commRank);
    EXPECT_EQ(1, rank->commNRanks);
    const char* payload = data.records + sizeof(*rank);
    if (index == 0) {
      const auto* gpu = reinterpret_cast<const rasDiagnosticsGpuModelData*>(payload);
      EXPECT_EQ(1, gpu->nGpus);
      EXPECT_STREQ("MI300X", gpu->model);
    } else if (index == 1) {
      EXPECT_EQ(70002000u, reinterpret_cast<const rasDiagnosticsCudaDriverVersionData*>(payload)->version);
    } else if (index == 2) {
      EXPECT_EQ(1, reinterpret_cast<const rasDiagnosticsEccData*>(payload)->available);
    } else if (index == 3) {
      const auto* links = reinterpret_cast<const rasDiagnosticsNvLinkData*>(payload);
      EXPECT_EQ(0, links->nLinks);
      EXPECT_EQ(0, links->nInactive);
    } else {
      EXPECT_STREQ("6.14.14", reinterpret_cast<const rasDiagnosticsNvidiaDriverVersionData*>(payload)->version);
    }
  }
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelSummaryCoversAvailabilityAndMismatchCases) {
  auto records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(1, 0, 1), GpuModel(8, "MI300X")},
     {MakeRank(2, 0, 1), GpuModel(8, RAS_DIAG_GPU_MODEL_UNKNOWN)},
     {MakeRank(3, 0, 1), GpuModel(0, "MI250")},
     {MakeRank(4, 0, 1), GpuModel(0, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "8x MI300X"));
  EXPECT_TRUE(Contains(state.lines, "GPU model unavailable"));
  EXPECT_TRUE(Contains(state.lines, "GPU count unavailable"));
  EXPECT_TRUE(Contains(state.lines, "unavailable via " RAS_DIAG_GPU_SOURCE));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(8, 0, 1), GpuModel(0, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[INFO] GPU inventory: unavailable via " RAS_DIAG_GPU_SOURCE " across 1 ranks in comm 0x8",
            state.lines[0]);

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(5, 1, 2), GpuModel(4, "MI250")}, {MakeRank(5, 0, 2), GpuModel(8, "MI300X")}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "rank(s) {1} differ from rank 0 (8)"));
  EXPECT_TRUE(Contains(state.lines, "rank(s) {1} differ from rank 0 (MI300X)"));

  ExpectIncomplete(rasDiagnosticsGpuModelSummarize, "GPU inventory", 6, GpuModel(8, "MI300X"));
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionSummaryCoversKnownUnknownMismatchAndIncomplete) {
  auto records = BuildRecords<rasDiagnosticsCudaDriverVersionData>(
    {{MakeRank(1, 0, 1), DriverVersion(70002000)},
     {MakeRank(2, 0, 1), DriverVersion(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "70002000 consistent"));
  EXPECT_TRUE(Contains(state.lines, "unavailable across"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsCudaDriverVersionData>(
    {{MakeRank(3, 1, 2), DriverVersion(2)}, {MakeRank(3, 0, 2), DriverVersion(1)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "rank(s) {1} differ from rank 0 (1)"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsCudaDriverVersionData>(
    {{MakeRank(5, 1, 2), DriverVersion(70002000)},
     {MakeRank(5, 0, 2), DriverVersion(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[INFO] " RAS_DIAG_DRIVER_LABEL ": mismatch across 2 ranks in comm 0x5, rank(s) {1} differ from rank 0 "
            "(unavailable)",
            state.lines[0]);

  ExpectIncomplete(rasDiagnosticsCudaDriverVersionSummarize, RAS_DIAG_DRIVER_LABEL, 4, DriverVersion(1));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvidiaDriverVersionSummaryReportsAvailabilityMismatchAndIncomplete) {
  auto records = BuildRecords<rasDiagnosticsNvidiaDriverVersionData>(
    {{MakeRank(2, 0, 1), NvidiaDriverVersion("")},
     {MakeRank(1, 1, 2), NvidiaDriverVersion("580.82.07")},
     {MakeRank(1, 0, 2), NvidiaDriverVersion("580.82.07")}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvidiaDriverVersionSummarize(
                          &ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(2u, state.lines.size());
  EXPECT_EQ("[OK]   " RAS_DIAG_GPU_DRIVER_LABEL ": 580.82.07 consistent across 2 ranks in comm 0x1", state.lines[0]);
  EXPECT_EQ("[INFO] " RAS_DIAG_GPU_DRIVER_LABEL ": unavailable via " RAS_DIAG_GPU_SOURCE " across 1 ranks in comm 0x2",
            state.lines[1]);

  for (const char* expected : {"580.82.07", ""}) {
    state.lines.clear();
    records = BuildRecords<rasDiagnosticsNvidiaDriverVersionData>(
      {{MakeRank(3, 1, 2), NvidiaDriverVersion("580.95.05")},
       {MakeRank(3, 0, 2), NvidiaDriverVersion(expected)}});
    ASSERT_EQ(ncclSuccess, rasDiagnosticsNvidiaDriverVersionSummarize(
                            &ctx, &reporter, records.data(), static_cast<int>(records.size())));
    ASSERT_EQ(1u, state.lines.size());
    EXPECT_EQ(std::string("[INFO] " RAS_DIAG_GPU_DRIVER_LABEL ": mismatch across 2 ranks in comm 0x3, "
                          "rank(s) {1} differ from rank 0 (") +
                (expected[0] != '\0' ? expected : "unavailable via " RAS_DIAG_GPU_SOURCE) + ")", state.lines[0]);
  }
  ExpectIncomplete(rasDiagnosticsNvidiaDriverVersionSummarize, RAS_DIAG_GPU_DRIVER_LABEL,
                   0xabc, NvidiaDriverVersion("580.82.07"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvidiaDriverVersionSummaryBoundsUnterminatedPeerData) {
  rasDiagnosticsNvidiaDriverVersionData version;
  memset(version.version, 'V', sizeof(version.version));
  auto records = BuildRecords<rasDiagnosticsNvidiaDriverVersionData>({{MakeRank(1, 0, 1), version}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvidiaDriverVersionSummarize(
                          &ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[OK]   " RAS_DIAG_GPU_DRIVER_LABEL ": " + std::string(sizeof(version.version) - 1, 'V') +
              " consistent across 1 ranks in comm 0x1", state.lines[0]);
}

TEST_F(RasDiagnosticsGpuMicrotest, EccSummaryCoversAvailabilityHealthyAndErrorCases) {
  auto records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(1, 0, 1), EccData(false)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "ECC: unavailable via " RAS_DIAG_GPU_SOURCE " across 1 ranks"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(2, 0, 1), EccData(true)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "no uncorrected volatile errors across 1 ranks"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>(
    {{MakeRank(3, 0, 2), EccData(true)}, {MakeRank(3, 1, 2), EccData(false)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "across 1 of 2 ranks"));

  state.lines.clear();
  g_eccThreshold = 5;
  records = BuildRecords<rasDiagnosticsEccData>(
    {{MakeRank(4, 1, 2), EccData(true, 0, 0, 5, 1)}, {MakeRank(4, 0, 2), EccData(true, 6, 2, 0, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "uncorrected volatile errors on rank(s) {0,1} (worst=2)"));
  EXPECT_TRUE(Contains(state.lines, "corrected volatile errors at or above threshold 5 on rank(s) {0,1} (worst=6)"));
  ASSERT_FALSE(g_paramEnvs.empty());
  for (const std::string& env : g_paramEnvs) EXPECT_EQ("DIAGNOSTICS_ECC_THRESHOLD", env);
  for (int64_t defaultValue : g_paramDefaults) EXPECT_EQ(0, defaultValue);

  ExpectIncomplete(rasDiagnosticsEccSummarize, "ECC", 5, EccData(true));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsEccData>(
    {{MakeRank(7, 0, 1), EccData(false)}, {MakeRank(6, 0, 1), EccData(true)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "no uncorrected volatile errors across 1 ranks in comm 0x6"));
  EXPECT_TRUE(Contains(state.lines, "ECC: unavailable via " RAS_DIAG_GPU_SOURCE " across 1 ranks in comm 0x7"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkSummaryCoversNoLinksHealthyMismatchInactiveAndIncomplete) {
  auto records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(1, 0, 1), NvLinkData(0, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(state.lines.empty());

  records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(2, 0, 1), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[OK]   " RAS_DIAG_LINK_LABEL ": 8 links per GPU, all active" RAS_DIAG_LINK_SPEED_OK
            " across 1 ranks in comm 0x2",
            state.lines[0]);

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>(
    {{MakeRank(3, 1, 2), NvLinkData(6, 0)}, {MakeRank(3, 0, 2), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "rank(s) {1} differ from rank 0 (8)"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>(
    {{MakeRank(3, 1, 2), NvLinkData(8, 1)}, {MakeRank(3, 0, 2), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "inactive link(s) on rank(s) {1}"));

  ExpectIncomplete(rasDiagnosticsNvLinkSummarize, RAS_DIAG_LINK_LABEL, 20, NvLinkData(8, 0));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>(
    {{MakeRank(6, 0, 1), NvLinkData(8, 0)}, {MakeRank(7, 0, 1), NvLinkData(8, 1)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "all active" RAS_DIAG_LINK_SPEED_OK " across 1 ranks in comm 0x6"));
  EXPECT_TRUE(Contains(state.lines, "inactive link(s) on rank(s) {0} across 1 ranks in comm 0x7"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkSummaryReportsSpeedDifferencesWithinAndBetweenRanks) {
  for (const auto& mismatch : {NvLinkData(8, 0, 25000, 50000), NvLinkData(8, 0, 25000, 25000)}) {
    state.lines.clear();
    auto records = BuildRecords<rasDiagnosticsNvLinkData>(
      {{MakeRank(30, 1, 2), mismatch}, {MakeRank(30, 0, 2), NvLinkData(8, 0)}});
    ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(),
                                                        static_cast<int>(records.size())));
    ASSERT_EQ(1u, state.lines.size());
    EXPECT_EQ("[INFO] " RAS_DIAG_LINK_LABEL ": inconsistent link speeds on rank(s) {1} across 2 ranks in comm 0x1e",
              state.lines[0]);
  }
}

namespace {

template <typename Payload>
std::vector<char> BuildOverflowRecords(const Payload& reference, const Payload& mismatch) {
  const int nRanks = RAS_DIAG_RANK_SET_MAX + 2;
  std::vector<std::pair<rasDiagnosticsRankHeader, Payload>> ranks;
  for (int rank = nRanks - 1; rank >= 0; --rank)
    ranks.push_back({MakeRank(0x20, rank, nRanks), rank == 0 ? reference : mismatch});
  return BuildRecords<Payload>(ranks);
}

std::string ExpectedTruncatedRanks() {
  std::string ranks = "{";
  for (int rank = 1; rank <= RAS_DIAG_RANK_SET_MAX; ++rank) {
    if (rank > 1) ranks += ",";
    ranks += std::to_string(rank);
  }
  return ranks + ",...} (N=" + std::to_string(RAS_DIAG_RANK_SET_MAX + 1) + ")";
}

}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelSummary_TooManyMismatchesTruncatesBothRankSets) {
  auto records = BuildOverflowRecords(GpuModel(8, "MI300X"), GpuModel(4, "MI250X"));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(),
                                                       static_cast<int>(records.size())));
  ASSERT_EQ(2u, state.lines.size());
  EXPECT_NE(std::string::npos, state.lines[0].find("count mismatch"));
  EXPECT_NE(std::string::npos, state.lines[1].find("model mismatch"));
  for (const auto& line : state.lines)
    EXPECT_NE(std::string::npos, line.find("rank(s) " + ExpectedTruncatedRanks() + " differ from rank 0"));
}

TEST_F(RasDiagnosticsGpuMicrotest, DriverVersionSummary_TooManyMismatchesTruncatesRankSet) {
  auto records = BuildOverflowRecords(DriverVersion(70002000), DriverVersion(70001000));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCudaDriverVersionSummarize(&ctx, &reporter, records.data(),
                                                                static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_NE(std::string::npos, state.lines[0].find("rank(s) " + ExpectedTruncatedRanks() + " differ from rank 0"));
}

TEST_F(RasDiagnosticsGpuMicrotest, EccSummary_TooManyErrorsTruncatesBothRankSets) {
  g_eccThreshold = 5;
  auto records = BuildOverflowRecords(EccData(true), EccData(true, 6, 2));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(),
                                                  static_cast<int>(records.size())));
  ASSERT_EQ(2u, state.lines.size());
  EXPECT_NE(std::string::npos, state.lines[0].find("uncorrected volatile errors on rank(s) " +
                                                ExpectedTruncatedRanks() + " (worst=2)"));
  EXPECT_NE(std::string::npos, state.lines[1].find("corrected volatile errors at or above threshold 5 on rank(s) " +
                                                ExpectedTruncatedRanks() + " (worst=6)"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkSummary_TooManyMismatchesTruncatesBothRankSets) {
  auto records = BuildOverflowRecords(NvLinkData(8, 0), NvLinkData(6, 1));
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(),
                                                     static_cast<int>(records.size())));
  ASSERT_EQ(2u, state.lines.size());
  EXPECT_NE(std::string::npos, state.lines[0].find("rank(s) " + ExpectedTruncatedRanks() + " differ from rank 0"));
  EXPECT_NE(std::string::npos, state.lines[1].find("inactive link(s) on rank(s) " + ExpectedTruncatedRanks()));
}

TEST_F(RasDiagnosticsGpuMicrotest, EveryReporterFailurePropagates) {
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(rasDiagnosticsGpuModelSummarize,
                                                     {{MakeRank(1, 0, 2), GpuModel(8, "MI300X")}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(rasDiagnosticsGpuModelSummarize,
                                                     {{MakeRank(2, 0, 1), GpuModel(8, "MI300X")}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(
    rasDiagnosticsGpuModelSummarize, {{MakeRank(3, 0, 1), GpuModel(8, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(rasDiagnosticsGpuModelSummarize,
                                                     {{MakeRank(4, 0, 1), GpuModel(0, "MI300X")}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(
    rasDiagnosticsGpuModelSummarize, {{MakeRank(5, 0, 1), GpuModel(0, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(
    rasDiagnosticsGpuModelSummarize,
    {{MakeRank(6, 0, 2), GpuModel(8, "MI300X")}, {MakeRank(6, 1, 2), GpuModel(4, "MI300X")}});
  ExpectReporterFailure<rasDiagnosticsGpuModelData>(
    rasDiagnosticsGpuModelSummarize,
    {{MakeRank(7, 0, 2), GpuModel(8, "MI300X")}, {MakeRank(7, 1, 2), GpuModel(8, "MI250")}});

  ExpectReporterFailure<rasDiagnosticsCudaDriverVersionData>(
    rasDiagnosticsCudaDriverVersionSummarize, {{MakeRank(8, 0, 2), DriverVersion(70002000)}});
  ExpectReporterFailure<rasDiagnosticsCudaDriverVersionData>(
    rasDiagnosticsCudaDriverVersionSummarize,
    {{MakeRank(9, 0, 1), DriverVersion(RAS_DIAG_CUDA_DRIVER_VERSION_UNKNOWN)}});
  ExpectReporterFailure<rasDiagnosticsCudaDriverVersionData>(
    rasDiagnosticsCudaDriverVersionSummarize, {{MakeRank(10, 0, 1), DriverVersion(70002000)}});
  ExpectReporterFailure<rasDiagnosticsCudaDriverVersionData>(
    rasDiagnosticsCudaDriverVersionSummarize,
    {{MakeRank(11, 0, 2), DriverVersion(70002000)}, {MakeRank(11, 1, 2), DriverVersion(70003000)}});

  ExpectReporterFailure<rasDiagnosticsNvidiaDriverVersionData>(
    rasDiagnosticsNvidiaDriverVersionSummarize, {{MakeRank(22, 0, 2), NvidiaDriverVersion("580.82.07")}});
  ExpectReporterFailure<rasDiagnosticsNvidiaDriverVersionData>(
    rasDiagnosticsNvidiaDriverVersionSummarize, {{MakeRank(23, 0, 1), NvidiaDriverVersion("")}});
  ExpectReporterFailure<rasDiagnosticsNvidiaDriverVersionData>(
    rasDiagnosticsNvidiaDriverVersionSummarize, {{MakeRank(24, 0, 1), NvidiaDriverVersion("580.82.07")}});
  ExpectReporterFailure<rasDiagnosticsNvidiaDriverVersionData>(
    rasDiagnosticsNvidiaDriverVersionSummarize,
    {{MakeRank(25, 0, 2), NvidiaDriverVersion("580.82.07")}, {MakeRank(25, 1, 2), NvidiaDriverVersion("580.95.05")}});

  ExpectReporterFailure<rasDiagnosticsEccData>(rasDiagnosticsEccSummarize,
                                               {{MakeRank(12, 0, 2), EccData(true)}});
  ExpectReporterFailure<rasDiagnosticsEccData>(rasDiagnosticsEccSummarize,
                                               {{MakeRank(13, 0, 1), EccData(false)}});
  ExpectReporterFailure<rasDiagnosticsEccData>(rasDiagnosticsEccSummarize,
                                               {{MakeRank(14, 0, 1), EccData(true)}});
  ExpectReporterFailure<rasDiagnosticsEccData>(
    rasDiagnosticsEccSummarize,
    {{MakeRank(15, 0, 2), EccData(true)}, {MakeRank(15, 1, 2), EccData(false)}});
  ExpectReporterFailure<rasDiagnosticsEccData>(rasDiagnosticsEccSummarize,
                                               {{MakeRank(16, 0, 1), EccData(true, 0, 1)}});
  g_eccThreshold = 5;
  ExpectReporterFailure<rasDiagnosticsEccData>(rasDiagnosticsEccSummarize,
                                               {{MakeRank(17, 0, 1), EccData(true, 5)}});

  ExpectReporterFailure<rasDiagnosticsNvLinkData>(rasDiagnosticsNvLinkSummarize,
                                                  {{MakeRank(18, 0, 2), NvLinkData(8, 0)}});
  ExpectReporterFailure<rasDiagnosticsNvLinkData>(rasDiagnosticsNvLinkSummarize,
                                                  {{MakeRank(19, 0, 1), NvLinkData(8, 0)}});
  ExpectReporterFailure<rasDiagnosticsNvLinkData>(
    rasDiagnosticsNvLinkSummarize,
    {{MakeRank(20, 0, 2), NvLinkData(8, 0)}, {MakeRank(20, 1, 2), NvLinkData(6, 0)}});
  ExpectReporterFailure<rasDiagnosticsNvLinkData>(rasDiagnosticsNvLinkSummarize,
                                                  {{MakeRank(21, 0, 1), NvLinkData(8, 1)}});
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuInitLoadsAmdSmi) {
  rasDiagnosticsGpuInit();
  EXPECT_EQ(1, g_amdSmiDiagInitCalls);
}

TEST_F(RasDiagnosticsGpuMicrotest, HealthyNodesReportOkForEveryCheck) {
  EXPECT_EQ(Lines{"[OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunGpuInventory());
  EXPECT_EQ(Lines{"[OK]   HIP driver version: 70253211 consistent across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunDriverVersion());
  EXPECT_EQ(Lines{"[OK]   AMD GPU driver version: 6.19.14.31400000 consistent across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunGpuDriverVersion());
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

TEST_F(RasDiagnosticsGpuMicrotest, GpuDriverVersionMismatchAcrossNodes) {
  nodes[1].gpuDriverVersion = "6.16.13.30300000";
  EXPECT_EQ(Lines{"[INFO] AMD GPU driver version: mismatch across 4 ranks in comm 0x5fa31c27a9e0d1b4, rank(s) {2,3} "
                  "differ from rank 0 (6.19.14.31400000)"},
            RunGpuDriverVersion());
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
  g_eccThreshold = 10;
  gpus[0].ecc.correctable = 9;
  gpus[1].ecc.correctable = 10;
  gpus[3].ecc.correctable = 12;
  EXPECT_EQ(Lines{"[INFO] ECC: corrected volatile errors at or above threshold 10 on rank(s) {1,3} (worst=12) "
                  "across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunEcc());
  EXPECT_FALSE(g_paramEnvs.empty());
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

TEST_F(RasDiagnosticsGpuMicrotest, PartitionAliasReadsThePhysicalGpu) {
  partitionFunction = 3;
  gpus[2].ecc.uncorrectable = 1;
  gpus[1].links.nDown = 1;
  EXPECT_EQ(Lines{"[OK]   GPU inventory: 8x AMD Instinct MI355X per node consistent across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunGpuInventory());
  EXPECT_EQ(Lines{"[INFO] ECC: uncorrected volatile errors on rank(s) {2} (worst=1) across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunEcc());
  EXPECT_EQ(Lines{"[INFO] XGMI: inactive link(s) on rank(s) {1} across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunXgmi());
  EXPECT_EQ(Lines{"[OK]   AMD GPU driver version: 6.19.14.31400000 consistent across 4 ranks in comm "
                  "0x5fa31c27a9e0d1b4"},
            RunGpuDriverVersion());
  EXPECT_EQ(4 * kRanks, aliasQueries);
}

TEST_F(RasDiagnosticsGpuMicrotest, PhysicalGpuUnknownToAmdSmiIsNotRetried) {
  gpus[3].unknownToAmdSmi = true;
  EXPECT_EQ(Lines{"[INFO] ECC: no uncorrected volatile errors across 3 of 4 ranks in comm 0x5fa31c27a9e0d1b4 (ECC "
                  "counters unavailable via AMD SMI on 1 ranks)"},
            RunEcc());
  EXPECT_EQ(0, aliasQueries);
}

TEST_F(RasDiagnosticsGpuMicrotest, AmdSmiUnavailableOnEveryNode) {
  for (FakeNode& node : nodes) node.amdSmiAvailable = false;
  EXPECT_EQ(Lines{"[INFO] GPU inventory: unavailable via AMD SMI across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunGpuInventory());
  EXPECT_EQ(Lines{"[OK]   HIP driver version: 70253211 consistent across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunDriverVersion());
  EXPECT_EQ(Lines{"[INFO] AMD GPU driver version: unavailable via AMD SMI across 4 ranks in comm 0x5fa31c27a9e0d1b4"},
            RunGpuDriverVersion());
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
