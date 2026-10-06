/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_gpu.cc.

#include <array>
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

int ncclNvmlDeviceCount = 0;
ncclNvmlDeviceInfo ncclNvmlDevices[ncclNvmlMaxDevices]{};
ncclNvmlDevicePairInfo ncclNvmlDevicePairs[ncclNvmlMaxDevices][ncclNvmlMaxDevices]{};

namespace {

ncclResult_t g_deviceCountResult = ncclSuccess;
unsigned int g_deviceCount = 0;
ncclResult_t g_handleResult = ncclSuccess;
ncclResult_t g_nameResult = ncclSuccess;
std::string g_deviceName;
std::array<std::array<unsigned long long, 2>, 2> g_eccCounters{};
std::array<std::array<ncclResult_t, 2>, 2> g_eccResults{};
struct NvLinkSample {
  nvmlEnableState_t state = NVML_FEATURE_ENABLED;
  unsigned int speedMBps = 50000;
  ncclResult_t result = ncclSuccess;
  nvmlReturn_t stateStatus = NVML_SUCCESS;
  nvmlReturn_t speedStatus = NVML_SUCCESS;
};
std::vector<NvLinkSample> g_nvLinks;
ncclResult_t g_nvLinkCountResult = ncclSuccess;
nvmlReturn_t g_nvLinkCountStatus = NVML_SUCCESS;
std::vector<unsigned int> g_handleIndices;
std::vector<int> g_nameDevices;
std::vector<int> g_eccDevices;
std::vector<nvmlMemoryErrorType_t> g_eccErrorTypes;
std::vector<nvmlEccCounterType_t> g_eccCounterTypes;
std::vector<nvmlMemoryLocation_t> g_eccLocations;
std::vector<int> g_nvLinkFieldDevices;
std::vector<std::pair<unsigned int, unsigned int>> g_nvLinkFieldQueries;

int DeviceIndex(nvmlDevice_t device) {
  return static_cast<int>(reinterpret_cast<uintptr_t>(device)) - 1;
}

int ErrorIndex(nvmlMemoryErrorType_t type) {
  return type == NVML_MEMORY_ERROR_TYPE_UNCORRECTED ? 1 : 0;
}

int LocationIndex(nvmlMemoryLocation_t location) {
  return location == NVML_MEMORY_LOCATION_DRAM ? 1 : 0;
}

}  // namespace

ncclResult_t DiagnosticsGpuTestCudaDriverVersion(int* version) {
  *version = 0x5EED;
  if (g_driverResult != ncclSuccess) return g_driverResult;
  *version = g_driverVersion;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetCount(unsigned int* deviceCount) {
  *deviceCount = 0x5EED;
  if (g_deviceCountResult != ncclSuccess) return g_deviceCountResult;
  *deviceCount = g_deviceCount;
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  g_handleIndices.push_back(index);
  *device = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(0x5EED));
  if (g_handleResult != ncclSuccess) return g_handleResult;
  *device = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index + 1));
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
  g_nameDevices.push_back(DeviceIndex(device));
  snprintf(name, length, "%s", "sentinel");
  if (g_nameResult != ncclSuccess) return g_nameResult;
  snprintf(name, length, "%s", g_deviceName.c_str());
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType,
                                                 nvmlEccCounterType_t counterType, nvmlMemoryLocation_t location,
                                                 unsigned long long* count) {
  g_eccDevices.push_back(DeviceIndex(device));
  g_eccErrorTypes.push_back(errorType);
  g_eccCounterTypes.push_back(counterType);
  g_eccLocations.push_back(location);
  *count = 0x5EED;
  const int error = ErrorIndex(errorType);
  const int memory = LocationIndex(location);
  if (g_eccResults[error][memory] != ncclSuccess) return g_eccResults[error][memory];
  *count = g_eccCounters[error][memory];
  return ncclSuccess;
}

ncclResult_t ncclNvmlDeviceGetFieldValues(nvmlDevice_t device, int count, nvmlFieldValue_t* values) {
  g_nvLinkFieldDevices.push_back(DeviceIndex(device));
  for (int i = 0; i < count; ++i) {
    auto& value = values[i];
    g_nvLinkFieldQueries.emplace_back(value.fieldId, value.scopeId);
    if (value.fieldId == NVML_FI_DEV_NVLINK_LINK_COUNT) {
      EXPECT_EQ(1, count);
      value.nvmlReturn = g_nvLinkCountStatus;
      value.value.uiVal = static_cast<unsigned int>(g_nvLinks.size());
      if (g_nvLinkCountResult != ncclSuccess) return g_nvLinkCountResult;
    } else {
      EXPECT_EQ(2, count);
      if (value.scopeId >= g_nvLinks.size()) {
        ADD_FAILURE() << "Unexpected NVLink scope " << value.scopeId;
        return ncclInternalError;
      }
      const auto& link = g_nvLinks[value.scopeId];
      if (link.result != ncclSuccess) return link.result;
      if (value.fieldId == NVML_FI_DEV_NVLINK_GET_STATE) {
        value.nvmlReturn = link.stateStatus;
        value.value.uiVal = link.state;
      } else {
        EXPECT_EQ(NVML_FI_DEV_NVLINK_GET_SPEED, value.fieldId);
        value.nvmlReturn = link.speedStatus;
        value.value.uiVal = link.speedMBps;
      }
    }
  }
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
  g_eccThreshold = 0;
  g_paramEnvs.clear();
  g_paramDefaults.clear();
  g_driverResult = ncclSuccess;
  g_driverVersion = 0;
  g_deviceCountResult = ncclSuccess;
  g_deviceCount = 0;
  g_handleResult = ncclSuccess;
  g_nameResult = ncclSuccess;
  g_deviceName.clear();
  ncclNvmlDeviceCount = 0;
  for (auto& byLocation : g_eccCounters) byLocation.fill(0);
  for (auto& byLocation : g_eccResults) byLocation.fill(ncclSuccess);
  g_nvLinks.clear();
  g_nvLinkCountResult = ncclSuccess;
  g_nvLinkCountStatus = NVML_SUCCESS;
  g_handleIndices.clear();
  g_nameDevices.clear();
  g_eccDevices.clear();
  g_eccErrorTypes.clear();
  g_eccCounterTypes.clear();
  g_eccLocations.clear();
  g_nvLinkFieldDevices.clear();
  g_nvLinkFieldQueries.clear();
}

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
};

}  // namespace

TEST_F(RasDiagnosticsGpuMicrotest, SummariesValidateArguments) {
  const std::array<rasDiagnosticsSummarizeFn, 4> summaries = {rasDiagnosticsGpuModelSummarize,
                                             rasDiagnosticsCudaDriverVersionSummarize,
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

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelCollectReportsKnownAndUnknownInventory) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  owned.comm->nvmlDev = 0;
  InstallNcclComms({owned.comm.get()});
  owned.comm->nvmlDev = 1;
  ncclNvmlDeviceCount = 2;
  g_deviceCount = 8;
  g_deviceName = "MI300X";
  rasDiagnosticsGpuModelData data{};

  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsGpuModelCollectLocal, &data));
  EXPECT_EQ(8, data.nGpus);
  EXPECT_STREQ("MI300X", data.model);
  ASSERT_EQ(1u, g_handleIndices.size());
  EXPECT_EQ(1u, g_handleIndices[0]);
  ASSERT_EQ(1u, g_nameDevices.size());
  EXPECT_EQ(1, g_nameDevices[0]);

  g_deviceCountResult = ncclSystemError;
  g_nameResult = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsGpuModelCollectLocal, &data));
  EXPECT_EQ(0, data.nGpus);
  EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);

  owned.comm->nvmlDev = -1;
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsGpuModelCollectLocal, &data));
  EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);
}

TEST_F(RasDiagnosticsGpuMicrotest, GpuModelCollectSkipsOutOfRangeDevicesAndHandlesLookupFailure) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  owned.comm->nvmlDev = 0;
  InstallNcclComms({owned.comm.get()});
  ncclNvmlDeviceCount = 2;
  g_deviceCount = 8;
  g_deviceName = "MI300X";
  rasDiagnosticsGpuModelData data{};

  for (int nvmlDev : {-1, ncclNvmlDeviceCount}) {
    owned.comm->nvmlDev = nvmlDev;
    data = GpuModel(1, "stale");
    ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsGpuModelCollectLocal, &data));
    EXPECT_EQ(8, data.nGpus);
    EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);
  }
  EXPECT_TRUE(g_handleIndices.empty());
  EXPECT_TRUE(g_nameDevices.empty());

  owned.comm->nvmlDev = 1;
  g_handleResult = ncclSystemError;
  data = GpuModel(1, "stale");
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsGpuModelCollectLocal, &data));
  EXPECT_EQ(8, data.nGpus);
  EXPECT_STREQ(RAS_DIAG_GPU_MODEL_UNKNOWN, data.model);
  ASSERT_EQ(1u, g_handleIndices.size());
  EXPECT_EQ(1u, g_handleIndices[0]);
  EXPECT_TRUE(g_nameDevices.empty());
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

TEST_F(RasDiagnosticsGpuMicrotest, EccCollectRequiresEveryCounter) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  owned.comm->nvmlDev = 0;
  InstallNcclComms({owned.comm.get()});
  owned.comm->nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_eccCounters = {{{{1, 2}}, {{3, 4}}}};
  rasDiagnosticsEccData data{};

  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsEccCollectLocal, &data));
  EXPECT_EQ(1u, data.correctedSram);
  EXPECT_EQ(2u, data.correctedDram);
  EXPECT_EQ(3u, data.uncorrectedSram);
  EXPECT_EQ(4u, data.uncorrectedDram);
  EXPECT_EQ(1, data.available);
  EXPECT_EQ((std::vector<int>{0, 0, 0, 0}), g_eccDevices);
  EXPECT_EQ((std::vector<nvmlMemoryErrorType_t>{NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                                NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                                NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                                NVML_MEMORY_ERROR_TYPE_UNCORRECTED}),
            g_eccErrorTypes);
  EXPECT_EQ((std::vector<nvmlMemoryLocation_t>{NVML_MEMORY_LOCATION_SRAM, NVML_MEMORY_LOCATION_SRAM,
                                               NVML_MEMORY_LOCATION_DRAM, NVML_MEMORY_LOCATION_DRAM}),
            g_eccLocations);
  for (nvmlEccCounterType_t counterType : g_eccCounterTypes) EXPECT_EQ(NVML_VOLATILE_ECC, counterType);

  const std::array<std::pair<int, int>, 4> failedSlots = {{{0, 0}, {1, 0}, {0, 1}, {1, 1}}};
  for (size_t slot = 0; slot < failedSlots.size(); slot++) {
    for (auto& byLocation : g_eccResults) byLocation.fill(ncclSuccess);
    g_eccResults[failedSlots[slot].first][failedSlots[slot].second] = ncclSystemError;
    data = EccData(true, 9, 9, 9, 9);

    ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsEccCollectLocal, &data));
    EXPECT_EQ(slot == 0 ? 0u : 1u, data.correctedSram);
    EXPECT_EQ(slot == 1 ? 0u : 3u, data.uncorrectedSram);
    EXPECT_EQ(slot == 2 ? 0u : 2u, data.correctedDram);
    EXPECT_EQ(slot == 3 ? 0u : 4u, data.uncorrectedDram);
    EXPECT_EQ(0, data.available);
  }

  for (auto& byLocation : g_eccResults) byLocation.fill(ncclSuccess);
  g_handleResult = ncclSystemError;
  data = EccData(true, 9, 9, 9, 9);
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsEccCollectLocal, &data));
  EXPECT_EQ(0u, data.correctedSram);
  EXPECT_EQ(0u, data.uncorrectedSram);
  EXPECT_EQ(0u, data.correctedDram);
  EXPECT_EQ(0u, data.uncorrectedDram);
  EXPECT_EQ(0, data.available);

  g_handleResult = ncclSuccess;
  g_handleIndices.clear();
  for (int nvmlDev : {-1, ncclNvmlDeviceCount}) {
    owned.comm->nvmlDev = nvmlDev;
    data = EccData(true, 9, 9, 9, 9);
    ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsEccCollectLocal, &data));
    EXPECT_EQ(0u, data.correctedSram);
    EXPECT_EQ(0u, data.uncorrectedSram);
    EXPECT_EQ(0u, data.correctedDram);
    EXPECT_EQ(0u, data.uncorrectedDram);
    EXPECT_EQ(0, data.available);
  }
  EXPECT_TRUE(g_handleIndices.empty());
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkCollectCountsValidAndInactiveLinks) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  owned.comm->nvmlDev = 0;
  InstallNcclComms({owned.comm.get()});
  owned.comm->nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_nvLinks.resize(4);
  g_nvLinks[1].state = NVML_FEATURE_DISABLED;
  g_nvLinks[2].speedMBps = 25000;
  g_nvLinks[3].result = ncclSystemError;
  rasDiagnosticsNvLinkData data{};

  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvLinkCollectLocal, &data));
  EXPECT_EQ(4, data.nLinks);
  EXPECT_EQ(2, data.nInactive);
  EXPECT_EQ(25000u, data.minSpeedMBps);
  EXPECT_EQ(50000u, data.maxSpeedMBps);
  EXPECT_EQ((std::vector<int>{0, 0, 0, 0, 0}), g_nvLinkFieldDevices);
  const std::vector<std::pair<unsigned int, unsigned int>> expectedQueries = {
    {NVML_FI_DEV_NVLINK_LINK_COUNT, 0},
    {NVML_FI_DEV_NVLINK_GET_STATE, 0}, {NVML_FI_DEV_NVLINK_GET_SPEED, 0},
    {NVML_FI_DEV_NVLINK_GET_STATE, 1}, {NVML_FI_DEV_NVLINK_GET_SPEED, 1},
    {NVML_FI_DEV_NVLINK_GET_STATE, 2}, {NVML_FI_DEV_NVLINK_GET_SPEED, 2},
    {NVML_FI_DEV_NVLINK_GET_STATE, 3}};
  EXPECT_EQ(expectedQueries, g_nvLinkFieldQueries);

  g_handleIndices.clear();
  for (int nvmlDev : {-1, ncclNvmlDeviceCount}) {
    owned.comm->nvmlDev = nvmlDev;
    data = NvLinkData(9, 9);
    ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvLinkCollectLocal, &data));
    EXPECT_EQ(0, data.nLinks);
    EXPECT_EQ(0, data.nInactive);
  }
  EXPECT_TRUE(g_handleIndices.empty());

  owned.comm->nvmlDev = 0;
  g_handleResult = ncclSystemError;
  g_nvLinkFieldDevices.clear();
  data = NvLinkData(9, 9);
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvLinkCollectLocal, &data));
  EXPECT_EQ(0, data.nLinks);
  EXPECT_EQ(0, data.nInactive);
  ASSERT_EQ(1u, g_handleIndices.size());
  EXPECT_EQ(0u, g_handleIndices[0]);
  EXPECT_TRUE(g_nvLinkFieldDevices.empty());
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkCollectHandlesUnavailableCountsAndUnreadableLinks) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  InstallNcclComms({owned.comm.get()});
  owned.comm->nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_nvLinks.resize(2);
  for (int failure = 0; failure < 4; ++failure) {
    SCOPED_TRACE(failure);
    g_nvLinkCountResult = failure == 0 ? ncclSystemError : ncclSuccess;
    g_nvLinkCountStatus = failure == 1 ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
    for (auto& link : g_nvLinks) {
      link.result = failure == 2 ? ncclSystemError : ncclSuccess;
      link.stateStatus = failure == 3 ? NVML_ERROR_UNKNOWN : NVML_SUCCESS;
    }
    rasDiagnosticsNvLinkData data = NvLinkData(9, 9);
    ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvLinkCollectLocal, &data));
    EXPECT_EQ(0, data.nLinks);
    EXPECT_EQ(0, data.nInactive);
    EXPECT_EQ(0u, data.minSpeedMBps);
    EXPECT_EQ(0u, data.maxSpeedMBps);
  }
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkCollectTreatsUnknownAndZeroSpeedsAsInactive) {
  OwnedComm owned(0x100, 0x101, 0x102, 0);
  owned.comm->nRanks = 1;
  InstallNcclComms({owned.comm.get()});
  owned.comm->nvmlDev = 0;
  ncclNvmlDeviceCount = 1;
  g_nvLinks.resize(3);
  g_nvLinks[0].speedStatus = NVML_ERROR_UNKNOWN;
  g_nvLinks[1].speedMBps = 0;
  rasDiagnosticsNvLinkData data{};
  ASSERT_NO_FATAL_FAILURE(CollectPayload(rasDiagnosticsNvLinkCollectLocal, &data));
  EXPECT_EQ(3, data.nLinks);
  EXPECT_EQ(2, data.nInactive);
  EXPECT_EQ(50000u, data.minSpeedMBps);
  EXPECT_EQ(50000u, data.maxSpeedMBps);
}

TEST_F(RasDiagnosticsGpuMicrotest, CollectLocalBuildsOneRecordForEachCheck) {
  OwnedComm comm(0x100, 0x101, 0x102, 0);
  comm.comm->nRanks = 1;
  comm.comm->nvmlDev = 0;
  InstallNcclComms({comm.comm.get()});
  ncclNvmlDeviceCount = 1;
  g_deviceCount = 1;
  g_deviceName = "MI300X";
  g_driverVersion = 70002000;

  const std::array<rasDiagnosticsCollectLocalFn, 4> collectors = {rasDiagnosticsGpuModelCollectLocal,
                                                                  rasDiagnosticsCudaDriverVersionCollectLocal,
                                                                  rasDiagnosticsEccCollectLocal,
                                                                  rasDiagnosticsNvLinkCollectLocal};
  const std::array<size_t, 4> payloadSizes = {sizeof(rasDiagnosticsGpuModelData),
                                              sizeof(rasDiagnosticsCudaDriverVersionData),
                                              sizeof(rasDiagnosticsEccData),
                                              sizeof(rasDiagnosticsNvLinkData)};
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
    } else {
      const auto* nvLink = reinterpret_cast<const rasDiagnosticsNvLinkData*>(payload);
      EXPECT_EQ(0, nvLink->nLinks);
      EXPECT_EQ(0, nvLink->nInactive);
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
  EXPECT_TRUE(Contains(state.lines, "unavailable via NVML"));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsGpuModelData>(
    {{MakeRank(8, 0, 1), GpuModel(0, RAS_DIAG_GPU_MODEL_UNKNOWN)}});
  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsGpuModelSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[INFO] GPU inventory: unavailable via NVML across 1 ranks in comm 0x8", state.lines[0]);

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
  EXPECT_EQ("[INFO] CUDA driver version: mismatch across 2 ranks in comm 0x5, rank(s) {1} differ from rank 0 "
            "(unavailable)",
            state.lines[0]);

  ExpectIncomplete(rasDiagnosticsCudaDriverVersionSummarize, "CUDA driver version", 4, DriverVersion(1));
}

TEST_F(RasDiagnosticsGpuMicrotest, EccSummaryCoversAvailabilityHealthyAndErrorCases) {
  auto records = BuildRecords<rasDiagnosticsEccData>({{MakeRank(1, 0, 1), EccData(false)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsEccSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "ECC: unavailable via NVML across 1 ranks"));

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
  EXPECT_TRUE(Contains(state.lines, "ECC: unavailable via NVML across 1 ranks in comm 0x7"));
}

TEST_F(RasDiagnosticsGpuMicrotest, NvLinkSummaryCoversNoLinksHealthyMismatchInactiveAndIncomplete) {
  auto records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(1, 0, 1), NvLinkData(0, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(state.lines.empty());

  records = BuildRecords<rasDiagnosticsNvLinkData>({{MakeRank(2, 0, 1), NvLinkData(8, 0)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[OK]   NVLink: 8 links per GPU, all active at consistent speed across 1 ranks in comm 0x2", state.lines[0]);

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

  ExpectIncomplete(rasDiagnosticsNvLinkSummarize, "NVLink", 20, NvLinkData(8, 0));

  state.lines.clear();
  records = BuildRecords<rasDiagnosticsNvLinkData>(
    {{MakeRank(6, 0, 1), NvLinkData(8, 0)}, {MakeRank(7, 0, 1), NvLinkData(8, 1)}});
  ASSERT_EQ(ncclSuccess, rasDiagnosticsNvLinkSummarize(&ctx, &reporter, records.data(), static_cast<int>(records.size())));
  EXPECT_TRUE(Contains(state.lines, "all active at consistent speed across 1 ranks in comm 0x6"));
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
    EXPECT_EQ("[INFO] NVLink: inconsistent link speeds on rank(s) {1} across 2 ranks in comm 0x1e", state.lines[0]);
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
