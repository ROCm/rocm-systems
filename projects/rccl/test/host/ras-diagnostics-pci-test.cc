/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/ras/diagnostics_pci.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../common/ProcessIsolatedTestRunner.hpp"
#include "alloc.h"
#include "comm.h"
#include "ScopedHook.h"
#include "fakes/diagnostics_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/ras_diagnostics_test_support.h"
#include "fakes/topo_net_stubs.h"
#include "fakes/topo_stubs.h"
#include "fakes/utils_fakes.h"
#include "graph/topo.h"
#include "ras/diagnostics_checks_common.h"

namespace {

int g_pciAllocationCalls = 0;
int g_pciFailAllocation = 0;

template <typename... Args>
ncclResult_t PciTestCalloc(Args&&... arguments) {
  if (++g_pciAllocationCalls == g_pciFailAllocation) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(arguments)..., __FILE__, __LINE__, __func__, false);
}

}

#undef ncclCalloc
#define ncclCalloc(...) PciTestCalloc(__VA_ARGS__)
#include "fakes/libc_seam.h"
#include RAS_DIAGNOSTICS_PCI_CC_PATH
#include "fakes/libc_seam_undef.h"
#undef ncclCalloc

namespace {

class RasDiagnosticsPciMicrotest : public ::testing::Test {
 protected:
  rasDiagnosticsContext context_{};
  ras_test::ReporterState output_;
  rasDiagnosticsReporter reporter_ = ras_test::MakeReporter(&output_);

  void SetUp() override {
    ResetLibcFakes();
    ResetDiagnosticsFakes();
    ResetTopoStubs();
    ResetUtilsFakes();
    EXPECT_EQ(RAS_DIAG_RDMA_TOPO_UNAVAILABLE, rasDiagnosticsRdmaTopoResult()->state);
    g_realpath = [](const char*, char*) -> char* { return nullptr; };
    g_pciAllocationCalls = 0;
    g_pciFailAllocation = 0;
  }

  void TearDown() override {
    ras_test::ResetNcclComms();
    ResetTopoStubs();
    ResetDiagnosticsFakes();
    ResetLibcFakes();
    g_pciFailAllocation = 0;
    ResetUtilsFakes();
  }

  template <typename Payload>
  std::vector<char> Records(const std::vector<Payload>& payloads) {
    const size_t stride = rasDiagnosticsLocalRecordStride(sizeof(Payload));
    std::vector<char> records(stride * payloads.size());
    for (size_t index = 0; index < payloads.size(); index++) {
      rasDiagnosticsRankHeader rank{};
      rank.commId = {0x123, 0x456, 0x789};
      rank.commRank = payloads.size() - index - 1;
      rank.commNRanks = payloads.size();
      std::memcpy(records.data() + index * stride, &rank, sizeof(rank));
      std::memcpy(records.data() + index * stride + sizeof(rank), &payloads[index], sizeof(Payload));
    }
    return records;
  }

  using Summarizer = ncclResult_t (*)(const rasDiagnosticsContext*, const rasDiagnosticsReporter*, const char*, int);

  static ncclResult_t NetProperties(ncclTopoNetPropertiesSnapshot** target, int* count, int device = 0) {
    *count = device + 1;
    *target = static_cast<ncclTopoNetPropertiesSnapshot*>(std::calloc(*count, sizeof(**target)));
    if (!*target) return ncclSystemError;
    (*target)[device].propertiesValid = true;
    std::strcpy((*target)[device].pciPath, "0000:81:00.0");
    return ncclSuccess;
  }

  void CheckReport(Summarizer summarize, const std::vector<char>& records,
                   const std::vector<std::string>& expected) {
    output_ = {};
    ASSERT_EQ(ncclSuccess, summarize(&context_, &reporter_, records.data(), records.size()));
    EXPECT_EQ(expected, output_.lines);
    for (size_t failure = 1; failure <= expected.size(); failure++) {
      SCOPED_TRACE(failure);
      output_ = {};
      output_.failCall = failure;
      output_.emitResult = ncclRemoteError;
      EXPECT_EQ(ncclRemoteError, summarize(&context_, &reporter_, records.data(), records.size()));
      EXPECT_EQ(failure, output_.calls);
      EXPECT_EQ((std::vector<std::string>(expected.begin(), expected.begin() + failure)), output_.lines);
    }
  }

  rasDiagnosticsRdmaTopoData Rdma(bool issue = false) {
    rasDiagnosticsRdmaTopoData data{};
    data.source = RAS_DIAG_RDMA_TOPO_SOURCE_TOOL;
    data.rdmaTopoState = RAS_DIAG_RDMA_TOPO_RAN;
    data.rdmaTopo.reportedIssue = issue;
    data.rdmaTopo.exitCode = issue ? 1 : 0;
    std::strcpy(data.rdmaTopo.sample, issue ? "FAIL\tlink" : "OK\tlink");
    return data;
  }

  rasDiagnosticsIommuData Iommu(rasDiagnosticsIommuMode mode = RAS_DIAG_IOMMU_MODE_PASSTHROUGH) {
    rasDiagnosticsIommuData data{};
    data.fallbackState = RAS_DIAG_PCI_FALLBACK_COLLECTED;
    data.nPairs = 1;
    std::strcpy(data.pairs[0].gpu.bdf, "0000:41:00.0");
    std::strcpy(data.pairs[0].nic.bdf, "0000:81:00.0");
    data.pairs[0].gpu.mode = data.pairs[0].nic.mode = mode;
    data.pairs[0].gpu.group = data.pairs[0].nic.group = {RAS_DIAG_IOMMU_GROUP_AVAILABLE, 37};
    return data;
  }

  rasDiagnosticsAtsData Ats(rasDiagnosticsAtsState state = RAS_DIAG_ATS_ON) {
    rasDiagnosticsAtsData data{};
    data.fallbackState = RAS_DIAG_PCI_FALLBACK_COLLECTED;
    data.nNics = 1;
    data.nics[0].state = state;
    std::strcpy(data.nics[0].bdf, "0000:81:00.0");
    return data;
  }
};

TEST(RasDiagnosticsPciIsolatedTest, Collect_ConclusiveToolResultSkipsFallbackAndUsesCache) {
  RUN_ISOLATED_TEST("Pci_ConclusiveToolResultSkipsFallbackAndUsesCache", []() {
    ResetLibcFakes();
    ResetDiagnosticsFakes();
    ResetTopoStubs();
    ras_test::OwnedComm comm(0x123, 0x456, 0x789, 0);
    ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
    ScopedHook child(g_ncclDiagChildRun,
      [](const char* command, int timeout, char* target, int size, bool* truncated) {
        EXPECT_STREQ("rdma_topo check", command);
        EXPECT_EQ(2, timeout);
        DeliverChildOutput("OK\tlink\n", target, size, nullptr, nullptr, truncated);
        return 0;
      });
    ScopedHook properties(g_ncclTopoGetNetPropertiesSnapshot,
      [](ncclComm*, bool, ncclTopoNetPropertiesSnapshot**, int*) {
        ADD_FAILURE() << "Conclusive rdma_topo must skip network snapshots";
        return ncclInternalError;
      });
    ScopedHook open(g_open, [](const char*, int, mode_t) {
      ADD_FAILURE() << "Conclusive rdma_topo must skip sysfs reads";
      return -1;
    });
    rasDiagnosticsContext context{};
    rasDiagnosticsLocalData data{};
    ASSERT_EQ(ncclSuccess, rasDiagnosticsRdmaTopoCollectLocal(&context, &data));
    ncclUniquePtr<char> rdmaRecords(data.records);
    ASSERT_EQ(1, data.nRecords);
    const auto* rdma = rasDiagnosticsRdmaTopoDataFromRecord(data.records);
    EXPECT_EQ(RAS_DIAG_RDMA_TOPO_SOURCE_TOOL, rdma->source);
    EXPECT_EQ(RAS_DIAG_RDMA_TOPO_RAN, rdma->rdmaTopoState);
    EXPECT_EQ(0, rdma->rdmaTopo.exitCode);
    EXPECT_FALSE(rdma->rdmaTopo.reportedIssue);
    EXPECT_STREQ("OK\tlink", rdma->rdmaTopo.sample);
    ASSERT_EQ(ncclSuccess, rasDiagnosticsIommuCollectLocal(&context, &data));
    ncclUniquePtr<char> iommuRecords(data.records);
    ASSERT_EQ(1, data.nRecords);
    EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_SKIPPED, rasDiagnosticsIommuDataFromRecord(data.records)->fallbackState);
    ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsCollectLocal(&context, &data));
    ncclUniquePtr<char> atsRecords(data.records);
    ASSERT_EQ(1, data.nRecords);
    EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_SKIPPED, rasDiagnosticsAtsDataFromRecord(data.records)->fallbackState);
    EXPECT_EQ(RAS_DIAG_RDMA_TOPO_RAN, rasDiagnosticsRdmaTopoResult()->state);
    EXPECT_EQ(1, child.calls);
    EXPECT_EQ(0, properties.calls);
    EXPECT_EQ(0, open.calls);
    ras_test::ResetNcclComms();
  });
}

TEST_F(RasDiagnosticsPciMicrotest, Bdf_ValidatesSeparatorsAndExtractsFinalPathComponent) {
  for (const char* valid : {"0000:81:00.0", "abcdef0:Ab:Cd.E"}) EXPECT_TRUE(rasDiagnosticsPciValidBdf(valid));
  for (const char* invalid : std::initializer_list<const char*>{
         nullptr, "", "000:81:00.0", "000000000:81:00.0", "0000/81:00.0",
         "0000:81/00.0", "0000:81:00/0", "0000:8g:00.0"}) {
    EXPECT_FALSE(rasDiagnosticsPciValidBdf(invalid));
  }
  char bdf[RAS_DIAG_PCI_BDF_BYTES] = {};
  ASSERT_TRUE(rasDiagnosticsPciBdfFromPath("/sys/devices/0000:81:00.0///", bdf, sizeof(bdf)));
  EXPECT_STREQ("0000:81:00.0", bdf);
  EXPECT_FALSE(rasDiagnosticsPciBdfFromPath("/", bdf, sizeof(bdf)));
  EXPECT_FALSE(rasDiagnosticsPciBdfFromPath(nullptr, bdf, sizeof(bdf)));
  EXPECT_FALSE(rasDiagnosticsPciBdfFromPath("0000:81:00.0", bdf, sizeof(bdf) - 1));
  char path[64];
  EXPECT_TRUE(rasDiagnosticsPciDevicePath(bdf, path, sizeof(path)));
  EXPECT_STREQ("/sys/bus/pci/devices/0000:81:00.0", path);
  EXPECT_FALSE(rasDiagnosticsPciDevicePath(bdf, path, 3));
}

TEST_F(RasDiagnosticsPciMicrotest, IommuGroup_UsesDirectoryRelativeLinkAndRejectsInvalidIds) {
  ScopedHook open(g_open, [&](const char* path, int flags, mode_t) {
    EXPECT_STREQ("/sys/bus/pci/devices/0000:81:00.0", path);
    EXPECT_EQ(O_RDONLY | O_DIRECTORY | O_CLOEXEC, flags);
    return 37;
  });
  for (const auto& item : std::vector<std::pair<std::string, int>>{
         {"../../../kernel/iommu_groups/0", 0}, {"/2147483647", INT32_MAX},
         {"/2147483648", -1}, {"/-1", -1}, {"/37x", -1}, {"/", -1}, {"/999999999999999999999", -1}}) {
    SCOPED_TRACE(item.first);
    ScopedHook link(g_readlinkat, [&](int fd, const char* name, char* target, size_t capacity) -> ssize_t {
      EXPECT_EQ(37, fd);
      EXPECT_STREQ("iommu_group", name);
      EXPECT_EQ(PATH_MAX - 1, capacity);
      std::memcpy(target, item.first.data(), item.first.size());
      return item.first.size();
    });
    const auto group = rasDiagnosticsPciReadIommuGroup("0000:81:00.0");
    EXPECT_EQ(item.second, group.id);
    EXPECT_EQ(item.second < 0 ? RAS_DIAG_IOMMU_GROUP_UNAVAILABLE : RAS_DIAG_IOMMU_GROUP_AVAILABLE, group.state);
    EXPECT_EQ(1, link.calls);
  }
  EXPECT_EQ(7, open.calls);
  EXPECT_EQ((std::vector<int>(7, 37)), g_closedFds);
}

TEST_F(RasDiagnosticsPciMicrotest, IommuGroup_MissingDeniedTruncatedAndOpenFailure) {
  ScopedHook open(g_open, [](const char*, int, mode_t) { return 38; });
  for (int error : {ENOENT, EACCES, EIO}) {
    ScopedHook link(g_readlinkat, [&](int, const char*, char*, size_t) -> ssize_t {
      errno = error;
      return -1;
    });
    const auto group = rasDiagnosticsPciReadIommuGroup("0000:81:00.0");
    EXPECT_EQ(error == ENOENT ? RAS_DIAG_IOMMU_GROUP_NONE : RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, group.state);
    EXPECT_EQ(-1, group.id);
    EXPECT_EQ(1, link.calls);
  }
  ScopedHook truncated(g_readlinkat, [&](int, const char*, char*, size_t capacity) -> ssize_t {
    return capacity;
  });
  EXPECT_EQ(RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, rasDiagnosticsPciReadIommuGroup("0000:81:00.0").state);
  g_open = [](const char*, int, mode_t) { return -1; };
  EXPECT_EQ(RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, rasDiagnosticsPciReadIommuGroup("0000:81:00.0").state);
  EXPECT_EQ(RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, rasDiagnosticsPciReadIommuGroup("bad").state);
  EXPECT_EQ(1, truncated.calls);
  EXPECT_EQ((std::vector<int>(4, 38)), g_closedFds);
}

TEST_F(RasDiagnosticsPciMicrotest, IommuMode_RetriesInterruptedReadsAndChecksCompleteInput) {
  rasDiagnosticsIommuGroup group{RAS_DIAG_IOMMU_GROUP_AVAILABLE, 93};
  ScopedHook open(g_open, [&](const char* path, int flags, mode_t) {
    EXPECT_STREQ("/sys/kernel/iommu_groups/93/type", path);
    EXPECT_EQ(O_RDONLY | O_CLOEXEC, flags);
    return 41;
  });
  for (const auto& item : std::vector<std::pair<std::string, rasDiagnosticsIommuMode>>{
         {"identity\n", RAS_DIAG_IOMMU_MODE_PASSTHROUGH}, {"DMA \t\n", RAS_DIAG_IOMMU_MODE_ON},
         {"DMA-FQ", RAS_DIAG_IOMMU_MODE_ON}, {"unknown", RAS_DIAG_IOMMU_MODE_UNAVAILABLE},
         {"", RAS_DIAG_IOMMU_MODE_UNAVAILABLE}}) {
    HookOf<decltype(g_read)> read(g_read, [&](int fd, void* target, size_t capacity) -> ssize_t {
      EXPECT_EQ(41, fd);
      if (read.calls == 1 || read.calls == 3) { errno = EINTR; return -1; }
      if (read.calls > 2) { EXPECT_EQ(1u, capacity); return 0; }
      EXPECT_EQ(31u, capacity);
      std::memcpy(target, item.first.data(), item.first.size());
      return item.first.size();
    });
    EXPECT_EQ(item.second, rasDiagnosticsIommuReadMode(&group));
    EXPECT_EQ(item.first.empty() || item.first.back() == '\n' ? 2 : 4, read.calls);
  }
  EXPECT_EQ(5, open.calls);
  EXPECT_EQ((std::vector<int>(5, 41)), g_closedFds);
  group.state = RAS_DIAG_IOMMU_GROUP_NONE;
  EXPECT_EQ(RAS_DIAG_IOMMU_MODE_OFF, rasDiagnosticsIommuReadMode(&group));
  group.state = RAS_DIAG_IOMMU_GROUP_UNAVAILABLE;
  EXPECT_EQ(RAS_DIAG_IOMMU_MODE_UNAVAILABLE, rasDiagnosticsIommuReadMode(&group));
  EXPECT_EQ(RAS_DIAG_IOMMU_MODE_UNAVAILABLE, rasDiagnosticsIommuReadMode(nullptr));
  EXPECT_EQ(5, open.calls);
}

TEST_F(RasDiagnosticsPciMicrotest, RdmaRun_ClassifiesToolResultsWithoutSpawningProcesses) {
  struct Case { const char* text; int status; bool truncated; rasDiagnosticsRdmaTopoState state; bool issue; };
  const Case cases[] = {
    {"OK\tlink\n", 0, false, RAS_DIAG_RDMA_TOPO_RAN, false},
    {"FAIL\tlink\n", 1, true, RAS_DIAG_RDMA_TOPO_RAN, true},
    {"OK\tlink\n", 0, true, RAS_DIAG_RDMA_TOPO_INCONCLUSIVE, false},
    {"No supported topology detected", 1, false, RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE, false},
    {"No ConnectX DMA Direct functions detected", 0, false, RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE, false},
    {"Need access to the PCI VPD", 1, false, RAS_DIAG_RDMA_TOPO_NEEDS_ROOT, false},
    {"Are You ROOT?", 1, false, RAS_DIAG_RDMA_TOPO_NEEDS_ROOT, false},
    {"E: cannot inspect", 0, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"not OK\tlink", 0, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"OK\tlink", 1, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"", 0, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"", 3, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"", 124, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"", -1, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
    {"FAIL\tlink", 128, false, RAS_DIAG_RDMA_TOPO_UNAVAILABLE, false},
  };
  for (const auto& item : cases) {
    SCOPED_TRACE(::testing::Message() << item.text << ':' << item.status);
    ScopedHook child(g_ncclDiagChildRun,
      [&](const char* command, int timeout, char* target, int size, bool* truncated) {
      EXPECT_STREQ("rdma_topo check", command);
      EXPECT_EQ(2, timeout);
      EXPECT_EQ(4096, size);
      DeliverChildOutput(item.text, target, size, nullptr, nullptr, truncated);
      *truncated = item.truncated;
      return item.status;
    });
    rasDiagnosticsRdmaTopoCache cache{};
    rasDiagnosticsRdmaTopoRun(&cache);
    EXPECT_EQ(1, child.calls);
    EXPECT_EQ(item.state, cache.state);
    EXPECT_EQ(item.status, cache.exitCode);
    EXPECT_EQ(item.issue, cache.reportedIssue);
    EXPECT_EQ(item.truncated, cache.outputTruncated);
    EXPECT_NE('\0', cache.sample[0]);
    if (item.status == 124) EXPECT_STREQ("rdma_topo timed out", cache.sample);
  }
}

TEST_F(RasDiagnosticsPciMicrotest, RdmaCache_ReusesResultWithoutRunningAnotherChild) {
  ScopedHook child(g_ncclDiagChildRun, [](const char*, int, char*, int, bool*) { return 0; });
  const auto* first = rasDiagnosticsRdmaTopoResult();
  EXPECT_EQ(first, rasDiagnosticsRdmaTopoResult());
  EXPECT_EQ(RAS_DIAG_RDMA_TOPO_UNAVAILABLE, first->state);
  EXPECT_EQ(0, child.calls);
}

TEST_F(RasDiagnosticsPciMicrotest, RdmaSample_PrefersIssueAndBoundsOutput) {
  char sample[20]{};
  rasDiagnosticsRdmaTopoSample(" first\n FAIL\tbroken \r\n last", sample, sizeof(sample));
  EXPECT_STREQ(" FAIL\tbroken", sample);
  rasDiagnosticsRdmaTopoSample(" first\n last \t\r\n", sample, sizeof(sample));
  EXPECT_STREQ(" last", sample);
  rasDiagnosticsRdmaTopoSample("abcdefghijklmnopqrstuvwxyz", sample, sizeof(sample));
  EXPECT_STREQ("abcdefghijklmnopqrs", sample);
  EXPECT_TRUE(rasDiagnosticsRdmaTopoHasLinePrefix("noise\nOK\tlink", "OK\t"));
  EXPECT_FALSE(rasDiagnosticsRdmaTopoHasLinePrefix("noise\n\t OK\tlink", "OK\t"));
  EXPECT_FALSE(rasDiagnosticsRdmaTopoHasLinePrefix("noise OK\tlink", "OK\t"));
}

TEST_F(RasDiagnosticsPciMicrotest, AddPair_ResolvesPhysicalDeviceAndDeduplicatesFullPair) {
  ScopedHook resolve(g_realpath, [&](const char* path, char* target) {
    EXPECT_STREQ("/sys/class/infiniband/mlx5_7/device", path);
    std::strcpy(target, "/sys/devices/0000:81:00.0");
    return target;
  });
  rasDiagnosticsPciCommSnapshot snapshot{};
  rasDiagnosticsPciAddPair(0x41000, "mlx5_7", "/wrong", &snapshot);
  rasDiagnosticsPciAddPair(0x41000, nullptr, "0000:81:00.0", &snapshot);
  rasDiagnosticsPciAddPair(0x42000, "bad/name", "0000:81:00.0", &snapshot);
  EXPECT_EQ(1, resolve.calls);
  ASSERT_EQ(2, snapshot.nPairs);
  EXPECT_EQ(0x41000, snapshot.pairs[0].gpuBusId);
  EXPECT_EQ(0x42000, snapshot.pairs[1].gpuBusId);
  EXPECT_STREQ("0000:81:00.0", snapshot.pairs[0].nicBdf);
  for (int index = 2; index < 17; index++) rasDiagnosticsPciAddPair(index, nullptr, "0000:81:00.0", &snapshot);
  EXPECT_EQ(16, snapshot.nPairs);
  EXPECT_EQ(1, snapshot.nPairsTruncated);
  rasDiagnosticsPciAddPair(99, nullptr, "/invalid", &snapshot);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
}

TEST_F(RasDiagnosticsPciMicrotest, AddNetDevice_ExpandsVirtualMembersAndMarksInvalidSelections) {
  std::array<ncclTopoNetPropertiesSnapshot, 3> properties{};
  for (auto& device : properties) device.propertiesValid = true;
  std::strcpy(properties[1].pciPath, "0000:81:00.0");
  std::strcpy(properties[2].pciPath, "0000:82:00.0");
  properties[0].vProps.ndevs = 2;
  properties[0].vProps.devs[0] = 2;
  properties[0].vProps.devs[1] = 1;
  rasDiagnosticsPciCommSnapshot snapshot{};
  rasDiagnosticsPciAddNetDevice(properties.data(), 3, 0, 0x41000, &snapshot);
  ASSERT_EQ(2, snapshot.nPairs);
  EXPECT_STREQ("0000:82:00.0", snapshot.pairs[0].nicBdf);
  EXPECT_STREQ("0000:81:00.0", snapshot.pairs[1].nicBdf);
  EXPECT_EQ(0, snapshot.netSelectionIncomplete);
  for (int index : {-1, 3}) {
    snapshot = {};
    rasDiagnosticsPciAddNetDevice(properties.data(), 3, index, 0x41000, &snapshot);
    EXPECT_EQ(0, snapshot.nPairs);
    EXPECT_EQ(1, snapshot.netSelectionIncomplete);
  }
  properties[0].vProps.devs[0] = 9;
  snapshot = {};
  rasDiagnosticsPciAddNetDevice(properties.data(), 3, 0, 0x41000, &snapshot);
  EXPECT_EQ(1, snapshot.nPairs);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
}

TEST_F(RasDiagnosticsPciMicrotest, SnapshotNet_UsesSelectedProxyGpuAndRejectsRemoteProxy) {
  auto comm = std::make_unique<ncclComm>();
  std::array<ncclPeerInfo, 3> peers{};
  comm->rank = 2;
  comm->nRanks = 3;
  comm->peerInfo = peers.data();
  peers[2].hostHash = peers[1].hostHash = 43;
  peers[1].busId = 0x51000;
  comm->graphs[NCCL_ALGO_RING].nChannels = 1;
  ScopedHook properties(g_ncclTopoGetNetPropertiesSnapshot,
    [&](ncclComm* actual, bool initialized, ncclTopoNetPropertiesSnapshot** target, int* count) {
      EXPECT_EQ(comm.get(), actual);
      EXPECT_TRUE(initialized);
      return NetProperties(target, count, 1);
    });
  ScopedHook select(g_ncclTopoGetNetDev,
    [&](ncclComm* actual, int rank, ncclTopoGraph* graph, int channel, int peer, int64_t* id, int* dev, int* proxy) {
      EXPECT_EQ(comm.get(), actual);
      EXPECT_EQ(2, rank);
      EXPECT_EQ(&comm->graphs[NCCL_ALGO_RING], graph);
      EXPECT_EQ(0, channel);
      EXPECT_EQ(-1, peer);
      EXPECT_EQ(nullptr, id);
      *dev = 1;
      *proxy = 1;
      return ncclSuccess;
    });
  rasDiagnosticsPciCommSnapshot snapshot{};
  rasDiagnosticsPciSnapshotNet(comm.get(), true, &snapshot);
  ASSERT_EQ(1, snapshot.nPairs);
  EXPECT_EQ(RAS_DIAG_PCI_NET_AVAILABLE, snapshot.netState);
  EXPECT_EQ(0x51000, snapshot.pairs[0].gpuBusId);
  EXPECT_STREQ("0000:81:00.0", snapshot.pairs[0].nicBdf);
  peers[1].hostHash = 44;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), true, &snapshot);
  EXPECT_EQ(0, snapshot.nPairs);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
  EXPECT_EQ(2, properties.calls);
  EXPECT_EQ(2, select.calls);
}

TEST_F(RasDiagnosticsPciMicrotest, SnapshotNet_HandlesMissingPropertiesAndNvlsHeadSelection) {
  auto comm = std::make_unique<ncclComm>();
  auto topology = std::make_unique<ncclTopoSystem>();
  std::array<ncclPeerInfo, 3> peers{};
  comm->rank = 2;
  comm->nRanks = 3;
  comm->peerInfo = peers.data();
  comm->topo = topology.get();
  peers[2].busId = 0x42000;
  ncclResult_t result = ncclRemoteError;
  int propertyCount = 0;
  ScopedHook properties(g_ncclTopoGetNetPropertiesSnapshot,
    [&](ncclComm* actual, bool initialized, ncclTopoNetPropertiesSnapshot** target, int* count) {
      EXPECT_EQ(comm.get(), actual);
      EXPECT_FALSE(initialized);
      *count = propertyCount;
      if (result == ncclSuccess && propertyCount) {
        return NetProperties(target, count);
      }
      return result;
    });
  std::vector<int> channels;
  ScopedHook select(g_ncclTopoGetNetDev,
    [&](ncclComm*, int rank, ncclTopoGraph* graph, int channel, int, int64_t*, int* dev, int* proxy) {
      EXPECT_EQ(2, rank);
      EXPECT_EQ(&comm->graphs[NCCL_ALGO_NVLS], graph);
      channels.push_back(channel);
      *dev = 0;
      *proxy = 2;
      return ncclSuccess;
    });
  rasDiagnosticsPciCommSnapshot snapshot{};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(RAS_DIAG_PCI_NET_UNAVAILABLE, snapshot.netState);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
  result = ncclSuccess;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(RAS_DIAG_PCI_NET_NO_NICS, snapshot.netState);
  propertyCount = 1;
  comm->graphs[NCCL_ALGO_RING].nChannels = MAXCHANNELS + 1;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
  comm->graphs[NCCL_ALGO_RING].nChannels = 0;
  comm->nvlsSupport = 1;
  auto& graph = comm->graphs[NCCL_ALGO_NVLS];
  graph.pattern = NCCL_TOPO_PATTERN_NVLS;
  graph.nChannels = 2;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(1, snapshot.netSelectionIncomplete);
  topology->nodes[GPU].count = 2;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(RAS_DIAG_PCI_NET_NO_NICS, snapshot.netState);
  EXPECT_EQ(0, select.calls);
  graph.intra[2] = 2;
  snapshot = {};
  rasDiagnosticsPciSnapshotNet(comm.get(), false, &snapshot);
  EXPECT_EQ(RAS_DIAG_PCI_NET_AVAILABLE, snapshot.netState);
  EXPECT_EQ(1, snapshot.nPairs);
  EXPECT_EQ((std::vector<int>{0, 1}), channels);
  EXPECT_EQ(6, properties.calls);
  EXPECT_EQ(2, select.calls);
}

TEST_F(RasDiagnosticsPciMicrotest, AtsParser_RestrictsCapabilitiesToExactDeviceBlock) {
  const char* listing = "0000:81:00.00 wrong\n\tATSCtl: Enable+\n"
                        "0000:81:00.0 target\n\tATSCtl: Enable-\n"
                        "0000:82:00.0 other\n\tATSCtl: Enable+\n";
  EXPECT_EQ(RAS_DIAG_ATS_OFF, rasDiagnosticsAtsStateFromLspci(listing, "0000:81:00.0"));
  EXPECT_EQ(RAS_DIAG_ATS_ON, rasDiagnosticsAtsStateFromLspci(listing, "0000:82:00.0"));
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE, rasDiagnosticsAtsStateFromLspci(listing, "0000:83:00.0"));
  EXPECT_EQ(RAS_DIAG_ATS_NEEDS_ROOT,
            rasDiagnosticsAtsStateFromLspci("0000:AB:00.0 NIC\n\tACCESS DENIED", "0000:ab:00.0"));
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE, rasDiagnosticsAtsStateFromLspci(nullptr, "0000:81:00.0"));
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE, rasDiagnosticsAtsStateFromLspci(listing, "bad"));
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE,
            rasDiagnosticsAtsStateFromLspci("0000:81:00.0 NIC\n0000:82:00.0 NIC\n\tATSCtl: Enable+", "0000:81:00.0"));
}

TEST_F(RasDiagnosticsPciMicrotest, AtsFill_DeduplicatesNicsAcrossGpusAndPreservesIncompleteFlags) {
  rasDiagnosticsPciCommSnapshot snapshot{};
  snapshot.netState = RAS_DIAG_PCI_NET_AVAILABLE;
  snapshot.nPairs = 2;
  snapshot.netSelectionIncomplete = snapshot.nPairsTruncated = 1;
  std::strcpy(snapshot.pairs[0].nicBdf, "0000:81:00.0");
  snapshot.pairs[1] = snapshot.pairs[0];
  snapshot.pairs[1].gpuBusId = 0x52000;
  rasDiagnosticsRdmaTopoCache cache{};
  cache.state = RAS_DIAG_RDMA_TOPO_UNAVAILABLE;
  rasDiagnosticsAtsData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsFillRecord(&snapshot, &cache,
    "0000:81:00.0 NIC\n\tATSCtl: Enable+", false, true, &data));
  EXPECT_EQ(1, data.nNics);
  EXPECT_EQ(RAS_DIAG_ATS_ON, data.nics[0].state);
  EXPECT_EQ(1, data.netSelectionIncomplete);
  EXPECT_EQ(1, data.netSelectionTruncated);
  EXPECT_EQ(1, data.lspciOutputTruncated);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsFillRecord(&snapshot, &cache,
    "0000:81:00.0 NIC\n\tATSCtl: Enable+", true, false, &data));
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE, data.nics[0].state);
  cache.state = RAS_DIAG_RDMA_TOPO_RAN;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsFillRecord(&snapshot, &cache, nullptr, false, false, &data));
  EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_SKIPPED, data.fallbackState);
  EXPECT_EQ(0, data.nNics);
  EXPECT_EQ(0, data.netSelectionIncomplete);
}

TEST_F(RasDiagnosticsPciMicrotest, RdmaSummary_PassFailureAndMixedResultsUseExactReports) {
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsRdmaTopoSummarize, Records(std::vector{Rdma(), Rdma()}),
    {"[OK]   rdma_topo check: passed on 2/2 ranks in comm 0x123"}));
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsRdmaTopoSummarize, Records(std::vector{Rdma(true), Rdma()}),
    {"[INFO] rdma_topo check: failed on rank(s) {1} across 2/2 ranks in comm 0x123; first rank 1 exit 1: FAIL\tlink"}));
  auto fallback = Rdma();
  fallback.source = RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK;
  fallback.rdmaTopoState = RAS_DIAG_RDMA_TOPO_NEEDS_ROOT;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsRdmaTopoSummarize, Records(std::vector{fallback, Rdma()}),
    {"[INFO] rdma_topo check: mixed results across 2 ranks in comm 0x123 (conclusive=1 not-usable=1)",
     "[INFO] rdma_topo check: passed on 1/2 ranks in comm 0x123",
     "[INFO] rdma_topo check: not usable on 1/2 ranks in comm 0x123 (needs root)"}));
}

TEST_F(RasDiagnosticsPciMicrotest, IommuSummary_MatchingAndMismatchedGroups) {
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{Iommu(), Iommu()}),
    {"[OK]   IOMMU mode: passthrough for 4 relevant device(s) across 2/2 ranks in comm 0x123",
     "[OK]   IOMMU groups: GPU and NIC groups match on 2 pair(s) across 2/2 ranks in comm 0x123"}));
  auto mismatch = Iommu(RAS_DIAG_IOMMU_MODE_ON);
  mismatch.pairs[0].nic.group.id = 39;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{mismatch}),
    {"[INFO] IOMMU mode: enabled (DMA remapping) for 2 relevant device(s) across 1/1 ranks in comm 0x123",
     "[INFO] IOMMU groups: GPU and NIC groups differ on 1 pair(s) in comm 0x123; "
     "first rank 0 GPU group 37 NIC 0000:81:00.0 group 39"}));
}

TEST_F(RasDiagnosticsPciMicrotest, AtsSummary_EnabledAndDisabledReportsDoNotContradictEachOther) {
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{Ats(), Ats()}),
    {"[OK]   ATS state: enabled on 2 rank/NIC pair(s) across 2 ranks in comm 0x123"}));
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{Ats(RAS_DIAG_ATS_OFF), Ats()}),
    {"[INFO] ATS state: disabled on 1 rank/NIC pair(s) across 1/2 ranks in comm 0x123",
     "[INFO] ATS state: ATSCtl disabled on 1 rank/NIC pair(s) in comm 0x123; first rank 1 NIC 0000:81:00.0"}));
}

TEST_F(RasDiagnosticsPciMicrotest, Summaries_RejectMalformedHeadersContextsAndAllocationFailure) {
  const Summarizer summaries[] = {rasDiagnosticsRdmaTopoSummarize, rasDiagnosticsIommuSummarize,
                                  rasDiagnosticsAtsSummarize};
  const std::vector<char> inputs[] = {
    Records(std::vector{Rdma()}), Records(std::vector{Iommu()}), Records(std::vector{Ats()})};
  for (size_t index = 0; index < std::size(summaries); index++) {
    SCOPED_TRACE(index);
    auto summarize = summaries[index];
    auto records = inputs[index];
    EXPECT_EQ(ncclInternalError, summarize(nullptr, &reporter_, records.data(), records.size()));
    EXPECT_EQ(ncclInternalError, summarize(&context_, nullptr, records.data(), records.size()));
    EXPECT_EQ(ncclInternalError, summarize(&context_, &reporter_, nullptr, records.size()));
    EXPECT_EQ(ncclInternalError, summarize(&context_, &reporter_, records.data(), records.size() - 1));
    EXPECT_EQ(ncclSuccess, summarize(&context_, &reporter_, nullptr, 0));
    context_.hasCommFilter = true;
    context_.commNRanks = 1;
    context_.commFilter = {0x123, 0x456, 0x788};
    EXPECT_EQ(ncclInternalError, summarize(&context_, &reporter_, records.data(), records.size()));
    context_ = {};
    auto* rank = reinterpret_cast<rasDiagnosticsRankHeader*>(records.data());
    rank->commRank = 1;
    EXPECT_EQ(ncclInternalError, summarize(&context_, &reporter_, records.data(), records.size()));
    records = inputs[index];
    records.insert(records.end(), inputs[index].begin(), inputs[index].end());
    EXPECT_EQ(ncclInternalError, summarize(&context_, &reporter_, records.data(), records.size()));
    g_pciFailAllocation = g_pciAllocationCalls + 1;
    EXPECT_EQ(ncclSystemError, summarize(&context_, &reporter_, inputs[index].data(), inputs[index].size()));
    g_pciFailAllocation = 0;
  }
  EXPECT_EQ(0, output_.calls);
}

TEST_F(RasDiagnosticsPciMicrotest, IommuSummary_MixedModesUnavailableGroupsAndDisabledDevices) {
  auto mixed = Iommu();
  mixed.pairs[0].nic.mode = RAS_DIAG_IOMMU_MODE_UNAVAILABLE;
  mixed.pairs[0].nic.group = {RAS_DIAG_IOMMU_GROUP_UNAVAILABLE, -1};
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{mixed}),
    {"[INFO] IOMMU mode: incomplete across 2 relevant device(s) on 1/1 ranks in comm 0x123",
     "[INFO] IOMMU mode: passthrough on 1 device(s); first rank 0 device 0000:41:00.0 in comm 0x123",
     "[INFO] IOMMU mode: unavailable on 1 device(s); first rank 0 device 0000:81:00.0 in comm 0x123",
     "[INFO] IOMMU groups: comparison unavailable on 1 GPU/NIC pair(s) in comm 0x123",
     "[INFO] IOMMU groups: unavailable on 1 pair(s) in comm 0x123; "
     "first rank 0 GPU group 37 NIC 0000:81:00.0 group -1"}));
  auto disabled = Iommu(RAS_DIAG_IOMMU_MODE_OFF);
  disabled.pairs[0].gpu.group = disabled.pairs[0].nic.group = {RAS_DIAG_IOMMU_GROUP_NONE, -1};
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{disabled}),
    {"[OK]   IOMMU mode: disabled for 2 relevant device(s) across 1/1 ranks in comm 0x123"}));
  mixed = Iommu();
  mixed.pairs[0].nic.mode = RAS_DIAG_IOMMU_MODE_ON;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{mixed}),
    {"[INFO] IOMMU mode: mismatch across 2 relevant device(s) on 1/1 ranks in comm 0x123",
     "[INFO] IOMMU mode: passthrough on 1 device(s); first rank 0 device 0000:41:00.0 in comm 0x123",
     "[INFO] IOMMU mode: enabled (DMA remapping) on 1 device(s); first rank 0 device 0000:81:00.0 in comm 0x123",
     "[OK]   IOMMU groups: GPU and NIC groups match on 1 pair(s) across 1/1 ranks in comm 0x123"}));
}

TEST_F(RasDiagnosticsPciMicrotest, IommuSummary_SharedDevicesAreCountedOncePerRank) {
  auto data = Iommu();
  data.nPairs = 3;
  data.pairs[1] = data.pairs[2] = data.pairs[0];
  std::strcpy(data.pairs[1].nic.bdf, "0000:82:00.0");
  std::strcpy(data.pairs[2].gpu.bdf, "0000:42:00.0");
  data.netSelectionIncomplete = 1;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{data}),
    {"[INFO] IOMMU mode: passthrough for 4 relevant device(s) across 1/1 ranks in comm 0x123",
     "[INFO] IOMMU groups: GPU and NIC groups match on 3 pair(s) across 1/1 ranks in comm 0x123"}));
}

TEST_F(RasDiagnosticsPciMicrotest, AtsSummary_ReportsRootUnavailableAndTruncatedInputs) {
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{Ats(RAS_DIAG_ATS_NEEDS_ROOT)}),
    {"[INFO] ATS state: PCIe capabilities require root access on 1 rank/NIC pair(s) across 1/1 ranks in comm 0x123",
     "[INFO] ATS state: PCIe capabilities inaccessible on 1 rank/NIC pair(s) in comm 0x123; "
     "first rank 0 NIC 0000:81:00.0"}));
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{Ats(RAS_DIAG_ATS_UNAVAILABLE)}),
    {"[INFO] ATS state: unavailable across 1/1 ranks in comm 0x123",
     "[INFO] ATS state: ATSCtl unavailable on 1 rank/NIC pair(s) in comm 0x123; first rank 0 NIC 0000:81:00.0"}));
  auto data = Ats(RAS_DIAG_ATS_UNAVAILABLE);
  data.lspciUnavailable = data.lspciOutputTruncated = data.netSelectionTruncated = data.netSelectionIncomplete = 1;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{data}),
    {"[INFO] ATS state: unavailable across 1/1 ranks in comm 0x123",
     "[INFO] ATS state: lspci unavailable on rank(s) {0} in comm 0x123",
     "[INFO] ATS state: lspci output truncated on rank(s) {0} in comm 0x123",
     "[INFO] ATS state: network device selection truncated on rank(s) {0} in comm 0x123",
     "[INFO] ATS state: network device selection incomplete on rank(s) {0} in comm 0x123"}));
}

TEST_F(RasDiagnosticsPciMicrotest, FallbackSummaries_DistinguishSkippedMissingAndNotReady) {
  for (auto state : {RAS_DIAG_PCI_FALLBACK_SKIPPED, RAS_DIAG_PCI_FALLBACK_NO_NICS,
                    RAS_DIAG_PCI_FALLBACK_UNAVAILABLE, RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY}) {
    SCOPED_TRACE(state);
    rasDiagnosticsIommuData iommu{};
    rasDiagnosticsAtsData ats{};
    iommu.fallbackState = ats.fallbackState = state;
    std::vector<std::string> iommuLines, atsLines;
    if (state == RAS_DIAG_PCI_FALLBACK_UNAVAILABLE) {
      iommuLines = {"[INFO] IOMMU mode: unable to identify relevant GPU/NIC pairs on 1/1 ranks in comm 0x123"};
      atsLines = {"[INFO] ATS state: unable to identify relevant NICs on 1/1 ranks in comm 0x123"};
    } else if (state == RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY) {
      iommuLines = {"[INFO] IOMMU mode: incomplete for comm 0x123: "
                    "topology-dependent data was unavailable for 1 of 1 ranks"};
      atsLines = {"[INFO] ATS state: incomplete for comm 0x123: "
                  "topology-dependent data was unavailable for 1 of 1 ranks"};
    }
    ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsIommuSummarize, Records(std::vector{iommu}), iommuLines));
    ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{ats}), atsLines));
  }
}

TEST_F(RasDiagnosticsPciMicrotest, Summaries_MissingRanksSuppressCompleteResults) {
  const Summarizer summaries[] = {
    rasDiagnosticsRdmaTopoSummarize, rasDiagnosticsIommuSummarize, rasDiagnosticsAtsSummarize};
  std::vector<char> records[] = {
    Records(std::vector{Rdma()}), Records(std::vector{Iommu()}), Records(std::vector{Ats()})};
  const char* names[] = {"rdma_topo check", "IOMMU mode", "ATS state"};
  for (size_t index = 0; index < std::size(summaries); index++) {
    reinterpret_cast<rasDiagnosticsRankHeader*>(records[index].data())->commNRanks = 2;
    ASSERT_NO_FATAL_FAILURE(CheckReport(summaries[index], records[index],
      {std::string("[INFO] ") + names[index] +
        ": diagnostics incomplete, gathered 1/2 ranks in comm 0x123/0x456/0x789"}));
  }
}

TEST_F(RasDiagnosticsPciMicrotest, Payloads_RejectContradictoryStatesAndDuplicateDevices) {
  auto rdma = Rdma();
  rdma.rdmaTopo.exitCode = 124;
  EXPECT_FALSE(rasDiagnosticsRdmaTopoPayloadValid(&rdma));
  rdma = Rdma();
  std::memset(rdma.rdmaTopo.sample, 'x', sizeof(rdma.rdmaTopo.sample));
  EXPECT_FALSE(rasDiagnosticsRdmaTopoPayloadValid(&rdma));
  auto iommu = Iommu();
  iommu.pairs[0].gpu.mode = RAS_DIAG_IOMMU_MODE_OFF;
  EXPECT_FALSE(rasDiagnosticsIommuPayloadValid(&iommu));
  iommu = Iommu();
  iommu.pairs[1] = iommu.pairs[0];
  iommu.nPairs = 2;
  EXPECT_FALSE(rasDiagnosticsIommuPayloadValid(&iommu));
  auto ats = Ats();
  ats.lspciUnavailable = 1;
  EXPECT_FALSE(rasDiagnosticsAtsPayloadValid(&ats));
  ats = Ats();
  ats.nics[1] = ats.nics[0];
  ats.nNics = 2;
  EXPECT_FALSE(rasDiagnosticsAtsPayloadValid(&ats));
}

TEST_F(RasDiagnosticsPciMicrotest, Summaries_RejectMalformedPayloadsBeforeReporting) {
  auto rdma = Rdma();
  rdma.source = RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK;
  auto iommu = Iommu();
  iommu.nPairsTruncated = 1;
  auto ats = Ats();
  ats.nNics = RAS_DIAG_PCI_MAX_NET_DEVS + 1;
  const Summarizer summaries[] = {
    rasDiagnosticsRdmaTopoSummarize, rasDiagnosticsIommuSummarize, rasDiagnosticsAtsSummarize};
  const std::vector<char> records[] = {
    Records(std::vector{rdma}), Records(std::vector{iommu}), Records(std::vector{ats})};
  for (size_t index = 0; index < std::size(summaries); index++) {
    SCOPED_TRACE(index);
    EXPECT_EQ(ncclInternalError, summaries[index](&context_, &reporter_, records[index].data(), records[index].size()));
  }
  EXPECT_EQ(0, output_.calls);
  iommu = Iommu();
  iommu.pairs[0].gpu.bdf[0] = 'x';
  EXPECT_FALSE(rasDiagnosticsIommuPayloadValid(&iommu));
  iommu.nPairs = RAS_DIAG_PCI_MAX_GPU_NIC_PAIRS + 1;
  EXPECT_FALSE(rasDiagnosticsIommuPayloadValid(&iommu));
}

TEST_F(RasDiagnosticsPciMicrotest, RdmaSummary_UniformAndMixedFallbackReasons) {
  const std::pair<rasDiagnosticsRdmaTopoState, const char*> cases[] = {
    {RAS_DIAG_RDMA_TOPO_NOT_APPLICABLE, "not applicable"}, {RAS_DIAG_RDMA_TOPO_NEEDS_ROOT, "needs root"},
    {RAS_DIAG_RDMA_TOPO_INCONCLUSIVE, "inconclusive"}, {RAS_DIAG_RDMA_TOPO_UNAVAILABLE, "unavailable"}};
  for (const auto& item : cases) {
    auto data = Rdma();
    data.source = RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK;
    data.rdmaTopoState = item.first;
    ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsRdmaTopoSummarize, Records(std::vector{data}),
      {std::string("[INFO] rdma_topo check: not usable on 1/1 ranks in comm 0x123 (") + item.second + ')'}));
  }
  auto first = Rdma(), second = Rdma();
  first.source = second.source = RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK;
  first.rdmaTopoState = RAS_DIAG_RDMA_TOPO_NEEDS_ROOT;
  second.rdmaTopoState = RAS_DIAG_RDMA_TOPO_UNAVAILABLE;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsRdmaTopoSummarize, Records(std::vector{first, second}),
    {"[INFO] rdma_topo check: not usable on 2/2 ranks in comm 0x123 (mixed results)"}));
}

TEST_F(RasDiagnosticsPciMicrotest, AtsSummary_PartialSelectionDoesNotReportOk) {
  rasDiagnosticsAtsData missing{};
  missing.fallbackState = RAS_DIAG_PCI_FALLBACK_UNAVAILABLE;
  ASSERT_NO_FATAL_FAILURE(CheckReport(rasDiagnosticsAtsSummarize, Records(std::vector{missing, Ats()}),
    {"[INFO] ATS state: enabled on 1 rank/NIC pair(s) across 1/2 ranks in comm 0x123",
     "[INFO] ATS state: unable to identify relevant NICs on 1 rank(s) in comm 0x123"}));
}

TEST_F(RasDiagnosticsPciMicrotest, Collect_NoCommunicatorsAndInvalidArguments) {
  rasDiagnosticsLocalData data{};
  for (auto collect :
       {rasDiagnosticsRdmaTopoCollectLocal, rasDiagnosticsIommuCollectLocal, rasDiagnosticsAtsCollectLocal}) {
    EXPECT_EQ(ncclInternalError, collect(nullptr, &data));
    EXPECT_EQ(ncclInternalError, collect(&context_, nullptr));
    ASSERT_EQ(ncclSuccess, collect(&context_, &data));
    EXPECT_EQ(0, data.nRecords);
    EXPECT_EQ(nullptr, data.records);
  }
}

TEST_F(RasDiagnosticsPciMicrotest, Collect_SelectedNicDrivesIommuAndAtsPayloads) {
  ras_test::OwnedComm owned(0x123, 0x456, 0x789, 0);
  owned.comm->nRanks = 1;
  owned.comm->initState = ncclSuccess;
  owned.comm->graphs[NCCL_ALGO_TREE].nChannels = 1;
  owned.peers[0].busId = 0x41000;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({owned.comm.get()}));
  ScopedHook properties(g_ncclTopoGetNetPropertiesSnapshot,
    [&](ncclComm* comm, bool initialized, ncclTopoNetPropertiesSnapshot** target, int* count) {
      EXPECT_EQ(owned.comm.get(), comm);
      EXPECT_TRUE(initialized);
      return NetProperties(target, count);
    });
  ScopedHook select(g_ncclTopoGetNetDev,
    [&](ncclComm* comm, int rank, ncclTopoGraph* graph, int channel, int peer, int64_t* id, int* dev, int* proxy) {
      EXPECT_EQ(owned.comm.get(), comm);
      EXPECT_EQ(0, rank);
      EXPECT_EQ(&owned.comm->graphs[NCCL_ALGO_TREE], graph);
      EXPECT_EQ(0, channel);
      EXPECT_EQ(-1, peer);
      EXPECT_EQ(nullptr, id);
      *dev = *proxy = 0;
      return ncclSuccess;
    });
  std::vector<std::string> opened;
  ScopedHook open(g_open, [&](const char* path, int flags, mode_t) {
    EXPECT_EQ(O_RDONLY | O_DIRECTORY | O_CLOEXEC, flags);
    opened.emplace_back(path);
    return 51;
  });
  ScopedHook link(g_readlinkat, [](int fd, const char* name, char*, size_t) -> ssize_t {
    EXPECT_EQ(51, fd);
    EXPECT_STREQ("iommu_group", name);
    errno = ENOENT;
    return -1;
  });
  rasDiagnosticsLocalData data{};
  ScopedHook busId(g_int64ToBusId, [](int64_t id, char* output) {
    EXPECT_EQ(0x41000, id);
    std::strcpy(output, "0000:41:00.0");
    return ncclSuccess;
  });
  ASSERT_EQ(ncclSuccess, rasDiagnosticsIommuCollectLocal(&context_, &data));
  ncclUniquePtr<char> iommuRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  const auto* iommu = rasDiagnosticsIommuDataFromRecord(data.records);
  EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_COLLECTED, iommu->fallbackState);
  ASSERT_EQ(1, iommu->nPairs);
  EXPECT_STREQ("0000:41:00.0", iommu->pairs[0].gpu.bdf);
  EXPECT_STREQ("0000:81:00.0", iommu->pairs[0].nic.bdf);
  EXPECT_EQ(RAS_DIAG_IOMMU_MODE_OFF, iommu->pairs[0].gpu.mode);
  EXPECT_EQ(RAS_DIAG_IOMMU_MODE_OFF, iommu->pairs[0].nic.mode);
  EXPECT_EQ((std::vector<std::string>{
              "/sys/bus/pci/devices/0000:41:00.0", "/sys/bus/pci/devices/0000:81:00.0"}), opened);
  EXPECT_EQ(2, link.calls);
  EXPECT_EQ(1, busId.calls);
  EXPECT_EQ((std::vector<int>{51, 51}), g_closedFds);
  int lspciExit = 0;
  ScopedHook child(g_ncclDiagChildRun, [&](const char* command, int timeout, char* target, int size, bool* truncated) {
    EXPECT_STREQ("lspci -D -vvv", command);
    EXPECT_EQ(2, timeout);
    EXPECT_EQ(1024 * 1024, size);
    DeliverChildOutput("0000:81:00.0 NIC\n\tATSCtl: Enable+\n", target, size, nullptr, nullptr, truncated);
    return lspciExit;
  });
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsCollectLocal(&context_, &data));
  ncclUniquePtr<char> atsRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  const auto* ats = rasDiagnosticsAtsDataFromRecord(data.records);
  EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_COLLECTED, ats->fallbackState);
  ASSERT_EQ(1, ats->nNics);
  EXPECT_STREQ("0000:81:00.0", ats->nics[0].bdf);
  EXPECT_EQ(RAS_DIAG_ATS_ON, ats->nics[0].state);
  EXPECT_EQ(0, ats->lspciUnavailable);
  EXPECT_EQ(1, child.calls);
  EXPECT_EQ(2, properties.calls);
  EXPECT_EQ(2, select.calls);
  lspciExit = 1;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsCollectLocal(&context_, &data));
  ncclUniquePtr<char> failedAtsRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  const auto* failedAts = rasDiagnosticsAtsDataFromRecord(data.records);
  ASSERT_EQ(1, failedAts->nNics);
  EXPECT_EQ(RAS_DIAG_ATS_UNAVAILABLE, failedAts->nics[0].state);
  EXPECT_EQ(1, failedAts->lspciUnavailable);
  EXPECT_EQ(2, child.calls);
  for (int failCall : {1, 2, 3}) {
    SCOPED_TRACE(failCall);
    g_pciAllocationCalls = 0;
    g_pciFailAllocation = failCall;
    const int previousCalls = child.calls;
    EXPECT_EQ(ncclSystemError, rasDiagnosticsAtsCollectLocal(&context_, &data));
    EXPECT_EQ(nullptr, data.records);
    EXPECT_EQ(0, data.nRecords);
    EXPECT_EQ(previousCalls + (failCall == 3), child.calls);
  }
}

TEST_F(RasDiagnosticsPciMicrotest, Collect_RecordsTopologyNotReadyAndFiltersCommunicatorIdentity) {
  ras_test::OwnedComm selected(0x123, 0x456, 0x789, 0);
  ras_test::OwnedComm ignored(0x123, 0x456, 0x788, 0);
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({ignored.comm.get(), nullptr, selected.comm.get()}));
  context_.hasCommFilter = true;
  context_.commFilter = {0x123, 0x456, 0x789};
  context_.commNRanks = 8;
  ScopedHook properties(g_ncclTopoGetNetPropertiesSnapshot,
    [&](ncclComm* comm, bool initialized, ncclTopoNetPropertiesSnapshot**, int*) {
      EXPECT_EQ(selected.comm.get(), comm);
      EXPECT_FALSE(initialized);
      return ncclInProgress;
    });
  rasDiagnosticsLocalData data{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsIommuCollectLocal(&context_, &data));
  ncclUniquePtr<char> iommuRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY, rasDiagnosticsIommuDataFromRecord(data.records)->fallbackState);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsAtsCollectLocal(&context_, &data));
  ncclUniquePtr<char> atsRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  EXPECT_EQ(RAS_DIAG_PCI_FALLBACK_TOPOLOGY_NOT_READY, rasDiagnosticsAtsDataFromRecord(data.records)->fallbackState);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsRdmaTopoCollectLocal(&context_, &data));
  ncclUniquePtr<char> rdmaRecords(data.records);
  ASSERT_EQ(1, data.nRecords);
  EXPECT_EQ(RAS_DIAG_RDMA_TOPO_SOURCE_FALLBACK, rasDiagnosticsRdmaTopoDataFromRecord(data.records)->source);
  EXPECT_EQ(2, properties.calls);
  for (int failCall : {1, 2}) {
    g_pciAllocationCalls = 0;
    g_pciFailAllocation = failCall;
    EXPECT_EQ(ncclSystemError, rasDiagnosticsIommuCollectLocal(&context_, &data));
    EXPECT_EQ(nullptr, data.records);
    EXPECT_EQ(0, data.nRecords);
  }
}

}
