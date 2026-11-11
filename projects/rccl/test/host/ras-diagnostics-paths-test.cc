/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/ras/diagnostics_paths.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alloc.h"
#include "comm.h"
#include "fakes/ras_diagnostics_test_support.h"
#include "graph/topo.h"
#include "ras/diagnostics_checks_common.h"

namespace {

bool g_failPathsAllocation = false;

template <typename... Args>
ncclResult_t PathsTestCalloc(Args&&... arguments) {
  if (g_failPathsAllocation) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(arguments)..., __FILE__, __LINE__, __func__, false);
}

}

#undef ncclCalloc
#define ncclCalloc(...) PathsTestCalloc(__VA_ARGS__)
#include RAS_DIAGNOSTICS_PATHS_CC_PATH
#undef ncclCalloc

namespace {

class RasDiagnosticsPathsMicrotest : public ::testing::Test {
 protected:
  ras_test::ReporterState output_;
  rasDiagnosticsReporter reporter_ = ras_test::MakeReporter(&output_);
  rasDiagnosticsContext context_{};

  struct Row {
    int rank;
    uint32_t mask;
    rasDiagnosticsPathsState state = RAS_DIAG_PATHS_AVAILABLE;
    rasCommId commId{0x123, 0x456, 0x789};
    int nRanks = 2;
  };

  void TearDown() override {
    ras_test::ResetNcclComms();
    g_failPathsAllocation = false;
  }

  std::vector<char> Records(const std::vector<Row>& rows) {
    std::vector<std::pair<rasDiagnosticsRankHeader, rasDiagnosticsPathsData>> records;
    for (const auto& row : rows) {
      rasDiagnosticsRankHeader rank{};
      rank.commId = row.commId;
      rank.commRank = row.rank;
      rank.commNRanks = row.nRanks;
      records.push_back({rank, {row.mask, row.state}});
    }
    return ras_test::BuildRecords(records);
  }

  ncclResult_t Summarize(const std::vector<Row>& rows) {
    const auto records = Records(rows);
    return rasDiagnosticsPathsSummarize(&context_, &reporter_, records.data(), records.size());
  }
};

TEST_F(RasDiagnosticsPathsMicrotest, Usage_MapsRankThroughTopologyAndClassifiesRemoteAndDisconnectedPeers) {
  auto topology = std::make_unique<ncclTopoSystem>();
  topology->nodes[GPU].count = 3;
  topology->nodes[GPU].nodes[0].gpu.rank = 3;
  topology->nodes[GPU].nodes[1].gpu.rank = 2;
  topology->nodes[GPU].nodes[2].gpu.rank = 0;
  std::array<ncclTopoLinkList, 3> paths{};
  paths[0].type = PATH_PIX;
  paths[2].type = PATH_DIS + 1;
  topology->nodes[GPU].nodes[1].paths[GPU] = paths.data();
  rasDiagnosticsPathsData data{};

  rasDiagnosticsPathsUsage(topology.get(), 2, 4, &data);

  EXPECT_EQ(RAS_DIAG_PATHS_AVAILABLE, data.state);
  EXPECT_EQ((1u << PATH_LOC) | (1u << PATH_PIX) | (1u << PATH_NET) | (1u << PATH_DIS), data.pathMask);
  paths[2].type = -1;
  rasDiagnosticsPathsUsage(topology.get(), 2, 4, &data);
  EXPECT_EQ((1u << PATH_LOC) | (1u << PATH_PIX) | (1u << PATH_NET) | (1u << PATH_DIS), data.pathMask);
}

TEST_F(RasDiagnosticsPathsMicrotest, Usage_MissingPathTableIsDisconnected) {
  auto topology = std::make_unique<ncclTopoSystem>();
  topology->nodes[GPU].count = 2;
  topology->nodes[GPU].nodes[0].gpu.rank = 1;
  topology->nodes[GPU].nodes[1].gpu.rank = 0;
  rasDiagnosticsPathsData data{};

  rasDiagnosticsPathsUsage(topology.get(), 0, 2, &data);

  EXPECT_EQ(RAS_DIAG_PATHS_AVAILABLE, data.state);
  EXPECT_EQ((1u << PATH_LOC) | (1u << PATH_DIS), data.pathMask);
}

TEST_F(RasDiagnosticsPathsMicrotest, Usage_MissingOwnGpuClearsMask) {
  auto topology = std::make_unique<ncclTopoSystem>();
  for (auto* candidate : {topology.get(), static_cast<ncclTopoSystem*>(nullptr)}) {
    rasDiagnosticsPathsData data{UINT32_MAX, RAS_DIAG_PATHS_AVAILABLE};
    rasDiagnosticsPathsUsage(candidate, 3, 8, &data);
    EXPECT_EQ(RAS_DIAG_PATHS_LOCAL_GPU_MISSING, data.state);
    EXPECT_EQ(0u, data.pathMask);
  }
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_FiltersRegistryAndPacksOnlyInitializedRanks) {
  ras_test::OwnedComm ready(0x123, 0x456, 0x789, 3);
  ras_test::OwnedComm failed(0x123, 0x456, 0x789, 1);
  ras_test::OwnedComm invalid(0x123, 0x456, 0x789, 0, false);
  ready.comm->initState = ncclSuccess;
  failed.comm->initState = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({nullptr, failed.comm.get(), invalid.comm.get(), ready.comm.get()}));
  rasDiagnosticsLocalData local{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  std::unique_ptr<char, decltype(&std::free)> owner(local.records, std::free);

  ASSERT_EQ(1, local.nRecords);
  EXPECT_EQ(local.recordStride, local.recordsBytes);
  const auto* rank = rasDiagnosticsRankHeaderFromRecord(local.records);
  EXPECT_EQ(3, rank->commRank);
  EXPECT_EQ(8, rank->commNRanks);
  EXPECT_EQ(0x123u, rank->commId.commHash);
  EXPECT_EQ(0x456u, rank->commId.hostHash);
  EXPECT_EQ(0x789u, rank->commId.pidHash);
  EXPECT_EQ(RAS_DIAG_PATHS_LOCAL_GPU_MISSING, rasDiagnosticsPathsDataFromRecord(local.records)->state);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_FilteredInProgressCommUsesAvailableTopology) {
  ras_test::OwnedComm selected(0x123, 0x456, 0x789, 0);
  ras_test::OwnedComm other(0x123, 0x456, 0x790, 0);
  auto topology = std::make_unique<ncclTopoSystem>();
  topology->nodes[GPU].count = 1;
  topology->nodes[GPU].nodes[0].gpu.rank = 0;
  selected.comm->topo = topology.get();
  selected.comm->nRanks = 1;
  selected.comm->initState = ncclInProgress;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({other.comm.get(), selected.comm.get()}));
  context_.hasCommFilter = true;
  context_.commFilter = {0x123, 0x456, 0x789};
  rasDiagnosticsLocalData local{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  std::unique_ptr<char, decltype(&std::free)> owner(local.records, std::free);

  ASSERT_EQ(1, local.nRecords);
  EXPECT_EQ(RAS_DIAG_PATHS_AVAILABLE, rasDiagnosticsPathsDataFromRecord(local.records)->state);
  EXPECT_EQ(1u << PATH_LOC, rasDiagnosticsPathsDataFromRecord(local.records)->pathMask);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_InProgressWithoutReadyTopologyReportsUnavailable) {
  ras_test::OwnedComm comm(0x123, 0x456, 0x789, 0);
  comm.comm->initState = ncclInProgress;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
  for (bool filtered : {false, true}) {
    SCOPED_TRACE(filtered);
    context_.hasCommFilter = filtered;
    context_.commFilter = {0x123, 0x456, 0x789};
    rasDiagnosticsLocalData local{};
    ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
    std::unique_ptr<char, decltype(&std::free)> owner(local.records, std::free);
    ASSERT_EQ(1, local.nRecords);
    EXPECT_EQ(RAS_DIAG_PATHS_TOPOLOGY_NOT_READY, rasDiagnosticsPathsDataFromRecord(local.records)->state);
    EXPECT_EQ(0u, rasDiagnosticsPathsDataFromRecord(local.records)->pathMask);
  }
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_UnfilteredInProgressCommDoesNotReadAvailableTopology) {
  ras_test::OwnedComm comm(0x123, 0x456, 0x789, 0);
  auto topology = std::make_unique<ncclTopoSystem>();
  topology->nodes[GPU].count = 1;
  topology->nodes[GPU].nodes[0].gpu.rank = 0;
  comm.comm->topo = topology.get();
  comm.comm->nRanks = 1;
  comm.comm->initState = ncclInProgress;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
  rasDiagnosticsLocalData local{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  std::unique_ptr<char, decltype(&std::free)> owner(local.records, std::free);

  ASSERT_EQ(1, local.nRecords);
  EXPECT_EQ(RAS_DIAG_PATHS_TOPOLOGY_NOT_READY, rasDiagnosticsPathsDataFromRecord(local.records)->state);
  EXPECT_EQ(0u, rasDiagnosticsPathsDataFromRecord(local.records)->pathMask);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_FilteredFailedCommStillUsesAvailableTopology) {
  ras_test::OwnedComm comm(0x123, 0x456, 0x789, 0);
  auto topology = std::make_unique<ncclTopoSystem>();
  topology->nodes[GPU].count = 1;
  topology->nodes[GPU].nodes[0].gpu.rank = 0;
  comm.comm->topo = topology.get();
  comm.comm->nRanks = 1;
  comm.comm->initState = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
  context_.hasCommFilter = true;
  context_.commFilter = {0x123, 0x456, 0x789};
  rasDiagnosticsLocalData local{};

  ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  std::unique_ptr<char, decltype(&std::free)> owner(local.records, std::free);

  ASSERT_EQ(1, local.nRecords);
  EXPECT_EQ(RAS_DIAG_PATHS_AVAILABLE, rasDiagnosticsPathsDataFromRecord(local.records)->state);
  EXPECT_EQ(1u << PATH_LOC, rasDiagnosticsPathsDataFromRecord(local.records)->pathMask);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_NoEligibleCommsPublishesNoRecords) {
  ras_test::OwnedComm comm(1, 2, 3, 0);
  comm.comm->initState = ncclSystemError;
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
  rasDiagnosticsLocalData local{};
  ASSERT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  EXPECT_EQ(nullptr, local.records);
  EXPECT_EQ(0, local.nRecords);
  ras_test::ResetNcclComms();
  EXPECT_EQ(ncclSuccess, rasDiagnosticsPathsCollectLocal(&context_, &local));
  EXPECT_EQ(nullptr, local.records);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_AllocationFailurePublishesNoRecords) {
  ras_test::OwnedComm comm(1, 2, 3, 0);
  ASSERT_NO_FATAL_FAILURE(ras_test::InstallNcclComms({comm.comm.get()}));
  g_failPathsAllocation = true;
  rasDiagnosticsLocalData local{};
  EXPECT_EQ(ncclSystemError, rasDiagnosticsPathsCollectLocal(&context_, &local));
  EXPECT_EQ(nullptr, local.records);
  EXPECT_EQ(0, local.recordsBytes);
}

TEST_F(RasDiagnosticsPathsMicrotest, CollectLocal_NullArgumentsAreRejected) {
  rasDiagnosticsLocalData local{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsCollectLocal(&context_, nullptr));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsCollectLocal(nullptr, &local));
  EXPECT_EQ(nullptr, local.records);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_CompleteUniformRanksReportOk) {
  const uint32_t mask = (1u << PATH_LOC) | (1u << PATH_SYS) | (1u << PATH_NET);
  ASSERT_EQ(ncclSuccess, Summarize({{1, mask}, {0, mask}}));
  EXPECT_EQ((std::vector<std::string>{"[OK]   Paths: self+SYS+NET across 2 ranks in comm 0x123"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_PartialPathTypesReportExactRankCounts) {
  ASSERT_EQ(ncclSuccess, Summarize({{1, (1u << PATH_LOC) | (1u << PATH_C2C)},
                                   {0, (1u << PATH_LOC) | (1u << PATH_NET)}}));
  EXPECT_EQ((std::vector<std::string>{"[INFO] Paths: self+C2C(1/2)+NET(1/2) across 2 ranks in comm 0x123"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_DisconnectedPathsAreInformationalEvenWhenUniform) {
  ASSERT_EQ(ncclSuccess, Summarize({{0, 1u << PATH_DIS}, {1, 1u << PATH_DIS}}));
  EXPECT_EQ((std::vector<std::string>{"[INFO] Paths: DIS across 2 ranks in comm 0x123"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_MissingGpuReportsRankAndPartialPaths) {
  ASSERT_EQ(ncclSuccess, Summarize({{1, 0, RAS_DIAG_PATHS_LOCAL_GPU_MISSING}, {0, 1u << PATH_LOC}}));
  EXPECT_EQ((std::vector<std::string>{
      "[INFO] Paths: could not locate the rank's own GPU in the topology, so its paths are unknown -- {1} in comm 0x123",
      "[INFO] Paths: self(1/2) across 2 ranks in comm 0x123"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_NotReadyAndIncompleteSuppressSuccessSummary) {
  ASSERT_EQ(ncclSuccess, Summarize({{1, 1u << PATH_LOC, RAS_DIAG_PATHS_TOPOLOGY_NOT_READY}}));
  EXPECT_EQ((std::vector<std::string>{
      "[INFO] Paths: diagnostics incomplete, gathered 1/2 ranks in comm 0x123/0x456/0x789",
      "[INFO] Paths: incomplete for comm 0x123: topology-dependent data was unavailable for 1 of 2 ranks"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_MissingGpuRankSetIsBoundedAndRetainsTotal) {
  const rasCommId id{0x123, 0x456, 0x789};
  const int rankCount = RAS_DIAG_RANK_SET_MAX + 2;
  std::vector<Row> rows;
  for (int rank = rankCount - 1; rank >= 0; rank--) {
    rows.push_back({rank, 0, RAS_DIAG_PATHS_LOCAL_GPU_MISSING, id, rankCount});
  }

  ASSERT_EQ(ncclSuccess, Summarize(rows));

  EXPECT_EQ((std::vector<std::string>{
              "[INFO] Paths: could not locate the rank's own GPU in the topology, so its paths are unknown -- "
              "{0,1,2,3,4,5,6,7,...} (N=10) in comm 0x123"}),
            output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_IncompleteReadyRanksSuppressSuccessSummary) {
  ASSERT_EQ(ncclSuccess, Summarize({{0, 1u << PATH_LOC}}));
  EXPECT_EQ((std::vector<std::string>{
      "[INFO] Paths: diagnostics incomplete, gathered 1/2 ranks in comm 0x123/0x456/0x789"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_IncompleteRanksSuppressMissingGpuReport) {
  const rasCommId id{0x123, 0x456, 0x789};
  ASSERT_EQ(ncclSuccess, Summarize({{0, 1u << PATH_LOC, RAS_DIAG_PATHS_AVAILABLE, id, 3},
                                   {1, 0, RAS_DIAG_PATHS_LOCAL_GPU_MISSING, id, 3}}));
  EXPECT_EQ((std::vector<std::string>{
              "[INFO] Paths: diagnostics incomplete, gathered 2/3 ranks in comm 0x123/0x456/0x789"}),
            output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_CompleteRanksWithUnavailableTopologySuppressSuccessSummary) {
  ASSERT_EQ(ncclSuccess, Summarize({{0, 1u << PATH_LOC}, {1, 0, RAS_DIAG_PATHS_TOPOLOGY_NOT_READY}}));
  EXPECT_EQ((std::vector<std::string>{
      "[INFO] Paths: incomplete for comm 0x123: topology-dependent data was unavailable for 1 of 2 ranks"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_GroupsByFullCommunicatorIdentity) {
  const rasCommId first{0x123, 0x456, 0x789};
  const rasCommId second{0x123, 0x456, 0x790};
  const rasCommId third{0x123, 0x457, 0x789};
  ASSERT_EQ(ncclSuccess, Summarize({{0, 1u << PATH_NET, RAS_DIAG_PATHS_AVAILABLE, third, 1},
                                   {0, 1u << PATH_SYS, RAS_DIAG_PATHS_AVAILABLE, second, 1},
                                   {0, 1u << PATH_LOC, RAS_DIAG_PATHS_AVAILABLE, first, 1}}));
  EXPECT_EQ((std::vector<std::string>{
      "[OK]   Paths: self across 1 ranks in comm 0x123",
      "[OK]   Paths: SYS across 1 ranks in comm 0x123",
      "[OK]   Paths: NET across 1 ranks in comm 0x123"}), output_.lines);
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_ReporterFailuresStopAtEachEmission) {
  const std::vector<std::vector<Row>> inputs = {
      {{0, 1u << PATH_LOC}},
      {{0, 0, RAS_DIAG_PATHS_TOPOLOGY_NOT_READY}, {1, 1u << PATH_LOC}},
      {{0, 0, RAS_DIAG_PATHS_LOCAL_GPU_MISSING}, {1, 1u << PATH_LOC}}};
  for (size_t input = 0; input < inputs.size(); input++) {
    output_ = {};
    ASSERT_EQ(ncclSuccess, Summarize(inputs[input]));
    const int calls = output_.calls;
    ASSERT_GT(calls, 0);
    for (int failCall = 1; failCall <= calls; failCall++) {
      SCOPED_TRACE(::testing::Message() << input << ',' << failCall);
      output_ = {};
      output_.failCall = failCall;
      output_.emitResult = ncclSystemError;
      EXPECT_EQ(ncclSystemError, Summarize(inputs[input]));
      EXPECT_EQ(failCall, output_.calls);
    }
  }
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_EmptyMasksEmitNothingAndInvalidStateFails) {
  ASSERT_EQ(ncclSuccess, Summarize({{0, 0}, {1, 0}}));
  EXPECT_TRUE(output_.lines.empty());
  EXPECT_EQ(ncclInternalError, Summarize({{0, 0, static_cast<rasDiagnosticsPathsState>(99)}}));
  EXPECT_TRUE(output_.lines.empty());
}

TEST_F(RasDiagnosticsPathsMicrotest, Summarize_InvalidArgumentsAndAllocationFailuresAreRejected) {
  auto records = Records({{0, 1u << PATH_LOC}});
  const int stride = records.size();
  rasDiagnosticsReporter invalid{};
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsSummarize(&context_, nullptr, records.data(), stride));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsSummarize(&context_, &invalid, records.data(), stride));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsSummarize(&context_, &reporter_, nullptr, stride));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsSummarize(&context_, &reporter_, records.data(), -stride));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsPathsSummarize(&context_, &reporter_, records.data(), stride - 1));
  EXPECT_EQ(ncclSuccess, rasDiagnosticsPathsSummarize(&context_, &reporter_, nullptr, 0));
  g_failPathsAllocation = true;
  EXPECT_EQ(ncclSystemError, Summarize({{0, 1u << PATH_LOC}}));
  EXPECT_TRUE(output_.lines.empty());
}

TEST_F(RasDiagnosticsPathsMicrotest, AppendType_BoundedBufferPreservesGuardsAndTerminator) {
  std::array<char, 8> buffer{};
  buffer.fill('z');
  EXPECT_EQ(3, rasDiagnosticsPathsAppendType(buffer.data(), 4, 0, true, PATH_SYS, 2, 2));
  EXPECT_STREQ("SYS", buffer.data());
  EXPECT_EQ('z', buffer[4]);
  EXPECT_EQ(7, rasDiagnosticsPathsAppendType(buffer.data(), 4, 3, false, PATH_NET, 2, 2));
  EXPECT_STREQ("SYS", buffer.data());
  EXPECT_EQ('z', buffer[4]);
  EXPECT_EQ(-1, rasDiagnosticsPathsAppendType(buffer.data(), 4, -1, false, PATH_NET, 2, 2));
  EXPECT_EQ(4, rasDiagnosticsPathsAppendType(buffer.data(), 4, 4, false, PATH_NET, 2, 2));
}

}
