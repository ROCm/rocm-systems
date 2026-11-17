/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/ras/diagnostics_checks_common.cc.

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "fakes/ras_diagnostics_test_support.h"
#include "compiler.h"
#include "fakes/fail_loud.h"
#include "fakes/signature-drift.h"
// NCCL 2.32.6: the unit includes graph/topo.h, whose xml.h has an inline xmlAlloc calling ncclCalloc(char**, n).
// Include it before the ncclCalloc redirect below so the redirect only reaches the unit's own allocations.
#include "graph/topo.h"
#include "ras/diagnostics_checks_common.h"
#include "ras/ras_internal.h"
#include "transport.h"

namespace {

int g_allocationCalls = 0;
int g_failAllocationCall = 0;
std::vector<size_t> g_allocationCounts;

template <typename T>
ncclResult_t DiagnosticsTestCalloc(ncclUniquePtr<T>& ptr, size_t count) {
  ++g_allocationCalls;
  g_allocationCounts.push_back(count);
  if (g_allocationCalls == g_failAllocationCall) return ncclSystemError;
  return ncclCallocDebug(ptr, count, __FILE__, __LINE__, __func__, false);
}

}  // namespace

#undef ncclCalloc
#define ncclCalloc(...) DiagnosticsTestCalloc(__VA_ARGS__)

#include RAS_DIAGNOSTICS_COMMON_CC_PATH

#undef ncclCalloc

// Link seams for rasDiagnosticsCollectCommSnapshots' net-device path. graph/topo.cc and transport/net_ib/ are not in
// this binary; the unit only reads the IB device table and compares the comm's plugin against &ncclNetIb.
ncclNet_t ncclNetIb = {.name = "IB"};
int ncclNIbDevs = 0;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];

static ncclResult_t DefaultTopoGetNetPropertiesSnapshot(struct ncclComm*, bool, struct ncclTopoNetPropertiesSnapshot**,
                                                        int*) {
  FailLoudUnfaked("ras-diagnostics-common-test", "ncclTopoGetNetPropertiesSnapshot");
}
static std::function<ncclResult_t(struct ncclComm*, bool, struct ncclTopoNetPropertiesSnapshot**, int*)>
    g_topoGetNetPropertiesSnapshot = DefaultTopoGetNetPropertiesSnapshot;
ASSERT_HOOK_MATCHES_PROD(g_topoGetNetPropertiesSnapshot, ncclTopoGetNetPropertiesSnapshot);
ncclResult_t ncclTopoGetNetPropertiesSnapshot(struct ncclComm* comm, bool requireCommInitialized,
                                              struct ncclTopoNetPropertiesSnapshot** snapshots, int* nSnapshots) {
  return g_topoGetNetPropertiesSnapshot(comm, requireCommInitialized, snapshots, nSnapshots);
}

namespace {

struct TestPayload {
  int cudaDev;
  int localRank;
  uint32_t marker;
};

using ras_test::OwnedComm;
using ras_test::ReporterState;
using ras_test::CaptureReport;
using ras_test::InstallNcclComms;

std::vector<rasDiagnosticsCommSnapshot> g_fillSnapshots;
int g_fillCalls = 0;
int g_failFillCall = 0;
bool g_fillObservedUnlockedMutex = true;

ncclResult_t FillTestPayload(const rasDiagnosticsCommSnapshot* snapshot, void* checkData) {
  ++g_fillCalls;
  bool mutexWasUnlocked = false;
  std::thread probe([&mutexWasUnlocked] {
    if (!ncclCommsMutex.try_lock()) return;
    mutexWasUnlocked = true;
    ncclCommsMutex.unlock();
  });
  probe.join();
  if (!mutexWasUnlocked) g_fillObservedUnlockedMutex = false;
  g_fillSnapshots.push_back(*snapshot);
  if (g_fillCalls == g_failFillCall) return ncclSystemError;

  const TestPayload payload{snapshot->cudaDev, snapshot->localRank, 0xc0ffeeu};
  std::memcpy(checkData, &payload, sizeof(payload));
  return ncclSuccess;
}

void ResetDiagnosticsCommonState() {
  ras_test::ResetNcclComms();
  g_topoGetNetPropertiesSnapshot = DefaultTopoGetNetPropertiesSnapshot;
  ncclNIbDevs = 0;
  for (struct ncclIbDev& dev : ncclIbDevs) dev.currSpeed = 0;
  g_allocationCalls = 0;
  g_failAllocationCall = 0;
  g_allocationCounts.clear();
  g_fillSnapshots.clear();
  g_fillCalls = 0;
  g_failFillCall = 0;
  g_fillObservedUnlockedMutex = true;
}

rasDiagnosticsContext UnfilteredContext() {
  rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = false;
  return ctx;
}

rasDiagnosticsContext FilteredContext(uint64_t commHash, uint64_t hostHash, uint64_t pidHash) {
  rasDiagnosticsContext ctx{};
  ctx.hasCommFilter = true;
  ctx.commFilter = {commHash, hostHash, pidHash};
  return ctx;
}

class RasDiagnosticsCommonMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetDiagnosticsCommonState(); }
  void TearDown() override { ResetDiagnosticsCommonState(); }
};

TEST_F(RasDiagnosticsCommonMicrotest, CommSnapshotCopiesEveryDiagnosticField) {
  OwnedComm owned(UINT64_C(0x1122334455667788), UINT64_C(0x99aabbccddeeff00),
                  UINT64_C(0x0123456789abcdef), 5);
  // rank stays 5: since NCCL 2.32.6 the snapshot reads its node identity from peerInfo[comm->rank], so the rank must
  // index the peer table. That entry gets its own hostHash to show which one the snapshot reads.
  owned.peers[5].hostHash = UINT64_C(0x5555aaaa5555aaaa);
  owned.comm->nRanks = 0x55667788;
  owned.comm->cudaDev = 7;
  owned.comm->nvmlDev = 9;
  owned.comm->busId = 0x123456;
  owned.comm->localRank = 2;
  owned.comm->localRanks = 6;

  rasDiagnosticsCommSnapshot snapshot{};
  rasDiagnosticsCommSnapshotInit(&snapshot, owned.comm.get());

  EXPECT_EQ(UINT64_C(0x1122334455667788), snapshot.rank.commId.commHash);
  EXPECT_EQ(UINT64_C(0x99aabbccddeeff00), snapshot.rank.commId.hostHash);
  EXPECT_EQ(UINT64_C(0x0123456789abcdef), snapshot.rank.commId.pidHash);
  EXPECT_EQ(5, snapshot.rank.commRank);
  EXPECT_EQ(0x55667788, snapshot.rank.commNRanks);
  EXPECT_EQ(7, snapshot.cudaDev);
  EXPECT_EQ(9, snapshot.nvmlDev);
  EXPECT_EQ(0x123456, snapshot.busId);
  EXPECT_EQ(2, snapshot.localRank);
  EXPECT_EQ(6, snapshot.localRanks);
  EXPECT_EQ(UINT64_C(0x5555aaaa5555aaaa), snapshot.hostHash);
  // Net-device fields: no plugin means not a verbs plugin, and no properties until a collector asks for them.
  EXPECT_FALSE(snapshot.netIbVerbs);
  EXPECT_EQ(nullptr, snapshot.netProps);
  EXPECT_EQ(0, snapshot.nNetProps);
}

TEST_F(RasDiagnosticsCommonMicrotest, CommIdCompareOrdersEveryIdentityComponent) {
  const rasCommId base{UINT64_C(0x1122334455667788), UINT64_C(0x99aabbccddeeff00),
                       UINT64_C(0x0123456789abcdef)};
  struct CompareCase {
    const char* field;
    rasCommId value;
  };
  const std::vector<CompareCase> less = {
    {"commHash high bits", {UINT64_C(0x0122334455667788), UINT64_MAX, UINT64_MAX}},
    {"hostHash high bits", {base.commHash, UINT64_C(0x89aabbccddeeff00), UINT64_MAX}},
    {"pidHash high bits", {base.commHash, base.hostHash, UINT64_C(0x0023456789abcdef)}},
    {"commHash low bits", {base.commHash - 1, UINT64_MAX, UINT64_MAX}},
    {"hostHash low bits", {base.commHash, base.hostHash - 1, UINT64_MAX}},
    {"pidHash low bits", {base.commHash, base.hostHash, base.pidHash - 1}}};
  const std::vector<CompareCase> greater = {
    {"commHash high bits", {UINT64_C(0x2122334455667788), 0, 0}},
    {"hostHash high bits", {base.commHash, UINT64_C(0xa9aabbccddeeff00), 0}},
    {"pidHash high bits", {base.commHash, base.hostHash, UINT64_C(0x0223456789abcdef)}},
    {"commHash low bits", {base.commHash + 1, 0, 0}},
    {"hostHash low bits", {base.commHash, base.hostHash + 1, 0}},
    {"pidHash low bits", {base.commHash, base.hostHash, base.pidHash + 1}}};

  EXPECT_EQ(0, rasDiagnosticsCommIdCompare(&base, &base));
  for (const CompareCase& test : less) {
    SCOPED_TRACE(test.field);
    EXPECT_LT(rasDiagnosticsCommIdCompare(&test.value, &base), 0);
    EXPECT_GT(rasDiagnosticsCommIdCompare(&base, &test.value), 0);
  }
  for (const CompareCase& test : greater) {
    SCOPED_TRACE(test.field);
    EXPECT_GT(rasDiagnosticsCommIdCompare(&test.value, &base), 0);
    EXPECT_LT(rasDiagnosticsCommIdCompare(&base, &test.value), 0);
  }

  const rasCommId signedCommLow{UINT64_C(0x7fffffffffffffff), 0, 0};
  const rasCommId signedCommHigh{UINT64_C(0x8000000000000000), 0, 0};
  EXPECT_LT(rasDiagnosticsCommIdCompare(&signedCommLow, &signedCommHigh), 0);
  EXPECT_GT(rasDiagnosticsCommIdCompare(&signedCommHigh, &signedCommLow), 0);
  const rasCommId signedHostLow{base.commHash, UINT64_C(0x7fffffffffffffff), 0};
  const rasCommId signedHostHigh{base.commHash, UINT64_C(0x8000000000000000), 0};
  EXPECT_LT(rasDiagnosticsCommIdCompare(&signedHostLow, &signedHostHigh), 0);
  EXPECT_GT(rasDiagnosticsCommIdCompare(&signedHostHigh, &signedHostLow), 0);
  const rasCommId signedPidLow{base.commHash, base.hostHash, UINT64_C(0x7fffffffffffffff)};
  const rasCommId signedPidHigh{base.commHash, base.hostHash, UINT64_C(0x8000000000000000)};
  EXPECT_LT(rasDiagnosticsCommIdCompare(&signedPidLow, &signedPidHigh), 0);
  EXPECT_GT(rasDiagnosticsCommIdCompare(&signedPidHigh, &signedPidLow), 0);
}

TEST_F(RasDiagnosticsCommonMicrotest, CommMatchesContextHonorsOptionalFilter) {
  OwnedComm owned(10, 20, 30, 0);
  rasDiagnosticsContext ctx = UnfilteredContext();

  EXPECT_TRUE(rasDiagnosticsCommMatchesContext(&ctx, nullptr));
  ctx = FilteredContext(10, 20, 30);
  EXPECT_TRUE(rasDiagnosticsCommMatchesContext(&ctx, owned.comm.get()));
  ctx.commFilter.commHash++;
  EXPECT_FALSE(rasDiagnosticsCommMatchesContext(&ctx, owned.comm.get()));
  ctx = FilteredContext(10, 21, 30);
  EXPECT_FALSE(rasDiagnosticsCommMatchesContext(&ctx, owned.comm.get()));
  ctx = FilteredContext(10, 20, 31);
  EXPECT_FALSE(rasDiagnosticsCommMatchesContext(&ctx, owned.comm.get()));
}

TEST_F(RasDiagnosticsCommonMicrotest, LocalRecordStrideRoundsHeaderAndPayloadToHeaderAlignment) {
  const size_t headerSize = sizeof(rasDiagnosticsRankHeader);
  const size_t alignment = alignof(rasDiagnosticsRankHeader);

  for (size_t payloadSize : {size_t{0}, size_t{1}, alignment - 1, alignment, alignment + 1}) {
    const size_t stride = rasDiagnosticsLocalRecordStride(payloadSize);
    EXPECT_GE(stride, headerSize + payloadSize);
    EXPECT_LT(stride, headerSize + payloadSize + alignment);
    EXPECT_EQ(0u, stride % alignment);
  }
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsNullOutput) {
  const rasDiagnosticsContext ctx = UnfilteredContext();
  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, nullptr));
  EXPECT_EQ(0, g_allocationCalls);
  EXPECT_EQ(0, g_fillCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsNullContextAndClearsOutput) {
  rasDiagnosticsLocalData data{reinterpret_cast<char*>(1), 2, 3, 4};

  EXPECT_EQ(ncclInternalError,
            rasDiagnosticsCollectLocalRecords(nullptr, sizeof(TestPayload), FillTestPayload, &data));
  EXPECT_EQ(nullptr, data.records);
  EXPECT_EQ(0, data.recordsBytes);
  EXPECT_EQ(0, data.recordStride);
  EXPECT_EQ(0, data.nRecords);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsNullFillCallback) {
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};

  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), nullptr, &data));
  EXPECT_EQ(0, g_allocationCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsPayloadWhoseHeaderWouldOverflowInt) {
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  const size_t payloadSize = static_cast<size_t>(INT_MAX) - sizeof(rasDiagnosticsRankHeader) + 1;

  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalRecords(&ctx, payloadSize, FillTestPayload, &data));
  EXPECT_EQ(0, g_allocationCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsStrideRoundedPastIntMax) {
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  const size_t payloadSize = static_cast<size_t>(INT_MAX) - sizeof(rasDiagnosticsRankHeader);

  ASSERT_GT(rasDiagnosticsLocalRecordStride(payloadSize), static_cast<size_t>(INT_MAX));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalRecords(&ctx, payloadSize, FillTestPayload, &data));
  EXPECT_EQ(0, g_allocationCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectRejectsCombinedRecordBytesPastIntMax) {
  OwnedComm first(1, 2, 3, 0);
  OwnedComm second(4, 5, 6, 1);
  InstallNcclComms({first.comm.get(), second.comm.get()});
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  const size_t payloadSize = static_cast<size_t>(INT_MAX) / 2;

  const size_t stride = rasDiagnosticsLocalRecordStride(payloadSize);
  ASSERT_LE(stride, static_cast<size_t>(INT_MAX));
  ASSERT_EQ(1, INT_MAX / static_cast<int>(stride));
  EXPECT_EQ(ncclInternalError, rasDiagnosticsCollectLocalRecords(&ctx, payloadSize, FillTestPayload, &data));
  // NCCL 2.32.6 counts comms by taking their snapshots (rasDiagnosticsCollectCommSnapshots) before this size check,
  // so the two-entry snapshot array is the one allocation made (and released on the error path); the record buffer
  // is never requested.
  EXPECT_EQ(1, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{2}), g_allocationCounts);
  EXPECT_EQ(0, g_fillCalls);
  EXPECT_EQ(nullptr, data.records);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectAcceptsRecordBytesExactlyAtIntMaxLimit) {
  OwnedComm first(1, 2, 3, 0);
  OwnedComm second(4, 5, 6, 1);
  InstallNcclComms({first.comm.get(), second.comm.get()});
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  const size_t payloadSize = 1000000000;

  const size_t recordStride = rasDiagnosticsLocalRecordStride(payloadSize);
  ASSERT_EQ(2u, static_cast<size_t>(INT_MAX) / recordStride);
  // Stop before requesting the approximately 2 GB record buffer.
  g_failAllocationCall = 2;
  EXPECT_EQ(ncclSystemError, rasDiagnosticsCollectLocalRecords(&ctx, payloadSize, FillTestPayload, &data));
  // Reaching allocation 2 proves equality passed the strict overflow guard.
  EXPECT_EQ(2, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{2, 2 * recordStride}), g_allocationCounts);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectReturnsEmptyForNullInvalidAndFilteredComms) {
  OwnedComm invalid(10, 20, 30, 0, false);
  OwnedComm mismatch(11, 20, 30, 1);
  InstallNcclComms({nullptr, invalid.comm.get(), mismatch.comm.get()});
  const rasDiagnosticsContext ctx = FilteredContext(10, 20, 30);
  rasDiagnosticsLocalData data{reinterpret_cast<char*>(1), 2, 3, 4};

  EXPECT_EQ(ncclSuccess, rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, &data));
  EXPECT_EQ(nullptr, data.records);
  EXPECT_EQ(0, data.recordsBytes);
  EXPECT_EQ(0, data.recordStride);
  EXPECT_EQ(0, data.nRecords);
  EXPECT_EQ(0, g_allocationCalls);
  EXPECT_EQ(0, g_fillCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectFiltersCommsAndBuildsAlignedRecordsAfterUnlockingRegistry) {
  OwnedComm first(10, 20, 30, 2);
  OwnedComm invalid(10, 20, 30, 3, false);
  OwnedComm mismatch(11, 20, 30, 4);
  OwnedComm second(10, 20, 30, 5);
  // Ranks stay 2 and 5: since NCCL 2.32.6 the snapshot reads peerInfo[comm->rank], so a rank must index the peer table.
  first.comm->nRanks = 0x55667788;
  second.comm->nRanks = 0x66778899;
  InstallNcclComms({first.comm.get(), nullptr, invalid.comm.get(), mismatch.comm.get(), second.comm.get(), nullptr});
  const rasDiagnosticsContext ctx = FilteredContext(10, 20, 30);
  rasDiagnosticsLocalData data{};

  ASSERT_EQ(ncclSuccess,
            rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, &data));
  ASSERT_EQ(2, data.nRecords);
  EXPECT_EQ(2, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{2, 2 * static_cast<size_t>(data.recordStride)}), g_allocationCounts);
  EXPECT_EQ(2, g_fillCalls);
  EXPECT_TRUE(g_fillObservedUnlockedMutex);
  EXPECT_EQ(2 * data.recordStride, data.recordsBytes);
  EXPECT_EQ(rasDiagnosticsLocalRecordStride(sizeof(TestPayload)), static_cast<size_t>(data.recordStride));

  const std::vector<OwnedComm*> expected = {&first, &second};
  for (int index = 0; index < data.nRecords; ++index) {
    const char* record = data.records + index * data.recordStride;
    const rasDiagnosticsRankHeader* header = rasDiagnosticsRankHeaderFromRecord(record);
    EXPECT_EQ(expected[index]->comm->commHash, header->commId.commHash);
    EXPECT_EQ(expected[index]->peers[0].hostHash, header->commId.hostHash);
    EXPECT_EQ(expected[index]->peers[0].pidHash, header->commId.pidHash);
    EXPECT_EQ(expected[index]->comm->rank, header->commRank);
    EXPECT_EQ(expected[index]->comm->nRanks, header->commNRanks);

    TestPayload payload{};
    std::memcpy(&payload, record + sizeof(*header), sizeof(payload));
    EXPECT_EQ(expected[index]->comm->cudaDev, payload.cudaDev);
    EXPECT_EQ(expected[index]->comm->localRank, payload.localRank);
    EXPECT_EQ(0xc0ffeeu, payload.marker);

    for (size_t offset = sizeof(*header) + sizeof(payload); offset < static_cast<size_t>(data.recordStride);
         ++offset) {
      EXPECT_EQ(0, record[offset]);
    }
  }

  ASSERT_EQ(2u, g_fillSnapshots.size());
  EXPECT_EQ(first.comm->busId, g_fillSnapshots[0].busId);
  EXPECT_EQ(second.comm->nvmlDev, g_fillSnapshots[1].nvmlDev);
  std::free(data.records);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectPropagatesSnapshotAllocationFailure) {
  OwnedComm owned(1, 2, 3, 0);
  InstallNcclComms({owned.comm.get()});
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  g_failAllocationCall = 1;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, &data));
  EXPECT_EQ(1, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{1}), g_allocationCounts);
  EXPECT_EQ(nullptr, data.records);
  EXPECT_EQ(0, g_fillCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectPropagatesRecordAllocationFailure) {
  OwnedComm owned(1, 2, 3, 0);
  InstallNcclComms({owned.comm.get()});
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  g_failAllocationCall = 2;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, &data));
  EXPECT_EQ(2, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{1, rasDiagnosticsLocalRecordStride(sizeof(TestPayload))}), g_allocationCounts);
  EXPECT_EQ(nullptr, data.records);
  EXPECT_EQ(0, g_fillCalls);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectPropagatesFillFailureWithoutPublishingPartialData) {
  OwnedComm owned(1, 2, 3, 0);
  InstallNcclComms({owned.comm.get()});
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsLocalData data{};
  g_failFillCall = 1;

  EXPECT_EQ(ncclSystemError, rasDiagnosticsCollectLocalRecords(&ctx, sizeof(TestPayload), FillTestPayload, &data));
  EXPECT_EQ(2, g_allocationCalls);
  EXPECT_EQ((std::vector<size_t>{1, rasDiagnosticsLocalRecordStride(sizeof(TestPayload))}), g_allocationCounts);
  EXPECT_EQ(1, g_fillCalls);
  EXPECT_TRUE(g_fillObservedUnlockedMutex);
  EXPECT_EQ(nullptr, data.records);
  EXPECT_EQ(0, data.recordsBytes);
  EXPECT_EQ(0, data.recordStride);
  EXPECT_EQ(0, data.nRecords);
}

TEST_F(RasDiagnosticsCommonMicrotest, RankHeaderCompareOrdersIdentityThenRank) {
  rasDiagnosticsRankHeader base{{10, 20, 30}, 4, 8};
  rasDiagnosticsRankHeader same = base;
  rasDiagnosticsRankHeader earlierComm{{9, 999, 999}, 99, 8};
  rasDiagnosticsRankHeader earlierRank = base;
  rasDiagnosticsRankHeader laterRank = base;
  earlierRank.commRank = 3;
  laterRank.commRank = 5;

  EXPECT_EQ(&base, rasDiagnosticsRankHeaderFromRecord(reinterpret_cast<const char*>(&base)));
  EXPECT_EQ(0, rasDiagnosticsRankHeaderCompare(&base, &same));
  EXPECT_LT(rasDiagnosticsRankHeaderCompare(&earlierComm, &base), 0);
  EXPECT_GT(rasDiagnosticsRankHeaderCompare(&base, &earlierComm), 0);
  EXPECT_LT(rasDiagnosticsRankHeaderCompare(&earlierRank, &base), 0);
  EXPECT_GT(rasDiagnosticsRankHeaderCompare(&laterRank, &base), 0);
}

TEST_F(RasDiagnosticsCommonMicrotest, FormatRankSetHandlesEmptyCompleteAndTruncatedSets) {
  const int ranks[] = {0, 2, 4, 6, 8, 10, 12, 14, 16};
  char buf[128];

  rasDiagnosticsFormatRankSet(buf, sizeof(buf), ranks, 0, 0);
  EXPECT_STREQ("{}", buf);
  rasDiagnosticsFormatRankSet(buf, sizeof(buf), ranks, 3, 3);
  EXPECT_STREQ("{0,2,4}", buf);
  rasDiagnosticsFormatRankSet(buf, sizeof(buf), ranks, 9, 9);
  EXPECT_STREQ("{0,2,4,6,8,10,12,14,...} (N=9)", buf);
  rasDiagnosticsFormatRankSet(buf, sizeof(buf), ranks, RAS_DIAG_RANK_SET_MAX, 20);
  EXPECT_STREQ("{0,2,4,6,8,10,12,14,...} (N=20)", buf);
  rasDiagnosticsFormatRankSet(buf, sizeof(buf), ranks, 9, 20);
  EXPECT_STREQ("{0,2,4,6,8,10,12,14,...} (N=20)", buf);
}

TEST_F(RasDiagnosticsCommonMicrotest, FormatRankSetRespectsZeroAndTinyBuffers) {
  const int ranks[] = {123};
  char untouched = 'x';
  char oneByte[2] = {'x', 'y'};

  rasDiagnosticsFormatRankSet(&untouched, 0, ranks, 1, 1);
  EXPECT_EQ('x', untouched);
  rasDiagnosticsFormatRankSet(oneByte, 1, ranks, 1, 1);
  EXPECT_EQ('\0', oneByte[0]);
  EXPECT_EQ('y', oneByte[1]);

  const int many[] = {1000, 2000, 3000};
  char small[32];
  std::memset(small, 'S', sizeof(small));
  rasDiagnosticsFormatRankSet(small, 8, many, 3, 3);
  EXPECT_STREQ("{1000,2", small);
  for (size_t index = 8; index < sizeof(small); ++index) EXPECT_EQ('S', small[index]);
}

TEST_F(RasDiagnosticsCommonMicrotest, ReportFormatsMessageAndPropagatesEmitterResult) {
  ReporterState state;
  const rasDiagnosticsReporter reporter{CaptureReport, nullptr, &state};

  EXPECT_EQ(ncclSuccess, rasDiagnosticsReport(&reporter, RAS_DIAG_TAG_OK, "rank %d: %s", 7, "ready"));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[OK]   rank 7: ready", state.lines[0]);

  state.emitResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, rasDiagnosticsReport(&reporter, RAS_DIAG_TAG_INFO, "second"));
  ASSERT_EQ(2u, state.lines.size());
  EXPECT_EQ("[INFO] second", state.lines[1]);
}

TEST_F(RasDiagnosticsCommonMicrotest, ReportRejectsTagThatConsumesWholeBuffer) {
  ReporterState state;
  const rasDiagnosticsReporter reporter{CaptureReport, nullptr, &state};
  const std::string tag(1024, 'x');

  EXPECT_EQ(ncclInternalError, rasDiagnosticsReport(&reporter, tag.c_str(), "ignored"));
  EXPECT_TRUE(state.lines.empty());
}

TEST_F(RasDiagnosticsCommonMicrotest, ReportTruncatesOversizedFormattedBodySafely) {
  ReporterState state;
  const rasDiagnosticsReporter reporter{CaptureReport, nullptr, &state};
  const std::string body(2048, 'z');

  EXPECT_EQ(ncclSuccess, rasDiagnosticsReport(&reporter, "T:", "%s", body.c_str()));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ(1023u, state.lines[0].size());
  EXPECT_EQ("T:", state.lines[0].substr(0, 2));
  EXPECT_EQ(std::string(1021, 'z'), state.lines[0].substr(2));
}

TEST_F(RasDiagnosticsCommonMicrotest, ReportIncompleteIncludesCountsAndFullCommIdentity) {
  ReporterState state;
  const rasDiagnosticsReporter reporter{CaptureReport, nullptr, &state};
  const rasDiagnosticsRankHeader rank{{UINT64_C(0x1122334455667788), UINT64_C(0x99aabbccddeeff00),
                                       UINT64_C(0x0123456789abcdef)},
                                      4,
                                      16};

  EXPECT_EQ(ncclSuccess, rasDiagnosticsReportIncomplete(&reporter, "GPU model", &rank, 12));
  ASSERT_EQ(1u, state.lines.size());
  EXPECT_EQ("[INFO] GPU model: diagnostics incomplete, gathered 12/16 ranks in comm "
            "0x1122334455667788/0x99aabbccddeeff00/0x123456789abcdef",  // NCCL 2.32 dropped the "(RAS overlay may not be ready)" suffix
            state.lines[0]);

  state.emitResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, rasDiagnosticsReportIncomplete(&reporter, "GPU model", &rank, 12));
}

// --- rasDiagnosticsCollectCommSnapshots' net-device path (NCCL 2.32.6) ---

// One property-snapshot array the way ncclTopoGetNetPropertiesSnapshot hands it out: heap-owned by the caller.
struct NetPropSpec {
  bool valid;
  std::vector<int> devs;
  int speed;
};
ncclTopoNetPropertiesSnapshot* MakeNetProps(const std::vector<NetPropSpec>& specs) {
  auto* props = static_cast<ncclTopoNetPropertiesSnapshot*>(std::calloc(specs.size(), sizeof(ncclTopoNetPropertiesSnapshot)));
  for (size_t i = 0; i < specs.size(); i++) {
    props[i].propertiesValid = specs[i].valid;
    props[i].vProps.ndevs = static_cast<int>(specs[i].devs.size());
    for (size_t d = 0; d < specs[i].devs.size(); d++) props[i].vProps.devs[d] = specs[i].devs[d];
    props[i].speed = specs[i].speed;
  }
  return props;
}

struct NetSnapshotCall {
  ncclComm* comm;
  bool requireCommInitialized;
};

// RCCL's internal verbs transport counts as a verbs plugin; a name that only shares its prefix does not.
TEST_F(RasDiagnosticsCommonMicrotest, CollectCommSnapshotsTreatsRcclIbCastAsVerbs) {
  ncclNet_t ibCast = {.name = "IB-CAST"};
  ncclNet_t ibCastPrefix = {.name = "IB-CASTX"};
  OwnedComm c0(10, 20, 30, 0), c1(10, 20, 30, 1);
  c0.comm->ncclNet = &ibCast;
  c1.comm->ncclNet = &ibCastPrefix;
  InstallNcclComms({c0.comm.get(), c1.comm.get()});

  const rasDiagnosticsContext unfiltered = UnfilteredContext();
  rasDiagnosticsCommSnapshot* snapshots = nullptr;
  int nSnapshots = -1;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&unfiltered, false, &snapshots, &nSnapshots));
  ASSERT_EQ(2, nSnapshots);
  EXPECT_TRUE(snapshots[0].netIbVerbs);
  EXPECT_FALSE(snapshots[1].netIbVerbs);
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectCommSnapshotsQueriesNetDevicesOnlyForVerbsPluginsWhenAsked) {
  // Verbs plugin names match exactly or followed by '_' or ' '; a name that merely shares the prefix does not.
  ncclNet_t ibExt = {.name = "IBext_v10"};
  ncclNet_t ibPrefix = {.name = "IBX"};
  ncclNet_t rdma = {.name = "NCCL RDMA Plugin v9"};
  ncclNet_t socket = {.name = "Socket"};
  OwnedComm c0(10, 20, 30, 0), c1(10, 20, 30, 1), c2(10, 20, 30, 2), c3(10, 20, 30, 3), c4(10, 20, 30, 4);
  c0.comm->ncclNet = &ibExt;
  c1.comm->ncclNet = &ibPrefix;
  c2.comm->ncclNet = nullptr;
  c3.comm->ncclNet = &rdma;
  c4.comm->ncclNet = &socket;
  InstallNcclComms({c0.comm.get(), c1.comm.get(), c2.comm.get(), c3.comm.get(), c4.comm.get()});
  const std::vector<bool> verbs = {true, false, false, true, false};

  // Not asked: the default seam fails loudly if reached.
  const rasDiagnosticsContext unfiltered = UnfilteredContext();
  rasDiagnosticsCommSnapshot* snapshots = nullptr;
  int nSnapshots = -1;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&unfiltered, false, &snapshots, &nSnapshots));
  ASSERT_EQ(5, nSnapshots);
  for (int i = 0; i < nSnapshots; i++) {
    EXPECT_EQ(verbs[i], snapshots[i].netIbVerbs) << i;
    EXPECT_EQ(nullptr, snapshots[i].netProps) << i;
    EXPECT_EQ(0, snapshots[i].nNetProps) << i;
  }
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);

  // Asked, unscoped: only the verbs comms are queried, and they must have finished initializing.
  std::vector<NetSnapshotCall> calls;
  std::vector<ncclTopoNetPropertiesSnapshot*> handedOut;
  g_topoGetNetPropertiesSnapshot = [&](ncclComm* comm, bool requireInit, ncclTopoNetPropertiesSnapshot** out, int* n) {
    calls.push_back({comm, requireInit});
    *out = MakeNetProps({{true, {0}, 100}, {true, {1}, 200}});
    *n = 2;
    handedOut.push_back(*out);
    return ncclSuccess;
  };
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&unfiltered, true, &snapshots, &nSnapshots));
  ASSERT_EQ(5, nSnapshots);
  ASSERT_EQ(2u, calls.size());
  EXPECT_EQ(c0.comm.get(), calls[0].comm);
  EXPECT_TRUE(calls[0].requireCommInitialized);
  EXPECT_EQ(c3.comm.get(), calls[1].comm);
  EXPECT_TRUE(calls[1].requireCommInitialized);
  EXPECT_EQ(handedOut[0], snapshots[0].netProps);
  EXPECT_EQ(2, snapshots[0].nNetProps);
  EXPECT_EQ(handedOut[1], snapshots[3].netProps);
  for (int i : {1, 2, 4}) EXPECT_EQ(nullptr, snapshots[i].netProps) << i;
  // ibExt is not the built-in IB plugin, so the plugin-reported speeds stand.
  EXPECT_EQ(100, snapshots[0].netProps[0].speed);
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);

  // Comm-scoped: the run is triggered by that comm's own initialization, so it must not wait for it.
  calls.clear();
  const rasDiagnosticsContext scoped = FilteredContext(10, 20, 30);
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&scoped, true, &snapshots, &nSnapshots));
  ASSERT_EQ(2u, calls.size());
  EXPECT_FALSE(calls[0].requireCommInitialized);
  EXPECT_FALSE(calls[1].requireCommInitialized);
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);

  // A failed property query is not fatal: the snapshot just carries no properties.
  g_topoGetNetPropertiesSnapshot = [](ncclComm*, bool, ncclTopoNetPropertiesSnapshot**, int*) {
    return ncclSystemError;
  };
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&unfiltered, true, &snapshots, &nSnapshots));
  ASSERT_EQ(5, nSnapshots);
  EXPECT_EQ(nullptr, snapshots[0].netProps);
  EXPECT_EQ(0, snapshots[0].nNetProps);
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);
}

TEST_F(RasDiagnosticsCommonMicrotest, CollectCommSnapshotsTakesLiveSpeedFromBuiltInIbDeviceTable) {
  ncclNIbDevs = 3;
  ncclIbDevs[0].currSpeed = 400000;
  ncclIbDevs[1].currSpeed = 0;  // not tracked: keep the plugin's speed
  ncclIbDevs[2].currSpeed = static_cast<uint64_t>(INT_MAX) + 1;  // does not fit the int field: keep the plugin's speed
  ncclNet_t notBuiltIn = {.name = "IB"};  // a verbs plugin by name, but not &ncclNetIb
  OwnedComm builtIn(10, 20, 30, 0), other(10, 20, 30, 1);
  builtIn.comm->ncclNet = &ncclNetIb;
  other.comm->ncclNet = &notBuiltIn;
  InstallNcclComms({builtIn.comm.get(), other.comm.get()});
  g_topoGetNetPropertiesSnapshot = [](ncclComm*, bool, ncclTopoNetPropertiesSnapshot** out, int* n) {
    *out = MakeNetProps({
        {true, {0}, 100},     // live speed applies
        {true, {1}, 200},     // currSpeed 0
        {false, {0}, 300},    // properties invalid
        {true, {0, 1}, 400},  // merged virtual device: no single IB device
        {true, {2}, 500},     // currSpeed past INT_MAX
        {true, {3}, 600},     // index past ncclNIbDevs
    });
    *n = 6;
    return ncclSuccess;
  };
  const rasDiagnosticsContext ctx = UnfilteredContext();
  rasDiagnosticsCommSnapshot* snapshots = nullptr;
  int nSnapshots = 0;
  ASSERT_EQ(ncclSuccess, rasDiagnosticsCollectCommSnapshots(&ctx, true, &snapshots, &nSnapshots));
  ASSERT_EQ(2, nSnapshots);
  ASSERT_EQ(6, snapshots[0].nNetProps);
  const std::vector<int> builtInSpeeds = {400000, 200, 300, 400, 500, 600};
  const std::vector<int> pluginSpeeds = {100, 200, 300, 400, 500, 600};
  for (int d = 0; d < 6; d++) {
    EXPECT_EQ(builtInSpeeds[d], snapshots[0].netProps[d].speed) << d;
    EXPECT_EQ(pluginSpeeds[d], snapshots[1].netProps[d].speed) << d;
  }
  rasDiagnosticsFreeCommSnapshots(snapshots, nSnapshots);
}

}  // namespace
