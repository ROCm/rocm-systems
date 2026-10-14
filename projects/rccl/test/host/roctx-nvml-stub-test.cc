/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/roctx.cc and src/misc/nvmlwrap_stub.cc. The -roctx variant
// builds roctx.cc's ROCTX_ENABLE arm; the range tests hold in either. stringify() inputs stay
// under MAX_MESSAGE_LENGTH: past it, it overflows its buffer (AICOMRCCL-2914).

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "debug.h"

std::function<int64_t(const char*, int64_t)> g_loadParam = [](const char*, int64_t deftVal) { return deftVal; };

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

#include "fakes/param_redirect.h"

// malloc seam for roctx.cc's own bodies only, so roctxAlloc's failure paths are reachable.
#include <cstdlib>
#include "roctx.h"
namespace {
std::function<void*(size_t)> g_roctxMalloc;
}  // namespace

// With ROCTX_ENABLE, record roctracer's push/pop instead of linking it.
#ifdef ROCTX_ENABLE
namespace {
std::vector<std::string> g_roctxPushed;
int g_roctxPops = 0;
}  // namespace
int roctxRangePushA(const char* message) {
  g_roctxPushed.push_back(message ? message : "<null>");
  return 0;
}
int roctxRangePop() {
  ++g_roctxPops;
  return 0;
}
#endif
static void* RoctxTestMalloc(size_t n) { return g_roctxMalloc ? g_roctxMalloc(n) : std::malloc(n); }
#define malloc(n) RoctxTestMalloc(n)
#include ROCTX_CC_PATH
#undef malloc

#include NVMLWRAP_STUB_CC_PATH

namespace {

class RoctxNvmlRoctxMicrotest : public ::testing::Test {
protected:
  void SetUp() override {
    ResetRecorders();
    savedNvtxToRoctx_ = nvtxToRoctx;
  }
  void TearDown() override {
    g_loadParam = [](const char*, int64_t deftVal) { return deftVal; };
    g_roctxMalloc = nullptr;
    // Restored here so a failed assertion can't leak entries into the next test.
    nvtxToRoctx = savedNvtxToRoctx_;
    ResetRecorders();
  }

  static void ResetRecorders() {
    g_logged.clear();
#ifdef ROCTX_ENABLE
    g_roctxPushed.clear();
    g_roctxPops = 0;
#endif
  }

  // Enter/exit events: roctracer push/pop with ROCTX_ENABLE, one WARN each without; none if off.
  static size_t RangeEvents() {
#ifdef ROCTX_ENABLE
    EXPECT_TRUE(g_logged.empty()) << "a ROCTX_ENABLE build must not fall back to WARN";
    return g_roctxPushed.size() + static_cast<size_t>(g_roctxPops);
#else
    return WarnCount();
#endif
  }

  // ROCTX_ENABLE builds can also check the message handed to roctracer; the fallback never sees it.
  static void ExpectPushedMessage(const char* expected) {
#ifdef ROCTX_ENABLE
    ASSERT_EQ(g_roctxPushed.size(), 1u);
    EXPECT_EQ(g_roctxPushed[0], expected);
#else
    (void)expected;
#endif
  }

  // One enter and one exit, counted apart so a destructor that pushes can't pass.
  static void ExpectOneRangeEnteredAndExited() {
#ifdef ROCTX_ENABLE
    EXPECT_TRUE(g_logged.empty()) << "a ROCTX_ENABLE build must not fall back to WARN";
    EXPECT_EQ(g_roctxPushed.size(), 1u);
    EXPECT_EQ(g_roctxPops, 1);
#else
    EXPECT_EQ(WarnCount(), 2u);
#endif
  }

  // Counts roctx.cc's mallocs while letting every one through.
  static int* CountMallocs() {
    static int calls;
    calls = 0;
    g_roctxMalloc = [](size_t n) -> void* {
      ++calls;
      return std::malloc(n);
    };
    return &calls;
  }

  // Lets the first `succeedFirst` roctx.cc mallocs through, then fails every one after.
  static int* FailMallocAfter(int succeedFirst) {
    static int calls;
    calls = 0;
    g_roctxMalloc = [succeedFirst](size_t n) -> void* { return calls++ < succeedFirst ? std::malloc(n) : nullptr; };
    return &calls;
  }

  static void SetLogRoctx(int64_t value) {
    g_loadParam = [value](const char* env, int64_t deftVal) {
      return std::strcmp(env, "RCCL_LOG_ROCTX") == 0 ? value : deftVal;
    };
  }

  static size_t WarnCount() {
    size_t n = 0;
    for (const auto& m : g_logged) n += (m.level == NCCL_LOG_WARN);
    return n;
  }

  // Runs stringify() over `entries` under `id` and returns the message it built.
  static std::string Stringify(const char* id, const std::vector<roctxPayloadSchemaEntryInfo>& entries) {
    roctxPayloadInfo info;
    roctxAlloc(&info, entries.empty() ? 1 : entries.size());
    info.id = id;
    info.numEntries = entries.size();
    for (size_t i = 0; i < entries.size(); ++i) info.payloadEntries[i] = entries[i];
    stringify(&info);
    std::string message = info.message;
    roctxFree(&info);
    return message;
  }

  static roctxPayloadSchemaEntryInfo IntEntry(const char* name, int value) {
    roctxPayloadSchemaEntryInfo e{name, ROCTX_PAYLOAD_ENTRY_TYPE_INT, {}};
    e.payload.typeInt = value;
    return e;
  }

  // Zero-initialised schema entry: the fields roctx.cc does not read stay at 0.
  static nvtxPayloadSchemaEntry_t Entry(uint64_t type, const char* name, uint64_t offset) {
    nvtxPayloadSchemaEntry_t e;
    std::memset(&e, 0, sizeof(e));
    e.type = type;
    e.name = name;
    e.offset = offset;
    return e;
  }

private:
  std::map<uint64_t, roctxPayloadEntryType> savedNvtxToRoctx_;
};

struct MixedPayload {
  int i;
  size_t s;
  ncclDevRedOp_t op;
  ncclDataType_t dt;
};

}  // namespace

// Lookup tables

TEST_F(RoctxNvmlRoctxMicrotest, NvtxToRoctx_MapsEachSupportedNvtxTypeToItsRoctxType) {
  EXPECT_EQ(nvtxToRoctx.size(), 4u);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_INT), ROCTX_PAYLOAD_ENTRY_TYPE_INT);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_SIZE), ROCTX_PAYLOAD_ENTRY_TYPE_SIZE);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_REDOP), ROCTX_PAYLOAD_ENTRY_TYPE_REDOP);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE), ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE);
}

// The names stringify() prints for redop/datatype payloads, indexed by enum value.
TEST_F(RoctxNvmlRoctxMicrotest, NameTables_MatchTheEnumOrder) {
  const char* redOps[] = {"Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv"};
  ASSERT_EQ(sizeof(redOps) / sizeof(redOps[0]), static_cast<size_t>(ncclNumDevRedOps));
  for (int i = 0; i < ncclNumDevRedOps; ++i) EXPECT_STREQ(ncclRedOpStr[i], redOps[i]) << "op " << i;

  const char* dataTypes[] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", "f64", "b16", "f8", "b8"};
  ASSERT_EQ(sizeof(dataTypes) / sizeof(dataTypes[0]), static_cast<size_t>(ncclNumTypes));
  for (int i = 0; i < ncclNumTypes; ++i) EXPECT_STREQ(ncclDataTypeStr[i], dataTypes[i]) << "type " << i;
}

// One slot per roctx entry type (DATATYPE has no initialiser; nothing reads it).
TEST_F(RoctxNvmlRoctxMicrotest, RoctxEntryTypeStr_HasOneSlotPerEntryType) {
  static_assert(sizeof(roctxEntryTypeStr) / sizeof(roctxEntryTypeStr[0]) == ROCTX_PAYLOAD_NUM_ENTRY_TYPES,
                "roctxEntryTypeStr must have one slot per roctxPayloadEntryType");
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_INT], "ROCTX_PAYLOAD_ENTRY_TYPE_INT");
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_SIZE], "ROCTX_PAYLOAD_ENTRY_TYPE_SIZE");
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_REDOP], "ROCTX_PAYLOAD_ENTRY_TYPE_REDOP");
}

// roctxAlloc / roctxFree

TEST_F(RoctxNvmlRoctxMicrotest, Alloc_AllocatesEntriesAndMessage_FreeReleasesBoth) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 3);
  ASSERT_NE(info.payloadEntries, nullptr);
  ASSERT_NE(info.message, nullptr);
  // Both buffers are usable at their full advertised size (under ASan, a short allocation fails here).
  std::memset(info.payloadEntries, 0, 3 * sizeof(roctxPayloadSchemaEntryInfo));
  std::memset(info.message, 'x', MAX_MESSAGE_LENGTH);
  roctxFree(&info);
}

// The entries allocation fails: both pointers end up null and the message is never allocated.
TEST_F(RoctxNvmlRoctxMicrotest, Alloc_EntriesAllocationFails_LeavesBothPointersNull) {
  int* calls = FailMallocAfter(0);
  roctxPayloadInfo info;
  info.message = reinterpret_cast<char*>(0x1);  // must be overwritten with nullptr
  roctxAlloc(&info, 2);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
  EXPECT_EQ(*calls, 1);
}

// The message allocation fails after the entries succeeded: entries are freed and nulled.
TEST_F(RoctxNvmlRoctxMicrotest, Alloc_MessageAllocationFails_ReleasesEntriesAndNullsBoth) {
  int* calls = FailMallocAfter(1);
  roctxPayloadInfo info;
  roctxAlloc(&info, 2);
  EXPECT_EQ(*calls, 2);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
  roctxFree(&info);
}

TEST_F(RoctxNvmlRoctxMicrotest, Free_ToleratesNullPointers) {
  roctxPayloadInfo info;  // both members default to nullptr
  roctxFree(&info);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
}

// extractPayloadInfo / stringify

TEST_F(RoctxNvmlRoctxMicrotest, Extract_NoEntriesBuffer_ReturnsWithoutTouchingInfo) {
  roctxPayloadInfo info;
  info.id = "untouched";
  info.numEntries = 77;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "a", 0)};
  int value = 1;
  nvtxPayloadData_t data{0, sizeof(value), &value};
  extractPayloadInfo(schema, &data, 1, "Sched", &info);
  EXPECT_STREQ(info.id, "untouched");
  EXPECT_EQ(info.numEntries, 77u);
}

TEST_F(RoctxNvmlRoctxMicrotest, Extract_ReadsEachTypeAtItsOffsetAndStringifies) {
  MixedPayload payload{-3, size_t{1} << 40, ncclDevProd, ncclFloat32};
  nvtxPayloadSchemaEntry_t schema[] = {
    Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "count", offsetof(MixedPayload, i)),
    Entry(NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "bytes", offsetof(MixedPayload, s)),
    Entry(NVTX_PAYLOAD_ENTRY_TYPE_REDOP, "op", offsetof(MixedPayload, op)),
    Entry(NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE, "type", offsetof(MixedPayload, dt)),
  };
  nvtxPayloadData_t data{0, sizeof(payload), &payload};
  roctxPayloadInfo info;
  roctxAlloc(&info, 4);
  extractPayloadInfo(schema, &data, 4, "AllReduce", &info);

  EXPECT_STREQ(info.id, "AllReduce");
  ASSERT_EQ(info.numEntries, 4u);
  EXPECT_STREQ(info.payloadEntries[0].name, "count");
  EXPECT_EQ(info.payloadEntries[0].type, ROCTX_PAYLOAD_ENTRY_TYPE_INT);
  EXPECT_EQ(info.payloadEntries[0].payload.typeInt, -3);
  EXPECT_EQ(info.payloadEntries[1].type, ROCTX_PAYLOAD_ENTRY_TYPE_SIZE);
  EXPECT_EQ(info.payloadEntries[1].payload.typeSize, size_t{1} << 40);
  EXPECT_EQ(info.payloadEntries[2].type, ROCTX_PAYLOAD_ENTRY_TYPE_REDOP);
  EXPECT_EQ(info.payloadEntries[2].payload.typeRedOp, ncclDevProd);
  EXPECT_EQ(info.payloadEntries[3].type, ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE);
  EXPECT_EQ(info.payloadEntries[3].payload.typeDataType, ncclFloat32);
  EXPECT_STREQ(info.message, "{AllReduce: count: -3, bytes: 1099511627776, op: Prod, type: f32}");
  roctxFree(&info);
}

// The separator is only written between entries: one entry has no trailing ", ".
TEST_F(RoctxNvmlRoctxMicrotest, Stringify_SingleEntryHasNoSeparator) {
  EXPECT_EQ(Stringify("Bcast", {IntEntry("root", 7)}), "{Bcast: root: 7}");
}

TEST_F(RoctxNvmlRoctxMicrotest, Stringify_ZeroEntriesIsJustTheId) {
  EXPECT_EQ(Stringify("Empty", {}), "{Empty: }");
}

// Out-of-range values print "unknown"; the top valid values sit next to them to pin the bounds.
TEST_F(RoctxNvmlRoctxMicrotest, Stringify_OutOfRangeRedOpAndDataTypePrintUnknown) {
  std::vector<roctxPayloadSchemaEntryInfo> entries(4);
  entries[0] = {"lastOp", ROCTX_PAYLOAD_ENTRY_TYPE_REDOP, {}};
  entries[0].payload.typeRedOp = static_cast<ncclDevRedOp_t>(ncclNumDevRedOps - 1);
  entries[1] = {"badOp", ROCTX_PAYLOAD_ENTRY_TYPE_REDOP, {}};
  entries[1].payload.typeRedOp = static_cast<ncclDevRedOp_t>(ncclNumDevRedOps);
  entries[2] = {"lastType", ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE, {}};
  entries[2].payload.typeDataType = static_cast<ncclDataType_t>(ncclNumTypes - 1);
  entries[3] = {"badType", ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE, {}};
  entries[3].payload.typeDataType = static_cast<ncclDataType_t>(ncclNumTypes);
  EXPECT_EQ(Stringify("X", entries), "{X: lastOp: SumPostDiv, badOp: unknown, lastType: b8, badType: unknown}");
}

TEST_F(RoctxNvmlRoctxMicrotest, Stringify_UnmappedRoctxTypeHitsTheDefaultBranch) {
  roctxPayloadSchemaEntryInfo weird{"weird", static_cast<roctxPayloadEntryType>(ROCTX_PAYLOAD_NUM_ENTRY_TYPES), {}};
  EXPECT_EQ(Stringify("X", {weird}), "{X: weird: unknown roctx payload type}");
}

TEST_F(RoctxNvmlRoctxMicrotest, Stringify_MissingBufferIsANoOp) {
  roctxPayloadInfo info;  // payloadEntries and message both null
  info.id = "X";
  info.numEntries = 0;
  stringify(&info);  // must not dereference the null message
  EXPECT_EQ(info.message, nullptr);
}

// Entries but no message: still a no-op (the guard is `!entries || !message`).
TEST_F(RoctxNvmlRoctxMicrotest, Stringify_EntriesWithoutMessageIsANoOp) {
  roctxPayloadSchemaEntryInfo entry{"n", ROCTX_PAYLOAD_ENTRY_TYPE_INT, {}};
  roctxPayloadInfo info;
  info.payloadEntries = &entry;
  info.id = "X";
  info.numEntries = 1;
  stringify(&info);
  EXPECT_EQ(info.message, nullptr);
}

// An out-of-enum roctx type stores no payload and prints the unknown-type text.
TEST_F(RoctxNvmlRoctxMicrotest, Extract_MappingToUnknownRoctxTypeTakesTheDefaultBranch) {
  const uint64_t nvtxType = NVTX_PAYLOAD_ENTRY_TYPE_INT + 2;
  ASSERT_EQ(nvtxToRoctx.count(nvtxType), 0u);
  const auto bogus = static_cast<roctxPayloadEntryType>(ROCTX_PAYLOAD_NUM_ENTRY_TYPES);
  nvtxToRoctx[nvtxType] = bogus;
  int value = 5;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(nvtxType, "w", 0)};
  nvtxPayloadData_t data{0, sizeof(value), &value};
  roctxPayloadInfo info;
  roctxAlloc(&info, 1);
  extractPayloadInfo(schema, &data, 1, "D", &info);
  EXPECT_EQ(info.payloadEntries[0].type, bogus);
  EXPECT_STREQ(info.message, "{D: w: unknown roctx payload type}");
  roctxFree(&info);
}

// Pinned as observed: an unmapped nvtx type is inserted by operator[] as INT and decoded as one.
TEST_F(RoctxNvmlRoctxMicrotest, Extract_UnmappedNvtxTypeIsTreatedAsIntAndInsertedIntoTheMap) {
  const uint64_t unmapped = NVTX_PAYLOAD_ENTRY_TYPE_INT + 1;
  ASSERT_EQ(nvtxToRoctx.count(unmapped), 0u);
  int value = 42;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(unmapped, "v", 0)};
  nvtxPayloadData_t data{0, sizeof(value), &value};
  roctxPayloadInfo info;
  roctxAlloc(&info, 1);
  extractPayloadInfo(schema, &data, 1, "U", &info);
  EXPECT_EQ(info.payloadEntries[0].type, ROCTX_PAYLOAD_ENTRY_TYPE_INT);
  EXPECT_STREQ(info.message, "{U: v: 42}");
  EXPECT_EQ(nvtxToRoctx.count(unmapped), 1u);  // TearDown restores the table
  roctxFree(&info);
}

// roctx_scoped_range_in

TEST_F(RoctxNvmlRoctxMicrotest, LogRoctxParam_IsReadFromRcclLogRoctxWithDefaultZero) {
  std::string seenEnv;
  int64_t seenDefault = -1;
  g_loadParam = [&](const char* env, int64_t deftVal) {
    seenEnv = env;
    seenDefault = deftVal;
    return deftVal;
  };
  EXPECT_EQ(rcclParamLogRoctx(), 0);
  EXPECT_EQ(seenEnv, "RCCL_LOG_ROCTX");
  EXPECT_EQ(seenDefault, 0);
}

TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_LoggingOff_DoesNothing) {
  SetLogRoctx(0);
  {
    roctx_scoped_range_in range("msg");
    roctx_scoped_range_in defaulted;
  }
  EXPECT_EQ(RangeEvents(), 0u);
  EXPECT_TRUE(g_logged.empty());
}

// One range: one enter event and one exit event.
TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_LoggingOn_OneEventOnEnterAndOneOnExit) {
  SetLogRoctx(1);
  {
    roctx_scoped_range_in range("msg");
    EXPECT_EQ(RangeEvents(), 1u);
    ExpectPushedMessage("msg");
  }
  ExpectOneRangeEnteredAndExited();
}

TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_DefaultConstructorDelegatesToTheMessageConstructor) {
  SetLogRoctx(1);
  {
    roctx_scoped_range_in range;
    ExpectPushedMessage("");
  }
  ExpectOneRangeEnteredAndExited();
}

// The constructor pushes the schema-built message; the destructor frees it in both builds.
TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_PayloadConstructorWithLoggingOn_PushesTheStringifiedPayload) {
  SetLogRoctx(1);
  int* mallocs = CountMallocs();
  MixedPayload payload{1, 2, ncclDevSum, ncclInt8};
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "n", offsetof(MixedPayload, i))};
  nvtxPayloadData_t data{0, sizeof(payload), &payload};
  {
    roctx_scoped_range_in range(schema, &data, 1, "RCCL_AllReduce");
    EXPECT_EQ(*mallocs, 2);
    EXPECT_EQ(RangeEvents(), 1u);
    ExpectPushedMessage("{RCCL_AllReduce: n: 1}");
  }
  ExpectOneRangeEnteredAndExited();
}

// roctxAlloc fails in the constructor: the range still enters and exits once, without crashing.
TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_PayloadConstructorWithLoggingOnAndAllocFailure_StillEntersAndExits) {
  SetLogRoctx(1);
  int* mallocs = FailMallocAfter(0);
  int value = 1;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "n", 0)};
  nvtxPayloadData_t data{0, sizeof(value), &value};
  { roctx_scoped_range_in range(schema, &data, 1, "RCCL_AllReduce"); }
  EXPECT_EQ(*mallocs, 1);
  ExpectOneRangeEnteredAndExited();
}

TEST_F(RoctxNvmlRoctxMicrotest, ScopedRange_PayloadConstructorWithLoggingOff_DoesNothing) {
  SetLogRoctx(0);
  int* mallocs = CountMallocs();
  int value = 1;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "n", 0)};
  nvtxPayloadData_t data{0, sizeof(value), &value};
  { roctx_scoped_range_in range(schema, &data, 1, "RCCL_AllReduce"); }
  EXPECT_EQ(*mallocs, 0);
  EXPECT_EQ(RangeEvents(), 0u);
  EXPECT_TRUE(g_logged.empty());
}

// nvmlwrap_stub.cc: no NVML on ROCm, so every query reports "no device".

namespace {
nvmlDevice_t FakeDevice() { return reinterpret_cast<nvmlDevice_t>(0x1234); }
}  // namespace

TEST(RoctxNvmlStubMicrotest, LifecycleCallsSucceed) {
  EXPECT_EQ(ncclNvmlSymbols(), ncclSuccess);
  EXPECT_EQ(ncclNvmlInit(), ncclSuccess);
  EXPECT_EQ(ncclNvmlShutdown(), ncclSuccess);
}

TEST(RoctxNvmlStubMicrotest, ReportsZeroDevices) {
  EXPECT_EQ(ncclNvmlDeviceCount, 0);
  unsigned int count = 99;
  EXPECT_EQ(ncclNvmlDeviceGetCount(&count), ncclSuccess);
  EXPECT_EQ(count, 0u);
}

// The successful getters write a fixed value; check it lands rather than a stale input.
TEST(RoctxNvmlStubMicrotest, SucceedingGettersWriteFixedValues) {
  unsigned index = 99;
  EXPECT_EQ(ncclNvmlDeviceGetIndex(FakeDevice(), &index), ncclSuccess);
  EXPECT_EQ(index, 0u);

  unsigned int minor = 99;
  EXPECT_EQ(ncclNvmlDeviceGetMinorNumber(FakeDevice(), &minor), ncclSuccess);
  EXPECT_EQ(minor, 0u);

  int major = 99, minorCc = 99;
  EXPECT_EQ(ncclNvmlDeviceGetCudaComputeCapability(FakeDevice(), &major, &minorCc), ncclSuccess);
  EXPECT_EQ(major, 1);
  EXPECT_EQ(minorCc, 1);
}

// The failing lookups return ncclSystemError and leave their out-params alone.
TEST(RoctxNvmlStubMicrotest, DeviceLookupsFailWithoutWritingOutputs) {
  nvmlDevice_t sentinel = FakeDevice();
  nvmlDevice_t dev = sentinel;
  EXPECT_EQ(ncclNvmlDeviceGetHandleByPciBusId("0000:00:00.0", &dev), ncclSystemError);
  EXPECT_EQ(dev, sentinel);
  EXPECT_EQ(ncclNvmlDeviceGetHandleByIndex(0, &dev), ncclSystemError);
  EXPECT_EQ(dev, sentinel);

  nvmlPciInfo_t pci;
  std::memset(&pci, 0xAB, sizeof(pci));
  EXPECT_EQ(ncclNvmlDeviceGetPciInfo(FakeDevice(), &pci), ncclSystemError);
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkRemotePciInfo(FakeDevice(), 0, &pci), ncclSystemError);
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&pci);
  for (size_t i = 0; i < sizeof(pci); ++i) ASSERT_EQ(bytes[i], 0xAB) << "byte " << i;

  nvmlEnableState_t state = NVML_FEATURE_ENABLED;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkState(FakeDevice(), 0, &state), ncclSystemError);
  EXPECT_EQ(state, NVML_FEATURE_ENABLED);

  unsigned int value = 99;
  EXPECT_EQ(ncclNvmlDeviceGetNvLinkCapability(FakeDevice(), 0, NVML_NVLINK_CAP_P2P_SUPPORTED, &value),
            ncclSystemError);
  EXPECT_EQ(ncclNvmlDeviceGetCurrPcieLinkGeneration(FakeDevice(), &value), ncclSystemError);
  EXPECT_EQ(ncclNvmlDeviceGetCurrPcieLinkWidth(FakeDevice(), &value), ncclSystemError);
  EXPECT_EQ(value, 99u);
}

// These fail too, but clear their output first.
TEST(RoctxNvmlStubMicrotest, FailingQueriesClearTheirOutputs) {
  char name[8] = "garbage";
  EXPECT_EQ(ncclNvmlDeviceGetName(FakeDevice(), name, sizeof(name)), ncclSystemError);
  EXPECT_EQ(name[0], '\0');

  char version[8] = "garbage";
  EXPECT_EQ(ncclNvmlSystemGetDriverVersion(version, sizeof(version)), ncclSystemError);
  EXPECT_EQ(version[0], '\0');

  unsigned long long count = 99;
  EXPECT_EQ(ncclNvmlDeviceGetMemoryErrorCounter(FakeDevice(), NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_VOLATILE_ECC,
                                                NVML_MEMORY_LOCATION_DRAM, &count),
            ncclSystemError);
  EXPECT_EQ(count, 0u);
}

// A zero length means "no buffer": nothing may be written, not even the terminator.
TEST(RoctxNvmlStubMicrotest, ZeroLengthBuffersAreNotWritten) {
  char name[2] = {'n', 'x'};
  EXPECT_EQ(ncclNvmlDeviceGetName(FakeDevice(), name, 0), ncclSystemError);
  EXPECT_EQ(name[0], 'n');

  char version[2] = {'v', 'x'};
  EXPECT_EQ(ncclNvmlSystemGetDriverVersion(version, 0), ncclSystemError);
  EXPECT_EQ(version[0], 'v');
}

TEST(RoctxNvmlStubMicrotest, FieldValuesMarksEveryRequestedFieldNotSupported) {
  nvmlFieldValue_t values[3];
  std::memset(values, 0, sizeof(values));
  for (auto& v : values) v.nvmlReturn = NVML_SUCCESS;
  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(FakeDevice(), 2, values), ncclSystemError);
  EXPECT_EQ(values[0].nvmlReturn, NVML_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(values[1].nvmlReturn, NVML_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(values[2].nvmlReturn, NVML_SUCCESS) << "only valuesCount entries may be touched";

  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(FakeDevice(), 0, values), ncclSystemError);
}
