/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/roctx.cc (NVTX payload -> roctx message
// translation and the RAII range used by NCCL_NVTX3_FUNC_RANGE*) and
// src/misc/nvmlwrap_stub.cc (the ROCm stand-in for NVML, which reports no
// devices so NVML-driven callers skip their checks).
//
// Both files are compiled for real into this binary. Neither needs the shared
// fakes: this TU supplies its own ncclDebugLog (to record WARNs) and
// g_loadParam (param_redirect.h routes rcclParamLogRoctx() through it on every
// call instead of caching, so each test picks LOG_ROCTX independently).
//
// The scoped range has two production arms. In-tree builds where roctracer is
// found inherit ROCTX_ENABLE from the rccl target and call roctxRangePushA/Pop,
// which this TU defines to record the messages pushed. The standalone build
// has no ROCTX_ENABLE and falls back to one WARN per enter/exit. The range
// tests are written to hold in either build.
//
// Every stringify() case here stays under MAX_MESSAGE_LENGTH on purpose: past
// it, stringify() writes beyond its 1024-byte buffer (confirmed under ASan),
// tracked as AICOMRCCL-2914 rather than pinned here as undefined behaviour.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
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

// malloc seam for roctx.cc only, so roctxAlloc's two failure paths are reachable. Every header
// roctx.cc includes is pulled in first, so their include guards keep the macro below from
// rewriting any declaration; it applies to roctx.cc's own function bodies and nothing else.
#include <cstdlib>
#include "roctx.h"
namespace {
std::function<void*(size_t)> g_roctxMalloc;
}  // namespace

// In-tree builds inherit the rccl target's COMPILE_DEFINITIONS (test/CMakeLists.txt), so
// ROCTX_ENABLE is defined here whenever roctracer was found, and roctx.cc then calls the real
// roctracer entry points. Nothing on the micro-test link line provides them, so record the calls
// instead; that also lets the range tests check the exact message pushed. The standalone build has
// no ROCTX_ENABLE and exercises roctx.cc's WARN fallback instead.
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

class RoctxMicrotest : public ::testing::Test {
protected:
  void SetUp() override { ResetRecorders(); }
  void TearDown() override {
    g_loadParam = [](const char*, int64_t deftVal) { return deftVal; };
    g_roctxMalloc = nullptr;
    ResetRecorders();
  }

  static void ResetRecorders() {
    g_logged.clear();
#ifdef ROCTX_ENABLE
    g_roctxPushed.clear();
    g_roctxPops = 0;
#endif
  }

  // Range enter/exit events, whichever form this build gives them: roctracer push/pop calls with
  // ROCTX_ENABLE, one WARN each without it. With logging off there must be none either way.
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

  // Zero-initialised schema entry: the fields roctx.cc does not read stay at 0.
  static nvtxPayloadSchemaEntry_t Entry(uint64_t type, const char* name, uint64_t offset) {
    nvtxPayloadSchemaEntry_t e;
    std::memset(&e, 0, sizeof(e));
    e.type = type;
    e.name = name;
    e.offset = offset;
    return e;
  }
};

struct MixedPayload {
  int i;
  size_t s;
  ncclDevRedOp_t op;
  ncclDataType_t dt;
};

}  // namespace

// ---------------------------------------------------------------------------
// Lookup tables
// ---------------------------------------------------------------------------

TEST_F(RoctxMicrotest, NvtxToRoctx_MapsEachSupportedNvtxTypeToItsRoctxType) {
  EXPECT_EQ(nvtxToRoctx.size(), 4u);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_INT), ROCTX_PAYLOAD_ENTRY_TYPE_INT);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_SIZE), ROCTX_PAYLOAD_ENTRY_TYPE_SIZE);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_REDOP), ROCTX_PAYLOAD_ENTRY_TYPE_REDOP);
  EXPECT_EQ(nvtxToRoctx.at(NVTX_PAYLOAD_ENTRY_TYPE_DATATYPE), ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE);
}

// The names stringify() prints for redop/datatype payloads, indexed by enum value.
TEST_F(RoctxMicrotest, NameTables_MatchTheEnumOrder) {
  const char* redOps[] = {"Sum", "Prod", "MinMax", "PreMulSum", "SumPostDiv"};
  ASSERT_EQ(sizeof(redOps) / sizeof(redOps[0]), static_cast<size_t>(ncclNumDevRedOps));
  for (int i = 0; i < ncclNumDevRedOps; ++i) EXPECT_STREQ(ncclRedOpStr[i], redOps[i]) << "op " << i;

  const char* dataTypes[] = {"i8", "u8", "i32", "u32", "i64", "u64", "f16", "f32", "f64", "b16", "f8", "b8"};
  ASSERT_EQ(sizeof(dataTypes) / sizeof(dataTypes[0]), static_cast<size_t>(ncclNumTypes));
  for (int i = 0; i < ncclNumTypes; ++i) EXPECT_STREQ(ncclDataTypeStr[i], dataTypes[i]) << "type " << i;
}

// Pinned as observed, not endorsed: the table has 4 slots but only 3 initialisers, so the
// DATATYPE slot is null. Nothing in the tree reads roctxEntryTypeStr today.
TEST_F(RoctxMicrotest, RoctxEntryTypeStr_DatatypeSlotIsNull) {
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_INT], "ROCTX_PAYLOAD_ENTRY_TYPE_INT");
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_SIZE], "ROCTX_PAYLOAD_ENTRY_TYPE_SIZE");
  EXPECT_STREQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_REDOP], "ROCTX_PAYLOAD_ENTRY_TYPE_REDOP");
  EXPECT_EQ(roctxEntryTypeStr[ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE], nullptr);
}

// ---------------------------------------------------------------------------
// roctxAlloc / roctxFree
// ---------------------------------------------------------------------------

TEST_F(RoctxMicrotest, Alloc_AllocatesEntriesAndMessage_FreeReleasesBoth) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 3);
  ASSERT_NE(info.payloadEntries, nullptr);
  ASSERT_NE(info.message, nullptr);
  // Both buffers are usable at their full advertised size (ASan flags a short allocation).
  std::memset(info.payloadEntries, 0, 3 * sizeof(roctxPayloadSchemaEntryInfo));
  std::memset(info.message, 'x', MAX_MESSAGE_LENGTH);
  roctxFree(&info);
}

// The entries allocation fails: both pointers end up null and the message is never allocated.
TEST_F(RoctxMicrotest, Alloc_EntriesAllocationFails_LeavesBothPointersNull) {
  int* calls = FailMallocAfter(0);
  roctxPayloadInfo info;
  info.message = reinterpret_cast<char*>(0x1);  // must be overwritten with nullptr
  roctxAlloc(&info, 2);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
  EXPECT_EQ(*calls, 1);
}

// The message allocation fails after the entries succeeded: the entries buffer is released
// (LeakSanitizer catches a missing free) and nulled, so a later roctxFree cannot double-free.
TEST_F(RoctxMicrotest, Alloc_MessageAllocationFails_ReleasesEntriesAndNullsBoth) {
  int* calls = FailMallocAfter(1);
  roctxPayloadInfo info;
  roctxAlloc(&info, 2);
  EXPECT_EQ(*calls, 2);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
  roctxFree(&info);
}

TEST_F(RoctxMicrotest, Free_ToleratesNullPointers) {
  roctxPayloadInfo info;  // both members default to nullptr
  roctxFree(&info);
  EXPECT_EQ(info.payloadEntries, nullptr);
  EXPECT_EQ(info.message, nullptr);
}

// ---------------------------------------------------------------------------
// extractPayloadInfo / stringify
// ---------------------------------------------------------------------------

TEST_F(RoctxMicrotest, Extract_NoEntriesBuffer_ReturnsWithoutTouchingInfo) {
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

TEST_F(RoctxMicrotest, Extract_ReadsEachTypeAtItsOffsetAndStringifies) {
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
TEST_F(RoctxMicrotest, Stringify_SingleEntryHasNoSeparator) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 1);
  info.id = "Bcast";
  info.numEntries = 1;
  info.payloadEntries[0].name = "root";
  info.payloadEntries[0].type = ROCTX_PAYLOAD_ENTRY_TYPE_INT;
  info.payloadEntries[0].payload.typeInt = 7;
  stringify(&info);
  EXPECT_STREQ(info.message, "{Bcast: root: 7}");
  roctxFree(&info);
}

TEST_F(RoctxMicrotest, Stringify_ZeroEntriesIsJustTheId) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 1);
  info.id = "Empty";
  info.numEntries = 0;
  stringify(&info);
  EXPECT_STREQ(info.message, "{Empty: }");
  roctxFree(&info);
}

// Out-of-range redop/datatype values print "unknown" instead of indexing past the tables;
// the highest valid values sit right next to them so an off-by-one in either bound shows.
TEST_F(RoctxMicrotest, Stringify_OutOfRangeRedOpAndDataTypePrintUnknown) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 4);
  info.id = "X";
  info.numEntries = 4;
  info.payloadEntries[0] = {"lastOp", ROCTX_PAYLOAD_ENTRY_TYPE_REDOP, {}};
  info.payloadEntries[0].payload.typeRedOp = static_cast<ncclDevRedOp_t>(ncclNumDevRedOps - 1);
  info.payloadEntries[1] = {"badOp", ROCTX_PAYLOAD_ENTRY_TYPE_REDOP, {}};
  info.payloadEntries[1].payload.typeRedOp = static_cast<ncclDevRedOp_t>(ncclNumDevRedOps);
  info.payloadEntries[2] = {"lastType", ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE, {}};
  info.payloadEntries[2].payload.typeDataType = static_cast<ncclDataType_t>(ncclNumTypes - 1);
  info.payloadEntries[3] = {"badType", ROCTX_PAYLOAD_ENTRY_TYPE_DATATYPE, {}};
  info.payloadEntries[3].payload.typeDataType = static_cast<ncclDataType_t>(ncclNumTypes);
  stringify(&info);
  EXPECT_STREQ(info.message, "{X: lastOp: SumPostDiv, badOp: unknown, lastType: b8, badType: unknown}");
  roctxFree(&info);
}

TEST_F(RoctxMicrotest, Stringify_UnmappedRoctxTypeHitsTheDefaultBranch) {
  roctxPayloadInfo info;
  roctxAlloc(&info, 1);
  info.id = "X";
  info.numEntries = 1;
  info.payloadEntries[0] = {"weird", static_cast<roctxPayloadEntryType>(ROCTX_PAYLOAD_NUM_ENTRY_TYPES), {}};
  stringify(&info);
  EXPECT_STREQ(info.message, "{X: weird: unknown roctx payload type}");
  roctxFree(&info);
}

TEST_F(RoctxMicrotest, Stringify_MissingBufferIsANoOp) {
  roctxPayloadInfo info;  // payloadEntries and message both null
  info.id = "X";
  info.numEntries = 0;
  stringify(&info);  // must not dereference the null message
  EXPECT_EQ(info.message, nullptr);
}

// Entries present but no message buffer: still a no-op, checked separately from the
// both-null case above because the guard is `!entries || !message`.
TEST_F(RoctxMicrotest, Stringify_EntriesWithoutMessageIsANoOp) {
  roctxPayloadSchemaEntryInfo entry{"n", ROCTX_PAYLOAD_ENTRY_TYPE_INT, {}};
  roctxPayloadInfo info;
  info.payloadEntries = &entry;
  info.id = "X";
  info.numEntries = 1;
  stringify(&info);
  EXPECT_EQ(info.message, nullptr);
}

// A mapping to a roctx type outside the enum reaches extractPayloadInfo's default branch,
// which stores the type but no payload, and stringify then prints the unknown-type text.
TEST_F(RoctxMicrotest, Extract_MappingToUnknownRoctxTypeTakesTheDefaultBranch) {
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
  nvtxToRoctx.erase(nvtxType);
  EXPECT_EQ(info.payloadEntries[0].type, bogus);
  EXPECT_STREQ(info.message, "{D: w: unknown roctx payload type}");
  roctxFree(&info);
}

// Pinned as observed: an nvtx type with no mapping is looked up with std::map::operator[],
// which inserts it as ROCTX_PAYLOAD_ENTRY_TYPE_INT (value 0) and decodes the field as an int.
TEST_F(RoctxMicrotest, Extract_UnmappedNvtxTypeIsTreatedAsIntAndInsertedIntoTheMap) {
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
  EXPECT_EQ(nvtxToRoctx.count(unmapped), 1u);
  nvtxToRoctx.erase(unmapped);  // keep the global table as other tests expect it
  roctxFree(&info);
}

// ---------------------------------------------------------------------------
// roctx_scoped_range_in
// ---------------------------------------------------------------------------

TEST_F(RoctxMicrotest, LogRoctxParam_IsReadFromRcclLogRoctxWithDefaultZero) {
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

TEST_F(RoctxMicrotest, ScopedRange_LoggingOff_DoesNothing) {
  SetLogRoctx(0);
  {
    roctx_scoped_range_in range("msg");
    roctx_scoped_range_in defaulted;
  }
  EXPECT_EQ(RangeEvents(), 0u);
  EXPECT_TRUE(g_logged.empty());
}

// One range is one event on enter and one on exit: a roctracer push/pop with ROCTX_ENABLE,
// otherwise the WARN that stands in for each.
TEST_F(RoctxMicrotest, ScopedRange_LoggingOn_OneEventOnEnterAndOneOnExit) {
  SetLogRoctx(1);
  {
    roctx_scoped_range_in range("msg");
    EXPECT_EQ(RangeEvents(), 1u);
    ExpectPushedMessage("msg");
  }
  EXPECT_EQ(RangeEvents(), 2u);
#ifndef ROCTX_ENABLE
  for (const auto& m : g_logged) EXPECT_NE(m.text.find("ROCTX_ENABLE is not defined"), std::string::npos);
#endif
}

TEST_F(RoctxMicrotest, ScopedRange_DefaultConstructorDelegatesToTheMessageConstructor) {
  SetLogRoctx(1);
  {
    roctx_scoped_range_in range;
    ExpectPushedMessage("");
  }
  EXPECT_EQ(RangeEvents(), 2u);
}

// The payload constructor builds the message from the schema (pushed verbatim with ROCTX_ENABLE)
// and the destructor frees it in both builds; LeakSanitizer would flag a missing roctxFree.
TEST_F(RoctxMicrotest, ScopedRange_PayloadConstructorWithLoggingOn_PushesTheStringifiedPayload) {
  SetLogRoctx(1);
  MixedPayload payload{1, 2, ncclDevSum, ncclInt8};
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "n", offsetof(MixedPayload, i))};
  nvtxPayloadData_t data{0, sizeof(payload), &payload};
  {
    roctx_scoped_range_in range(schema, &data, 1, "RCCL_AllReduce");
    EXPECT_EQ(RangeEvents(), 1u);
    ExpectPushedMessage("{RCCL_AllReduce: n: 1}");
  }
  EXPECT_EQ(RangeEvents(), 2u);
}

TEST_F(RoctxMicrotest, ScopedRange_PayloadConstructorWithLoggingOff_DoesNothing) {
  SetLogRoctx(0);
  int value = 1;
  nvtxPayloadSchemaEntry_t schema[] = {Entry(NVTX_PAYLOAD_ENTRY_TYPE_INT, "n", 0)};
  nvtxPayloadData_t data{0, sizeof(value), &value};
  { roctx_scoped_range_in range(schema, &data, 1, "RCCL_AllReduce"); }
  EXPECT_EQ(RangeEvents(), 0u);
  EXPECT_TRUE(g_logged.empty());
}

// ---------------------------------------------------------------------------
// nvmlwrap_stub.cc: no NVML on ROCm, so every query reports "no device".
// ---------------------------------------------------------------------------

namespace {
nvmlDevice_t FakeDevice() { return reinterpret_cast<nvmlDevice_t>(0x1234); }
}  // namespace

TEST(NvmlStubMicrotest, LifecycleCallsSucceed) {
  EXPECT_EQ(ncclNvmlSymbols(), ncclSuccess);
  EXPECT_EQ(ncclNvmlInit(), ncclSuccess);
  EXPECT_EQ(ncclNvmlShutdown(), ncclSuccess);
}

TEST(NvmlStubMicrotest, ReportsZeroDevices) {
  EXPECT_EQ(ncclNvmlDeviceCount, 0);
  unsigned int count = 99;
  EXPECT_EQ(ncclNvmlDeviceGetCount(&count), ncclSuccess);
  EXPECT_EQ(count, 0u);
}

// The successful getters write a fixed value; check it lands rather than a stale input.
TEST(NvmlStubMicrotest, SucceedingGettersWriteFixedValues) {
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
TEST(NvmlStubMicrotest, DeviceLookupsFailWithoutWritingOutputs) {
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

// These fail too, but clear their output first so a caller that ignores the error
// reads a defined value instead of stack garbage.
TEST(NvmlStubMicrotest, FailingQueriesClearTheirOutputs) {
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
TEST(NvmlStubMicrotest, ZeroLengthBuffersAreNotWritten) {
  char name[2] = {'n', 'x'};
  EXPECT_EQ(ncclNvmlDeviceGetName(FakeDevice(), name, 0), ncclSystemError);
  EXPECT_EQ(name[0], 'n');

  char version[2] = {'v', 'x'};
  EXPECT_EQ(ncclNvmlSystemGetDriverVersion(version, 0), ncclSystemError);
  EXPECT_EQ(version[0], 'v');
}

TEST(NvmlStubMicrotest, FieldValuesMarksEveryRequestedFieldNotSupported) {
  nvmlFieldValue_t values[3];
  std::memset(values, 0, sizeof(values));
  for (auto& v : values) v.nvmlReturn = NVML_SUCCESS;
  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(FakeDevice(), 2, values), ncclSystemError);
  EXPECT_EQ(values[0].nvmlReturn, NVML_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(values[1].nvmlReturn, NVML_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(values[2].nvmlReturn, NVML_SUCCESS) << "only valuesCount entries may be touched";

  EXPECT_EQ(ncclNvmlDeviceGetFieldValues(FakeDevice(), 0, values), ncclSystemError);
}
