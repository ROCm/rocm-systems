/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/api_trace.cc: the rcclApiFuncTable layout and every public entry
// point's dispatch through it. Each _impl is a recording fake defined here.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "nccl.h"
#include "debug.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
std::vector<std::string> g_warnings;
std::vector<std::string> g_recorded;
std::vector<std::string> g_rangeNames;
std::vector<size_t> g_rangeBytes;
int g_rangesOpen = 0;

struct Call {
  std::string name;
  std::vector<uintptr_t> args;
};
std::vector<Call> g_calls;
ncclResult_t g_result = ncclSuccess;

template <typename T>
uintptr_t W(T v) {
  if constexpr (std::is_pointer_v<T>) return reinterpret_cast<uintptr_t>(v);
  else return static_cast<uintptr_t>(v);
}
uintptr_t W(ncclUniqueId id) {
  uintptr_t v;
  std::memcpy(&v, id.internal, sizeof(v));
  return v;
}

template <typename... A>
ncclResult_t Hit(const char* name, A... a) {
  g_calls.push_back({name, {W(a)...}});
  return g_result;
}
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (level == NCCL_LOG_WARN) g_warnings.push_back(buf);
}

#include API_TRACE_CC_PATH

namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder inst;
  return inst;
}
void Recorder::record(const char* name) { g_recorded.push_back(name); }
}  // namespace rccl

roctx_scoped_range_in::roctx_scoped_range_in(const nvtxPayloadSchemaEntry_t* /*schema*/,
                                             const nvtxPayloadData_t* data, const size_t /*numEntries*/,
                                             const char* schemaName) noexcept {
  g_rangeNames.push_back(schemaName);
  g_rangeBytes.push_back(static_cast<const NcclNvtxParamsMSCCL*>(data[0].payload)->bytes);
  ++g_rangesOpen;
}
roctx_scoped_range_in::~roctx_scoped_range_in() { --g_rangesOpen; }

// clang-format off
ncclResult_t ncclAllGather_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclComm_t c, hipStream_t st) {
  return Hit("AllGather", s, r, n, t, c, st);
}
ncclResult_t ncclAllReduce_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t op, ncclComm* c,
                                hipStream_t st) {
  return Hit("AllReduce", s, r, n, t, op, c, st);
}
ncclResult_t ncclAllReduceWithBias_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t op,
                                        ncclComm* c, hipStream_t st, const void* acc) {
  return Hit("AllReduceWithBias", s, r, n, t, op, c, st, acc);
}
ncclResult_t ncclAlltoAll_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclComm_t c, hipStream_t st) {
  return Hit("AlltoAll", s, r, n, t, c, st);
}
ncclResult_t ncclAlltoAllv_impl(const void* s, const size_t sc[], const size_t sd[], void* r, const size_t rc[],
                                const size_t rd[], ncclDataType_t t, ncclComm_t c, hipStream_t st) {
  return Hit("AlltoAllv", s, sc, sd, r, rc, rd, t, c, st);
}
ncclResult_t ncclBroadcast_impl(const void* s, void* r, size_t n, ncclDataType_t t, int root, ncclComm_t c,
                                hipStream_t st) {
  return Hit("Broadcast", s, r, n, t, root, c, st);
}
ncclResult_t ncclGather_impl(const void* s, void* r, size_t n, ncclDataType_t t, int root, ncclComm_t c,
                             hipStream_t st) {
  return Hit("Gather", s, r, n, t, root, c, st);
}
ncclResult_t ncclReduce_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t op, int root,
                             ncclComm_t c, hipStream_t st) {
  return Hit("Reduce", s, r, n, t, op, root, c, st);
}
ncclResult_t ncclReduceScatter_impl(const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t op, ncclComm* c,
                                    hipStream_t st) {
  return Hit("ReduceScatter", s, r, n, t, op, c, st);
}
ncclResult_t ncclScatter_impl(const void* s, void* r, size_t n, ncclDataType_t t, int root, ncclComm_t c,
                              hipStream_t st) {
  return Hit("Scatter", s, r, n, t, root, c, st);
}
ncclResult_t ncclSend_impl(const void* s, size_t n, ncclDataType_t t, int peer, ncclComm_t c, hipStream_t st) {
  return Hit("Send", s, n, t, peer, c, st);
}
ncclResult_t ncclRecv_impl(void* r, size_t n, ncclDataType_t t, int peer, ncclComm_t c, hipStream_t st) {
  return Hit("Recv", r, n, t, peer, c, st);
}
ncclResult_t ncclPutSignal_impl(const void* l, size_t n, ncclDataType_t t, int peer, ncclWindow_t w, size_t off,
                                int sig, int ctx, unsigned int flags, ncclComm_t c, hipStream_t st) {
  return Hit("PutSignal", l, n, t, peer, w, off, sig, ctx, flags, c, st);
}
ncclResult_t ncclSignal_impl(int peer, int sig, int ctx, unsigned int flags, ncclComm_t c, hipStream_t st) {
  return Hit("Signal", peer, sig, ctx, flags, c, st);
}
ncclResult_t ncclWaitSignal_impl(int nDesc, ncclWaitSignalDesc_t* d, ncclComm_t c, hipStream_t st) {
  return Hit("WaitSignal", nDesc, d, c, st);
}
ncclResult_t ncclRedOpCreatePreMulSum_impl(ncclRedOp_t* op, void* scalar, ncclDataType_t t,
                                           ncclScalarResidence_t res, ncclComm_t c) {
  return Hit("RedOpCreatePreMulSum", op, scalar, t, res, c);
}
ncclResult_t ncclRedOpDestroy_impl(ncclRedOp_t op, ncclComm_t c) { return Hit("RedOpDestroy", op, c); }
ncclResult_t ncclGroupStart_impl() { return Hit("GroupStart"); }
ncclResult_t ncclGroupEnd_impl() { return Hit("GroupEnd"); }
ncclResult_t ncclGetVersion_impl(int* v) { return Hit("GetVersion", v); }
ncclResult_t ncclGetUniqueId_impl(ncclUniqueId* out) { return Hit("GetUniqueId", out); }
ncclResult_t ncclCommInitRank_impl(ncclComm_t* nc, int n, ncclUniqueId id, int rank) {
  return Hit("CommInitRank", nc, n, id, rank);
}
ncclResult_t ncclCommInitAll_impl(ncclComm_t* comms, int ndev, const int* devs) {
  return Hit("CommInitAll", comms, ndev, devs);
}
ncclResult_t ncclCommInitRankConfig_impl(ncclComm_t* nc, int n, ncclUniqueId id, int rank, ncclConfig_t* cfg) {
  return Hit("CommInitRankConfig", nc, n, id, rank, cfg);
}
ncclResult_t ncclCommFinalize_impl(ncclComm_t c) { return Hit("CommFinalize", c); }
ncclResult_t ncclCommDestroy_impl(ncclComm_t c) { return Hit("CommDestroy", c); }
ncclResult_t ncclCommAbort_impl(ncclComm_t c) { return Hit("CommAbort", c); }
ncclResult_t ncclCommRevoke_impl(ncclComm_t c, int flags) { return Hit("CommRevoke", c, flags); }
ncclResult_t ncclCommShrink_impl(ncclComm_t c, int* ex, int nEx, ncclComm_t* nc, ncclConfig_t* cfg, int flags) {
  return Hit("CommShrink", c, ex, nEx, nc, cfg, flags);
}
ncclResult_t ncclCommSplit_impl(ncclComm_t c, int color, int key, ncclComm_t* nc, ncclConfig_t* cfg) {
  return Hit("CommSplit", c, color, key, nc, cfg);
}
const char* ncclGetErrorString_impl(ncclResult_t code) {
  Hit("GetErrorString", code);
  return "error-string";
}
const char* ncclGetLastError_impl(const ncclComm_t c) {
  Hit("GetLastError", c);
  return "last-error";
}
ncclResult_t ncclCommGetAsyncError_impl(ncclComm_t c, ncclResult_t* e) { return Hit("CommGetAsyncError", c, e); }
ncclResult_t ncclCommCount_impl(const ncclComm_t c, int* n) { return Hit("CommCount", c, n); }
ncclResult_t ncclCommCuDevice_impl(const ncclComm_t c, int* d) { return Hit("CommCuDevice", c, d); }
ncclResult_t ncclCommUserRank_impl(const ncclComm_t c, int* r) { return Hit("CommUserRank", c, r); }
ncclResult_t ncclMemAlloc_impl(void** p, size_t n) { return Hit("MemAlloc", p, n); }
ncclResult_t ncclMemFree_impl(void* p) { return Hit("MemFree", p); }
ncclResult_t ncclCommRegister_impl(const ncclComm_t c, void* b, size_t n, void** h) {
  return Hit("CommRegister", c, b, n, h);
}
ncclResult_t ncclCommDeregister_impl(const ncclComm_t c, void* h) { return Hit("CommDeregister", c, h); }
ncclResult_t ncclCommWindowRegister_impl(ncclComm_t c, void* b, size_t n, ncclWindow_t* w, int flags) {
  return Hit("CommWindowRegister", c, b, n, w, flags);
}
ncclResult_t ncclCommWindowDeregister_impl(ncclComm_t c, ncclWindow_t w) { return Hit("CommWindowDeregister", c, w); }
ncclResult_t ncclCommSuspend_impl(ncclComm_t c, int flags) { return Hit("CommSuspend", c, flags); }
ncclResult_t ncclCommResume_impl(ncclComm_t c) { return Hit("CommResume", c); }
ncclResult_t ncclCommMemStats_impl(ncclComm_t c, ncclCommMemStat_t s, uint64_t* v) {
  return Hit("CommMemStats", c, s, v);
}
ncclResult_t ncclCommGetUniqueId_impl(ncclComm_t c, ncclUniqueId* id) { return Hit("CommGetUniqueId", c, id); }
ncclResult_t ncclCommGrow_impl(ncclComm_t c, int n, const ncclUniqueId* id, int rank, ncclComm_t* nc,
                               ncclConfig_t* cfg) {
  return Hit("CommGrow", c, n, id, rank, nc, cfg);
}
// clang-format on

namespace {

constexpr ncclResult_t kResult = ncclRemoteError;  // every fake returns this, so propagation is checked too

// Distinct non-null handles; only their values are compared.
template <typename T>
T H(uintptr_t v) {
  return reinterpret_cast<T>(v);
}
const auto kComm = H<ncclComm_t>(0xc0);
const auto kStream = H<hipStream_t>(0x57);
const auto kSend = H<const void*>(0x5e);
const auto kRecv = H<void*>(0x4e);

#define EXPECT_FORWARDED(expr, fn, ...)                                     \
  do {                                                                      \
    g_calls.clear();                                                        \
    EXPECT_EQ((expr), kResult) << #expr;                                    \
    ASSERT_EQ(g_calls.size(), 1u) << #expr;                                 \
    EXPECT_EQ(g_calls[0].name, fn) << #expr;                                \
    EXPECT_EQ(g_calls[0].args, (std::vector<uintptr_t>{__VA_ARGS__})) << #expr; \
  } while (0)

class ApiTraceMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_calls.clear();
    g_warnings.clear();
    g_recorded.clear();
    g_rangeNames.clear();
    g_rangeBytes.clear();
    g_result = kResult;
  }
};

TEST_F(ApiTraceMicrotest, Table_IsBuiltOnceInPlace) {
  const rcclApiFuncTable* tbl = rccl::RcclGetFunctionTable();
  EXPECT_EQ(tbl, rccl::RcclGetFunctionTable());
  EXPECT_EQ(static_cast<const void*>(tbl), static_cast<const void*>(rccl::m_buffer.data()));
  EXPECT_EQ(tbl->size, sizeof(rcclApiFuncTable));
  volatile size_t members = 52;  // evaluated at run time too, not only in the static_asserts
  EXPECT_EQ(rccl::compute_table_size(members), sizeof(rcclApiFuncTable));
  EXPECT_EQ(rccl::compute_table_offset(members - 1), offsetof(rcclApiFuncTable, ncclCommGrow_fn));
}

// The legacy ncclAllToAll(v) entry points dispatch through the ncclAlltoAll(v) slots.
TEST_F(ApiTraceMicrotest, Table_SlotsHoldTheMatchingImplementations) {
  const rcclApiFuncTable* t = rccl::RcclGetFunctionTable();
  EXPECT_EQ(t->ncclAllGather_fn, &ncclAllGather_impl);
  EXPECT_EQ(t->ncclAllReduce_fn, &ncclAllReduce_impl);
  EXPECT_EQ(t->ncclAllToAll_fn, nullptr);
  EXPECT_EQ(t->ncclAllToAllv_fn, nullptr);
  EXPECT_EQ(t->ncclBroadcast_fn, &ncclBroadcast_impl);
  EXPECT_EQ(t->ncclGather_fn, &ncclGather_impl);
  EXPECT_EQ(t->ncclReduce_fn, &ncclReduce_impl);
  EXPECT_EQ(t->ncclReduceScatter_fn, &ncclReduceScatter_impl);
  EXPECT_EQ(t->ncclScatter_fn, &ncclScatter_impl);
  EXPECT_EQ(t->ncclSend_fn, &ncclSend_impl);
  EXPECT_EQ(t->ncclRecv_fn, &ncclRecv_impl);
  EXPECT_EQ(t->ncclRedOpCreatePreMulSum_fn, &ncclRedOpCreatePreMulSum_impl);
  EXPECT_EQ(t->ncclRedOpDestroy_fn, &ncclRedOpDestroy_impl);
  EXPECT_EQ(t->ncclGroupStart_fn, &ncclGroupStart_impl);
  EXPECT_EQ(t->ncclGroupEnd_fn, &ncclGroupEnd_impl);
  EXPECT_EQ(t->ncclGetVersion_fn, &ncclGetVersion_impl);
  EXPECT_EQ(t->ncclGetUniqueId_fn, &ncclGetUniqueId_impl);
  EXPECT_EQ(t->ncclCommInitRank_fn, &ncclCommInitRank_impl);
  EXPECT_EQ(t->ncclCommInitAll_fn, &ncclCommInitAll_impl);
  EXPECT_EQ(t->ncclCommInitRankConfig_fn, &ncclCommInitRankConfig_impl);
  EXPECT_EQ(t->ncclCommFinalize_fn, &ncclCommFinalize_impl);
  EXPECT_EQ(t->ncclCommDestroy_fn, &ncclCommDestroy_impl);
  EXPECT_EQ(t->ncclCommAbort_fn, &ncclCommAbort_impl);
  EXPECT_EQ(t->ncclCommSplit_fn, &ncclCommSplit_impl);
  EXPECT_EQ(t->ncclGetErrorString_fn, &ncclGetErrorString_impl);
  EXPECT_EQ(t->ncclGetLastError_fn, &ncclGetLastError_impl);
  EXPECT_EQ(t->ncclCommGetAsyncError_fn, &ncclCommGetAsyncError_impl);
  EXPECT_EQ(t->ncclCommCount_fn, &ncclCommCount_impl);
  EXPECT_EQ(t->ncclCommCuDevice_fn, &ncclCommCuDevice_impl);
  EXPECT_EQ(t->ncclCommUserRank_fn, &ncclCommUserRank_impl);
  EXPECT_EQ(t->ncclMemAlloc_fn, &ncclMemAlloc_impl);
  EXPECT_EQ(t->ncclMemFree_fn, &ncclMemFree_impl);
  EXPECT_EQ(t->mscclLoadAlgo_fn, &mscclLoadAlgo_impl);
  EXPECT_EQ(t->mscclRunAlgo_fn, &mscclRunAlgo_impl);
  EXPECT_EQ(t->mscclUnloadAlgo_fn, &mscclUnloadAlgo_impl);
  EXPECT_EQ(t->ncclCommRegister_fn, &ncclCommRegister_impl);
  EXPECT_EQ(t->ncclCommDeregister_fn, &ncclCommDeregister_impl);
  EXPECT_EQ(t->ncclAllReduceWithBias_fn, &ncclAllReduceWithBias_impl);
  EXPECT_EQ(t->ncclCommShrink_fn, &ncclCommShrink_impl);
  EXPECT_EQ(t->ncclCommWindowRegister_fn, &ncclCommWindowRegister_impl);
  EXPECT_EQ(t->ncclCommWindowDeregister_fn, &ncclCommWindowDeregister_impl);
  EXPECT_EQ(t->ncclAlltoAll_fn, &ncclAlltoAll_impl);
  EXPECT_EQ(t->ncclAlltoAllv_fn, &ncclAlltoAllv_impl);
  EXPECT_EQ(t->ncclCommRevoke_fn, &ncclCommRevoke_impl);
  EXPECT_EQ(t->ncclCommSuspend_fn, &ncclCommSuspend_impl);
  EXPECT_EQ(t->ncclCommResume_fn, &ncclCommResume_impl);
  EXPECT_EQ(t->ncclCommMemStats_fn, &ncclCommMemStats_impl);
  EXPECT_EQ(t->ncclPutSignal_fn, &ncclPutSignal_impl);
  EXPECT_EQ(t->ncclSignal_fn, &ncclSignal_impl);
  EXPECT_EQ(t->ncclWaitSignal_fn, &ncclWaitSignal_impl);
  EXPECT_EQ(t->ncclCommGetUniqueId_fn, &ncclCommGetUniqueId_impl);
  EXPECT_EQ(t->ncclCommGrow_fn, &ncclCommGrow_impl);
}

TEST_F(ApiTraceMicrotest, Dispatch_Collectives) {
  const auto t = ncclFloat16;
  const auto op = ncclMax;
  const auto acc = H<const void*>(0xacc);
  EXPECT_FORWARDED(ncclAllGather(kSend, kRecv, 11, t, kComm, kStream), "AllGather", W(kSend), W(kRecv), 11, t,
                   W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclAllReduce(kSend, kRecv, 12, t, op, kComm, kStream), "AllReduce", W(kSend), W(kRecv), 12, t,
                   op, W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclAllReduceWithBias(kSend, kRecv, 13, t, op, kComm, kStream, acc), "AllReduceWithBias",
                   W(kSend), W(kRecv), 13, t, op, W(kComm), W(kStream), W(acc));
  EXPECT_FORWARDED(ncclAlltoAll(kSend, kRecv, 14, t, kComm, kStream), "AlltoAll", W(kSend), W(kRecv), 14, t,
                   W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclBroadcast(kSend, kRecv, 15, t, 2, kComm, kStream), "Broadcast", W(kSend), W(kRecv), 15, t, 2,
                   W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclGather(kSend, kRecv, 16, t, 3, kComm, kStream), "Gather", W(kSend), W(kRecv), 16, t, 3,
                   W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclReduce(kSend, kRecv, 17, t, op, 4, kComm, kStream), "Reduce", W(kSend), W(kRecv), 17, t, op,
                   4, W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclReduceScatter(kSend, kRecv, 18, t, op, kComm, kStream), "ReduceScatter", W(kSend), W(kRecv),
                   18, t, op, W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclScatter(kSend, kRecv, 19, t, 5, kComm, kStream), "Scatter", W(kSend), W(kRecv), 19, t, 5,
                   W(kComm), W(kStream));
  EXPECT_TRUE(g_warnings.empty());
}

TEST_F(ApiTraceMicrotest, Dispatch_AlltoAllv) {
  size_t sc[1], sd[1], rc[1], rd[1];
  EXPECT_FORWARDED(ncclAlltoAllv(kSend, sc, sd, kRecv, rc, rd, ncclInt64, kComm, kStream), "AlltoAllv", W(kSend),
                   W(sc), W(sd), W(kRecv), W(rc), W(rd), ncclInt64, W(kComm), W(kStream));
  EXPECT_TRUE(g_warnings.empty());
}

TEST_F(ApiTraceMicrotest, Dispatch_DeprecatedAllToAllWarnsAndForwards) {
  EXPECT_FORWARDED(ncclAllToAll(kSend, kRecv, 7, ncclInt8, kComm, kStream), "AlltoAll", W(kSend), W(kRecv), 7,
                   ncclInt8, W(kComm), W(kStream));
  ASSERT_EQ(g_warnings.size(), 1u);
  EXPECT_NE(g_warnings[0].find("ncclAllToAll is deprecated, please use ncclAlltoAll"), std::string::npos);

  size_t sc[1], sd[1], rc[1], rd[1];
  EXPECT_FORWARDED(ncclAllToAllv(kSend, sc, sd, kRecv, rc, rd, ncclInt8, kComm, kStream), "AlltoAllv", W(kSend),
                   W(sc), W(sd), W(kRecv), W(rc), W(rd), ncclInt8, W(kComm), W(kStream));
  ASSERT_EQ(g_warnings.size(), 2u);
  EXPECT_NE(g_warnings[1].find("ncclAllToAllv is deprecated, please use ncclAlltoAllv"), std::string::npos);
}

TEST_F(ApiTraceMicrotest, Dispatch_PointToPointAndSignals) {
  const auto win = H<ncclWindow_t>(0x3d);
  ncclWaitSignalDesc_t descs[2];
  EXPECT_FORWARDED(ncclSend(kSend, 21, ncclUint8, 6, kComm, kStream), "Send", W(kSend), 21, ncclUint8, 6, W(kComm),
                   W(kStream));
  EXPECT_FORWARDED(ncclRecv(kRecv, 22, ncclUint8, 7, kComm, kStream), "Recv", W(kRecv), 22, ncclUint8, 7, W(kComm),
                   W(kStream));
  EXPECT_FORWARDED(ncclPutSignal(kSend, 23, ncclInt32, 1, win, 64, 2, 3, 4u, kComm, kStream), "PutSignal", W(kSend),
                   23, ncclInt32, 1, W(win), 64, 2, 3, 4u, W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclSignal(5, 6, 7, 8u, kComm, kStream), "Signal", 5, 6, 7, 8u, W(kComm), W(kStream));
  EXPECT_FORWARDED(ncclWaitSignal(2, descs, kComm, kStream), "WaitSignal", 2, W(descs), W(kComm), W(kStream));
}

TEST_F(ApiTraceMicrotest, Dispatch_RedOpsGroupsAndVersion) {
  ncclRedOp_t op;
  float scalar;
  int version;
  EXPECT_FORWARDED(ncclRedOpCreatePreMulSum(&op, &scalar, ncclFloat32, ncclScalarHostImmediate, kComm),
                   "RedOpCreatePreMulSum", W(&op), W(&scalar), ncclFloat32, ncclScalarHostImmediate, W(kComm));
  EXPECT_FORWARDED(ncclRedOpDestroy(ncclAvg, kComm), "RedOpDestroy", ncclAvg, W(kComm));
  EXPECT_FORWARDED(ncclGroupStart(), "GroupStart");
  EXPECT_FORWARDED(ncclGroupEnd(), "GroupEnd");
  EXPECT_FORWARDED(ncclGetVersion(&version), "GetVersion", W(&version));
}

TEST_F(ApiTraceMicrotest, Dispatch_CommunicatorLifecycle) {
  ncclUniqueId id = {};
  const uintptr_t idWord = 0x1122334455667788ull;
  std::memcpy(id.internal, &idWord, sizeof(idWord));
  ncclComm_t comms[2];
  int devs[2];
  ncclConfig_t cfg;
  int excluded[1];
  EXPECT_FORWARDED(ncclGetUniqueId(&id), "GetUniqueId", W(&id));
  EXPECT_FORWARDED(ncclCommInitRank(comms, 8, id, 3), "CommInitRank", W(comms), 8, idWord, 3);
  EXPECT_FORWARDED(ncclCommInitAll(comms, 2, devs), "CommInitAll", W(comms), 2, W(devs));
  EXPECT_FORWARDED(ncclCommInitRankConfig(comms, 9, id, 4, &cfg), "CommInitRankConfig", W(comms), 9, idWord, 4,
                   W(&cfg));
  EXPECT_FORWARDED(ncclCommFinalize(kComm), "CommFinalize", W(kComm));
  EXPECT_FORWARDED(ncclCommDestroy(kComm), "CommDestroy", W(kComm));
  EXPECT_FORWARDED(ncclCommAbort(kComm), "CommAbort", W(kComm));
  EXPECT_FORWARDED(ncclCommRevoke(kComm, 1), "CommRevoke", W(kComm), 1);
  EXPECT_FORWARDED(ncclCommShrink(kComm, excluded, 1, comms, &cfg, 2), "CommShrink", W(kComm), W(excluded), 1,
                   W(comms), W(&cfg), 2);
  EXPECT_FORWARDED(ncclCommSplit(kComm, 5, 6, comms, &cfg), "CommSplit", W(kComm), 5, 6, W(comms), W(&cfg));
  EXPECT_FORWARDED(ncclCommSuspend(kComm, 3), "CommSuspend", W(kComm), 3);
  EXPECT_FORWARDED(ncclCommResume(kComm), "CommResume", W(kComm));
  EXPECT_FORWARDED(ncclCommGetUniqueId(kComm, &id), "CommGetUniqueId", W(kComm), W(&id));
  EXPECT_FORWARDED(ncclCommGrow(kComm, 10, &id, 7, comms, &cfg), "CommGrow", W(kComm), 10, W(&id), 7, W(comms),
                   W(&cfg));
}

TEST_F(ApiTraceMicrotest, Dispatch_CommunicatorQueries) {
  ncclResult_t asyncErr;
  int value;
  uint64_t stat;
  EXPECT_FORWARDED(ncclCommGetAsyncError(kComm, &asyncErr), "CommGetAsyncError", W(kComm), W(&asyncErr));
  EXPECT_FORWARDED(ncclCommCount(kComm, &value), "CommCount", W(kComm), W(&value));
  EXPECT_FORWARDED(ncclCommCuDevice(kComm, &value), "CommCuDevice", W(kComm), W(&value));
  EXPECT_FORWARDED(ncclCommUserRank(kComm, &value), "CommUserRank", W(kComm), W(&value));
  EXPECT_FORWARDED(ncclCommMemStats(kComm, (ncclCommMemStat_t)1, &stat), "CommMemStats", W(kComm), 1, W(&stat));
}

TEST_F(ApiTraceMicrotest, Dispatch_MemoryAndRegistration) {
  void* ptr;
  void* handle;
  ncclWindow_t win;
  const auto winIn = H<ncclWindow_t>(0x3e);
  EXPECT_FORWARDED(ncclMemAlloc(&ptr, 4096), "MemAlloc", W(&ptr), 4096);
  EXPECT_FORWARDED(ncclMemFree(kRecv), "MemFree", W(kRecv));
  EXPECT_FORWARDED(ncclCommRegister(kComm, kRecv, 512, &handle), "CommRegister", W(kComm), W(kRecv), 512, W(&handle));
  EXPECT_FORWARDED(ncclCommDeregister(kComm, kRecv), "CommDeregister", W(kComm), W(kRecv));
  EXPECT_FORWARDED(ncclCommWindowRegister(kComm, kRecv, 1024, &win, 1), "CommWindowRegister", W(kComm), W(kRecv),
                   1024, W(&win), 1);
  EXPECT_FORWARDED(ncclCommWindowDeregister(kComm, winIn), "CommWindowDeregister", W(kComm), W(winIn));
}

TEST_F(ApiTraceMicrotest, Dispatch_ErrorStringsReturnTheImplementationsText) {
  EXPECT_STREQ(ncclGetErrorString(ncclInvalidUsage), "error-string");
  ASSERT_EQ(g_calls.size(), 1u);
  EXPECT_EQ(g_calls[0].name, "GetErrorString");
  EXPECT_EQ(g_calls[0].args, std::vector<uintptr_t>{ncclInvalidUsage});
  g_calls.clear();
  EXPECT_STREQ(ncclGetLastError(kComm), "last-error");
  ASSERT_EQ(g_calls.size(), 1u);
  EXPECT_EQ(g_calls[0].name, "GetLastError");
  EXPECT_EQ(g_calls[0].args, std::vector<uintptr_t>{W(kComm)});
}


TEST_F(ApiTraceMicrotest, Msccl_LoadAndUnloadAreRecordedNoOps) {
  mscclAlgoHandle_t handle = 77;
  EXPECT_EQ(mscclLoadAlgo("/no/such/algo.xml", &handle, 0), ncclSuccess);
  EXPECT_EQ(mscclUnloadAlgo(handle), ncclSuccess);
  EXPECT_EQ(handle, 77);
  EXPECT_EQ(g_recorded, (std::vector<std::string>{"mscclLoadAlgo", "mscclUnloadAlgo"}));
  ASSERT_EQ(g_warnings.size(), 2u);
  EXPECT_NE(g_warnings[0].find("mscclLoadAlgo has no effect"), std::string::npos);
  EXPECT_NE(g_warnings[1].find("mscclUnloadAlgo has no effect"), std::string::npos);
  EXPECT_TRUE(g_calls.empty());
}

TEST_F(ApiTraceMicrotest, Msccl_RunIsARecordedNoOp) {
  size_t counts[2] = {1, 2};
  EXPECT_EQ(mscclRunAlgo(kSend, counts, counts, kRecv, counts, counts, 64, ncclFloat32, 0, 1, ncclSum, 3, kComm,
                         kStream),
            ncclSuccess);
  EXPECT_EQ(g_recorded, std::vector<std::string>{"mscclRunAlgo"});
  ASSERT_EQ(g_warnings.size(), 1u);
  EXPECT_NE(g_warnings[0].find("mscclRunAlgo has no effect"), std::string::npos);
  EXPECT_EQ(g_rangeNames, std::vector<std::string>{"RCCL_MSCCL"});
  EXPECT_EQ(g_rangeBytes, std::vector<size_t>{64 * sizeof(float)});
  EXPECT_EQ(g_rangesOpen, 0);
  EXPECT_TRUE(g_calls.empty());
}

}  // namespace
