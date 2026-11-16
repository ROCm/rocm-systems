/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/argcheck.cc: argument and pointer checks, and the global
// symmetric-registration check. Peers are simulated through the bootstrapAllGather seam.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "argcheck.h"
#include "bootstrap.h"
#include "comm.h"
#include "debug.h"
#include "dev_runtime.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

struct ArgcheckLog {
  ncclDebugLogLevel level;
  std::string text;
};
static std::vector<ArgcheckLog> g_argcheckLog;
void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_argcheckLog.push_back({level, buf});
}

// Mirrors registrationCheck's per-buffer record: two per rank, send then recv.
struct SymBufInfo {
  bool isSymRegistered;
  uintptr_t bigOffset;
  uintptr_t userOffset;
};

static hipError_t g_attrResult = hipSuccess;
static hipPointerAttribute_t g_attr{};
static std::vector<const void*> g_attrPtrs;
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attr, const void* ptr) {
  g_attrPtrs.push_back(ptr);
  *attr = g_attr;
  return g_attrResult;
}

static std::map<const void*, ncclDevrWindow*> g_windows;
static int g_findWindowCalls = 0;
static int g_findWindowFailAt = 0;
ncclResult_t ncclDevrFindWindow(struct ncclComm*, void const* userPtr, struct ncclDevrWindow** outWin) {
  if (++g_findWindowCalls == g_findWindowFailAt) return ncclInternalError;
  auto it = g_windows.find(userPtr);
  *outWin = it == g_windows.end() ? nullptr : it->second;
  return ncclSuccess;
}

static std::vector<SymBufInfo> g_peerInfo;  // what every other rank contributes
static std::vector<SymBufInfo> g_gathered;  // this rank's two records as the gather saw them
static int g_gatherCalls = 0;
static int g_gatherSize = 0;
static ncclResult_t g_gatherResult = ncclSuccess;
static int g_myRank = 0;
ncclResult_t bootstrapAllGather(void*, void* allData, int size) {
  g_gatherCalls++;
  g_gatherSize = size;
  SymBufInfo* all = static_cast<SymBufInfo*>(allData);
  g_gathered.assign(all + 2 * g_myRank, all + 2 * g_myRank + 2);
  for (int i = 0; i < (int)g_peerInfo.size(); i++) {
    if (i / 2 != g_myRank) all[i] = g_peerInfo[i];
  }
  return g_gatherResult;
}

static std::set<void*> g_argcheckLive;
static int g_callocCalls = 0;
static int g_callocFailAt = 0;
template <typename T>
static ncclResult_t ArgcheckCalloc(T** ptr, size_t nelem) {
  if (++g_callocCalls == g_callocFailAt) return ncclSystemError;
  *ptr = static_cast<T*>(std::calloc(nelem, sizeof(T)));
  g_argcheckLive.insert(*ptr);
  return ncclSuccess;
}
static void ArgcheckFree(void* p) {
  g_argcheckLive.erase(p);
  std::free(p);
}

#undef ncclCalloc
#define ncclCalloc(...) ArgcheckCalloc(__VA_ARGS__)
#define free(p) ArgcheckFree(p)

#include ARGCHECK_CC_PATH

#undef free
#undef ncclCalloc

namespace {

constexpr int kRanks = 4;
constexpr int kDev = 1;

class ArgcheckMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_argcheckLog.clear();
    g_attrResult = hipSuccess;
    g_attr = hipPointerAttribute_t{};
    g_attr.type = hipMemoryTypeDevice;
    g_attr.device = kDev;
    g_attr.devicePointer = &devMem_;
    g_attrPtrs.clear();
    g_windows.clear();
    g_findWindowCalls = 0;
    g_findWindowFailAt = 0;
    g_gathered.clear();
    g_gatherCalls = 0;
    g_gatherResult = ncclSuccess;
    g_myRank = 0;
    g_argcheckLive.clear();
    g_callocCalls = 0;
    g_callocFailAt = 0;

    comm_ = std::make_unique<ncclComm>();
    comm_->startMagic = comm_->endMagic = NCCL_MAGIC;
    comm_->nRanks = kRanks;
    comm_->cudaDev = kDev;
    comm_->userRedOps = userRedOps_;
    ncclIntruQueueConstruct(&comm_->argsInfoQueue);
    info_.coll = ncclFuncAllReduce;
    info_.opName = "AllReduce";
    info_.sendbuff = &send_[16];
    info_.recvbuff = &recv_[32];
    info_.count = 8;
    info_.datatype = ncclFloat32;
    info_.op = ncclSum;
    info_.comm = comm_.get();
    SetPeers(Sym(0x1000, 16), Sym(0x2000, 32));
  }
  void TearDown() override {
    while (ncclArgsInfo* a = ncclIntruQueueTryDequeue(&comm_->argsInfoQueue)) std::free(a);
  }

  static SymBufInfo Sym(uintptr_t bigOffset, uintptr_t userOffset) { return {true, bigOffset, userOffset}; }
  void SetRank(int rank) {
    comm_->rank = rank;
    g_myRank = rank;
  }
  // Every rank registers send and recv symmetrically with these records.
  void SetPeers(SymBufInfo send, SymBufInfo recv) { g_peerInfo.assign({send, recv, send, recv, send, recv, send, recv}); }
  void RegisterOwn(int sendFlags = NCCL_WIN_COLL_SYMMETRIC, int recvFlags = NCCL_WIN_COLL_SYMMETRIC) {
    sendWin_.userPtr = send_;
    sendWin_.bigOffset = 0x1000;
    sendWin_.winFlags = sendFlags;
    recvWin_.userPtr = recv_;
    recvWin_.bigOffset = 0x2000;
    recvWin_.winFlags = recvFlags;
    g_windows = {{info_.sendbuff, &sendWin_}, {info_.recvbuff, &recvWin_}};
  }
  ncclResult_t GlobalCheck() {
    args_.info = info_;
    return ncclArgsGlobalCheck(&args_);
  }
  bool Logged(ncclDebugLogLevel level, const char* needle) {
    for (const auto& m : g_argcheckLog) {
      if (m.level == level && m.text.find(needle) != std::string::npos) return true;
    }
    return false;
  }
  // The op a user would pass for userRedOps_[index] on this comm.
  ncclRedOp_t UserOp(int index) {
    int k = ncclNumOps;
    while (int(ncclUserRedOpMangle(comm_.get(), ncclRedOp_t(k))) == k) k++;
    int hash = int(ncclUserRedOpMangle(comm_.get(), ncclRedOp_t(k))) ^ k;
    ncclRedOp_t op = ncclRedOp_t(hash ^ (int(ncclNumOps) + index));
    EXPECT_EQ(int(ncclNumOps) + index, int(ncclUserRedOpMangle(comm_.get(), op)));
    return op;
  }

  std::unique_ptr<ncclComm> comm_;
  ncclUserRedOp userRedOps_[2] = {};
  ncclInfo info_{};
  ncclArgsInfo args_{};
  char send_[64] = {};
  char recv_[64] = {};
  int devMem_ = 0;
  ncclDevrWindow sendWin_{};
  ncclDevrWindow recvWin_{};
};

// Pointer and comm checks

TEST_F(ArgcheckMicrotest, PtrCheck_RejectsNull) {
  EXPECT_EQ(ncclInvalidArgument, PtrCheck(nullptr, "ncclAllReduce", "comm"));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "ncclAllReduce : comm argument is NULL"));
  EXPECT_EQ(ncclSuccess, PtrCheck(this, "ncclAllReduce", "comm"));
}

TEST_F(ArgcheckMicrotest, CommCheck_RejectsNullAndCorruptedComms) {
  EXPECT_EQ(ncclSuccess, CommCheck(comm_.get(), "op", "comm"));
  EXPECT_EQ(ncclInvalidArgument, CommCheck(nullptr, "op", "comm"));
  comm_->startMagic = 0;
  EXPECT_EQ(ncclInvalidArgument, CommCheck(comm_.get(), "op", "comm"));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "corrupted comm object detected"));
  comm_->startMagic = NCCL_MAGIC;
  comm_->endMagic = 0;
  EXPECT_EQ(ncclInvalidArgument, CommCheck(comm_.get(), "op", "comm"));
}

TEST_F(ArgcheckMicrotest, CudaPtrCheck_AcceptsMemoryOnTheCommsDevice) {
  EXPECT_EQ(ncclSuccess, CudaPtrCheck(send_, comm_.get(), "sendbuff", "AllReduce"));
  EXPECT_EQ((std::vector<const void*>{send_}), g_attrPtrs);
  g_attr.type = hipMemoryTypeHost;
  g_attr.device = kDev + 1;  // host memory may report any device
  EXPECT_EQ(ncclSuccess, CudaPtrCheck(send_, comm_.get(), "sendbuff", "AllReduce"));
}

TEST_F(ArgcheckMicrotest, CudaPtrCheck_RejectsInvalidPointers) {
  g_attrResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclInvalidArgument, CudaPtrCheck(send_, comm_.get(), "sendbuff", "AllReduce"));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "AllReduce : sendbuff"));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "is not a valid pointer"));
  g_attrResult = hipSuccess;
  g_attr.devicePointer = nullptr;
  EXPECT_EQ(ncclInvalidArgument, CudaPtrCheck(send_, comm_.get(), "sendbuff", "AllReduce"));
}

TEST_F(ArgcheckMicrotest, CudaPtrCheck_RejectsDeviceMemoryOfAnotherDevice) {
  g_attr.device = kDev + 1;
  EXPECT_EQ(ncclInvalidArgument, CudaPtrCheck(recv_, comm_.get(), "recvbuff", "AllReduce"));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "AllReduce : recvbuff allocated on device 2 mismatchs with NCCL device 1"));
}

// ArgsCheck

TEST_F(ArgcheckMicrotest, ArgsCheck_RejectsOutOfRangeRootTypeAndOp) {
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  for (int root : {-1, kRanks}) {
    info_.root = root;
    EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_)) << root;
  }
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "AllReduce : invalid root 4 (root should be in the 0..4 range)"));
  info_.root = kRanks - 1;
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  for (int type : {-1, int(ncclNumTypes)}) {
    info_.datatype = ncclDataType_t(type);
    EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_)) << type;
  }
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "invalid type"));
  info_.datatype = ncclDataType_t(ncclNumTypes - 1);
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  info_.op = ncclRedOp_t(-1);
  EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "invalid reduction operation -1"));
  info_.op = ncclRedOp_t(ncclNumOps - 1);
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
}

TEST_F(ArgcheckMicrotest, ArgsCheck_UserOpMustBeAllocatedOnThisComm) {
  comm_->userRedOpCapacity = 2;
  userRedOps_[0].freeNext = 1;  // free
  userRedOps_[1].freeNext = -1;  // allocated
  info_.op = UserOp(1);
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  info_.op = UserOp(0);
  EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "unknown to this communicator"));
  comm_->userRedOpCapacity = 1;
  info_.op = UserOp(1);
  EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_));
}

TEST_F(ArgcheckMicrotest, ArgsCheck_DefaultModeSkipsPointerChecks) {
  g_attrResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  EXPECT_TRUE(g_attrPtrs.empty());
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->argsInfoQueue));
}

TEST_F(ArgcheckMicrotest, ArgsCheck_LocalModeChecksBothBuffers) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
  EXPECT_EQ((std::vector<const void*>{info_.sendbuff, info_.recvbuff}), g_attrPtrs);
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->argsInfoQueue));

  g_attrResult = hipErrorInvalidValue;
  EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "AllReduce : sendbuff"));
}

TEST_F(ArgcheckMicrotest, ArgsCheck_RootedCollsCheckOnlyTheBuffersTheRankUses) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  info_.root = 2;
  struct Case {
    ncclFunc_t coll;
    int rank;
    std::vector<const void*> checked;
  };
  const std::vector<Case> cases = {
      {ncclFuncBroadcast, 2, {info_.sendbuff, info_.recvbuff}},
      {ncclFuncBroadcast, 0, {info_.recvbuff}},
      {ncclFuncReduce, 2, {info_.sendbuff, info_.recvbuff}},
      {ncclFuncReduce, 0, {info_.sendbuff}},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(std::to_string(c.coll) + " rank " + std::to_string(c.rank));
    g_attrPtrs.clear();
    info_.coll = c.coll;
    comm_->rank = c.rank;
    EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
    EXPECT_EQ(c.checked, g_attrPtrs);
  }
}

TEST_F(ArgcheckMicrotest, ArgsCheck_SendRecvCheckOnlyANonEmptyRecvbuff) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  for (ncclFunc_t coll : {ncclFuncSend, ncclFuncRecv}) {
    g_attrPtrs.clear();
    info_.coll = coll;
    info_.count = 0;
    EXPECT_EQ(ncclSuccess, ArgsCheck(&info_));
    EXPECT_TRUE(g_attrPtrs.empty());
    info_.count = 1;
    g_attrResult = hipErrorInvalidValue;
    EXPECT_EQ(ncclInvalidArgument, ArgsCheck(&info_));
    EXPECT_EQ((std::vector<const void*>{info_.recvbuff}), g_attrPtrs);
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, "AllReduce : buff"));
    g_attrResult = hipSuccess;
  }
}

TEST_F(ArgcheckMicrotest, ArgsCheck_OneSidedOpsSkipBufferChecks) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  g_attrResult = hipErrorInvalidValue;
  for (ncclFunc_t coll : {ncclFuncPutSignal, ncclFuncSignal, ncclFuncWaitSignal}) {
    info_.coll = coll;
    EXPECT_EQ(ncclSuccess, ArgsCheck(&info_)) << coll;
  }
  EXPECT_TRUE(g_attrPtrs.empty());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "skipping sendbuff/recvbuff pointer check"));
}

TEST_F(ArgcheckMicrotest, ArgsCheck_GlobalModeQueuesACopyForTheGroupCheck) {
  comm_->checkMode = ncclCheckModeDebugGlobal;
  ASSERT_EQ(ncclSuccess, ArgsCheck(&info_));
  info_.count = 99;
  ASSERT_EQ(ncclSuccess, ArgsCheck(&info_));
  EXPECT_EQ(4u, g_attrPtrs.size());
  ncclArgsInfo* first = ncclIntruQueueTryDequeue(&comm_->argsInfoQueue);
  ncclArgsInfo* second = ncclIntruQueueTryDequeue(&comm_->argsInfoQueue);
  ASSERT_NE(nullptr, first);
  ASSERT_NE(nullptr, second);
  EXPECT_EQ(8u, first->info.count);
  EXPECT_EQ(info_.sendbuff, first->info.sendbuff);
  EXPECT_EQ(99u, second->info.count);
  EXPECT_EQ(nullptr, second->next);
  std::free(first);
  std::free(second);
}

TEST_F(ArgcheckMicrotest, ArgsCheck_GlobalModeAllocationFailurePropagates) {
  comm_->checkMode = ncclCheckModeDebugGlobal;
  g_callocFailAt = 1;
  EXPECT_EQ(ncclSystemError, ArgsCheck(&info_));
  EXPECT_TRUE(ncclIntruQueueEmpty(&comm_->argsInfoQueue));
}

// Global symmetric-registration check

TEST_F(ArgcheckMicrotest, GlobalCheck_SkipsP2pAndOneSidedOps) {
  for (ncclFunc_t coll : {ncclFuncSend, ncclFuncRecv, ncclFuncPutSignal, ncclFuncSignal, ncclFuncWaitSignal}) {
    info_.coll = coll;
    EXPECT_EQ(ncclSuccess, GlobalCheck()) << coll;
  }
  EXPECT_EQ(0, g_gatherCalls);
}

TEST_F(ArgcheckMicrotest, GlobalCheck_PublishesEachBuffersWindowAndUserOffset) {
  RegisterOwn();
  SetRank(2);
  ASSERT_EQ(ncclSuccess, GlobalCheck());
  EXPECT_EQ(1, g_gatherCalls);
  EXPECT_EQ(int(2 * sizeof(SymBufInfo)), g_gatherSize);
  ASSERT_EQ(2u, g_gathered.size());
  EXPECT_TRUE(g_gathered[0].isSymRegistered);
  EXPECT_EQ(0x1000u, g_gathered[0].bigOffset);
  EXPECT_EQ(16u, g_gathered[0].userOffset);
  EXPECT_TRUE(g_gathered[1].isSymRegistered);
  EXPECT_EQ(0x2000u, g_gathered[1].bigOffset);
  EXPECT_EQ(32u, g_gathered[1].userOffset);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "SymCheck: coll AllReduce size 32 rank 2 bigOffset 1000 userOffset 10"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "SymCheck: coll AllReduce size 32 rank 2 bigOffset 2000 userOffset 20"));
  EXPECT_TRUE(g_argcheckLive.empty());
}

// Only NCCL_WIN_COLL_SYMMETRIC windows count; anything else publishes an unregistered buffer.
TEST_F(ArgcheckMicrotest, GlobalCheck_NonSymmetricWindowsAreUnregistered) {
  RegisterOwn(0, 0);
  g_peerInfo.assign(2 * kRanks, SymBufInfo{false, 0, 0});
  ASSERT_EQ(ncclSuccess, GlobalCheck());
  EXPECT_FALSE(g_gathered[0].isSymRegistered);
  EXPECT_FALSE(g_gathered[1].isSymRegistered);
  EXPECT_EQ(0u, g_gathered[0].bigOffset);
  g_windows.clear();
  ASSERT_EQ(ncclSuccess, GlobalCheck());
  EXPECT_FALSE(g_gathered[0].isSymRegistered);
}

TEST_F(ArgcheckMicrotest, GlobalCheck_RegistrationMustMatchOnEveryRank) {
  RegisterOwn();
  for (int buf : {0, 1}) {
    SCOPED_TRACE(buf);
    SetPeers(Sym(0x1000, 16), Sym(0x2000, 32));
    g_peerInfo[2 * 3 + buf].isSymRegistered = false;
    g_argcheckLog.clear();
    EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
    EXPECT_TRUE(Logged(NCCL_LOG_WARN, buf == 0 ? "check failed on rank 3: sendReg 0 recvReg 1 mismatch with rank 0 "
                                                 "sendReg 1 recvReg 1"
                                               : "check failed on rank 3: sendReg 1 recvReg 0"));
  }
  EXPECT_TRUE(g_argcheckLive.empty());
}

// Every rank fails the check; only rank 0 says why.
TEST_F(ArgcheckMicrotest, GlobalCheck_OnlyRankZeroWarns) {
  RegisterOwn();
  SetRank(1);
  SetPeers(Sym(0x1000, 16), Sym(0x2000, 32));
  g_peerInfo[2 * 3].isSymRegistered = false;
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  SetPeers(Sym(0x1001, 16), Sym(0x2000, 32));
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  SetPeers(Sym(0x1000, 17), Sym(0x2000, 32));
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  SetPeers(Sym(0x1000, 16), Sym(0x2001, 32));
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  SetPeers(Sym(0x1000, 16), Sym(0x2000, 33));
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  for (const auto& m : g_argcheckLog) EXPECT_NE(NCCL_LOG_WARN, m.level) << m.text;
}

// Send buffers must line up for colls that read them symmetrically, recv buffers for colls that
// write them; the last mismatching rank is the one reported.
TEST_F(ArgcheckMicrotest, GlobalCheck_OffsetsMustMatchWhereTheCollUsesThem) {
  struct Case {
    ncclFunc_t coll;
    bool checksSend;
    bool checksRecv;
  };
  const std::vector<Case> cases = {
      {ncclFuncAllReduce, true, true},   {ncclFuncReduceScatter, true, false}, {ncclFuncAlltoAll, true, true},
      {ncclFuncGather, true, false},     {ncclFuncAllGather, false, true},     {ncclFuncScatter, false, true},
      {ncclFuncBroadcast, false, false}, {ncclFuncReduce, false, false},
  };
  struct Mismatch {
    int index;
    bool big;
    const char* warn;
  };
  const std::vector<Mismatch> mismatches = {
      {0, true, "rank 3: send buffer window (0x1001) mismatch with rank 0 (0x1000)"},
      {0, false, "rank 3: send buffer user offset (0x11) mismatch with rank 0 (0x10)"},
      {1, true, "rank 3: recv buffer window (0x2001) mismatch with rank 0 (0x2000)"},
      {1, false, "rank 3: recv buffer user offset (0x21) mismatch with rank 0 (0x20)"},
  };
  RegisterOwn();
  for (const auto& c : cases) {
    for (const auto& m : mismatches) {
      SCOPED_TRACE(std::to_string(c.coll) + " " + m.warn);
      SetPeers(Sym(0x1000, 16), Sym(0x2000, 32));
      for (int r : {2, 3}) {
        SymBufInfo& peer = g_peerInfo[2 * r + m.index];
        (m.big ? peer.bigOffset : peer.userOffset) += r - 2;  // rank 2 matches, rank 3 is off by one
      }
      g_argcheckLog.clear();
      info_.coll = c.coll;
      bool checked = m.index == 0 ? c.checksSend : c.checksRecv;
      EXPECT_EQ(checked ? ncclInvalidArgument : ncclSuccess, GlobalCheck());
      EXPECT_EQ(checked, Logged(NCCL_LOG_WARN, m.warn));
    }
  }
  EXPECT_TRUE(g_argcheckLive.empty());
}

TEST_F(ArgcheckMicrotest, GlobalCheck_ReportsTheLastMismatchingRank) {
  RegisterOwn();
  g_peerInfo[2 * 1].bigOffset = 0x1001;
  g_peerInfo[2 * 2].bigOffset = 0x1002;
  g_peerInfo[2 * 1 + 1].userOffset = 0x21;
  g_peerInfo[2 * 3 + 1].userOffset = 0x23;
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "rank 2: send buffer window (0x1002)"));
  info_.coll = ncclFuncAllGather;
  EXPECT_EQ(ncclInvalidArgument, GlobalCheck());
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "rank 3: recv buffer user offset (0x23)"));
}

// Mismatched offsets do not matter for a buffer nobody registered.
TEST_F(ArgcheckMicrotest, GlobalCheck_UnregisteredBuffersAreNotCompared) {
  g_peerInfo.assign(2 * kRanks, SymBufInfo{false, 0, 0});
  g_peerInfo[2 * 3].userOffset = 5;
  g_peerInfo[2 * 3 + 1].bigOffset = 5;
  EXPECT_EQ(ncclSuccess, GlobalCheck());
}

TEST_F(ArgcheckMicrotest, GlobalCheck_FailuresPropagateAndFreeTheGatherBuffer) {
  RegisterOwn();
  g_callocFailAt = 1;
  EXPECT_EQ(ncclSystemError, GlobalCheck());
  EXPECT_EQ(0, g_findWindowCalls);
  for (int failAt : {1, 2}) {
    g_findWindowCalls = 0;
    g_findWindowFailAt = failAt;
    EXPECT_EQ(ncclInternalError, GlobalCheck()) << failAt;
  }
  g_findWindowFailAt = 0;
  g_gatherResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, GlobalCheck());
  EXPECT_EQ(1, g_gatherCalls);
  EXPECT_TRUE(g_argcheckLive.empty());
}

}  // namespace
