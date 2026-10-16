/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/enqueue/mgmt_task_enq.cc: ncclMgmtTaskEnqueue's task setup, group join and
// blocking-mode bookkeeping. The group thread-locals group.cc owns are defined in this TU.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "group.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local ncclComm* ncclGroupCommPreconnectHead = nullptr;
thread_local int ncclGroupBlocking = -1;

namespace {
std::vector<std::string> g_warnings;
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (level == NCCL_LOG_WARN) g_warnings.push_back(buf);
}

// Reached only when a collective or raw-task join pushes a memory scope, never for a management task.
void* ncclMemoryStack::allocateSpilled(ncclMemoryStack*, size_t, size_t) {
  ADD_FAILURE() << "management tasks must not push a memory scope";
  return nullptr;
}

#include MGMT_TASK_ENQ_CC_PATH

namespace {

std::vector<void*> g_destroyed;
ncclResult_t TaskFunc(ncclAsyncJob*) { return ncclSuccess; }
void Destroy(void* task) { g_destroyed.push_back(task); }

class MgmtTaskEnqMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_warnings.clear();
    g_destroyed.clear();
    for (ncclComm*& head : ncclGroupCommHead) head = nullptr;
    ncclGroupBlocking = -1;
  }
  void TearDown() override {
    for (ncclComm*& head : ncclGroupCommHead) head = nullptr;
    ncclGroupBlocking = -1;
  }

  std::unique_ptr<ncclComm> MakeComm(uint64_t commHash, int blocking) {
    auto comm = std::make_unique<ncclComm>();
    for (ncclComm*& next : comm->groupNext) next = reinterpret_cast<ncclComm*>(NCCL_COMM_GROUP_INVALID);
    comm->intraComm0 = comm.get();
    comm->commHash = commHash;
    comm->config.blocking = blocking;
    comm->abortFlag = &abort_;
    comm->abortFlagDev = &abortDev_;
    comm->childAbortFlag = &childAbort_;
    comm->childAbortFlagDev = &childAbortDev_;
    return comm;
  }

  uint32_t abort_ = 0;
  uint32_t abortDev_ = 0;
  uint32_t childAbort_ = 0;
  uint32_t childAbortDev_ = 0;
};

TEST_F(MgmtTaskEnqMicrotest, Enqueue_ArmsTheTaskFromItsComm) {
  auto comm = MakeComm(5, 0);
  comm->destroyFlag = 0;
  ncclAsyncJob task{};
  task.state = ncclGroupJobDone;
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&task, TaskFunc, Destroy, comm.get()));
  EXPECT_EQ(TaskFunc, task.func);
  EXPECT_EQ(Destroy, task.destructor);
  EXPECT_EQ(&abort_, task.abortFlag);
  EXPECT_EQ(&abortDev_, task.abortFlagDev);
  EXPECT_EQ(&childAbort_, task.childAbortFlag);
  EXPECT_EQ(&childAbortDev_, task.childAbortFlagDev);
  EXPECT_EQ(ncclGroupJobRunning, task.state);
  EXPECT_EQ(comm.get(), task.comm);
  EXPECT_EQ(0, task.destroyFlag);
  EXPECT_EQ(&task, ncclIntruQueueHead(&comm->mgmtTaskQueue));
  EXPECT_EQ(comm.get(), ncclGroupCommHead[ncclGroupTaskTypeMgmtTask]);
  EXPECT_EQ(nullptr, ncclGroupCommHead[ncclGroupTaskTypeCollective]);
  EXPECT_EQ(0, ncclGroupBlocking);
  EXPECT_TRUE(g_destroyed.empty());
}

TEST_F(MgmtTaskEnqMicrotest, Enqueue_DestroyingCommForcesABlockingGroup) {
  auto comm = MakeComm(5, 0);
  comm->destroyFlag = 1;
  ncclAsyncJob task{};
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&task, TaskFunc, nullptr, comm.get()));
  EXPECT_EQ(1, task.destroyFlag);
  EXPECT_EQ(1, ncclGroupBlocking);
}

TEST_F(MgmtTaskEnqMicrotest, Enqueue_QueuesEveryTaskOfAJoinedCommInOrder) {
  auto comm = MakeComm(5, 1);
  ncclAsyncJob first{};
  ncclAsyncJob second{};
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&first, TaskFunc, Destroy, comm.get()));
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&second, TaskFunc, Destroy, comm.get()));
  EXPECT_EQ(&first, ncclIntruQueueHead(&comm->mgmtTaskQueue));
  EXPECT_EQ(&second, first.next);
  EXPECT_EQ(comm.get(), ncclGroupCommHead[ncclGroupTaskTypeMgmtTask]);
  EXPECT_EQ(nullptr, comm->groupNext[ncclGroupTaskTypeMgmtTask]);  // joined once
  EXPECT_EQ(1, ncclGroupBlocking);
}

TEST_F(MgmtTaskEnqMicrotest, Enqueue_JoinsCommsInCommHashOrder) {
  auto high = MakeComm(9, 1);
  auto low = MakeComm(3, 1);
  ncclAsyncJob a{};
  ncclAsyncJob b{};
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&a, TaskFunc, Destroy, high.get()));
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&b, TaskFunc, Destroy, low.get()));
  EXPECT_EQ(low.get(), ncclGroupCommHead[ncclGroupTaskTypeMgmtTask]);
  EXPECT_EQ(high.get(), low->groupNext[ncclGroupTaskTypeMgmtTask]);
}

// ncclGroupCommJoin already stored this comm's mode, so the mixed-mode check never fires; a fix flips this pin.
TEST_F(MgmtTaskEnqMicrotest, Enqueue_KnownBug_MixedBlockingModesAreNotRejected) {
  auto blocking = MakeComm(3, 1);
  auto nonblocking = MakeComm(9, 0);
  ncclAsyncJob a{};
  ncclAsyncJob b{};
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&a, TaskFunc, Destroy, blocking.get()));
  EXPECT_EQ(ncclSuccess, ncclMgmtTaskEnqueue(&b, TaskFunc, Destroy, nonblocking.get()));
  EXPECT_EQ(0, ncclGroupBlocking);
  EXPECT_EQ(&b, ncclIntruQueueHead(&nonblocking->mgmtTaskQueue));
  EXPECT_TRUE(g_destroyed.empty());
  EXPECT_TRUE(g_warnings.empty());
}

}  // namespace
