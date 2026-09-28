/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/enqueue/task_sched/: a dispatch layer with zero production
// callers today. ncclTaskSchedule's real per-queue routing is commented out (dead code, not
// gated by any macro), and every one of the six ncclScheduleXTasks wrappers is an 18-line file
// whose entire body is `return doLaunches(comm);`. These tests pin exactly that forwarding
// contract and ncclTaskSchedule's four argument guards, not doLaunches itself, which is the
// same kernel-launch machinery already out of scope for a host-only binary.
// Hipify leaves all seven task_sched files byte-identical.

#include <gtest/gtest.h>

#include "ScopedHook.h"
#include "TaskPrepScene.h"

#include TASK_SCHED_DISPATCH_CC_PATH
#include TASK_SCHED_SYM_CC_PATH
#include TASK_SCHED_LEGACY_CC_PATH
#include TASK_SCHED_ALLGATHERV_CC_PATH
#include TASK_SCHED_P2P_CC_PATH
#include TASK_SCHED_RMA_CC_PATH
#include TASK_SCHED_CE_CC_PATH

namespace {

// Non-null sentinels. None of these are ever dereferenced by the code under test: sharedRes and
// context are only null-checked (the lines that would read through them are the commented-out
// dead block in task_sched.cc), and queue/depStream/launchOrderStream are unused parameters in
// every one of the six wrappers.
constexpr uintptr_t kSharedResAddr = 0x1;
constexpr uintptr_t kContextAddr = 0x2;
constexpr uintptr_t kQueueAddr = 0x10;
constexpr uintptr_t kDepStreamAddr = 0x20;
constexpr uintptr_t kLaunchOrderStreamAddr = 0x30;

using ScheduleFn = ncclResult_t (*)(struct ncclComm*, TaskTuningInfoQueue*, struct ncclStrongStream*,
                                    struct ncclStrongStream*);

struct TaskSchedWrapperCase {
  const char* name;
  ScheduleFn fn;
};

const TaskSchedWrapperCase kWrapperCases[] = {
    {"ncclScheduleSymTasks", ncclScheduleSymTasks},
    {"ncclScheduleLegacyTasks", ncclScheduleLegacyTasks},
    {"ncclScheduleAllGatherVTasks", ncclScheduleAllGatherVTasks},
    {"ncclScheduleP2pTasks", ncclScheduleP2pTasks},
    {"ncclScheduleRmaTasks", ncclScheduleRmaTasks},
    {"ncclScheduleCeTasks", ncclScheduleCeTasks},
};

// A comm past ncclTaskSchedule's two guards (sharedRes/context both non-null), plus its own
// ncclClassifiedTaskQueues. The two guard tests that deliberately null one field reach in and
// clear it back out after construction.
class TaskSched_ValidScene {
 public:
  TaskSched_ValidScene() {
    scene_.comm()->sharedRes = reinterpret_cast<struct ncclSharedResources*>(TaskPrep_Addr(kSharedResAddr));
    scene_.comm()->context = reinterpret_cast<struct ncclCudaContext*>(TaskPrep_Addr(kContextAddr));
  }

  struct ncclComm* comm() { return scene_.comm(); }
  struct ncclClassifiedTaskQueues* ctq() { return &ctq_; }

 private:
  TaskPrepScene scene_;
  struct ncclClassifiedTaskQueues ctq_{};
};

class TaskSchedMicrotest : public TaskPrepFakesFixture {};

TEST_F(TaskSchedMicrotest, WrapperFunctions_Called_ForwardToDoLaunchesWithCollectiveTaskTypeAndIgnoreOtherArgs) {
  TaskPrepScene scene;
  TaskTuningInfoQueue* const queue = reinterpret_cast<TaskTuningInfoQueue*>(TaskPrep_Addr(kQueueAddr));
  struct ncclStrongStream* const depStream =
    reinterpret_cast<struct ncclStrongStream*>(TaskPrep_Addr(kDepStreamAddr));
  struct ncclStrongStream* const launchOrderStream =
    reinterpret_cast<struct ncclStrongStream*>(TaskPrep_Addr(kLaunchOrderStreamAddr));
  for (const TaskSchedWrapperCase& c : kWrapperCases) {
    SCOPED_TRACE(c.name);
    struct ncclComm* seenComm = nullptr;
    int seenTaskType = -1;
    ScopedHook hook(g_doLaunches, [&](struct ncclComm* comm, int taskType) {
      seenComm = comm;
      seenTaskType = taskType;
      return ncclSuccess;
    });

    ncclResult_t rc = c.fn(scene.comm(), queue, depStream, launchOrderStream);

    EXPECT_EQ(ncclSuccess, rc);
    EXPECT_EQ(1, hook.calls);
    EXPECT_EQ(scene.comm(), seenComm);
    EXPECT_EQ(ncclGroupTaskTypeCollective, seenTaskType);
  }
}

TEST_F(TaskSchedMicrotest, WrapperFunctions_DoLaunchesFails_PropagatesTheError) {
  TaskPrepScene scene;
  for (const TaskSchedWrapperCase& c : kWrapperCases) {
    SCOPED_TRACE(c.name);
    ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclUnhandledCudaError; });

    ncclResult_t rc = c.fn(scene.comm(), nullptr, nullptr, nullptr);

    EXPECT_EQ(ncclUnhandledCudaError, rc);
    EXPECT_EQ(1, hook.calls);
  }
}

TEST_F(TaskSchedMicrotest, WrapperFunctions_DoLaunchesReturnsInProgress_PropagatesItVerbatim) {
  TaskPrepScene scene;
  for (const TaskSchedWrapperCase& c : kWrapperCases) {
    SCOPED_TRACE(c.name);
    ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclInProgress; });

    ncclResult_t rc = c.fn(scene.comm(), nullptr, nullptr, nullptr);

    EXPECT_EQ(ncclInProgress, rc);
    EXPECT_EQ(1, hook.calls);
  }
}

TEST_F(TaskSchedMicrotest, TaskSchedule_NullComm_ReturnsInvalidArgumentWithoutCallingDoLaunches) {
  struct ncclClassifiedTaskQueues ctq{};
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclSuccess; });

  EXPECT_EQ(ncclInvalidArgument, ncclTaskSchedule(nullptr, &ctq));
  EXPECT_EQ(0, hook.calls);
}

TEST_F(TaskSchedMicrotest, TaskSchedule_NullQueues_ReturnsInvalidArgumentWithoutCallingDoLaunches) {
  TaskPrepScene scene;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclSuccess; });

  EXPECT_EQ(ncclInvalidArgument, ncclTaskSchedule(scene.comm(), nullptr));
  EXPECT_EQ(0, hook.calls);
}

TEST_F(TaskSchedMicrotest, TaskSchedule_NullSharedRes_ReturnsInternalErrorWithoutCallingDoLaunches) {
  TaskSched_ValidScene scene;
  scene.comm()->sharedRes = nullptr;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclSuccess; });

  EXPECT_EQ(ncclInternalError, ncclTaskSchedule(scene.comm(), scene.ctq()));
  EXPECT_EQ(0, hook.calls);
}

TEST_F(TaskSchedMicrotest, TaskSchedule_NullContext_ReturnsInternalErrorWithoutCallingDoLaunches) {
  TaskSched_ValidScene scene;
  scene.comm()->context = nullptr;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclSuccess; });

  EXPECT_EQ(ncclInternalError, ncclTaskSchedule(scene.comm(), scene.ctq()));
  EXPECT_EQ(0, hook.calls);
}

TEST_F(TaskSchedMicrotest, TaskSchedule_ValidArguments_CallsDoLaunchesOnceWithCollectiveTaskTypeAndPropagatesSuccess) {
  TaskSched_ValidScene scene;
  struct ncclComm* seenComm = nullptr;
  int seenTaskType = -1;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm* comm, int taskType) {
    seenComm = comm;
    seenTaskType = taskType;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, ncclTaskSchedule(scene.comm(), scene.ctq()));
  EXPECT_EQ(1, hook.calls);
  EXPECT_EQ(scene.comm(), seenComm);
  EXPECT_EQ(ncclGroupTaskTypeCollective, seenTaskType);
}

TEST_F(TaskSchedMicrotest, TaskSchedule_ValidArguments_DoLaunchesFails_PropagatesTheError) {
  TaskSched_ValidScene scene;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclRemoteError; });

  EXPECT_EQ(ncclRemoteError, ncclTaskSchedule(scene.comm(), scene.ctq()));
  EXPECT_EQ(1, hook.calls);
}

// doLaunches returning ncclInProgress falls through NCCLCHECK (which only early-returns on a
// result that is neither ncclSuccess nor ncclInProgress) and hits the literal `return ncclSuccess`
// on the next line, so the in-progress status is silently dropped here. The six wrappers above,
// whose bare `return doLaunches(comm);` propagates ncclInProgress verbatim, do not have this gap.
TEST_F(TaskSchedMicrotest, TaskSchedule_ValidArguments_DoLaunchesReturnsInProgress_CurrentlyDropsItAndReturnsSuccess) {
  TaskSched_ValidScene scene;
  ScopedHook hook(g_doLaunches, [&](struct ncclComm*, int) { return ncclInProgress; });

  EXPECT_EQ(ncclSuccess, ncclTaskSchedule(scene.comm(), scene.ctq()));
  EXPECT_EQ(1, hook.calls);
}

}  // namespace
