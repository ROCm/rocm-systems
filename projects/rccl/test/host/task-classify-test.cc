/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; the UUT is #include'd. Hipify rewrites only the two cudaStream_t spellings.

#include <gtest/gtest.h>

#include <cstring>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <vector>

#include "ScopedHook.h"
#include "TaskPrepScene.h"

#include TASK_CLASSIFY_CC_PATH

namespace {

constexpr int kRoot = 2;
constexpr int kOffRootRank = 1;
constexpr int kAboveRootRank = 3;
constexpr int kPeer = 3;
constexpr ncclDataType_t kP2pDatatype = ncclFloat64;
constexpr size_t kP2pElemSize = 8;
constexpr size_t kP2pBytes = kCount * kP2pElemSize;
constexpr size_t kCollSliceBytes = kCount * sizeof(float);

const hipStream_t kStream = reinterpret_cast<hipStream_t>(0x5eedULL);

const void* OffsetBy(const void* base, size_t bytes) {
  return static_cast<const char*>(base) + bytes;
}

// Fails the Nth call and every one after it, so a caller that aborts part-way still runs the earlier ones.
auto TaskClassify_FailRegLocalIsValidFrom(int nth) {
  auto calls = std::make_shared<int>(0);
  return [calls, nth](struct ncclReg*, bool* out) -> ncclResult_t {
    if (++*calls >= nth) {
      return ncclInternalError;
    }
    *out = false;
    return ncclSuccess;
  };
}

// Self-constructing: a site that forgets ncclIntruQueueConstruct gets a garbage head, not an empty queue.
struct TaskClassify_Queue : TaskTuningInfoQueue {
  TaskClassify_Queue() { ncclIntruQueueConstruct(static_cast<TaskTuningInfoQueue*>(this)); }
};

struct TaskClassify_TuningInfoQueue : ncclTaskTuningInfoQueue {
  TaskClassify_TuningInfoQueue() { ncclIntruQueueConstruct(&queue); }
};

struct TaskClassify_P2p {
  ncclFunc_t func;
  const void* buff;
  int peer;
};

bool operator==(const TaskClassify_P2p& a, const TaskClassify_P2p& b) {
  return a.func == b.func && a.buff == b.buff && a.peer == b.peer;
}

std::ostream& operator<<(std::ostream& os, const TaskClassify_P2p& task) {
  return os << "{func=" << task.func << " buff=" << task.buff << " peer=" << task.peer << "}";
}

std::vector<TaskClassify_P2p> TaskClassify_P2pShape(TaskTuningInfoQueue* queue) {
  std::vector<TaskClassify_P2p> shape;
  for (struct ncclTaskTuningInfo* task : QueueTasks(queue)) {
    shape.push_back({task->raw->sendRecv.func, task->raw->sendRecv.buff, task->raw->sendRecv.peer});
  }
  return shape;
}

std::vector<struct ncclTaskTuningInfo*> TaskClassify_List(
  std::initializer_list<struct ncclTaskTuningInfo*> tasks) {
  return std::vector<struct ncclTaskTuningInfo*>(tasks);
}

// The shape triple says nothing about what each lowered task carries, which is what enqueue.cc branches on.
void TaskClassify_ExpectLoweredPayload(TaskTuningInfoQueue* queue, ncclFunc_t collAPI,
                                       ncclDataType_t datatype, size_t bytes) {
  for (struct ncclTaskTuningInfo* task : QueueTasks(queue)) {
    EXPECT_EQ(collAPI, task->raw->sendRecv.collAPI);
    EXPECT_EQ(datatype, task->raw->sendRecv.datatype);
    EXPECT_EQ(kCount, task->raw->sendRecv.count);
    EXPECT_EQ(bytes, task->raw->sendRecv.bytes);
    EXPECT_EQ(kStream, task->raw->sendRecv.stream);
  }
}

struct ncclRawTask* TaskClassify_NewP2pRaw(TaskPrepScene* scene) {
  struct ncclRawTask* raw = scene->NewSendRecv(ncclFuncRecv, kPeer);
  raw->sendRecv.collAPI = ncclFuncAlltoAll;
  raw->sendRecv.datatype = kP2pDatatype;
  raw->sendRecv.bytes = kP2pBytes;
  raw->sendRecv.stream = kStream;
  return raw;
}

struct ncclTaskTuningInfo* TaskClassify_NewCollInfo(TaskPrepScene* scene, ncclFunc_t func, int root,
                                                    ncclDataType_t datatype = ncclFloat32) {
  struct ncclRawTask* raw = scene->NewColl(func, datatype);
  raw->coll.root = root;
  raw->coll.stream = kStream;
  return scene->NewTuningInfo(raw);
}

void TaskClassify_SelectSymKernel(struct ncclTaskTuningInfo* tInfo, int symKernelId) {
  tInfo->tuningOut.valid = kTunedValid;
  tInfo->tuningOut.symKernelId = symKernelId;
}

void TaskClassify_SelectCeMethod(struct ncclTaskTuningInfo* tInfo, int ceMethodId) {
  tInfo->tuningOut.valid = kTunedValid;
  tInfo->tuningOut.ceMethodId = ceMethodId;
}

class TaskClassifyMicrotest : public TaskPrepFakesFixture {};

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_UnregisteredBuffer_CopiesTheRawFieldsAndClearsRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(scene.comm(), in.comm);
  EXPECT_EQ(NCCL_TUNING_MASK_ALL, in.tuningMask);
  EXPECT_EQ(ncclFuncRecv, in.func);
  EXPECT_EQ(kP2pDatatype, in.datatype);
  EXPECT_EQ(kCount, in.count);
  EXPECT_EQ(kCount, in.countMax);
  EXPECT_EQ(1, in.nWorks);
  EXPECT_EQ(1, in.numPipeOps);
  EXPECT_EQ(kP2pBytes, in.nBytes);
  EXPECT_EQ(0, in.regBuff);
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_RegisteredAndLocallyValid_SetsRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  RegisteredRanges registered(&scene, {{raw->sendRecv.buff, kP2pBytes}});
  struct ncclReg* probed = nullptr;
  ScopedHook isValid(g_regLocalIsValid, [&probed](struct ncclReg* reg, bool* out) {
    probed = reg;
    *out = true;
    return ncclSuccess;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(1, in.regBuff);
  EXPECT_EQ(1, isValid.calls);
  EXPECT_NE(nullptr, probed);
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_RegisteredButNotLocallyValid_ClearsRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  RegisteredRanges registered(&scene, {{raw->sendRecv.buff, kP2pBytes}});
  struct ncclReg* probed = nullptr;
  ScopedHook isValid(g_regLocalIsValid, [&probed](struct ncclReg* reg, bool* out) {
    probed = reg;
    *out = false;
    return ncclSuccess;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(0, in.regBuff);
  EXPECT_NE(nullptr, probed) << "the registration must still have been found";
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_RegistrationShorterThanTheBuffer_ClearsRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  RegisteredRanges registered(&scene, {{raw->sendRecv.buff, kP2pBytes - 1}});
  ScopedHook isValid(g_regLocalIsValid, [](struct ncclReg*, bool* out) {
    *out = true;
    return ncclSuccess;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(0, in.regBuff);
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_CapturingGraphWithGraphRegister_SetsRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(true);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, "GRAPH_REGISTER") == 0 ? 1 : deft;
  });
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(1, in.regBuff);
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_GraphArmHalfSatisfied_ClearsRegBuff) {
  for (bool capturing : {true, false}) {
    const int64_t graphRegister = capturing ? 0 : 1;
    TaskPrepScene scene;
    scene.SetGraphCapture(capturing);
    struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
    ScopedHook param(g_loadParam, [graphRegister](const char* env, int64_t deft) -> int64_t {
      return std::strcmp(env, "GRAPH_REGISTER") == 0 ? graphRegister : deft;
    });
    ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

    ASSERT_EQ(ncclSuccess, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

    EXPECT_EQ(0, in.regBuff) << "capturing = " << capturing;
  }
}

TEST_F(TaskClassifyMicrotest, FillSendRecvTuningInput_RegLocalIsValidFails_PropagatesBeforeWritingRegBuff) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclRawTask* raw = TaskClassify_NewP2pRaw(&scene);
  ScopedHook isValid(g_regLocalIsValid, TaskClassify_FailRegLocalIsValidFrom(1));
  ncclTuningInput_t in = TaskPrep_Poisoned<ncclTuningInput_t>();

  EXPECT_EQ(ncclInternalError, classifyFillSendRecvTuningInput(scene.comm(), &raw->sendRecv, &in));

  EXPECT_EQ(TaskPrep_Poisoned<int>(), in.regBuff);
  EXPECT_EQ(kP2pBytes, in.nBytes);
}

TEST_F(TaskClassifyMicrotest, EnqueueP2pTask_Called_BuildsTheRawTaskAndItsTuningInput) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  std::vector<double> buff(kCount);
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess,
            classifyEnqueueP2pTask(scene.comm(), &queue, ncclFuncRecv, ncclFuncAlltoAll, buff.data(),
                                   kCount, kP2pDatatype, kPeer, kStream));

  std::vector<struct ncclTaskTuningInfo*> tasks = QueueTasks(&queue);
  ASSERT_EQ(1u, tasks.size());
  struct ncclRawTask* raw = tasks[0]->raw;
  ASSERT_NE(nullptr, raw);
  EXPECT_EQ(ncclTaskKindSendRecv, raw->kind);
  EXPECT_EQ(ncclFuncRecv, raw->sendRecv.func);
  EXPECT_EQ(ncclFuncAlltoAll, raw->sendRecv.collAPI);
  EXPECT_EQ(buff.data(), raw->sendRecv.buff);
  EXPECT_EQ(kCount, raw->sendRecv.count);
  EXPECT_EQ(kP2pDatatype, raw->sendRecv.datatype);
  EXPECT_EQ(kPeer, raw->sendRecv.peer);
  EXPECT_EQ(kP2pBytes, raw->sendRecv.bytes);
  EXPECT_EQ(kStream, raw->sendRecv.stream);
  EXPECT_TRUE(CarriesNoTuningEstimate(tasks[0]->tuningOut));
  EXPECT_EQ(ncclSymkKernelId_Count, tasks[0]->tuningOut.symKernelId);
  EXPECT_EQ(ncclFuncRecv, tasks[0]->tuningIn.func);
  EXPECT_EQ(kP2pBytes, tasks[0]->tuningIn.nBytes);
}

TEST_F(TaskClassifyMicrotest, EnqueueP2pTask_TwoCalls_AppendInCallOrderWithDistinctRawTasks) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  std::vector<double> sendBuff(kCount);
  std::vector<double> recvBuff(kCount);
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess,
            classifyEnqueueP2pTask(scene.comm(), &queue, ncclFuncSend, ncclFuncGather, sendBuff.data(),
                                   kCount, kP2pDatatype, kRoot, kStream));
  ASSERT_EQ(ncclSuccess,
            classifyEnqueueP2pTask(scene.comm(), &queue, ncclFuncRecv, ncclFuncGather, recvBuff.data(),
                                   kCount, kP2pDatatype, kPeer, kStream));

  const std::vector<TaskClassify_P2p> expected = {{ncclFuncSend, sendBuff.data(), kRoot},
                                                  {ncclFuncRecv, recvBuff.data(), kPeer}};
  EXPECT_EQ(expected, TaskClassify_P2pShape(&queue));
  std::vector<struct ncclTaskTuningInfo*> tasks = QueueTasks(&queue);
  ASSERT_EQ(2u, tasks.size());
  EXPECT_NE(tasks[0]->raw, tasks[1]->raw);
}

TEST_F(TaskClassifyMicrotest, EnqueueP2pTask_TuningInputFails_PropagatesAndLeavesTheQueueEmpty) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  std::vector<double> buff(kCount);
  TaskClassify_Queue queue;
  ScopedHook isValid(g_regLocalIsValid, TaskClassify_FailRegLocalIsValidFrom(1));

  EXPECT_EQ(ncclInternalError,
            classifyEnqueueP2pTask(scene.comm(), &queue, ncclFuncSend, ncclFuncAlltoAll, buff.data(),
                                   kCount, kP2pDatatype, kPeer, kStream));

  EXPECT_TRUE(ncclIntruQueueEmpty(&queue));
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_AllToAll_EnqueuesAStridedSendRecvPairPerRank) {
  TaskPrepScene scene(kRanks, kOffRootRank);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncAlltoAll, kRoot);
  const void* sendBase = tInfo->raw->coll.sendbuff;
  const void* recvBase = tInfo->raw->coll.recvbuff;
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  std::vector<TaskClassify_P2p> expected;
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncSend, OffsetBy(sendBase, r * kCollSliceBytes), r});
    expected.push_back({ncclFuncRecv, OffsetBy(recvBase, r * kCollSliceBytes), r});
  }
  EXPECT_EQ(expected, TaskClassify_P2pShape(&queue));
  TaskClassify_ExpectLoweredPayload(&queue, ncclFuncAlltoAll, ncclFloat32, kCollSliceBytes);
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_AllToAllAtANarrowerDatatype_ScalesTheOffsetsAndBytes) {
  TaskPrepScene scene(kRanks, kOffRootRank);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo =
    TaskClassify_NewCollInfo(&scene, ncclFuncAlltoAll, kRoot, ncclInt8);
  const void* sendBase = tInfo->raw->coll.sendbuff;
  const void* recvBase = tInfo->raw->coll.recvbuff;
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  std::vector<TaskClassify_P2p> expected;
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncSend, OffsetBy(sendBase, r * kCount), r});
    expected.push_back({ncclFuncRecv, OffsetBy(recvBase, r * kCount), r});
  }
  EXPECT_EQ(expected, TaskClassify_P2pShape(&queue));
  TaskClassify_ExpectLoweredPayload(&queue, ncclFuncAlltoAll, ncclInt8, kCount);
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_GatherAtRoot_SendsToTheRootThenReceivesFromEveryRank) {
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncGather, kRoot);
  const void* sendBase = tInfo->raw->coll.sendbuff;
  const void* recvBase = tInfo->raw->coll.recvbuff;
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  std::vector<TaskClassify_P2p> expected = {{ncclFuncSend, sendBase, kRoot}};
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncRecv, OffsetBy(recvBase, r * kCollSliceBytes), r});
  }
  EXPECT_EQ(expected, TaskClassify_P2pShape(&queue));
  TaskClassify_ExpectLoweredPayload(&queue, ncclFuncGather, ncclFloat32, kCollSliceBytes);
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_GatherOffRoot_EnqueuesOnlyTheSendToTheRoot) {
  for (int rank : {kOffRootRank, kAboveRootRank}) {
    TaskPrepScene scene(kRanks, rank);
    scene.SetGraphCapture(false);
    struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncGather, kRoot);
    const void* sendBase = tInfo->raw->coll.sendbuff;
    TaskClassify_Queue queue;

    ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

    const std::vector<TaskClassify_P2p> expected = {{ncclFuncSend, sendBase, kRoot}};
    EXPECT_EQ(expected, TaskClassify_P2pShape(&queue)) << "rank = " << rank;
  }
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_ScatterAtRoot_SendsToEveryRankThenReceivesFromTheRoot) {
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncScatter, kRoot);
  const void* sendBase = tInfo->raw->coll.sendbuff;
  const void* recvBase = tInfo->raw->coll.recvbuff;
  TaskClassify_Queue queue;

  ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  std::vector<TaskClassify_P2p> expected;
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncSend, OffsetBy(sendBase, r * kCollSliceBytes), r});
  }
  expected.push_back({ncclFuncRecv, recvBase, kRoot});
  EXPECT_EQ(expected, TaskClassify_P2pShape(&queue));
  TaskClassify_ExpectLoweredPayload(&queue, ncclFuncScatter, ncclFloat32, kCollSliceBytes);
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_ScatterOffRoot_EnqueuesOnlyTheRecvFromTheRoot) {
  for (int rank : {kOffRootRank, kAboveRootRank}) {
    TaskPrepScene scene(kRanks, rank);
    scene.SetGraphCapture(false);
    struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncScatter, kRoot);
    const void* recvBase = tInfo->raw->coll.recvbuff;
    TaskClassify_Queue queue;

    ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

    const std::vector<TaskClassify_P2p> expected = {{ncclFuncRecv, recvBase, kRoot}};
    EXPECT_EQ(expected, TaskClassify_P2pShape(&queue)) << "rank = " << rank;
  }
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_Lowered_ReturnsTheCollRawTaskToThePoolAndClearsTheLink) {
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncGather, kRoot);
  struct ncclRawTask* raw = tInfo->raw;
  TaskClassify_Queue queue;
  ASSERT_EQ(nullptr, scene.comm()->memPool_ncclRawTask.head);

  ASSERT_EQ(ncclSuccess, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  EXPECT_EQ(nullptr, tInfo->raw);
  EXPECT_EQ(static_cast<void*>(raw), static_cast<void*>(scene.comm()->memPool_ncclRawTask.head));
}

// The ladder routes only AllToAll/Gather/Scatter here, so this arm is reachable by direct call alone.
TEST_F(TaskClassifyMicrotest, CollToP2pTasks_UnsupportedFunc_ReturnsInternalErrorAndKeepsTheRawTask) {
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncAllReduce, kRoot);
  struct ncclRawTask* raw = tInfo->raw;
  TaskClassify_Queue queue;

  EXPECT_EQ(ncclInternalError, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  EXPECT_EQ(raw, tInfo->raw);
  EXPECT_TRUE(ncclIntruQueueEmpty(&queue));
  EXPECT_EQ(nullptr, scene.comm()->memPool_ncclRawTask.head);
}

TEST_F(TaskClassifyMicrotest, CollToP2pTasks_EnqueueFails_PropagatesAndKeepsTheRawTask) {
  TaskPrepScene scene(kRanks, kOffRootRank);
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, ncclFuncAlltoAll, kRoot);
  struct ncclRawTask* raw = tInfo->raw;
  TaskClassify_Queue queue;
  ScopedHook isValid(g_regLocalIsValid, TaskClassify_FailRegLocalIsValidFrom(1));

  EXPECT_EQ(ncclInternalError, classifyCollToP2pTasks(scene.comm(), tInfo, &queue));

  EXPECT_EQ(raw, tInfo->raw);
  EXPECT_NE(static_cast<void*>(raw), static_cast<void*>(scene.comm()->memPool_ncclRawTask.head));
}

TEST_F(TaskClassifyMicrotest, TaskUsesSymKernel_ValidResultWithInRangeKernelId_IsTrue) {
  TaskPrepScene scene;
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  for (int symKernelId : {0, static_cast<int>(ncclSymkKernelId_Count) - 1}) {
    TaskClassify_SelectSymKernel(tInfo, symKernelId);
    EXPECT_TRUE(taskUsesSymKernel(tInfo)) << "symKernelId = " << symKernelId;
  }
}

TEST_F(TaskClassifyMicrotest, TaskUsesSymKernel_KernelIdOutOfRange_IsFalse) {
  TaskPrepScene scene;
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  for (int symKernelId : {-1, static_cast<int>(ncclSymkKernelId_Count)}) {
    TaskClassify_SelectSymKernel(tInfo, symKernelId);
    EXPECT_FALSE(taskUsesSymKernel(tInfo)) << "symKernelId = " << symKernelId;
  }
}

TEST_F(TaskClassifyMicrotest, TaskUsesSymKernel_TuningResultNotValid_IsFalse) {
  TaskPrepScene scene;
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  TaskClassify_SelectSymKernel(tInfo, 0);
  tInfo->tuningOut.valid = 0;
  EXPECT_FALSE(taskUsesSymKernel(tInfo));
}

// task_pretuning.cc:36-37 omits NCCL_TUNING_MASK_CE, so no production tuning result reaches this arm.
TEST_F(TaskClassifyMicrotest, TaskUsesCe_ValidResultNamingAnAllGatherCeMethod_IsTrue) {
  TaskPrepScene scene;
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  for (int ceMethodId : {static_cast<int>(ncclCeMethodId_AllGather_UC),
                         static_cast<int>(ncclCeMethodId_AllGather_MC)}) {
    TaskClassify_SelectCeMethod(tInfo, ceMethodId);
    EXPECT_TRUE(taskUsesCe(tInfo)) << "ceMethodId = " << ceMethodId;
  }
}

TEST_F(TaskClassifyMicrotest, TaskUsesCe_OtherCeMethodOrInvalidResult_IsFalse) {
  TaskPrepScene scene;
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  for (int ceMethodId : {-1, static_cast<int>(ncclCeMethodId_Count)}) {
    TaskClassify_SelectCeMethod(tInfo, ceMethodId);
    EXPECT_FALSE(taskUsesCe(tInfo)) << "ceMethodId = " << ceMethodId;
  }

  TaskClassify_SelectCeMethod(tInfo, ncclCeMethodId_AllGather_UC);
  tInfo->tuningOut.valid = 0;
  EXPECT_FALSE(taskUsesCe(tInfo));
}

TEST_F(TaskClassifyMicrotest, TaskUsesAllGatherV_MatchesOnlyTheAllGatherVRawTaskKind) {
  TaskPrepScene scene;
  EXPECT_TRUE(taskUsesAllGatherV(scene.NewTuningInfo(scene.NewAllGatherV())));
  EXPECT_FALSE(taskUsesAllGatherV(scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather))));
  EXPECT_FALSE(taskUsesAllGatherV(scene.NewTuningInfo(scene.NewSendRecv(ncclFuncSend, kPeer))));
  EXPECT_FALSE(taskUsesAllGatherV(scene.NewTuningInfo(scene.NewRma(ncclFuncPutSignal))));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_NullArgument_ReturnsInvalidArgumentBeforeTouchingTheQueues) {
  TaskPrepScene scene;
  TaskClassify_TuningInfoQueue tiq;
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  EXPECT_EQ(ncclInvalidArgument, ncclTaskClassification(nullptr, &tiq, &ctq));
  EXPECT_EQ(ncclInvalidArgument, ncclTaskClassification(scene.comm(), nullptr, &ctq));
  EXPECT_EQ(ncclInvalidArgument, ncclTaskClassification(scene.comm(), &tiq, nullptr));
  for (TaskTuningInfoQueue* queue : {&ctq.symTaskQueue, &ctq.legacyTaskQueue, &ctq.allgathervTaskQueue,
                                     &ctq.p2pTaskQueue, &ctq.rmaTaskQueue, &ctq.ceTaskQueue}) {
    EXPECT_EQ(TaskPrep_Poisoned<void*>(), static_cast<void*>(queue->head));
  }
}

TEST_F(TaskClassifyMicrotest, TaskClassification_PrepopulatedQueues_AreReconstructedEmpty) {
  TaskPrepScene scene;
  TaskClassify_TuningInfoQueue tiq;
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  for (TaskTuningInfoQueue* queue : {&ctq.symTaskQueue, &ctq.legacyTaskQueue, &ctq.allgathervTaskQueue,
                                     &ctq.p2pTaskQueue, &ctq.rmaTaskQueue, &ctq.ceTaskQueue}) {
    EXPECT_TRUE(ncclIntruQueueEmpty(queue));
  }
}

TEST_F(TaskClassifyMicrotest, TaskClassification_NullRawTask_ReturnsInternalError) {
  TaskPrepScene scene;
  TaskClassify_TuningInfoQueue tiq;
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();
  ncclIntruQueueEnqueue(&tiq.queue, scene.NewTuningInfo(nullptr));

  EXPECT_EQ(ncclInternalError, ncclTaskClassification(scene.comm(), &tiq, &ctq));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_EachTaskKind_LandsInItsOwnQueue) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* p2p = scene.NewTuningInfo(scene.NewSendRecv(ncclFuncSend, kPeer));
  struct ncclTaskTuningInfo* rma = scene.NewTuningInfo(scene.NewRma(ncclFuncPutSignal));
  struct ncclTaskTuningInfo* sym = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  TaskClassify_SelectSymKernel(sym, ncclSymkKernelId_AllGather_LL);
  struct ncclTaskTuningInfo* agv = scene.NewTuningInfo(scene.NewAllGatherV());
  struct ncclTaskTuningInfo* legacy = scene.NewTuningInfo(scene.NewColl(ncclFuncAllReduce));
  TaskClassify_TuningInfoQueue tiq;
  for (struct ncclTaskTuningInfo* task : {p2p, rma, sym, agv, legacy}) {
    ncclIntruQueueEnqueue(&tiq.queue, task);
  }
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  EXPECT_EQ(TaskClassify_List({p2p}), QueueTasks(&ctq.p2pTaskQueue));
  EXPECT_EQ(TaskClassify_List({rma}), QueueTasks(&ctq.rmaTaskQueue));
  EXPECT_EQ(TaskClassify_List({sym}), QueueTasks(&ctq.symTaskQueue));
  EXPECT_EQ(TaskClassify_List({agv}), QueueTasks(&ctq.allgathervTaskQueue));
  EXPECT_EQ(TaskClassify_List({legacy}), QueueTasks(&ctq.legacyTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.ceTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&tiq.queue));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_TaskKind_OutranksTheTuningResultArms) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* p2p = scene.NewTuningInfo(scene.NewSendRecv(ncclFuncSend, kPeer));
  struct ncclTaskTuningInfo* rma = scene.NewTuningInfo(scene.NewRma(ncclFuncPutSignal));
  for (struct ncclTaskTuningInfo* task : {p2p, rma}) {
    TaskClassify_SelectSymKernel(task, ncclSymkKernelId_AllGather_LL);
    TaskClassify_SelectCeMethod(task, ncclCeMethodId_AllGather_MC);
  }
  TaskClassify_TuningInfoQueue tiq;
  ncclIntruQueueEnqueue(&tiq.queue, p2p);
  ncclIntruQueueEnqueue(&tiq.queue, rma);
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  EXPECT_EQ(TaskClassify_List({p2p}), QueueTasks(&ctq.p2pTaskQueue));
  EXPECT_EQ(TaskClassify_List({rma}), QueueTasks(&ctq.rmaTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.symTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.ceTaskQueue));
}

// Only a synthesized tuning result reaches the CE arm; the production tuningMask never selects one.
TEST_F(TaskClassifyMicrotest, TaskClassification_CeMethodResult_OutranksTheSymKernelArm) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(scene.NewColl(ncclFuncAllGather));
  TaskClassify_SelectSymKernel(tInfo, ncclSymkKernelId_AllGather_LL);
  TaskClassify_SelectCeMethod(tInfo, ncclCeMethodId_AllGather_MC);
  TaskClassify_TuningInfoQueue tiq;
  ncclIntruQueueEnqueue(&tiq.queue, tInfo);
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  EXPECT_EQ(TaskClassify_List({tInfo}), QueueTasks(&ctq.ceTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.symTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.legacyTaskQueue));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_AllToAllScatterGatherColls_AreLoweredIntoThePeerQueue) {
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  TaskClassify_TuningInfoQueue tiq;
  const void* sendBase = nullptr;
  const void* recvBase = nullptr;
  for (ncclFunc_t func : {ncclFuncAlltoAll, ncclFuncScatter, ncclFuncGather}) {
    struct ncclTaskTuningInfo* tInfo = TaskClassify_NewCollInfo(&scene, func, kRoot);
    sendBase = tInfo->raw->coll.sendbuff;
    recvBase = tInfo->raw->coll.recvbuff;
    ncclIntruQueueEnqueue(&tiq.queue, tInfo);
  }
  struct ncclTaskTuningInfo* legacy = TaskClassify_NewCollInfo(&scene, ncclFuncAllReduce, kRoot);
  ncclIntruQueueEnqueue(&tiq.queue, legacy);
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  std::vector<TaskClassify_P2p> expected;
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncSend, OffsetBy(sendBase, r * kCollSliceBytes), r});
    expected.push_back({ncclFuncRecv, OffsetBy(recvBase, r * kCollSliceBytes), r});
  }
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncSend, OffsetBy(sendBase, r * kCollSliceBytes), r});
  }
  expected.push_back({ncclFuncRecv, recvBase, kRoot});
  expected.push_back({ncclFuncSend, sendBase, kRoot});
  for (int r = 0; r < kRanks; r++) {
    expected.push_back({ncclFuncRecv, OffsetBy(recvBase, r * kCollSliceBytes), r});
  }
  EXPECT_EQ(expected, TaskClassify_P2pShape(&ctq.p2pTaskQueue));
  for (struct ncclTaskTuningInfo* task : QueueTasks(&ctq.p2pTaskQueue)) {
    EXPECT_EQ(ncclTaskKindSendRecv, task->raw->kind);
  }
  EXPECT_EQ(TaskClassify_List({legacy}), QueueTasks(&ctq.legacyTaskQueue));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_LoweringFails_PropagatesAndStrandsTheEarlierTasks) {
  constexpr int kFailingProbe = 3;
  TaskPrepScene scene(kRanks, kRoot);
  scene.SetGraphCapture(false);
  TaskClassify_TuningInfoQueue tiq;
  ncclIntruQueueEnqueue(&tiq.queue, TaskClassify_NewCollInfo(&scene, ncclFuncAlltoAll, kRoot));
  ncclIntruQueueEnqueue(&tiq.queue, TaskClassify_NewCollInfo(&scene, ncclFuncAllReduce, kRoot));
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();
  ScopedHook isValid(g_regLocalIsValid, TaskClassify_FailRegLocalIsValidFrom(kFailingProbe));

  EXPECT_EQ(ncclInternalError, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  // Lowering has no rollback, so whatever was enqueued before the failing probe stays on the peer queue.
  EXPECT_EQ(static_cast<size_t>(kFailingProbe - 1), QueueTasks(&ctq.p2pTaskQueue).size());
  EXPECT_TRUE(ncclIntruQueueEmpty(&ctq.legacyTaskQueue));
}

TEST_F(TaskClassifyMicrotest, TaskClassification_SeveralTasksPerQueue_PreserveTheirEnqueueOrder) {
  TaskPrepScene scene;
  scene.SetGraphCapture(false);
  struct ncclTaskTuningInfo* firstLegacy = scene.NewTuningInfo(scene.NewColl(ncclFuncAllReduce));
  struct ncclTaskTuningInfo* firstRma = scene.NewTuningInfo(scene.NewRma(ncclFuncPutSignal));
  struct ncclTaskTuningInfo* secondLegacy = scene.NewTuningInfo(scene.NewColl(ncclFuncBroadcast));
  struct ncclTaskTuningInfo* secondRma = scene.NewTuningInfo(scene.NewRma(ncclFuncSignal));
  struct ncclTaskTuningInfo* thirdLegacy = scene.NewTuningInfo(scene.NewColl(ncclFuncReduce));
  TaskClassify_TuningInfoQueue tiq;
  for (struct ncclTaskTuningInfo* task :
       {firstLegacy, firstRma, secondLegacy, secondRma, thirdLegacy}) {
    ncclIntruQueueEnqueue(&tiq.queue, task);
  }
  struct ncclClassifiedTaskQueues ctq = TaskPrep_Poisoned<ncclClassifiedTaskQueues>();

  ASSERT_EQ(ncclSuccess, ncclTaskClassification(scene.comm(), &tiq, &ctq));

  EXPECT_EQ(TaskClassify_List({firstLegacy, secondLegacy, thirdLegacy}),
            QueueTasks(&ctq.legacyTaskQueue));
  EXPECT_EQ(TaskClassify_List({firstRma, secondRma}), QueueTasks(&ctq.rmaTaskQueue));
}

}  // namespace
