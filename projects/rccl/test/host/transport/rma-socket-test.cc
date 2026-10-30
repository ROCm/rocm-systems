/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/rma_socket.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <vector>

#include "ScopedHook.h"
#include "fakes/gdrwrap_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/os_fakes.h"
#include "fakes/socket_fakes.h"
#include "fakes/transport_net_socket_common_fakes.h"
#include "fakes/transport_net_socket_fakes.h"
#include "transport/RmaSocketTest.h"

#include RMA_SOCKET_CC_PATH

namespace {

using rma_socket_test::Scene;

class RmaSocketMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    ncclRmaSocketProxyRefCount = 0;
    ncclGdrCopy = nullptr;
    ResetTransportNetSocketCommonFakes();
    ResetTransportNetSocketFakes();
    ResetSocketFakes();
    ResetOsFakes();
    ResetGdrwrapFakes();
    ResetHipFakes();
    ResetNcclFakes();
  }
};

// A live request in slot 0 of context 0. Every receiver is closed so the
// progress pass inside ncclRmaSocketProxyTest reads no socket.
ncclRmaSocketProxyRequest& ArmRequest(Scene& scene, ncclRmaSocketProxyRequestType type, int peer, uint64_t id) {
  for (auto& receiver : scene.receivers) receiver.recvState = ncclRmaSocketProxyRecvStateClosed;
  auto& req = scene.requestStorage[0][0];
  req.type = type;
  req.ctx = &scene.contexts[0];
  req.peer = peer;
  req.globalRequestId = id;
  scene.contexts[0].activeOps = 1;
  return req;
}

TEST_F(RmaSocketMicrotest, Init_AmdPlatformReportsUnsupportedAndClearsContext) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ScopedHook initDevices(g_ncclNetSocketInitDevices, [](const char*) { return ncclSuccess; });
  void* ctx = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyInit(&ctx, 7, nullptr), ncclInternalError);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(ncclRmaSocketProxyRefCount, 0);
  EXPECT_EQ(initDevices.calls, 0);
}

TEST_F(RmaSocketMicrotest, Init_ExistingInitializationAddsReferenceWithoutReinitializingDevices) {
  ncclRmaSocketProxyRefCount = 1;
  void* ctx = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyInit(&ctx, 7, nullptr), ncclSuccess);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(ncclRmaSocketProxyRefCount, 2);
}

TEST_F(RmaSocketMicrotest, Finalize_ReferencesAreReleasedAndZeroCountIsRejected) {
  ncclRmaSocketProxyRefCount = 2;
  EXPECT_EQ(ncclRmaSocketProxyFinalize(nullptr), ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyRefCount, 1);
  EXPECT_EQ(ncclRmaSocketProxyFinalize(nullptr), ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyRefCount, 0);
  EXPECT_EQ(ncclRmaSocketProxyFinalize(nullptr), ncclInternalError);
}

TEST_F(RmaSocketMicrotest, Devices_ReturnsDiscoveredSocketDeviceCount) {
  ncclNetIfs = 3;
  int devices = -1;

  EXPECT_EQ(ncclRmaSocketProxyDevices(&devices), ncclSuccess);
  EXPECT_EQ(devices, 3);
}

TEST_F(RmaSocketMicrotest, GetRmaProperties_ReportsSignalsDoNotFlushAllPuts) {
  ncclRmaProperties_t props{};
  props.flushesAllPutsOnAnySignal = true;

  EXPECT_EQ(ncclRmaSocketProxyGetRmaProperties(nullptr, &props), ncclSuccess);
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(RmaSocketMicrotest, GetProperties_DelegatesThenMarksGinProxyDevice) {
  ScopedHook getProperties(g_ncclNetSocketGetProperties, [](int dev, ncclNetProperties_t* props) {
    EXPECT_EQ(dev, 2);
    std::memset(props, 0, sizeof(*props));
    props->speed = 123;
    return ncclSuccess;
  });
  ncclNetProperties_t props{};

  EXPECT_EQ(ncclRmaSocketProxyGetProperties(2, &props), ncclSuccess);
  EXPECT_EQ(props.speed, 123);
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_PROXY);
  EXPECT_EQ(getProperties.calls, 1);
}

TEST_F(RmaSocketMicrotest, GetProperties_DelegateFailureIsPropagated) {
  ScopedHook getProperties(g_ncclNetSocketGetProperties,
                           [](int, ncclNetProperties_t*) { return ncclSystemError; });
  ncclNetProperties_t props{};

  EXPECT_EQ(ncclRmaSocketProxyGetProperties(0, &props), ncclSystemError);
}

TEST_F(RmaSocketMicrotest, Listen_CreatesListenerAndPublishesOpaqueHandle) {
  ScopedHook create(g_ncclNetSocketCreateListener,
                    [](const char* prefix, int dev, ncclSocket*, ncclSocketAddress*, uint64_t* magic) {
                      EXPECT_STREQ(prefix, "RMA/Socket");
                      EXPECT_EQ(dev, 1);
                      *magic = 0xabc;
                      return ncclSuccess;
                    });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  ncclRmaSocketProxyHandle handle{};
  void* listen = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyListen(nullptr, 1, &handle, &listen), ncclSuccess);
  EXPECT_NE(listen, nullptr);
  EXPECT_EQ(handle.magic, 0xabcu);
  EXPECT_EQ(create.calls, 1);
  EXPECT_EQ(ncclRmaSocketProxyCloseListen(listen), ncclSuccess);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(RmaSocketMicrotest, Listen_CreateFailureClosesAllocatedSocketAndLeavesOutputUntouched) {
  ScopedHook create(g_ncclNetSocketCreateListener,
                    [](const char*, int, ncclSocket*, ncclSocketAddress*, uint64_t*) {
                      return ncclSystemError;
                    });
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclSuccess; });
  ncclRmaSocketProxyHandle handle{};
  void* listen = nullptr;

  EXPECT_EQ(ncclRmaSocketProxyListen(nullptr, 0, &handle, &listen), ncclSystemError);
  EXPECT_EQ(listen, nullptr);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(RmaSocketMicrotest, CloseListen_NullIsNoOpAndCloseFailureIsReturned) {
  EXPECT_EQ(ncclRmaSocketProxyCloseListen(nullptr), ncclSuccess);

  auto* sock = static_cast<ncclSocket*>(std::calloc(1, sizeof(ncclSocket)));
  ASSERT_NE(sock, nullptr);
  ScopedHook close(g_ncclSocketClose, [](ncclSocket*, bool) { return ncclRemoteError; });
  EXPECT_EQ(ncclRmaSocketProxyCloseListen(sock), ncclRemoteError);
  EXPECT_EQ(close.calls, 1);
}

TEST_F(RmaSocketMicrotest, RecordError_StoresOnlyTerminalFailures) {
  ncclRmaSocketProxyCollComm comm{};
  comm.lastError = ncclSuccess;

  EXPECT_EQ(ncclRmaSocketProxyRecordError(&comm, ncclSuccess), ncclSuccess);
  EXPECT_EQ(comm.lastError, ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyRecordError(&comm, ncclInProgress), ncclInProgress);
  EXPECT_EQ(comm.lastError, ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyRecordError(&comm, ncclRemoteError), ncclRemoteError);
  EXPECT_EQ(comm.lastError, ncclRemoteError);
  EXPECT_EQ(ncclRmaSocketProxyRecordError(nullptr, ncclSystemError), ncclSystemError);
}

TEST_F(RmaSocketMicrotest, ActiveOpAccounting_ReleaseReturnsSlotToPoolAndNeverLeavesNegativeCount) {
  ncclRmaSocketProxyCtx ctx{};
  ncclRmaSocketProxyRequest req{};
  req.type = ncclRmaSocketProxyRequestTypePut;
  req.ctx = &ctx;
  ctx.activeOps = 1;

  ncclRmaSocketProxyReleaseRequest(&req);
  EXPECT_EQ(ctx.activeOps, 0);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(req.ctx, nullptr);

  req.type = ncclRmaSocketProxyRequestTypeGet;
  req.ctx = &ctx;
  ncclRmaSocketProxyReleaseRequest(&req);
  EXPECT_EQ(ctx.activeOps, 0);
  ncclRmaSocketProxyReleaseRequest(nullptr);
}

TEST_F(RmaSocketMicrotest, GetRequest_InvalidArgumentsAreRejected) {
  Scene scene;
  ncclRmaSocketProxyRequest* request = reinterpret_cast<ncclRmaSocketProxyRequest*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyGetRequest(nullptr, 0, ncclRmaSocketProxyRequestTypePut, &request),
            ncclInvalidArgument);
  scene.contexts[0].collComm = nullptr;
  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], 0, ncclRmaSocketProxyRequestTypePut, &request),
            ncclInvalidArgument);
  scene.contexts[0].collComm = &scene.comm;
  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], -1, ncclRmaSocketProxyRequestTypePut, &request),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], scene.comm.nranks,
                                         ncclRmaSocketProxyRequestTypePut, &request),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], 0, ncclRmaSocketProxyRequestTypeFree, &request),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], 0, ncclRmaSocketProxyRequestTypePut, nullptr),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, GetRequest_FreeSlotGetsIdentityAndAdvancesRollingId) {
  Scene scene;
  scene.comm.nextGlobalRequestId = std::numeric_limits<uint64_t>::max();
  ncclRmaSocketProxyRequest* request = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], 1, ncclRmaSocketProxyRequestTypeGet, &request),
            ncclSuccess);
  ASSERT_NE(request, nullptr);
  EXPECT_EQ(request, &scene.requestStorage[0][0]);
  EXPECT_EQ(request->type, ncclRmaSocketProxyRequestTypeGet);
  EXPECT_EQ(request->ctx, &scene.contexts[0]);
  EXPECT_EQ(request->peer, 1);
  EXPECT_EQ(request->globalRequestId, std::numeric_limits<uint64_t>::max());
  EXPECT_EQ(scene.comm.nextGlobalRequestId, 0u);
  EXPECT_EQ(scene.contexts[0].activeOps, 1);
  ncclRmaSocketProxyReleaseRequest(request);
}

TEST_F(RmaSocketMicrotest, GetRequest_FullPoolReturnsErrorAndNullOutput) {
  Scene scene;
  for (auto& request : scene.requestStorage[0]) request.type = ncclRmaSocketProxyRequestTypePut;
  ncclRmaSocketProxyRequest* request = reinterpret_cast<ncclRmaSocketProxyRequest*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyGetRequest(&scene.contexts[0], 0, ncclRmaSocketProxyRequestTypeGet, &request),
            ncclInternalError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
}

TEST_F(RmaSocketMicrotest, BuildHeader_ValidPutPopulatesEveryWireField) {
  ncclRmaSocketProxyHeader header{};

  ASSERT_EQ(ncclRmaSocketProxyBuildHeader(&header, ncclRmaSocketProxyMsgTypeIput, ncclSuccess, 4,
                                          NCCL_RMA_SOCKET_OP_HAS_SIGNAL, 5, 6, 7, 8, 9, 10,
                                          NCCL_NET_SIGNAL_OP_ADD),
            ncclSuccess);
  EXPECT_EQ(header.type, ncclRmaSocketProxyMsgTypeIput);
  EXPECT_EQ(header.status, static_cast<uint32_t>(ncclSuccess));
  EXPECT_EQ(header.requestId, 4u);
  EXPECT_EQ(header.opFlags, NCCL_RMA_SOCKET_OP_HAS_SIGNAL);
  EXPECT_EQ(header.mrId, 5u);
  EXPECT_EQ(header.dstOff, 6u);
  EXPECT_EQ(header.size, 7u);
  EXPECT_EQ(header.signalMrId, 8u);
  EXPECT_EQ(header.signalOff, 9u);
  EXPECT_EQ(header.signalValue, 10u);
  EXPECT_EQ(header.signalOp, NCCL_NET_SIGNAL_OP_ADD);
}

TEST_F(RmaSocketMicrotest, BuildHeader_OversizeOrDisallowedFlagsAreRejected) {
  ncclRmaSocketProxyHeader header{};
  EXPECT_EQ(ncclRmaSocketProxyBuildHeader(&header, ncclRmaSocketProxyMsgTypeIput, 0, 0, 0, 0, 0,
                                          static_cast<size_t>(INT_MAX) + 1, 0, 0, 0, 0),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyBuildHeader(&header, ncclRmaSocketProxyMsgTypeRespondGetData, 0, 0,
                                          NCCL_RMA_SOCKET_OP_HAS_SIGNAL, 0, 0, 1, 0, 0, 0, 0),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyBuildHeader(&header, ncclRmaSocketProxyMsgTypeIput, 0, 0, 0x80, 0, 0, 1, 0, 0, 0, 0),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, AcquireSendTask_ValidQueueReservesTailAndWraps) {
  Scene scene(/*nranks=*/1, /*nContexts=*/1, /*queueCapacity=*/2);
  auto& sender = scene.senders[0];
  sender.tail = 1;
  ncclRmaSocketProxySendTask* task = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyAcquireSendTask(&sender, &task), ncclSuccess);
  EXPECT_EQ(task, &sender.tasks[1]);
  EXPECT_EQ(sender.tail, 0);
  EXPECT_EQ(sender.count, 1);
}

TEST_F(RmaSocketMicrotest, AcquireSendTask_InvalidOrFullQueueIsRejected) {
  ncclRmaSocketProxySendTask* task = reinterpret_cast<ncclRmaSocketProxySendTask*>(0x1);
  EXPECT_EQ(ncclRmaSocketProxyAcquireSendTask(nullptr, &task), ncclInvalidArgument);

  ncclRmaSocketProxyPeerSender sender{};
  EXPECT_EQ(ncclRmaSocketProxyAcquireSendTask(&sender, &task), ncclInternalError);
  EXPECT_EQ(task, nullptr);
  std::array<ncclRmaSocketProxySendTask, 1> storage{};
  sender.tasks = storage.data();
  sender.capacity = 1;
  sender.count = 1;
  EXPECT_EQ(ncclRmaSocketProxyAcquireSendTask(&sender, &task), ncclInternalError);
}

TEST_F(RmaSocketMicrotest, PopAndCompleteSendTask_PutAdvancesWatermarkAndReturnsSlot) {
  Scene scene(/*nranks=*/1);
  auto& sender = scene.senders[0];
  sender.count = 1;
  sender.head = 0;
  sender.tasks[0].controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  sender.tasks[0].controlMsg.requestId = 88;

  ncclRmaSocketProxyPopAndCompleteSendTask(&sender);

  EXPECT_EQ(sender.lastCompletedRequestGlobalId, 88u);
  EXPECT_EQ(sender.count, 0);
  EXPECT_EQ(sender.head, 1);
  EXPECT_EQ(sender.tasks[0].controlMsg.type, 0);
  ncclRmaSocketProxyPopAndCompleteSendTask(&sender);
  ncclRmaSocketProxyPopAndCompleteSendTask(nullptr);
}

TEST_F(RmaSocketMicrotest, EnqueueRespondGetData_QueuesReplyWithRequestDestinationMetadata) {
  Scene scene(/*nranks=*/1);
  ncclRmaSocketProxyHeader request{};
  request.requestId = 17;
  request.localMrId = 22;
  request.localOff = 3;
  int source = 0;
  ncclRmaSocketProxyMrHandle handle{};

  ASSERT_EQ(ncclRmaSocketProxyEnqueueRespondGetData(&scene.senders[0], &request, ncclSuccess, &source,
                                                    NCCL_PTR_CUDA, &handle, sizeof(source)),
            ncclSuccess);
  const auto& task = scene.senders[0].tasks[0];
  EXPECT_EQ(task.controlMsg.type, ncclRmaSocketProxyMsgTypeRespondGetData);
  EXPECT_EQ(task.controlMsg.requestId, 17u);
  EXPECT_EQ(task.controlMsg.mrId, 22u);
  EXPECT_EQ(task.controlMsg.dstOff, 3u);
  EXPECT_EQ(task.srcPtr, &source);
  EXPECT_EQ(task.srcHandle, &handle);
  EXPECT_EQ(task.srcType, NCCL_PTR_CUDA);
  EXPECT_EQ(task.size, sizeof(source));
  EXPECT_EQ(ncclRmaSocketProxyEnqueueRespondGetData(nullptr, &request, 0, nullptr, 0, nullptr, 0),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_PartialControlMessageKeepsTaskQueued) {
  Scene scene(/*nranks=*/1);
  auto& sender = scene.senders[0];
  sender.count = 1;
  auto& task = sender.tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  ScopedHook progress(g_ncclSocketProgress, [](int op, ncclSocket*, void*, int size, int* offset, bool* closed) {
    EXPECT_EQ(op, NCCL_SOCKET_SEND);
    *offset = size - 1;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(sender.count, 1);
  EXPECT_EQ(task.controlMsgOffset, static_cast<int>(sizeof(task.controlMsg)) - 1);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_ControlCloseReturnsRemoteError) {
  Scene scene(/*nranks=*/1);
  auto& task = scene.senders[0].tasks[0];
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void*, int size, int* offset, bool* closed) {
    *offset = size;
    *closed = true;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclRemoteError);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_PutAckCompletesEmptyTaskAndAdvancesWatermark) {
  Scene scene(/*nranks=*/1);
  auto& sender = scene.senders[0];
  sender.count = 1;
  auto& task = sender.tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  task.controlMsg.requestId = 12;
  task.controlMsgOffset = sizeof(task.controlMsg);
  ScopedHook progress(g_ncclSocketProgress, [&task](int op, ncclSocket*, void* ptr, int size, int* offset,
                                                    bool* closed) {
    EXPECT_EQ(op, NCCL_SOCKET_RECV);
    EXPECT_EQ(ptr, &task.controlAck);
    *static_cast<uint32_t*>(ptr) = NCCL_RMA_SOCKET_CONTROL_ACK_MAGIC;
    *offset = size;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(sender.count, 0);
  EXPECT_EQ(sender.lastCompletedRequestGlobalId, 12u);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_InvalidOrClosedPutAckReturnsRemoteError) {
  Scene scene(/*nranks=*/1);
  auto& task = scene.senders[0].tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  task.controlMsgOffset = sizeof(task.controlMsg);
  bool close = false;
  ScopedHook progress(g_ncclSocketProgress, [&close](int, ncclSocket*, void* ptr, int size, int* offset,
                                                     bool* closed) {
    *static_cast<uint32_t*>(ptr) = close ? NCCL_RMA_SOCKET_CONTROL_ACK_MAGIC : 0;
    *offset = size;
    *closed = close;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclRemoteError);
  close = true;
  task.controlAckOffset = 0;
  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclRemoteError);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_CudaPayloadReadsChunkThroughGdrMappingAndKeepsLocalWatermark) {
  Scene scene(/*nranks=*/1);
  std::array<unsigned char, 12> device{};
  const std::array<unsigned char, 12> bar{{0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0}};
  auto source =
      Scene::MakeGdrMappedMr(device.data(), device.size(), const_cast<unsigned char*>(bar.data()), gdr_mh_t{31});
  std::array<unsigned char, 8> staging{};
  auto& sender = scene.senders[0];
  sender.buffer = staging.data();
  sender.count = 1;
  auto& task = sender.tasks[0];
  // The remote requester's id: completing the reply must not move this side's PUT watermark.
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  task.controlMsg.requestId = 99;
  task.controlMsgOffset = sizeof(task.controlMsg);
  task.srcPtr = device.data() + 2;
  task.srcHandle = &source;
  task.srcType = NCCL_PTR_CUDA;
  task.size = staging.size();
  ScopedHook copyFrom(g_wrapGdrCopyFromMapping, [&bar, &staging](gdr_mh_t mh, void* host, const void* map,
                                                                 size_t size) {
    EXPECT_EQ(mh.h, 31u);
    EXPECT_EQ(host, staging.data());
    EXPECT_EQ(map, bar.data() + 2);
    std::memcpy(host, map, size);
    return ncclSuccess;
  });
  ScopedHook progress(g_ncclSocketProgress, [&staging](int op, ncclSocket*, void* ptr, int size, int* offset,
                                                       bool* closed) {
    EXPECT_EQ(op, NCCL_SOCKET_SEND);
    EXPECT_EQ(ptr, staging.data());
    EXPECT_EQ(size, static_cast<int>(staging.size()));
    *offset = size;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(copyFrom.calls, 1);
  EXPECT_EQ(staging, (std::array<unsigned char, 8>{{1, 2, 3, 4, 5, 6, 7, 8}}));
  EXPECT_EQ(sender.count, 0);
  EXPECT_EQ(sender.lastCompletedRequestGlobalId, UINT64_MAX);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_MultiChunkPayloadAdvancesSourceAndCompletesOnSecondChunk) {
  Scene scene(/*nranks=*/1);
  const size_t payloadSize = NCCL_RMA_SOCKET_CHUNK_SIZE + 3;
  std::vector<unsigned char> source(payloadSize, 0x11);
  std::fill(source.end() - 3, source.end(), 0x22);
  std::vector<unsigned char> staging(NCCL_RMA_SOCKET_CHUNK_SIZE);
  auto& sender = scene.senders[0];
  sender.buffer = staging.data();
  sender.count = 1;
  auto& task = sender.tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  task.controlMsgOffset = sizeof(task.controlMsg);
  task.srcPtr = source.data();
  task.srcType = NCCL_PTR_HOST;
  task.size = source.size();
  std::vector<int> sentSizes;
  ScopedHook progress(g_ncclSocketProgress,
                      [&sentSizes](int op, ncclSocket*, void*, int size, int* offset, bool* closed) {
                        EXPECT_EQ(op, NCCL_SOCKET_SEND);
                        sentSizes.push_back(size);
                        *offset = size;
                        *closed = false;
                        return ncclSuccess;
                      });

  ASSERT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(sender.count, 1);
  EXPECT_EQ(task.payloadOffset, static_cast<size_t>(NCCL_RMA_SOCKET_CHUNK_SIZE));
  EXPECT_EQ(staging.front(), 0x11);
  EXPECT_EQ(staging.back(), 0x11);

  ASSERT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(sender.count, 0);
  EXPECT_EQ(sentSizes, (std::vector<int>{NCCL_RMA_SOCKET_CHUNK_SIZE, 3}));
  EXPECT_EQ(staging[0], 0x22);
  EXPECT_EQ(staging[1], 0x22);
  EXPECT_EQ(staging[2], 0x22);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_MissingStagingBufferReturnsError) {
  Scene scene(/*nranks=*/1);
  auto& task = scene.senders[0].tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  task.controlMsgOffset = sizeof(task.controlMsg);
  task.size = 1;

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclInternalError);
}

TEST_F(RmaSocketMicrotest, ProgressSendTask_PartialPayloadKeepsChunkAndClosedPayloadReturnsError) {
  Scene scene(/*nranks=*/1);
  std::array<unsigned char, 4> source{{1, 2, 3, 4}};
  std::array<unsigned char, 4> staging{};
  auto& sender = scene.senders[0];
  sender.buffer = staging.data();
  sender.count = 1;
  auto& task = sender.tasks[0];
  task.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  task.controlMsgOffset = sizeof(task.controlMsg);
  task.srcPtr = source.data();
  task.srcType = NCCL_PTR_HOST;
  task.size = source.size();
  bool close = false;
  ScopedHook progress(g_ncclSocketProgress, [&close](int, ncclSocket*, void*, int size, int* offset, bool* closed) {
    *offset = size - 1;
    *closed = close;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclSuccess);
  EXPECT_EQ(task.chunkSize, static_cast<int>(source.size()));
  EXPECT_EQ(task.payloadOffset, 0u);
  close = true;
  EXPECT_EQ(ncclRmaSocketProxyProgressSendTask(&scene.comm, 0, &task), ncclRemoteError);
  EXPECT_EQ(sender.count, 1);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_DataPutPublishesResolvedSourceAndDestination) {
  Scene scene;
  std::array<unsigned char, 16> source{};
  std::array<unsigned char, 16> destination{};
  auto src = scene.MakeMr(source.data(), source.size(), NCCL_PTR_CUDA, 3);
  auto dst = scene.MakeMr(destination.data(), destination.size(), NCCL_PTR_HOST, 4);
  ncclRmaSocketProxyPutArgs args{};

  ASSERT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 2, &src, 5, 4, &dst, 1, false, 0,
                                              nullptr, 0, &args),
            ncclSuccess);
  EXPECT_EQ(args.ctx, &scene.contexts[0]);
  EXPECT_EQ(args.srcPtr, source.data() + 2);
  EXPECT_EQ(args.srcHandle, &src);
  EXPECT_EQ(args.srcType, NCCL_PTR_CUDA);
  EXPECT_EQ(args.dstMrId, 4u);
  EXPECT_EQ(args.dstOff, 4u);
  EXPECT_EQ(args.opFlags, 0u);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_InvalidHandlesContextRankSizeOrRangeAreRejected) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  auto src = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 1);
  auto dst = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2);
  ncclRmaSocketProxyPutArgs args{};

  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 1, 0, &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), -1, 0, &src, 1, 0, &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, 1, 0, &dst,
                                              scene.comm.nranks, false, 0, nullptr, 0, &args),
            ncclInvalidArgument);
  const size_t oversized = static_cast<size_t>(INT_MAX) + 1;
  src.size = oversized;
  dst.size = oversized;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, oversized, 0, &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  src.size = data.size();
  dst.size = data.size();
  src.type = NCCL_PTR_DMABUF;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, 1, 0, &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  src.type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, data.size(), &src, 1, 0, &dst, 0,
                                              false, 0, nullptr, 0, &args),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_DestinationHandleTypeAndRangeAreValidated) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  auto src = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 1);
  auto dst = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2);
  ncclRmaSocketProxyPutArgs args{};

  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, 1, 0, nullptr, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  dst.type = NCCL_PTR_DMABUF;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, 1, 0, &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
  dst.type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, &src, 1, data.size(), &dst, 0, false, 0,
                                              nullptr, 0, &args),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_ContextAtUpperBoundIsRejected) {
  Scene scene;
  ncclRmaSocketProxyPutArgs args{};

  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), scene.contexts[0].nContexts, 0, nullptr, 0, 0,
                                              nullptr, 0, false, 0, nullptr, 0, &args),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_SignalOnlyPutPublishesSignalMetadata) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 16> signalData{};
  auto signal = scene.MakeMr(signalData.data(), signalData.size(), NCCL_PTR_CUDA, 7);
  ncclRmaSocketProxyPutArgs args{};

  ASSERT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 1, true,
                                              sizeof(uint64_t), &signal, NCCL_NET_SIGNAL_OP_ADD, &args),
            ncclSuccess);
  EXPECT_EQ(args.srcType, NCCL_PTR_HOST);
  EXPECT_EQ(args.opFlags, NCCL_RMA_SOCKET_OP_HAS_SIGNAL);
  EXPECT_EQ(args.signalMrId, 7u);
}

TEST_F(RmaSocketMicrotest, ValidatePutArgs_InvalidSignalHandleOperationTypeAlignmentOrRangeIsRejected) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 16> signalData{};
  auto signal = scene.MakeMr(signalData.data(), signalData.size(), NCCL_PTR_CUDA, 7);
  ncclRmaSocketProxyPutArgs args{};

  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, true, 0,
                                              nullptr, NCCL_NET_SIGNAL_OP_INC, &args),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, true, 0,
                                              &signal, 99, &args),
            ncclInvalidArgument);
  signal.type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, true, 0,
                                              &signal, NCCL_NET_SIGNAL_OP_INC, &args),
            ncclInvalidUsage);
  signal.type = NCCL_PTR_CUDA;
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, true, 1,
                                              &signal, NCCL_NET_SIGNAL_OP_INC, &args),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidatePutArgs(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, true,
                                              signalData.size(), &signal, NCCL_NET_SIGNAL_OP_INC, &args),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, IPut_ValidRequestQueuesWireTaskAndReturnsCompletionObject) {
  Scene scene;
  std::array<unsigned char, 16> source{};
  std::array<unsigned char, 16> destination{};
  auto src = scene.MakeMr(source.data(), source.size(), NCCL_PTR_CUDA, 3);
  auto dst = scene.MakeMr(destination.data(), destination.size(), NCCL_PTR_HOST, 4);
  void* request = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyIPut(scene.contexts.data(), 0, 2, &src, 5, 4, &dst, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<ncclRmaSocketProxyRequest*>(request);
  ASSERT_NE(req, nullptr);
  EXPECT_EQ(req->type, ncclRmaSocketProxyRequestTypePut);
  EXPECT_EQ(req->peer, 1);
  EXPECT_EQ(scene.contexts[0].activeOps, 1);
  ASSERT_EQ(scene.senders[1].count, 1);
  const auto& task = scene.senders[1].tasks[0];
  EXPECT_EQ(task.srcPtr, source.data() + 2);
  EXPECT_EQ(task.srcHandle, &src);
  EXPECT_EQ(task.srcType, NCCL_PTR_CUDA);
  EXPECT_EQ(task.size, 5u);
  EXPECT_EQ(task.controlMsg.type, ncclRmaSocketProxyMsgTypeIput);
  EXPECT_EQ(task.controlMsg.size, 5u);
  EXPECT_EQ(task.controlMsg.opFlags, 0u);
  EXPECT_EQ(task.controlMsg.mrId, 4u);
  EXPECT_EQ(task.controlMsg.dstOff, 4u);
  EXPECT_EQ(task.controlMsg.requestId, req->globalRequestId);
  ncclRmaSocketProxyReleaseRequest(req);
}

TEST_F(RmaSocketMicrotest, IPutSignal_ValidSignalOnlyRequestQueuesSignalHeader) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 16> signalData{};
  auto signal = scene.MakeMr(signalData.data(), signalData.size(), NCCL_PTR_CUDA, 9);
  void* request = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyIPutSignal(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 1, sizeof(uint64_t),
                                        &signal, 6, NCCL_NET_SIGNAL_OP_ADD, true, 0, &request),
            ncclSuccess);
  const auto& task = scene.senders[1].tasks[0];
  EXPECT_EQ(task.size, 0u);
  EXPECT_EQ(task.controlMsg.opFlags, NCCL_RMA_SOCKET_OP_HAS_SIGNAL);
  EXPECT_EQ(task.controlMsg.signalMrId, 9u);
  EXPECT_EQ(task.controlMsg.signalOff, sizeof(uint64_t));
  EXPECT_EQ(task.controlMsg.signalValue, 6u);
  EXPECT_EQ(task.controlMsg.signalOp, NCCL_NET_SIGNAL_OP_ADD);
  ncclRmaSocketProxyReleaseRequest(static_cast<ncclRmaSocketProxyRequest*>(request));
}

TEST_F(RmaSocketMicrotest, IPut_NullTopLevelArgumentsAreRejected) {
  Scene scene;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyIPut(nullptr, 0, 0, nullptr, 0, 0, nullptr, 0, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(ncclRmaSocketProxyIPut(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, 0, nullptr),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, IPut_FullSendQueueReleasesAllocatedRequest) {
  Scene scene;
  scene.senders[0].count = scene.senders[0].capacity;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyIPut(scene.contexts.data(), 0, 0, nullptr, 0, 0, nullptr, 0, 0, &request),
            ncclInternalError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
  EXPECT_EQ(scene.requestStorage[0][0].type, ncclRmaSocketProxyRequestTypeFree);
}

TEST_F(RmaSocketMicrotest, ValidateGet_ValidRequestReturnsContextAndHandles) {
  Scene scene;
  std::array<unsigned char, 16> remoteData{};
  std::array<unsigned char, 16> localData{};
  auto remote = scene.MakeMr(remoteData.data(), remoteData.size(), NCCL_PTR_HOST, 1);
  auto local = scene.MakeMr(localData.data(), localData.size(), NCCL_PTR_CUDA, 2);
  ncclRmaSocketProxyCtx* ctx = nullptr;
  ncclRmaSocketProxyMrHandle* remoteOut = nullptr;
  ncclRmaSocketProxyMrHandle* localOut = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 2, &remote, 4, 3, &local, 1, &ctx,
                                          &remoteOut, &localOut),
            ncclSuccess);
  EXPECT_EQ(ctx, &scene.contexts[0]);
  EXPECT_EQ(remoteOut, &remote);
  EXPECT_EQ(localOut, &local);
}

TEST_F(RmaSocketMicrotest, ValidateGet_InvalidArgumentsClosedPeerTypesOrRangesAreRejected) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  auto remote = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 1);
  auto local = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2);
  ncclRmaSocketProxyCtx* ctx = nullptr;
  ncclRmaSocketProxyMrHandle* remoteOut = nullptr;
  ncclRmaSocketProxyMrHandle* localOut = nullptr;

  EXPECT_EQ(ncclRmaSocketProxyValidateGet(nullptr, 0, 0, &remote, 1, 0, &local, 0, &ctx, &remoteOut, &localOut),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 0, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, nullptr, 1, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), -1, 0, &remote, 1, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, 0, &local,
                                          scene.comm.nranks, &ctx, &remoteOut, &localOut),
            ncclInvalidArgument);
  scene.receivers[0].recvState = ncclRmaSocketProxyRecvStateClosed;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclRemoteError);
  scene.receivers[0].recvState = ncclRmaSocketProxyRecvStateControlMsg;
  remote.type = NCCL_PTR_DMABUF;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  remote.type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, data.size(), &remote, 1, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ValidateGet_LocalHandleContextAndMaximumSizeAreValidated) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  auto remote = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 1);
  auto local = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2);
  ncclRmaSocketProxyCtx* ctx = nullptr;
  ncclRmaSocketProxyMrHandle* remoteOut = nullptr;
  ncclRmaSocketProxyMrHandle* localOut = nullptr;

  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, 0, nullptr, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), scene.contexts[0].nContexts, 0, &remote, 1, 0,
                                          &local, 0, &ctx, &remoteOut, &localOut),
            ncclInvalidArgument);
  const size_t oversized = static_cast<size_t>(INT_MAX) + 1;
  remote.size = oversized;
  local.size = oversized;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, oversized, 0, &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
  remote.size = data.size();
  local.size = data.size();
  local.type = NCCL_PTR_DMABUF;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, 0, &local, 0, &ctx, &remoteOut,
                                          &localOut),
            ncclInvalidArgument);
  local.type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyValidateGet(scene.contexts.data(), 0, 0, &remote, 1, data.size(), &local, 0, &ctx,
                                          &remoteOut, &localOut),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, IGet_ValidRequestQueuesRemoteReadAndLocalDestinationMetadata) {
  Scene scene;
  std::array<unsigned char, 16> remoteData{};
  std::array<unsigned char, 16> localData{};
  auto remote = scene.MakeMr(remoteData.data(), remoteData.size(), NCCL_PTR_HOST, 4);
  auto local = scene.MakeMr(localData.data(), localData.size(), NCCL_PTR_HOST, 5);
  void* request = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyIGet(scene.contexts.data(), 0, 2, &remote, 6, 3, &local, 1, 0, &request),
            ncclSuccess);
  auto* req = static_cast<ncclRmaSocketProxyRequest*>(request);
  ASSERT_NE(req, nullptr);
  EXPECT_EQ(req->type, ncclRmaSocketProxyRequestTypeGet);
  const auto& task = scene.senders[1].tasks[0];
  EXPECT_EQ(task.controlMsg.type, ncclRmaSocketProxyMsgTypeRequestGetData);
  EXPECT_EQ(task.controlMsg.mrId, 4u);
  EXPECT_EQ(task.controlMsg.dstOff, 2u);
  EXPECT_EQ(task.controlMsg.size, 6u);
  EXPECT_EQ(task.controlMsg.localMrId, 5u);
  EXPECT_EQ(task.controlMsg.localOff, 3u);
  ncclRmaSocketProxyReleaseRequest(req);
}

TEST_F(RmaSocketMicrotest, IGet_NullRequestPointerIsRejected) {
  Scene scene;
  EXPECT_EQ(ncclRmaSocketProxyIGet(scene.contexts.data(), 0, 0, nullptr, 1, 0, nullptr, 0, 0, nullptr),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, IGet_FullSendQueueReleasesAllocatedRequest) {
  Scene scene;
  std::array<unsigned char, 4> data{};
  auto remote = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 1);
  auto local = scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2);
  scene.senders[0].count = scene.senders[0].capacity;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyIGet(scene.contexts.data(), 0, 0, &remote, 1, 0, &local, 0, 0, &request),
            ncclInternalError);
  EXPECT_EQ(request, nullptr);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
  EXPECT_EQ(scene.requestStorage[0][0].type, ncclRmaSocketProxyRequestTypeFree);
}

TEST_F(RmaSocketMicrotest, IFlush_ValidRequestCompletesSynchronouslyWithoutRequestObject) {
  Scene scene;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyIFlush(scene.contexts.data(), 0, nullptr, 1, &request), ncclSuccess);
  EXPECT_EQ(request, nullptr);
}

TEST_F(RmaSocketMicrotest, IFlush_InvalidContextOrRankIsRejected) {
  Scene scene;
  void* request = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyIFlush(nullptr, 0, nullptr, 0, &request), ncclInvalidArgument);
  EXPECT_EQ(request, reinterpret_cast<void*>(0x1));
  EXPECT_EQ(ncclRmaSocketProxyIFlush(scene.contexts.data(), 0, nullptr, 0, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyIFlush(scene.contexts.data(), -1, nullptr, 0, &request), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyIFlush(scene.contexts.data(), scene.contexts[0].nContexts, nullptr, 0, &request),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyIFlush(scene.contexts.data(), 0, nullptr, scene.comm.nranks, &request),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, Test_CompletedPutSetsDoneAndReleasesRequest) {
  Scene scene;
  auto& req = ArmRequest(scene, ncclRmaSocketProxyRequestTypePut, /*peer=*/1, /*id=*/6);
  scene.senders[1].lastCompletedRequestGlobalId = 6;
  int done = 0;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
}

TEST_F(RmaSocketMicrotest, Test_CompletedGetReadsReceiverWatermark) {
  Scene scene;
  auto& req = ArmRequest(scene, ncclRmaSocketProxyRequestTypeGet, /*peer=*/1, /*id=*/6);
  scene.receivers[1].lastCompletedRequestGlobalId = 6;
  int done = 0;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
  EXPECT_EQ(scene.senders[1].lastCompletedRequestGlobalId, UINT64_MAX);
}

TEST_F(RmaSocketMicrotest, Test_RequestIdBeforeWrappedWatermarkCompletes) {
  Scene scene;
  auto& req = ArmRequest(scene, ncclRmaSocketProxyRequestTypePut, /*peer=*/0, /*id=*/UINT64_MAX);
  scene.senders[0].lastCompletedRequestGlobalId = 0;
  int done = 0;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclSuccess);
  EXPECT_EQ(done, 1);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
}

TEST_F(RmaSocketMicrotest, Test_IncompleteRequestRemainsLive) {
  Scene scene;
  auto& req = ArmRequest(scene, ncclRmaSocketProxyRequestTypePut, /*peer=*/0, /*id=*/7);
  scene.senders[0].lastCompletedRequestGlobalId = 6;
  int done = -1;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclSuccess);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypePut);
  EXPECT_EQ(scene.contexts[0].activeOps, 1);
}

TEST_F(RmaSocketMicrotest, Test_IncompleteGetFromClosedPeerReturnsErrorAndReleasesRequest) {
  Scene scene;
  auto& req = ArmRequest(scene, ncclRmaSocketProxyRequestTypeGet, /*peer=*/0, /*id=*/2);
  scene.receivers[0].lastCompletedRequestGlobalId = 1;
  int done = -1;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclRemoteError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
}

TEST_F(RmaSocketMicrotest, Test_InvalidPointersOrOwnershipAreRejected) {
  Scene scene;
  for (auto& receiver : scene.receivers) receiver.recvState = ncclRmaSocketProxyRecvStateClosed;
  int done = 0;
  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, nullptr, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, nullptr, &done), ncclInternalError);

  ncclRmaSocketProxyRequest req{};
  req.type = ncclRmaSocketProxyRequestTypePut;
  req.ctx = &scene.contexts[0];
  EXPECT_EQ(ncclRmaSocketProxyTest(nullptr, &req, &done), ncclInvalidArgument);
  ncclRmaSocketProxyCollComm other{};
  EXPECT_EQ(ncclRmaSocketProxyTest(&other, &req, &done), ncclInvalidArgument);

  ncclRmaSocketProxyRequest freed{};
  freed.ctx = &scene.contexts[0];
  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &freed, &done), ncclInvalidArgument);
  ncclRmaSocketProxyRequest orphan{};
  orphan.type = ncclRmaSocketProxyRequestTypePut;
  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &orphan, &done), ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, Test_UnknownLiveRequestTypeReturnsErrorAndReleasesRequest) {
  Scene scene;
  auto& req = ArmRequest(scene, static_cast<ncclRmaSocketProxyRequestType>(3), /*peer=*/0, /*id=*/0);
  int done = -1;

  EXPECT_EQ(ncclRmaSocketProxyTest(&scene.comm, &req, &done), ncclInternalError);
  EXPECT_EQ(done, 0);
  EXPECT_EQ(req.type, ncclRmaSocketProxyRequestTypeFree);
  EXPECT_EQ(scene.contexts[0].activeOps, 0);
}

TEST_F(RmaSocketMicrotest, ProgressComm_ReceiveFailureRecordsLastError) {
  Scene scene(/*nranks=*/1);
  ScopedHook progress(g_ncclSocketProgress,
                      [](int, ncclSocket*, void*, int, int*, bool*) { return ncclSystemError; });

  EXPECT_EQ(ncclRmaSocketProxyProgressComm(&scene.comm), ncclSystemError);
  EXPECT_EQ(scene.comm.lastError, ncclSystemError);
  EXPECT_EQ(ncclRmaSocketProxyProgressComm(nullptr), ncclInvalidArgument);
}

TEST_F(RmaSocketMicrotest, ProgressComm_SendFailureRecordsLastError) {
  Scene scene(/*nranks=*/1);
  scene.receivers[0].recvState = ncclRmaSocketProxyRecvStateClosed;
  scene.senders[0].count = 1;
  ScopedHook progress(g_ncclSocketProgress,
                      [](int, ncclSocket*, void*, int, int*, bool*) { return ncclRemoteError; });

  EXPECT_EQ(ncclRmaSocketProxyProgressComm(&scene.comm), ncclRemoteError);
  EXPECT_EQ(scene.comm.lastError, ncclRemoteError);
}

TEST_F(RmaSocketMicrotest, Progress_NullContextIsNoOpAndValidContextUsesOwningComm) {
  EXPECT_EQ(ncclRmaSocketProxyProgress(nullptr), ncclSuccess);

  Scene scene;
  for (auto& receiver : scene.receivers) receiver.recvState = ncclRmaSocketProxyRecvStateClosed;
  scene.senders[1].count = 1;
  ncclSocket* progressed = nullptr;
  ScopedHook progress(g_ncclSocketProgress, [&progressed](int, ncclSocket* sock, void*, int, int*, bool*) {
    progressed = sock;
    return ncclRemoteError;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgress(scene.contexts.data()), ncclRemoteError);
  EXPECT_EQ(progressed, &scene.senders[1].sock);
  EXPECT_EQ(scene.comm.lastError, ncclRemoteError);
}

TEST_F(RmaSocketMicrotest, QueryLastError_ReportsStoredTerminalError) {
  bool hasError = true;
  EXPECT_EQ(ncclRmaSocketProxyQueryLastError(nullptr, &hasError), ncclSuccess);
  EXPECT_FALSE(hasError);
  EXPECT_EQ(ncclRmaSocketProxyQueryLastError(nullptr, nullptr), ncclInvalidArgument);

  Scene scene;
  EXPECT_EQ(ncclRmaSocketProxyQueryLastError(scene.contexts.data(), &hasError), ncclSuccess);
  EXPECT_FALSE(hasError);
  scene.comm.lastError = ncclRemoteError;
  EXPECT_EQ(ncclRmaSocketProxyQueryLastError(scene.contexts.data(), &hasError), ncclSuccess);
  EXPECT_TRUE(hasError);
}

TEST_F(RmaSocketMicrotest, PluginTable_ExportsSocketRmaCallbacks) {
  EXPECT_STREQ(ncclRmaSocketProxy.name, "RMA_SOCKET_PROXY");
  EXPECT_EQ(ncclRmaSocketProxy.init, ncclRmaSocketProxyInit);
  EXPECT_EQ(ncclRmaSocketProxy.devices, ncclRmaSocketProxyDevices);
  EXPECT_EQ(ncclRmaSocketProxy.getRmaProperties, ncclRmaSocketProxyGetRmaProperties);
  EXPECT_EQ(ncclRmaSocketProxy.getProperties, ncclRmaSocketProxyGetProperties);
  EXPECT_EQ(ncclRmaSocketProxy.listen, ncclRmaSocketProxyListen);
  EXPECT_EQ(ncclRmaSocketProxy.connect, ncclRmaSocketProxyConnect);
  EXPECT_EQ(ncclRmaSocketProxy.createContext, ncclRmaSocketProxyCreateContext);
  EXPECT_EQ(ncclRmaSocketProxy.regMrSym, ncclRmaSocketProxyRegMrSym);
  EXPECT_EQ(ncclRmaSocketProxy.regMrSymDmaBuf, ncclRmaSocketProxyRegMrSymDmaBuf);
  EXPECT_EQ(ncclRmaSocketProxy.deregMrSym, ncclRmaSocketProxyDeregMrSym);
  EXPECT_EQ(ncclRmaSocketProxy.destroyContext, ncclRmaSocketProxyDestroyContext);
  EXPECT_EQ(ncclRmaSocketProxy.closeColl, ncclRmaSocketProxyCloseColl);
  EXPECT_EQ(ncclRmaSocketProxy.closeListen, ncclRmaSocketProxyCloseListen);
  EXPECT_EQ(ncclRmaSocketProxy.iput,
            static_cast<decltype(ncclRmaSocketProxy.iput)>(&ncclRmaSocketProxyIPut));
  EXPECT_EQ(ncclRmaSocketProxy.iputSignal,
            static_cast<decltype(ncclRmaSocketProxy.iputSignal)>(&ncclRmaSocketProxyIPutSignal));
  EXPECT_EQ(ncclRmaSocketProxy.iget,
            static_cast<decltype(ncclRmaSocketProxy.iget)>(&ncclRmaSocketProxyIGet));
  EXPECT_EQ(ncclRmaSocketProxy.iflush, ncclRmaSocketProxyIFlush);
  EXPECT_EQ(ncclRmaSocketProxy.test, ncclRmaSocketProxyTest);
  EXPECT_EQ(ncclRmaSocketProxy.rmaProgress, ncclRmaSocketProxyProgress);
  EXPECT_EQ(ncclRmaSocketProxy.queryLastError, ncclRmaSocketProxyQueryLastError);
  EXPECT_EQ(ncclRmaSocketProxy.finalize, ncclRmaSocketProxyFinalize);
}

}  // namespace
