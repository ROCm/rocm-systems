/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/rma_socket_progress_recver.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <functional>
#include <vector>

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/socket_fakes.h"
#include "transport/RmaSocketTest.h"

namespace {

static ncclResult_t DefaultProgressGdrCopy(gdr_mh_t, void*, const void*, size_t) {
  return ncclInternalError;
}

std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_testProgressGdrCopyTo =
    DefaultProgressGdrCopy;
std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_testProgressGdrCopyFrom =
    DefaultProgressGdrCopy;

ncclResult_t TestProgressGdrCopyTo(gdr_mh_t mh, void* map, const void* host, size_t size) {
  return g_testProgressGdrCopyTo(mh, map, host, size);
}

ncclResult_t TestProgressGdrCopyFrom(gdr_mh_t mh, void* host, const void* map, size_t size) {
  return g_testProgressGdrCopyFrom(mh, host, map, size);
}

}  // namespace

#define wrap_gdr_copy_to_mapping TestProgressGdrCopyTo
#define wrap_gdr_copy_from_mapping TestProgressGdrCopyFrom
#include RMA_SOCKET_PROGRESS_RECVER_CC_PATH
#undef wrap_gdr_copy_from_mapping
#undef wrap_gdr_copy_to_mapping

namespace {

using rma_socket_test::Scene;

class RmaSocketProgressRecverMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    g_testProgressGdrCopyTo = DefaultProgressGdrCopy;
    g_testProgressGdrCopyFrom = DefaultProgressGdrCopy;
    ResetSocketFakes();
    ResetNcclFakes();
  }
};

TEST_F(RmaSocketProgressRecverMicrotest, FindMr_KnownIdReturnsHandleAndUnknownIdClearsOutput) {
  Scene scene;
  std::array<unsigned char, 8> firstData{};
  std::array<unsigned char, 8> secondData{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(firstData.data(), firstData.size(), NCCL_PTR_HOST, 11),
      scene.MakeMr(secondData.data(), secondData.size(), NCCL_PTR_HOST, 12)};
  scene.LinkMrs(handles);
  ncclRmaSocketProxyMrHandle* found = nullptr;

  EXPECT_EQ(ncclRmaSocketProxyFindMr(&scene.comm, 12, &found), ncclSuccess);
  EXPECT_EQ(found, &handles[1]);
  EXPECT_EQ(ncclRmaSocketProxyFindMr(&scene.comm, 99, &found), ncclInvalidUsage);
  EXPECT_EQ(found, nullptr);
  EXPECT_EQ(ncclRmaSocketProxyFindMr(nullptr, 11, &found), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyFindMr(&scene.comm, 11, nullptr), ncclInvalidArgument);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResolveDataTarget_ValidRangePublishesTargetAndRetainsMr) {
  Scene scene;
  std::array<unsigned char, 16> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 7)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[1];
  receiver.controlMsg.mrId = 7;
  receiver.controlMsg.dstOff = 3;
  receiver.controlMsg.size = 5;

  EXPECT_EQ(ncclRmaSocketProxyResolveDataTarget(&scene.comm, 1, &receiver), ncclSuccess);
  EXPECT_EQ(receiver.dataPtr, data.data() + 3);
  EXPECT_EQ(receiver.dataType, NCCL_PTR_HOST);
  EXPECT_EQ(receiver.dataHandle, &handles[0]);
  EXPECT_EQ(handles[0].recvActiveOps, 1);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResolveDataTarget_UnknownOrOutOfRangeMrIsRejectedWithoutRetain) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 7)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.mrId = 8;
  receiver.controlMsg.size = 1;
  EXPECT_EQ(ncclRmaSocketProxyResolveDataTarget(&scene.comm, 0, &receiver), ncclInvalidUsage);

  receiver.controlMsg.mrId = 7;
  receiver.controlMsg.dstOff = data.size();
  receiver.controlMsg.size = 1;
  EXPECT_EQ(ncclRmaSocketProxyResolveDataTarget(&scene.comm, 0, &receiver), ncclInvalidArgument);
  EXPECT_EQ(handles[0].recvActiveOps, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResolveSignalTarget_IncAndAddPublishExpectedDelta) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 24> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_CUDA, 21)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.signalMrId = 21;
  receiver.controlMsg.signalOff = sizeof(uint64_t);
  receiver.controlMsg.signalOp = NCCL_NET_SIGNAL_OP_INC;
  receiver.controlMsg.signalValue = 9;

  ASSERT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclSuccess);
  EXPECT_EQ(receiver.signalPtr, data.data() + sizeof(uint64_t));
  EXPECT_EQ(receiver.signalDelta, 1u);
  EXPECT_EQ(receiver.hasSignal, 1);
  EXPECT_EQ(handles[0].recvActiveOps, 1);
  ncclRmaSocketProxyResetRecv(&receiver);

  receiver.controlMsg.signalMrId = 21;
  receiver.controlMsg.signalOff = 0;
  receiver.controlMsg.signalOp = NCCL_NET_SIGNAL_OP_ADD;
  receiver.controlMsg.signalValue = 9;
  ASSERT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclSuccess);
  EXPECT_EQ(receiver.signalDelta, 9u);
  ncclRmaSocketProxyResetRecv(&receiver);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResolveSignalTarget_InvalidOperationTypeOrRangeIsRejected) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 16> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_CUDA, 1)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.signalMrId = 1;
  receiver.controlMsg.signalOp = 99;
  EXPECT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclInvalidUsage);

  receiver.controlMsg.signalOp = NCCL_NET_SIGNAL_OP_INC;
  handles[0].type = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclInvalidUsage);

  handles[0].type = NCCL_PTR_CUDA;
  receiver.controlMsg.signalOff = data.size();
  EXPECT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclInvalidArgument);
  EXPECT_EQ(handles[0].recvActiveOps, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResolveSignalTarget_UnknownMrIsRejected) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.signalMrId = 404;
  receiver.controlMsg.signalOp = NCCL_NET_SIGNAL_OP_INC;

  EXPECT_EQ(ncclRmaSocketProxyResolveSignalTarget(&scene.comm, 0, &receiver), ncclInvalidUsage);
  EXPECT_EQ(receiver.signalHandle, nullptr);
  EXPECT_EQ(receiver.hasSignal, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ApplySignal_NoSignalIsNoOpAndInvalidTargetIsRejected) {
  ncclRmaSocketProxyPeerReceiver receiver{};

  EXPECT_EQ(ncclRmaSocketProxyApplySignal(&receiver), ncclSuccess);
  receiver.hasSignal = 1;
  receiver.signalType = NCCL_PTR_HOST;
  EXPECT_EQ(ncclRmaSocketProxyApplySignal(&receiver), ncclInvalidArgument);
  receiver.signalType = NCCL_PTR_CUDA;
  EXPECT_EQ(ncclRmaSocketProxyApplySignal(&receiver), ncclInvalidArgument);
}

TEST_F(RmaSocketProgressRecverMicrotest, ApplySignal_UnalignedBarMappingIsRejectedBeforeCopy) {
  alignas(uint64_t) std::array<unsigned char, 16> data{};
  alignas(uint64_t) std::array<unsigned char, 17> bar{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.type = NCCL_PTR_CUDA;
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data() + 1;
  ncclRmaSocketProxyPeerReceiver receiver{};
  receiver.hasSignal = 1;
  receiver.signalType = NCCL_PTR_CUDA;
  receiver.signalPtr = data.data();
  receiver.signalHandle = &handle;
  ScopedHook copyFrom(g_testProgressGdrCopyFrom, [](gdr_mh_t, void*, const void*, size_t) {
    ADD_FAILURE() << "unaligned mapping must be rejected before reading";
    return ncclSuccess;
  });
  ScopedHook copyTo(g_testProgressGdrCopyTo, [](gdr_mh_t, void*, const void*, size_t) {
    ADD_FAILURE() << "unaligned mapping must be rejected before writing";
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyApplySignal(&receiver), ncclInternalError);
  EXPECT_EQ(copyFrom.calls, 0);
  EXPECT_EQ(copyTo.calls, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ApplySignal_AlignedMappingReadsAddsAndWritesSignal) {
  alignas(uint64_t) std::array<unsigned char, 16> data{};
  alignas(uint64_t) std::array<unsigned char, 16> bar{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.type = NCCL_PTR_CUDA;
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data();
  ncclRmaSocketProxyPeerReceiver receiver{};
  receiver.hasSignal = 1;
  receiver.signalType = NCCL_PTR_CUDA;
  receiver.signalPtr = data.data();
  receiver.signalDelta = 5;
  receiver.signalHandle = &handle;
  ScopedHook copyFrom(g_testProgressGdrCopyFrom,
                      [&bar](gdr_mh_t, void* host, const void* map, size_t size) {
                        EXPECT_EQ(map, bar.data());
                        EXPECT_EQ(size, sizeof(uint64_t));
                        *static_cast<uint64_t*>(host) = 10;
                        return ncclSuccess;
                      });
  ScopedHook copyTo(g_testProgressGdrCopyTo, [&bar](gdr_mh_t, void* map, const void* host, size_t size) {
    EXPECT_EQ(map, bar.data());
    EXPECT_EQ(size, sizeof(uint64_t));
    EXPECT_EQ(*static_cast<const uint64_t*>(host), 15u);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyApplySignal(&receiver), ncclSuccess);
  EXPECT_EQ(copyFrom.calls, 1);
  EXPECT_EQ(copyTo.calls, 1);
}

TEST_F(RmaSocketProgressRecverMicrotest, ResetRecv_ReleasesTargetsAndRestoresControlState) {
  ncclRmaSocketProxyMrHandle data{};
  ncclRmaSocketProxyMrHandle signal{};
  data.recvActiveOps = 1;
  signal.recvActiveOps = 1;
  ncclRmaSocketProxyPeerReceiver receiver{};
  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  receiver.controlMsgOffset = 4;
  receiver.recvPayloadOffset = 9;
  receiver.dataHandle = &data;
  receiver.signalHandle = &signal;
  receiver.hasSignal = 1;

  ncclRmaSocketProxyResetRecv(&receiver);

  EXPECT_EQ(data.recvActiveOps, 0);
  EXPECT_EQ(signal.recvActiveOps, 0);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
  EXPECT_EQ(receiver.controlMsg.type, 0);
  EXPECT_EQ(receiver.controlMsgOffset, 0);
  EXPECT_EQ(receiver.recvPayloadOffset, 0u);
  EXPECT_EQ(receiver.dataHandle, nullptr);
  EXPECT_EQ(receiver.signalHandle, nullptr);
  EXPECT_EQ(receiver.hasSignal, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ReleaseRecvMr_ZeroReferenceDoesNotUnderflow) {
  ncclRmaSocketProxyMrHandle handle{};
  handle.mrId = 4;

  ncclRmaSocketProxyReleaseRecvMr(&handle);
  ncclRmaSocketProxyReleaseRecvMr(nullptr);

  EXPECT_EQ(handle.recvActiveOps, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, CompleteRecvPayload_PutResetsReceiverAndGetResponseAdvancesWatermark) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  EXPECT_EQ(ncclRmaSocketProxyCompleteRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);

  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  receiver.controlMsg.requestId = 77;
  EXPECT_EQ(ncclRmaSocketProxyCompleteRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.lastCompletedRequestGlobalId, 77u);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, CompleteRecvPayload_UnknownTypeReturnsErrorWithoutReset) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = 99;

  EXPECT_EQ(ncclRmaSocketProxyCompleteRecvPayload(&scene.comm, 0), ncclInternalError);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStatePayload);
}

TEST_F(RmaSocketProgressRecverMicrotest, StartRecvPut_DataAndSignalResolveBeforePublishingAckState) {
  Scene scene;
  std::array<unsigned char, 16> data{};
  alignas(uint64_t) std::array<unsigned char, 16> signal{};
  std::array<unsigned char, 32> staging{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2),
      scene.MakeMr(signal.data(), signal.size(), NCCL_PTR_CUDA, 3)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.buffer = staging.data();
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeIput;
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.opFlags = NCCL_RMA_SOCKET_OP_HAS_SIGNAL;
  receiver.controlMsg.mrId = 2;
  receiver.controlMsg.size = 4;
  receiver.controlMsg.signalMrId = 3;
  receiver.controlMsg.signalOp = NCCL_NET_SIGNAL_OP_ADD;
  receiver.controlMsg.signalValue = 5;

  ASSERT_EQ(ncclRmaSocketProxyStartRecvPut(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlAck);
  EXPECT_EQ(receiver.controlAck, NCCL_RMA_SOCKET_CONTROL_ACK_MAGIC);
  EXPECT_EQ(receiver.dataHandle, &handles[0]);
  EXPECT_EQ(receiver.signalHandle, &handles[1]);
  ncclRmaSocketProxyResetRecv(&receiver);
}

TEST_F(RmaSocketProgressRecverMicrotest, StartRecvPut_InvalidWireFieldsOrMissingBufferAreRejected) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 2)};
  scene.LinkMrs(handles);
  auto& msg = scene.receivers[0].controlMsg;
  msg.size = static_cast<uint64_t>(INT_MAX) + 1;
  EXPECT_EQ(ncclRmaSocketProxyStartRecvPut(&scene.comm, 0), ncclRemoteError);
  msg.size = 1;
  msg.status = ncclRemoteError;
  EXPECT_EQ(ncclRmaSocketProxyStartRecvPut(&scene.comm, 0), ncclRemoteError);
  msg.status = ncclSuccess;
  msg.opFlags = 0x80;
  EXPECT_EQ(ncclRmaSocketProxyStartRecvPut(&scene.comm, 0), ncclRemoteError);
  msg.opFlags = 0;
  msg.mrId = 2;
  EXPECT_EQ(ncclRmaSocketProxyStartRecvPut(&scene.comm, 0), ncclInternalError);
  EXPECT_EQ(handles[0].recvActiveOps, 1);
  ncclRmaSocketProxyResetRecv(&scene.receivers[0]);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressControlAck_PartialThenCompleteTransitionsToPayload) {
  ncclRmaSocketProxyPeerReceiver receiver{};
  receiver.recvState = ncclRmaSocketProxyRecvStateControlAck;
  bool complete = false;
  ScopedHook progress(g_ncclSocketProgress,
                      [&complete](int op, ncclSocket*, void*, int size, int* offset, bool* closed) {
                        EXPECT_EQ(op, NCCL_SOCKET_SEND);
                        *closed = false;
                        *offset = complete ? size : size - 1;
                        return ncclSuccess;
                      });

  EXPECT_EQ(ncclRmaSocketProxyProgressControlAck(&receiver, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlAck);
  complete = true;
  EXPECT_EQ(ncclRmaSocketProxyProgressControlAck(&receiver, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStatePayload);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressControlAck_ClosedSocketReturnsRemoteError) {
  ncclRmaSocketProxyPeerReceiver receiver{};
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void*, int, int*, bool* closed) {
    *closed = true;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressControlAck(&receiver, 1), ncclRemoteError);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRespondGetData_ValidResponseResolvesDestinationAndStartsPayload) {
  Scene scene;
  std::array<unsigned char, 16> data{};
  std::array<unsigned char, 16> staging{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 8)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.buffer = staging.data();
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.size = 4;
  receiver.controlMsg.mrId = 8;
  receiver.controlMsg.dstOff = 2;

  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStatePayload);
  EXPECT_EQ(receiver.dataPtr, data.data() + 2);
  ncclRmaSocketProxyResetRecv(&receiver);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRespondGetData_ErrorOrInvalidPayloadIsRejected) {
  Scene scene;
  auto& msg = scene.receivers[0].controlMsg;
  msg.status = ncclRemoteError;
  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclRemoteError);
  msg.status = ncclSuccess;
  msg.size = 0;
  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclRemoteError);
  msg.size = static_cast<uint64_t>(INT_MAX) + 1;
  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclRemoteError);
  msg.size = 1;
  msg.opFlags = 1;
  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclRemoteError);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRespondGetData_MissingStagingBufferIsRejectedAfterResolvingMr) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 8)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.size = 1;
  receiver.controlMsg.mrId = 8;

  EXPECT_EQ(ncclRmaSocketProxyHandleRespondGetData(&scene.comm, 0), ncclInternalError);
  EXPECT_EQ(handles[0].recvActiveOps, 1);
  ncclRmaSocketProxyResetRecv(&receiver);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRequestGetData_ValidRequestQueuesReplyAndResetsReceiver) {
  Scene scene;
  std::array<unsigned char, 16> source{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(source.data(), source.size(), NCCL_PTR_HOST, 5)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[1];
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRequestGetData;
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.requestId = 33;
  receiver.controlMsg.mrId = 5;
  receiver.controlMsg.dstOff = 3;
  receiver.controlMsg.size = 4;

  ASSERT_EQ(ncclRmaSocketProxyHandleRequestGetData(&scene.comm, 1), ncclSuccess);
  const auto& sender = scene.senders[1];
  ASSERT_EQ(sender.count, 1);
  const auto& task = sender.tasks[sender.head];
  EXPECT_EQ(task.controlMsg.type, ncclRmaSocketProxyMsgTypeRespondGetData);
  EXPECT_EQ(task.controlMsg.requestId, 33u);
  EXPECT_EQ(task.controlMsg.status, ncclSuccess);
  EXPECT_EQ(task.srcPtr, source.data() + 3);
  EXPECT_EQ(task.srcHandle, &handles[0]);
  EXPECT_EQ(task.size, 4u);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRequestGetData_InvalidRequestQueuesErrorReplyWithoutPayload) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRequestGetData;
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.requestId = 9;
  receiver.controlMsg.mrId = 404;
  receiver.controlMsg.size = 4;

  ASSERT_EQ(ncclRmaSocketProxyHandleRequestGetData(&scene.comm, 0), ncclSuccess);
  const auto& task = scene.senders[0].tasks[0];
  EXPECT_EQ(task.controlMsg.status, ncclRemoteError);
  EXPECT_EQ(task.size, 0u);
  EXPECT_EQ(task.srcPtr, nullptr);
  EXPECT_EQ(task.srcHandle, nullptr);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRequestGetData_InvalidWireFieldsQueueErrorReply) {
  Scene scene;
  std::array<unsigned char, 8> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 5)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRequestGetData;
  receiver.controlMsg.status = ncclRemoteError;
  receiver.controlMsg.requestId = 10;
  receiver.controlMsg.mrId = 5;
  receiver.controlMsg.size = 1;

  ASSERT_EQ(ncclRmaSocketProxyHandleRequestGetData(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(scene.senders[0].tasks[0].controlMsg.status, ncclRemoteError);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRequestGetData_InvalidSizeOrFlagsQueueErrorReply) {
  auto expectErrorReply = [](uint64_t size, uint32_t opFlags) {
    Scene scene;
    std::array<unsigned char, 8> data{};
    const size_t advertisedSize =
        size > static_cast<uint64_t>(INT_MAX) ? static_cast<size_t>(INT_MAX) + 2 : data.size();
    std::vector<ncclRmaSocketProxyMrHandle> handles{
        scene.MakeMr(data.data(), advertisedSize, NCCL_PTR_HOST, 5)};
    scene.LinkMrs(handles);
    auto& receiver = scene.receivers[0];
    receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRequestGetData;
    receiver.controlMsg.status = ncclSuccess;
    receiver.controlMsg.requestId = 10;
    receiver.controlMsg.mrId = 5;
    receiver.controlMsg.size = size;
    receiver.controlMsg.opFlags = opFlags;

    ASSERT_EQ(ncclRmaSocketProxyHandleRequestGetData(&scene.comm, 0), ncclSuccess);
    EXPECT_EQ(scene.senders[0].tasks[0].controlMsg.status, ncclRemoteError);
    EXPECT_EQ(scene.senders[0].tasks[0].size, 0u);
  };

  expectErrorReply(/*size=*/0, /*opFlags=*/0);
  expectErrorReply(/*size=*/static_cast<uint64_t>(INT_MAX) + 1, /*opFlags=*/0);
  expectErrorReply(/*size=*/1, /*opFlags=*/1);
}

TEST_F(RmaSocketProgressRecverMicrotest, HandleRequestGetData_OutOfRangeSourceQueuesErrorReply) {
  Scene scene;
  std::array<unsigned char, 4> data{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 5)};
  scene.LinkMrs(handles);
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRequestGetData;
  receiver.controlMsg.status = ncclSuccess;
  receiver.controlMsg.requestId = 11;
  receiver.controlMsg.mrId = 5;
  receiver.controlMsg.dstOff = data.size();
  receiver.controlMsg.size = 1;

  ASSERT_EQ(ncclRmaSocketProxyHandleRequestGetData(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(scene.senders[0].tasks[0].controlMsg.status, ncclRemoteError);
  EXPECT_EQ(scene.senders[0].tasks[0].size, 0u);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressRecvControl_PartialMessageIsInProgressAndCleanCloseMarksPeerClosed) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  enum class Action { Partial, Closed };
  Action action = Action::Partial;
  ScopedHook progress(g_ncclSocketProgress,
                      [&action](int op, ncclSocket*, void*, int size, int* offset, bool* closed) {
                        EXPECT_EQ(op, NCCL_SOCKET_RECV);
                        if (action == Action::Partial) {
                          *offset = size - 1;
                          *closed = false;
                        } else {
                          *offset = 0;
                          *closed = true;
                        }
                        return ncclSuccess;
                      });

  EXPECT_EQ(ncclRmaSocketProxyProgressRecvControl(&scene.comm, 0), ncclInProgress);
  action = Action::Closed;
  receiver.controlMsgOffset = 0;
  EXPECT_EQ(ncclRmaSocketProxyProgressRecvControl(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateClosed);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressRecvControl_CloseAfterPartialMessageReturnsRemoteError) {
  Scene scene;
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void*, int, int* offset, bool* closed) {
    *offset = 1;
    *closed = true;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressRecvControl(&scene.comm, 0), ncclRemoteError);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressRecvPayload_CompleteHostChunkCopiesDataAndResetsReceiver) {
  Scene scene;
  std::array<unsigned char, 8> destination{};
  std::array<unsigned char, 8> staging{};
  auto& receiver = scene.receivers[0];
  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  receiver.controlMsg.requestId = 55;
  receiver.controlMsg.size = destination.size();
  receiver.buffer = staging.data();
  receiver.dataPtr = destination.data();
  receiver.dataType = NCCL_PTR_HOST;
  ScopedHook progress(g_ncclSocketProgress, [](int op, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
    EXPECT_EQ(op, NCCL_SOCKET_RECV);
    std::memset(ptr, 0x5a, size);
    *offset = size;
    *closed = false;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclRmaSocketProxyProgressRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(destination, (std::array<unsigned char, 8>{{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a}}));
  EXPECT_EQ(receiver.lastCompletedRequestGlobalId, 55u);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressRecvPayload_MultiChunkDataAdvancesDestinationAndCompletes) {
  Scene scene;
  const size_t payloadSize = NCCL_RMA_SOCKET_CHUNK_SIZE + 3;
  std::vector<unsigned char> destination(payloadSize);
  std::vector<unsigned char> staging(NCCL_RMA_SOCKET_CHUNK_SIZE);
  auto& receiver = scene.receivers[0];
  receiver.recvState = ncclRmaSocketProxyRecvStatePayload;
  receiver.controlMsg.type = ncclRmaSocketProxyMsgTypeRespondGetData;
  receiver.controlMsg.requestId = 56;
  receiver.controlMsg.size = payloadSize;
  receiver.buffer = staging.data();
  receiver.dataPtr = destination.data();
  receiver.dataType = NCCL_PTR_HOST;
  int chunks = 0;
  ScopedHook progress(g_ncclSocketProgress,
                      [&chunks](int op, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
                        EXPECT_EQ(op, NCCL_SOCKET_RECV);
                        std::memset(ptr, chunks++ == 0 ? 0x11 : 0x22, size);
                        *offset = size;
                        *closed = false;
                        return ncclSuccess;
                      });

  ASSERT_EQ(ncclRmaSocketProxyProgressRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStatePayload);
  EXPECT_EQ(receiver.recvPayloadOffset, static_cast<size_t>(NCCL_RMA_SOCKET_CHUNK_SIZE));
  EXPECT_EQ(destination.front(), 0x11);
  EXPECT_EQ(destination[NCCL_RMA_SOCKET_CHUNK_SIZE - 1], 0x11);
  EXPECT_EQ(destination[NCCL_RMA_SOCKET_CHUNK_SIZE], 0);

  ASSERT_EQ(ncclRmaSocketProxyProgressRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
  EXPECT_EQ(receiver.lastCompletedRequestGlobalId, 56u);
  EXPECT_EQ(destination[NCCL_RMA_SOCKET_CHUNK_SIZE], 0x22);
  EXPECT_EQ(destination.back(), 0x22);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressRecvPayload_PartialOrClosedSocketDoesNotApplyData) {
  Scene scene;
  std::array<unsigned char, 4> destination{};
  destination.fill(0x7a);
  const auto unchanged = destination;
  std::array<unsigned char, 4> staging{};
  auto& receiver = scene.receivers[0];
  receiver.controlMsg.size = destination.size();
  receiver.buffer = staging.data();
  receiver.dataPtr = destination.data();
  receiver.dataType = NCCL_PTR_HOST;
  bool close = false;
  ScopedHook progress(g_ncclSocketProgress, [&close](int, ncclSocket*, void*, int size, int* offset, bool* closed) {
    *offset = size - 1;
    *closed = close;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressRecvPayload(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.recvPayloadOffset, 0u);
  EXPECT_EQ(destination, unchanged);
  close = true;
  EXPECT_EQ(ncclRmaSocketProxyProgressRecvPayload(&scene.comm, 0), ncclRemoteError);
  EXPECT_EQ(destination, unchanged);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_UnknownMessageReturnsRemoteErrorAndResetsState) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
    auto* msg = static_cast<ncclRmaSocketProxyHeader*>(ptr);
    msg->type = 99;
    *offset = size;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclRemoteError);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
  EXPECT_EQ(receiver.controlMsgOffset, 0);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_PartialControlMessageReturnsSuccessAndKeepsProgress) {
  Scene scene;
  auto& receiver = scene.receivers[0];
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void*, int size, int* offset, bool* closed) {
    *offset = size - 1;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(receiver.controlMsgOffset, static_cast<int>(sizeof(receiver.controlMsg)) - 1);
  EXPECT_EQ(receiver.recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_CleanControlCloseMarksPeerClosed) {
  Scene scene;
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void*, int, int* offset, bool* closed) {
    *offset = 0;
    *closed = true;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(scene.receivers[0].recvState, ncclRmaSocketProxyRecvStateClosed);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_PutControlRunsAckAndCompletesEmptyPut) {
  Scene scene;
  int calls = 0;
  ScopedHook progress(g_ncclSocketProgress,
                      [&calls](int op, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
                        ++calls;
                        *closed = false;
                        *offset = size;
                        if (op == NCCL_SOCKET_RECV) {
                          auto* msg = static_cast<ncclRmaSocketProxyHeader*>(ptr);
                          msg->type = ncclRmaSocketProxyMsgTypeIput;
                          msg->status = ncclSuccess;
                          msg->size = 0;
                        }
                        return ncclSuccess;
                      });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(scene.receivers[0].recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_GetRequestControlQueuesErrorResponse) {
  Scene scene;
  ScopedHook progress(g_ncclSocketProgress, [](int, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
    auto* msg = static_cast<ncclRmaSocketProxyHeader*>(ptr);
    msg->type = ncclRmaSocketProxyMsgTypeRequestGetData;
    msg->status = ncclSuccess;
    msg->requestId = 12;
    msg->mrId = 404;
    msg->size = 1;
    *offset = size;
    *closed = false;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclSuccess);
  ASSERT_EQ(scene.senders[0].count, 1);
  EXPECT_EQ(scene.senders[0].tasks[0].controlMsg.status, ncclRemoteError);
  EXPECT_EQ(scene.receivers[0].recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

TEST_F(RmaSocketProgressRecverMicrotest, ProgressPeerRecv_GetResponseControlReceivesAndPublishesPayload) {
  Scene scene;
  std::array<unsigned char, 4> data{};
  std::array<unsigned char, 4> staging{};
  std::vector<ncclRmaSocketProxyMrHandle> handles{
      scene.MakeMr(data.data(), data.size(), NCCL_PTR_HOST, 6)};
  scene.LinkMrs(handles);
  scene.receivers[0].buffer = staging.data();
  int calls = 0;
  ScopedHook progress(g_ncclSocketProgress,
                      [&calls](int, ncclSocket*, void* ptr, int size, int* offset, bool* closed) {
                        if (calls++ == 0) {
                          auto* msg = static_cast<ncclRmaSocketProxyHeader*>(ptr);
                          msg->type = ncclRmaSocketProxyMsgTypeRespondGetData;
                          msg->status = ncclSuccess;
                          msg->requestId = 13;
                          msg->mrId = 6;
                          msg->size = 4;
                        } else {
                          std::memset(ptr, 0x2a, size);
                        }
                        *offset = size;
                        *closed = false;
                        return ncclSuccess;
                      });

  EXPECT_EQ(ncclRmaSocketProxyProgressPeerRecv(&scene.comm, 0), ncclSuccess);
  EXPECT_EQ(data, (std::array<unsigned char, 4>{{0x2a, 0x2a, 0x2a, 0x2a}}));
  EXPECT_EQ(scene.receivers[0].lastCompletedRequestGlobalId, 13u);
  EXPECT_EQ(scene.receivers[0].recvState, ncclRmaSocketProxyRecvStateControlMsg);
}

}  // namespace
