/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib/gin.cc's ncclGinIbAllGather(),
// the ring allgather the IB RMA/GIN proxy uses to exchange handles and rkeys.
//
// NCCL 2.32.3 (799ca46, "gin: poll completions while waiting for CTS"): the
// allgather spun on isend() until the peer's CTS arrived and tested no request
// before that, so a dropped CTS hung it even after the receive had failed. It
// now polls both requests while it waits and returns the receive's error.
//
// ncclGinIbAllGather is file-static, so this TU #includes the hipified net_ib
// gin.cc (NET_IB_GIN_CC_PATH); a scripted ncclNetIb stands in for the IB NIC.

#include <gtest/gtest.h>

#include "nccl.h"

#include "fakes/nccl_fakes.h"

// gin-plugin-init-test.cc already defines ncclParamGinType() for plugin/gin.cc;
// rename the copy this gin.cc's NCCL_PARAM emits so both link into one binary.
#define ncclParamGinType netIbGinParamGinType
#include "fakes/param_redirect.h"

#include NET_IB_GIN_CC_PATH

// The rest of net_ib, which gin.cc references but these tests never reach.
// ncclSocketToString comes from ras-test.cc.
ncclNet_t ncclNetIb;
int ncclNIbDevs;
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclNMergedIbDevs;
struct ncclIbMergedDev ncclIbMergedDevs[MAX_IB_VDEVS];
const char* ncclIbReqTypeStr[] = {""};
int64_t ncclParamIbTc() { return -1; }
size_t ncclOsGetPageSize() { return 4096; }
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t, ncclProfilerCallback_t) { return ncclInternalError; }
ncclResult_t ncclIbFinalizeDevices(void) { return ncclSuccess; }
ncclResult_t ncclIbGdrSupport() { return ncclSystemError; }
ncclResult_t ncclIbPeerMemSupport() { return ncclSystemError; }
ncclResult_t ncclIbDmaBufSupport(int) { return ncclSystemError; }
ncclResult_t ncclIbDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
ncclResult_t ncclIbGetProperties(int, ncclNetProperties_t*) { return ncclInternalError; }
ncclResult_t ncclIbListen(void*, int, void*, void**) { return ncclInternalError; }
ncclResult_t ncclIbConnectImpl(void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) { return ncclInternalError; }
ncclResult_t ncclIbAcceptImpl(void*, void**, ncclNetDeviceHandle_t**, int) { return ncclInternalError; }
ncclResult_t ncclIbCloseListen(void*) { return ncclSuccess; }
ncclResult_t ncclIbGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*) { return ncclInternalError; }
void ncclIbSetTrafficClass(void*, int) {}
ncclResult_t ncclIbRegMrDmaBufInternal(void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
  return ncclInternalError;
}
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase*, struct ncclIbRequest**) { return ncclInternalError; }
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest*) { return ncclSuccess; }
void ncclIbAddEvent(struct ncclIbRequest*, int) {}
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm*) { return ncclInternalError; }
ncclResult_t ncclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress*) { return ncclInternalError; }

namespace {

// One rank of the ring. isend() returns a request only once the peer's CTS has
// landed, ctsAfter polls into each step (-1: never); test() completes a request
// on its first poll, or fails a receive with recvResult as an error CQE would.
struct FakeIbNet {
  static constexpr int kMaxSendPolls = 1 << 16;  // past this, the CTS wait would never end
  int ctsAfter = 0;
  ncclResult_t recvResult = ncclSuccess;
  int* base = nullptr;  // the allgather's buffer, one int per rank
  int* recvSlot = nullptr;
  int sendPolls = 0, waited = 0, tests = 0;
  char sendReq = 0, recvReq = 0, mr = 0;
};
FakeIbNet* g_ibNet = nullptr;

ncclResult_t FakeRegMr(void*, void*, size_t, int, void** mhandle) {
  *mhandle = &g_ibNet->mr;
  return ncclSuccess;
}
ncclResult_t FakeDeregMr(void*, void*) { return ncclSuccess; }
ncclResult_t FakeIsend(void*, void*, size_t, int, void*, void*, void** request) {
  *request = nullptr;
  if (++g_ibNet->sendPolls > FakeIbNet::kMaxSendPolls) return ncclSystemError;
  if (g_ibNet->ctsAfter < 0 || g_ibNet->waited++ < g_ibNet->ctsAfter) return ncclSuccess;  // no CTS yet
  g_ibNet->waited = 0;
  *request = &g_ibNet->sendReq;
  return ncclSuccess;
}
ncclResult_t FakeIrecv(void*, int, void** data, size_t*, int*, void**, void**, void** request) {
  g_ibNet->recvSlot = (int*)data[0];
  *request = &g_ibNet->recvReq;
  return ncclSuccess;
}
ncclResult_t FakeTest(void* request, int* done, int*) {
  ++g_ibNet->tests;
  if (request == &g_ibNet->recvReq) {
    if (g_ibNet->recvResult != ncclSuccess) return g_ibNet->recvResult;
    *g_ibNet->recvSlot = 100 + (int)(g_ibNet->recvSlot - g_ibNet->base);  // the block that rank sent
  }
  *done = 1;
  return ncclSuccess;
}

class GinIbAllGatherTest : public ::testing::Test {
 protected:
  FakeIbNet fake_;
  struct ncclGinIbCollComm cComm_ {};

  void SetUp() override {
    g_ibNet = &fake_;
    ncclNetIb.regMr = FakeRegMr;
    ncclNetIb.deregMr = FakeDeregMr;
    ncclNetIb.isend = FakeIsend;
    ncclNetIb.irecv = FakeIrecv;
    ncclNetIb.test = FakeTest;
    cComm_.sendComm = cComm_.recvComm = &fake_;
  }
  void TearDown() override {
    ncclNetIb = ncclNet_t{};
    g_ibNet = nullptr;
  }

  // Rank `rank` contributes 100 + rank; buf receives one int per rank.
  ncclResult_t AllGather(int rank, int nranks, int* buf) {
    cComm_.rank = rank;
    cComm_.nranks = nranks;
    fake_.base = buf;
    int src = 100 + rank;
    return ncclGinIbAllGather(&cComm_, &src, buf, sizeof(int));
  }
};

// 799ca46: the peer's CTS never arrives and the receive has already failed.
// The allgather must poll the receive and return its error; before the fix it
// spun on isend() and tested nothing until both requests were posted.
TEST_F(GinIbAllGatherTest, DroppedCtsReturnsRecvError) {
  fake_.ctsAfter = -1;
  fake_.recvResult = ncclRemoteError;
  int buf[2] = {};
  EXPECT_EQ(AllGather(0, 2, buf), ncclRemoteError);
  EXPECT_GT(fake_.tests, 0) << "no completion was polled while waiting for the CTS";
  EXPECT_LE(fake_.sendPolls, FakeIbNet::kMaxSendPolls) << "isend() was polled until the fake gave up";
}

// Control: a CTS that lands a few polls late still completes a 4-rank ring and
// fills every slot, with or without the fix.
TEST_F(GinIbAllGatherTest, ControlLateCtsCompletesRing) {
  fake_.ctsAfter = 3;
  int buf[4] = {};
  ASSERT_EQ(AllGather(1, 4, buf), ncclSuccess);
  for (int i = 0; i < 4; i++) EXPECT_EQ(buf[i], 100 + i) << "slot " << i;
}

}  // namespace
