/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for ncclRmaIbProxyTest() (src/transport/net_ib/gin.cc), which polls the
// CQ behind the GIN proxy and host RMA on IB/RoCE. On a failed completion it logs the peer GIDs
// and returns ncclRemoteError. Before NCCL 2.32.3 (b4a53400, "gin/proxy: fix latent bugs") the
// log took the completion's index in the polled batch as a device index: req->devBases[i] and
// remDevs[i]. A request on a single-port NIC only has devBases[0], so a failure behind a
// successful completion dereferenced NULL in the proxy thread.
//
// The suite compiles the hipified gin.cc directly (via NET_IB_GIN_CC_PATH) and calls only
// ncclRmaIbProxyTest(). The CQ is a fake ibv_context whose poll_cq returns a scripted batch:
// no NIC, no GPU, no librccl.so.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include NET_IB_GIN_CC_PATH

static std::string gWarnLog;
uint64_t ncclDebugMask = 0;
uint32_t ncclDebugLevelMask = 0; // debug.h from 2.32
int ncclDebugLevel = 0;          // debug.h before 2.32
thread_local int ncclDebugNoWarn = 0;
const char* ncclIbReqTypeStr[] = {"Unused", "Send", "Recv", "Flush", "IPut", "IGet", "Failed"};
void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  if (level == NCCL_LOG_WARN) gWarnLog += buf;
}
ncclResult_t ncclIbFreeRequest(struct ncclIbRequest* r) {
  r->type = NCCL_NET_IB_REQ_UNUSED;
  return ncclSuccess;
}
ncclResult_t ncclSocketGetAddr(struct ncclSocket*, union ncclSocketAddress* addr) {
  memset(addr, 0, sizeof(*addr));
  return ncclSuccess;
}
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, const int) { return strcpy(buf, "peer"); }

// The rest of gin.cc's externs. Nothing below runs in these tests, but coverage and sanitizer
// builds keep all of gin.cc live, so every symbol it references needs a definition.
ncclResult_t ncclIbAcceptImpl(void*, void**, ncclNetDeviceHandle_t**, int) { return ncclInternalError; }
ncclResult_t ncclIbConnectImpl(void*, int, void*, void**, ncclNetDeviceHandle_t**, int, int) {
  return ncclInternalError;
}
void ncclIbAddEvent(struct ncclIbRequest*, int) {}
ncclResult_t ncclIbCloseListen(void*) { return ncclInternalError; }
ncclResult_t ncclIbCreateFlushQp(struct ncclIbRecvComm*) { return ncclInternalError; }
ncclResult_t ncclIbDevices(int*) { return ncclInternalError; }
ncclResult_t ncclIbDmaBufSupport(int) { return ncclInternalError; }
ncclResult_t ncclIbFinalizeDevices(void) { return ncclInternalError; }
ncclResult_t ncclIbGdrSupport() { return ncclInternalError; }
ncclResult_t ncclIbGetGidIndex(struct ibv_context*, uint8_t, struct ibv_port_attr*, int*) { return ncclInternalError; }
ncclResult_t ncclIbGetProperties(int, ncclNetProperties_t*) { return ncclInternalError; }
ncclResult_t ncclIbGetRequest(struct ncclIbNetCommBase*, struct ncclIbRequest**) { return ncclInternalError; }
ncclResult_t ncclIbInitDevices(ncclDebugLogger_t, ncclProfilerCallback_t) { return ncclInternalError; }
ncclResult_t ncclIbListen(void*, int, void*, void**) { return ncclInternalError; }
ncclResult_t ncclIbPeerMemSupport() { return ncclInternalError; }
ncclResult_t ncclIbRegMrDmaBufInternal(void*, void*, size_t, int, uint64_t, int, uint64_t, void**) {
  return ncclInternalError;
}
void ncclIbSetTrafficClass(void*, int) {}
int64_t ncclLoadParam(char const*, int64_t deftVal, int64_t, int64_t*, int8_t*) { return deftVal; }
int64_t ncclParamIbTc() { return -1; }
size_t ncclOsGetPageSize() { return 4096; }
struct ncclIbDev ncclIbDevs[MAX_IB_DEVS];
int ncclNIbDevs = 0;
ncclNet_t ncclNetIb = {};

namespace {

std::vector<ibv_wc> gBatch;
int fakePollCq(ibv_cq*, int n, ibv_wc* wc) {
  int k = std::min<int>(n, gBatch.size());
  std::copy(gBatch.begin(), gBatch.begin() + k, wc);
  gBatch.erase(gBatch.begin(), gBatch.begin() + k);
  return k;
}

ibv_wc completion(uint64_t reqSlot, ibv_wc_status status) {
  ibv_wc wc = {};
  wc.wr_id = reqSlot;
  wc.status = status;
  return wc;
}

class RmaIbProxyTest : public ::testing::Test {
protected:
  void SetUp() override {
    strcpy(dev.name, "rocep1s0");
    ctx.device = &dev;
    ctx.ops.poll_cq = fakePollCq;
    pd.context = cq.context = &ctx;
    ncclIbNetCommDevBase* devBase = &comm->devs[0].base;
    devBase->pd = &pd;
    devBase->cq = &cq;
    devBase->gidInfo.link_layer = IBV_LINK_LAYER_ETHERNET;
    inet_pton(AF_INET6, "fe80::1", &devBase->gidInfo.localGid);
    inet_pton(AF_INET6, "fe80::2", &comm->base.remDevs[0].remoteGid);
    sendComms[0] = comm.get();
    proxyCtx.fullSendComm = sendComms;
    // An IPut and an IGet in flight, as ncclRmaIbProxyIPut/IGet leave them on a single-port NIC:
    // ncclIbGetRequest() zeroes devBases[] and only devBases[0] is set.
    for (int slot = 0; slot < 2; slot++) {
      ncclIbRequest* req = &comm->base.reqs[slot];
      req->base = &comm->base;
      req->type = slot ? NCCL_NET_IB_REQ_GIN_IGET : NCCL_NET_IB_REQ_GIN_IPUT;
      req->sock = &comm->base.sock;
      req->rmaProxyCtx = &proxyCtx;
      req->id = slot + 1;
      req->events[0] = 1;
      req->devBases[0] = devBase;
    }
    gWarnLog.clear();
    gBatch.clear();
  }

  void expectReportedFailure(ncclIbRequest* req) {
    int done = -1;
    EXPECT_EQ(ncclRemoteError, ncclRmaIbProxyTest(nullptr, req, &done));
    EXPECT_EQ(0, done);
    EXPECT_NE(std::string::npos, gWarnLog.find("status=10 ")) << gWarnLog; // IBV_WC_REM_ACCESS_ERR
    EXPECT_NE(std::string::npos, gWarnLog.find("fe80::1")) << gWarnLog;    // devs[0] local GID
    EXPECT_NE(std::string::npos, gWarnLog.find("fe80::2")) << gWarnLog;    // remDevs[0] remote GID
    EXPECT_NE(std::string::npos, gWarnLog.find("hca rocep1s0")) << gWarnLog;
  }

  ibv_device dev = {};
  ibv_context ctx = {};
  ibv_pd pd = {};
  ibv_cq cq = {};
  std::unique_ptr<ncclIbSendComm> comm{new ncclIbSendComm()};
  void* sendComms[1] = {};
  ncclRmaIbProxyCtx proxyCtx = {};
};

} // namespace

// Control: the failed completion is first in the batch, the one index the old code got right.
TEST_F(RmaIbProxyTest, FailedFirstCompletionIsReported) {
  gBatch = {completion(1, IBV_WC_REM_ACCESS_ERR)};
  expectReportedFailure(&comm->base.reqs[1]);
}

// The failure follows the IPut's successful completion, so it is wc[1] of the batch.
TEST_F(RmaIbProxyTest, FailedCompletionBehindSuccessIsReported) {
  gBatch = {completion(0, IBV_WC_SUCCESS), completion(1, IBV_WC_REM_ACCESS_ERR)};
  expectReportedFailure(&comm->base.reqs[1]);
}
