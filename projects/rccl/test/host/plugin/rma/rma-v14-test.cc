/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/rma/rma_v14.cc, the adapter that presents a v14 RMA plugin as the current
// ncclRma_t: iput, iputSignal and iget drop optFlags, the missing getRmaProperties reports no flush-on-signal, and
// every other call is used as exported.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstring>
#include <string>
#include <vector>

#include "checks.h"
#include "nccl_rma.h"
#include "os.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x14);
ncclRma_v14_t* g_export = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclRmaPlugin_v14")) return g_export;
  return nullptr;
}

}  // namespace

// rma_v14.cc calls dlsym directly; its headers are included above so the rename reaches only rma_v14.cc.
#define dlsym FakeDlsym
#include RMA_V14_CC_PATH
#undef dlsym

namespace {

// What the v14 data-path call was last given.
struct Call {
  void* ctx = nullptr;
  int context = -1;
  uint64_t srcOff = 0, dstOff = 0, signalOff = 0, signalValue = 0;
  void* srcMhandle = nullptr;
  void* dstMhandle = nullptr;
  void* signalMhandle = nullptr;
  size_t size = 0;
  uint32_t rank = 0, signalOp = 0;
  bool strong = false;
  void** request = nullptr;
};
Call g_call;
void* const kCtx = reinterpret_cast<void*>(0x100);
void* const kSrc = reinterpret_cast<void*>(0x200);
void* const kDst = reinterpret_cast<void*>(0x300);
void* const kSig = reinterpret_cast<void*>(0x400);

// Records the fields every data-path call shares; the signal fields are set by V14IputSignal.
void RecordTransfer(void* ctx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                    void* dstMhandle, uint32_t rank, void** request) {
  g_call = {};
  g_call.ctx = ctx;
  g_call.context = context;
  g_call.srcOff = srcOff;
  g_call.srcMhandle = srcMhandle;
  g_call.size = size;
  g_call.dstOff = dstOff;
  g_call.dstMhandle = dstMhandle;
  g_call.rank = rank;
  g_call.request = request;
}

ncclResult_t V14Iput(void* ctx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                     void* dstMhandle, uint32_t rank, void** request) {
  RecordTransfer(ctx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
  return ncclInProgress;
}

ncclResult_t V14IputSignal(void* ctx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                           void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle,
                           uint64_t signalValue, uint32_t signalOp, bool isStrongSignal, void** request) {
  RecordTransfer(ctx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
  g_call.signalOff = signalOff;
  g_call.signalMhandle = signalMhandle;
  g_call.signalValue = signalValue;
  g_call.signalOp = signalOp;
  g_call.strong = isStrongSignal;
  return ncclInProgress;
}

// remote maps to src and local to dst.
ncclResult_t V14Iget(void* ctx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size, uint64_t localOff,
                     void* localMhandle, uint32_t rank, void** request) {
  RecordTransfer(ctx, context, remoteOff, remoteMhandle, size, localOff, localMhandle, rank, request);
  return ncclInProgress;
}

// A distinct function per member, so a swapped assignment is visible.
#define FAKE [](auto...) { return ncclSuccess; }

class RmaV14Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    plugin_ = {"fake-v14", FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE,
               FAKE,       FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE};
    plugin_.iput = V14Iput;
    plugin_.iputSignal = V14IputSignal;
    plugin_.iget = V14Iget;
    g_export = &plugin_;
    g_dlsyms.clear();
    g_call = {};
  }
  void TearDown() override { g_export = nullptr; }

  ncclRma_v14_t plugin_;
  void* request_ = nullptr;
};

#undef FAKE

TEST_F(RmaV14Microtest, MissingSymbolReturnsNull) {
  g_export = nullptr;
  EXPECT_EQ(nullptr, getNcclRma_v14(kLib));
  EXPECT_EQ(g_dlsyms, std::vector<std::string>{"ncclRmaPlugin_v14"});
}

TEST_F(RmaV14Microtest, AdapterUsesControlCallsAsExported) {
  ncclRma_t* rma = getNcclRma_v14(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_STREQ("fake-v14", rma->name);
  EXPECT_EQ(plugin_.init, rma->init);
  EXPECT_EQ(plugin_.devices, rma->devices);
  EXPECT_EQ(plugin_.getProperties, rma->getProperties);
  EXPECT_EQ(plugin_.listen, rma->listen);
  EXPECT_EQ(plugin_.connect, rma->connect);
  EXPECT_EQ(plugin_.createContext, rma->createContext);
  EXPECT_EQ(plugin_.regMrSym, rma->regMrSym);
  EXPECT_EQ(plugin_.regMrSymDmaBuf, rma->regMrSymDmaBuf);
  EXPECT_EQ(plugin_.deregMrSym, rma->deregMrSym);
  EXPECT_EQ(plugin_.destroyContext, rma->destroyContext);
  EXPECT_EQ(plugin_.closeColl, rma->closeColl);
  EXPECT_EQ(plugin_.closeListen, rma->closeListen);
  EXPECT_EQ(plugin_.iflush, rma->iflush);
  EXPECT_EQ(plugin_.test, rma->test);
  EXPECT_EQ(plugin_.rmaProgress, rma->rmaProgress);
  EXPECT_EQ(plugin_.queryLastError, rma->queryLastError);
  EXPECT_EQ(plugin_.finalize, rma->finalize);
}

TEST_F(RmaV14Microtest, RmaPropertiesReportNoFlushOnSignal) {
  ncclRma_t* rma = getNcclRma_v14(kLib);
  ASSERT_NE(nullptr, rma);
  ncclRmaProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  EXPECT_EQ(ncclSuccess, rma->getRmaProperties(nullptr, &props));
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

TEST_F(RmaV14Microtest, IputForwardsEverythingButOptFlags) {
  ncclRma_t* rma = getNcclRma_v14(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_EQ(ncclInProgress, rma->iput(kCtx, 3, 10, kSrc, 64, 20, kDst, 5, /*optFlags=*/0xff, &request_));
  EXPECT_EQ(kCtx, g_call.ctx);
  EXPECT_EQ(3, g_call.context);
  EXPECT_EQ(10u, g_call.srcOff);
  EXPECT_EQ(kSrc, g_call.srcMhandle);
  EXPECT_EQ(64u, g_call.size);
  EXPECT_EQ(20u, g_call.dstOff);
  EXPECT_EQ(kDst, g_call.dstMhandle);
  EXPECT_EQ(5u, g_call.rank);
  EXPECT_EQ(&request_, g_call.request);
}

TEST_F(RmaV14Microtest, IputSignalForwardsEverythingButOptFlags) {
  ncclRma_t* rma = getNcclRma_v14(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_EQ(ncclInProgress, rma->iputSignal(kCtx, 3, 10, kSrc, 64, 20, kDst, 5, 30, kSig, 7, 2,
                                            /*isStrongSignal=*/true, /*optFlags=*/0xff, &request_));
  EXPECT_EQ(kCtx, g_call.ctx);
  EXPECT_EQ(3, g_call.context);
  EXPECT_EQ(10u, g_call.srcOff);
  EXPECT_EQ(kSrc, g_call.srcMhandle);
  EXPECT_EQ(64u, g_call.size);
  EXPECT_EQ(20u, g_call.dstOff);
  EXPECT_EQ(kDst, g_call.dstMhandle);
  EXPECT_EQ(5u, g_call.rank);
  EXPECT_EQ(30u, g_call.signalOff);
  EXPECT_EQ(kSig, g_call.signalMhandle);
  EXPECT_EQ(7u, g_call.signalValue);
  EXPECT_EQ(2u, g_call.signalOp);
  EXPECT_TRUE(g_call.strong);
  EXPECT_EQ(&request_, g_call.request);
  EXPECT_EQ(ncclInProgress, rma->iputSignal(kCtx, 3, 10, kSrc, 64, 20, kDst, 5, 30, kSig, 7, 2,
                                            /*isStrongSignal=*/false, 0, &request_));
  EXPECT_FALSE(g_call.strong);
}

TEST_F(RmaV14Microtest, IgetForwardsEverythingButOptFlags) {
  ncclRma_t* rma = getNcclRma_v14(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_EQ(ncclInProgress, rma->iget(kCtx, 3, 10, kSrc, 64, 20, kDst, 5, /*optFlags=*/0xff, &request_));
  EXPECT_EQ(kCtx, g_call.ctx);
  EXPECT_EQ(3, g_call.context);
  EXPECT_EQ(10u, g_call.srcOff);
  EXPECT_EQ(kSrc, g_call.srcMhandle);
  EXPECT_EQ(64u, g_call.size);
  EXPECT_EQ(20u, g_call.dstOff);
  EXPECT_EQ(kDst, g_call.dstMhandle);
  EXPECT_EQ(5u, g_call.rank);
  EXPECT_EQ(&request_, g_call.request);
}

}  // namespace
