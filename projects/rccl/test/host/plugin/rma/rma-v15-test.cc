/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/rma/rma_v15.cc, the v15-to-current ncclRma_t adapter.

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <cstring>
#include <string>
#include <vector>

#include "nccl_rma.h"
#include "proxy.h"

namespace {

void* const kLib = reinterpret_cast<void*>(0x15);
ncclRma_v15_t* g_export = nullptr;
std::vector<std::string> g_dlsyms;

void* FakeDlsym(void* lib, const char* symbol) {
  g_dlsyms.push_back(symbol);
  if (lib != kLib) return nullptr;
  if (!std::strcmp(symbol, "ncclRmaPlugin_v15")) return g_export;
  return nullptr;
}

}  // namespace

// rma_v15.cc calls dlsym directly; its headers are included above so the rename reaches only rma_v15.cc.
#define dlsym FakeDlsym
#include RMA_V15_CC_PATH
#undef dlsym

namespace {

// What the v15 connect and createContext were last given.
struct ControlCall {
  void* ctx = nullptr;
  void** handles = nullptr;
  int nranks = -1, rank = -1;
  void* listenComm = nullptr;
  void** collComm = nullptr;
  ncclRmaConfig_v15_t config = {};
  void** rmaCtx = nullptr;
};
ControlCall g_control;

ncclResult_t V15Connect(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm) {
  g_control.ctx = ctx;
  g_control.handles = handles;
  g_control.nranks = nranks;
  g_control.rank = rank;
  g_control.listenComm = listenComm;
  g_control.collComm = collComm;
  return ncclInProgress;
}

ncclResult_t V15CreateContext(void* collComm, ncclRmaConfig_v15_t* config, void** rmaCtx) {
  g_control.ctx = collComm;
  g_control.config = *config;
  g_control.rmaCtx = rmaCtx;
  return ncclInProgress;
}

// A distinct function per member, so a swapped assignment is visible.
#define FAKE [](auto...) { return ncclSuccess; }

class RmaV15Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    plugin_ = {"fake-v15", FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE,
               FAKE,       FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE};
    plugin_.connect = V15Connect;
    plugin_.createContext = V15CreateContext;
    g_export = &plugin_;
    g_dlsyms.clear();
    g_control = {};
  }
  void TearDown() override { g_export = nullptr; }

  ncclRma_v15_t plugin_;
};

#undef FAKE

TEST_F(RmaV15Microtest, MissingSymbolReturnsNull) {
  g_export = nullptr;
  EXPECT_EQ(nullptr, getNcclRma_v15(kLib));
  EXPECT_EQ(g_dlsyms, std::vector<std::string>{"ncclRmaPlugin_v15"});
}

TEST_F(RmaV15Microtest, AdapterUsesTheOtherCallsAsExported) {
  ncclRma_t* rma = getNcclRma_v15(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_STREQ("fake-v15", rma->name);
  EXPECT_EQ(plugin_.init, rma->init);
  EXPECT_EQ(plugin_.devices, rma->devices);
  EXPECT_EQ(plugin_.getProperties, rma->getProperties);
  EXPECT_EQ(plugin_.listen, rma->listen);
  EXPECT_EQ(plugin_.regMrSym, rma->regMrSym);
  EXPECT_EQ(plugin_.regMrSymDmaBuf, rma->regMrSymDmaBuf);
  EXPECT_EQ(plugin_.deregMrSym, rma->deregMrSym);
  EXPECT_EQ(plugin_.destroyContext, rma->destroyContext);
  EXPECT_EQ(plugin_.closeColl, rma->closeColl);
  EXPECT_EQ(plugin_.closeListen, rma->closeListen);
  EXPECT_EQ(plugin_.iput, rma->iput);
  EXPECT_EQ(plugin_.iputSignal, rma->iputSignal);
  EXPECT_EQ(plugin_.iget, rma->iget);
  EXPECT_EQ(plugin_.iflush, rma->iflush);
  EXPECT_EQ(plugin_.test, rma->test);
  EXPECT_EQ(plugin_.rmaProgress, rma->rmaProgress);
  EXPECT_EQ(plugin_.queryLastError, rma->queryLastError);
  EXPECT_EQ(plugin_.finalize, rma->finalize);
}

// connect gained an abort flag in v17; v15 has none, so the adapter drops it and forwards the rest.
TEST_F(RmaV15Microtest, ConnectForwardsEverythingButTheAbortFlag) {
  ncclRma_t* rma = getNcclRma_v15(kLib);
  ASSERT_NE(nullptr, rma);
  void* handles[4] = {};
  void* const listen = reinterpret_cast<void*>(0x500);
  void* collComm = nullptr;
  volatile uint32_t abortFlag = 1;
  EXPECT_EQ(ncclInProgress,
            rma->connect(reinterpret_cast<void*>(0x100), handles, 4, 2, listen, &collComm, &abortFlag));
  EXPECT_EQ(reinterpret_cast<void*>(0x100), g_control.ctx);
  EXPECT_EQ(handles, g_control.handles);
  EXPECT_EQ(4, g_control.nranks);
  EXPECT_EQ(2, g_control.rank);
  EXPECT_EQ(listen, g_control.listenComm);
  EXPECT_EQ(&collComm, g_control.collComm);
}

// The v17 config carries a peer list instead of a rank stride: the adapter keeps nContexts and trafficClass and
// asks v15 for every rank (rankStride 1).
TEST_F(RmaV15Microtest, CreateContextDownConvertsTheConfig) {
  ncclRma_t* rma = getNcclRma_v15(kLib);
  ASSERT_NE(nullptr, rma);
  int peers[2] = {1, 3};
  ncclRmaConfig_v17_t config = {};
  config.nContexts = 3;
  config.trafficClass = 5;
  config.peerArray = peers;
  void* rmaCtx = nullptr;
  EXPECT_EQ(ncclInProgress, rma->createContext(reinterpret_cast<void*>(0x600), &config, &rmaCtx));
  EXPECT_EQ(reinterpret_cast<void*>(0x600), g_control.ctx);
  EXPECT_EQ(3, g_control.config.nContexts);
  EXPECT_EQ(5, g_control.config.trafficClass);
  EXPECT_EQ(1, g_control.config.rankStride);
  EXPECT_EQ(&rmaCtx, g_control.rmaCtx);
}

TEST_F(RmaV15Microtest, RmaPropertiesReportNoFlushOnSignal) {
  ncclRma_t* rma = getNcclRma_v15(kLib);
  ASSERT_NE(nullptr, rma);
  ncclRmaProperties_t props;
  std::memset(&props, 0xff, sizeof(props));
  EXPECT_EQ(ncclSuccess, rma->getRmaProperties(nullptr, &props));
  EXPECT_FALSE(props.flushesAllPutsOnAnySignal);
}

}  // namespace
