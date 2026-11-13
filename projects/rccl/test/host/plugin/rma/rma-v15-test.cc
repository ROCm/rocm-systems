/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/plugin/rma/rma_v15.cc, the adapter that presents a v15 RMA plugin as the current
// ncclRma_t: every call is used as exported, and the missing getRmaProperties reports no flush-on-signal.

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

// A distinct function per member, so a swapped assignment is visible.
#define FAKE [](auto...) { return ncclSuccess; }

class RmaV15Microtest : public ::testing::Test {
 protected:
  void SetUp() override {
    plugin_ = {"fake-v15", FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE,
               FAKE,       FAKE, FAKE, FAKE, FAKE, FAKE, FAKE, FAKE};
    g_export = &plugin_;
    g_dlsyms.clear();
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

TEST_F(RmaV15Microtest, AdapterUsesEveryCallAsExported) {
  ncclRma_t* rma = getNcclRma_v15(kLib);
  ASSERT_NE(nullptr, rma);
  EXPECT_STREQ("fake-v15", rma->name);
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
  EXPECT_EQ(plugin_.iput, rma->iput);
  EXPECT_EQ(plugin_.iputSignal, rma->iputSignal);
  EXPECT_EQ(plugin_.iget, rma->iget);
  EXPECT_EQ(plugin_.iflush, rma->iflush);
  EXPECT_EQ(plugin_.test, rma->test);
  EXPECT_EQ(plugin_.rmaProgress, rma->rmaProgress);
  EXPECT_EQ(plugin_.queryLastError, rma->queryLastError);
  EXPECT_EQ(plugin_.finalize, rma->finalize);
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
