/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/ionicdvwrap.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "ScopedHook.h"
#include "fakes/dl_fakes.h"
#include "fakes/transport_stubs.h"

#include IONICDVWRAP_CC_PATH

namespace {

// The wrappers only forward these, so they are never dereferenced.
ibv_qp* const kQp = reinterpret_cast<ibv_qp*>(0xC0);
ibv_pd* const kPd = reinterpret_cast<ibv_pd*>(0xD0);
ionic_dv_puec_route* const kRoute = reinterpret_cast<ionic_dv_puec_route*>(0xE0);

int g_ret;
int g_calls;

// One wrapper and the fake verb behind it. The fake checks the arguments it was forwarded.
struct WrapCase {
  const char* name;
  void (*install)();
  ncclResult_t (*call)();
};

const WrapCase kCases[] = {
    {"qp_set_gda",
     [] {
       ionicdvSymbols.ionicdv_internal_qp_set_gda = [](ibv_qp* qp, bool send, bool recv) {
         EXPECT_EQ(qp, kQp);
         EXPECT_FALSE(send);
         EXPECT_TRUE(recv);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ionicdv_qp_set_gda(kQp, false, true); }},
    {"pd_set_udma_mask",
     [] {
       ionicdvSymbols.ionicdv_internal_pd_set_udma_mask = [](ibv_pd* pd, uint8_t mask) {
         EXPECT_EQ(pd, kPd);
         EXPECT_EQ(mask, 0x5);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ionicdv_pd_set_udma_mask(kPd, 0x5); }},
    {"qp_set_puec_plane_route",
     [] {
       ionicdvSymbols.ionicdv_internal_qp_set_puec_plane_route = [](ibv_qp* qp, uint8_t plane, ionic_dv_puec_route* r) {
         EXPECT_EQ(qp, kQp);
         EXPECT_EQ(plane, 2);
         EXPECT_EQ(r, kRoute);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ionicdv_qp_set_puec_plane_route(kQp, 2, kRoute); }},
};

class IonicdvwrapMicrotest : public ::testing::TestWithParam<WrapCase> {
 protected:
  void SetUp() override {
    saved_ = ionicdvSymbols;
    std::memset(&ionicdvSymbols, 0, sizeof(ionicdvSymbols));
    g_ret = 0;
    g_calls = 0;
  }
  void TearDown() override { ionicdvSymbols = saved_; }

 private:
  ncclIonicdvSymbols saved_;
};

TEST_P(IonicdvwrapMicrotest, VerbReturnsZero_Succeeds) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IonicdvwrapMicrotest, VerbReturnsErrno_ReturnsSystemError) {
  GetParam().install();
  g_ret = EINVAL;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
}

// Callers read errno to tell an old libionic from a real failure.
TEST_P(IonicdvwrapMicrotest, SymbolNotLoaded_ReturnsSystemErrorWithEopnotsupp) {
  errno = 0;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
  EXPECT_EQ(errno, EOPNOTSUPP);
}

INSTANTIATE_TEST_SUITE_P(Ionicdvwrap, IonicdvwrapMicrotest, ::testing::ValuesIn(kCases),
                         [](const ::testing::TestParamInfo<WrapCase>& info) { return std::string(info.param.name); });

TEST(IonicdvwrapSymbolsMicrotest, NoAinic_SucceedsWithoutLoadingLibionic) {
  ScopedHook noAinic(g_useAinic, [] { return false; });
  ScriptedDl dl;
  EXPECT_EQ(wrap_ionicdv_symbols(), ncclSuccess);
  EXPECT_TRUE(dl.opens.empty());
}

// Builds once per process, so runs isolated. Relies on ionicdvsymbols-test.cc's dl_seam.h build.
TEST(IonicdvwrapSymbolsMicrotestIsolated, Ainic_BuildsOnceAndCachesFailure) {
  RUN_ISOLATED_TEST("IonicdvwrapSymbolsMicrotestIsolated_Ainic_BuildsOnceAndCachesFailure", []() {
    ScopedHook ainic(g_useAinic, [] { return true; });
    ScriptedDl dl;
    EXPECT_EQ(wrap_ionicdv_symbols(), ncclSystemError);
    size_t opens = dl.opens.size();
    EXPECT_GT(opens, 0u);
    EXPECT_EQ(wrap_ionicdv_symbols(), ncclSystemError);
    EXPECT_EQ(dl.opens.size(), opens);
  });
}

TEST(IonicdvwrapSymbolsMicrotestIsolated, Ainic_LoadsLibionicSymbols) {
  RUN_ISOLATED_TEST("IonicdvwrapSymbolsMicrotestIsolated_Ainic_LoadsLibionicSymbols", []() {
    ScopedHook ainic(g_useAinic, [] { return true; });
    ScriptedDl dl;
    void* lib = dl.AddLibrary("libionic.so");
    dl.Export(lib, "ionic_dv_pd_set_udma_mask", "IONIC_1.0", reinterpret_cast<void*>(0x1));
    EXPECT_EQ(wrap_ionicdv_symbols(), ncclSuccess);
    EXPECT_EQ(reinterpret_cast<void*>(ionicdvSymbols.ionicdv_internal_pd_set_udma_mask),
              reinterpret_cast<void*>(0x1));
  });
}

}  // namespace
