/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only checks that the cached LSA CTA search in src/tuning/sym_model.cc matches an uncached ncclSymkLsaModel.

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/devcomm_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#if defined(__HIP_PLATFORM_AMD__)
#include "tdm/tdmCopy.h"  // TDM_TOOLCHAIN_AVAILABLE
#endif

#include SYM_MODEL_CC_PATH
#include SYM_MODEL_LSA_CC_PATH
#include SYM_MODEL_LSA_BASE_CC_PATH
#include SYM_MODEL_LSA_A2A_CC_PATH

// sym_model/gin.cc is not compiled here; the LSA path never reaches it.
ncclResult_t ncclSymkGinModel(struct ncclTuningInput_t*, enum ncclSymkKernelId, size_t, float* timeUs,
                              int* nBlocks) {
  *timeUs = -1.0f;
  *nBlocks = 0;
  return ncclSuccess;
}

namespace {

struct Uncached {
  float timeUs;
  float selectionTimeUs;
  int nBlocks;
};

std::function<int64_t(const char*, int64_t)> Params(int64_t symCtas, int64_t symTmaEnable) {
  return [=](const char* env, int64_t deftVal) -> int64_t {
    std::string e(env);
    if (e == "SYM_CTAS") return symCtas;
    if (e == "SYM_TMA_ENABLE") return symTmaEnable;
    return deftVal;
  };
}

class SymModelCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_ = MakeComm(/*nRanks=*/8, "gfx950", /*commHash=*/0x5a5aull);
  }
  void TearDown() override {
    ResetDevcommFakes();
    ResetNcclFakes();
  }

  static std::unique_ptr<ncclComm> MakeComm(int nRanks, const char* arch, uint64_t commHash) {
    auto comm = std::make_unique<ncclComm>();
    comm->nRanks = nRanks;
    comm->rank = 0;
    comm->commHash = commHash;
    comm->symmetricSupport = true;
    comm->isAllDirectNvlink = true;
    comm->symkState.hasLsaMultimem = false;
    comm->WarpSize = 64;
    comm->archName = const_cast<char*>(arch);
    comm->maxSharedMemOptin = ncclTmaShmemScratchWarpSize() * 16;
    return comm;
  }

  static ncclFunc_t FuncOf(ncclSymkKernelId k) {
    if (ncclSymkARKernelMask() >> k & 1) return ncclFuncAllReduce;
    if (ncclSymkAGKernelMask() >> k & 1) return ncclFuncAllGather;
    return ncclFuncReduceScatter;
  }

  static ncclTuningInput_t Input(ncclComm* comm, ncclSymkKernelId k, size_t nBytes, int inPlace = 0) {
    ncclTuningInput_t in{};
    in.comm = comm;
    in.func = FuncOf(k);
    in.devRedOp = ncclDevSum;
    in.datatype = ncclFloat32;
    in.nBytes = nBytes;
    in.count = nBytes / ncclTypeSize(in.datatype);
    in.countMax = in.count;
    in.nWorks = 1;
    in.winRegType = ncclSymSendRegRecvReg;
    in.symAligned16B = true;
    in.inPlace = inPlace;
    in.minCTAs = 1;
    in.maxCTAs = ncclSymkMaxBlocks;
    return in;
  }

  static Uncached Direct(ncclTuningInput_t in, ncclSymkKernelId k) {
    Uncached u{};
    EXPECT_EQ(ncclSymkLsaModel(&in, k, in.nBytes, &u.timeUs, &u.selectionTimeUs, &u.nBlocks), ncclSuccess);
    return u;
  }

  // Returns false when ncclTuningSymkModelSim rejects the kernel before reaching the CTA search.
  static bool Cached(ncclTuningInput_t in, ncclSymkKernelId k, ncclTuningResult_t* out) {
    *out = NCCL_TUNING_RESULT_INIT;
    out->symKernelId = k;
    EXPECT_EQ(ncclTuningSymkModelSim(&in, out), ncclSuccess);
    return out->valid != 0;
  }

  static void ExpectSame(const ncclTuningResult_t& r, const Uncached& u, const std::string& what) {
    EXPECT_EQ(r.timeUs, u.timeUs) << what;
    EXPECT_EQ(r.selectionTimeUs, u.selectionTimeUs) << what;
    EXPECT_EQ(r.nChannels, u.nBlocks) << what;
  }

  std::unique_ptr<ncclComm> comm_;
};

const ncclSymkKernelId kLsaKernels[] = {
  ncclSymkKernelId_AllReduce_AGxLL_R, ncclSymkKernelId_AllReduce_RSxLD_AGxST, ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_ST,      ncclSymkKernelId_ReduceScatter_LL,      ncclSymkKernelId_ReduceScatter_LD,
};

TEST_F(SymModelCacheTest, MissAndHitMatchUncachedAcrossKernelsAndSizes) {
  ScopedHook params(g_loadParam, Params(0, 0));
  int compared = 0;
  for (ncclSymkKernelId k : kLsaKernels) {
    for (size_t bytes = 1 << 10; bytes <= (size_t(1) << 30); bytes <<= 2) {
      for (int inPlace : {0, 1}) {
        ncclTuningInput_t in = Input(comm_.get(), k, bytes, inPlace);
        Uncached u = Direct(in, k);
        ncclTuningResult_t miss, hit;
        bool valid = Cached(in, k, &miss);
        ASSERT_EQ(Cached(in, k, &hit), valid);
        if (!valid) continue;
        std::string what = std::string(ncclSymkKernelIdToString(k)) + " bytes=" + std::to_string(bytes) +
                           " inPlace=" + std::to_string(inPlace);
        ExpectSame(miss, u, what + " (miss)");
        ExpectSame(hit, u, what + " (hit)");
        ++compared;
      }
    }
  }
  EXPECT_GT(compared, 40);
}

TEST_F(SymModelCacheTest, ChangingAKeyFieldIsNeverServedStale) {
  const ncclSymkKernelId k = ncclSymkKernelId_AllGather_LL;
  const size_t baseBytes = 256 << 10;
  auto otherComm = MakeComm(8, "gfx950", 0x5a5aull);

  struct Mutation {
    std::string name;
    std::function<void(ncclTuningInput_t&)> apply;
    int64_t symCtas;
    bool changesResult;  // proves a stale hit would have been caught
  };
  const std::vector<Mutation> mutations = {
    {"nBytes", [](ncclTuningInput_t& in) { in.nBytes *= 2; in.count *= 2; in.countMax *= 2; }, 0, true},
    {"countMax", [](ncclTuningInput_t& in) { in.countMax *= 2; }, 0, false},
    {"datatype", [](ncclTuningInput_t& in) { in.datatype = ncclBfloat16; in.count *= 2; in.countMax *= 2; }, 0,
     false},
    {"inPlace", [](ncclTuningInput_t& in) { in.inPlace = 1; }, 0, false},
    {"minCTAs", [](ncclTuningInput_t& in) { in.minCTAs = 48; }, 0, false},
    {"maxCTAs", [](ncclTuningInput_t& in) { in.maxCTAs = 2; }, 0, true},
    {"nRanks", [this](ncclTuningInput_t&) { comm_->nRanks = 4; }, 0, true},
    {"comm", [&](ncclTuningInput_t& in) { in.comm = otherComm.get(); }, 0, false},
    {"NCCL_SYM_CTAS", [](ncclTuningInput_t&) {}, 6, true},
  };

  for (const Mutation& m : mutations) {
    comm_->nRanks = 8;
    ncclTuningInput_t base = Input(comm_.get(), k, baseBytes);
    ncclTuningResult_t primed;
    {
      ScopedHook params(g_loadParam, Params(0, 0));
      ASSERT_TRUE(Cached(base, k, &primed)) << m.name;
      ExpectSame(primed, Direct(base, k), m.name + " (primed)");
    }

    ncclTuningInput_t mutated = base;
    m.apply(mutated);
    ScopedHook params(g_loadParam, Params(m.symCtas, 0));
    Uncached fresh = Direct(mutated, k);
    ncclTuningResult_t r;
    ASSERT_TRUE(Cached(mutated, k, &r)) << m.name;
    ExpectSame(r, fresh, m.name);
    if (m.changesResult) {
      EXPECT_TRUE(fresh.nBlocks != primed.nChannels || fresh.timeUs != primed.timeUs)
        << m.name << " no longer changes the model's answer, so it cannot detect a stale entry";
    }
  }
}

TEST_F(SymModelCacheTest, KernelsDoNotShareEntries) {
  ScopedHook params(g_loadParam, Params(0, 0));
  const ncclSymkKernelId a = ncclSymkKernelId_AllGather_LL, b = ncclSymkKernelId_AllGather_ST;
  ncclTuningInput_t in = Input(comm_.get(), a, 512 << 10);
  for (int round = 0; round < 3; ++round) {
    for (ncclSymkKernelId k : {a, b}) {
      ncclTuningResult_t r;
      ASSERT_TRUE(Cached(in, k, &r)) << ncclSymkKernelIdToString(k);
      ExpectSame(r, Direct(in, k), ncclSymkKernelIdToString(k));
    }
  }
}

TEST_F(SymModelCacheTest, TmaKernelsMatchUncachedIncludingForcedMode) {
#if defined(__HIP_PLATFORM_AMD__)
  if (!TDM_TOOLCHAIN_AVAILABLE) GTEST_SKIP() << "SDK lacks the gfx1250 TDM toolchain; no Tma kernels to tune";
#endif
  auto tmaComm = MakeComm(8, "gfx1250", 0x7b7bull);
  tmaComm->minCompCap = 100;
  const ncclSymkKernelId tmaKernels[] = {ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST,
                                         ncclSymkKernelId_AllGather_TmaST, ncclSymkKernelId_ReduceScatter_TmaLD};
  int compared = 0;
  for (int64_t tmaEnable : {1, 2, 1}) {
    ScopedHook params(g_loadParam, Params(0, tmaEnable));
    for (ncclSymkKernelId k : tmaKernels) {
      for (size_t bytes = 64 << 10; bytes <= (size_t(1) << 30); bytes <<= 4) {
        ncclTuningInput_t in = Input(tmaComm.get(), k, bytes);
        Uncached u = Direct(in, k);
        ncclTuningResult_t r;
        if (!Cached(in, k, &r)) continue;
        ExpectSame(r, u, std::string(ncclSymkKernelIdToString(k)) + " bytes=" + std::to_string(bytes) +
                           " SYM_TMA_ENABLE=" + std::to_string(tmaEnable));
        ++compared;
      }
    }
  }
  EXPECT_GT(compared, 0);
}

}  // namespace
