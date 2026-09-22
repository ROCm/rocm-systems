/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for ncclIbProbeGdrSupport (src/misc/gdr_probe.cc), the
// runtime fallback the net_ib and net_ib_cast transports use when the sysfs
// memory_peers scan finds no peer-memory client.
//
// Contract: given an ibv_context and a relaxed-ordering setting, the probe
// returns 1 iff a GPU buffer registration actually succeeds, registering with
// the same entry point and access flags production registration (reg.cc) would
// use for that setting, and releases exactly what it acquired -- never a
// resource it failed to obtain -- whichever step fails.
//
// The suite compiles the hipified gdr_probe.cc directly (via GDR_PROBE_CC_PATH)
// and drives hipMalloc/hipFree through fakes/hip_fakes and the verbs wrappers
// through fakes/ibvwrap_fakes. No GPU, no NIC, no librccl.so.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "fakes/hip_fakes.h"
#include "fakes/ibvwrap_fakes.h"
#include "ScopedHook.h"

#include GDR_PROBE_CC_PATH

namespace {

// The base access flags of production registration (transport/net_ib/reg.cc and
// transport/net_ib_cast/reg.cc). test/host/CMakeLists.txt checks at configure time
// that both reg.cc files and this line still spell the identical set, so the
// probe is pinned to production rather than to a private copy.
constexpr int kRegAccessFlags =
    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;

struct RegCall {
  struct ibv_pd* pd;
  void* addr;
  size_t length;
  uint64_t iova;
  int access;
};

class GdrProbeTest : public ::testing::Test {
 protected:
  // Per-test outcome knobs for each dependency; every step succeeds by default.
  hipError_t mallocResult_ = hipSuccess;
  ncclResult_t allocPdResult_ = ncclSuccess;
  bool regSucceeds_ = true;
  ncclResult_t deregResult_ = ncclSuccess;
  ncclResult_t deallocPdResult_ = ncclSuccess;

  // Stand-ins for the device buffer, the verbs context and the objects the
  // verbs layer would hand back.
  alignas(4096) unsigned char gpuMem_[4096] = {};
  struct ibv_context ctx_ = {};
  struct ibv_pd pd_ = {};
  struct ibv_mr mr_ = {};

  // What the probe did to its dependencies.
  std::vector<struct ibv_context*> allocPdContexts_;
  std::vector<RegCall> regCalls_;
  std::vector<RegCall> regIova2Calls_;
  std::vector<struct ibv_mr*> deregged_;
  std::vector<struct ibv_pd*> deallocated_;
  std::vector<void*> freed_;
  std::vector<std::string> releases_;  // release operations, in the order issued

  std::optional<ScopedHook<hipError_t(void**, std::size_t)>> hipMalloc_;
  std::optional<ScopedHook<hipError_t(void*)>> hipFree_;
  std::optional<ScopedHook<ncclResult_t(struct ibv_pd**, struct ibv_context*)>> allocPd_;
  std::optional<ScopedHook<ncclResult_t(struct ibv_pd*)>> deallocPd_;
  std::optional<ScopedHook<struct ibv_mr*(struct ibv_pd*, void*, size_t, int)>> regMr_;
  std::optional<ScopedHook<struct ibv_mr*(struct ibv_pd*, void*, size_t, uint64_t, int)>> regMrIova2_;
  std::optional<ScopedHook<ncclResult_t(struct ibv_mr*)>> deregMr_;

  void SetUp() override {
    hipMalloc_.emplace(g_hipMalloc, [this](void** ptr, std::size_t) {
      *ptr = (mallocResult_ == hipSuccess) ? gpuMem_ : nullptr;
      return mallocResult_;
    });
    hipFree_.emplace(g_hipFree, [this](void* ptr) {
      freed_.push_back(ptr);
      releases_.push_back("hipFree");
      return hipSuccess;
    });
    allocPd_.emplace(g_ibvAllocPd, [this](struct ibv_pd** ret, struct ibv_context* context) {
      allocPdContexts_.push_back(context);
      *ret = (allocPdResult_ == ncclSuccess) ? &pd_ : nullptr;
      return allocPdResult_;
    });
    deallocPd_.emplace(g_ibvDeallocPd, [this](struct ibv_pd* pd) {
      deallocated_.push_back(pd);
      releases_.push_back("dealloc_pd");
      return deallocPdResult_;
    });
    regMr_.emplace(g_ibvDirectRegMr, [this](struct ibv_pd* pd, void* addr, size_t length, int access) {
      regCalls_.push_back({pd, addr, length, 0, access});
      return regSucceeds_ ? &mr_ : nullptr;
    });
    regMrIova2_.emplace(g_ibvDirectRegMrIova2,
                        [this](struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access) {
                          regIova2Calls_.push_back({pd, addr, length, iova, access});
                          return regSucceeds_ ? &mr_ : nullptr;
                        });
    deregMr_.emplace(g_ibvDeregMr, [this](struct ibv_mr* mr) {
      deregged_.push_back(mr);
      releases_.push_back("dereg_mr");
      return deregResult_;
    });
  }

  void TearDown() override {
    hipMalloc_.reset();
    hipFree_.reset();
    allocPd_.reset();
    deallocPd_.reset();
    regMr_.reset();
    regMrIova2_.reset();
    deregMr_.reset();
    ResetIbvwrapFakes();
  }

  size_t ReleaseIndex(const std::string& op) const {
    return std::find(releases_.begin(), releases_.end(), op) - releases_.begin();
  }

  // Every resource the probe acquired in a run where all acquisitions succeed
  // is released exactly once.
  void ExpectAllReleasedOnce() {
    EXPECT_EQ(deregged_, std::vector<struct ibv_mr*>{&mr_});
    EXPECT_EQ(deallocated_, std::vector<struct ibv_pd*>{&pd_});
    EXPECT_EQ(freed_, std::vector<void*>{gpuMem_});
  }
};

// Default (strict-ordering) arm: production reg.cc registers through plain
// ibv_reg_mr with the base flags, so a successful probe must have exercised that
// same call, on the device buffer, against a PD of the device it was handed.
TEST_F(GdrProbeTest, RelaxedOrderingOff_RegistersLikeProductionAndReportsSupported) {
  EXPECT_EQ(1, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/0));

  EXPECT_EQ(allocPdContexts_, std::vector<struct ibv_context*>{&ctx_});
  ASSERT_EQ(1u, regCalls_.size());
  EXPECT_EQ(&pd_, regCalls_[0].pd);
  EXPECT_EQ(static_cast<void*>(gpuMem_), regCalls_[0].addr);
  EXPECT_EQ(kRegAccessFlags, regCalls_[0].access);
  EXPECT_TRUE(regIova2Calls_.empty());
  ExpectAllReleasedOnce();
}

// Relaxed-ordering arm: production switches to ibv_reg_mr_iova2 (iova == addr)
// and adds IBV_ACCESS_RELAXED_ORDERING. A probe that registered any other way
// could succeed where production registration then fails, which is the flag
// mismatch this probe was once shipped with.
TEST_F(GdrProbeTest, RelaxedOrderingOn_RegistersLikeProductionAndReportsSupported) {
  EXPECT_EQ(1, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/1));

  EXPECT_EQ(allocPdContexts_, std::vector<struct ibv_context*>{&ctx_});
  ASSERT_EQ(1u, regIova2Calls_.size());
  EXPECT_EQ(&pd_, regIova2Calls_[0].pd);
  EXPECT_EQ(static_cast<void*>(gpuMem_), regIova2Calls_[0].addr);
  EXPECT_EQ(reinterpret_cast<uint64_t>(gpuMem_), regIova2Calls_[0].iova);
  EXPECT_EQ(kRegAccessFlags | IBV_ACCESS_RELAXED_ORDERING, regIova2Calls_[0].access);
  EXPECT_TRUE(regCalls_.empty());
  ExpectAllReleasedOnce();
}

// A rejected registration is the "no working peer memory" answer the probe
// exists to give. It must report 0 and still release the PD and buffer, with no
// deregistration of an MR that never existed.
TEST_F(GdrProbeTest, RegistrationRejected_ReportsUnsupportedAndReleasesPdAndBuffer) {
  regSucceeds_ = false;

  EXPECT_EQ(0, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/0));

  EXPECT_TRUE(deregged_.empty());
  EXPECT_EQ(deallocated_, std::vector<struct ibv_pd*>{&pd_});
  EXPECT_EQ(freed_, std::vector<void*>{gpuMem_});
}

// Same guarantee on the relaxed-ordering arm, which reaches the result through
// a different registration entry point.
TEST_F(GdrProbeTest, RelaxedOrderingRegistrationRejected_ReportsUnsupportedAndReleasesPdAndBuffer) {
  regSucceeds_ = false;

  EXPECT_EQ(0, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/1));

  EXPECT_TRUE(deregged_.empty());
  EXPECT_EQ(deallocated_, std::vector<struct ibv_pd*>{&pd_});
  EXPECT_EQ(freed_, std::vector<void*>{gpuMem_});
}

// No device buffer means there is nothing to register: report 0 without
// touching the verbs layer, and free nothing, since nothing was acquired.
TEST_F(GdrProbeTest, DeviceAllocFails_ReportsUnsupportedAndAcquiresNothing) {
  mallocResult_ = hipErrorOutOfMemory;

  EXPECT_EQ(0, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/1));

  EXPECT_TRUE(allocPdContexts_.empty());
  EXPECT_TRUE(regCalls_.empty());
  EXPECT_TRUE(regIova2Calls_.empty());
  EXPECT_TRUE(releases_.empty());
}

// Partial acquisition: the buffer exists but the PD does not. The probe must not
// attempt registration or deallocate the PD it never got, but must still free the
// buffer rather than leak device memory on every failed probe.
TEST_F(GdrProbeTest, PdAllocFails_ReportsUnsupportedAndFreesOnlyTheBuffer) {
  allocPdResult_ = ncclSystemError;

  EXPECT_EQ(0, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/1));

  EXPECT_TRUE(regCalls_.empty());
  EXPECT_TRUE(regIova2Calls_.empty());
  EXPECT_TRUE(deregged_.empty());
  EXPECT_TRUE(deallocated_.empty());
  EXPECT_EQ(freed_, std::vector<void*>{gpuMem_});
}

// The registration already proved GDR works; a failure tearing the MR down is
// logged, not turned into a false "unsupported", and must not skip releasing
// the PD and buffer.
TEST_F(GdrProbeTest, DeregFails_StillReportsSupportedAndReleasesPdAndBuffer) {
  deregResult_ = ncclSystemError;

  EXPECT_EQ(1, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/0));

  ExpectAllReleasedOnce();
}

// Likewise for a failed PD teardown: the answer stays 1 and the device buffer is
// still freed.
TEST_F(GdrProbeTest, PdDeallocFails_StillReportsSupportedAndFreesBuffer) {
  deallocPdResult_ = ncclSystemError;

  EXPECT_EQ(1, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/0));

  ExpectAllReleasedOnce();
}

// Resource lifetime: verbs refuses to deallocate a PD that still has an MR on
// it, and freeing device memory under a live registration leaves the NIC mapped
// to freed memory. So the MR must be released before both its PD and its
// buffer. The relative order of the PD and buffer releases is not constrained.
TEST_F(GdrProbeTest, Success_ReleasesMrBeforeItsPdAndBuffer) {
  ASSERT_EQ(1, ncclIbProbeGdrSupport(&ctx_, /*relaxedOrderingEnabled=*/1));

  ASSERT_EQ(3u, releases_.size());
  EXPECT_LT(ReleaseIndex("dereg_mr"), ReleaseIndex("dealloc_pd"));
  EXPECT_LT(ReleaseIndex("dereg_mr"), ReleaseIndex("hipFree"));
}

}  // namespace
