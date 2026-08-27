/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the devcomm backward-compatibility shims. The two
// production .cc files are #included so their file-static filter/copy callbacks
// are directly callable; every external they reach is faked in devcomm_fakes.
#include <gtest/gtest.h>

#include <cstring>
#include <memory>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/devcomm_fakes.h"

#include "comm.h"
#include "dev_runtime.h"
#include "nccl.h"

// The units under test. v22902 is hipified (one line prepended); v22907 is
// copied verbatim, so its hipified line numbers match src/ exactly.
#include DEVCOMM_V22902_CC_PATH
#include DEVCOMM_V22907_CC_PATH

namespace {

// Shared base: heap-allocates the comm (ncclComm has inline channel storage, so
// a stack instance overflows) and restores every devcomm seam after each test.
class DevcommMicrotest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_;

  void SetUp() override {
    // make_unique value-initializes: ncclComm's default ctor is implicit, so the whole object is
    // zeroed first. Do not memset over it -- rmaState.rmaProxyState holds a live thread/mutex/condvar.
    comm_ = std::make_unique<ncclComm>();
    comm_->rank = 0;
    comm_->nRanks = 8;
  }

  void TearDown() override { ResetDevcommFakes(); }
};

}  // namespace

// ---- block: ncclCommPropertiesFilter_v22902 ----
namespace {

// The modern layout the caller passes; the shim writes ginType through the
// v22902 layout instead, so the two ginType offsets must be checked apart.
constexpr std::size_t kPropsSize = sizeof(struct ncclCommProperties);
constexpr std::size_t kDeviceApiOff = offsetof(struct ncclCommProperties, deviceApiSupport);
constexpr std::size_t kModernGinOff = offsetof(struct ncclCommProperties, ginType);
constexpr std::size_t kV22902GinOff = 34;  // offsetof(ncclCommProperties_v22902, ginType), static_asserted in the UUT

static_assert(kDeviceApiOff == 32, "deviceApiSupport moved; the byte-level oracle below needs updating");
static_assert(kModernGinOff == 36, "modern ginType moved; the byte-level oracle below needs updating");
static_assert(kV22902GinOff != kModernGinOff, "the type-pun this block exists for has collapsed");

// Every byte starts at kPoison so a store of NCCL_GIN_TYPE_NONE_v22902 (== 0) is
// distinguishable from no store at all.
constexpr unsigned char kPoison = 0xA5;

class PropertiesFilter22902Test : public DevcommMicrotest {
 protected:
  alignas(alignof(struct ncclCommProperties)) unsigned char raw_[kPropsSize];
  unsigned char before_[kPropsSize];

  struct ncclCommProperties* props() { return reinterpret_cast<struct ncclCommProperties*>(raw_); }

  // Poisons the whole buffer, then writes only the two inputs the block reads.
  void ArmProps(bool deviceApiSupport) {
    std::memset(raw_, kPoison, sizeof(raw_));
    props()->size = kPropsSize;
    props()->rank = 3;
    props()->nRanks = 8;
    props()->deviceApiSupport = deviceApiSupport;
    std::memcpy(before_, raw_, sizeof(raw_));
  }

  // Fails for every byte the block changed other than the ones named.
  void ExpectBytesUnchangedExcept(std::initializer_list<std::size_t> allowed) {
    for (std::size_t i = 0; i < kPropsSize; ++i) {
      bool skip = false;
      for (std::size_t a : allowed) skip = skip || (a == i);
      if (skip) continue;
      EXPECT_EQ(raw_[i], before_[i]) << "byte " << i << " was modified";
    }
  }
};

// Arm: deviceApiSupport true AND lsa team spans the comm -> support survives.
TEST_F(PropertiesFilter22902Test, LsaSpansComm_KeepsDeviceApiSupport) {
  ncclComm_t seen = nullptr;
  ScopedHook teamLsa(g_ncclTeamLsa, [&](ncclComm_t c) {
    seen = c;
    ncclTeam_t t{};
    t.nRanks = 8;
    t.rank = 0;
    t.stride = 1;
    return t;
  });
  comm_->nRanks = 8;
  comm_->rank = 5;  // distinct from nRanks so comparing against the wrong member is visible
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_TRUE(props()->deviceApiSupport);
  EXPECT_EQ(teamLsa.calls, 1);
  EXPECT_EQ(seen, comm_.get());
}

// Arm: deviceApiSupport true but lsa team is a strict subset -> support cleared.
TEST_F(PropertiesFilter22902Test, LsaSmallerThanComm_ClearsDeviceApiSupport) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 4;
    return t;
  });
  comm_->nRanks = 8;
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_FALSE(props()->deviceApiSupport);
  EXPECT_EQ(teamLsa.calls, 1);
}

// Arm: lsa team larger than the comm is still a mismatch (pins ==, not >=).
TEST_F(PropertiesFilter22902Test, LsaLargerThanComm_ClearsDeviceApiSupport) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 16;
    return t;
  });
  comm_->nRanks = 8;
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_FALSE(props()->deviceApiSupport);
  EXPECT_EQ(teamLsa.calls, 1);
}

// Arm: deviceApiSupport already false short-circuits the && -- the team is never queried.
TEST_F(PropertiesFilter22902Test, DeviceApiAlreadyFalse_ShortCircuitsTeamQuery) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 8;  // would satisfy the right operand if it were ever evaluated
    return t;
  });
  comm_->nRanks = 8;
  ArmProps(false);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_FALSE(props()->deviceApiSupport);
  EXPECT_EQ(teamLsa.calls, 0);
}

// Arm: both operands false -> false, and still no team query.
TEST_F(PropertiesFilter22902Test, BothOperandsFalse_StaysFalseWithoutTeamQuery) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 4;
    return t;
  });
  comm_->nRanks = 8;
  ArmProps(false);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_FALSE(props()->deviceApiSupport);
  EXPECT_EQ(teamLsa.calls, 0);
}

// Arm: the ginType store. It goes through the v22902 layout, so exactly one byte
// at offset 34 is zeroed and the modern 4-byte ginType at 36 keeps its poison.
TEST_F(PropertiesFilter22902Test, ZeroesGinTypeByteAtV22902OffsetOnly) {
  comm_->nRanks = 8;
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_EQ(raw_[kV22902GinOff], 0x00) << "v22902 ginType byte not zeroed";
  for (std::size_t i = kModernGinOff; i < kModernGinOff + sizeof(ncclGinType_t); ++i) {
    EXPECT_EQ(raw_[i], kPoison) << "byte " << i << ": the store went through the modern struct layout";
  }
  EXPECT_EQ(raw_[33], kPoison) << "multimemSupport clobbered";
  ExpectBytesUnchangedExcept({kDeviceApiOff, kV22902GinOff});
}

// Same store, on the cleared-support arm: the ginType write is unconditional.
TEST_F(PropertiesFilter22902Test, ZeroesGinTypeByteEvenWhenSupportCleared) {
  ScopedHook teamLsa(g_ncclTeamLsa, [](ncclComm_t) {
    ncclTeam_t t{};
    t.nRanks = 2;
    return t;
  });
  comm_->nRanks = 8;
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_FALSE(props()->deviceApiSupport);
  EXPECT_EQ(raw_[kV22902GinOff], 0x00);
  EXPECT_EQ(raw_[kModernGinOff], kPoison);
  ExpectBytesUnchangedExcept({kDeviceApiOff, kV22902GinOff});
}

// The default seam models an lsa team spanning the comm, so support survives it.
TEST_F(PropertiesFilter22902Test, DefaultTeamSeam_SpansComm) {
  comm_->nRanks = 8;
  comm_->rank = 5;
  ArmProps(true);

  EXPECT_EQ(ncclCommPropertiesFilter_v22902(comm_.get(), props()), ncclSuccess);

  EXPECT_TRUE(props()->deviceApiSupport);
  EXPECT_EQ(raw_[kV22902GinOff], 0x00);
}

}  // namespace
// ---- end block ----

// ---- block: ncclDevCommRequirementsFilter_v22902 ----
namespace {

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

// Distinctive poison so a deleted store is visible; never 0, never equal to a
// value the block could legitimately compute.
constexpr int kRailGinPoison = 0x5A5A;

class ReqsFilterV22902Microtest : public DevcommMicrotest {
 protected:
  ncclDevCommRequirements_t reqs_{};
  // Backing storage for the resource-requirements linked list under test.
  ncclDevResourceRequirements_t rr_[4]{};

  void SetUp() override {
    DevcommMicrotest::SetUp();
    reqs_ = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs_.version = NCCL_VERSION(2, 29, 2);
    reqs_.railGinBarrierCount = kRailGinPoison;
    for (auto& rr : rr_) {
      rr = ncclDevResourceRequirements_t{};
    }
  }

  // Chains rr_[0..n-1] into reqs_.resourceRequirementsList, nullptr-terminated.
  void ChainResourceRequirements(int n) {
    reqs_.resourceRequirementsList = n > 0 ? &rr_[0] : nullptr;
    for (int i = 0; i < n; ++i) {
      rr_[i].next = (i + 1 < n) ? &rr_[i + 1] : nullptr;
    }
  }

  ncclResult_t Run() { return ncclDevCommRequirementsFilter_v22902(comm_.get(), &reqs_); }

  static std::string VersionString(int version) {
    char buf[16];
    return std::string(ncclVersionToString(version, buf, sizeof(buf)));
  }

  // The compiled/runtime halves of the WARN, asserted as label+value pairs.
  std::string CompiledNeedle() const {
    return "compiled with NCCL version " + VersionString(reqs_.version) + ", but is";
  }
  static std::string RuntimeNeedle() {
    return "running with NCCL library version " + VersionString(NCCL_VERSION_CODE) + ".";
  }
};

// Arm (a) trigger via ginForceEnable alone -> arm (b).
TEST_F(ReqsFilterV22902Microtest, GinForceEnableAlone_Rejects) {
  reqs_.ginForceEnable = true;
  reqs_.barrierCount = 7;
  reqs_.lsaBarrierCount = 3;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, CompiledNeedle().c_str())) << log;
  EXPECT_TRUE(LogHas(log, RuntimeNeedle().c_str())) << log;
  // Arm (b) returns before arm (c), so none of the fixups may have run.
  EXPECT_EQ(7, reqs_.barrierCount);
  EXPECT_EQ(3, reqs_.lsaBarrierCount);
  EXPECT_EQ(kRailGinPoison, reqs_.railGinBarrierCount);
}

// Arm (a) trigger via reqs->ginSignalCount alone.
TEST_F(ReqsFilterV22902Microtest, GinSignalCountAlone_Rejects) {
  reqs_.ginSignalCount = 2;
  reqs_.barrierCount = 7;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, CompiledNeedle().c_str())) << log;
  EXPECT_EQ(7, reqs_.barrierCount);
  EXPECT_EQ(kRailGinPoison, reqs_.railGinBarrierCount);
}

// Arm (a) trigger via reqs->ginCounterCount alone.
TEST_F(ReqsFilterV22902Microtest, GinCounterCountAlone_Rejects) {
  reqs_.ginCounterCount = 5;
  reqs_.barrierCount = 7;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, RuntimeNeedle().c_str())) << log;
  EXPECT_EQ(7, reqs_.barrierCount);
  EXPECT_EQ(kRailGinPoison, reqs_.railGinBarrierCount);
}

// Arm (a) the WARN reports the caller's reqs->version, not the runtime version.
TEST_F(ReqsFilterV22902Microtest, WarnReportsCallerVersion_NotRuntime) {
  reqs_.version = NCCL_VERSION(2, 29, 3);
  reqs_.ginForceEnable = true;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, "compiled with NCCL version 2.29.3, but is")) << log;
  EXPECT_FALSE(LogHas(log, "compiled with NCCL version 2.29.2, but is")) << log;
}

// Arm (a) empty list, all reqs counters zero -> falls through to arm (c).
TEST_F(ReqsFilterV22902Microtest, NoGinAndNullList_TakesSuccessArm) {
  reqs_.resourceRequirementsList = nullptr;
  reqs_.barrierCount = 7;
  reqs_.lsaBarrierCount = 3;

  ncclResult_t res = ncclInvalidUsage;
  const std::string log = CaptureLog([&]() {
    res = Run();
    WARN("reqsfilter-v22902 capture anchor");  // proves the capture is live
  });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(LogHas(log, "reqsfilter-v22902 capture anchor")) << log;
  EXPECT_FALSE(LogHas(log, "needs to be recompiled")) << log;
  EXPECT_EQ(7, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (a) a four-node list where no node requests GIN -> arm (c), list untouched.
TEST_F(ReqsFilterV22902Microtest, ListWithNoGinNode_TakesSuccessArm) {
  ChainResourceRequirements(4);
  for (auto& rr : rr_) {
    rr.bufferSize = 4096;  // non-GIN work, must not trip the detector
  }
  reqs_.barrierCount = 7;
  reqs_.lsaBarrierCount = 3;

  ncclResult_t res = ncclInvalidUsage;
  const std::string log = CaptureLog([&]() {
    res = Run();
    WARN("reqsfilter-v22902 capture anchor");
  });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(LogHas(log, "reqsfilter-v22902 capture anchor")) << log;
  EXPECT_FALSE(LogHas(log, "needs to be recompiled")) << log;
  EXPECT_EQ(7, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
  // The filter must not rewrite the caller's list.
  EXPECT_EQ(&rr_[0], reqs_.resourceRequirementsList);
  EXPECT_EQ(&rr_[3], rr_[2].next);
  EXPECT_EQ(4096u, rr_[3].bufferSize);
}

// Arm (a) the third of four nodes requests GIN signals; the fourth is benign, so a
// loop that failed to stop early would overwrite the flag back to false.
TEST_F(ReqsFilterV22902Microtest, ThirdListNodeSignals_RejectsDespiteBenignTail) {
  ChainResourceRequirements(4);
  rr_[2].ginSignalCount = 3;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, CompiledNeedle().c_str())) << log;
  EXPECT_TRUE(LogHas(log, RuntimeNeedle().c_str())) << log;
  EXPECT_EQ(kRailGinPoison, reqs_.railGinBarrierCount);
}

// Arm (a) same shape, but the trigger is ginCounterCount on the second node.
TEST_F(ReqsFilterV22902Microtest, SecondListNodeCounters_RejectsDespiteBenignTail) {
  ChainResourceRequirements(4);
  rr_[1].ginCounterCount = 6;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, CompiledNeedle().c_str())) << log;
  EXPECT_EQ(kRailGinPoison, reqs_.railGinBarrierCount);
}

// Arm (a) only the last node requests GIN: the walk must reach the tail.
TEST_F(ReqsFilterV22902Microtest, LastListNodeSignals_Rejects) {
  ChainResourceRequirements(4);
  rr_[3].ginSignalCount = 1;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Run(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, RuntimeNeedle().c_str())) << log;
}

// Arm (c) barrierCount > lsaBarrierCount: std::max picks barrierCount.
TEST_F(ReqsFilterV22902Microtest, BarrierCountGreater_RaisesLsaBarrierCount) {
  reqs_.barrierCount = 7;
  reqs_.lsaBarrierCount = 3;

  EXPECT_EQ(ncclSuccess, Run());
  EXPECT_EQ(7, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (c) lsaBarrierCount > barrierCount: std::max keeps lsaBarrierCount.
TEST_F(ReqsFilterV22902Microtest, LsaBarrierCountGreater_IsPreserved) {
  reqs_.barrierCount = 4;
  reqs_.lsaBarrierCount = 9;

  EXPECT_EQ(ncclSuccess, Run());
  EXPECT_EQ(9, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (c) equal counts: the merge is idempotent.
TEST_F(ReqsFilterV22902Microtest, BarrierCountsEqual_KeepsValue) {
  reqs_.barrierCount = 5;
  reqs_.lsaBarrierCount = 5;

  EXPECT_EQ(ncclSuccess, Run());
  EXPECT_EQ(5, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (c) barrierCount == 0: the if body is skipped but railGin is still zeroed.
TEST_F(ReqsFilterV22902Microtest, ZeroBarrierCount_StillZeroesRailGinBarrierCount) {
  reqs_.barrierCount = 0;
  reqs_.lsaBarrierCount = 6;

  EXPECT_EQ(ncclSuccess, Run());
  EXPECT_EQ(6, reqs_.lsaBarrierCount);  // untouched: the if body never ran
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (c) pins current behaviour for a negative barrierCount: `if (barrierCount)` is a
// truthiness test, so the merge runs and std::max discards the negative.
TEST_F(ReqsFilterV22902Microtest, NegativeBarrierCount_IsClearedAndDiscarded) {
  reqs_.barrierCount = -3;
  reqs_.lsaBarrierCount = 2;

  EXPECT_EQ(ncclSuccess, Run());
  EXPECT_EQ(2, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

// Arm (a) negative GIN counters are not a request: `> 0`, not `>= 0`.
TEST_F(ReqsFilterV22902Microtest, NegativeGinCounts_DoNotRequestGin) {
  reqs_.ginSignalCount = -1;
  reqs_.ginCounterCount = -1;
  ChainResourceRequirements(2);
  rr_[1].ginSignalCount = -2;
  rr_[1].ginCounterCount = -2;
  reqs_.barrierCount = 7;
  reqs_.lsaBarrierCount = 3;

  ncclResult_t res = ncclInvalidUsage;
  const std::string log = CaptureLog([&]() {
    res = Run();
    WARN("reqsfilter-v22902 capture anchor");
  });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(LogHas(log, "reqsfilter-v22902 capture anchor")) << log;
  EXPECT_FALSE(LogHas(log, "needs to be recompiled")) << log;
  EXPECT_EQ(7, reqs_.lsaBarrierCount);
  EXPECT_EQ(0, reqs_.railGinBarrierCount);
}

}  // namespace
// ---- end block ----

// ---- block: ncclDevCommCopyNewToOld_v22902 ----
namespace {

// The shim memcpys this many bytes into a 200-byte v22902 struct whose railGinBarrier sits at 128.
// If the modern ncclDevComm ever grows a field before railGinBarrier the copy silently overruns into
// the old struct's GIN tail, so pin the span rather than derive it.
constexpr size_t kCopyNewToOld22902LsaSpan =
    offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank);
static_assert(kCopyNewToOld22902LsaSpan == 128, "LSA prefix no longer fits below v22902 railGinBarrier");
static_assert(offsetof(struct ncclDevComm_v22902, railGinBarrier) == 128);
static_assert(sizeof(struct ncclDevComm_v22902) == 200);

constexpr size_t kCopyNewToOld22902Guard = 64;
constexpr unsigned char kCopyNewToOld22902Poison = 0xA5;

// Destination arena: the v22902 struct followed by a poisoned guard, so an oversized memset shows up.
struct CopyNewToOld22902Dest {
  alignas(16) unsigned char bytes[sizeof(struct ncclDevComm_v22902) + kCopyNewToOld22902Guard];

  CopyNewToOld22902Dest() { std::memset(bytes, kCopyNewToOld22902Poison, sizeof(bytes)); }
  struct ncclDevComm_v22902* old() { return reinterpret_cast<struct ncclDevComm_v22902*>(bytes); }
};

// Distinguishable, non-zero source. Every LSA field gets its own value so a shifted or dropped copy
// cannot alias onto a neighbour's expected value.
std::unique_ptr<struct ncclDevComm> MakeCopyNewToOld22902Source() {
  auto dev = std::make_unique<struct ncclDevComm>();
  std::memset(static_cast<void*>(dev.get()), 0x5C, sizeof(*dev));
  dev->rank = 7;
  dev->nRanks = 64;
  dev->nRanks_rcp32 = 0xDEADBEEFu;
  dev->lsaRank = 3;
  dev->lsaSize = 16;
  dev->lsaSize_rcp32 = 0xCAFEBABEu;
  dev->windowTable = reinterpret_cast<ncclDevCommWindowTable_t>(0x1122334455667788ull);
  dev->resourceWindow = reinterpret_cast<ncclWindow_t>(0x99aabbccddeeff00ull);
  return dev;
}

bool CopyNewToOld22902AllBytesAre(const unsigned char* p, size_t n, unsigned char v) {
  for (size_t i = 0; i < n; ++i) {
    if (p[i] != v) return false;
  }
  return true;
}

}  // namespace

// Arm: the whole body on the real CopyLsaData default -- memset zeroes all 200 bytes, then the LSA
// prefix is copied in from newDevComm, leaving the GIN tail zero and the source untouched.
TEST_F(DevcommMicrotest, CopyNewToOld22902_ZeroesWholeStructThenCopiesLsaPrefix) {
  CopyNewToOld22902Dest dst;
  auto src = MakeCopyNewToOld22902Source();
  unsigned char srcSnapshot[sizeof(struct ncclDevComm)];
  std::memcpy(srcSnapshot, static_cast<void*>(src.get()), sizeof(srcSnapshot));

  EXPECT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22902(comm_.get(), dst.bytes, src.get()));

  struct ncclDevComm_v22902* old = dst.old();
  EXPECT_EQ(7, old->rank);
  EXPECT_EQ(64, old->nRanks);
  EXPECT_EQ(0xDEADBEEFu, old->nRanks_rcp32);
  EXPECT_EQ(3, old->lsaRank);
  EXPECT_EQ(16, old->lsaSize);
  EXPECT_EQ(0xCAFEBABEu, old->lsaSize_rcp32);
  EXPECT_EQ(reinterpret_cast<void*>(0x1122334455667788ull), static_cast<void*>(old->windowTable));
  EXPECT_EQ(reinterpret_cast<void*>(0x99aabbccddeeff00ull), static_cast<void*>(old->resourceWindow));
  // Byte-exact: the copied prefix must equal the source's rank..railGinBarrier window verbatim.
  EXPECT_EQ(0, std::memcmp(dst.bytes,
                           reinterpret_cast<const unsigned char*>(src.get()) +
                               offsetof(struct ncclDevComm, rank),
                           kCopyNewToOld22902LsaSpan));

  // The GIN tail past the copy is never written by CopyLsaData, so only the memset can have cleared it.
  EXPECT_TRUE(CopyNewToOld22902AllBytesAre(dst.bytes + kCopyNewToOld22902LsaSpan,
                                           sizeof(struct ncclDevComm_v22902) - kCopyNewToOld22902LsaSpan,
                                           0x00));
  EXPECT_EQ(0u, old->ginContextCount);
  EXPECT_EQ(0, old->ginSignalCount);
  EXPECT_EQ(0u, old->ginSignalBase);
  EXPECT_EQ(0, old->ginCounterCount);
  EXPECT_EQ(0u, old->ginCounterBase);
  EXPECT_EQ(nullptr, old->ginSignalShadows);

  // sizeof(*old), not more: bytes past the struct keep the poison.
  EXPECT_TRUE(CopyNewToOld22902AllBytesAre(dst.bytes + sizeof(struct ncclDevComm_v22902),
                                           kCopyNewToOld22902Guard, kCopyNewToOld22902Poison));
  // Direction: newDevComm is read-only here.
  EXPECT_EQ(0, std::memcmp(srcSnapshot, static_cast<void*>(src.get()), sizeof(srcSnapshot)));
}

// Arm: the memset alone, isolated from the copy -- with an inert seam every one of the 200 bytes
// must be zero, which no partial-size memset can achieve.
TEST_F(DevcommMicrotest, CopyNewToOld22902_MemsetClearsEveryByteOfTheStruct) {
  CopyNewToOld22902Dest dst;
  auto src = MakeCopyNewToOld22902Source();
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData, [](void*, void const*) {});

  EXPECT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22902(comm_.get(), dst.bytes, src.get()));

  EXPECT_EQ(1, copyLsa.calls);
  EXPECT_TRUE(CopyNewToOld22902AllBytesAre(dst.bytes, sizeof(struct ncclDevComm_v22902), 0x00));
  EXPECT_TRUE(CopyNewToOld22902AllBytesAre(dst.bytes + sizeof(struct ncclDevComm_v22902),
                                           kCopyNewToOld22902Guard, kCopyNewToOld22902Poison));
}

// Arm: the argument computation -- CopyLsaData is handed &old->rank and &newDevComm->rank exactly
// once, in that order. Both are first fields, so a swap is only visible in the captured pointers.
TEST_F(DevcommMicrotest, CopyNewToOld22902_PassesDestThenSourceRankPointers) {
  CopyNewToOld22902Dest dst;
  auto src = MakeCopyNewToOld22902Source();
  void* seenDst = nullptr;
  void const* seenSrc = nullptr;
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData, [&](void* d, void const* s) {
    seenDst = d;
    seenSrc = s;
  });

  EXPECT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22902(comm_.get(), dst.bytes, src.get()));

  EXPECT_EQ(1, copyLsa.calls);
  EXPECT_EQ(static_cast<void*>(dst.bytes), seenDst);
  EXPECT_EQ(static_cast<void const*>(reinterpret_cast<const unsigned char*>(src.get()) +
                                     offsetof(struct ncclDevComm, rank)),
            seenSrc);
  EXPECT_NE(seenDst, seenSrc);
}

// Arm: statement order -- the seam observes a fully zeroed destination, proving the memset runs
// before the copy rather than after it (which would erase everything the copy just wrote).
TEST_F(DevcommMicrotest, CopyNewToOld22902_MemsetRunsBeforeTheCopy) {
  CopyNewToOld22902Dest dst;
  auto src = MakeCopyNewToOld22902Source();
  bool destZeroedOnEntry = false;
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData, [&](void* d, void const* s) {
    destZeroedOnEntry = CopyNewToOld22902AllBytesAre(static_cast<unsigned char*>(d),
                                                     sizeof(struct ncclDevComm_v22902), 0x00);
    DefaultNcclDevCommCopyLsaData(d, s);
  });

  EXPECT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22902(comm_.get(), dst.bytes, src.get()));

  EXPECT_EQ(1, copyLsa.calls);
  EXPECT_TRUE(destZeroedOnEntry);
  EXPECT_EQ(7, dst.old()->rank);
  EXPECT_EQ(0xCAFEBABEu, dst.old()->lsaSize_rcp32);
}
// ---- end block ----

// ---- block: ncclDevCommCopyOldToNew_v22902 ----

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

// Offsets the shim's copy is defined in terms of. Length comes from the NEW struct
// but is applied to the OLD one, so kOldRankOff + kLsaLen must stay inside the old struct.
constexpr std::size_t kNewRankOff = offsetof(struct ncclDevComm, rank);
constexpr std::size_t kNewRailOff = offsetof(struct ncclDevComm, railGinBarrier);
constexpr std::size_t kLsaLen = kNewRailOff - kNewRankOff;
constexpr std::size_t kOldRankOff = offsetof(struct ncclDevComm_v22902, rank);
static_assert(kOldRankOff + kLsaLen <= sizeof(struct ncclDevComm_v22902),
              "the shim would read past the end of the v22902 devComm");

// Position-dependent fills; the odd strides guarantee source and poison differ at
// every byte of the copied range, so "dst changed" and "src intact" stay decidable.
void FillPattern(void* p, std::size_t n, uint8_t base, uint8_t step) {
  uint8_t* b = static_cast<uint8_t*>(p);
  for (std::size_t i = 0; i < n; ++i) b[i] = static_cast<uint8_t>(base + step * i);
}

// -1 when equal, else the index of the first differing byte (a bare memcmp says only "no").
long FirstDiff(void const* a, void const* b, std::size_t n) {
  uint8_t const* x = static_cast<uint8_t const*>(a);
  uint8_t const* y = static_cast<uint8_t const*>(b);
  for (std::size_t i = 0; i < n; ++i) {
    if (x[i] != y[i]) return static_cast<long>(i);
  }
  return -1;
}

class CopyOldToNewV22902 : public DevcommMicrotest {
 protected:
  std::unique_ptr<struct ncclDevComm> dst_;
  std::unique_ptr<struct ncclDevComm_v22902> src_;
  std::vector<uint8_t> dstPoison_, srcBefore_;

  void SetUp() override {
    DevcommMicrotest::SetUp();
    dst_ = std::make_unique<struct ncclDevComm>();
    src_ = std::make_unique<struct ncclDevComm_v22902>();
    FillPattern(dst_.get(), sizeof(*dst_), 0xA0, 7);
    FillPattern(src_.get(), sizeof(*src_), 0x11, 3);
    dstPoison_.assign(dstBytes(), dstBytes() + sizeof(*dst_));
    srcBefore_.assign(srcBytes(), srcBytes() + sizeof(*src_));
  }

  uint8_t* dstBytes() { return reinterpret_cast<uint8_t*>(dst_.get()); }
  uint8_t* srcBytes() { return reinterpret_cast<uint8_t*>(src_.get()); }

  ncclResult_t Invoke() {
    return ncclDevCommCopyOldToNew_v22902(comm_.get(), dst_.get(), src_.get());
  }
};

// Arm: the single straight-line body, running the real memcpy default of the seam.
TEST_F(CopyOldToNewV22902, MovesLsaPrefixFromOldToNew_LeavesOldUntouched) {
  ASSERT_NE(-1, FirstDiff(dstBytes() + kNewRankOff, srcBefore_.data() + kOldRankOff, kLsaLen))
      << "poison and source coincide; the copy would be unobservable";

  EXPECT_EQ(ncclSuccess, Invoke());

  EXPECT_EQ(-1, FirstDiff(dstBytes() + kNewRankOff, srcBefore_.data() + kOldRankOff, kLsaLen))
      << "destination LSA prefix does not match the source it was copied from";
  EXPECT_EQ(-1, FirstDiff(srcBytes(), srcBefore_.data(), sizeof(*src_)))
      << "source was modified; the copy ran in the wrong direction";
}

// Arm: same body, asserting the deliberate absence of a memset. The caller
// (ncclDevCommDestroy) stamps magic/version before calling and must get them back.
TEST_F(CopyOldToNewV22902, PreservesDestinationOutsideLsaPrefix) {
  dst_->magic = NCCL_API_MAGIC;
  dst_->version = NCCL_VERSION_CODE;

  EXPECT_EQ(ncclSuccess, Invoke());

  EXPECT_EQ(NCCL_API_MAGIC, dst_->magic);
  EXPECT_EQ(static_cast<unsigned int>(NCCL_VERSION_CODE), dst_->version);
  EXPECT_EQ(-1, FirstDiff(dstBytes() + kNewRailOff, dstPoison_.data() + kNewRailOff,
                          sizeof(*dst_) - kNewRailOff))
      << "bytes past the LSA prefix were clobbered; the shim must not memset";
}

// Arm: same body, pinning the exact copy length at both ends of the range.
TEST_F(CopyOldToNewV22902, CopiesLastByteOfRange_NotTheByteAfterIt) {
  srcBytes()[kOldRankOff + kLsaLen - 1] = 0x5A;
  srcBytes()[kOldRankOff + kLsaLen] = 0x5B;
  dstBytes()[kNewRankOff + kLsaLen - 1] = 0xC3;
  dstBytes()[kNewRailOff] = 0xC4;

  EXPECT_EQ(ncclSuccess, Invoke());

  EXPECT_EQ(0x5A, dstBytes()[kNewRankOff + kLsaLen - 1]) << "last byte in range was not copied";
  EXPECT_EQ(0xC4, dstBytes()[kNewRailOff]) << "copy ran one byte past the LSA prefix";
}

// Arm: same body, observed at the seam -- exactly one call, destination argument first.
TEST_F(CopyOldToNewV22902, InvokesCopyLsaDataExactlyOnce_DestinationFirst) {
  void* seenDst = nullptr;
  void const* seenSrc = nullptr;
  ScopedHook copy(g_ncclDevCommCopyLsaData, [&](void* d, void const* s) {
    seenDst = d;
    seenSrc = s;
  });

  EXPECT_EQ(ncclSuccess, Invoke());

  EXPECT_EQ(1, copy.calls);
  EXPECT_EQ(static_cast<void*>(&dst_->rank), seenDst);
  EXPECT_EQ(static_cast<void const*>(&src_->rank), seenSrc);
  EXPECT_NE(seenDst, seenSrc);
  EXPECT_EQ(-1, FirstDiff(dstBytes(), dstPoison_.data(), sizeof(*dst_)))
      << "the shim wrote to the destination outside ncclDevCommCopyLsaData";
}

// Arm: the callback slots of both compat tables. This pins the wiring only; the actual "v22907
// defers to v22902" fallback lives in ncclDevCommDestroy (src/dev_runtime.cc), which is not in
// this binary, so a null devCommCopyOldToNew here is asserted but its consequence is not.
TEST_F(CopyOldToNewV22902, CompatTables_WireTheV22902Callbacks_AndLeaveV22907CopyOldToNewNull) {
  EXPECT_EQ(&ncclDevCommCopyOldToNew_v22902, ncclDevCommCompat_v22902.devCommCopyOldToNew);
  EXPECT_EQ(nullptr, ncclDevCommCompat_v22907.devCommCopyOldToNew);
  EXPECT_EQ(ncclDevCommCompat_v22902.devCommCopyNewToOld, &ncclDevCommCopyNewToOld_v22902);
  EXPECT_EQ(&ncclCommPropertiesFilter_v22902, ncclDevCommCompat_v22902.commPropertiesFilter);
  EXPECT_EQ(&ncclDevCommRequirementsFilter_v22902, ncclDevCommCompat_v22902.devCommRequirementsFilter);
  EXPECT_EQ(&ncclCommPropertiesFilter_v22907, ncclDevCommCompat_v22907.commPropertiesFilter);
  EXPECT_EQ(&ncclDevCommRequirementsFilter_v22907, ncclDevCommCompat_v22907.devCommRequirementsFilter);
  EXPECT_EQ(&ncclDevCommCopyNewToOld_v22907, ncclDevCommCompat_v22907.devCommCopyNewToOld);
}

// The version ranges select which shim set an old application gets. getNcclVersionCompat is
// first-match-wins over devCommCompat[] (src/dev_runtime.cc), so widening v22902's range would
// silently route 2.29.5-2.29.7 apps through the v22902 shims: memset of 200 bytes over a 224-byte
// ncclDevComm_v22907, and abortFlag never copied. Nothing else in the suite pins these four fields.
TEST_F(CopyOldToNewV22902, CompatTables_PinTheVersionRangesThatSelectEachShimSet) {
  EXPECT_EQ(NCCL_VERSION(2, 29, 2), ncclDevCommCompat_v22902.minVersion);
  EXPECT_EQ(NCCL_VERSION(2, 29, 3), ncclDevCommCompat_v22902.maxVersion);
  EXPECT_EQ(NCCL_VERSION(2, 29, 5), ncclDevCommCompat_v22907.minVersion);
  EXPECT_EQ(NCCL_VERSION(2, 29, 7), ncclDevCommCompat_v22907.maxVersion);

  // Disjoint and ordered, so first-match-wins over devCommCompat[] is order-independent.
  EXPECT_LT(ncclDevCommCompat_v22902.maxVersion, ncclDevCommCompat_v22907.minVersion);
  EXPECT_LE(ncclDevCommCompat_v22902.minVersion, ncclDevCommCompat_v22902.maxVersion);
  EXPECT_LE(ncclDevCommCompat_v22907.minVersion, ncclDevCommCompat_v22907.maxVersion);
  // The gap is deliberate: 2.29.4 matches no table and falls through to ncclInvalidUsage. If a
  // 2.29.4 shim set is ever added, this assertion is the one that must be revisited.
  EXPECT_EQ(NCCL_VERSION(2, 29, 4), ncclDevCommCompat_v22902.maxVersion + 1);
}

// Pins a latent layout hazard, not desired behaviour: v22902's inlined resource window
// is 8 bytes longer than the new one, so its tail lands in hybridWorldGinBarrier.
TEST_F(CopyOldToNewV22902, PinsInlinedResourceWindowTailSpillIntoHybridWorldGinBarrier) {
  constexpr std::size_t kOldInl = offsetof(struct ncclDevComm_v22902, resourceWindow_inlined);
  constexpr std::size_t kNewHybrid = offsetof(struct ncclDevComm, hybridWorldGinBarrier);
  ASSERT_EQ(sizeof(struct ncclWindow_vidmem_v22902),
            sizeof(ncclResourceWindow_vidmem_t) + sizeof(ncclGinBarrierHandle_t));

  EXPECT_EQ(ncclSuccess, Invoke());

  EXPECT_EQ(-1, FirstDiff(dstBytes() + kNewHybrid,
                          srcBefore_.data() + kOldInl + sizeof(ncclResourceWindow_vidmem_t),
                          sizeof(ncclGinBarrierHandle_t)))
      << "spill target moved; re-derive the v22902 <-> ncclDevComm prefix mapping";
  EXPECT_EQ(offsetof(struct ncclDevComm_v22902, lsaMultimem) - kOldRankOff,
            offsetof(struct ncclDevComm, lsaMultimem) - kNewRankOff)
      << "lsaMultimem no longer realigns; the whole prefix copy is now wrong";
}

}  // namespace

// ---- end block ----

// ---- block: ncclCommPropertiesFilter_v22907 ----
namespace {

// Distinct nonzero poisons: NCCL_GIN_TYPE_NONE is 0, so a props zeroed at setup cannot tell the two
// stores from a no-op, and equal poisons cannot tell one field being written into the other.
constexpr ncclGinType_t kGinPoison = NCCL_GIN_TYPE_PROXY;         // 2
constexpr ncclGinType_t kRailedGinPoison = NCCL_GIN_TYPE_GDAKI;   // 3

ncclCommProperties PoisonedProps22907(bool deviceApiSupport) {
  ncclCommProperties props{};
  props.size = sizeof(props);
  props.magic = NCCL_API_MAGIC;
  props.version = NCCL_VERSION_CODE;
  props.rank = 3;
  props.nRanks = 8;
  props.cudaDev = 5;
  props.nvmlDev = 6;
  props.deviceApiSupport = deviceApiSupport;
  props.multimemSupport = true;
  props.ginType = kGinPoison;
  props.nLsaTeams = 7;
  props.hostRmaSupport = true;
  props.railedGinType = kRailedGinPoison;
  return props;
}

ncclTeam_t TeamWithRanks22907(int nRanks) {
  ncclTeam_t t{};
  t.rank = 1;
  t.nRanks = nRanks;
  t.stride = 1;
  return t;
}

// Arm: deviceApiSupport already true, LSA team spans the whole comm -> stays true, RHS evaluated once.
TEST_F(DevcommMicrotest, PropsFilter22907_TrueAndTeamEqual_KeepsDeviceApiSupport) {
  ncclComm_t seen = nullptr;
  ScopedHook teamLsa(g_ncclTeamLsa, [&](ncclComm_t c) {
    seen = c;
    return TeamWithRanks22907(comm_->nRanks);
  });

  ncclCommProperties props = PoisonedProps22907(true);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_TRUE(props.deviceApiSupport);
  EXPECT_EQ(1, teamLsa.calls);
  EXPECT_EQ(comm_.get(), seen);
}

// Arm: deviceApiSupport already true, LSA team smaller than the comm -> cleared, RHS evaluated once.
TEST_F(DevcommMicrotest, PropsFilter22907_TrueAndTeamSmaller_ClearsDeviceApiSupport) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks / 2); });

  ncclCommProperties props = PoisonedProps22907(true);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_FALSE(props.deviceApiSupport);
  EXPECT_EQ(1, teamLsa.calls);
}

// Arm: same as above but the team is larger, so the comparison is != rather than a magnitude test.
TEST_F(DevcommMicrotest, PropsFilter22907_TrueAndTeamLarger_ClearsDeviceApiSupport) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks + 1); });

  ncclCommProperties props = PoisonedProps22907(true);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_FALSE(props.deviceApiSupport);
  EXPECT_EQ(1, teamLsa.calls);
}

// Arm: deviceApiSupport already false, team equal -> stays false and && short-circuits, so no call.
TEST_F(DevcommMicrotest, PropsFilter22907_FalseAndTeamEqual_ShortCircuitsWithoutQueryingTeam) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks); });

  ncclCommProperties props = PoisonedProps22907(false);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_FALSE(props.deviceApiSupport);
  EXPECT_EQ(0, teamLsa.calls);
}

// Arm: deviceApiSupport already false, team unequal -> stays false, still short-circuits.
TEST_F(DevcommMicrotest, PropsFilter22907_FalseAndTeamUnequal_StaysFalseWithoutQueryingTeam) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks / 2); });

  ncclCommProperties props = PoisonedProps22907(false);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_FALSE(props.deviceApiSupport);
  EXPECT_EQ(0, teamLsa.calls);
}

// Arm: both GIN stores, asserted independently from distinct poisons so dropping either one dies.
TEST_F(DevcommMicrotest, PropsFilter22907_ClearsBothGinTypesIndependently) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks); });

  ncclCommProperties props = PoisonedProps22907(true);
  ASSERT_NE(props.ginType, props.railedGinType);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_EQ(NCCL_GIN_TYPE_NONE, props.ginType);
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, props.railedGinType);
}

// The GIN stores must not depend on which deviceApiSupport arm ran.
TEST_F(DevcommMicrotest, PropsFilter22907_ClearsBothGinTypesOnTheClearedArm) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks / 2); });

  ncclCommProperties props = PoisonedProps22907(true);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_FALSE(props.deviceApiSupport);
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, props.ginType);
  EXPECT_EQ(NCCL_GIN_TYPE_NONE, props.railedGinType);
}

// Unlike the v22902 sibling this filter writes real fields, so no neighbouring field may be scribbled.
TEST_F(DevcommMicrotest, PropsFilter22907_LeavesEveryOtherPropertyUntouched) {
  ScopedHook teamLsa(g_ncclTeamLsa,
                     [&](ncclComm_t) { return TeamWithRanks22907(comm_->nRanks); });

  ncclCommProperties props = PoisonedProps22907(true);
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_EQ(sizeof(props), props.size);
  EXPECT_EQ(NCCL_API_MAGIC, props.magic);
  EXPECT_EQ(static_cast<unsigned>(NCCL_VERSION_CODE), props.version);
  EXPECT_EQ(3, props.rank);
  EXPECT_EQ(8, props.nRanks);
  EXPECT_EQ(5, props.cudaDev);
  EXPECT_EQ(6, props.nvmlDev);
  EXPECT_TRUE(props.multimemSupport);
  EXPECT_EQ(7, props.nLsaTeams);
  EXPECT_TRUE(props.hostRmaSupport);
}

// comm->nRanks is the right-hand operand: with props.nRanks deliberately different from comm->nRanks,
// a filter reading props.nRanks (or comm->rank) instead reaches the opposite verdict.
TEST_F(DevcommMicrotest, PropsFilter22907_ComparesAgainstCommNRanksNotPropsNRanks) {
  comm_->rank = 2;
  comm_->nRanks = 4;
  ScopedHook teamLsa(g_ncclTeamLsa, [&](ncclComm_t) { return TeamWithRanks22907(4); });

  ncclCommProperties props = PoisonedProps22907(true);  // props.nRanks == 8, comm->nRanks == 4
  EXPECT_EQ(ncclSuccess, ncclCommPropertiesFilter_v22907(comm_.get(), &props));

  EXPECT_TRUE(props.deviceApiSupport);
  EXPECT_EQ(1, teamLsa.calls);
}

}  // namespace
// ---- end block ----

// ---- block: ncclDevCommRequirementsFilter_v22907 ----
namespace {

// Formats to "123.45.67": nine chars, so a shrunken version buffer truncates it, and unmistakable next to the runtime version.
constexpr int kCompiledVersion22907 = 1234567;
// WARN routes __FILE__ through the log fake, so this pins the v22907 copy of a diagnostic v22902 emits verbatim.
constexpr const char kWarnSite22907[] = "devcomm_v22907.cc:";
constexpr const char kWarnLead22907[] = "The application was compiled with too old version of NCCL.";

std::string ExpectedVersionPhrase22907() {
  char compiled[16], runtime[16], phrase[160];
  snprintf(phrase, sizeof(phrase),
           "compiled with NCCL version %s, but is running with NCCL library version %s.",
           ncclVersionToString(kCompiledVersion22907, compiled, sizeof(compiled)),
           ncclVersionToString(NCCL_VERSION_CODE, runtime, sizeof(runtime)));
  return phrase;
}

class DevcommReqsFilterV22907Microtest : public DevcommMicrotest {
 protected:
  ncclDevCommRequirements_t reqs_;
  ncclDevResourceRequirements_t nodes_[3];
  unsigned char reqsSnapshot_[sizeof(ncclDevCommRequirements_t)];
  unsigned char nodesSnapshot_[sizeof(ncclDevResourceRequirements_t[3])];

  void SetUp() override {
    DevcommMicrotest::SetUp();
    std::memset(&reqs_, 0, sizeof(reqs_));
    reqs_ = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs_.version = kCompiledVersion22907;
    std::memset(nodes_, 0, sizeof(nodes_));
  }

  // Chain nodes_[0..n-1] and hang them off reqs_; the tail's next stays null.
  void LinkNodes(int n) {
    for (int i = 0; i < n; ++i) {
      nodes_[i].next = (i + 1 < n) ? &nodes_[i + 1] : nullptr;
    }
    reqs_.resourceRequirementsList = &nodes_[0];
  }

  // Byte snapshots: v22907 promotes nothing, so every arm must leave both intact.
  void Snapshot() {
    std::memcpy(reqsSnapshot_, &reqs_, sizeof(reqs_));
    std::memcpy(nodesSnapshot_, nodes_, sizeof(nodes_));
  }
  bool ReqsUnchanged() const { return std::memcmp(reqsSnapshot_, &reqs_, sizeof(reqs_)) == 0; }
  bool NodesUnchanged() const { return std::memcmp(nodesSnapshot_, nodes_, sizeof(nodes_)) == 0; }

  ncclResult_t Call() { return ncclDevCommRequirementsFilter_v22907(comm_.get(), &reqs_); }

  // Rejecting configuration used only as a positive anchor, so a no-warn assertion cannot pass on a dead capture.
  ncclResult_t CallRejectingProbe() {
    ncclDevCommRequirements_t probe = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    probe.version = kCompiledVersion22907;
    probe.ginSignalCount = 1;
    probe.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
    return ncclDevCommRequirementsFilter_v22907(comm_.get(), &probe);
  }
};

using RcclUnitTesting::CaptureLog;
using RcclUnitTesting::LogHas;

// Arm: reqs->ginSignalCount > 0 alone trips the initial expression.
TEST_F(DevcommReqsFilterV22907Microtest, SignalCountAloneRejects) {
  reqs_.ginSignalCount = 3;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_EQ(3, reqs_.ginSignalCount);
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: reqs->ginCounterCount > 0 alone trips the initial expression.
TEST_F(DevcommReqsFilterV22907Microtest, CounterCountAloneRejects) {
  reqs_.ginCounterCount = 5;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_EQ(0, reqs_.ginSignalCount);
  EXPECT_EQ(5, reqs_.ginCounterCount);
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: reqs->barrierCount > 0 alone trips it -- v22907 counts plain barriers as GIN and, unlike v22902, never promotes them.
TEST_F(DevcommReqsFilterV22907Microtest, BarrierCountAloneRejectsAndIsNotPromoted) {
  reqs_.barrierCount = 2;
  reqs_.lsaBarrierCount = 0;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_EQ(2, reqs_.barrierCount);
  EXPECT_EQ(0, reqs_.lsaBarrierCount);
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: reqs->railGinBarrierCount > 0 alone trips it, and v22907 does not zero it the way v22902 does.
TEST_F(DevcommReqsFilterV22907Microtest, RailGinBarrierCountAloneRejectsAndIsNotZeroed) {
  reqs_.railGinBarrierCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_EQ(1, reqs_.railGinBarrierCount);
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: no GIN request and an empty resource list -- the while loop is never entered.
TEST_F(DevcommReqsFilterV22907Microtest, NoRequestEmptyListReturnsSuccessWithoutWarning) {
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs_.ginForceEnable = true;
  ASSERT_EQ(nullptr, reqs_.resourceRequirementsList);
  Snapshot();

  ncclResult_t res = ncclInvalidUsage;
  const std::string quiet = CaptureLog([&]() { res = Call(); });
  const std::string anchor = CaptureLog([&]() { EXPECT_EQ(ncclInvalidUsage, CallRejectingProbe()); });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_FALSE(LogHas(quiet, kWarnSite22907)) << quiet;
  EXPECT_TRUE(LogHas(anchor, kWarnSite22907)) << anchor;
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: the loop walks a three-node list in which no node requests GIN, exits on node == nullptr, and still reports success.
TEST_F(DevcommReqsFilterV22907Microtest, AllZeroNodeListWalksToEndAndSucceeds) {
  LinkNodes(3);
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclInvalidUsage;
  const std::string quiet = CaptureLog([&]() { res = Call(); });
  const std::string anchor = CaptureLog([&]() { EXPECT_EQ(ncclInvalidUsage, CallRejectingProbe()); });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_FALSE(LogHas(quiet, kWarnSite22907)) << quiet;
  EXPECT_TRUE(LogHas(anchor, kWarnSite22907)) << anchor;
  EXPECT_EQ(&nodes_[0], reqs_.resourceRequirementsList);
  EXPECT_TRUE(ReqsUnchanged());
  EXPECT_TRUE(NodesUnchanged());
}

// Arm: a middle node's ginSignalCount trips the loop; the trailing all-zero node must not overwrite that result.
TEST_F(DevcommReqsFilterV22907Microtest, MiddleNodeSignalCountRejectsAndLoopStopsThere) {
  LinkNodes(3);
  nodes_[1].ginSignalCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_TRUE(ReqsUnchanged());
  EXPECT_TRUE(NodesUnchanged());
}

// Arm: the loop body's second operand -- a middle node's ginCounterCount alone trips it.
TEST_F(DevcommReqsFilterV22907Microtest, MiddleNodeCounterCountRejects) {
  LinkNodes(3);
  nodes_[1].ginCounterCount = 7;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  Snapshot();

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_EQ(0, nodes_[1].ginSignalCount);
  EXPECT_TRUE(NodesUnchanged());
}

// Arm: GIN resources requested but the outer gate's second operand is false on both sides -- v22907's distinguishing arm.
TEST_F(DevcommReqsFilterV22907Microtest, GinRequestedWithoutConnectionOrForceSucceedsSilently) {
  reqs_.ginSignalCount = 4;
  reqs_.ginCounterCount = 6;
  reqs_.barrierCount = 2;
  reqs_.railGinBarrierCount = 3;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_NONE;
  reqs_.ginForceEnable = false;
  Snapshot();

  ncclResult_t res = ncclInvalidUsage;
  const std::string quiet = CaptureLog([&]() { res = Call(); });
  const std::string anchor = CaptureLog([&]() { EXPECT_EQ(ncclInvalidUsage, CallRejectingProbe()); });

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_FALSE(LogHas(quiet, kWarnSite22907)) << quiet;
  EXPECT_TRUE(LogHas(anchor, kWarnSite22907)) << anchor;
  EXPECT_EQ(2, reqs_.barrierCount);
  EXPECT_EQ(3, reqs_.railGinBarrierCount);
  EXPECT_EQ(0, reqs_.lsaBarrierCount);
  EXPECT_TRUE(ReqsUnchanged());
}

// Arm: outer gate opened by ginConnectionType alone (FULL), ginForceEnable false.
TEST_F(DevcommReqsFilterV22907Microtest, ConnectionFullWithoutForceRejects) {
  reqs_.ginSignalCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs_.ginForceEnable = false;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
}

// Arm: RAIL is the other non-NONE connection type and must open the gate identically.
TEST_F(DevcommReqsFilterV22907Microtest, ConnectionRailWithoutForceRejects) {
  reqs_.ginSignalCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  reqs_.ginForceEnable = false;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
}

// Arm: outer gate opened by ginForceEnable alone, with ginConnectionType left at NONE.
TEST_F(DevcommReqsFilterV22907Microtest, ForceEnableWithoutConnectionRejects) {
  reqs_.ginSignalCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_NONE;
  reqs_.ginForceEnable = true;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
}

// Arm: the WARN itself -- both version arguments, in order, formatted through the 16-byte buffers.
TEST_F(DevcommReqsFilterV22907Microtest, WarnNamesCompiledThenRuntimeVersion) {
  reqs_.ginCounterCount = 1;
  reqs_.ginConnectionType = NCCL_GIN_CONNECTION_FULL;

  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&]() { res = Call(); });

  EXPECT_EQ(ncclInvalidUsage, res);
  EXPECT_TRUE(LogHas(log, kWarnSite22907)) << log;
  EXPECT_TRUE(LogHas(log, kWarnLead22907)) << log;
  EXPECT_TRUE(LogHas(log, ExpectedVersionPhrase22907().c_str())) << log;
  EXPECT_TRUE(LogHas(log, "Because of its use of GIN device kernels, it needs to be recompiled")) << log;
}

}  // namespace
// ---- end block ----

// ---- block: ncclDevCommCopyNewToOld_v22907 ----
#include <cstddef>
#include <cstdint>

namespace {

constexpr size_t kOld907Size = sizeof(struct ncclDevComm_v22907);
constexpr size_t kOld907AbortOff = offsetof(struct ncclDevComm_v22907, abortFlag);
// What ncclDevCommCopyLsaData actually moves; anything past it in `old` can only be zero or abortFlag.
constexpr size_t kLsaPrefixLen =
    offsetof(struct ncclDevComm, railGinBarrier) - offsetof(struct ncclDevComm, rank);
constexpr size_t kGuardLen = 32;

// Never nullptr: a 0x00 abortFlag is indistinguishable from the memset result.
uint32_t* const kAbortA = reinterpret_cast<uint32_t*>(0xABCDEF0012345678ull);
uint32_t* const kAbortB = reinterpret_cast<uint32_t*>(0x0FEDCBA987654321ull);

// Destination flanked by 0x5A guard bands so a memset running past sizeof(*old) is visible.
class Old907Buf {
 public:
  explicit Old907Buf(unsigned char poison) {
    std::memset(bytes_, 0x5A, sizeof(bytes_));
    std::memset(bytes_ + kGuardLen, poison, kOld907Size);
  }
  void* raw() { return bytes_ + kGuardLen; }
  struct ncclDevComm_v22907* old() { return reinterpret_cast<struct ncclDevComm_v22907*>(raw()); }
  unsigned char byteAt(size_t i) const { return bytes_[kGuardLen + i]; }
  bool GuardsIntact() const {
    for (size_t i = 0; i < kGuardLen; ++i) {
      if (bytes_[i] != 0x5A || bytes_[kGuardLen + kOld907Size + i] != 0x5A) return false;
    }
    return true;
  }

 private:
  alignas(16) unsigned char bytes_[kGuardLen + kOld907Size + kGuardLen];
};

std::unique_ptr<ncclDevComm> MakeZeroedNewDevComm() {
  auto p = std::make_unique<ncclDevComm>();
  std::memset(static_cast<void*>(p.get()), 0, sizeof(*p));
  return p;
}

// Arm: the memset. With CopyLsaData neutralised, every byte outside abortFlag must be zero.
TEST_F(DevcommMicrotest, CopyNewToOld22907_ClearsEveryByte_ThenStoresAbortFlagAfterwards) {
  auto newDev = MakeZeroedNewDevComm();
  newDev->abortFlag = kAbortA;

  Old907Buf buf(0xA5);
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData, [](void*, void const*) {});

  EXPECT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22907(comm_.get(), buf.raw(), newDev.get()));
  EXPECT_EQ(1, copyLsa.calls);

  for (size_t i = 0; i < kOld907Size; ++i) {
    if (i >= kOld907AbortOff && i < kOld907AbortOff + sizeof(uint32_t*)) continue;
    EXPECT_EQ(0u, buf.byteAt(i)) << "byte " << i << " of old was not cleared";
  }
  EXPECT_EQ(kAbortA, buf.old()->abortFlag);
  EXPECT_TRUE(buf.GuardsIntact());
}

// Arm: the ncclDevCommCopyLsaData call site -- exactly once, dst=&old->rank, src=&newDevComm->rank.
TEST_F(DevcommMicrotest, CopyNewToOld22907_CallsCopyLsaDataOnceWithOldRankAndNewRank) {
  auto newDev = MakeZeroedNewDevComm();
  newDev->abortFlag = kAbortA;

  Old907Buf buf(0xA5);
  void* seenDst = nullptr;
  void const* seenSrc = nullptr;
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData, [&](void* dst, void const* src) {
    seenDst = dst;
    seenSrc = src;
  });

  ASSERT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22907(comm_.get(), buf.raw(), newDev.get()));
  EXPECT_EQ(1, copyLsa.calls);
  EXPECT_EQ(static_cast<void*>(&buf.old()->rank), seenDst);
  EXPECT_EQ(static_cast<void const*>(&newDev->rank), seenSrc);
}

// Arm: the real LSA memcpy -- prefix fields land, the tail stays zero, the source is not written.
TEST_F(DevcommMicrotest, CopyNewToOld22907_RealCopyMovesLsaPrefixAndLeavesSourceUnchanged) {
  auto newDev = MakeZeroedNewDevComm();
  newDev->magic = 0xDEADBEEFu;
  newDev->version = 0x0002'1D07u;
  newDev->rank = 3;
  newDev->nRanks = 8;
  newDev->nRanks_rcp32 = 0x11223344u;
  newDev->lsaRank = 2;
  newDev->lsaSize = 4;
  newDev->lsaSize_rcp32 = 0x55667788u;
  newDev->windowTable = reinterpret_cast<ncclDevCommWindowTable_t>(0x1000200030004000ull);
  newDev->resourceWindow = reinterpret_cast<ncclWindow_t>(0x2000300040005000ull);
  newDev->abortFlag = kAbortA;

  auto srcBefore = std::make_unique<ncclDevComm>();
  std::memcpy(static_cast<void*>(srcBefore.get()), newDev.get(), sizeof(*newDev));

  Old907Buf buf(0xA5);
  ScopedHook copyLsa(g_ncclDevCommCopyLsaData,
                     [](void* dst, void const* src) { DefaultNcclDevCommCopyLsaData(dst, src); });

  ASSERT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22907(comm_.get(), buf.raw(), newDev.get()));
  EXPECT_EQ(1, copyLsa.calls);

  EXPECT_EQ(3, buf.old()->rank);
  EXPECT_EQ(8, buf.old()->nRanks);
  EXPECT_EQ(0x11223344u, buf.old()->nRanks_rcp32);
  EXPECT_EQ(2, buf.old()->lsaRank);
  EXPECT_EQ(4, buf.old()->lsaSize);
  EXPECT_EQ(0x55667788u, buf.old()->lsaSize_rcp32);
  EXPECT_EQ(reinterpret_cast<void*>(0x1000200030004000ull),
            static_cast<void*>(buf.old()->windowTable));
  EXPECT_EQ(reinterpret_cast<void*>(0x2000300040005000ull),
            static_cast<void*>(buf.old()->resourceWindow));
  EXPECT_EQ(kAbortA, buf.old()->abortFlag);

  for (size_t i = kLsaPrefixLen; i < kOld907AbortOff; ++i) {
    EXPECT_EQ(0u, buf.byteAt(i)) << "byte " << i << " past the LSA prefix was not cleared";
  }
  EXPECT_EQ(0, std::memcmp(srcBefore.get(), newDev.get(), sizeof(*newDev)));
  EXPECT_TRUE(buf.GuardsIntact());
}

// Arm: memset-before-store ordering -- a dirty old struct is fully reinitialised, GIN fields dropped.
TEST_F(DevcommMicrotest, CopyNewToOld22907_ClearsStaleGinFieldsAndOverwritesStaleAbortFlag) {
  Old907Buf buf(0x00);
  buf.old()->ginConnectionCount = 3;
  buf.old()->ginNetDeviceTypes[0] = 7;
  buf.old()->ginHandles[0] = reinterpret_cast<void*>(0x1111222233334444ull);
  buf.old()->ginSignalBase = 0x99u;
  buf.old()->ginSignalCount = 5;
  buf.old()->ginCounterBase = 0x77u;
  buf.old()->ginCounterCount = 6;
  buf.old()->ginSignalShadows = reinterpret_cast<uint64_t*>(0x5555666677778888ull);
  buf.old()->ginContextCount = 9;
  buf.old()->ginContextBase = 0x33u;
  buf.old()->ginIsRailed = true;
  buf.old()->abortFlag = kAbortB;

  auto newDev = MakeZeroedNewDevComm();
  newDev->rank = 1;
  newDev->abortFlag = kAbortA;

  ASSERT_EQ(ncclSuccess, ncclDevCommCopyNewToOld_v22907(comm_.get(), buf.raw(), newDev.get()));

  EXPECT_EQ(0u, buf.old()->ginConnectionCount);
  EXPECT_EQ(0u, buf.old()->ginNetDeviceTypes[0]);
  EXPECT_EQ(nullptr, buf.old()->ginHandles[0]);
  EXPECT_EQ(0u, buf.old()->ginSignalBase);
  EXPECT_EQ(0, buf.old()->ginSignalCount);
  EXPECT_EQ(0u, buf.old()->ginCounterBase);
  EXPECT_EQ(0, buf.old()->ginCounterCount);
  EXPECT_EQ(nullptr, buf.old()->ginSignalShadows);
  EXPECT_EQ(0u, buf.old()->ginContextCount);
  EXPECT_EQ(0u, buf.old()->ginContextBase);
  EXPECT_FALSE(buf.old()->ginIsRailed);
  EXPECT_EQ(1, buf.old()->rank);
  EXPECT_EQ(kAbortA, buf.old()->abortFlag);
  EXPECT_TRUE(buf.GuardsIntact());
}

}  // namespace
// ---- end block ----
