/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/init.cc (AICOMRCCL-1685).
//
// Like p2p-test.cc, this TU compiles the hipified unit-under-test source
// directly (`#include INIT_CC_PATH`) so static helpers become callable, links
// NO librccl/HIP, and routes every GPU/environment dependency through the fake
// seams. Coverage is grown incrementally: pick an uncovered branch out of the
// llvm-cov/BRDA report for the hipified init.cc, work out the state that
// reaches it, and add a test (see test/host/MICROTEST_README.md, "Coverage-
// driven workflow").

#include <gtest/gtest.h>

// Standard headers used by the test body; their include guards make init.cc's
// transitive re-includes no-ops.
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fakes/init_fakes.h"
#include "../common/LogCapture.hpp"                 // CaptureLog: assert on WARN/INFO text
#include "../common/ProcessIsolatedTestRunner.hpp"  // fork+execv process isolation

// Pull in alloc.h now so its macros are visible to be #undef'd before init.cc's
// transitive includes see them (same rationale as p2p-test.cc).
#include "alloc.h"

// NCCL_PARAM redirector (shared with p2p-test.cc): routes every generated
// ncclParamXxx() through g_loadParam on each call so tests can flip params
// per-case. Also pulls in param.h (needed by the RCCL_PARAM redirectors below).
#include "fakes/param_redirect.h"

// RCCL_PARAM / RCCL_PARAM_NCCL_ALIAS redirectors: init.cc uses these heavily
// (Gfx9CheapFenceOff, InitChannels, LL128ForceEnable, InjectFaults, ...). The
// real macros cache and declare a pthread_mutex_t global; redirect to
// g_loadParam so params stay per-test controllable and no mutex globals leak.
#undef RCCL_PARAM
#define RCCL_PARAM(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }
#undef RCCL_PARAM_NCCL_ALIAS
#define RCCL_PARAM_NCCL_ALIAS(name, env, deftVal) \
  int64_t rcclParam##name() { return g_loadParam(("RCCL_" env), (deftVal)); }

// ncclCalloc redirector: lets a test fail the Nth allocation so the
// NCCLCHECK(ncclCalloc(...)) early-return arms become reachable. ncclCalloc is a
// macro (alloc.h), so retargeting it here -- the same trick the PARAM
// redirectors above use -- intercepts RCCL's ncclCalloc call sites rather than
// every heap allocation. Interposing libc malloc instead would also catch
// gtest's and libstdc++'s allocations, and they would consume the failure
// counter before the code under test ever ran.
//
// Scope caveat: the #define is textual and stays live for the rest of this TU,
// so g_callocCallIndex counts EVERY ncclCalloc the running test reaches --
// including ones in a test body and inside commAlloc/devCommSetup. "0-based call
// index" therefore means "Nth ncclCalloc in this test", which equals "Nth in the
// UUT" only when the UUT is the first caller. True for the P2pSchedule tests
// that use it; verify before relying on it elsewhere.
//
// Reset lives in the fixture TearDown, not ResetInitFakes(): these two are
// static to this TU and ResetInitFakes() is compiled in init_fakes.cc, which
// cannot see them.
static int g_callocCallIndex = 0;
static int g_callocFailAt = -1;  // -1 = never fail; otherwise 0-based call index
template <typename... Args>
static ncclResult_t MicroCalloc(const char* file, int line, const char* fn, Args&&... args) {
  if (g_callocCallIndex++ == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(args)..., file, line, fn, true);
}
#undef ncclCalloc
#define ncclCalloc(...) MicroCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

// Neutralize the NVTX3 range macros so init.cc references no roctx_scoped_range_in
// symbols (empty expansion: init.cc uses them without a trailing ';'). ONLY when
// the build hasn't already disabled NVTX -- if NVTX_NO_IMPL / NVTX_DISABLE are set
// (e.g. ROCm 7.2+ configs) nvtx.h already makes these no-ops, and pre-including it
// here would double-define its types (nccl_domain).
#if !defined(NVTX_NO_IMPL) && !defined(NVTX_DISABLE)
#include "nvtx.h"  // guarded; init.cc's re-include is a no-op
#undef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#undef NVTX3_RANGE
#define NVTX3_RANGE(...)
#undef NVTX3_RANGE_ADD_PAYLOAD
#define NVTX3_RANGE_ADD_PAYLOAD(...)
#undef NVTX3_FUNC_WITH_PARAMS
#define NVTX3_FUNC_WITH_PARAMS(...)
#else
// NVTX already disabled by the build (ROCm 7.2+ defines NVTX_NO_IMPL/NVTX_DISABLE).
// On those configs core.h pulls src/include/nvtx_stub.h, which defines `struct
// nccl_domain` -- but init.cc ALSO does a direct `#include "nvtx.h"`, which defines
// `nccl_domain` again under a DIFFERENT include guard (NCCL_NVTX_H_), so the TU sees
// two definitions and fails to compile. Pre-set that guard so init.cc's direct
// nvtx.h include is a no-op (nvtx_stub.h stays the sole definer), and supply the one
// range macro nvtx.h would have provided that nvtx_stub.h does not.
#define NCCL_NVTX_H_
#ifndef NCCL_NVTX3_FUNC_RANGE
#define NCCL_NVTX3_FUNC_RANGE
#endif
#endif

// getenv seam: init.cc's direct getenv()/std::getenv() reads are intercepted at
// link time by the extern "C" getenv() override in init_fakes.cc (routes through
// the controllable microEnvMap; unmapped names hit the real libc getenv). No
// per-call-site macro, so both getenv() and std::getenv() spellings are caught.

// Pull in the hipified copy of init.cc (cudaXxx -> hipXxx already applied by
// the hipify pass in the main RCCL build). INIT_CC_PATH is defined by this
// target's CMakeLists.txt as ${PROJECT_BINARY_DIR}/hipify/src/init.cc
// (NOT init_tmp.cc -- src/init.cc is the first of the duplicate basenames).
#include INIT_CC_PATH

// -------------------------------------------------------------------------
// ncclNetInit()/ncclNetInitFromParent() fakes. Defined HERE (not in
// init_fakes.cc) because they set comm->ncclNet, which needs the full
// ncclComm/ncclNet_t layout that only this UUT TU sees. commAlloc() reads
// comm->ncclNet again in dmaBufSupported(), so a non-null net is required on
// the success path. g_ncclNetInitResult (init_fakes.h) makes the result
// injectable; on failure comm->ncclNet is left null and commAlloc returns early.
// -------------------------------------------------------------------------
static ncclNet_t g_microFakeNet = [] {
  ncclNet_t n{};
  n.name = "microfake";
  return n;
}();
ncclResult_t ncclNetInit(struct ncclComm* comm) {
  if (g_ncclNetInitResult == ncclSuccess && comm) comm->ncclNet = &g_microFakeNet;
  return g_ncclNetInitResult;
}
ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  if (g_ncclNetInitResult == ncclSuccess && comm)
    comm->ncclNet = parent ? parent->ncclNet : &g_microFakeNet;
  return g_ncclNetInitResult;
}
// Note: ncclGdrCopy is defined by init.cc itself (= NULL by default), so
// devCommSetup()'s `ncclGdrCopy != NULL` check takes the host-workFifo arm.

// ===========================================================================
// Fixture: resets all init-layer fakes between tests (TearDown). Tests that
// exercise ncclInit()/call_once outcomes run process-isolated.
// ===========================================================================
class InitMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Hermetic entry. micro_getenv falls through to the real libc getenv on a
    // map miss, so anything the developer exports reaches the unit under test:
    // envConfigOverride (init.cc:3359, reached from parseCommConfig) reads
    // NCCL_CTA_POLICY via ncclGetEnv and overwrites comm->config.CTAPolicy.
    // Worse, onceEnvCtaPolicy (init.cc:3066) is a function-local once_flag no
    // test can reset, so a host value would be latched by whichever test runs
    // envConfigOverride first -- an order-dependent failure under
    // --gtest_shuffle. Mask every name the BINARY reads through ncclGetEnv/getenv
    // so no test depends on the ambient environment.
    //
    // The list spans every real TU this target links, not just init.cc --
    // NCCL_HOSTID is read by the real utils.cc (getHostHash) and reaches the unit
    // under test via fillInfo, latched behind its own once_flag. Re-derive with:
    //   grep -n 'ncclGetEnv(\|getenv(' src/init.cc src/misc/{utils,argcheck,archinfo}.cc
    ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
    for (const char* name : {"NCCL_CTA_POLICY", "NCCL_COLLNET_ENABLE", "NCCL_CHECK_MODE",
                             "NCCL_COMM_ID", "NCCL_LAUNCH_MODE", "NCCL_NET", "NCCL_PAT_ENABLE",
                             "NCCL_TOPO_DUMP_FILE", "HSA_FORCE_FINE_GRAIN_PCIE",
                             "HSA_NO_SCRATCH_RECLAIM", "ROCSHMEM_HEAP_SIZE", "NCCL_HOSTID"}) {
      SetMicroEnvAbsent(name);
    }
  }
  void TearDown() override {
    ResetInitFakes();
    // ctaPolicyEnv (init.cc:143) is file-scope static that getEnvCtaPolicyOnce
    // only ever assigns or OR-accumulates -- it is never cleared. Production is
    // protected by the call_once at init.cc:3067; tests calling the helper
    // directly are not, and envConfigOverride reads it at init.cc:3068. Reset it
    // here so no test can leak a policy into another under --gtest_shuffle.
    ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
    // Disarm the ncclCalloc redirector so an injected failure cannot leak into
    // the next test.
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
  }
};

namespace {
// Minimal comm builder for uniformRanksPerHost, which reads ONLY
// peerInfo[i].hostHash. Each initializer value is a host id; ranks sharing an
// id are "on the same host". Nothing else on ncclComm is touched.
class HostPattern {
 public:
  explicit HostPattern(std::initializer_list<uint64_t> hosts)
      : peers_(hosts.size()), comm_(new ncclComm{}) {
    int i = 0;
    for (uint64_t h : hosts) peers_[i++].hostHash = h;
    comm_->peerInfo = peers_.data();
  }
  const ncclComm* comm() const { return comm_.get(); }
  int nranks() const { return static_cast<int>(peers_.size()); }
 private:
  std::vector<ncclPeerInfo> peers_;
  std::unique_ptr<ncclComm> comm_;
};
}  // namespace

// --- uniformRanksPerHost (init.cc:1249) -- pure host logic, no seams ---
// Returns true iff every host has the same (>0) number of ranks and they cover
// all nranks.

TEST_F(InitMicrotest, UniformRanksPerHost_TwoHostsTwoRanksEach_ReturnsTrue) {
  HostPattern p{1, 1, 2, 2};
  EXPECT_TRUE(uniformRanksPerHost(p.comm(), p.nranks()));
}

TEST_F(InitMicrotest, UniformRanksPerHost_UnevenRanksPerHost_ReturnsFalse) {
  HostPattern p{1, 1, 2};  // host 1 has 2 ranks, host 2 has 1 -> non-uniform
  EXPECT_FALSE(uniformRanksPerHost(p.comm(), p.nranks()));
}

TEST_F(InitMicrotest, UniformRanksPerHost_AllRanksOneHost_ReturnsTrue) {
  HostPattern p{7, 7, 7, 7};
  EXPECT_TRUE(uniformRanksPerHost(p.comm(), p.nranks()));
}

TEST_F(InitMicrotest, UniformRanksPerHost_SingleRank_ReturnsTrue) {
  HostPattern p{42};
  EXPECT_TRUE(uniformRanksPerHost(p.comm(), p.nranks()));
}

TEST_F(InitMicrotest, UniformRanksPerHost_ZeroRanks_ReturnsFalse) {
  HostPattern p{};  // ranksPerHost stays -1 -> ranksPerHost>0 fails
  EXPECT_FALSE(uniformRanksPerHost(p.comm(), 0));
}

// ---------------------------------------------------------------------------
// Shared log-assertion helpers, used by the showVersion, ncclP2pSchedule and
// getEnvCtaPolicyOnce sections below.
// ---------------------------------------------------------------------------
namespace {
// Substring check for captured WARN/INFO/VERSION text. These targets link
// GTest::GTest only (no gmock), so ::testing::HasSubstr is unavailable.
bool LogHas(const std::string& log, const char* needle) {
  return log.find(needle) != std::string::npos;
}

// INFO() (debug.h:50) is gated on ncclDebugLevel, which nccl_fakes.cc pins to
// NCCL_LOG_NONE -- so INFO never reaches the stderr-writing ncclDebugLog fake
// and CaptureLog would see nothing. Set the level/mask for the scope of a
// capture, then put both back. (WARN and VERSION are ungated, which is why the
// other log-asserting suites need no such guard.) The level is a parameter
// because showVersion branches on ncclDebugLevel itself.
class ScopedDebugLogging {
 public:
  explicit ScopedDebugLogging(int level = NCCL_LOG_INFO, uint64_t mask = NCCL_ALL)
      : level_(ncclDebugLevel), mask_(ncclDebugMask) {
    ncclDebugLevel = level;
    ncclDebugMask = mask;
  }
  ~ScopedDebugLogging() {
    ncclDebugLevel = level_;
    ncclDebugMask = mask_;
  }
  ScopedDebugLogging(const ScopedDebugLogging&) = delete;
  ScopedDebugLogging& operator=(const ScopedDebugLogging&) = delete;

 private:
  int level_;
  uint64_t mask_;
};
}  // namespace

// ===========================================================================
// showVersion (init.cc:1009) -- builds the once-per-process banner (version +
// git hash, HIP/ROCm compile-vs-runtime versions, hostname, librccl path) and
// emits it at VERSION or INFO level depending on ncclDebugLevel. Production
// reaches it only via std::call_once (init.cc:3399, :4251), so tests call the
// static helper directly.
//
// decodeHipVer/fmtExtVer are the real header-inline code
// (hip_rocm_version_info.h); only the four inputs are seams -- gethostname,
// dladdr, hipRuntimeGetVersion and getROCmVersion.
// ===========================================================================
namespace {
// Runs showVersion() at a given debug level and returns the emitted banner.
// VERSION is ungated but INFO is not, so the level doubles as both the branch
// input under test and the gate that lets CaptureLog see the INFO arm.
std::string RunShowVersion(int debugLevel) {
  ScopedDebugLogging log(debugLevel);
  return RcclUnitTesting::CaptureLog([] { showVersion(); });
}
}  // namespace

TEST_F(InitMicrotest, ShowVersion_HappyPath_LogsVersionHostAndLibPath) {
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  // Banner shape from the fmt::format at init.cc:1040-1041.
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "microtest")) << "git hash missing:\n" << log;  // rcclGitHash fake
  EXPECT_TRUE(LogHas(log, "Hostname")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "Librccl path")) << "actual log:\n" << log;
  // Both lookups succeeded, so neither field fell back.
  EXPECT_FALSE(LogHas(log, "Unknown")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelVersion_UsesVersionLog) {
  // First arm of the disjunction at init.cc:1043 short-circuits.
  const std::string log = RunShowVersion(NCCL_LOG_VERSION);
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelWarn_UsesVersionLog) {
  // First arm false, second true -- only reachable at exactly NCCL_LOG_WARN.
  const std::string log = RunShowVersion(NCCL_LOG_WARN);
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_HipRuntimeVersionUnavailable_StillReportsCompileTime) {
  // hipRuntimeGetVersion failing leaves hipRt default-constructed; the banner
  // still carries the compile-time HIP version.
  g_hipRuntimeGetVersion = [](int*) { return hipErrorInvalidValue; };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "HIP version")) << "actual log:\n" << log;
}

#if ROCM_VERSION >= 60000
TEST_F(InitMicrotest, ShowVersion_RocmVersionAvailable_ReportsRuntimeRocm) {
  // The getROCmVersion block is preprocessed out below ROCm 6, mirroring the
  // guard at init.cc:1027.
  g_getROCmVersionResult = 0;  // VerSuccess
  g_rocmVersionMajor = 9;
  g_rocmVersionMinor = 8;
  g_rocmVersionPatch = 7;
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "9.8.7")) << "runtime ROCm version not reported:\n" << log;
}
#endif

TEST_F(InitMicrotest, ShowVersion_GethostnameFails_ReportsUnknownHost) {
  SetGethostnameFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "Unknown")) << "hostname did not fall back:\n" << log;
  EXPECT_TRUE(LogHas(log, "Hostname")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DladdrFails_ReportsUnknownLibPath) {
  SetDladdrFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "Unknown")) << "lib path did not fall back:\n" << log;
  EXPECT_TRUE(LogHas(log, "Librccl path")) << "actual log:\n" << log;
}

// ===========================================================================
// setupChannel (init.cc:1183) -- rotates a channel's ring so it starts at the
// local rank, and builds the forward (userRanks) and inverse (rankToIndex) maps
// every ring collective indexes through. Only called from initTransportsRank
// (init.cc:2207/2215), out of reach host-only.
//
// initChannel() is faked (g_initChannelResult) because the real one needs strong
// streams, memory stacks and device allocations. It therefore does NOT allocate
// ring->userRanks/rankToIndex the way production does (channel.cc:61-62), so the
// builder below owns that storage -- which also keeps it leak-free.
// ===========================================================================
namespace {
// comm->channels is a fixed inline array (comm.h:610), so a value-initialised
// ncclComm already has the channel; only the two ring arrays need backing.
class SetupChannelComm {
 public:
  SetupChannelComm(int nranks, int channelId)
      : userRanks_(nranks > 0 ? nranks : 0, -1),
        rankToIndex_(nranks > 0 ? nranks : 0, -1),
        channelId_(channelId),
        comm_(new ncclComm{}) {
    comm_->channels[channelId].ring.userRanks = userRanks_.data();
    comm_->channels[channelId].ring.rankToIndex = rankToIndex_.data();
  }
  ncclComm* get() { return comm_.get(); }
  const ncclRing& ring() const { return comm_->channels[channelId_].ring; }
  const std::vector<int>& userRanks() const { return userRanks_; }
  const std::vector<int>& rankToIndex() const { return rankToIndex_; }

 private:
  std::vector<int> userRanks_;
  std::vector<int> rankToIndex_;
  int channelId_;
  std::unique_ptr<ncclComm> comm_;
};
}  // namespace

TEST_F(InitMicrotest, SetupChannel_RotatedRing_ReindexesFromLocalRank) {
  // ring {2,0,3,1} as seen by rank 3: rank 0 sits at index 1, rank 3 at index 2,
  // so our ring distance from rank 0 is (2-1+4)%4 == 1, and userRanks is the ring
  // rotated to start at us.
  int ringRanks[4] = {2, 0, 3, 1};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/3, /*nranks=*/4, ringRanks));

  EXPECT_EQ(1, c.ring().index);
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
  EXPECT_EQ(3, c.userRanks()[0]) << "userRanks must start at the local rank";
  // rankToIndex is the exact inverse of userRanks.
  for (int i = 0; i < 4; ++i) EXPECT_EQ(i, c.rankToIndex()[c.userRanks()[i]]) << "i=" << i;
}

TEST_F(InitMicrotest, SetupChannel_IdentityRing_Rank0_IsIdentity) {
  // Both `if`s inside the scan hit true at the SAME index (i==0).
  int ringRanks[4] = {0, 1, 2, 3};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/0, /*nranks=*/4, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.userRanks());
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.rankToIndex());
}

TEST_F(InitMicrotest, SetupChannel_RotatedRing_Rank0_IndexIsZero) {
  // Invariant: rank 0's ring distance from rank 0 is 0 whatever the rotation,
  // because ixRank and ixZero are then the same index.
  int ringRanks[4] = {2, 3, 0, 1};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/0, /*nranks=*/4, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.userRanks());
}

TEST_F(InitMicrotest, SetupChannel_SingleRank_TrivialRing) {
  int ringRanks[1] = {0};
  SetupChannelComm c(/*nranks=*/1, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/0, /*nranks=*/1, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0}), c.userRanks());
  EXPECT_EQ(std::vector<int>({0}), c.rankToIndex());
}

TEST_F(InitMicrotest, SetupChannel_InitChannelFails_PropagatesAndLeavesRingUntouched) {
  g_initChannelResult = ncclInternalError;
  int ringRanks[4] = {2, 0, 3, 1};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  EXPECT_EQ(ncclInternalError, setupChannel(c.get(), 0, /*rank=*/3, /*nranks=*/4, ringRanks));
  // NCCLCHECK returned before the scan, so the builder's -1 fill survives.
  EXPECT_EQ(std::vector<int>({-1, -1, -1, -1}), c.userRanks());
}

TEST_F(InitMicrotest, SetupChannel_InitChannelInProgress_ContinuesAndSucceeds) {
  // NCCLCHECK (checks.h) returns only when the result is neither ncclSuccess NOR
  // ncclInProgress, so ncclInProgress falls through and the ring is still built.
  // That arm is invisible to a plain success/failure pair.
  g_initChannelResult = ncclInProgress;
  int ringRanks[4] = {2, 0, 3, 1};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  EXPECT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/3, /*nranks=*/4, ringRanks));
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
}

// setupChannel does not validate that `rank` is actually a member of the ring:
// ixRank keeps its 0 initialiser, so the ring is silently left unrotated and
// userRanks[0] is whatever happened to be first -- a well-formed but wrong ring,
// with no diagnostic. (The mirror case, rank 0 missing from the ring, is not
// safely constructible: rankToIndex is indexed by rank value and sized nranks,
// so every entry must be < nranks, and nranks distinct such values must include 0.)
TEST_F(InitMicrotest, SetupChannel_RankNotInRing_SilentlyTreatsIndexZeroAsSelf) {
  int ringRanks[4] = {1, 2, 3, 0};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/7, /*nranks=*/4, ringRanks));

  EXPECT_EQ(1, c.ring().index);                              // (0 - 3 + 4) % 4
  EXPECT_EQ(std::vector<int>({1, 2, 3, 0}), c.userRanks());   // unrotated
  EXPECT_NE(7, c.userRanks()[0]) << "no validation that rank is in the ring";
}

// ===========================================================================
// commGetSplitInfo (init.cc:2485) + getParentRanks (init.cc:2528) -- the two
// helpers that decide who ends up in a split/shrunk communicator. Called
// together from ncclCommInitRankFunc (init.cc:2607/2611), out of reach
// host-only.
//
// commGetSplitInfo allgathers every parent rank's (color, key), keeps those
// matching our colour, and INSERTION-SORTS them by key. getParentRanks is pure
// integer logic with no dependencies at all.
//
// Lesson from the round-2 review of the P2P schedule tests: for a computed
// result, assert the CONTENTS. A return-code oracle here would be blind to a
// broken comparator, a wrong insert position, or a dropped shift.
// ===========================================================================
namespace {
// Parent comm for commGetSplitInfo: it reads only nRanks, rank and bootstrap.
std::unique_ptr<ncclComm> MakeParentComm(int nRanks, int rank) {
  auto parent = std::unique_ptr<ncclComm>(new ncclComm{});
  parent->nRanks = nRanks;
  parent->rank = rank;
  parent->bootstrap = nullptr;  // the allgather seam ignores it
  return parent;
}

// Scripts the post-allgather table. Production's allgather returns every rank's
// contribution INCLUDING our own, so the caller makes entry [parent->rank] agree
// with the color/key it passes to commGetSplitInfo. Lives here, not in
// init_fakes.cc: commSplitInfo is a typedef inside init.cc (2481-2484) and only
// exists in this TU.
void InstallSplitInfoTable(std::vector<std::pair<int, int>> colorKey) {
  g_bootstrapAllGather = [colorKey](void*, void* allData, int size) {
    EXPECT_EQ(static_cast<int>(sizeof(commSplitInfo)), size);
    auto* info = static_cast<commSplitInfo*>(allData);
    for (size_t i = 0; i < colorKey.size(); ++i) {
      info[i].color = colorKey[i].first;
      info[i].key = colorKey[i].second;
    }
    return ncclSuccess;
  };
}
}  // namespace

// --- getParentRanks: filters a sorted exclusion list out of the parent ranks ---

TEST_F(InitMicrotest, GetParentRanks_NoExclusions_KeepsEveryRank) {
  int out[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/4, /*parentRank=*/2, nullptr,
                                        /*excludeRanksCount=*/0, &nRanks, &myRank, out));
  EXPECT_EQ(4, nRanks);
  EXPECT_EQ(2, myRank);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), std::vector<int>(out, out + 4));
}

TEST_F(InitMicrotest, GetParentRanks_ExcludesListedRanks_CompactsAndRemapsMyRank) {
  int exclude[2] = {1, 3};
  int out[5] = {-1, -1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/5, /*parentRank=*/4, exclude,
                                        /*excludeRanksCount=*/2, &nRanks, &myRank, out));
  EXPECT_EQ(3, nRanks);
  EXPECT_EQ(2, myRank) << "rank 4 is the 3rd survivor, so its new rank is 2";
  EXPECT_EQ(std::vector<int>({0, 2, 4}), std::vector<int>(out, out + 3));
}

// LATENT BUG (init.cc:2537): *myRankRet is only written when the loop reaches
// i == parentRank, so a caller that excludes itself gets its previous value back
// with no diagnostic. The count is still computed as if nothing were wrong.
TEST_F(InitMicrotest, GetParentRanks_ExcludingCaller_LeavesMyRankUnwritten) {
  int exclude[1] = {2};
  int out[4] = {-1, -1, -1, -1};
  int nRanks = -1;
  int myRank = 0x5EED;  // sentinel: must survive untouched
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/4, /*parentRank=*/2, exclude,
                                        /*excludeRanksCount=*/1, &nRanks, &myRank, out));
  EXPECT_EQ(3, nRanks);
  EXPECT_EQ(0x5EED, myRank) << "no validation that the caller survives the exclusion";
  EXPECT_EQ(std::vector<int>({0, 1, 3}), std::vector<int>(out, out + 3));
}

TEST_F(InitMicrotest, GetParentRanks_ExcludeAll_ReturnsZeroRanks) {
  int exclude[3] = {0, 1, 2};
  int out[3] = {-1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/3, /*parentRank=*/0, exclude,
                                        /*excludeRanksCount=*/3, &nRanks, &myRank, out));
  EXPECT_EQ(0, nRanks);
  EXPECT_EQ(std::vector<int>({-1, -1, -1}), std::vector<int>(out, out + 3))
      << "nothing should have been written";
}

// --- commGetSplitInfo: colour filter + insertion sort by key ---

TEST_F(InitMicrotest, CommGetSplitInfo_NoColor_ReturnsEarlyWithoutTouchingOutputs) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable({{7, 0}, {7, 1}, {7, 2}, {7, 3}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = 0x1234, myRank = 0x5678;  // sentinels
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), NCCL_SPLIT_NOCOLOR,
                                          /*key=*/0, &nRanks, &myRank, parentRanks));
  EXPECT_EQ(0x1234, nRanks) << "NOCOLOR must leave the outputs alone";
  EXPECT_EQ(0x5678, myRank);
}

TEST_F(InitMicrotest, CommGetSplitInfo_SortedKeys_PreservesOrder) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable({{7, 0}, {7, 1}, {7, 2}, {7, 3}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(4, nRanks);
  EXPECT_EQ(1, myRank);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), std::vector<int>(parentRanks, parentRanks + 4));
}

// The one that matters. With already-sorted keys the insert position is always
// the end, so the shift loop at init.cc:2508 never runs and a broken comparator
// would go unnoticed -- exactly the blind spot the round-2 review found in the
// P2P schedule tests.
//
// Mutation ceiling for this function: `r > insert` -> `r >= insert` at
// init.cc:2508 is an EQUIVALENT mutant and no assertion can kill it. The extra
// iteration writes parentRanksRet[insert] = parentRanksRet[insert-1], which
// init.cc:2510 immediately overwrites with `i`. (It does add an out-of-bounds
// read of parentRanksRet[-1] when insert == 0 -- visible to ASan, invisible to
// any output oracle.) Everything else here is killed: both comparator flips,
// deleting the shift, append-instead-of-insert, and the colour/myRank/NOCOLOR
// inversions.
TEST_F(InitMicrotest, CommGetSplitInfo_UnsortedKeys_InsertionSortsByKey) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable({{7, 3}, {7, 1}, {7, 2}, {7, 0}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(4, nRanks);
  // Parent ranks ordered by key: rank3(key0), rank1(key1), rank2(key2), rank0(key3).
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), std::vector<int>(parentRanks, parentRanks + 4));
  EXPECT_EQ(1, myRank) << "our parent rank 1 holds key 1, so we sort to position 1";
}

TEST_F(InitMicrotest, CommGetSplitInfo_MixedColors_SelectsOnlyMatching) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/2);
  InstallSplitInfoTable({{7, 0}, {9, 1}, {7, 2}, {9, 3}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/2,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(2, nRanks);
  EXPECT_EQ(std::vector<int>({0, 2}), std::vector<int>(parentRanks, parentRanks + 2));
  EXPECT_EQ(1, myRank);
  EXPECT_EQ(-1, parentRanks[2]) << "the 0xff memset tail must be untouched";
}

// The comparator is `<=`, so an equal key inserts AFTER the incumbent: ties break
// by parent rank. `<` would reverse them.
TEST_F(InitMicrotest, CommGetSplitInfo_EqualKeys_TieBreaksByParentRank) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/3);
  InstallSplitInfoTable({{7, 5}, {7, 5}, {7, 5}, {7, 5}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/5,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(4, nRanks);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), std::vector<int>(parentRanks, parentRanks + 4));
  EXPECT_EQ(3, myRank);
}

TEST_F(InitMicrotest, CommGetSplitInfo_CallocFails_ReturnsSystemError) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable({{7, 0}, {7, 1}, {7, 2}, {7, 3}});
  g_callocFailAt = 0;  // the info table
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  EXPECT_EQ(ncclSystemError, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                              &nRanks, &myRank, parentRanks));
}

TEST_F(InitMicrotest, CommGetSplitInfo_AllGatherFails_PropagatesError) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  g_bootstrapAllGather = [](void*, void*, int) { return ncclInternalError; };
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  EXPECT_EQ(ncclInternalError, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                                &nRanks, &myRank, parentRanks));
  EXPECT_EQ(-1, nRanks) << "outputs untouched on the failure path";
}

// ===========================================================================
// ncclP2pSchedule (init.cc:1311) -- builds comm->p2pSchedule, the per-round
// send/recv rank pairing every P2P collective walks. Its only production caller
// is initTransportsRank (init.cc:2188), far out of reach host-only; the
// #include-the-.cc model lets us call the static helper directly with a
// hand-built comm. gcd (utils.h), pow2Up (bitops.h) and ncclCalloc (alloc.h)
// are all real header code here -- nothing about the algorithm is faked.
// ===========================================================================
namespace {
// The minimal comm ncclP2pSchedule() reads: the nodeRanks table, the four
// scalars it indexes with, and the p2pSchedule output array. Global ranks are
// handed out node by node, matching how the real topology numbers them.
class P2pScheduleComm {
 public:
  P2pScheduleComm(int nNodes, int node, int localRank, int nRanks, int maxLocalRanks,
                  std::initializer_list<int> localRanksPerNode)
      : nodeRanks_(localRanksPerNode.size()), comm_(new ncclComm{}) {
    // Both UUT loops iterate to comm->nNodes while nodeRanks_ is sized from the
    // list, so a mismatch would read past the end -- UB in the harness that
    // would be misattributed to the unit. Fail loudly instead. (nRanks is NOT
    // checked against sum(localRanksPerNode): two tests deliberately make them
    // disagree to reach the groupCount and round mismatch arms.)
    assert(nNodes <= static_cast<int>(localRanksPerNode.size()) &&
           "P2pScheduleComm: nNodes exceeds the localRanksPerNode list");
    rankTables_.reserve(localRanksPerNode.size());  // keep .data() pointers stable
    int nextRank = 0, i = 0;
    for (int lr : localRanksPerNode) {
      rankTables_.emplace_back(lr > 0 ? lr : 0);
      for (int r = 0; r < lr; ++r) rankTables_.back()[r] = nextRank++;
      nodeRanks_[i].localRanks = lr;
      nodeRanks_[i].localRankToRank = rankTables_.back().data();
      ++i;
    }
    schedule_.resize(nRanks > 0 ? nRanks : 0);
    comm_->nNodes = nNodes;
    comm_->node = node;
    comm_->localRank = localRank;
    comm_->nRanks = nRanks;
    comm_->maxLocalRanks = maxLocalRanks;
    comm_->nodeRanks = nodeRanks_.data();
    comm_->p2pSchedule = schedule_.data();
  }
  ncclComm* get() { return comm_.get(); }
  int sendRank(int round) const { return schedule_[round].sendRank; }
  int recvRank(int round) const { return schedule_[round].recvRank; }

 private:
  std::vector<ncclNodeRanks> nodeRanks_;
  std::vector<std::vector<int>> rankTables_;
  std::vector<ncclComm::P2pSchedulePair> schedule_;
  std::unique_ptr<ncclComm> comm_;
};

}  // namespace

// NOTE -- init.cc:1331 (`if (0 != nodeRanks[n].localRanks % groupSize)`) is DEAD
// CODE, not a coverage gap. The loop at init.cc:1314-1317 normalizes groupSize
// first:
//
//   if (localRanks % groupSize != 0 || localRanks < groupSize)
//     groupSize = gcd(groupSize, nodeRanks[node].localRanks);
//
// After each node either the condition was false (divisibility already held) or
// gcd made it hold; and since every new groupSize divides the previous one, it
// still divides every earlier node's localRanks. So on exit groupSize divides
// ALL localRanks and the 1331 check can never be true. Verified against the
// Euclid implementation at src/include/utils.h:105 for negative and zero inputs
// too. Do not try to flip this branch -- reaching it requires a state the
// function's own preceding loop makes impossible.
//
// The escape hatch is groupSize == 0, which crashes before 1331 can be reached
// -- at 1316 (`localRanks % groupSize`) normally, or at 1321
// (`localRank % groupSize`) when nNodes == 0 and the normalization loop never
// runs. That is operator-triggerable, not theoretical: ncclLoadParam does a bare
// strtoll with no range check (param.cc:96), so NCCL_P2P_SCHEDULE_GROUP_SIZE=0
// SIGFPEs, and so does any multiple of 2^32, which narrows to 0 on the
// int64_t -> int conversion at 1313. Negative values do not crash but produce a
// negative nGroups; both are pinned by the two tests below. The source bug is
// pre-existing on develop and out of scope for this test-only change.

// Pins the escape hatch the dead-code argument above leans on. A negative
// NCCL_P2P_SCHEDULE_GROUP_SIZE does not crash -- it yields a negative nGroups,
// so ncclCalloc is asked for a wildly out-of-range size and fails.
TEST_F(InitMicrotest, P2pSchedule_NegativeGroupSizeParam_FailsInAllocation) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(-2) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4, 4});
  EXPECT_EQ(ncclSystemError, ncclP2pSchedule(c.get()));
}

// ...and the crashing half. Death tests fork, so this neither corrupts the
// parent nor depends on the SIGFPE being catchable.
TEST_F(InitMicrotest, P2pSchedule_ZeroGroupSizeParam_DiesOnDivideByZero) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(0) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4, 4});
  EXPECT_DEATH(ncclP2pSchedule(c.get()), "");
}

TEST_F(InitMicrotest, P2pSchedule_SingleNode_BuildsFullSchedule) {
  // nNodes == 1 -> groupSize comes from maxLocalRanks, not the param.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4});
  ASSERT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
  // groupSize=4, one group: send walks +delta, recv walks -delta (mod 4).
  const int expectSend[4] = {0, 1, 2, 3};
  const int expectRecv[4] = {0, 3, 2, 1};
  for (int r = 0; r < 4; ++r) {
    EXPECT_EQ(expectSend[r], c.sendRank(r)) << "round " << r;
    EXPECT_EQ(expectRecv[r], c.recvRank(r)) << "round " << r;
  }
}

TEST_F(InitMicrotest, P2pSchedule_MultiNode_UsesGroupSizeParam) {
  // nNodes > 1 takes the ncclParamGroupSize() arm; comm->node=1 makes the
  // `n < comm->node` group-offset branch fire on node 0.
  //
  // localRank=1 matters: with localRank=0 both `local` (localRank % groupSize)
  // and `group` (localRank / groupSize) are identically 0, which hides the whole
  // group-level walk from any assertion. Asserting the full schedule here is
  // what kills the sendGroup/recvGroup swap and the `n < comm->node` deletion --
  // return-code assertions cannot see either.
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(2) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/1, /*localRank=*/1, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4, 4});
  ASSERT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
  // groupSize=2, nGroups=4, group=2 after the node-1 offset, local=1.
  const int expectSend[8] = {5, 4, 7, 6, 3, 2, 1, 0};
  const int expectRecv[8] = {5, 4, 3, 2, 7, 6, 1, 0};
  for (int r = 0; r < 8; ++r) {
    EXPECT_EQ(expectSend[r], c.sendRank(r)) << "send round " << r;
    EXPECT_EQ(expectRecv[r], c.recvRank(r)) << "recv round " << r;
  }
}

TEST_F(InitMicrotest, P2pSchedule_IndivisibleLocalRanks_ShrinksGroupSizeByGcd) {
  // 6 % 4 != 0 -> the first arm of the 1316 disjunction; gcd(4,6)=2 rescues it.
  // The chosen groupSize is otherwise unobservable from outside the function --
  // a wrong gcd would still produce a well-formed schedule -- so pin it via the
  // INFO at init.cc:1348, which is the only place the value is reported.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/6,
                    /*maxLocalRanks=*/4, {6});
  std::string log;
  ncclResult_t res = ncclInternalError;
  {
    ScopedDebugLogging dbg;
    log = RcclUnitTesting::CaptureLog([&] { res = ncclP2pSchedule(c.get()); });
  }
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(LogHas(log, "group size used is 2")) << "actual log:\n" << log;
}

// The `|| localRanks < groupSize` disjunct at init.cc:1316 is EFFECTIVELY DEAD,
// like the check at :1331. For any non-negative localRanks < groupSize the first
// disjunct has already fired (localRanks % groupSize == localRanks != 0), and
// the one case it uniquely adds is localRanks == 0 -- where gcd(groupSize, 0)
// == groupSize (utils.h:105, plain Euclid) makes taking the arm a no-op.
// Deleting the disjunct is an equivalent mutant for every reachable input. (It
// IS load-bearing for negative localRanks, since gcd(4,-4) returns -4, but
// localRanks is a rank count and can never be negative in production.)
//
// So this test does not claim to observe that disjunct. What it does pin is the
// nGroupsInNode == 0 case skipping the inner group loop entirely.
TEST_F(InitMicrotest, P2pSchedule_EmptyNode_SkipsGroupLoop) {
  // localRanks=0 on node 1 -> nGroupsInNode = 0/4 = 0, so the inner loop at
  // init.cc:1337 is entered zero times and contributes no groups.
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(4) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4, 0});
  EXPECT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
}

TEST_F(InitMicrotest, P2pSchedule_NonPow2Groups_ScheduleContents) {
  // nGroups=3 but nGroupsPow2=4, so the delta walk visits a value >= nGroups
  // and takes the false arm of `if (groupDelta < nGroups)`. groupSize divides
  // cleanly here, so no gcd shrink is involved. localRank=3 gives local=1,
  // group=1 -- a non-power-of-2 nGroups is what makes the `+ nGroups` wrap bias
  // at init.cc:1355 observable, and only a content assertion can see it.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/3, /*nRanks=*/6,
                    /*maxLocalRanks=*/2, {6});
  ASSERT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
  const int expectSend[6] = {3, 2, 5, 4, 1, 0};
  const int expectRecv[6] = {3, 2, 1, 0, 5, 4};
  for (int r = 0; r < 6; ++r) {
    EXPECT_EQ(expectSend[r], c.sendRank(r)) << "send round " << r;
    EXPECT_EQ(expectRecv[r], c.recvRank(r)) << "recv round " << r;
  }
}

// LATENT BUG (init.cc:1346): this `return ncclInternalError` fires before the
// free() at init.cc:1370-1371, leaking both groupToNode and groupToLocal. This
// is the LIVE one of the two leaking returns (:1334 sits behind the dead branch
// above); LSan attributes 16 bytes to this test.
TEST_F(InitMicrotest, P2pSchedule_GroupCountMismatch_ReturnsInternalError) {
  // nRanks=8 with only 4 local ranks -> nGroups=2 but groupCount=1.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4});
  ncclResult_t res = ncclSuccess;
  const std::string log =
      RcclUnitTesting::CaptureLog([&] { res = ncclP2pSchedule(c.get()); });
  EXPECT_EQ(ncclInternalError, res);
  EXPECT_TRUE(LogHas(log, "Group creation failed")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, P2pSchedule_RoundMismatch_ReturnsInternalError) {
  // nRanks=6 with groupSize=4 -> nGroups=1 (integer division), so the schedule
  // only emits 4 rounds and the final round==nRanks check fails.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/6,
                    /*maxLocalRanks=*/4, {4});
  ncclResult_t res = ncclSuccess;
  const std::string log =
      RcclUnitTesting::CaptureLog([&] { res = ncclP2pSchedule(c.get()); });
  EXPECT_EQ(ncclInternalError, res);
  EXPECT_TRUE(LogHas(log, "P2p schedule creation has bugs")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, P2pSchedule_FirstCallocFails_ReturnsSystemError) {
  g_callocFailAt = 0;  // groupToNode
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4});
  EXPECT_EQ(ncclSystemError, ncclP2pSchedule(c.get()));
}

// LATENT BUG (init.cc:1328): the NCCLCHECK embedded in the second ncclCalloc
// returns with groupToNode already allocated and never freed -- a third leaking
// return path, reachable in production under memory pressure during init.
TEST_F(InitMicrotest, P2pSchedule_SecondCallocFails_ReturnsSystemError) {
  g_callocFailAt = 1;  // groupToLocal
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4});
  EXPECT_EQ(ncclSystemError, ncclP2pSchedule(c.get()));
}

TEST_F(InitMicrotest, P2pSchedule_ZeroNodes_AllocatesNothingAndSucceeds) {
  // nGroups=0 drives ncclCalloc's nelem==0 arm (returns NULL without malloc),
  // and both node loops are entered zero times. pow2Up(0)==1 (log2Up returns 0
  // for 0), so the delta walk terminates after one skipped iteration.
  P2pScheduleComm c(/*nNodes=*/0, /*node=*/0, /*localRank=*/0, /*nRanks=*/0,
                    /*maxLocalRanks=*/1, {});
  EXPECT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
}

// --- ctaPolicyIsValid (init.cc:132) -- pure; valid range is
// [0, DEFAULT|EFFICIENCY|ZERO]. -----------------------------------------------
TEST_F(InitMicrotest, CtaPolicyIsValid_Default_True) {
  EXPECT_TRUE(ctaPolicyIsValid(NCCL_CTA_POLICY_DEFAULT));
}
TEST_F(InitMicrotest, CtaPolicyIsValid_CombinedFlags_True) {
  EXPECT_TRUE(ctaPolicyIsValid(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO));
}
TEST_F(InitMicrotest, CtaPolicyIsValid_Negative_False) {
  EXPECT_FALSE(ctaPolicyIsValid(-1));
}
TEST_F(InitMicrotest, CtaPolicyIsValid_AboveMax_False) {
  const int maxPolicy =
      NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO;
  EXPECT_FALSE(ctaPolicyIsValid(maxPolicy + 1));
}

// ===========================================================================
// getEnvCtaPolicyOnce (init.cc:144) -- parses NCCL_CTA_POLICY into the static
// ctaPolicyEnv. Two mutually exclusive syntaxes: a legacy leading digit
// (0/1/2) and a '|'-separated mode list ("EFFICIENCY|ZERO").
//
// Production reaches it only via std::call_once (init.cc:3067), and the
// EnvConfigOverride_* tests below already burn that flag for the process --
// so these tests call the helper DIRECTLY, which is exactly what the
// #include-the-.cc model exists to enable.
// ===========================================================================
namespace {
// Drives the unit from a known-clean ctaPolicyEnv with NCCL_CTA_POLICY set to
// `value`; nullptr scripts the variable as *absent*. Returns the resulting
// ctaPolicyEnv. The reset matters because the unit only ever assigns or
// OR-accumulates -- see the fixture TearDown.
int RunCtaPolicyEnv(const char* value) {
  ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
  if (value) SetMicroEnv("NCCL_CTA_POLICY", value);
  else SetMicroEnvAbsent("NCCL_CTA_POLICY");
  getEnvCtaPolicyOnce();
  return ctaPolicyEnv;
}

// As above, but returns everything the parse wrote to stderr. Several inputs
// are state-indistinguishable ("7" and an unset variable both leave UNDEF), so
// the diagnostic is the only thing that separates them.
std::string RunCtaPolicyEnvCapturingLog(const char* value, int* policyOut) {
  ScopedDebugLogging dbg;
  return RcclUnitTesting::CaptureLog([&] { *policyOut = RunCtaPolicyEnv(value); });
}
}  // namespace

// --- env unset: the early return at init.cc:146. SetMicroEnvAbsent masks the
// real environment, so this holds even on a host that exports the variable. ---
TEST_F(InitMicrotest, GetEnvCtaPolicy_Unset_LeavesPolicyUndefined) {
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, RunCtaPolicyEnv(nullptr));
}

// --- legacy single-digit syntax (init.cc:149-163) ---
TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitZero_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("0"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitOne_SelectsEfficiency) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, RunCtaPolicyEnv("1"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitTwo_SelectsZero) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, RunCtaPolicyEnv("2"));
}

// LATENT BUG (init.cc:160-162): the switch `default:` arm logs "Using DEFAULT
// instead" but never assigns NCCL_CTA_POLICY_DEFAULT, so ctaPolicyEnv is left
// UNDEF and envConfigOverride (init.cc:3068) then skips the override entirely.
// The message and the behaviour disagree.
TEST_F(InitMicrotest, GetEnvCtaPolicy_UnknownDigit_LogsDefaultButLeavesUnset) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("7", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);  // pins today's behaviour
  // "Unknown CTA policy" alone is a shared prefix of the legacy-digit message
  // (init.cc:161) and the per-token one (init.cc:174), so it cannot tell this
  // arm from the combine path. "Using DEFAULT instead" occurs exactly once in
  // init.cc and is the phrase that constitutes the latent bug.
  EXPECT_TRUE(LogHas(log, "Using DEFAULT instead")) << "actual log:\n" << log;

  // WHEN FIXED (assign DEFAULT in the `default:` arm), this is the test:
  // EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, policy);
}

// --- combine syntax (init.cc:164-186) -- one token, each strcasecmp arm ---
TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedDefault_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("DEFAULT"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedEfficiency_SelectsEfficiencyAndLogsParse) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("EFFICIENCY", &policy);
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, policy);
  // Assert the interpolated payload too, not just the sentence -- otherwise a
  // wrong policy value or env string still satisfies the needle.
  EXPECT_TRUE(LogHas(log, "NCCL_CTA_POLICY=EFFICIENCY to 1")) << "actual log:\n" << log;
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedZero_SelectsZero) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, RunCtaPolicyEnv("ZERO"));
}

// Second and later recognized tokens take the `|=` arm at init.cc:176.
TEST_F(InitMicrotest, GetEnvCtaPolicy_TwoNamedModes_AccumulatesBoth) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO,
            RunCtaPolicyEnv("EFFICIENCY|ZERO"));
}
// Comparison is strcasecmp, so spelling case is irrelevant. All THREE tokens are
// lower/mixed-cased deliberately: with DEFAULT spelled all-caps everywhere else,
// `strcasecmp(token,"DEFAULT") -> strcmp` survives the whole suite.
TEST_F(InitMicrotest, GetEnvCtaPolicy_MixedCaseModes_AccumulatesBoth) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO,
            RunCtaPolicyEnv("default|efficiency|ZeRo"));
}
// An unrecognized token is reported and skipped; the recognized ones around it
// still apply, covering the skip (174-F), first-assign and accumulate arms in
// a single parse. This is the suite's only three-token input, so it is also the
// only test that can observe strtok's continuation delimiter -- which is why
// the third token must be EFFICIENCY (0x01) and not DEFAULT: DEFAULT is 0x00,
// so ZERO|DEFAULT is bit-identical to ZERO alone and the oracle would be blind
// to the third token being dropped entirely.
TEST_F(InitMicrotest, GetEnvCtaPolicy_UnknownTokenAmongValid_IgnoresOnlyTheUnknown) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO | NCCL_CTA_POLICY_EFFICIENCY,
            RunCtaPolicyEnv("ZERO|BOGUS|EFFICIENCY"));
}
// Nothing recognized at all -> both the per-token and the summary diagnostic.
// "Logs", not "Warns": getEnvCtaPolicyOnce emits only INFO -- there is no WARN
// anywhere in init.cc:144-187, and the two are gated differently (debug.h:40
// vs :50), which is the whole reason ScopedDebugLogging exists.
TEST_F(InitMicrotest, GetEnvCtaPolicy_OnlyUnknownToken_LeavesUnsetAndLogsTwice) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("BOGUS", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);
  // Name the offending token, not just the verb.
  EXPECT_TRUE(LogHas(log, "Unknown CTA policy BOGUS")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "No valid CTA policies found")) << "actual log:\n" << log;
}
// Empty string: isdigit('\0') is false so it takes the combine arm, but strtok
// yields no token at all -- the while loop body never runs.
TEST_F(InitMicrotest, GetEnvCtaPolicy_EmptyString_ParsesNoTokens) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);
  EXPECT_TRUE(LogHas(log, "No valid CTA policies found")) << "actual log:\n" << log;
}

// LATENT BUG (init.cc:149): isdigit(env[0]) short-circuits the ENTIRE combine
// syntax, so a leading digit makes everything after the first character
// unreachable -- "0|EFFICIENCY" silently drops EFFICIENCY with no diagnostic.
TEST_F(InitMicrotest, GetEnvCtaPolicy_LeadingDigit_DropsCombinedModes) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("0|EFFICIENCY"));  // pins today

  // WHEN FIXED (take the legacy arm only for a 1-char value, or parse digits as
  // tokens), this is the test:
  // EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY,
  //           RunCtaPolicyEnv("0|EFFICIENCY"));
}

// LATENT BUG (init.cc:168-173): tokens are compared with strcasecmp but never
// trimmed, so the natural spelling "DEFAULT | ZERO" yields "DEFAULT " and
// " ZERO" -- both unknown. Nothing is applied and the user gets only the
// generic "No valid CTA policies" line.
TEST_F(InitMicrotest, GetEnvCtaPolicy_SpacesAroundPipe_NoTokensRecognized) {
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, RunCtaPolicyEnv("DEFAULT | ZERO"));  // pins today

  // WHEN FIXED (trim each token before strcasecmp), this is the test:
  // EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_ZERO,
  //           RunCtaPolicyEnv("DEFAULT | ZERO"));
}

// The unit never clears ctaPolicyEnv, so a second call ORs into the first --
// which is precisely why production guards it with call_once, and why
// RunCtaPolicyEnv resets before each parse.
TEST_F(InitMicrotest, GetEnvCtaPolicy_CalledTwice_AccumulatesAcrossCalls) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, RunCtaPolicyEnv("EFFICIENCY"));
  SetMicroEnv("NCCL_CTA_POLICY", "ZERO");
  getEnvCtaPolicyOnce();  // deliberately NOT reset in between
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO, ctaPolicyEnv);
}

// --- parseCommConfig (init.cc:3041) -- one validation arm per test.
// A fresh NCCL_CONFIG_INITIALIZER (current version) passes every arm; each test
// perturbs exactly one field and expects ncclInvalidArgument. ---------------
namespace {
ncclResult_t ParseWith(const std::function<void(ncclConfig_t&)>& tweak) {
  auto comm = std::make_unique<ncclComm>();
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  tweak(cfg);
  return parseCommConfig(comm.get(), &cfg);
}
}  // namespace

TEST_F(InitMicrotest, ParseCommConfig_BadMagic_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.magic = 0; }));
}
TEST_F(InitMicrotest, ParseCommConfig_BadBlocking_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.blocking = 2; }));
}
TEST_F(InitMicrotest, ParseCommConfig_NegativeCgaClusterSize_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.cgaClusterSize = -5; }));
}
TEST_F(InitMicrotest, ParseCommConfig_MinGreaterThanMaxCTAs_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.minCTAs = 8; c.maxCTAs = 4; }));
}
TEST_F(InitMicrotest, ParseCommConfig_BadCollnetEnable_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.collnetEnable = 2; }));
}
TEST_F(InitMicrotest, ParseCommConfig_InvalidCTAPolicy_ReturnsInvalidArgument) {
  const int maxPolicy =
      NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO;
  EXPECT_EQ(ncclInvalidArgument,
            ParseWith([&](ncclConfig_t& c) { c.CTAPolicy = maxPolicy + 1; }));
}
TEST_F(InitMicrotest, ParseCommConfig_BadMaxP2pPeers_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ParseWith([](ncclConfig_t& c) { c.maxP2pPeers = 0; }));
}
TEST_F(InitMicrotest, ParseCommConfig_ValidDefault_ReturnsSuccessAndAssigns) {
  auto comm = std::make_unique<ncclComm>();
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclSuccess, parseCommConfig(comm.get(), &cfg));
  EXPECT_EQ(1, comm->config.blocking);                       // undef -> default 1
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, comm->config.CTAPolicy);  // undef -> default
}

// --- getters + version + async-error (init.cc:4224+) ---
// CommCheck/PtrCheck are the REAL argcheck.cc oracles; a "ready" comm has valid
// magics and abortFlag->0 so ncclCommEnsureReady takes the async-error path,
// which returns ncclSuccess for a zero-inited comm (no proxy/gin/groupJob).
namespace {
class ReadyComm {
 public:
  ReadyComm() : comm_(new ncclComm{}) {
    comm_->startMagic = comm_->endMagic = NCCL_MAGIC;
    comm_->abortFlag = &abortFlag_;  // COMPILER_ATOMIC_LOAD derefs this -> 0
  }
  ncclComm* get() { return comm_.get(); }
 private:
  uint32_t abortFlag_ = 0;
  std::unique_ptr<ncclComm> comm_;
};
}  // namespace

TEST_F(InitMicrotest, CommCount_NullComm_ReturnsInvalidArgument) {
  int c = -1;
  EXPECT_EQ(ncclInvalidArgument, ncclCommCount_impl(nullptr, &c));
}
TEST_F(InitMicrotest, CommCount_CorruptedMagic_ReturnsInvalidArgument) {
  auto comm = std::make_unique<ncclComm>();  // magics 0 -> corrupt
  int c = -1;
  EXPECT_EQ(ncclInvalidArgument, ncclCommCount_impl(comm.get(), &c));
}
TEST_F(InitMicrotest, CommCount_NullOut_ReturnsInvalidArgument) {
  ReadyComm rc;
  EXPECT_EQ(ncclInvalidArgument, ncclCommCount_impl(rc.get(), nullptr));
}
TEST_F(InitMicrotest, CommCount_ReadyComm_ReturnsNRanks) {
  ReadyComm rc;
  rc.get()->nRanks = 8;
  int c = -1;
  EXPECT_EQ(ncclSuccess, ncclCommCount_impl(rc.get(), &c));
  EXPECT_EQ(8, c);
}
TEST_F(InitMicrotest, CommCuDevice_ReadyComm_ReturnsCudaDev) {
  ReadyComm rc;
  rc.get()->cudaDev = 3;
  int d = -1;
  EXPECT_EQ(ncclSuccess, ncclCommCuDevice_impl(rc.get(), &d));
  EXPECT_EQ(3, d);
}
TEST_F(InitMicrotest, CommUserRank_ReadyComm_ReturnsRank) {
  ReadyComm rc;
  rc.get()->rank = 5;
  int r = -1;
  EXPECT_EQ(ncclSuccess, ncclCommUserRank_impl(rc.get(), &r));
  EXPECT_EQ(5, r);
}
TEST_F(InitMicrotest, GetVersion_NullOut_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ncclGetVersion_impl(nullptr));
}
TEST_F(InitMicrotest, GetVersion_ReturnsVersionCode) {
  int v = 0;
  EXPECT_EQ(ncclSuccess, ncclGetVersion_impl(&v));
  EXPECT_EQ(NCCL_VERSION_CODE, v);
}
TEST_F(InitMicrotest, GetAsyncError_NullComm_ReturnsInvalidArgument) {
  ncclResult_t e = ncclSuccess;
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetAsyncError_impl(nullptr, &e));
}
TEST_F(InitMicrotest, GetAsyncError_ReadyComm_ReturnsSuccess) {
  ReadyComm rc;
  ncclResult_t e = ncclInProgress;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(rc.get(), &e));
  EXPECT_EQ(ncclSuccess, e);
}

// --- ncclCommSetAsyncError (init.cc:3415) -- pure guard + atomic store -
TEST_F(InitMicrotest, SetAsyncError_ValidState_StoresAndSucceeds) {
  auto comm = std::make_unique<ncclComm>();
  EXPECT_EQ(ncclSuccess, ncclCommSetAsyncError(comm.get(), ncclInProgress));
  EXPECT_EQ(ncclInProgress, comm->asyncResult);
}
TEST_F(InitMicrotest, SetAsyncError_NegativeState_ReturnsInvalidArgument) {
  auto comm = std::make_unique<ncclComm>();
  EXPECT_EQ(ncclInvalidArgument, ncclCommSetAsyncError(comm.get(), (ncclResult_t)-1));
}
TEST_F(InitMicrotest, SetAsyncError_OutOfRangeState_ReturnsInvalidArgument) {
  auto comm = std::make_unique<ncclComm>();
  EXPECT_EQ(ncclInvalidArgument, ncclCommSetAsyncError(comm.get(), (ncclResult_t)ncclNumResults));
}
TEST_F(InitMicrotest, SetAsyncError_NullComm_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ncclCommSetAsyncError(nullptr, ncclSuccess));
}

// --- envConfigOverride (init.cc:2793) -- per-env override branches.
// g_loadParam dispatches on the NCCL_PARAM env key; TearDown restores defaults. -
namespace {
// Fresh comm whose embedded config is all-UNDEF (via NCCL_CONFIG_INITIALIZER).
std::unique_ptr<ncclComm> UndefConfigComm() {
  auto comm = std::make_unique<ncclComm>();
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  comm->config = cfg;
  return comm;
}
}  // namespace

TEST_F(InitMicrotest, EnvConfigOverride_BlockingEnv_OverridesBlocking) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "COMM_BLOCKING") == 0 ? int64_t(1) : deft;
  };
  auto comm = UndefConfigComm();
  EXPECT_EQ(ncclSuccess, envConfigOverride(comm.get()));
  EXPECT_EQ(1, comm->config.blocking);
}
TEST_F(InitMicrotest, EnvConfigOverride_CgaClusterSizeTooBig_ClampedToMax) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "CGA_CLUSTER_SIZE") == 0 ? int64_t(NCCL_MAX_CGA_CLUSTER_SIZE + 1) : deft;
  };
  auto comm = UndefConfigComm();
  EXPECT_EQ(ncclSuccess, envConfigOverride(comm.get()));
  EXPECT_EQ(NCCL_MAX_CGA_CLUSTER_SIZE, comm->config.cgaClusterSize);
}
TEST_F(InitMicrotest, EnvConfigOverride_CgaClusterSizeInRange_Applied) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "CGA_CLUSTER_SIZE") == 0 ? int64_t(2) : deft;
  };
  auto comm = UndefConfigComm();
  EXPECT_EQ(ncclSuccess, envConfigOverride(comm.get()));
  EXPECT_EQ(2, comm->config.cgaClusterSize);
}
// TODO(AICOMRCCL-1685): a MIN_CTAS override test does not apply the value even
// though COMM_BLOCKING/CGA_CLUSTER_SIZE (same g_loadParam path) do -- investigate
// whether ncclParamMinCTAs resolves outside the redirected NCCL_PARAM. Deferred.

// --- computeBuffSizes (init.cc:1180) -- buff/chunk selection. Params
// return -2 (default) so buffSizes come from rcclSetDefaultBuffSizes; single
// node + owner==comm sets the shared tpP2pChunkSize. ---------------------------
TEST_F(InitMicrotest, ComputeBuffSizes_SingleNodeOwner_UsesDefaultsAndSetsShared) {
  auto comm = std::make_unique<ncclComm>();
  auto sr = std::make_unique<ncclSharedResources>();
  comm->sharedRes = sr.get();
  sr->owner = comm.get();      // owner==comm -> sets sharedRes->tpP2pChunkSize
  comm->nNodes = 1;
  comm->isAllNvlink = false;   // -> p2pChunkSize from P2P_PCI_CHUNKSIZE default
  EXPECT_EQ(ncclSuccess, computeBuffSizes(comm.get()));
  EXPECT_EQ(1 << 22, comm->buffSizes[NCCL_PROTO_SIMPLE]);   // default from fake
  EXPECT_EQ(1 << 17, comm->p2pChunkSize);                   // P2P_PCI default
  EXPECT_EQ(comm->p2pChunkSize, sr->tpP2pChunkSize);
}

// --- ncclCommGetAsyncError precedence (init.cc:4260) ------------------
TEST_F(InitMicrotest, GetAsyncError_CommErrorWins_OverProxyGin) {
  ReadyComm rc;
  rc.get()->asyncResult = ncclSystemError;  // set before proxy/gin checks
  ncclResult_t e = ncclSuccess;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(rc.get(), &e));
  EXPECT_EQ(ncclSystemError, e);
}
TEST_F(InitMicrotest, GetAsyncError_ProxyError_Propagates) {
  ReadyComm rc;
  auto ps = std::make_unique<ncclProxyState>();
  ps->asyncResult = ncclSystemError;
  rc.get()->proxyState = ps.get();
  ncclResult_t e = ncclSuccess;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(rc.get(), &e));
  EXPECT_EQ(ncclSystemError, e);
}
TEST_F(InitMicrotest, GetAsyncError_GinError_ReturnsRemoteError) {
  ReadyComm rc;
  auto sr = std::make_unique<ncclSharedResources>();
  sr->ginState.connected = true;
  rc.get()->sharedRes = sr.get();
  g_ginHasError = true;  // ncclGinQueryLastError -> hasError
  ncclResult_t e = ncclSuccess;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(rc.get(), &e));
  EXPECT_EQ(ncclRemoteError, e);
}
TEST_F(InitMicrotest, GetAsyncError_GroupJobCompletes_AndClears) {
  ReadyComm rc;
  auto gj = std::make_unique<ncclGroupJob>();
  rc.get()->groupJob = gj.get();
  ncclResult_t e = ncclInProgress;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(rc.get(), &e));
  EXPECT_EQ(ncclSuccess, e);
  EXPECT_EQ(nullptr, rc.get()->groupJob);  // completed -> cleared
}

// ===========================================================================
// GPU-facing helpers via the controllable device model + fault
// injection. "mock various GPUs": g_hipGetDeviceProperties supplies gcnArchName;
// g_hipRuntimeGetVersion / g_hipGetDeviceProperties inject fatal HIP errors.
// ===========================================================================

// checkHsaEnvSetting (init.cc:229): getenv + hipRuntimeGetVersion + firmware +
// hipGetDeviceProperties + validHsaScratchEnvSetting. Returns success (WARN-only
// on invalid setting); CUDACHECK maps a HIP error to ncclUnhandledCudaError.
TEST_F(InitMicrotest, CheckHsaEnvSetting_AllSucceed_ReturnsSuccess) {
  EXPECT_EQ(ncclSuccess, checkHsaEnvSetting());  // defaults: gfx942, valid
}
TEST_F(InitMicrotest, CheckHsaEnvSetting_RuntimeVersionFault_ReturnsUnhandledCudaError) {
  g_hipRuntimeGetVersion = [](int*) { return hipErrorInvalidValue; };
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) -> hipError_t {
    ADD_FAILURE() << "hipGetDeviceProperties must not be called after runtime fault";
    return hipErrorInvalidValue;
  };
  EXPECT_EQ(ncclUnhandledCudaError, checkHsaEnvSetting());
}
TEST_F(InitMicrotest, CheckHsaEnvSetting_PropertiesFault_ReturnsUnhandledCudaError) {
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclUnhandledCudaError, checkHsaEnvSetting());
}
TEST_F(InitMicrotest, CheckHsaEnvSetting_InvalidSetting_WarnsButSucceeds) {
  g_validHsaScratch = false;  // validator false -> WARN, still ncclSuccess
  EXPECT_EQ(ncclSuccess, checkHsaEnvSetting());
}

// checkHostUncacheMemSetting (init.cc:253): #if HIP_HOST_UNCACHED_MEMORY returns
// success unconditionally; otherwise IsArchMatch(gcn,"gfx950") -> ncclSystemError.
// Real IsArchMatch oracle reads comm->topo->nodes[GPU].nodes[0].gpu.gcn.
namespace {
class TopoComm {
 public:
  explicit TopoComm(const char* gcn) : comm_(new ncclComm{}), topo_(new ncclTopoSystem{}) {
    topo_->nodes[GPU].count = 1;
    std::strncpy(topo_->nodes[GPU].nodes[0].gpu.gcn, gcn,
                 sizeof(topo_->nodes[GPU].nodes[0].gpu.gcn) - 1);
    comm_->topo = topo_.get();
  }
  ncclComm* get() { return comm_.get(); }
 private:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclTopoSystem> topo_;
};
}  // namespace

// fillInfo (init.cc:1023): cheap GPU-return branches. Properties fault returns
// before the alloc probe; hipFree fault returns before downstream topology/smi.
// (Rich success/DMA-BUF/GDR scenarios need the sharedRes/rmaState/ncclNet
// fixture and land next.) getHostHash/getPidHash use the real utils.cc oracle.
TEST_F(InitMicrotest, FillInfo_PropertiesFault_DoesNotAllocate) {
  auto comm = std::make_unique<ncclComm>();
  ncclPeerInfo info{};
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) { return hipErrorInvalidValue; };
  g_hipExtMallocWithFlags = [](void**, std::size_t, unsigned) -> hipError_t {
    ADD_FAILURE() << "alloc probe must not run after a properties fault";
    return hipErrorInvalidValue;
  };
  EXPECT_EQ(ncclUnhandledCudaError, fillInfo(comm.get(), &info, 0));
}
TEST_F(InitMicrotest, FillInfo_FreeFault_ReturnsUnhandledCudaError) {
  auto comm = std::make_unique<ncclComm>();
  ncclPeerInfo info{};
  // properties + alloc succeed (defaults); the post-alloc hipFree faults.
  g_hipFree = [](void*) { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclUnhandledCudaError, fillInfo(comm.get(), &info, 0));
}

// Rich fillInfo scenarios reach the end (need sharedRes for ginState; ncclNet
// for the alloc-success dmaBuf probe). rmaState is embedded (value-inited).
namespace {
class FillInfoComm {
 public:
  FillInfoComm()
      : comm_(new ncclComm{}), sr_(new ncclSharedResources{}), net_(new ncclNet_t{}) {
    comm_->sharedRes = sr_.get();
    comm_->ncclNet = net_.get();  // regMrDmaBuf == NULL -> dmaBuf unsupported
  }
  ncclComm* get() { return comm_.get(); }
  ncclNet_t* net() { return net_.get(); }
 private:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclSharedResources> sr_;
  std::unique_ptr<ncclNet_t> net_;
};

// A valid hsa export fn so pfn_hsa != NULL selects the DMA-BUF-supported arm.
hsa_status_t FakeExportDmaBuf(const void*, size_t, int*, uint64_t*) {
  return hsa_status_t(0);
}
}  // namespace

TEST_F(InitMicrotest, FillInfo_ExtMallocFails_DisablesFineGrainAndContinues) {
  FillInfoComm c;
  ncclPeerInfo info{};
  g_hipExtMallocWithFlags = [](void**, std::size_t, unsigned) { return hipErrorOutOfMemory; };
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_FALSE(info.hasFineGrain);
  EXPECT_EQ(0, info.gdrSupport);
  EXPECT_EQ(0, g_gdrSupportCalls);  // dmaBuf/GDR probe skipped entirely
}
TEST_F(InitMicrotest, FillInfo_AllocOk_DmaBufUnsupported_UsesGdrFallback) {
  FillInfoComm c;
  ncclPeerInfo info{};
  g_gdrSupportValue = 1;  // fallback reports GDR-capable
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_TRUE(info.hasFineGrain);        // alloc succeeded
  EXPECT_EQ(1, g_gdrSupportCalls);       // dmaBuf unsupported -> fallback taken
  EXPECT_EQ(1, info.gdrSupport);         // value from the fallback
}
TEST_F(InitMicrotest, FillInfo_AllocOk_DmaBufSupported_EnablesGdrDirectly) {
  FillInfoComm c;
  // AMD dmaBufSupported arm: regMrDmaBuf present + pfn_hsa non-null -> ncclSuccess.
  c.net()->regMrDmaBuf =
      reinterpret_cast<decltype(c.net()->regMrDmaBuf)>(static_cast<uintptr_t>(0x1));
  pfn_hsa_amd_portable_export_dmabuf = &FakeExportDmaBuf;
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_TRUE(info.hasFineGrain);
  EXPECT_EQ(1, info.gdrSupport);     // set directly by the dmaBuf-supported path
  EXPECT_EQ(0, g_gdrSupportCalls);   // GDR fallback NOT called
}

// ===========================================================================
// ncclCommInitAll_impl (init.cc:3350) validation arms + device-count model.
// cudaGetDevice runs before any validation; cudaGetDeviceCount gates the devlist
// checks. (The full happy path is not covered here: the loop calls
// ncclCommInitRankDev, which runs the whole init tree.)
// ===========================================================================
TEST_F(InitMicrotest, CommInitAll_GetDeviceFault_StopsBeforeCount) {
  g_hipGetDevice = [](int*) { return hipErrorInvalidValue; };
  g_hipGetDeviceCount = [](int*) -> hipError_t {
    ADD_FAILURE() << "device-count query must not run after a getDevice fault";
    return hipErrorInvalidValue;
  };
  ncclComm_t comms[2] = {};
  EXPECT_EQ(ncclUnhandledCudaError, ncclCommInitAll_impl(comms, 2, nullptr));
}
TEST_F(InitMicrotest, CommInitAll_NegativeNdev_ReturnsInvalidArgument) {
  ncclComm_t comms[1] = {};
  EXPECT_EQ(ncclInvalidArgument, ncclCommInitAll_impl(comms, -1, nullptr));
}
TEST_F(InitMicrotest, CommInitAll_NullComms_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ncclCommInitAll_impl(nullptr, 2, nullptr));
}
TEST_F(InitMicrotest, CommInitAll_GetDeviceCountFault_ReturnsUnhandledCudaError) {
  g_hipGetDeviceCount = [](int*) { return hipErrorInvalidValue; };
  ncclComm_t comms[2] = {};
  int devlist[2] = {0, 1};
  EXPECT_EQ(ncclUnhandledCudaError, ncclCommInitAll_impl(comms, 2, devlist));
}
TEST_F(InitMicrotest, CommInitAll_InvalidDeviceInList_ReturnsInvalidArgument) {
  g_deviceCount = 2;  // valid devices are 0..1
  ncclComm_t comms[1] = {};
  int devlist[1] = {5};  // out of range
  EXPECT_EQ(ncclInvalidArgument, ncclCommInitAll_impl(comms, 1, devlist));
}
TEST_F(InitMicrotest, CommInitAll_DuplicateDevice_ReturnsInvalidUsage) {
  g_deviceCount = 8;  // MULTI_RANK_GPU_ENABLE defaults 0 -> duplicates rejected
  ncclComm_t comms[2] = {};
  int devlist[2] = {0, 0};
  EXPECT_EQ(ncclInvalidUsage, ncclCommInitAll_impl(comms, 2, devlist));
}
// Device restoration: the in-loop cudaSetDevice(dev) faults BEFORE
// ncclCommInitRankDev (so we don't enter the real init path), and the exit arm
// must still restore the original device via cudaSetDevice(oldDev).
TEST_F(InitMicrotest, CommInitAll_SetTargetDeviceFault_RestoresOriginalDevice) {
  int setCalls = 0, restoredTo = -99;
  g_hipGetDevice = [](int* d) { if (d) *d = 3; return hipSuccess; };  // oldDev = 3
  g_hipSetDevice = [&](int dev) -> hipError_t {
    if (++setCalls == 1) return hipErrorInvalidValue;  // loop set(dev) faults
    restoredTo = dev;                                  // exit set(oldDev)
    return hipSuccess;
  };
  ncclComm_t comms[2] = {};
  EXPECT_EQ(ncclUnhandledCudaError, ncclCommInitAll_impl(comms, 2, nullptr));
  EXPECT_EQ(3, restoredTo);   // restored to the captured oldDev
  EXPECT_GE(setCalls, 2);     // in-loop set + restore set
}

// --- ncclCommInitRankDev (init.cc:3244) pre-ncclInit validation arms. Only the
// nId checks run before ncclInit(); the post-init arms (null newcomm/config,
// nranks/myrank) run after the real ncclInit() and are covered below. -
TEST_F(InitMicrotest, CommInitRankDev_NIdNonPositive_ReturnsInvalidArgument) {
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/0, &id, /*myrank=*/0, 0, &cfg, "t"));
}
TEST_F(InitMicrotest, CommInitRankDev_NIdGreaterThanNranks_ReturnsInvalidArgument) {
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/5, &id, /*myrank=*/0, 0, &cfg, "t"));
}

// --- real ncclInit() runs host-only. This drives the whole init-once tree
// (checkHsaEnvSetting, initEnv, setCpuStackSize->ncclOsInitialize, initGdrCopy,
// bootstrapNetInit) via the controllable success seams, THEN the post-ncclInit
// null-newcomm validation arm returns before ncclCalloc/ncclAsyncLaunch.
// NOTE: ncclInit uses std::call_once, so this is the single in-process ncclInit
// trigger; additional ncclInit outcomes (e.g. bootstrapNetInit failure) will be
// process-isolated. ----------------------------------------------------------
TEST_F(InitMicrotest, CommInitRankDev_PostInit_NullNewcomm_ReturnsInvalidArgument) {
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(/*newcomm=*/nullptr, /*nranks=*/1, /*nId=*/1, &id, /*myrank=*/0, 0, &cfg, "t"));
}
// More post-init arms (ncclInit succeeds by default; order-independent). Note
// nranks<1 is unreachable -- the pre-init guard requires 0 < nId <= nranks.
TEST_F(InitMicrotest, CommInitRankDev_PostInit_NullConfig_ReturnsInvalidArgument) {
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(&nc, /*nranks=*/1, /*nId=*/1, &id, /*myrank=*/0, 0, /*config=*/nullptr, "t"));
}
TEST_F(InitMicrotest, CommInitRankDev_PostInit_NegativeMyrank_ReturnsInvalidArgument) {
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(&nc, /*nranks=*/1, /*nId=*/1, &id, /*myrank=*/-1, 0, &cfg, "t"));
}
TEST_F(InitMicrotest, CommInitRankDev_PostInit_MyrankGeNranks_ReturnsInvalidArgument) {
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(&nc, /*nranks=*/1, /*nId=*/1, &id, /*myrank=*/1, 0, &cfg, "t"));
}

// --- (process-isolated) ncclInit() FAILURE arm. ncclInit uses call_once, so
// once any in-process test runs it (success), it can't be made to fail. The
// ProcessIsolatedTestRunner re-execs a fresh binary image (pristine call_once)
// per registered test, so bootstrapNetInit failure genuinely fails ncclInit. --
TEST(InitMicrotestIsolated, NcclInit_BootstrapNetInitFailure_ReturnsSystemError) {
  // Canonical RUN_ISOLATED_TEST (fork+execv fresh image -> pristine call_once).
  // If isolation failed (cached ncclInit success), the valid-args path would
  // reach ncclAsyncLaunch -> fail-loud abort -> child crash -> RED. Green proves
  // ncclInit genuinely failed via bootstrapNetInit.
  //
  // The env mask lives in the lambda, not in InitMicrotest::SetUp: the child
  // re-execs and runs THIS lambda directly off a marker env var, never through
  // gtest, so no fixture SetUp ever executes there. Converting this to a TEST_F
  // would mask the parent (which does not run the body) and leave the child --
  // which reaches initOnceFunc's HSA_* reads and envConfigOverride's
  // NCCL_CTA_POLICY read -- still exposed to the real environment.
  RUN_ISOLATED_TEST(
      "Init_NcclInit_BootstrapNetInitFailure",
      []() {
        for (const char* name : {"NCCL_CTA_POLICY", "NCCL_COLLNET_ENABLE", "NCCL_CHECK_MODE",
                                 "NCCL_COMM_ID", "NCCL_LAUNCH_MODE", "NCCL_NET",
                                 "NCCL_PAT_ENABLE", "NCCL_TOPO_DUMP_FILE",
                                 "HSA_FORCE_FINE_GRAIN_PCIE", "HSA_NO_SCRATCH_RECLAIM",
                                 "ROCSHMEM_HEAP_SIZE", "NCCL_HOSTID"}) {
          SetMicroEnvAbsent(name);
        }
        g_bootstrapNetInitFail = true;
        ncclComm_t nc = nullptr;
        ncclUniqueId id{};
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        ncclResult_t r = ncclCommInitRankDev(&nc, 1, 1, &id, 0, 0, &cfg, "t");
        ASSERT_EQ(ncclSystemError, r);  // ncclInit -> initOnceFunc -> bootstrapNetInit fails
      });
}

#if defined(HIP_HOST_UNCACHED_MEMORY)
TEST_F(InitMicrotest, CheckHostUncacheMemSetting_Uncached_AlwaysSucceeds) {
  TopoComm t("gfx950:sramecc+");  // even gfx950 is OK when the build flag is set
  EXPECT_EQ(ncclSuccess, checkHostUncacheMemSetting(t.get()));
}
#else
TEST_F(InitMicrotest, CheckHostUncacheMemSetting_Cached_Gfx950_ReturnsSystemError) {
  TopoComm t("gfx950:sramecc+");
  EXPECT_EQ(ncclSystemError, checkHostUncacheMemSetting(t.get()));
}
TEST_F(InitMicrotest, CheckHostUncacheMemSetting_Cached_Gfx942_ReturnsSuccess) {
  TopoComm t("gfx942:sramecc+:xnack-");
  EXPECT_EQ(ncclSuccess, checkHostUncacheMemSetting(t.get()));
}
#endif

// ===========================================================================
// commAlloc() -- the deep allocation entry point. Runs host-only via
// the controllable deep seams (InstallCommAllocSuccess). The two arg-check arms
// need no seams; the happy path needs all seams; each failure arm injects ONE
// failure so the corresponding early-return is covered in isolation.
// ===========================================================================
namespace {
std::unique_ptr<ncclComm> FreshComm() { return std::unique_ptr<ncclComm>(new ncclComm{}); }
}  // namespace

TEST_F(InitMicrotest, CommAlloc_NdevZero_ReturnsInvalidArgument) {
  auto comm = FreshComm();
  EXPECT_EQ(ncclInvalidArgument, commAlloc(comm.get(), /*parent=*/nullptr, /*ndev=*/0, /*rank=*/0));
}

TEST_F(InitMicrotest, CommAlloc_RankOutOfRange_ReturnsInvalidArgument) {
  auto comm = FreshComm();
  EXPECT_EQ(ncclInvalidArgument, commAlloc(comm.get(), nullptr, /*ndev=*/4, /*rank=*/4));
}

TEST_F(InitMicrotest, CommAlloc_RankNegative_ReturnsInvalidArgument) {
  auto comm = FreshComm();
  EXPECT_EQ(ncclInvalidArgument, commAlloc(comm.get(), nullptr, /*ndev=*/4, /*rank=*/-1));
}

TEST_F(InitMicrotest, CommAlloc_HappyPath_ReturnsSuccessAndSetsIdentity) {
  InstallCommAllocSuccess();
  auto comm = FreshComm();
  EXPECT_EQ(ncclSuccess, commAlloc(comm.get(), nullptr, /*ndev=*/8, /*rank=*/3));
  EXPECT_EQ(8, comm->nRanks);
  EXPECT_EQ(3, comm->rank);
  EXPECT_NE(nullptr, comm->ncclNet);            // ncclNetInit installed the fake net
  EXPECT_NE(nullptr, comm->sharedRes);          // parent==NULL -> owns sharedRes
}

TEST_F(InitMicrotest, CommAlloc_NetInitFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_ncclNetInitResult = ncclSystemError;
  auto comm = FreshComm();
  EXPECT_EQ(ncclSystemError, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_GinInitFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_ncclGinInitResult = ncclInternalError;
  auto comm = FreshComm();
  EXPECT_EQ(ncclInternalError, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_MemManagerInitFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_ncclMemManagerInitResult = ncclInternalError;
  auto comm = FreshComm();
  EXPECT_EQ(ncclInternalError, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_GetDeviceFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_hipGetDevice = [](int*) { return hipErrorInvalidValue; };  // first CUDACHECK
  auto comm = FreshComm();
  EXPECT_NE(ncclSuccess, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_EventCreateFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_hipEventCreateResult = hipErrorInvalidValue;  // cudaEventCreateWithFlags arm
  auto comm = FreshComm();
  EXPECT_NE(ncclSuccess, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_WarpSizeAttrFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_hipDeviceGetAttributeResult = hipErrorInvalidValue;  // WarpSize CUDACHECK
  auto comm = FreshComm();
  EXPECT_NE(ncclSuccess, commAlloc(comm.get(), nullptr, 8, 0));
}

TEST_F(InitMicrotest, CommAlloc_MemPoolCreateFails_ReturnsError) {
  InstallCommAllocSuccess();
  g_hipMemPoolResult = hipErrorInvalidValue;  // cudaMemPoolCreate arm
  auto comm = FreshComm();
  EXPECT_NE(ncclSuccess, commAlloc(comm.get(), nullptr, 8, 0));
}

// ===========================================================================
// devCommSetup() -- builds the device-side comm/channels image. Runs on
// a comm produced by commAlloc()'s happy path; the device work goes through the
// controllable HIP async-op seams (InstallDevCommSetupSuccess). cuMem stays off
// (default) so the alloc templates take the hipExtMallocWithFlags host arm.
// ===========================================================================
namespace {
// A comm brought through commAlloc()'s happy path, ready for devCommSetup().
// void + out-param (not a return value) so a commAlloc failure -- a broken
// precondition, not the thing under test -- can ASSERT: gtest's ASSERT_* only
// unwinds a void-returning function. Callers wrap the call in
// ASSERT_NO_FATAL_FAILURE so that unwind propagates out of the test too.
void AllocedComm(std::unique_ptr<ncclComm>& comm, int ndev = 8, int rank = 0) {
  comm = std::unique_ptr<ncclComm>(new ncclComm{});
  ASSERT_EQ(ncclSuccess, commAlloc(comm.get(), /*parent=*/nullptr, ndev, rank));
}
}  // namespace

TEST_F(InitMicrotest, DevCommSetup_HappyPath_ReturnsSuccessAndSetsDevComm) {
  InstallDevCommSetupSuccess();
  std::unique_ptr<ncclComm> comm;
  ASSERT_NO_FATAL_FAILURE(AllocedComm(comm));
  EXPECT_EQ(ncclSuccess, devCommSetup(comm.get()));
  EXPECT_NE(nullptr, comm->devComm);            // devCommAndChans allocated
  EXPECT_NE(nullptr, comm->workFifoBuf);         // host workFifo arm taken
}

TEST_F(InitMicrotest, DevCommSetup_AsyncOpFails_ReturnsError) {
  InstallDevCommSetupSuccess();
  std::unique_ptr<ncclComm> comm;
  ASSERT_NO_FATAL_FAILURE(AllocedComm(comm));
  g_hipAsyncOpsResult = hipErrorInvalidValue;    // first calloc-async capture-mode fails
  EXPECT_NE(ncclSuccess, devCommSetup(comm.get()));
}

TEST_F(InitMicrotest, DevCommSetup_DevCommAllocFails_ReturnsError) {
  InstallDevCommSetupSuccess();
  std::unique_ptr<ncclComm> comm;
  ASSERT_NO_FATAL_FAILURE(AllocedComm(comm));
  g_hipExtMallocWithFlags = [](void**, std::size_t, unsigned) { return hipErrorMemoryAllocation; };
  EXPECT_NE(ncclSuccess, devCommSetup(comm.get()));
}

// ===========================================================================
// commFree() -- the teardown path. commFree() ends with free(comm), so
// the comm MUST be ncclCalloc()'d (malloc-backed) exactly like production, NOT
// new'd. It also dereferences comm->abortFlag / abortFlagRefCount (set by the
// init path, not commAlloc); we point them at locals with refCount>1 so the
// abortFlag free-branch is skipped. The teardown finalizers are benign-success
// stubs and the HIP destroys succeed, so a commAlloc'd comm frees cleanly.
// ===========================================================================
TEST_F(InitMicrotest, CommFree_Null_ReturnsSuccess) {
  EXPECT_EQ(ncclSuccess, commFree(nullptr));
}

TEST_F(InitMicrotest, CommFree_AfterCommAlloc_ReturnsSuccessAndFrees) {
  InstallCommAllocSuccess();
  ncclComm* comm = nullptr;
  ASSERT_EQ(ncclSuccess, ncclCalloc(&comm, 1));   // malloc-backed, matches commFree's free()
  ASSERT_EQ(ncclSuccess, commAlloc(comm, /*parent=*/nullptr, /*ndev=*/8, /*rank=*/0));
  uint32_t abortFlag = 0;
  int abortRef = 2;                                // >1 so commFree skips the abortFlag free-branch
  comm->abortFlag = &abortFlag;
  comm->abortFlagRefCount = &abortRef;
  EXPECT_EQ(ncclSuccess, commFree(comm));          // frees comm; do not touch it afterwards
  EXPECT_EQ(1, abortRef);                          // refcount was decremented
}
