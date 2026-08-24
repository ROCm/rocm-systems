/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/init.cc (AICOMRCCL-1685).
//
// Compiles the hipified UUT directly (#include INIT_CC_PATH) so static helpers are
// callable, links no librccl/HIP, and routes every dependency through fakes/.
// Workflow for adding coverage: test/host/MICROTEST_README.md, "Coverage-driven".

#include <gtest/gtest.h>

// Standard headers used by the test body; their include guards make init.cc's
// transitive re-includes no-ops.
#include <algorithm>
#include <cassert>
#include <csignal>
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

// ncclCalloc redirector: fail the Nth allocation to reach the NCCLCHECK(ncclCalloc)
// arms. Retargets the macro, not libc malloc -- gtest/libstdc++ would eat the counter.
// TRAP: the #define is textual and TU-wide, so the index counts every ncclCalloc the
// TEST reaches, not just the UUT's; "Nth in the UUT" holds only if the UUT allocates first.
// TRAP: reset lives in TearDown, not ResetInitFakes -- both statics are TU-local.
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

// ASan defaults break three tests below. Two are fixed here: its allocator ABORTS on
// an allocation above its 1 TiB cap instead of returning NULL (killing the binary
// mid-run at NegativeGroupSizeParam -- no smaller negative avoids it, any negative int
// widens to >= 2^63), and it intercepts SIGFPE so KilledBySignal(SIGFPE) cannot match.
// The third is SIGSEGV, handled by DEATH_BY_SEGV below rather than handle_segv=0:
// that option would disable ASan's null-deref reporting process-wide to fix one test.
// Reachable via install.sh --address-sanitizer; the standalone CMake has no sanitizer
// branch (nor an ASan rpath), so run_host_tests.sh cannot reproduce any of it.
extern "C" const char* __asan_default_options() {
  return "allocator_may_return_null=1:handle_sigfpe=0";
}

// ASan turns a SIGSEGV into a report plus exit(1), so the signal predicate never
// matches under it. Match the exit instead there; the plain build keeps the stronger
// predicate. Under ASan this loses only the _exit(1) mutant (indistinguishable from
// ASan's own exit); abort, the upstream-segv and the real fix all still fail the test.
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define DEATH_BY_SEGV ::testing::ExitedWithCode(1)
#else
#define DEATH_BY_SEGV ::testing::KilledBySignal(SIGSEGV)
#endif

// ===========================================================================
// Fixture: resets all init-layer fakes between tests (TearDown). Tests that
// exercise ncclInit()/call_once outcomes run process-isolated.
// ===========================================================================
class InitMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Mask every env name the BINARY reads: micro_getenv falls through to real getenv,
    // and onceEnvCtaPolicy (:3066) latches the first value seen -- an order-dependent
    // failure under --gtest_shuffle. Spans every linked TU, not just init.cc.
    // Re-derive with:
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

// Deriving from InitMicrotest gets the env mask into the RE-EXEC'D CHILD: it re-enters
// RUN_ALL_TESTS(), so SetUp() runs before the body reaches handleReexecEntrypoint
// (ProcessIsolatedTestRunner.cpp:754). Distinct suite NAME keeps the yaml's
// "InitMicrotestIsolated.*" matching -- gtest's `*` does not cross the literal '.'.
class InitMicrotestIsolated : public InitMicrotest {};

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
// showVersion (init.cc:1009) -- the once-per-process banner, emitted at VERSION or INFO
// depending on ncclDebugLevel. decodeHipVer/fmtExtVer are real header-inline code;
// only gethostname, dladdr, hipRuntimeGetVersion and getROCmVersion are seams.
// ===========================================================================
namespace {
// Runs showVersion() at a given debug level and returns the emitted banner.
// VERSION is ungated but INFO is not, so the level doubles as both the branch
// input under test and the gate that lets CaptureLog see the INFO arm.
std::string RunShowVersion(int debugLevel) {
  ScopedDebugLogging dbg(debugLevel);
  return RcclUnitTesting::CaptureLog([] { showVersion(); });
}
}  // namespace

// LATENT BUG (init.cc:1011-1012): hostBuf is uninitialised and gethostname need not
// NUL-terminate on truncation; the sizeof-1 pinned below is all that reserves a byte.
// Labels are format-string literals, so these assert label+separator+VALUE as one
// string -- that is what observes gethostname/dladdr/hipRuntimeGetVersion at all.
TEST_F(InitMicrotest, ShowVersion_HappyPath_LogsVersionHostAndLibPath) {
  char host[HOST_NAME_MAX] = {};
  ASSERT_EQ(0, gethostname(host, sizeof(host) - 1));
  Dl_info self{};
  ASSERT_NE(0, dladdr((void*)ncclCommInitRank, &self));

  const std::string log = RunShowVersion(NCCL_LOG_INFO);

  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "microtest")) << "git hash missing:\n" << log;  // rcclGitHash fake
  // Label, separator and value together: a swapped label pair, a substituted
  // value, a changed separator or a changed field width all break this.
  EXPECT_TRUE(LogHas(log, (std::string("Hostname     : ") + host).c_str()))
      << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, (std::string("Librccl path : ") + self.dli_fname).c_str()))
      << "actual log:\n" << log;
  // INFO passes __func__ (debug.h:50-54); VERSION passes __FILE__ (debug.h:39).
  // The captured prefix is therefore what distinguishes the two arms of the
  // ncclDebugLevel test at init.cc:1043.
  EXPECT_TRUE(LogHas(log, "showVersion:")) << "INFO arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "init.cc:")) << "took the VERSION arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelVersion_UsesVersionLog) {
  // First arm of the disjunction at init.cc:1043 short-circuits.
  const std::string log = RunShowVersion(NCCL_LOG_VERSION);
  // Asserted here, not in HappyPath: that test calls gethostname itself with the
  // same length, so it could not tell which call it observed. showVersion passes
  // sizeof(hostBuf)-1, reserving the byte gethostname need not NUL-terminate.
  EXPECT_EQ(static_cast<size_t>(HOST_NAME_MAX - 1), LastGethostnameLen());
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "init.cc:")) << "VERSION arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "showVersion:")) << "took the INFO arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelWarn_UsesVersionLog) {
  // First arm false, second true -- only reachable at exactly NCCL_LOG_WARN.
  const std::string log = RunShowVersion(NCCL_LOG_WARN);
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "init.cc:")) << "VERSION arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "showVersion:")) << "took the INFO arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_HipRuntimeVersionUnavailable_OmitsRuntimeHip) {
  // hipRuntimeGetVersion failing leaves hipRt default-constructed (valid=false),
  // so fmtExtVer prints no runtime line at all. Without the guard hipRuntimeVer
  // is still 0, hipRt becomes {true,0,0,0} and the banner gains a FABRICATED
  // "HIP runtime : 0.0.0" -- which is what this EXPECT_FALSE catches.
  g_hipRuntimeGetVersion = [](int*) { return hipErrorInvalidValue; };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "HIP version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "HIP runtime")) << "fabricated a runtime version:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_HipRuntimeDiffersFromCompileTime_ReportsRuntime) {
  // 90807006 decodes to 9.8.7006 -- deliberately not the compile-time version,
  // so swapping hipRt/hipCt in fmtExtVer is observable.
  g_hipRuntimeGetVersion = [](int* v) {
    if (v) *v = 90807006;
    return hipSuccess;
  };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "9.8.7006")) << "runtime HIP version not reported:\n" << log;
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
  EXPECT_TRUE(LogHas(log, "ROCm runtime : 9.8.7")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_RocmVersionUnavailable_OmitsRuntimeRocm) {
  // The default g_getROCmVersionResult is 1 (!= VerSuccess), so rocmRt stays
  // invalid and no runtime ROCm line is printed. Taking the arm unconditionally
  // would emit "ROCm runtime : 0.0.0".
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "ROCm runtime")) << "fabricated a runtime version:\n" << log;
}
#endif

TEST_F(InitMicrotest, ShowVersion_HipRuntimeMatchesCompileTime_OmitsRuntime) {
  // fmtExtVer suppresses the runtime line only when it equals the compile-time
  // value, so without this the compile-time baseline is never asserted at all.
  g_hipRuntimeGetVersion = [](int* v) {
    if (v) *v = HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH;
    return hipSuccess;
  };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "HIP version")) << "actual log:\n" << log;  // fails on an empty log
  EXPECT_FALSE(LogHas(log, "HIP runtime")) << "actual log:\n" << log;
}

#if ROCM_VERSION >= 60000
TEST_F(InitMicrotest, ShowVersion_RocmRuntimeMatchesCompileTime_OmitsRuntime) {
  g_getROCmVersionResult = 0;  // VerSuccess
  g_rocmVersionMajor = ROCM_VERSION_MAJOR;
  g_rocmVersionMinor = ROCM_VERSION_MINOR;
  g_rocmVersionPatch = ROCM_VERSION_PATCH;
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm version")) << "actual log:\n" << log;  // fails on an empty log
  EXPECT_FALSE(LogHas(log, "ROCm runtime")) << "actual log:\n" << log;
}
#endif

TEST_F(InitMicrotest, ShowVersion_GethostnameFails_ReportsUnknownHost) {
  SetGethostnameFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  // Trailing \n matters: LogHas is a substring check, so without it "UnknownHostX"
  // would still match. Hostname is followed by \n in the format string itself.
  EXPECT_TRUE(LogHas(log, "Hostname     : Unknown\n")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DladdrFails_ReportsUnknownLibPath) {
  SetDladdrFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  // Librccl path is the last field, so the terminator here is the logger's own \n.
  EXPECT_TRUE(LogHas(log, "Librccl path : Unknown\n")) << "actual log:\n" << log;
}

// ===========================================================================
// setupChannel (init.cc:1183) -- rotates a channel's ring to start at the local rank and
// builds the userRanks/rankToIndex maps. initChannel is faked, and unlike the real
// one (channel.cc:61-62) it does NOT allocate those two arrays -- the builder owns them.
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

// LATENT BUG (init.cc:1189-1193): setupChannel validates neither that `rank` nor
// that rank 0 is a member of the ring. ixRank/ixZero keep their 0 initialisers
// and the result is a well-formed but WRONG ring, with no diagnostic. Both
// halves are pinned below.
TEST_F(InitMicrotest, SetupChannel_RankNotInRing_SilentlyTreatsIndexZeroAsSelf) {
  int ringRanks[4] = {1, 2, 3, 0};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/7, /*nranks=*/4, ringRanks));

  EXPECT_EQ(1, c.ring().index);                              // (0 - 3 + 4) % 4
  EXPECT_EQ(std::vector<int>({1, 2, 3, 0}), c.userRanks());   // unrotated: rank 7 is not first
  EXPECT_EQ(std::vector<int>({3, 0, 1, 2}), c.rankToIndex());
}

// The mirror half, and the more dangerous one. Memory safety only requires every
// entry < nranks -- NOT distinctness -- so a ring with no rank 0 is perfectly
// constructible. ixZero stays 0, the ring is misrotated, and the duplicate makes
// rankToIndex[0] never written at all: production leaves it 0 from initChannel's
// calloc, i.e. an index aliased onto position 0.
TEST_F(InitMicrotest, SetupChannel_RankZeroNotInRing_MisrotatesAndLeavesHole) {
  int ringRanks[4] = {1, 1, 2, 3};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/2, /*nranks=*/4, ringRanks));

  EXPECT_EQ(2, c.ring().index) << "measured from a rank 0 that is not in the ring";
  EXPECT_EQ(std::vector<int>({2, 3, 1, 1}), c.userRanks());
  // rankToIndex[0] is the builder's -1 fill: nothing ever wrote it.
  EXPECT_EQ(std::vector<int>({-1, 3, 0, 1}), c.rankToIndex());
}

// Every rotation the other tests use is a SELF-INVERSE permutation, so the
// forward and inverse maps coincide and `rankToIndex[userRanks[i]] = i` cannot be
// told apart from `rankToIndex[i] = userRanks[i]`. A 4-cycle is the smallest
// shape that separates them.
TEST_F(InitMicrotest, SetupChannel_NonInvolutiveRotation_PinsRankToIndex) {
  int ringRanks[4] = {0, 1, 2, 3};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/0);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 0, /*rank=*/1, /*nranks=*/4, ringRanks));

  EXPECT_EQ(std::vector<int>({1, 2, 3, 0}), c.userRanks());
  EXPECT_EQ(std::vector<int>({3, 0, 1, 2}), c.rankToIndex()) << "not the forward map";
}

// Every other setupChannel test uses channelId 0, which makes both
// `channels[channelId] -> channels[0]` and a wrong channelId forwarded to
// initChannel unobservable.
TEST_F(InitMicrotest, SetupChannel_NonZeroChannelId_UsesThatChannel) {
  int ringRanks[4] = {2, 0, 3, 1};
  SetupChannelComm c(/*nranks=*/4, /*channelId=*/2);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), 2, /*rank=*/3, /*nranks=*/4, ringRanks));

  EXPECT_EQ(1, c.ring().index);
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
  EXPECT_EQ(2, g_initChannelLastId) << "the channelId must be forwarded, not hardcoded";
}

// ===========================================================================
// commGetSplitInfo (init.cc:2485) + getParentRanks (:2528) -- who ends up in a split or
// shrunk comm. commGetSplitInfo allgathers each rank's (color,key), keeps matching
// colours, and INSERTION-SORTS by key; getParentRanks is pure integer logic.
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

// Scripts the post-allgather table. A real allgather GATHERS our slot, it does not
// write it -- so entry [selfRank] is ASSERTED, not overwritten. That assertion is the
// only oracle for commGetSplitInfo's own writes at init.cc:2494-2495; overwriting the
// slot instead leaves six mutants on those two lines alive. Lives here because
// commSplitInfo is a typedef inside init.cc and exists only in this TU.
void InstallSplitInfoTable(int selfRank, std::vector<std::pair<int, int>> colorKey) {
  g_bootstrapAllGather = [selfRank, colorKey](void*, void* allData, int size) {
    if (size != static_cast<int>(sizeof(commSplitInfo))) {
      ADD_FAILURE() << "allgather element size " << size << ", expected "
                    << sizeof(commSplitInfo);
      return ncclInternalError;
    }
    if (selfRank < 0 || selfRank >= static_cast<int>(colorKey.size())) {
      ADD_FAILURE() << "selfRank " << selfRank << " outside the " << colorKey.size()
                    << "-entry table: the own-slot oracle would be silently skipped";
    }
    auto* info = static_cast<commSplitInfo*>(allData);
    for (size_t i = 0; i < colorKey.size(); ++i) {
      if (static_cast<int>(i) == selfRank) {
        if (info[i].color != colorKey[i].first || info[i].key != colorKey[i].second) {
          ADD_FAILURE() << "rank " << selfRank << " published (" << info[i].color << ","
                        << info[i].key << "), expected (" << colorKey[i].first << ","
                        << colorKey[i].second << ")";
        }
        continue;
      }
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

// NOTE: getParentRanks leaves *myRankRet unwritten when the caller excludes itself;
// its sole caller rejects that first (bsearch at init.cc:4039), so this is missing
// defence-in-depth in the helper, not a reachable defect.
TEST_F(InitMicrotest, GetParentRanks_ExcludingCaller_LeavesMyRankUnwritten) {
  int exclude[1] = {2};
  int out[4] = {-1, -1, -1, -1};
  int nRanks = -1;
  int myRank = 0x5EED;  // sentinel: must survive untouched
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/4, /*parentRank=*/2, exclude,
                                        /*excludeRanksCount=*/1, &nRanks, &myRank, out));
  EXPECT_EQ(3, nRanks);
  EXPECT_EQ(0x5EED, myRank) << "getParentRanks itself does not re-check self-exclusion";
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

// LATENT BUG (init.cc:2540): *nRanksRet is parentRanks - excludeRanksCount, computed
// without reference to the loop, and nothing dedupes or range-checks the list
// (init.cc:4034-4043 checks only non-null, count > 0, self-exclusion). A duplicate
// silently evicts a real rank; the evicted rank then gets myRank == nRanks -- an index
// one past the end, returned with ncclSuccess and handed to child-comm init.
TEST_F(InitMicrotest, GetParentRanks_DuplicateExclusion_EvictsAnExtraRank) {
  int exclude[2] = {1, 1};  // sorted, but the same rank twice
  int out[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, getParentRanks(/*parentRanks=*/4, /*parentRank=*/3, exclude,
                                        /*excludeRanksCount=*/2, &nRanks, &myRank, out));
  EXPECT_EQ(2, nRanks) << "count comes from the arithmetic, not the loop";
  EXPECT_EQ(std::vector<int>({0, 2, 3}), std::vector<int>(out, out + 3))
      << "three ranks were written, but the caller is told there are two";
  EXPECT_EQ(2, myRank);
  EXPECT_GE(myRank, nRanks) << "rank 3's index is one past the end of its own comm";
}

// --- commGetSplitInfo: colour filter + insertion sort by key ---

TEST_F(InitMicrotest, CommGetSplitInfo_NoColor_ReturnsEarlyWithoutTouchingOutputs) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  // Our own slot carries the colour we pass -- NOCOLOR -- not a fabricated 7.
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {NCCL_SPLIT_NOCOLOR, 0}, {7, 2}, {7, 3}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = 0x1234, myRank = 0x5678;  // sentinels
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), NCCL_SPLIT_NOCOLOR,
                                          /*key=*/0, &nRanks, &myRank, parentRanks));
  EXPECT_EQ(0x1234, nRanks) << "NOCOLOR must leave the outputs alone";
  EXPECT_EQ(0x5678, myRank);
}

TEST_F(InitMicrotest, CommGetSplitInfo_SortedKeys_PreservesOrder) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {7, 1}, {7, 2}, {7, 3}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(4, nRanks);
  EXPECT_EQ(1, myRank);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), std::vector<int>(parentRanks, parentRanks + 4));
}

// The one that matters: sorted keys always insert at the end, so the shift loop at
// :2508 never runs and a broken comparator goes unnoticed.
// Mutation ceiling: `r > insert` -> `r >= insert` is EQUIVALENT -- the extra write is
// overwritten by :2510. (It does add an OOB read at insert==0; ASan-only.)
TEST_F(InitMicrotest, CommGetSplitInfo_UnsortedKeys_InsertionSortsByKey) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 3}, {7, 1}, {7, 2}, {7, 0}});
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
  InstallSplitInfoTable(/*selfRank=*/2, {{7, 0}, {9, 1}, {7, 2}, {9, 3}});
  int parentRanks[4] = {0, 0, 0, 0};  // NOT -1: that is what the 0xff memset writes
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/2,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(2, nRanks);
  EXPECT_EQ(std::vector<int>({0, 2}), std::vector<int>(parentRanks, parentRanks + 2));
  EXPECT_EQ(1, myRank);
  EXPECT_EQ(-1, parentRanks[2]) << "the 0xff memset must have filled the tail";
}

// The comparator is `<=`, so an equal key inserts AFTER the incumbent: ties break
// by parent rank. `<` would reverse them.
TEST_F(InitMicrotest, CommGetSplitInfo_EqualKeys_TieBreaksByParentRank) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/3);
  InstallSplitInfoTable(/*selfRank=*/3, {{7, 5}, {7, 5}, {7, 5}, {7, 5}});
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
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {7, 1}, {7, 2}, {7, 3}});
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

// init.cc:1331 is DEAD: the gcd loop at :1314-1317 leaves groupSize dividing every
// node's localRanks, so the check can never be true. Do not try to flip it.
// Escape hatch is groupSize==0, which crashes first (:1316, or :1321 when nNodes==0).
// Operator-triggerable: ERANGE is checked (param.cc:93-97) but the int64->int narrowing
// at :1313 is not, so GROUP_SIZE=0 or any multiple of 2^32 SIGFPEs. Both pinned below.

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
  // EXPECT_DEATH("") would accept ANY death: ::abort(), _exit(1) and a null
  // deref injected at the same spot all satisfy it. Pin the signal so the test
  // actually asserts "divide by zero", and so an inert guard (e.g. an assert()
  // that vanishes under NDEBUG) cannot quietly go back to passing.
  EXPECT_EXIT(ncclP2pSchedule(c.get()), ::testing::KilledBySignal(SIGFPE), "");
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

TEST_F(InitMicrotest, P2pSchedule_MultiNode_Rank1OnNode1_FullScheduleContents) {
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

// `|| localRanks < groupSize` at :1316 is an EQUIVALENT mutant for every reachable
// input: the first disjunct already fired unless localRanks==0, where gcd(g,0)==g makes
// it a no-op. This test pins the nGroupsInNode==0 case skipping the inner loop instead.
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
// Comparison is strcasecmp, so spelling case is irrelevant.
TEST_F(InitMicrotest, GetEnvCtaPolicy_MixedCaseModes_AccumulatesBoth) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO,
            RunCtaPolicyEnv("efficiency|ZeRo"));
}
// NCCL_CTA_POLICY_DEFAULT is 0x00, which makes a lower-cased DEFAULT alongside
// other tokens BIT-INVISIBLE: match-and-contribute-0 and skip-then-let-the-next
// -token-take-the-first-assign-arm produce the same final value, and no test
// captures the log to see the extra "Unknown CTA policy" line. So
// `strcasecmp(token,"DEFAULT") -> strcmp` survives a mixed-case combined input.
// DEFAULT ALONE is the only spelling where 0 and NCCL_CONFIG_UNDEF_INT differ.
TEST_F(InitMicrotest, GetEnvCtaPolicy_LowerCaseDefaultAlone_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("default"));
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
// ===========================================================================
// ncclCommGetUniqueId (init.cc:4160) -- mints a grow handle, hands it back as a uniqueId.
// bootstrapGetUniqueId was a fail-loud abort; bcastGrowHandle had no fake at all and
// only linked because --gc-sections dropped this whole function.
// ===========================================================================

TEST_F(InitMicrotest, CommGetUniqueId_NullComm_ReturnsInvalidArgument) {
  ncclUniqueId id{};
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetUniqueId_impl(nullptr, &id));
}

TEST_F(InitMicrotest, CommGetUniqueId_NullUniqueId_ReturnsInvalidArgument) {
  ReadyComm rc;
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetUniqueId_impl(rc.get(), nullptr));
}

TEST_F(InitMicrotest, CommGetUniqueId_NotReadyComm_ReturnsInvalidArgument) {
  // Without this, deleting NCCLCHECK(ncclCommEnsureReady) at init.cc:4162 survives:
  // a comm still mid-async-init would mint and broadcast a handle.
  ReadyComm rc;
  rc.get()->asyncResult = ncclInProgress;
  ncclUniqueId id{};
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetUniqueId_impl(rc.get(), &id));
  EXPECT_EQ(0, g_bcastGrowHandleCalls) << "must not broadcast for a not-ready comm";
}

TEST_F(InitMicrotest, CommGetUniqueId_HappyPath_CopiesHandleAndBroadcastsAsRoot) {
  ReadyComm rc;
  ncclUniqueId id{};
  std::memset(&id, 0xAB, sizeof(id));  // poison: the memset at init.cc:4171 must clear it
  g_bootstrapHandleMagic = 0xFEEDFACEULL;

  ASSERT_EQ(ncclSuccess, ncclCommGetUniqueId_impl(rc.get(), &id));

  // The handle bootstrapGetUniqueId stamped is what lands in the caller's id.
  ncclBootstrapHandle out{};
  std::memcpy(&out, &id, sizeof(out));
  EXPECT_EQ(0xFEEDFACEULL, out.magic);
  // init.cc:4171 zeroes the WHOLE id first, so the bytes past the handle are 0,
  // not the 0xAB poison.
  for (size_t i = sizeof(ncclBootstrapHandle); i < sizeof(ncclUniqueId); ++i) {
    ASSERT_EQ('\0', id.internal[i]) << "id byte " << i << " was not cleared";
  }
  EXPECT_EQ(1, g_bcastGrowHandleCalls);
  EXPECT_TRUE(g_bcastGrowHandleIsRoot) << "the id owner broadcasts as root";
}

TEST_F(InitMicrotest, CommGetUniqueId_BootstrapGetUniqueIdFails_PropagatesError) {
  ReadyComm rc;
  ncclUniqueId id{};
  g_bootstrapGetUniqueIdResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclCommGetUniqueId_impl(rc.get(), &id));
  EXPECT_EQ(0, g_bcastGrowHandleCalls) << "must not broadcast a handle it failed to mint";
}

TEST_F(InitMicrotest, CommGetUniqueId_BcastGrowHandleFails_PropagatesError) {
  ReadyComm rc;
  ncclUniqueId id{};
  g_bcastGrowHandleResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, ncclCommGetUniqueId_impl(rc.get(), &id));
  EXPECT_EQ(1, g_bcastGrowHandleCalls);
}

// --- ncclCommShrink exit path (:4147-4156) ---
// NVTX3_RANGE_ADD_PAYLOAD at :4152-4153 expands to NOTHING here (the preamble
// neutralizes it), so those lines carry no logic and the guard is 2/4 by design:
// *newcomm != NULL needs a successful child-comm init, and newcomm == NULL is
// UNREACHABLE because it crashes at :4113 first -- a symptom of the bug below.
TEST_F(InitMicrotest, CommShrink_NotReadyComm_ReturnsInvalidArgument) {
  // The sibling readiness check, at init.cc:4044 inside ncclCommInitChildComm.
  // rank 1 so the self-exclusion bsearch at init.cc:4039 does not reject first.
  ReadyComm rc;
  rc.get()->rank = 1;
  rc.get()->asyncResult = ncclInProgress;
  ncclComm_t out = nullptr;
  int exclude[1] = {0};
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommShrink_impl(rc.get(), exclude, /*excludeRanksCount=*/1, &out,
                                /*config=*/nullptr, /*shrinkFlags=*/0));
  EXPECT_EQ(nullptr, out);
}

TEST_F(InitMicrotest, CommShrink_ZeroExcludeCount_ReturnsInvalidArgumentViaExitPath) {
  ReadyComm rc;
  ncclComm_t out = nullptr;  // stays null -> the guard takes its FALSE arm
  int exclude[1] = {0};
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommShrink_impl(rc.get(), exclude, /*excludeRanksCount=*/0, &out,
                                /*config=*/nullptr, /*shrinkFlags=*/0));
  EXPECT_EQ(nullptr, out) << "a failed shrink must not hand back a comm";
}

// LATENT BUG (init.cc:4113): PtrCheck(newcomm) at :4032 rejects a null out-param by
// jumping to exit:, which then dereferences *newcomm -- so ncclCommShrink(.., NULL, ..)
// and ncclCommSplit(.., NULL, ..) SEGFAULT instead of returning ncclInvalidArgument.
// exit: is the only route: fail: ends in `goto exit` (:4126) and is unreachable when
// newcomm is null, so the `if (newcomm)` at :4125 guards nothing here.
// The arg is evaluated at the call site, so the no-op Recorder fake does not mask it.
TEST_F(InitMicrotest, CommShrink_NullNewcomm_DiesOnNullDeref) {
  ReadyComm rc;
  int exclude[1] = {0};
  // Signal AND message: the signal alone still accepts a crash arriving BEFORE the
  // newcomm validation -- a real death by the wrong mechanism. WARN is ungated
  // (debug.h:40), so the text reaches the child's stderr.
  EXPECT_EXIT(ncclCommShrink_impl(rc.get(), exclude, /*excludeRanksCount=*/1, nullptr,
                                  /*config=*/nullptr, /*shrinkFlags=*/0),
              DEATH_BY_SEGV, "newcomm argument is NULL");

  // WHEN FIXED (guard the Recorder call at init.cc:4113 the way `fail:` guards
  // its own store), this is the test:
  // ncclComm_t* nullOut = nullptr;
  // EXPECT_EQ(ncclInvalidArgument,
  //           ncclCommShrink_impl(rc.get(), exclude, 1, nullOut, nullptr, 0));
}

// ===========================================================================
// ncclGetErrorString (init.cc:4369) -- a pure switch, no seams. Full strings, not
// substrings: a prefix check cannot see a changed NCCL_DEBUG level or a dropped
// "(run with ...)" hint, both of which users read in production.
// ===========================================================================
TEST_F(InitMicrotest, GetErrorString_EveryResultCode_HasDistinctMessage) {
  const struct { ncclResult_t code; const char* expect; } kCases[] = {
      {ncclSuccess, "no error"},
      {ncclUnhandledCudaError, "unhandled cuda error (run with NCCL_DEBUG=INFO for details)"},
      {ncclSystemError, "unhandled system error (run with NCCL_DEBUG=INFO for details)"},
      {ncclInternalError, "internal error - please report this issue to the NCCL developers"},
      {ncclInvalidArgument, "invalid argument (run with NCCL_DEBUG=WARN for details)"},
      {ncclInvalidUsage, "invalid usage (run with NCCL_DEBUG=WARN for details)"},
      {ncclRemoteError, "remote process exited or there was a network error"},
      {ncclInProgress, "NCCL operation in progress"},
      {ncclTimeout, "timeout"},
  };
  // A new enumerator in nccl.h without a row here (and a case in init.cc) fails
  // the build rather than silently falling into "unknown result code".
  static_assert(sizeof(kCases) / sizeof(kCases[0]) == ncclNumResults,
                "a new ncclResult_t needs a case in init.cc and a row here");
  std::vector<std::string> seen;
  for (const auto& c : kCases) {
    const char* msg = ncclGetErrorString_impl(c.code);
    ASSERT_NE(nullptr, msg) << "code " << c.code;
    EXPECT_STREQ(c.expect, msg) << "code " << c.code;
    seen.emplace_back(msg);
  }
  // Distinctness too: a copy-paste slip returning one message from two cases
  // would satisfy every per-code check above.
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen.end(), std::unique(seen.begin(), seen.end()))
      << "two result codes share a message";
}

TEST_F(InitMicrotest, GetErrorString_UnmappedCode_FallsBackToUnknown) {
  // ncclNumResults is one past the last mapped enumerator, so it must hit the
  // default arm rather than any case.
  EXPECT_STREQ("unknown result code",
               ncclGetErrorString_impl(static_cast<ncclResult_t>(ncclNumResults)));
  EXPECT_STREQ("unknown result code", ncclGetErrorString_impl(static_cast<ncclResult_t>(-1)));
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
TEST_F(InitMicrotestIsolated, NcclInit_BootstrapNetInitFailure_ReturnsSystemError) {
  // Canonical RUN_ISOLATED_TEST (fork+execv fresh image -> pristine call_once).
  // If isolation failed (cached ncclInit success), the valid-args path would
  // reach ncclAsyncLaunch -> fail-loud abort -> child crash -> RED. Green proves
  // ncclInit genuinely failed via bootstrapNetInit.
  //
  // The env mask comes from InitMicrotest::SetUp and applies in the CHILD too --
  // see the fixture comment above. No mask list is duplicated here.
  RUN_ISOLATED_TEST(
      "Init_NcclInit_BootstrapNetInitFailure",
      []() {
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

// ===========================================================================
// initTransportsRank's supporting cast: four helpers it calls that are already
// compiled into this TU and need no seams. Every one computes a result, so the
// oracle is the whole output -- a "did it return ncclSuccess?" check here would
// be green and prove nothing.
// ===========================================================================

// --- initNvlDomainInfo (init.cc:1272), reached at init.cc:2047 ---
TEST_F(InitMicrotest, InitNvlDomainInfo_CopiesNodeAndRankCounts) {
  auto comm = std::make_unique<ncclComm>();
  comm->nNodes = 3;  // three pairwise-distinct values, so any field-to-field swap is visible
  comm->minLocalRanks = 5;
  comm->maxLocalRanks = 7;
  EXPECT_EQ(ncclSuccess, initNvlDomainInfo(comm.get()));
  EXPECT_EQ(3, comm->nvlDomainInfo.nNvlDomains);
  EXPECT_EQ(5, comm->nvlDomainInfo.minRanksPerNvlDomain);
  EXPECT_EQ(7, comm->nvlDomainInfo.maxRanksPerNvlDomain);
}

TEST_F(InitMicrotest, InitNvlDomainInfo_SingleNodeUniform_MinEqualsMax) {
  auto comm = std::make_unique<ncclComm>();
  comm->nNodes = 1;
  comm->minLocalRanks = comm->maxLocalRanks = 8;
  EXPECT_EQ(ncclSuccess, initNvlDomainInfo(comm.get()));
  EXPECT_EQ(1, comm->nvlDomainInfo.nNvlDomains);
  EXPECT_EQ(8, comm->nvlDomainInfo.minRanksPerNvlDomain);
  EXPECT_EQ(8, comm->nvlDomainInfo.maxRanksPerNvlDomain);
}

// --- rcclComputeCheapPostSendFenceOff (rccl_common.h:261), reached at init.cc:1804 ---
// Returns 1 = full __threadfence_system(), 0 = cheap fence. Each test picks an arch whose
// AUTO verdict is the opposite of the expected answer, so an arm cannot pass by coincidence.
TEST_F(InitMicrotest, CheapPostSendFenceOff_NoUncachedSupport_ForcesFullFence) {
  EXPECT_EQ(1, rcclComputeCheapPostSendFenceOff(940, /*param=*/2, /*uncachedMemSupported=*/false));
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_ParamTwo_ForcesCheapFenceOnAnyArch) {
  EXPECT_EQ(0, rcclComputeCheapPostSendFenceOff(950, /*param=*/2, true));  // 950 auto would be 1
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_ParamOne_ForcesFullFenceOnAnyArch) {
  EXPECT_EQ(1, rcclComputeCheapPostSendFenceOff(940, /*param=*/1, true));  // 940 auto would be 0
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_AutoGfx942_EnablesCheapFence) {
  EXPECT_EQ(0, rcclComputeCheapPostSendFenceOff(940, /*param=*/0, true));
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_AutoGfx1250_EnablesCheapFence) {
  EXPECT_EQ(0, rcclComputeCheapPostSendFenceOff(1250, /*param=*/0, true));
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_AutoGfx950_KeepsFullFence) {
  EXPECT_EQ(1, rcclComputeCheapPostSendFenceOff(950, /*param=*/0, true));
}
TEST_F(InitMicrotest, CheapPostSendFenceOff_AutoUnknownArch_KeepsFullFence) {
  EXPECT_EQ(1, rcclComputeCheapPostSendFenceOff(0, /*param=*/0, true));
}

// --- ncclP2pChannelForPart (device.h:388), reached at init.cc:2293/2298 ---
namespace {
// The channel each part maps to. Asserting the WHOLE vector is the point: a single-entry
// oracle survives almost any base/shift mutation, since one entry often still matches.
std::vector<int> P2pPartMap(int nChannels, int base, int nParts, int nNodes, int shiftSize) {
  std::vector<int> out;
  for (int p = 0; p < nParts; ++p)
    out.push_back(ncclP2pChannelForPart(nChannels, base, p, nParts, nNodes, shiftSize));
  return out;
}
}  // namespace

TEST_F(InitMicrotest, P2pChannelForPart_MultiNodeNoShift_BitReversesAndRotatesByBase) {
  // shiftSize==-1 && nNodes>2 -> reverseBits(part, log2(8)) rotated by base=3.
  EXPECT_EQ(std::vector<int>({3, 7, 5, 1, 4, 0, 6, 2}),
            P2pPartMap(/*nChannels=*/8, /*base=*/3, /*nParts=*/8, /*nNodes=*/4, /*shiftSize=*/-1));
}
TEST_F(InitMicrotest, P2pChannelForPart_BitReversal_IsAPermutationOfAllChannels) {
  auto map = P2pPartMap(8, /*base=*/3, /*nParts=*/8, /*nNodes=*/4, /*shiftSize=*/-1);
  std::sort(map.begin(), map.end());
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7}), map);  // every channel used exactly once
}
TEST_F(InitMicrotest, P2pChannelForPart_MultiNodeWithShift_UsesLinearShiftMapping) {
  EXPECT_EQ(std::vector<int>({4, 6, 0, 2}),
            P2pPartMap(/*nChannels=*/8, /*base=*/2, /*nParts=*/4, /*nNodes=*/4, /*shiftSize=*/1));
}
TEST_F(InitMicrotest, P2pChannelForPart_TwoNodes_UsesBaseTimesNPartsMapping) {
  EXPECT_EQ(std::vector<int>({6, 7}),
            P2pPartMap(/*nChannels=*/8, /*base=*/3, /*nParts=*/2, /*nNodes=*/2, /*shiftSize=*/1));
}
TEST_F(InitMicrotest, P2pChannelForPart_SingleNodeNoShift_StillUsesBaseTimesNParts) {
  // shiftSize==-1 alone is not enough: nNodes>2 must also hold, else the third arm wins.
  EXPECT_EQ(std::vector<int>({2, 3}),
            P2pPartMap(/*nChannels=*/8, /*base=*/1, /*nParts=*/2, /*nNodes=*/1, /*shiftSize=*/-1));
}
TEST_F(InitMicrotest, P2pChannelForPart_NNodesTwoVsThree_SelectsDifferentArms) {
  // Pins the `nNodes > 2` boundary: a `>=` mutation makes these two maps identical.
  EXPECT_EQ(std::vector<int>({4, 5, 6, 7}), P2pPartMap(8, 3, 4, /*nNodes=*/2, /*shiftSize=*/1));
  EXPECT_EQ(std::vector<int>({5, 7, 1, 3}), P2pPartMap(8, 3, 4, /*nNodes=*/3, /*shiftSize=*/1));
}

// --- collNetSupport (coll_net.h:87), reached at init.cc:1614/1868 ---
TEST_F(InitMicrotest, CollNetSupport_NoCollNetPlugin_ReturnsZero) {
  auto comm = std::make_unique<ncclComm>();
  EXPECT_EQ(0, collNetSupport(comm.get()));
}
TEST_F(InitMicrotest, CollNetSupport_CollNetPluginPresent_ReturnsOne) {
  auto comm = std::make_unique<ncclComm>();
  ncclCollNet_t collNet{};
  comm->ncclCollNet = &collNet;
  EXPECT_EQ(1, collNetSupport(comm.get()));
}

// ===========================================================================
// initTransportsRank (init.cc:1386) -- error-injection ladder, rung 1.
// 52 of its 121 calls are still fail-loud stubs, so instead of driving the happy path each test
// arms the LAST seam it needs to fail and everything before that runs for real. Covers :1462-1576,
// stopping at ncclTopoGetSystem, whose seam defaults to failure to serve as the terminator.
// Only :1488 returns without reaching the single exit: block, so g_ncclOsCpuCountCalls -- which
// exit::2403 always bumps -- is the oracle for "cleanup ran".
// ===========================================================================
namespace {

// Survives fillInfo (sharedRes for ginState, ncclNet for the dmaBuf probe) and the exit: block.
// Owns the peerInfo table the UUT ncclCalloc's at :1464 -- production frees it in commFree, so
// without this destructor every test here leaks it.
class TransportsRankComm {
 public:
  TransportsRankComm(int nRanks, int rank, const char* archName = "gfx942")
      : comm_(new ncclComm{}), sr_(new ncclSharedResources{}), net_(new ncclNet_t{}),
        archName_(archName) {
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->sharedRes = sr_.get();  // owner stays null, so exit::2405 short-circuits before ncclCuMemEnable
    comm_->ncclNet = net_.get();
    comm_->archName = archName_.empty() ? nullptr : &archName_[0];
    comm_->commHash = 0xC0FFEEULL;
    comm_->compCap = 90;
    std::memset(timers_, 0, sizeof(timers_));
  }
  ~TransportsRankComm() { free(comm_->peerInfo); }
  TransportsRankComm(const TransportsRankComm&) = delete;
  TransportsRankComm& operator=(const TransportsRankComm&) = delete;
  ncclComm* get() { return comm_.get(); }

  // Point ncclTopoGetSystem at a system this object owns, so the ladder can run past :1576 into the
  // topology block. Returns it so tests can assert the fields :1577-1589 and :1622-1639 stamp on it.
  // Nothing in :1386-1648 frees comm->topo (commFree does, and no test calls it), so ownership stays here.
  ncclTopoSystem* installTopo() {
    topo_ = std::make_unique<ncclTopoSystem>();
    ncclTopoSystem* t = topo_.get();
    g_ncclTopoGetSystem = [t](ncclComm*, ncclTopoSystem** out, const char*) {
      if (out) *out = t;  // the :1573 dump-file call site passes NULL
      return ncclSuccess;
    };
    return t;
  }
  uint64_t* timers() { return timers_; }
  int rank() const { return comm_->rank; }
  int nRanks() const { return comm_->nRanks; }

 private:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclSharedResources> sr_;
  std::unique_ptr<ncclNet_t> net_;
  std::unique_ptr<ncclTopoSystem> topo_;
  std::string archName_;
  uint64_t timers_[TIMERS_INIT_COUNT];
};

// Per-peer knobs for the scripted AllGather1. Defaults describe the dullest possible peer: same
// node as self, its own process, its own GPU, matching version -- so nothing fires unless asked.
// The entry at selfRank is ignored; fillInfo owns that slot.
struct PeerSpec {
  int node = 0;         // equal values share a hostHash; 0 == self's node
  int proc = -1;        // -1 = its own process; equal values share a pidHash; 0 == self's process
  int version = -1;     // -1 = same as self
  int cuMemSupport = 1;
  int mloPart = -1;     // -1 == NCCL_TOPO_UNDEF, i.e. not an MLOPart GPU
  int nvmlDev = -1;     // -1 = its own device; equal values collide for the :1482 check
  int uuidTag = -1;     // -1 = its own UUID; equal values (>=1) make two peers the same GPU
  int compCap = -1;     // -1 = same as self
  ncclComm* comm = nullptr;  // only read for same-process peers (:1527, :1549)
};

// Scripts g_bootstrapAllGather for the sizeof(ncclPeerInfo) site at :1466.
//
// It deliberately does NOT write the calling rank's slot. fillInfo filled it one line earlier and
// a real allgather gathers your contribution rather than inventing it; overwriting it would erase
// the oracle for everything fillInfo computed. Peer values are derived FROM the self slot so the
// modelled table stays self-consistent. ADD_FAILURE (not EXPECT_*) because this runs inside a
// std::function, where an EXPECT_ reports at a confusing site and lets the UUT run on bad data.
void InstallPeerInfoAllGather(TransportsRankComm& c, std::vector<PeerSpec> specs) {
  const int selfRank = c.rank();
  const int nranks = c.nRanks();
  if (static_cast<int>(specs.size()) != nranks || selfRank < 0 || selfRank >= nranks) {
    ADD_FAILURE() << "InstallPeerInfoAllGather: " << specs.size() << " specs for nranks " << nranks
                  << ", selfRank " << selfRank << " -- the self-slot oracle would be skipped silently";
    return;
  }
  g_bootstrapAllGather = [specs, selfRank, nranks](void*, void* allData, int size) -> ncclResult_t {
    if (size != static_cast<int>(sizeof(ncclPeerInfo))) {
      ADD_FAILURE() << "AllGather1 payload size " << size << ", expected sizeof(ncclPeerInfo)";
      return ncclInternalError;
    }
    auto* info = static_cast<ncclPeerInfo*>(allData);
    const ncclPeerInfo self = info[selfRank];  // copied first: every peer is derived from it
    if (self.version != NCCL_VERSION_CODE) ADD_FAILURE() << "fillInfo did not stamp version";
    if (self.comm == nullptr) ADD_FAILURE() << "fillInfo did not stamp comm (needed by :1549)";
    for (int i = 0; i < nranks; ++i) {
      if (i == selfRank) continue;  // assert-only above; never overwrite our own contribution
      const PeerSpec& s = specs[i];
      info[i] = ncclPeerInfo{};
      info[i].rank = i;
      info[i].version = s.version < 0 ? self.version : s.version;
      info[i].hostHash = self.hostHash + s.node;
      info[i].pidHash = self.pidHash + (s.proc < 0 ? 1000 + i : s.proc);
      info[i].cuMemSupport = s.cuMemSupport;
      info[i].mloPart = s.mloPart;
      info[i].nvmlDev = s.nvmlDev < 0 ? 100 + i : s.nvmlDev;  // self is 0, so -1 never collides
      std::memset(&info[i].gpuUuid, 0, sizeof(info[i].gpuUuid));
      reinterpret_cast<unsigned char*>(&info[i].gpuUuid)[0] =
          static_cast<unsigned char>(s.uuidTag < 0 ? i + 1 : s.uuidTag);  // 1-based: never matches self's zeros
      info[i].cudaCompCap = s.compCap < 0 ? self.cudaCompCap : s.compCap;
      info[i].comm = s.comm;
    }
    return ncclSuccess;
  };
}

// Override specific NCCL_PARAM/RCCL_PARAM values; everything else keeps its compiled-in default.
void SetParams(std::vector<std::pair<std::string, int64_t>> overrides) {
  g_loadParam = [overrides](const char* env, int64_t deft) {
    for (const auto& o : overrides)
      if (o.first == env) return o.second;
    return deft;
  };
}
}  // namespace

// --- AllGather1: allocation, fillInfo, the allgather itself (init.cc:1462-1466) ---

TEST_F(InitMicrotest, InitTransportsRank_PeerInfoCallocFails_ReturnsSystemError) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  g_callocFailAt = 0;  // the UUT's :1464 is the first ncclCalloc this test reaches
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);  // left through fail: -> exit:
}

TEST_F(InitMicrotest, InitTransportsRank_FillInfoFails_PropagatesAndSkipsAllGather) {
  TransportsRankComm c(4, 0);
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) { return hipErrorInvalidValue; };
  g_bootstrapAllGather = [](void*, void*, int) -> ncclResult_t {
    ADD_FAILURE() << "AllGather1 must not run after fillInfo failed";
    return ncclInternalError;
  };
  EXPECT_EQ(ncclUnhandledCudaError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_AllGatherFails_PropagatesAndLeavesPeerInfoInvalid) {
  TransportsRankComm c(4, 0);  // g_bootstrapAllGather defaults to ncclInternalError
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->peerInfoValid);  // :1467 sits past the failure point
}

TEST_F(InitMicrotest, InitTransportsRank_AllGatherSucceeds_MarksPeerInfoValid) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));  // stops at :1576
  EXPECT_TRUE(c.get()->peerInfoValid);  // positive anchor for the negative assertion above
}

// --- The peer-scan loop (init.cc:1470-1495) ---

TEST_F(InitMicrotest, InitTransportsRank_PeerVersionMismatch_WarnsWithBothRanksAndVersions) {
  TransportsRankComm c(4, /*rank=*/1);
  std::vector<PeerSpec> specs(4);
  specs[2].version = 12345;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  // Anchor on the VALUES, not the label: LogHas(log, "Mismatched") passes whatever the ranks were.
  const std::string want = "rank 2 version 12345 rank 1 version " + std::to_string(NCCL_VERSION_CODE);
  EXPECT_TRUE(LogHas(log, want.c_str())) << "actual log:\n" << log;
}

// Both need g_cuMemEnable armed: fillInfo stamps OUR slot from it (:1062) and the scan at :1478
// reads every slot including our own, so with the default of 0 self clears comm->cuMemSupport and
// the peer under test proves nothing.
TEST_F(InitMicrotest, InitTransportsRank_PeerWithoutCuMemSupport_ClearsCommCuMemSupport) {
  TransportsRankComm c(4, 0);
  g_cuMemEnable = [] { return 1; };
  std::vector<PeerSpec> specs(4);
  specs[2].cuMemSupport = 0;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->cuMemSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_AllPeersCuMemSupport_KeepsCommCuMemSupportSet) {
  TransportsRankComm c(4, 0);
  g_cuMemEnable = [] { return 1; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->cuMemSupport);  // :1469 sets it, the loop never clears it
}

TEST_F(InitMicrotest, InitTransportsRank_PeerWithMloPart_SetsHasMloPart) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[3].mloPart = 2;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(c.get()->hasMloPart);
}

TEST_F(InitMicrotest, InitTransportsRank_NoPeerWithMloPart_LeavesHasMloPartUnset) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->hasMloPart);
}

TEST_F(InitMicrotest, InitTransportsRank_ComputesMinAndMaxCompCapAcrossPeers) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[1].compCap = 42;   // asymmetric around self's 90, so swapping min/max is visible
  specs[2].compCap = 110;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(42, c.get()->minCompCap);
  EXPECT_EQ(110, c.get()->maxCompCap);
}

// --- The duplicate-GPU guard (init.cc:1484-1489) ---

TEST_F(InitMicrotest, InitTransportsRank_DuplicateGpuUuidSameHost_ReturnsInvalidUsageAndSkipsCleanup) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[1].uuidTag = 9;  // ranks 1 and 2: same node (default) and the same GPU UUID
  specs[2].uuidTag = 9;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  EXPECT_TRUE(LogHas(log, "Multiple Ranks are using the same GPU/Partition")) << "actual log:\n" << log;
  // :1488 is the ONE place that returns without reaching exit:. A return-code oracle cannot see this.
  EXPECT_EQ(0, g_ncclOsCpuCountCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_DuplicateGpuUuidDifferentHosts_IsAllowed) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[1].uuidTag = 9;
  specs[2].uuidTag = 9;
  specs[2].node = 1;  // same UUID but a different host -- the hostHash conjunct at :1484 saves it
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));  // ran on to :1576
  });
  EXPECT_FALSE(LogHas(log, "Multiple Ranks are using the same GPU/Partition")) << "actual log:\n" << log;
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);  // positive anchor: it really did reach exit:
}

TEST_F(InitMicrotest, InitTransportsRank_DuplicateGpuUuid_MultiRankGpuEnabled_Continues) {
  TransportsRankComm c(4, 0);
  SetParams({{"MULTI_RANK_GPU_ENABLE", 1}});
  std::vector<PeerSpec> specs(4);
  specs[1].uuidTag = 9;
  specs[2].uuidTag = 9;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  EXPECT_FALSE(LogHas(log, "Multiple Ranks are using the same GPU/Partition")) << "actual log:\n" << log;
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
}

// --- hasMultiRankNvml (init.cc:1482) ---
// PINS CURRENT BEHAVIOUR, WHICH LOOKS WRONG: the assignment is `=`, not `|=`, inside the (i,j)
// double loop, so only the FINAL pair survives and an earlier collision is erased. Consumed by
// src/transport/nvls.cc:252. Reported separately; these two tests document what ships today.
TEST_F(InitMicrotest, InitTransportsRank_MultiRankNvml_EarlyCollisionOverwrittenByLastPair) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[1].nvmlDev = 7;  // ranks 1 and 2 really do share a device on the same host...
  specs[2].nvmlDev = 7;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->hasMultiRankNvml);  // ...but the last pair (3,2) does not, and it wins
}

TEST_F(InitMicrotest, InitTransportsRank_MultiRankNvml_LastPairCollision_IsTheOnlyOneObserved) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[2].nvmlDev = 7;  // the final (i,j) pair examined is (3,2)
  specs[3].nvmlDev = 7;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(c.get()->hasMultiRankNvml);
}

// --- The MNNVL scope test (init.cc:1500-1510) ---
// mnnvlEnable==1 forces the check; ==0 forbids it; anything else (default 2) means "auto", which
// needs (multi-node OR gfx1250) AND p2pLevel != 0. g_ncclMnnvlCheckCalls is the whole oracle --
// the return code is identical either way.

TEST_F(InitMicrotest, InitTransportsRank_UserP2pLevelFails_PropagatesAndSkipsMnnvlCheck) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  g_ncclGetUserP2pLevel = [](int*) { return ncclInvalidArgument; };
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlForcedOn_ChecksEvenOnSingleNode) {
  TransportsRankComm c(4, 0);
  SetParams({{"MNNVL_ENABLE", 1}});
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));  // all one node, arch gfx942
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlForcedOff_SkipsCheckEvenOnMultiNode) {
  TransportsRankComm c(4, 0);
  SetParams({{"MNNVL_ENABLE", 0}});  // the zero-valued arm, tested alone: 0 and "auto" differ only here
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoMultiNode_ChecksAndCountsNodes) {
  TransportsRankComm c(4, 0);
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;  // one peer off-node -> nNodes becomes 2 at :1477
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);  // the only observable that nNodes>1 was computed
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoSingleNode_SkipsCheck) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoGfx1250SingleNode_Checks) {
  TransportsRankComm c(4, 0, /*archName=*/"gfx1250");
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);  // single node, but the arch arm at :1504 opens auto scope
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoP2pLevelZero_SkipsCheckDespiteMultiNode) {
  TransportsRankComm c(4, 0);
  g_ncclGetUserP2pLevel = [](int* level) { *level = 0; return ncclSuccess; };
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoNullArchName_DoesNotDereference) {
  TransportsRankComm c(4, 0, /*archName=*/"");  // comm->archName == NULL; :1504 must short-circuit
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlCheckFails_PropagatesError) {
  TransportsRankComm c(4, 0);
  SetParams({{"MNNVL_ENABLE", 1}});
  g_ncclMnnvlCheckResult = ncclSystemError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
}

// --- The intra-process block (init.cc:1512-1565) ---

TEST_F(InitMicrotest, InitTransportsRank_SameHostAndPid_ClearsNvlsRegSupport) {
  TransportsRankComm c(4, 0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}});  // else :1545 forces it straight back to 1
  std::vector<PeerSpec> specs(4);
  auto peer1 = std::make_unique<ncclComm>();  // heap: ncclComm embeds channels[MAXCHANNELS], far too big for the stack
  specs[1].proc = 0;  // shares our process, so the :1533 inner scan finds a colliding pair
  specs[1].comm = peer1.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_AllDistinctProcesses_KeepsNvlsRegSupport) {
  TransportsRankComm c(4, 0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}});
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->nvlsRegSupport);  // positive anchor for the test above
}

TEST_F(InitMicrotest, InitTransportsRank_SingleProcMemRegEnabled_RestoresNvlsRegSupport) {
  TransportsRankComm c(4, 0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 1}});
  std::vector<PeerSpec> specs(4);
  auto peer1 = std::make_unique<ncclComm>();
  specs[1].proc = 0;  // the scan clears it at :1536; :1545 then puts it back
  specs[1].comm = peer1.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlComm_ClearsNvlsRegSupportBeforeSingleProcOverride) {
  TransportsRankComm c(4, 0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 1}});  // would set it to 1, but MNNVL wins the else-if
  c.get()->MNNVL = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_IntraProcPeers_LinkIntraNextInReverseRankOrder) {
  TransportsRankComm c(4, /*rank=*/0);
  auto peer1 = std::make_unique<ncclComm>();
  auto peer2 = std::make_unique<ncclComm>();
  auto peer3 = std::make_unique<ncclComm>();
  std::vector<PeerSpec> specs(4);
  specs[1].proc = 0; specs[1].comm = peer1.get();
  specs[2].proc = 0; specs[2].comm = peer2.get();
  specs[3].proc = 0; specs[3].comm = peer3.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->intraRank);
  EXPECT_EQ(4, c.get()->intraRanks);
  EXPECT_EQ(c.get(), c.get()->intraComm0);
  // :1527-1528 pushes each peer onto the head, so the chain comes out in reverse rank order.
  // Asserting the whole chain, not just intraNext != NULL, is what makes a link-order bug visible.
  EXPECT_EQ(peer3.get(), c.get()->intraNext);
  EXPECT_EQ(peer2.get(), peer3->intraNext);
  EXPECT_EQ(peer1.get(), peer2->intraNext);
  EXPECT_EQ(nullptr, peer1->intraNext);
}

// Self is NOT the first same-process rank here, which is the only shape that separates
// `intraProcRank = intraProcRanks` (:1524) from `= i`: with rank 0 leading they are both 0.
TEST_F(InitMicrotest, InitTransportsRank_SelfNotFirstInProcess_IntraRankIsTheLocalIndex) {
  TransportsRankComm c(4, /*rank=*/2);
  auto peer1 = std::make_unique<ncclComm>();
  auto peer3 = std::make_unique<ncclComm>();
  std::vector<PeerSpec> specs(4);
  specs[1].proc = 0; specs[1].comm = peer1.get();  // rank 0 stays in its own process, so ranks
  specs[3].proc = 0; specs[3].comm = peer3.get();  // 1,2,3 form the group and rank 1 leads it
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->intraRank);   // local index within the process, not the global rank 2
  EXPECT_EQ(3, c.get()->intraRanks);
  EXPECT_EQ(peer1.get(), c.get()->intraComm0);
  EXPECT_EQ(nullptr, c.get()->intraNext);  // only intraProcRank0 builds the chain, and that is rank 1
}

// The guard at :1549 is `intraProcRank == -1 || intraProcRank0 == -1 || peerInfo[...].comm == NULL`.
// Only the first and third disjuncts are reachable. The second is DEAD: intraProcRank is assigned
// only inside the same `if` body that assigns intraProcRank0, so intraProcRank != -1 implies
// intraProcRanks was incremented at least once, which implies intraProcRank0 != -1. Do not try to
// cover it. The first needs an empty comm -- see the ZeroRanks test below.
TEST_F(InitMicrotest, InitTransportsRank_IntraProcRank0PeerHasNullComm_ReturnsInternalErrorAndWarns) {
  TransportsRankComm c(4, /*rank=*/1);
  std::vector<PeerSpec> specs(4);
  specs[0].proc = 0;  // rank 0 shares our process but never registered its comm; :1526 cannot fire
  specs[0].comm = nullptr;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  // The WARN is the discriminator: ncclInternalError alone is also what the :1576 terminator returns.
  EXPECT_TRUE(LogHas(log, "intraProcRank 1 intraProcRanks 2 intraProcRank0 0")) << "actual log:\n" << log;
}

// nRanks == 0 is rejected upstream by argcheck, so this models a defensive guard rather than a
// reachable production state -- but it is memory-safe (:1464 still allocates nranks+1 entries) and
// it is the only way to reach the `intraProcRank == -1` disjunct, the scan loop never running.
TEST_F(InitMicrotest, InitTransportsRank_ZeroRanks_ReturnsInternalErrorFromIntraProcGuard) {
  TransportsRankComm c(/*nRanks=*/0, /*rank=*/0);
  g_bootstrapAllGather = [](void*, void*, int) { return ncclSuccess; };  // nothing to gather
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  EXPECT_TRUE(LogHas(log, "intraProcRank -1 intraProcRanks 0 intraProcRank0 -1")) << "actual log:\n" << log;
}

// --- The ladder terminator and the exit: block (init.cc:1569-1576, :2402-2419) ---

TEST_F(InitMicrotest, InitTransportsRank_TopoGetSystemFails_PropagatesAndRunsCleanup) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  g_ncclTopoGetSystem = [](ncclComm*, ncclTopoSystem**, const char*) { return ncclSystemError; };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));  // not the default error
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);          // fail: fell through to exit:
  EXPECT_EQ(0u, g_ncclOsSetAffinityMasks.size());  // cpu count 0, so :2404 stayed unreached
}

TEST_F(InitMicrotest, InitTransportsRank_NoTopoDumpFile_PassesNullPathToTopoGetSystem) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  std::vector<std::string> paths;
  g_ncclTopoGetSystem = [&paths](ncclComm*, ncclTopoSystem**, const char* f) {
    paths.push_back(f ? f : "<null>");
    return ncclSystemError;
  };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(std::vector<std::string>({"<null>"}), paths);  // :1572 false, so only :1576 runs
}

TEST_F(InitMicrotest, InitTransportsRank_TopoDumpFileSet_PassesThatPathToTopoGetSystem) {
  TransportsRankComm c(4, 0);
  SetMicroEnv("NCCL_TOPO_DUMP_FILE", "/tmp/rccl-topo-microtest.xml");
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  std::vector<std::string> paths;
  g_ncclTopoGetSystem = [&paths](ncclComm*, ncclTopoSystem**, const char* f) {
    paths.push_back(f ? f : "<null>");
    return ncclSystemError;
  };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  // Recording the argument is the only way to tell the :1573 call site from the :1576 one.
  EXPECT_EQ(std::vector<std::string>({"/tmp/rccl-topo-microtest.xml"}), paths);
}

TEST_F(InitMicrotest, InitTransportsRank_CpuAffinitySet_RestoresThatAffinityAtExit) {
  TransportsRankComm c(4, 0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  CPU_ZERO(&c.get()->cpuAffinity);
  CPU_SET(5, &c.get()->cpuAffinity);
  g_ncclOsCpuCountValue = 1;  // non-zero, so exit::2404 restores the mask
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  // Only exit::2404 runs here -- :1607-1610 are past the rung-1 terminator.
  ASSERT_EQ(1u, g_ncclOsSetAffinityMasks.size());
  // Assert the MASK, not just the call: forwarding any other affinity would otherwise pass.
  EXPECT_TRUE(CPU_ISSET(5, &g_ncclOsSetAffinityMasks[0]));
  EXPECT_EQ(1, CPU_COUNT(&g_ncclOsSetAffinityMasks[0]));
}

// ===========================================================================
// initTransportsRank rung 2: topology detection, CPU affinity, CollNet and the
// host-index computation (src :1576-1648). Same ladder -- ncclTopoGetSystem now
// succeeds and hands back a test-owned ncclTopoSystem, and ncclTopoCompute
// (:1648) takes over as the terminator with its failure default.
// ===========================================================================

// --- Topology detection (init.cc:1576-1603) ---

TEST_F(InitMicrotest, InitTransportsRank_TopoDetected_StampsTopoDefaults) {
  TransportsRankComm c(4, 0);
  ncclTopoSystem* topo = c.installTopo();
  g_tuningIndexValue = 7;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));  // stops at :1648
  EXPECT_EQ(topo, c.get()->topo);
  EXPECT_EQ(4, topo->nRanks);
  EXPECT_EQ(7, topo->tuning);
  EXPECT_EQ(-2, topo->netGdrLevel);
  EXPECT_FALSE(topo->pivotA2AEnabled);
  EXPECT_EQ(0, topo->pivotA2ANumBiRings);
  EXPECT_FALSE(topo->ll128Enabled);
  EXPECT_FALSE(topo->treeDefined);
}

TEST_F(InitMicrotest, InitTransportsRank_TuningIndex_IsLookedUpByCommArchName) {
  TransportsRankComm c(4, 0, /*archName=*/"gfx90a");
  ncclTopoSystem* topo = c.installTopo();
  g_tuningIndexValue = 3;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(3, topo->tuning);
  EXPECT_EQ("gfx90a", g_tuningIndexLastArch);  // pins that :1577 forwards archName, not a constant
}

TEST_F(InitMicrotest, InitTransportsRank_ComputePathsFailsBeforeTrim_StopsAtFirstCall) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoComputePathsFailAt = 0;  // the :1591 call
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputePathsCalls);  // never reached the post-trim recompute
}

TEST_F(InitMicrotest, InitTransportsRank_ComputePathsFailsAfterTrim_RunsBothCalls) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoComputePathsFailAt = 1;  // the :1596 recompute; only a call index separates it from :1591
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoComputePathsCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_TrimSystemFails_PropagatesBetweenThePathsCalls) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoTrimSystemResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputePathsCalls);  // trim sits between :1591 and :1596
}

TEST_F(InitMicrotest, InitTransportsRank_TopoSearchInitFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoSearchInitResult = ncclInvalidUsage;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoComputePathsCalls);  // both path computations already ran
}

TEST_F(InitMicrotest, InitTransportsRank_TopoComputeCommCpuFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoComputeCommCPUResult = ncclUnhandledCudaError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclUnhandledCudaError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_TopoPrintFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoPrintResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(-1, g_ncclTopoGetCpuAffinityLastRank);  // :1607 is past the failure
}

// --- CPU affinity (init.cc:1607-1611) ---

TEST_F(InitMicrotest, InitTransportsRank_TopoGetCpuAffinityFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity*) { return ncclSystemError; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_CpuAffinityLookedUpForThisRank) {
  TransportsRankComm c(4, /*rank=*/2);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoGetCpuAffinityLastRank);  // :1607 forwards comm->rank, not a constant
}

TEST_F(InitMicrotest, InitTransportsRank_EmptyCpuAffinity_SkipsSaveAndApply) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));  // cpu count 0 by default
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0u, g_ncclOsSetAffinityMasks.size());  // neither :1610 nor exit::2404 ran
}

TEST_F(InitMicrotest, InitTransportsRank_NonEmptyCpuAffinity_AppliesAtBothSites) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity* a) {
    CPU_ZERO(a); CPU_SET(6, a); return ncclSuccess;
  };
  g_ncclOsCpuCountValue = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  ASSERT_EQ(2u, g_ncclOsSetAffinityMasks.size());  // :1610 on the way in, exit::2404 on the way out
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsSetAffinityMasks[0]));
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsSetAffinityMasks[1]));
  EXPECT_TRUE(CPU_ISSET(6, &c.get()->cpuAffinity));  // :1607 wrote through to the comm
}

TEST_F(InitMicrotest, InitTransportsRank_OsGetAffinityFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclOsGetAffinity = [](ncclAffinity*) { return ncclSystemError; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_OsSetAffinityFails_Propagates) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclOsSetAffinityResult = ncclInvalidUsage;  // :1610 is NCCLCHECKGOTO'd, unlike exit::2404
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
}

// affinitySave (:1395) is written at :1609 and never read: exit::2404 re-applies comm->cpuAffinity,
// not the saved mask, so the caller's original affinity is deliberately NOT restored. Pinned here so a
// future "restore affinitySave" change has to update a test rather than pass silently.
TEST_F(InitMicrotest, InitTransportsRank_AffinitySaveIsCapturedButNeverRestored) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity* a) {
    CPU_ZERO(a); CPU_SET(6, a); return ncclSuccess;   // the GPU-local mask
  };
  g_ncclOsGetAffinity = [](ncclAffinity* a) {
    CPU_ZERO(a); CPU_SET(9, a); return ncclSuccess;   // the caller's original mask
  };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  // BOTH call sites must be checked: exit::2404 always passes comm->cpuAffinity, so asserting only
  // the final mask cannot see :1610 handing over affinitySave instead.
  ASSERT_EQ(2u, g_ncclOsSetAffinityMasks.size());
  for (const ncclAffinity& m : g_ncclOsSetAffinityMasks) {
    EXPECT_TRUE(CPU_ISSET(6, &m));   // the GPU-local mask, at :1610 and at exit::2404...
    EXPECT_FALSE(CPU_ISSET(9, &m));  // ...and the saved mask is never re-applied anywhere
  }
}

// --- CollNet, host index and the ring graph (init.cc:1613-1648) ---

TEST_F(InitMicrotest, InitTransportsRank_NoCollNetPlugin_ClearsCollnetEnable) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  c.get()->config.collnetEnable = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->config.collnetEnable);
}

TEST_F(InitMicrotest, InitTransportsRank_CollNetPluginPresent_KeepsCollnetEnable) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  ncclCollNet_t collNet{};
  c.get()->ncclCollNet = &collNet;
  c.get()->config.collnetEnable = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->config.collnetEnable);  // positive anchor for the test above
}

TEST_F(InitMicrotest, InitTransportsRank_NvlsInitFails_ReturnsWithoutRunningCleanup) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclNvlsInitResult = ncclSystemError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclNvlsInitCalls);
  // :1618 is a bare NCCLCHECK, not NCCLCHECKGOTO -- the second cleanup bypass in this function.
  // :1608 already called ncclOsCpuCount once; reaching exit: would make it two.
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_ComputesHostCountAndHostIndex) {
  TransportsRankComm c(4, /*rank=*/3);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[0].node = 1;  // hosts, in rank order: B B C A -- self (rank 3) is on host A, seen last,
  specs[1].node = 1;  // so nHosts and hostIdx take different values and neither is 0
  specs[2].node = 2;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(3, topo->nHosts);
  EXPECT_EQ(2, topo->hostIdx);
}

TEST_F(InitMicrotest, InitTransportsRank_SingleHost_HostIndexIsZero) {
  TransportsRankComm c(4, 0);
  ncclTopoSystem* topo = c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, topo->nHosts);
  EXPECT_EQ(0, topo->hostIdx);
}

TEST_F(InitMicrotest, InitTransportsRank_UniformRanksPerHost_KeepsPresetTopoMatching) {
  TransportsRankComm c(4, 0);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;  // 2 hosts x 2 ranks
  specs[3].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(topo->skipPresetTopoMatching);
}

TEST_F(InitMicrotest, InitTransportsRank_NonUniformRanksPerHost_SkipsPresetTopoMatching) {
  TransportsRankComm c(4, 0);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[3].node = 1;  // 3 ranks on one host, 1 on the other
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(topo->skipPresetTopoMatching);
}

TEST_F(InitMicrotest, InitTransportsRank_TopoComputeFails_PropagatesAndRunsCleanup) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclTopoComputeResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputeCalls);
  EXPECT_EQ(2, g_ncclOsCpuCountCalls);  // :1608 and exit::2403 -- unlike the :1618 bypass above
}

TEST_F(InitMicrotest, InitTransportsRank_RingGraphSeededBeforeTopoCompute) {
  TransportsRankComm c(4, 0);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  const ncclTopoGraph& ring = c.get()->graphs[NCCL_ALGO_RING];
  EXPECT_EQ(0, ring.id);
  EXPECT_EQ(NCCL_TOPO_PATTERN_RING, ring.pattern);
  EXPECT_EQ(1, ring.minChannels);
  EXPECT_EQ(MAXCHANNELS / 2, ring.maxChannels);
}
