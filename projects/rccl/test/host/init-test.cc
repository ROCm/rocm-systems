/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/init.cc: the UUT is #include'd, so static helpers are directly callable.

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "fakes/init_fakes.h"
#include "../common/LogCapture.hpp"                 // CaptureLog: assert on WARN/INFO text
#include "../common/ProcessIsolatedTestRunner.hpp"  // fork+execv process isolation

// alloc.h first, so its macros are visible to be #undef'd before init.cc's transitive includes see them.
#include "alloc.h"

#include "fakes/param_redirect.h"  // redirects NCCL_PARAM and both RCCL_PARAM spellings

// TRAP: this #define is textual and TU-wide, so the index counts every ncclCalloc the TEST reaches, not just the UUT's.
// TRAP: the reset lives in TearDown, not ResetInitFakes -- both statics are TU-local.
static int g_callocCallIndex = 0;
static int g_callocFailAt = -1;  // -1 = never fail; otherwise 0-based call index
template <typename... Args>
static ncclResult_t MicroCalloc(const char* file, int line, const char* fn, Args&&... args) {
  if (g_callocCallIndex++ == g_callocFailAt) return ncclSystemError;
  return ncclCallocDebug(std::forward<Args>(args)..., file, line, fn, true);
}
#undef ncclCalloc
#define ncclCalloc(...) MicroCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)

// Seam for the UUT's bare malloc(). Armed by exact byte count, not a call index, so a test targets one
// allocation without depending on how many mallocs the surrounding code happens to make.
// TRAP: textual and TU-wide -- pick a request size no unrelated allocation can plausibly match.
static std::size_t g_mallocFailSize = 0;  // 0 = never fail; reset in TearDown, both statics are TU-local
static void* MicroMalloc(std::size_t n) {
  return (g_mallocFailSize != 0 && n == g_mallocFailSize) ? nullptr : std::malloc(n);
}
#define malloc(n) MicroMalloc(n)
// commCleanup's tuner->finalize seam. TU-local (needs ncclTuner_t), so the reset lives in TearDown.
static int g_tunerFinalizeCalls = 0;
static void* g_tunerFinalizeLastContext = nullptr;
static ncclResult_t g_tunerFinalizeResult = ncclSuccess;

#include "fakes/nvtx_redirect.h"  // neuter / block nvtx.h before init.cc includes it

// Pulled in ahead of the redirects below so those macros cannot mangle the declarations they redirect.
#include <cpuid.h>

#include "kernel_config.h"
#include "os.h"

#include "ScopedHook.h"

// Each default forwards to what the UUT would otherwise call, so an uninstalled redirect is a no-op.
static std::function<void(void*)> g_microFree = [](void* p) { ::free(p); };
static void MicroFree(void* p) { g_microFree(p); }

static std::function<int(unsigned, unsigned*, unsigned*, unsigned*, unsigned*)> g_microCpuid =
    [](unsigned leaf, unsigned* a, unsigned* b, unsigned* c, unsigned* d) {
      return __get_cpuid(leaf, a, b, c, d);
    };
static int MicroCpuid(unsigned leaf, unsigned* a, unsigned* b, unsigned* c, unsigned* d) {
  return g_microCpuid(leaf, a, b, c, d);
}

static std::function<ncclResult_t(const char*, const char*, char*, int)> g_microTopoGetStrFromSys =
    [](const char* path, const char* file, char* out, int maxLen) {
      return ncclOsTopoGetStrFromSys(path, file, out, maxLen);
    };
static ncclResult_t MicroTopoGetStrFromSys(const char* path, const char* file, char* out, int maxLen) {
  return g_microTopoGetStrFromSys(path, file, out, maxLen);
}

static std::function<bool(const char*)> g_microIommuPassthroughOk = [](const char* c) {
  return ncclIommuPassthroughOk(c);
};
static bool MicroIommuPassthroughOk(const char* cmdline) { return g_microIommuPassthroughOk(cmdline); }

#define free(p) MicroFree(p)
#define __get_cpuid(l, a, b, c, d) MicroCpuid(l, a, b, c, d)
#define ncclOsTopoGetStrFromSys(p, f, o, n) MicroTopoGetStrFromSys(p, f, o, n)
#define ncclIommuPassthroughOk(c) MicroIommuPassthroughOk(c)

// INIT_CC_PATH is ${PROJECT_BINARY_DIR}/hipify/src/init.cc -- NOT init_tmp.cc, which shares the basename.
#include INIT_CC_PATH

#undef malloc  // scoped to the UUT: leaving it defined would silently reroute every malloc() in the tests below
#undef free
#undef __get_cpuid
#undef ncclOsTopoGetStrFromSys
#undef ncclIommuPassthroughOk

// These net fakes live here, not in init_fakes.cc: they set comm->ncclNet, which needs the layout only this TU sees.
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
// ASan aborts on allocations above its 1 TiB cap rather than returning NULL, and eats SIGFPE; both break death tests.
extern "C" const char* __asan_default_options() {
  return "allocator_may_return_null=1:handle_sigfpe=0";
}

// Must stay below #undef malloc/#undef free, or these bodies are rewritten into the UUT's seams.
static const void* g_deleteWatchPtr = nullptr;
static int g_deleteWatchHits = 0;

static void MicroOperatorDelete(void* p) noexcept {
  if (p != nullptr && p == g_deleteWatchPtr) ++g_deleteWatchHits;
  std::free(p);
}
// Replaced as a matched pair and in every form: `delete p` emits the SIZED overload, so void* alone counts nothing.
void* operator new(std::size_t n) {
  void* p = std::malloc(n != 0 ? n : 1);
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n != 0 ? n : 1); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return std::malloc(n != 0 ? n : 1); }
void operator delete(void* p) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p) noexcept { MicroOperatorDelete(p); }
void operator delete(void* p, std::size_t) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p, std::size_t) noexcept { MicroOperatorDelete(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { MicroOperatorDelete(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { MicroOperatorDelete(p); }

class DeleteWatch {
 public:
  explicit DeleteWatch(const void* p) {
    g_deleteWatchPtr = p;
    g_deleteWatchHits = 0;
  }
  ~DeleteWatch() { g_deleteWatchPtr = nullptr; }
  DeleteWatch(const DeleteWatch&) = delete;
  DeleteWatch& operator=(const DeleteWatch&) = delete;
  void Arm(const void* p) {
    g_deleteWatchPtr = p;
    g_deleteWatchHits = 0;
  }
  int hits() const { return g_deleteWatchHits; }
};

// ASan turns a SIGSEGV into a report plus exit(1), so KilledBySignal never matches under it; match the exit instead.
#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#define DEATH_BY_SEGV ::testing::ExitedWithCode(1)
#else
#define DEATH_BY_SEGV ::testing::KilledBySignal(SIGSEGV)
#endif

class InitMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    // ncclGetEnv reads are hermetic already (micro_getenv is strict). Only the bare getenv/std::getenv
    // sites reach the real environment, so mask those. Re-derive with: grep -n '\bgetenv(' src/init.cc
    ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
    for (const char* name : {"HSA_NO_SCRATCH_RECLAIM", "HSA_FORCE_FINE_GRAIN_PCIE", "ROCSHMEM_HEAP_SIZE"}) {
      SetMicroEnvAbsent(name);
    }
  }
  void TearDown() override {
    ResetInitFakes();
    // ctaPolicyEnv (init.cc:143) is only ever assigned or OR-accumulated, never cleared, so it must be reset here.
    ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
    g_callocCallIndex = 0;
    g_callocFailAt = -1;
    g_mallocFailSize = 0;
    g_deleteWatchPtr = nullptr;
    g_deleteWatchHits = 0;
    g_tunerFinalizeCalls = 0;
    g_tunerFinalizeLastContext = nullptr;
    g_tunerFinalizeResult = ncclSuccess;
  }
};

// Derives so SetUp's env mask reaches the re-exec'd child; distinct name keeps the yaml filter InitMicrotestIsolated.*
class InitMicrotestIsolated : public InitMicrotest {};

namespace {
// uniformRanksPerHost reads ONLY peerInfo[i].hostHash; each initializer value is a host id.
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

TEST_F(InitMicrotest, UniformRanksPerHost_TwoHostsTwoRanksEach_ReturnsTrue) {
  HostPattern p{1, 1, 2, 2};
  EXPECT_TRUE(uniformRanksPerHost(p.comm(), p.nranks()));
}

TEST_F(InitMicrotest, UniformRanksPerHost_UnevenRanksPerHost_ReturnsFalse) {
  HostPattern p{1, 1, 2};
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
  HostPattern p{};
  EXPECT_FALSE(uniformRanksPerHost(p.comm(), 0));
}

namespace {
using RcclUnitTesting::LogHas;
using RcclUnitTesting::ScopedDebugLogging;

// The single log LINE containing `needle`, or "" if none does. Two separate LogHas calls over one
// buffer cannot tell which line satisfied which, and init.cc has format strings that share a tail --
// :1547's TRACE and :1550's WARN both end in "intraProcRank %d intraProcRanks %d intraProcRank0 %d".
std::string LogLineWith(const std::string& log, const char* needle) {
  const size_t hit = log.find(needle);
  if (hit == std::string::npos) return "";
  const size_t begin = log.rfind('\n', hit);
  const size_t end = log.find('\n', hit);
  const size_t from = (begin == std::string::npos) ? 0 : begin + 1;
  return log.substr(from, (end == std::string::npos ? log.size() : end) - from);
}

std::string RunShowVersion(int debugLevel) {
  ScopedDebugLogging dbg(debugLevel, NCCL_ALL);
  return RcclUnitTesting::CaptureLog([] { showVersion(); });
}
}  // namespace

// LATENT BUG (init.cc:1011-1012): hostBuf is uninitialised and gethostname need not NUL-terminate on truncation.
TEST_F(InitMicrotest, ShowVersion_HappyPath_LogsVersionHostAndLibPath) {
  char host[HOST_NAME_MAX] = {};
  ASSERT_EQ(0, gethostname(host, sizeof(host) - 1));
  Dl_info self{};
  ASSERT_NE(0, dladdr((void*)ncclCommInitRank, &self));

  const std::string log = RunShowVersion(NCCL_LOG_INFO);

  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "microtest")) << "git hash missing:\n" << log;
  EXPECT_TRUE(LogHas(log, (std::string("Hostname     : ") + host).c_str()))
      << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, (std::string("Librccl path : ") + self.dli_fname).c_str()))
      << "actual log:\n" << log;
  // INFO logs __func__ (debug.h:50-54) and VERSION logs __FILE__ (debug.h:39): the prefix tells the arms apart.
  EXPECT_TRUE(LogHas(log, "showVersion:")) << "INFO arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "init.cc:")) << "took the VERSION arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelVersion_UsesVersionLog) {
  const std::string log = RunShowVersion(NCCL_LOG_VERSION);
  // showVersion passes sizeof(hostBuf)-1, reserving the byte gethostname need not NUL-terminate.
  EXPECT_EQ(static_cast<size_t>(HOST_NAME_MAX - 1), LastGethostnameLen());
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "init.cc:")) << "VERSION arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "showVersion:")) << "took the INFO arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DebugLevelWarn_UsesVersionLog) {
  const std::string log = RunShowVersion(NCCL_LOG_WARN);
  EXPECT_TRUE(LogHas(log, "RCCL version")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "init.cc:")) << "VERSION arm not taken:\n" << log;
  EXPECT_FALSE(LogHas(log, "showVersion:")) << "took the INFO arm:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_HipRuntimeVersionUnavailable_OmitsRuntimeHip) {
  g_hipRuntimeGetVersion = [](int*) { return hipErrorInvalidValue; };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "HIP version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "HIP runtime")) << "fabricated a runtime version:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_HipRuntimeDiffersFromCompileTime_ReportsRuntime) {
  // 90807006 decodes to 9.8.7006, deliberately not the compile-time version.
  g_hipRuntimeGetVersion = [](int* v) {
    if (v) *v = 90807006;
    return hipSuccess;
  };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "9.8.7006")) << "runtime HIP version not reported:\n" << log;
}

#if ROCM_VERSION >= 60000
TEST_F(InitMicrotest, ShowVersion_RocmVersionAvailable_ReportsRuntimeRocm) {
  g_getROCmVersionResult = 0;  // VerSuccess
  g_rocmVersionMajor = 9;
  g_rocmVersionMinor = 8;
  g_rocmVersionPatch = 7;
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm runtime : 9.8.7")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_RocmVersionUnavailable_OmitsRuntimeRocm) {
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "ROCm runtime")) << "fabricated a runtime version:\n" << log;
}
#endif

TEST_F(InitMicrotest, ShowVersion_HipRuntimeMatchesCompileTime_OmitsRuntime) {
  g_hipRuntimeGetVersion = [](int* v) {
    if (v) *v = HIP_VERSION_MAJOR * 10000000 + HIP_VERSION_MINOR * 100000 + HIP_VERSION_PATCH;
    return hipSuccess;
  };
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "HIP version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "HIP runtime")) << "actual log:\n" << log;
}

#if ROCM_VERSION >= 60000
TEST_F(InitMicrotest, ShowVersion_RocmRuntimeMatchesCompileTime_OmitsRuntime) {
  g_getROCmVersionResult = 0;  // VerSuccess
  g_rocmVersionMajor = ROCM_VERSION_MAJOR;
  g_rocmVersionMinor = ROCM_VERSION_MINOR;
  g_rocmVersionPatch = ROCM_VERSION_PATCH;
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "ROCm version")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "ROCm runtime")) << "actual log:\n" << log;
}
#endif

TEST_F(InitMicrotest, ShowVersion_GethostnameFails_ReportsUnknownHost) {
  SetGethostnameFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  // The trailing \n matters: LogHas is a substring check, so without it "UnknownHostX" would also match.
  EXPECT_TRUE(LogHas(log, "Hostname     : Unknown\n")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, ShowVersion_DladdrFails_ReportsUnknownLibPath) {
  SetDladdrFail(true);
  const std::string log = RunShowVersion(NCCL_LOG_INFO);
  EXPECT_TRUE(LogHas(log, "Librccl path : Unknown\n")) << "actual log:\n" << log;
}

// Unlike the real initChannel (channel.cc:61-62), the fake does NOT allocate the ring arrays: the builder owns them.
namespace {
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
  const int nRanks = 4, localRank = 3, channelId = 0;
  int ringRanks[nRanks] = {2, 0, 3, 1};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(1, c.ring().index);
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
  EXPECT_EQ(localRank, c.userRanks()[0]) << "userRanks must start at the local rank";
  for (int i = 0; i < nRanks; ++i) EXPECT_EQ(i, c.rankToIndex()[c.userRanks()[i]]) << "i=" << i;
}

TEST_F(InitMicrotest, SetupChannel_IdentityRing_Rank0_IsIdentity) {
  const int nRanks = 4, localRank = 0, channelId = 0;
  int ringRanks[nRanks] = {0, 1, 2, 3};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.userRanks());
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.rankToIndex());
}

TEST_F(InitMicrotest, SetupChannel_RotatedRing_Rank0_IndexIsZero) {
  const int nRanks = 4, localRank = 0, channelId = 0;
  int ringRanks[nRanks] = {2, 3, 0, 1};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), c.userRanks());
}

TEST_F(InitMicrotest, SetupChannel_SingleRank_TrivialRing) {
  const int nRanks = 1, localRank = 0, channelId = 0;
  int ringRanks[nRanks] = {0};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(0, c.ring().index);
  EXPECT_EQ(std::vector<int>({0}), c.userRanks());
  EXPECT_EQ(std::vector<int>({0}), c.rankToIndex());
}

TEST_F(InitMicrotest, SetupChannel_InitChannelFails_PropagatesAndLeavesRingUntouched) {
  const int nRanks = 4, localRank = 3, channelId = 0;
  g_initChannelResult = ncclInternalError;
  int ringRanks[nRanks] = {2, 0, 3, 1};
  SetupChannelComm c(nRanks, channelId);
  EXPECT_EQ(ncclInternalError, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));
  EXPECT_EQ(std::vector<int>({-1, -1, -1, -1}), c.userRanks());
}

// NCCLCHECK returns only when the result is neither ncclSuccess NOR ncclInProgress, so this arm still builds the ring.
TEST_F(InitMicrotest, SetupChannel_InitChannelInProgress_ContinuesAndSucceeds) {
  const int nRanks = 4, localRank = 3, channelId = 0;
  g_initChannelResult = ncclInProgress;
  int ringRanks[nRanks] = {2, 0, 3, 1};
  SetupChannelComm c(nRanks, channelId);
  EXPECT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
}

// LATENT BUG (init.cc:1189-1193): setupChannel checks neither that `rank` nor that rank 0 is a member of the ring.
TEST_F(InitMicrotest, SetupChannel_RankNotInRing_SilentlyTreatsIndexZeroAsSelf) {
  const int nRanks = 4, localRank = 7, channelId = 0;  // localRank 7 is not in the ring
  int ringRanks[nRanks] = {1, 2, 3, 0};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(1, c.ring().index);
  EXPECT_EQ(std::vector<int>({1, 2, 3, 0}), c.userRanks());
  EXPECT_EQ(std::vector<int>({3, 0, 1, 2}), c.rankToIndex());
}

// Memory safety needs every entry < nranks but NOT distinctness, so a ring with a duplicate and no rank 0 is legal.
TEST_F(InitMicrotest, SetupChannel_RankZeroNotInRing_MisrotatesAndLeavesHole) {
  const int nRanks = 4, localRank = 2, channelId = 0;
  int ringRanks[nRanks] = {1, 1, 2, 3};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(2, c.ring().index) << "measured from a rank 0 that is not in the ring";
  EXPECT_EQ(std::vector<int>({2, 3, 1, 1}), c.userRanks());
  // rankToIndex[0] is the builder's -1 fill: nothing ever wrote it.
  EXPECT_EQ(std::vector<int>({-1, 3, 0, 1}), c.rankToIndex());
}

// A 4-cycle is the smallest shape that tells `rankToIndex[userRanks[i]]=i` apart from `rankToIndex[i]=userRanks[i]`.
TEST_F(InitMicrotest, SetupChannel_NonInvolutiveRotation_PinsRankToIndex) {
  const int nRanks = 4, localRank = 1, channelId = 0;
  int ringRanks[nRanks] = {0, 1, 2, 3};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(std::vector<int>({1, 2, 3, 0}), c.userRanks());
  EXPECT_EQ(std::vector<int>({3, 0, 1, 2}), c.rankToIndex()) << "not the forward map";
}

TEST_F(InitMicrotest, SetupChannel_NonZeroChannelId_UsesThatChannel) {
  const int nRanks = 4, localRank = 3, channelId = 2;
  int ringRanks[nRanks] = {2, 0, 3, 1};
  SetupChannelComm c(nRanks, channelId);
  ASSERT_EQ(ncclSuccess, setupChannel(c.get(), channelId, localRank, nRanks, ringRanks));

  EXPECT_EQ(1, c.ring().index);
  EXPECT_EQ(std::vector<int>({3, 1, 2, 0}), c.userRanks());
  EXPECT_EQ(channelId, g_initChannelLastId) << "the channelId must be forwarded, not hardcoded";
}

namespace {
std::unique_ptr<ncclComm> MakeParentComm(int nRanks, int rank) {
  auto parent = std::unique_ptr<ncclComm>(new ncclComm{});
  parent->nRanks = nRanks;
  parent->rank = rank;
  parent->bootstrap = nullptr;  // the allgather seam ignores it
  return parent;
}

// A real allgather GATHERS our slot rather than writing it, so entry [selfRank] is ASSERTED here, never overwritten.
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

// Not a reachable defect: the sole caller rejects self-exclusion first (bsearch at init.cc:4039).
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

// LATENT BUG (init.cc:2540): *nRanksRet is parentRanks - excludeRanksCount and nothing dedupes, so myRank can == nRanks
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

TEST_F(InitMicrotest, CommGetSplitInfo_NoColor_ReturnsEarlyWithoutTouchingOutputs) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
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

// EQUIVALENT MUTANT: `r > insert` -> `r >= insert` at init.cc:2508 -- the extra write is overwritten by :2510.
TEST_F(InitMicrotest, CommGetSplitInfo_UnsortedKeys_InsertionSortsByKey) {
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 3}, {7, 1}, {7, 2}, {7, 0}});
  int parentRanks[4] = {-1, -1, -1, -1};
  int nRanks = -1, myRank = -1;
  ASSERT_EQ(ncclSuccess, commGetSplitInfo(nullptr, parent.get(), /*color=*/7, /*key=*/1,
                                          &nRanks, &myRank, parentRanks));
  EXPECT_EQ(4, nRanks);
  // Ordered by key: rank3(key0), rank1(key1), rank2(key2), rank0(key3).
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

// The comparator is `<=`, so an equal key inserts AFTER the incumbent; `<` would reverse the tie order.
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

namespace {
// Global ranks are handed out node by node, matching how the real topology numbers them.
class P2pScheduleComm {
 public:
  P2pScheduleComm(int nNodes, int node, int localRank, int nRanks, int maxLocalRanks,
                  std::initializer_list<int> localRanksPerNode)
      : nodeRanks_(localRanksPerNode.size()), comm_(new ncclComm{}) {
    // A mismatch would read past the end of nodeRanks_ -- harness UB that would be misattributed to the unit.
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

// DEAD BRANCH (init.cc:1331): the gcd loop at :1314-1317 forces groupSize to divide every localRanks. Do not chase it.

// A negative NCCL_P2P_SCHEDULE_GROUP_SIZE yields a negative nGroups, so ncclCalloc is asked for a bogus size and fails.
TEST_F(InitMicrotest, P2pSchedule_NegativeGroupSizeParam_FailsInAllocation) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(-2) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4, 4});
  EXPECT_EQ(ncclSystemError, ncclP2pSchedule(c.get()));
}

// ERANGE is checked (param.cc:93-97) but the int64->int narrowing at :1313 is not, so GROUP_SIZE=0 or 2^32 SIGFPEs.
TEST_F(InitMicrotest, P2pSchedule_ZeroGroupSizeParam_DiesOnDivideByZero) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(0) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/8,
                    /*maxLocalRanks=*/4, {4, 4});
  // Pin the signal: EXPECT_DEATH("") would also accept ::abort(), _exit(1) or a null deref at the same spot.
  EXPECT_EXIT(ncclP2pSchedule(c.get()), ::testing::KilledBySignal(SIGFPE), "");
}

TEST_F(InitMicrotest, P2pSchedule_SingleNode_BuildsFullSchedule) {
  // nNodes == 1 -> groupSize comes from maxLocalRanks, not the param.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4});
  ASSERT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
  const int expectSend[4] = {0, 1, 2, 3};
  const int expectRecv[4] = {0, 3, 2, 1};
  for (int r = 0; r < 4; ++r) {
    EXPECT_EQ(expectSend[r], c.sendRank(r)) << "round " << r;
    EXPECT_EQ(expectRecv[r], c.recvRank(r)) << "round " << r;
  }
}

TEST_F(InitMicrotest, P2pSchedule_MultiNode_Rank1OnNode1_FullScheduleContents) {
  // localRank=1 matters: at localRank=0 both `local` and `group` are identically 0, hiding the whole group walk.
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
  // The chosen groupSize is unobservable from outside, so pin it via the INFO at init.cc:1348.
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/6,
                    /*maxLocalRanks=*/4, {6});
  std::string log;
  ncclResult_t res = ncclInternalError;
  {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    log = RcclUnitTesting::CaptureLog([&] { res = ncclP2pSchedule(c.get()); });
  }
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(LogHas(log, "group size used is 2")) << "actual log:\n" << log;
}

// EQUIVALENT MUTANT: dropping `|| localRanks < groupSize` at :1316 -- the first disjunct fires unless localRanks==0.
TEST_F(InitMicrotest, P2pSchedule_EmptyNode_SkipsGroupLoop) {
  // localRanks=0 on node 1 -> nGroupsInNode = 0, so the inner loop at init.cc:1337 is entered zero times.
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_SCHEDULE_GROUP_SIZE") == 0 ? int64_t(4) : deft;
  };
  P2pScheduleComm c(/*nNodes=*/2, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4, 0});
  EXPECT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
}

TEST_F(InitMicrotest, P2pSchedule_NonPow2Groups_ScheduleContents) {
  // nGroups=3 with nGroupsPow2=4 takes the false arm of `if (groupDelta < nGroups)` and exposes the :1355 wrap bias.
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

// LATENT BUG (init.cc:1346): this return fires before the free() at :1370-1371, leaking groupToNode and groupToLocal.
// LATENT BUG (init.cc:1334): the same leak on the sibling return, but behind the dead branch above.
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
  // nRanks=6 with groupSize=4 -> nGroups=1, so only 4 rounds are emitted and the final round==nRanks check fails.
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

// LATENT BUG (init.cc:1328): the second ncclCalloc's NCCLCHECK returns with groupToNode allocated and never freed.
TEST_F(InitMicrotest, P2pSchedule_SecondCallocFails_ReturnsSystemError) {
  g_callocFailAt = 1;  // groupToLocal
  P2pScheduleComm c(/*nNodes=*/1, /*node=*/0, /*localRank=*/0, /*nRanks=*/4,
                    /*maxLocalRanks=*/4, {4});
  EXPECT_EQ(ncclSystemError, ncclP2pSchedule(c.get()));
}

TEST_F(InitMicrotest, P2pSchedule_ZeroNodes_AllocatesNothingAndSucceeds) {
  // nGroups=0 drives ncclCalloc's nelem==0 arm; pow2Up(0)==1, so the delta walk ends after one skipped iteration.
  P2pScheduleComm c(/*nNodes=*/0, /*node=*/0, /*localRank=*/0, /*nRanks=*/0,
                    /*maxLocalRanks=*/1, {});
  EXPECT_EQ(ncclSuccess, ncclP2pSchedule(c.get()));
}

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

namespace {
// nullptr scripts NCCL_CTA_POLICY as absent. The reset matters: the unit only ever assigns or OR-accumulates.
int RunCtaPolicyEnv(const char* value) {
  ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
  if (value) SetMicroEnv("NCCL_CTA_POLICY", value);
  else SetMicroEnvAbsent("NCCL_CTA_POLICY");
  getEnvCtaPolicyOnce();
  return ctaPolicyEnv;
}

// Several inputs are state-indistinguishable ("7" and unset both leave UNDEF), so the diagnostic is the only oracle.
std::string RunCtaPolicyEnvCapturingLog(const char* value, int* policyOut) {
  ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
  return RcclUnitTesting::CaptureLog([&] { *policyOut = RunCtaPolicyEnv(value); });
}
}  // namespace

TEST_F(InitMicrotest, GetEnvCtaPolicy_Unset_LeavesPolicyUndefined) {
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, RunCtaPolicyEnv(nullptr));
}

TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitZero_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("0"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitOne_SelectsEfficiency) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, RunCtaPolicyEnv("1"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_DigitTwo_SelectsZero) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, RunCtaPolicyEnv("2"));
}

// LATENT BUG (init.cc:160): the `default:` arm logs "Using DEFAULT instead" but never assigns it, leaving UNDEF.
TEST_F(InitMicrotest, GetEnvCtaPolicy_UnknownDigit_LogsDefaultButLeavesUnset) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("7", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);
  // "Unknown CTA policy" is shared with the per-token message at :174; this phrase occurs exactly once in init.cc.
  EXPECT_TRUE(LogHas(log, "Using DEFAULT instead")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedDefault_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("DEFAULT"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedEfficiency_SelectsEfficiencyAndLogsParse) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("EFFICIENCY", &policy);
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, policy);
  // Assert the interpolated payload, not just the sentence: a wrong policy value still satisfies the bare needle.
  EXPECT_TRUE(LogHas(log, "NCCL_CTA_POLICY=EFFICIENCY to 1")) << "actual log:\n" << log;
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_NamedZero_SelectsZero) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, RunCtaPolicyEnv("ZERO"));
}

TEST_F(InitMicrotest, GetEnvCtaPolicy_TwoNamedModes_AccumulatesBoth) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO,
            RunCtaPolicyEnv("EFFICIENCY|ZERO"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_MixedCaseModes_AccumulatesBoth) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO,
            RunCtaPolicyEnv("efficiency|ZeRo"));
}
// DEFAULT is 0x00 and so bit-invisible beside other tokens: alone is the only input that observes the strcasecmp arm.
TEST_F(InitMicrotest, GetEnvCtaPolicy_LowerCaseDefaultAlone_SelectsDefault) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("default"));
}
// The third token must NOT be DEFAULT: it is 0x00, so ZERO|DEFAULT is bit-identical to ZERO and dropping it is unseen.
TEST_F(InitMicrotest, GetEnvCtaPolicy_UnknownTokenAmongValid_IgnoresOnlyTheUnknown) {
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO | NCCL_CTA_POLICY_EFFICIENCY,
            RunCtaPolicyEnv("ZERO|BOGUS|EFFICIENCY"));
}
TEST_F(InitMicrotest, GetEnvCtaPolicy_OnlyUnknownToken_LeavesUnsetAndLogsTwice) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("BOGUS", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);
  EXPECT_TRUE(LogHas(log, "Unknown CTA policy BOGUS")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "No valid CTA policies found")) << "actual log:\n" << log;
}
// isdigit('\0') is false, so an empty value takes the combine arm but strtok yields no token and the loop never runs.
TEST_F(InitMicrotest, GetEnvCtaPolicy_EmptyString_ParsesNoTokens) {
  int policy = NCCL_CONFIG_UNDEF_INT;
  const std::string log = RunCtaPolicyEnvCapturingLog("", &policy);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, policy);
  EXPECT_TRUE(LogHas(log, "No valid CTA policies found")) << "actual log:\n" << log;
}

// LATENT BUG (init.cc:149): isdigit(env[0]) short-circuits the combine syntax, so "0|EFFICIENCY" drops EFFICIENCY.
TEST_F(InitMicrotest, GetEnvCtaPolicy_LeadingDigit_DropsCombinedModes) {
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, RunCtaPolicyEnv("0|EFFICIENCY"));
}

// LATENT BUG (init.cc:168): tokens are never trimmed, so "DEFAULT | ZERO" yields two unknown tokens and applies none.
TEST_F(InitMicrotest, GetEnvCtaPolicy_SpacesAroundPipe_NoTokensRecognized) {
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, RunCtaPolicyEnv("DEFAULT | ZERO"));
}

// The unit never clears ctaPolicyEnv, so a second call ORs into the first; production guards it with call_once.
TEST_F(InitMicrotest, GetEnvCtaPolicy_CalledTwice_AccumulatesAcrossCalls) {
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, RunCtaPolicyEnv("EFFICIENCY"));
  SetMicroEnv("NCCL_CTA_POLICY", "ZERO");
  getEnvCtaPolicyOnce();  // deliberately NOT reset in between
  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO, ctaPolicyEnv);
}

// A fresh NCCL_CONFIG_INITIALIZER passes every arm; each test perturbs exactly one field.
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

// CommCheck/PtrCheck are the REAL argcheck.cc oracles, so a "ready" comm needs valid magics and abortFlag -> 0.
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
TEST_F(InitMicrotest, CommGetUniqueId_NullComm_ReturnsInvalidArgument) {
  ncclUniqueId id{};
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetUniqueId_impl(nullptr, &id));
}

TEST_F(InitMicrotest, CommGetUniqueId_NullUniqueId_ReturnsInvalidArgument) {
  ReadyComm rc;
  EXPECT_EQ(ncclInvalidArgument, ncclCommGetUniqueId_impl(rc.get(), nullptr));
}

TEST_F(InitMicrotest, CommGetUniqueId_NotReadyComm_ReturnsInvalidArgument) {
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

  ncclBootstrapHandle out{};
  std::memcpy(&out, &id, sizeof(out));
  EXPECT_EQ(0xFEEDFACEULL, out.magic);
  // init.cc:4171 zeroes the WHOLE id first, so the bytes past the handle are 0, not the 0xAB poison.
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

TEST_F(InitMicrotest, CommShrink_NotReadyComm_ReturnsInvalidArgument) {
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

// LATENT BUG (init.cc:4113): PtrCheck rejects a null newcomm by jumping to exit:, which derefs *newcomm and SEGFAULTs.
TEST_F(InitMicrotest, CommShrink_NullNewcomm_DiesOnNullDeref) {
  ReadyComm rc;
  int exclude[1] = {0};
  // Match the message too: the signal alone would also accept a crash arriving BEFORE the newcomm validation.
  EXPECT_EXIT(ncclCommShrink_impl(rc.get(), exclude, /*excludeRanksCount=*/1, nullptr,
                                  /*config=*/nullptr, /*shrinkFlags=*/0),
              DEATH_BY_SEGV, "newcomm argument is NULL");
}

// Full strings, not substrings: a prefix check cannot see a changed NCCL_DEBUG level or a dropped "(run with)" hint.
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
  static_assert(sizeof(kCases) / sizeof(kCases[0]) == ncclNumResults,
                "a new ncclResult_t needs a case in init.cc and a row here");
  std::vector<std::string> seen;
  for (const auto& c : kCases) {
    const char* msg = ncclGetErrorString_impl(c.code);
    ASSERT_NE(nullptr, msg) << "code " << c.code;
    EXPECT_STREQ(c.expect, msg) << "code " << c.code;
    seen.emplace_back(msg);
  }
  // Distinctness too: a copy-paste slip returning one message from two cases satisfies every per-code check above.
  std::sort(seen.begin(), seen.end());
  EXPECT_EQ(seen.end(), std::unique(seen.begin(), seen.end()))
      << "two result codes share a message";
}

TEST_F(InitMicrotest, GetErrorString_UnmappedCode_FallsBackToUnknown) {
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

namespace {
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
// TODO(AICOMRCCL-1685): a MIN_CTAS override does not apply, unlike COMM_BLOCKING on the same g_loadParam path.

namespace {
constexpr char kParentNetName[] = "parent-net";
constexpr char kChildNetName[] = "child-net";
// Deliberately odd length: the malloc seam is armed by request size, so this must not collide with another alloc.
constexpr char kUnallocatableNetName[] = "net-whose-strlen-plus-one-is-the-armed-malloc-failure-size-xyz";

// Every field differs from FillChildConfig's AND is post-validation stable, so envConfigOverride's clamps
// rewrite none of them; anything that differs after the copy is a defect, not a validation rewrite.
void FillParentConfig(ncclConfig_t& c) {
  c.size = 0x5A5A;
  c.magic = 0x00C0FFEEu;
  c.version = 0x00BEEF01u;
  c.blocking = 0;
  c.cgaClusterSize = 3;
  c.minCTAs = 5;
  c.maxCTAs = 11;
  c.netName = kParentNetName;
  c.splitShare = 1;
  c.trafficClass = 17;
  c.commName = "parent-comm";
  c.collnetEnable = 1;
  c.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  c.shrinkShare = 1;
  c.nvlsCTAs = 7;
  c.nChannelsPerNetPeer = 9;
  c.nvlinkCentricSched = 1;
  c.graphUsageMode = 2;
  c.numRmaCtx = 4;
  c.maxP2pPeers = 13;
  c.graphStreamOrdering = 1;
}

void FillChildConfig(ncclConfig_t& c) {
  c.size = 0xA5A5;
  c.magic = 0x00DEAD02u;
  c.version = 0x00FEED03u;
  c.blocking = 1;
  c.cgaClusterSize = 2;
  c.minCTAs = 6;
  c.maxCTAs = 12;
  c.netName = kChildNetName;
  c.splitShare = 0;
  c.trafficClass = 18;
  c.commName = "child-comm";
  c.collnetEnable = 0;
  c.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
  c.shrinkShare = 0;
  c.nvlsCTAs = 8;
  c.nChannelsPerNetPeer = 10;
  c.nvlinkCentricSched = 0;
  c.graphUsageMode = 1;
  c.numRmaCtx = 5;
  c.maxP2pPeers = 14;
  c.graphStreamOrdering = 0;
}

// TRIPWIRE: a new ncclConfig_t field must be added to both fills and to ExpectConfigFieldsEqual, or a
// memcpy truncated just before it would go unnoticed. Update all four sites together.
static_assert(sizeof(ncclConfig_t) == 96, "ncclConfig_t layout changed -- extend the copyCommConfig field checks");

// Field-by-field so a failure names the field. netName by content: envConfigOverride re-mallocs it.
void ExpectConfigFieldsEqual(const ncclConfig_t& want, const ncclConfig_t& got) {
  EXPECT_EQ(want.size, got.size);
  EXPECT_EQ(want.magic, got.magic);
  EXPECT_EQ(want.version, got.version);
  EXPECT_EQ(want.blocking, got.blocking);
  EXPECT_EQ(want.cgaClusterSize, got.cgaClusterSize);
  EXPECT_EQ(want.minCTAs, got.minCTAs);
  EXPECT_EQ(want.maxCTAs, got.maxCTAs);
  ASSERT_NE(nullptr, got.netName);
  EXPECT_STREQ(want.netName, got.netName);
  EXPECT_EQ(want.splitShare, got.splitShare);
  EXPECT_EQ(want.trafficClass, got.trafficClass);
  EXPECT_STREQ(want.commName, got.commName);
  EXPECT_EQ(want.collnetEnable, got.collnetEnable);
  EXPECT_EQ(want.CTAPolicy, got.CTAPolicy);
  EXPECT_EQ(want.shrinkShare, got.shrinkShare);
  EXPECT_EQ(want.nvlsCTAs, got.nvlsCTAs);
  EXPECT_EQ(want.nChannelsPerNetPeer, got.nChannelsPerNetPeer);
  EXPECT_EQ(want.nvlinkCentricSched, got.nvlinkCentricSched);
  EXPECT_EQ(want.graphUsageMode, got.graphUsageMode);
  EXPECT_EQ(want.numRmaCtx, got.numRmaCtx);
  EXPECT_EQ(want.maxP2pPeers, got.maxP2pPeers);
  EXPECT_EQ(want.graphStreamOrdering, got.graphStreamOrdering);
}

// envConfigOverride always replaces config.netName with a fresh malloc; free it so the copy tests do not leak.
// TRAP: skip the literals -- a defect that stops the override leaves one in place, and free()ing it would
// abort the process instead of letting the assertions report what actually went wrong.
struct ScopedNetName {
  explicit ScopedNetName(ncclComm* c) : comm(c) {}
  ~ScopedNetName() {
    const char* p = comm->config.netName;
    if (p != kParentNetName && p != kChildNetName && p != kUnallocatableNetName) free(const_cast<char*>(p));
  }
  ncclComm* comm;
};
}  // namespace

TEST_F(InitMicrotest, CopyCommConfig_OverwritesEveryChildConfigField) {
  auto parent = std::make_unique<ncclComm>();
  auto child = std::make_unique<ncclComm>();
  FillParentConfig(parent->config);
  FillChildConfig(child->config);  // every field differs, so a truncated memcpy leaves a stale value behind
  const ncclConfig_t want = parent->config;
  ScopedNetName freeNetName(child.get());

  ASSERT_EQ(ncclSuccess, copyCommConfig(child.get(), parent.get()));

  ExpectConfigFieldsEqual(want, child->config);
  EXPECT_STRNE(kChildNetName, child->config.netName);
  EXPECT_NE(parent->config.netName, child->config.netName);  // the override re-mallocs it; aliasing means it skipped
}

TEST_F(InitMicrotest, CopyCommConfig_LeavesParentConfigUntouched) {
  auto parent = std::make_unique<ncclComm>();
  auto child = std::make_unique<ncclComm>();
  FillParentConfig(parent->config);
  FillChildConfig(child->config);
  const ncclConfig_t want = parent->config;
  ScopedNetName freeNetName(child.get());

  ASSERT_EQ(ncclSuccess, copyCommConfig(child.get(), parent.get()));

  ExpectConfigFieldsEqual(want, parent->config);       // src/dst swapped would push the child's values here
  EXPECT_EQ(kParentNetName, parent->config.netName);   // and would retarget the parent at the child's literal
}

TEST_F(InitMicrotest, CopyCommConfig_EnvOverrideLandsOnTopOfTheCopy) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "COMM_BLOCKING") == 0 ? int64_t(1) : deft;
  };
  auto parent = std::make_unique<ncclComm>();
  auto child = std::make_unique<ncclComm>();
  FillParentConfig(parent->config);
  FillChildConfig(child->config);
  child->config.blocking = 0;  // both sides 0, so blocking==1 can only come from the override running after the copy
  ScopedNetName freeNetName(child.get());

  ASSERT_EQ(ncclSuccess, copyCommConfig(child.get(), parent.get()));

  EXPECT_EQ(1, child->config.blocking);
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, child->config.CTAPolicy);  // anchor: the copy still landed
  // The override re-mallocs netName, so aliasing the parent's buffer would mean the copy ran last.
  EXPECT_NE(parent->config.netName, child->config.netName);
  EXPECT_STREQ(kParentNetName, child->config.netName);
}

TEST_F(InitMicrotest, CopyCommConfig_CtaPolicyEnvOverridesCopiedPolicy) {
  SetMicroEnv("NCCL_CTA_POLICY", "EFFICIENCY");
  // envConfigOverride's call_once may already be burnt; parse here so ctaPolicyEnv is set whatever the shuffle order.
  getEnvCtaPolicyOnce();
  auto parent = std::make_unique<ncclComm>();
  auto child = std::make_unique<ncclComm>();
  FillParentConfig(parent->config);
  FillChildConfig(child->config);
  child->config.CTAPolicy = NCCL_CTA_POLICY_ZERO;  // both sides ZERO, so EFFICIENCY can only come from the env
  ScopedNetName freeNetName(child.get());

  ASSERT_EQ(ncclSuccess, copyCommConfig(child.get(), parent.get()));

  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, child->config.CTAPolicy);
  EXPECT_EQ(13, child->config.maxP2pPeers);  // anchor: the copy still landed
}

TEST_F(InitMicrotest, CopyCommConfig_EnvConfigOverrideFails_PropagatesErrorAfterCopying) {
  auto parent = std::make_unique<ncclComm>();
  auto child = std::make_unique<ncclComm>();
  FillParentConfig(parent->config);
  FillChildConfig(child->config);
  parent->config.netName = kUnallocatableNetName;
  g_mallocFailSize = std::strlen(kUnallocatableNetName) + 1;  // the only allocation envConfigOverride makes

  EXPECT_EQ(ncclSystemError, copyCommConfig(child.get(), parent.get()));

  EXPECT_EQ(nullptr, child->config.netName);                 // the failing allocation, not a stale child value
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, child->config.CTAPolicy);  // the copy still ran before the failure
  EXPECT_EQ(13, child->config.maxP2pPeers);
}

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

TEST_F(InitMicrotest, CheckHsaEnvSetting_AllSucceed_ReturnsSuccess) {
  ncclResult_t res = ncclUnhandledCudaError;
  const std::string log = RcclUnitTesting::CaptureLog([&] { res = checkHsaEnvSetting(); });
  EXPECT_EQ(ncclSuccess, res);  // defaults: gfx942, valid
  EXPECT_FALSE(LogHas(log, "HSA_NO_SCRATCH_RECLAIM=1 must be set")) << "actual log:\n" << log;
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
  SetMicroEnv("HSA_NO_SCRATCH_RECLAIM", "0");
  g_validHsaScratch = false;
  ncclResult_t res = ncclUnhandledCudaError;
  const std::string log = RcclUnitTesting::CaptureLog([&] { res = checkHsaEnvSetting(); });
  EXPECT_EQ(ncclSuccess, res);  // the WARN arm still succeeds, so the log is the only observable
  EXPECT_TRUE(LogHas(log, "HSA_NO_SCRATCH_RECLAIM=1 must be set")) << "actual log:\n" << log;
  ASSERT_NE(nullptr, g_lastHsaScratchEnv) << "checkHsaEnvSetting must read the environment";
  EXPECT_STREQ("0", g_lastHsaScratchEnv);
}

// The real IsArchMatch oracle reads comm->topo->nodes[GPU].nodes[0].gpu.gcn.
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
  g_hipFree = [](void*) { return hipErrorInvalidValue; };
  EXPECT_EQ(ncclUnhandledCudaError, fillInfo(comm.get(), &info, 0));
}

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
  EXPECT_TRUE(info.hasFineGrain);
  EXPECT_EQ(1, g_gdrSupportCalls);       // dmaBuf unsupported -> fallback taken
  EXPECT_EQ(1, info.gdrSupport);
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
  EXPECT_EQ(1, info.gdrSupport);
  EXPECT_EQ(0, g_gdrSupportCalls);   // GDR fallback NOT called
}

// MLOPart detection in fillInfo(). It exists for DPX/XCP/CPX, where HIP exposes each logical GPU as
// PCI function .N of one physical device and typically only .0 exists in sysfs as a GPU.
//
// Two signals, in order. The compute partition mode read from the physical device answers "is this
// hardware partitioned", and when it is, every function is a partition -- including function 0,
// which carries index 0 rather than staying undefined, so all of a device's partitions land in one
// DEV overlay group. Only when the platform reports no mode does the older sysfs-class probe decide,
// and that one can only recognise an alias at fn>0, never partition 0.
//
// The distinction matters beyond bookkeeping. An unpartitioned .0 GPU that gets stamped takes the
// MLOPart path in ncclTopoCheckGdr(), which reads its distance from the parent DEV node and applies
// NCCL_NET_GDR_MLOPART; more importantly the DEV overlay changes its topology id, which breaks Rome
// gpuId matching on an 8-GPU/8-NIC node. So SPX must stay undefined and CPX must not.
namespace {
constexpr int64_t kPhysGpuBusId = 0x11000;  // 0000:11:00.0 -- physical function
constexpr int64_t kAliasBusId   = 0x11001;  // 0000:11:00.1 -- a CPX HIP alias function
constexpr int64_t kHighFnBusId  = 0x1100f;  // 0000:11:00.f -- function >= NCCL_TOPO_MLOPART_DEV_MAX
}  // namespace

TEST_F(InitMicrotest, FillInfo_PhysicalFunctionZeroGpu_LeavesMloPartUndefined) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  g_pciDeviceClass = PCI_ACCELERATOR_CLASS;  // .0 is a real GPU in sysfs
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(NCCL_TOPO_UNDEF, info.mloPart);
  // SPX settles it: the device is not partitioned, so fn==0 never reaches the class probe.
  EXPECT_EQ(0, g_pciDeviceClassCalls);
}

TEST_F(InitMicrotest, FillInfo_AliasFunctionNotGpuInSysfs_StampsMloPartFromFunction) {
  FillInfoComm c;
  c.get()->busId = kAliasBusId;
  g_pciDeviceClass = "";  // the alias BDF has no sysfs entry
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(1, info.mloPart);
  // The probe must ask about the alias BDF; asking about .0 would report a GPU and lose the partition.
  EXPECT_EQ("0000:11:00.1", g_lastPciDeviceClassBusId);
}

TEST_F(InitMicrotest, FillInfo_GpuAtNonZeroFunction_LeavesMloPartUndefined) {
  FillInfoComm c;
  c.get()->busId = kAliasBusId;
  g_pciDeviceClass = "0x030000";  // a display-class GPU really lives at .1, so it is no HIP alias
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(NCCL_TOPO_UNDEF, info.mloPart);
  EXPECT_EQ(1, g_pciDeviceClassCalls);  // fn>0 does reach the probe; the class is what rejects it
}

TEST_F(InitMicrotest, FillInfo_FunctionAboveMloPartMax_LeavesMloPartUndefined) {
  FillInfoComm c;
  c.get()->busId = kHighFnBusId;
  g_pciDeviceClass = "";  // no sysfs entry, so only the fn bound can reject the stamp
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  // 0xf does not fit the 3-bit overlay index, so it must not be stamped even though sysfs is empty.
  EXPECT_EQ(NCCL_TOPO_UNDEF, info.mloPart);
  EXPECT_EQ(0, g_pciDeviceClassCalls);
}

TEST_F(InitMicrotest, FillInfo_CpxPartitionZero_StampsMloPartZero) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  g_pciDeviceClass = PCI_ACCELERATOR_CLASS;  // .0 is a real GPU in sysfs, exactly as in SPX
  g_pciComputePartition = "CPX";
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  // Partition 0 of a partitioned device is a partition. Only the mode tells it apart from the SPX
  // GPU above, which has an identical BDF and sysfs class.
  EXPECT_EQ(0, info.mloPart);
  EXPECT_EQ(0, g_pciDeviceClassCalls);  // the mode is decisive; the class probe is not reached
}

TEST_F(InitMicrotest, FillInfo_CpxAliasFunction_ProbesPhysicalFunctionForMode) {
  FillInfoComm c;
  c.get()->busId = kAliasBusId;
  g_pciDeviceClass = PCI_ACCELERATOR_CLASS;  // would reject the stamp if the class probe decided
  g_pciComputePartition = "CPX";
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(1, info.mloPart);
  // A CPX alias is usually absent from sysfs, so the mode has to be read from function 0.
  EXPECT_EQ("0000:11:00.0", g_lastPciComputePartitionBusId);
  EXPECT_EQ(0, g_pciDeviceClassCalls);
}

TEST_F(InitMicrotest, FillInfo_DpxPartitionZero_StampsMloPartZero) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  g_pciComputePartition = "DPX";  // any non-SPX mode means partitioned
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(0, info.mloPart);
}

TEST_F(InitMicrotest, FillInfo_NoPartitionModeReported_FallsBackToClassProbe) {
  FillInfoComm c;
  c.get()->busId = kAliasBusId;
  g_pciComputePartition = "";  // platform reports no mode, e.g. a VM or an older kernel
  g_pciDeviceClass = "";       // the alias BDF has no sysfs entry
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(1, info.mloPart);
  EXPECT_EQ(1, g_pciComputePartitionCalls);
  EXPECT_EQ(1, g_pciDeviceClassCalls);
}

TEST_F(InitMicrotest, FillInfo_NoPartitionModeReported_FunctionZeroLeavesMloPartUndefined) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  g_pciComputePartition = "";
  g_pciDeviceClass = PCI_ACCELERATOR_CLASS;
  ncclPeerInfo info{};
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  // The blind spot of the fallback: with no mode reported, partition 0 is indistinguishable from an
  // unpartitioned GPU, and staying undefined is the safe answer of the two.
  EXPECT_EQ(NCCL_TOPO_UNDEF, info.mloPart);
}

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
// The in-loop cudaSetDevice(dev) faults BEFORE ncclCommInitRankDev, so the real init tree is never entered.
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
  EXPECT_EQ(3, restoredTo);
  EXPECT_GE(setCalls, 2);
}

// Only the nId checks run before ncclInit(); the arms below it run after the real ncclInit().
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

// ncclInit uses std::call_once, so this is the single in-process ncclInit trigger; other outcomes must be isolated.
TEST_F(InitMicrotest, CommInitRankDev_PostInit_NullNewcomm_ReturnsInvalidArgument) {
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankDev(/*newcomm=*/nullptr, /*nranks=*/1, /*nId=*/1, &id, /*myrank=*/0, 0, &cfg, "t"));
}
// nranks < 1 is unreachable: the pre-init guard requires 0 < nId <= nranks.
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

// Process-isolated because ncclInit's call_once has already been burned (successfully) by the in-process tests above.
TEST_F(InitMicrotestIsolated, NcclInit_BootstrapNetInitFailure_ReturnsSystemError) {
  RUN_ISOLATED_TEST(
      "Init_NcclInit_BootstrapNetInitFailure",
      []() {
        g_bootstrapNetInitFail = true;
        ncclComm_t nc = nullptr;
        ncclUniqueId id{};
        ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
        ncclResult_t r = ncclCommInitRankDev(&nc, 1, 1, &id, 0, 0, &cfg, "t");
        ASSERT_EQ(ncclSystemError, r);
      });
}

#if defined(HIP_HOST_UNCACHED_MEMORY)
TEST_F(InitMicrotest, CheckHostUncacheMemSetting_Uncached_AlwaysSucceeds) {
  TopoComm t("gfx950:sramecc+");
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
  EXPECT_NE(nullptr, comm->ncclNet);
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

namespace {
// void + out-param, not a return value: gtest's ASSERT_* only unwinds a void-returning function.
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
  EXPECT_NE(nullptr, comm->devComm);
  EXPECT_NE(nullptr, comm->workFifoBuf);         // host workFifo arm taken
}

TEST_F(InitMicrotest, DevCommSetup_AsyncOpFails_ReturnsError) {
  InstallDevCommSetupSuccess();
  std::unique_ptr<ncclComm> comm;
  ASSERT_NO_FATAL_FAILURE(AllocedComm(comm));
  g_hipAsyncOpsResult = hipErrorInvalidValue;    // first calloc-async capture-mode fails
  EXPECT_NE(ncclSuccess, devCommSetup(comm.get()));
}

TEST_F(InitMicrotest, DevCommSetup_HostAllocFails_ReturnsError) {
  InstallDevCommSetupSuccess();
  std::unique_ptr<ncclComm> comm;
  ASSERT_NO_FATAL_FAILURE(AllocedComm(comm));
  g_hipHostMalloc = [](void**, std::size_t, unsigned) { return hipErrorMemoryAllocation; };
  EXPECT_NE(ncclSuccess, devCommSetup(comm.get()));
}

// commFree() ends in free(comm), so the comm MUST be ncclCalloc()'d (malloc-backed) exactly like production, NOT new'd.
TEST_F(InitMicrotest, CommFree_Null_ReturnsSuccess) {
  EXPECT_EQ(ncclSuccess, commFree(nullptr));
}

TEST_F(InitMicrotest, CommFree_AfterCommAlloc_ReturnsSuccessAndFrees) {
  InstallCommAllocSuccess();
  ncclComm* comm = nullptr;
  ASSERT_EQ(ncclSuccess, ncclCalloc(&comm, 1));
  ASSERT_EQ(ncclSuccess, commAlloc(comm, /*parent=*/nullptr, /*ndev=*/8, /*rank=*/0));
  uint32_t abortFlag = 0;
  int abortRef = 2;                                // >1 so commFree skips the abortFlag free-branch
  comm->abortFlag = &abortFlag;
  comm->abortFlagRefCount = &abortRef;
  EXPECT_EQ(ncclSuccess, commFree(comm));          // frees comm; do not touch it afterwards
  EXPECT_EQ(1, abortRef);
}

// ===========================================================================
// commCleanup (init.cc:3696). Pure ordering + error propagation: set the device,
// optionally finalize-then-unload the tuner, then commFree. Every assertion below
// keys on g_cleanupCallOrder, because a return code alone cannot see a swapped or
// dropped step. See init_fakes.h for who appends which name.
// ===========================================================================

namespace {
int kTunerContextSentinel = 0;  // distinct from comm, so forwarding the wrong pointer is visible

ncclResult_t FakeTunerFinalize(void* context) {
  ++g_tunerFinalizeCalls;
  g_tunerFinalizeLastContext = context;
  g_cleanupCallOrder.push_back("tunerFinalize");
  return g_tunerFinalizeResult;
}

// commCleanup ends in commFree, which free()s comm, so comm MUST be ncclCalloc'd, never new'd.
// abortFlag/abortRef are members so they outlive the call the comm points at them for.
struct CleanupComm {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;  // >1 so commFree skips the abortFlag free-branch
  ncclTuner_t tuner{};
};

// cudaDev defaults to 5, never 0: g_currentDevice starts at 0, so dev 0 could not tell a real
// hipSetDevice(comm->cudaDev) from a dropped call.
void MakeCleanupComm(CleanupComm& c, bool withTuner, int cudaDev = 5) {
  InstallCommAllocSuccess();
  ASSERT_EQ(ncclSuccess, ncclCalloc(&c.comm, 1));
  ASSERT_EQ(ncclSuccess, commAlloc(c.comm, /*parent=*/nullptr, /*ndev=*/8, /*rank=*/0));
  c.comm->abortFlag = &c.abortFlag;
  c.comm->abortFlagRefCount = &c.abortRef;
  c.comm->cudaDev = cudaDev;
  if (withTuner) {
    c.tuner.finalize = FakeTunerFinalize;
    c.comm->tuner = &c.tuner;
    c.comm->tunerContext = &kTunerContextSentinel;
  }
}

// Release a comm whose commCleanup returned early, so the test does not leak it. The abortRef guard
// keeps a regression that let commFree run anyway from turning into a double-free crash here.
void ReleaseUncleanedComm(CleanupComm& c) {
  ASSERT_EQ(2, c.abortRef) << "commFree already ran to completion; the comm is gone";
  g_ncclCeFinalizeResult = ncclSuccess;
  ASSERT_EQ(ncclSuccess, commFree(c.comm));
}
}  // namespace

// --- tuner == NULL arm ---
TEST_F(InitMicrotest, CommCleanup_NoTuner_SetsDeviceAndFreesCommOnly) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/false));
  g_currentDevice = 3;  // != cudaDev, so the set is observable

  EXPECT_EQ(ncclSuccess, commCleanup(c.comm));  // frees c.comm; do not touch it afterwards

  EXPECT_EQ(std::vector<std::string>({"commFree"}), g_cleanupCallOrder);
  EXPECT_EQ(5, g_currentDevice);
  EXPECT_EQ(0, g_tunerFinalizeCalls);
  EXPECT_EQ(nullptr, g_ncclTunerPluginUnloadLastComm);
  EXPECT_EQ(1, c.abortRef);  // commFree really ran, not just the marker
}

// --- tuner != NULL arm: the whole ordering contract ---
TEST_F(InitMicrotest, CommCleanup_WithTuner_FinalizesThenUnloadsThenFrees) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/true));
  ncclComm* const commAddr = c.comm;
  g_currentDevice = 3;

  EXPECT_EQ(ncclSuccess, commCleanup(c.comm));  // frees c.comm; do not touch it afterwards

  EXPECT_EQ(std::vector<std::string>({"tunerFinalize", "tunerUnload", "commFree"}), g_cleanupCallOrder);
  EXPECT_EQ(5, g_currentDevice);
  EXPECT_EQ(1, g_tunerFinalizeCalls);
  EXPECT_EQ(static_cast<void*>(&kTunerContextSentinel), g_tunerFinalizeLastContext);
  EXPECT_EQ(commAddr, g_ncclTunerPluginUnloadLastComm);
  EXPECT_EQ(1, c.abortRef);
}

// --- cudaSetDevice failure arm: short-circuits before the tuner guard ---
TEST_F(InitMicrotest, CommCleanup_SetDeviceFails_ReturnsCudaErrorAndRunsNothingElse) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/true));
  g_currentDevice = 3;
  auto savedSetDevice = g_hipSetDevice;
  g_hipSetDevice = [](int) { return hipErrorInvalidDevice; };

  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { res = commCleanup(c.comm); });

  EXPECT_EQ(ncclUnhandledCudaError, res);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "HIP failure:"));  // positive anchor for the empty-order check
  EXPECT_TRUE(g_cleanupCallOrder.empty());
  EXPECT_EQ(0, g_tunerFinalizeCalls);
  EXPECT_EQ(3, g_currentDevice);
  EXPECT_EQ(2, c.abortRef);  // pins current behaviour: a device error skips commFree, so the comm leaks

  // Restore only this hook: ResetHipFakes() would also revert the InstallCommAllocSuccess results commFree needs.
  g_hipSetDevice = savedSetDevice;
  ASSERT_NO_FATAL_FAILURE(ReleaseUncleanedComm(c));
}

// --- tuner->finalize failure arm: unload and commFree must not run ---
TEST_F(InitMicrotest, CommCleanup_TunerFinalizeFails_SkipsUnloadAndFree) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/true));
  g_currentDevice = 3;
  g_tunerFinalizeResult = ncclInvalidUsage;  // distinct from every other arm's code

  EXPECT_EQ(ncclInvalidUsage, commCleanup(c.comm));

  EXPECT_EQ(std::vector<std::string>({"tunerFinalize"}), g_cleanupCallOrder);
  EXPECT_EQ(nullptr, g_ncclTunerPluginUnloadLastComm);
  EXPECT_EQ(5, g_currentDevice);  // the device was set before the failure
  EXPECT_EQ(2, c.abortRef);       // pins current behaviour: a tuner plugin error leaks the whole comm

  ASSERT_NO_FATAL_FAILURE(ReleaseUncleanedComm(c));
}

// --- ncclTunerPluginUnload failure arm: commFree must not run ---
TEST_F(InitMicrotest, CommCleanup_TunerUnloadFails_SkipsFree) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/true));
  ScopedHook unload(g_ncclTunerPluginUnload, [](ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, commCleanup(c.comm));

  EXPECT_EQ(std::vector<std::string>({"tunerFinalize", "tunerUnload"}), g_cleanupCallOrder);
  EXPECT_EQ(1, g_tunerFinalizeCalls);
  EXPECT_EQ(2, c.abortRef);

  ASSERT_NO_FATAL_FAILURE(ReleaseUncleanedComm(c));
}

// --- commFree failure arm: the error reaches the caller unchanged ---
TEST_F(InitMicrotest, CommCleanup_CommFreeFails_PropagatesError) {
  CleanupComm c;
  ASSERT_NO_FATAL_FAILURE(MakeCleanupComm(c, /*withTuner=*/true));
  g_ncclCeFinalizeResult = ncclInternalError;  // commFree's first NCCLCHECK; it bails before free(comm)

  EXPECT_EQ(ncclInternalError, commCleanup(c.comm));

  EXPECT_EQ(std::vector<std::string>({"tunerFinalize", "tunerUnload", "commFree"}), g_cleanupCallOrder);
  EXPECT_EQ(2, c.abortRef);

  ASSERT_NO_FATAL_FAILURE(ReleaseUncleanedComm(c));
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
    comm_->sharedRes = sr_.get();  // owner stays null, so exit::2408 short-circuits before ncclCuMemEnable
    comm_->ncclNet = net_.get();
    comm_->archName = archName_.empty() ? nullptr : &archName_[0];
    comm_->commHash = 0xC0FFEEULL;
    comm_->compCap = 90;
    std::memset(timers_, 0, sizeof(timers_));
  }
  // Resetting the seam is not optional bookkeeping: installTopo() parks topo_.get() in a file-scope
  // global, so without this the global outlives the object and holds freed memory until TearDown.
  ~TransportsRankComm() {
    if (topo_) {
      g_ncclTopoGetSystem = [](ncclComm*, ncclTopoSystem**, const char*) { return ncclRemoteError; };
    }
    free(comm_->peerInfo);
  }
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
      return ncclInvalidArgument;  // distinct from every seam default, so a stray hit is traceable
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

// Restores g_ncclTopoGetSystem when it goes out of scope. Needed only where the installed lambda
// captures a stack local by reference -- the global would otherwise outlive what it points at.
class ScopedTopoGetSystem {
 public:
  ~ScopedTopoGetSystem() {
    g_ncclTopoGetSystem = [](ncclComm*, ncclTopoSystem**, const char*) { return ncclRemoteError; };
  }
};

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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) { return hipErrorInvalidValue; };
  g_bootstrapAllGather = [](void*, void*, int) -> ncclResult_t {
    ADD_FAILURE() << "AllGather1 must not run after fillInfo failed";
    return ncclInternalError;
  };
  EXPECT_EQ(ncclUnhandledCudaError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_AllGatherFails_PropagatesAndLeavesPeerInfoInvalid) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);  // g_bootstrapAllGather defaults to ncclInternalError
  EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->peerInfoValid);  // :1467 sits past the failure point
}

TEST_F(InitMicrotest, InitTransportsRank_AllGatherSucceeds_MarksPeerInfoValid) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));  // stops at :1576
  EXPECT_TRUE(c.get()->peerInfoValid);  // positive anchor for the negative assertion above
}

// --- The peer-scan loop (init.cc:1470-1495) ---

TEST_F(InitMicrotest, InitTransportsRank_PeerVersionMismatch_WarnsWithBothRanksAndVersions) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/1);
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  g_cuMemEnable = [] { return 1; };
  std::vector<PeerSpec> specs(4);
  specs[2].cuMemSupport = 0;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->cuMemSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_AllPeersCuMemSupport_KeepsCommCuMemSupportSet) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  g_cuMemEnable = [] { return 1; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->cuMemSupport);  // :1469 sets it, the loop never clears it
}

TEST_F(InitMicrotest, InitTransportsRank_PeerWithMloPart_SetsHasMloPart) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[3].mloPart = 2;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(c.get()->hasMloPart);
}

TEST_F(InitMicrotest, InitTransportsRank_NoPeerWithMloPart_LeavesHasMloPartUnset) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->hasMloPart);
}

// NOT ASSERTABLE FROM THIS RUNG, deliberately: the four `global*Support` accumulators at :1491-1494 are
// function-locals first read at :2347-2363, ~700 lines past the terminator. They execute (so they count
// as covered) but nothing here can observe them, and deleting any of the four leaves the suite green.
// Adding PeerSpec knobs would not help without an oracle -- assert them from the rung reaching :2347.
// Tracked under AICOMRCCL-1685 along with the rest of this suite.
TEST_F(InitMicrotest, InitTransportsRank_ComputesMinAndMaxCompCapAcrossPeers) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[1].compCap = 42;   // asymmetric around self's 90, so swapping min/max is visible
  specs[2].compCap = 110;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(42, c.get()->minCompCap);
  EXPECT_EQ(110, c.get()->maxCompCap);
}

// --- The duplicate-GPU guard (init.cc:1484-1489) ---

TEST_F(InitMicrotest, InitTransportsRank_DuplicateGpuUuidSameHost_ReturnsInvalidUsageAndSkipsCleanup) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[1].uuidTag = 9;
  specs[2].uuidTag = 9;
  specs[2].node = 1;  // same UUID but a different host -- the hostHash conjunct at :1484 saves it
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));  // ran on to :1576
  });
  // No log-liveness anchor is possible in the DEFAULT build: the only line this path logs is the :1547
  // TRACE, compiled out unless ENABLE_TRACE. The oracle is the two positive anchors below instead --
  // the ncclRemoteError sentinel (proves :1576 was reached) and the exit: call count.
  EXPECT_FALSE(LogHas(log, "Multiple Ranks are using the same GPU/Partition")) << "actual log:\n" << log;
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);  // positive anchor: it really did reach exit:
}

TEST_F(InitMicrotest, InitTransportsRank_DuplicateGpuUuid_MultiRankGpuEnabled_Continues) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"MULTI_RANK_GPU_ENABLE", 1}});
  std::vector<PeerSpec> specs(4);
  specs[1].uuidTag = 9;
  specs[2].uuidTag = 9;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  EXPECT_FALSE(LogHas(log, "Multiple Ranks are using the same GPU/Partition")) << "actual log:\n" << log;
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
}

// --- hasMultiRankNvml (init.cc:1482) ---
// PINS CURRENT BEHAVIOUR, WHICH LOOKS ODD: the assignment is `=`, not `|=`, inside the (i,j) double
// loop, so only the FINAL pair survives and an earlier collision is erased. On AMD this is write-only
// dead state, not a live wrong answer: the sole reader (src/transport/nvls.cc:252) sits inside
// `#if CUDART_VERSION >= 12010`, and CUDART_VERSION is not defined under hipcc. These two tests
// document what ships today so a `|=` change has to update a test rather than pass silently.
TEST_F(InitMicrotest, InitTransportsRank_MultiRankNvml_EarlyCollisionOverwrittenByLastPair) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[1].nvmlDev = 7;  // ranks 1 and 2 really do share a device on the same host...
  specs[2].nvmlDev = 7;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_FALSE(c.get()->hasMultiRankNvml);  // ...but the last pair (3,2) does not, and it wins
}

TEST_F(InitMicrotest, InitTransportsRank_MultiRankNvml_LastPairCollision_IsTheOnlyOneObserved) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[2].nvmlDev = 7;  // the final (i,j) pair examined is (3,2)
  specs[3].nvmlDev = 7;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(c.get()->hasMultiRankNvml);
}

// --- The MNNVL scope test (init.cc:1500-1510) ---
// mnnvlEnable==1 forces the check; ==0 forbids it; anything else (default 2) means "auto", which
// needs (multi-node OR gfx1250) AND p2pLevel != 0. g_ncclMnnvlCheckCalls is the whole oracle --
// the return code is identical either way.

TEST_F(InitMicrotest, InitTransportsRank_UserP2pLevelFails_PropagatesAndSkipsMnnvlCheck) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  g_ncclGetUserP2pLevel = [](int*) { return ncclInvalidArgument; };
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlForcedOn_ChecksEvenOnSingleNode) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"MNNVL_ENABLE", 1}});
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));  // all one node, arch gfx942
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlForcedOff_SkipsCheckEvenOnMultiNode) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"MNNVL_ENABLE", 0}});  // the zero-valued arm, tested alone: 0 and "auto" differ only here
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoMultiNode_ChecksAndCountsNodes) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;  // one peer off-node -> nNodes becomes 2 at :1477
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);  // the only observable that nNodes>1 was computed
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoSingleNode_SkipsCheck) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoGfx1250SingleNode_Checks) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0, /*archName=*/"gfx1250");
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclMnnvlCheckCalls);  // single node, but the arch arm at :1504 opens auto scope
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoP2pLevelZero_SkipsCheckDespiteMultiNode) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  g_ncclGetUserP2pLevel = [](int* level) { *level = 0; return ncclSuccess; };
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlAutoNullArchName_DoesNotDereference) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0, /*archName=*/"");  // comm->archName == NULL; :1504 must short-circuit
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclMnnvlCheckCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlCheckFails_PropagatesError) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"MNNVL_ENABLE", 1}});
  g_ncclMnnvlCheckResult = ncclSystemError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);
}

// --- The intra-process block (init.cc:1512-1565) ---

TEST_F(InitMicrotest, InitTransportsRank_SameHostAndPid_ClearsNvlsRegSupport) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}});  // else :1545 forces it straight back to 1
  std::vector<PeerSpec> specs(4);
  auto peer1 = std::make_unique<ncclComm>();  // heap: ncclComm embeds channels[MAXCHANNELS], far too big for the stack
  specs[1].proc = 0;  // shares our process, so the :1533 inner scan finds a colliding pair
  specs[1].comm = peer1.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_AllDistinctProcesses_KeepsNvlsRegSupport) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}});
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->nvlsRegSupport);  // positive anchor for the test above
}

TEST_F(InitMicrotest, InitTransportsRank_SingleProcMemRegEnabled_RestoresNvlsRegSupport) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 1}});
  std::vector<PeerSpec> specs(4);
  auto peer1 = std::make_unique<ncclComm>();
  specs[1].proc = 0;  // the scan clears it at :1536; :1545 then puts it back
  specs[1].comm = peer1.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_MnnvlComm_ClearsNvlsRegSupportBeforeSingleProcOverride) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetParams({{"SINGLE_PROC_MEM_REG_ENABLE", 1}});  // would set it to 1, but MNNVL wins the else-if
  c.get()->MNNVL = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->nvlsRegSupport);
}

TEST_F(InitMicrotest, InitTransportsRank_IntraProcPeers_LinkIntraNextInReverseRankOrder) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  auto peer1 = std::make_unique<ncclComm>();
  auto peer2 = std::make_unique<ncclComm>();
  auto peer3 = std::make_unique<ncclComm>();
  std::vector<PeerSpec> specs(4);
  specs[1].proc = 0;
  specs[1].comm = peer1.get();
  specs[2].proc = 0;
  specs[2].comm = peer2.get();
  specs[3].proc = 0;
  specs[3].comm = peer3.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/2);
  auto peer1 = std::make_unique<ncclComm>();
  auto peer3 = std::make_unique<ncclComm>();
  std::vector<PeerSpec> specs(4);
  // Rank 0 stays in its own process, so ranks 1,2,3 form the group and rank 1 leads it.
  specs[1].proc = 0;
  specs[1].comm = peer1.get();
  specs[3].proc = 0;
  specs[3].comm = peer3.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->intraRank);   // local index within the process, not the global rank 2
  EXPECT_EQ(3, c.get()->intraRanks);
  EXPECT_EQ(peer1.get(), c.get()->intraComm0);
  EXPECT_EQ(nullptr, c.get()->intraNext);  // only intraProcRank0 builds the chain, and that is rank 1
}

// The leader test at :1526 is `intraProcRank0 == rank`. Every other case here has either rank 0 leading
// or a non-leader, so the conjunct is indistinguishable from a bare `intraProcRank0 == 0`; here rank 1
// leads its own group, which is the only shape where the two differ.
TEST_F(InitMicrotest, InitTransportsRank_NonZeroRankLeadsItsProcess_StillBuildsTheChain) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/1);
  auto peer2 = std::make_unique<ncclComm>();
  auto peer3 = std::make_unique<ncclComm>();
  std::vector<PeerSpec> specs(4);
  // Rank 0 sits in its own process, so ranks 1,2,3 group up and rank 1 leads them.
  specs[0].proc = -1;
  specs[2].proc = 0;
  specs[2].comm = peer2.get();
  specs[3].proc = 0;
  specs[3].comm = peer3.get();
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->intraRank);
  EXPECT_EQ(3, c.get()->intraRanks);
  EXPECT_EQ(c.get(), c.get()->intraComm0);
  EXPECT_EQ(peer3.get(), c.get()->intraNext);  // chain built, in reverse rank order
  EXPECT_EQ(peer2.get(), peer3->intraNext);
  EXPECT_EQ(nullptr, peer2->intraNext);
}

// nRanks == 1 is the most common non-trivial production shape, and the only rank layout the suite
// otherwise never builds -- every other TransportsRankComm here uses 4 ranks or 0.
TEST_F(InitMicrotest, InitTransportsRank_SingleRank_IsItsOwnIntraProcGroup) {
  TransportsRankComm c(/*nRanks=*/1, /*rank=*/0);
  g_cuMemEnable = [] { return 1; };  // fillInfo stamps our own slot from this, and :1478 reads it back
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(1));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->intraRank);
  EXPECT_EQ(1, c.get()->intraRanks);
  EXPECT_EQ(c.get(), c.get()->intraComm0);
  EXPECT_EQ(nullptr, c.get()->intraNext);  // nobody else to link
  EXPECT_EQ(1, c.get()->cuMemSupport);     // the scan ran over exactly our own slot
}

// The guard at :1549 is `intraProcRank == -1 || intraProcRank0 == -1 || peerInfo[...].comm == NULL`.
// Only the first and third disjuncts are reachable. The second is DEAD: intraProcRank is assigned
// only inside the same `if` body that assigns intraProcRank0, so intraProcRank != -1 implies
// intraProcRanks was incremented at least once, which implies intraProcRank0 != -1. Do not try to
// cover it. The first needs an empty comm -- see the ZeroRanks test below.
TEST_F(InitMicrotest, InitTransportsRank_IntraProcRank0PeerHasNullComm_ReturnsInternalErrorAndWarns) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/1);
  std::vector<PeerSpec> specs(4);
  specs[0].proc = 0;  // rank 0 shares our process but never registered its comm; :1526 cannot fire
  specs[0].comm = nullptr;
  InstallPeerInfoAllGather(c, specs);
  const std::string log = RcclUnitTesting::CaptureLog([&] {
    EXPECT_EQ(ncclInternalError, initTransportsRank(c.get(), nullptr, c.timers()));
  });
  // Lead text AND value tail must come from the SAME line. The :1547 TRACE ends in an identical
  // "intraProcRank %d intraProcRanks %d intraProcRank0 %d" and the ncclDebugLog fake ignores level, so
  // two separate searches over the whole buffer let the TRACE satisfy the tail under ENABLE_TRACE.
  const std::string warn = LogLineWith(log, "Failed to determine intra proc ranks rank 1 ");
  ASSERT_FALSE(warn.empty()) << "no WARN line in log:\n" << log;
  EXPECT_NE(std::string::npos, warn.find("intraProcRank 1 intraProcRanks 2 intraProcRank0 0")) << warn;
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
  // Same trap as above: one line must carry both halves.
  const std::string warn = LogLineWith(log, "Failed to determine intra proc ranks rank 0 ");
  ASSERT_FALSE(warn.empty()) << "no WARN line in log:\n" << log;
  EXPECT_NE(std::string::npos, warn.find("intraProcRank -1 intraProcRanks 0 intraProcRank0 -1")) << warn;
}

// --- The ladder terminator and the exit: block (init.cc:1569-1576, :2402-2419) ---

TEST_F(InitMicrotest, InitTransportsRank_TopoGetSystemFails_PropagatesAndRunsCleanup) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  g_ncclTopoGetSystem = [](ncclComm*, ncclTopoSystem**, const char*) { return ncclSystemError; };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));  // not the default error
  EXPECT_EQ(1, g_ncclOsCpuCountCalls);          // fail: fell through to exit:
  EXPECT_EQ(0u, g_ncclOsSetAffinityMasks.size());  // cpu count 0, so :2404 stayed unreached
}

TEST_F(InitMicrotest, InitTransportsRank_NoTopoDumpFile_PassesNullPathToTopoGetSystem) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  std::vector<std::string> paths;
  // The seam captures a reference to `paths`, a local. Drop it when this scope ends rather than
  // relying on nothing reaching topo detection between here and TearDown's ResetInitFakes().
  ScopedTopoGetSystem restore_topo_seam;
  g_ncclTopoGetSystem = [&paths](ncclComm*, ncclTopoSystem**, const char* f) {
    paths.push_back(f ? f : "<null>");
    return ncclSystemError;
  };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(std::vector<std::string>({"<null>"}), paths);  // :1572 false, so only :1576 runs
}

TEST_F(InitMicrotest, InitTransportsRank_TopoDumpFileSet_PassesThatPathToTopoGetSystem) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  SetMicroEnv("NCCL_TOPO_DUMP_FILE", "/tmp/rccl-topo-microtest.xml");
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  std::vector<std::string> paths;
  // The seam captures a reference to `paths`, a local. Drop it when this scope ends rather than
  // relying on nothing reaching topo detection between here and TearDown's ResetInitFakes().
  ScopedTopoGetSystem restore_topo_seam;
  g_ncclTopoGetSystem = [&paths](ncclComm*, ncclTopoSystem**, const char* f) {
    paths.push_back(f ? f : "<null>");
    return ncclSystemError;
  };
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  // Recording the argument is the only way to tell the :1573 call site from the :1576 one.
  EXPECT_EQ(std::vector<std::string>({"/tmp/rccl-topo-microtest.xml"}), paths);
}

TEST_F(InitMicrotest, InitTransportsRank_CpuAffinitySet_RestoresThatAffinityAtExit) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  CPU_ZERO(&c.get()->cpuAffinity);
  CPU_SET(5, &c.get()->cpuAffinity);
  g_ncclOsCpuCountValue = 1;  // non-zero, so exit::2404 restores the mask
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  g_tuningIndexValue = 7;
  // installTopo() value-initialises the system, so the four "reset to false/0" fields below would pass
  // whether or not :1584-1589 ran. Poison them first, exactly as the graphs[] test poisons its slots.
  topo->pivotA2AEnabled = true;
  topo->pivotA2ANumBiRings = 7;
  topo->ll128Enabled = true;
  topo->treeDefined = true;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));  // stops at :1648
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0, /*archName=*/"gfx90a");
  ncclTopoSystem* topo = c.installTopo();
  g_tuningIndexValue = 3;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(3, topo->tuning);
  EXPECT_EQ("gfx90a", g_tuningIndexLastArch);  // pins that :1577 forwards archName, not a constant
}

TEST_F(InitMicrotest, InitTransportsRank_ComputePathsFailsBeforeTrim_StopsAtFirstCall) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoComputePathsFailAt = 0;  // the :1591 call
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputePathsCalls);  // never reached the post-trim recompute
}

TEST_F(InitMicrotest, InitTransportsRank_ComputePathsFailsAfterTrim_RunsBothCalls) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoComputePathsFailAt = 1;  // the :1596 recompute; only a call index separates it from :1591
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoComputePathsCalls);
}

TEST_F(InitMicrotest, InitTransportsRank_TrimSystemFails_PropagatesBetweenThePathsCalls) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoTrimSystemResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputePathsCalls);  // trim sits between :1591 and :1596
}

TEST_F(InitMicrotest, InitTransportsRank_TopoSearchInitFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoSearchInitResult = ncclInvalidUsage;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoComputePathsCalls);  // both path computations already ran
}

TEST_F(InitMicrotest, InitTransportsRank_TopoComputeCommCpuFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoComputeCommCPUResult = ncclUnhandledCudaError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclUnhandledCudaError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_TopoPrintFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoPrintResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(-1, g_ncclTopoGetCpuAffinityLastRank);  // :1607 is past the failure
}

// --- CPU affinity (init.cc:1607-1611) ---

TEST_F(InitMicrotest, InitTransportsRank_TopoGetCpuAffinityFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity*) { return ncclSystemError; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_CpuAffinityLookedUpForThisRank) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/2);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoGetCpuAffinityLastRank);  // :1607 forwards comm->rank, not a constant
}

TEST_F(InitMicrotest, InitTransportsRank_EmptyCpuAffinity_SkipsSaveAndApply) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));  // cpu count 0 by default
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  // Both cpu-count sites ran and both saw an empty mask, which is what "skips save and apply" means;
  // asserting only the empty setAffinity vector would match its .clear() reset value.
  ASSERT_EQ(2u, g_ncclOsCpuCountMasks.size());  // :1608 and exit::2403
  EXPECT_EQ(0, CPU_COUNT(&g_ncclOsCpuCountMasks[0]));
  EXPECT_EQ(0, CPU_COUNT(&g_ncclOsCpuCountMasks[1]));
  EXPECT_EQ(0u, g_ncclOsSetAffinityMasks.size());  // so neither :1610 nor exit::2404 ran
}

TEST_F(InitMicrotest, InitTransportsRank_NonEmptyCpuAffinity_AppliesAtBothSites) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity* a) {
    CPU_ZERO(a);
    CPU_SET(6, a);
    return ncclSuccess;
  };
  g_ncclOsCpuCountValue = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  ASSERT_EQ(2u, g_ncclOsSetAffinityMasks.size());  // :1610 on the way in, exit::2404 on the way out
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsSetAffinityMasks[0]));
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsSetAffinityMasks[1]));
  EXPECT_TRUE(CPU_ISSET(6, &c.get()->cpuAffinity));  // :1607 wrote through to the comm
  // ncclOsCpuCount is asked about the SAME mask at both its call sites; without this, handing either
  // :1608 or exit::2403 the zeroed affinitySave instead of comm->cpuAffinity goes unnoticed.
  ASSERT_EQ(2u, g_ncclOsCpuCountMasks.size());
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsCpuCountMasks[0]));
  EXPECT_TRUE(CPU_ISSET(6, &g_ncclOsCpuCountMasks[1]));
}

TEST_F(InitMicrotest, InitTransportsRank_OsGetAffinityFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclOsGetAffinity = [](ncclAffinity*) { return ncclSystemError; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
}

TEST_F(InitMicrotest, InitTransportsRank_OsSetAffinityFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclTopoGetCpuAffinity = [](ncclTopoSystem*, int, ncclAffinity* a) {
    CPU_ZERO(a);
    CPU_SET(6, a);  // the GPU-local mask
    return ncclSuccess;
  };
  g_ncclOsGetAffinity = [](ncclAffinity* a) {
    CPU_ZERO(a);
    CPU_SET(9, a);  // the caller's original mask
    return ncclSuccess;
  };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  c.get()->config.collnetEnable = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, c.get()->config.collnetEnable);
}

TEST_F(InitMicrotest, InitTransportsRank_CollNetPluginPresent_KeepsCollnetEnable) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  ncclCollNet_t collNet{};
  c.get()->ncclCollNet = &collNet;
  c.get()->config.collnetEnable = 1;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, c.get()->config.collnetEnable);  // :1614 left it alone, unlike the test above
  EXPECT_EQ(1, g_ncclNvlsInitCalls);            // and execution really did carry on past :1614
}

TEST_F(InitMicrotest, InitTransportsRank_NvlsInitFails_ReturnsWithoutRunningCleanup) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
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
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/3);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[0].node = 1;  // hosts, in rank order: B B C A -- self (rank 3) is on host A, seen last,
  specs[1].node = 1;  // so nHosts and hostIdx take different values and neither is 0
  specs[2].node = 2;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(3, topo->nHosts);
  EXPECT_EQ(2, topo->hostIdx);
}

TEST_F(InitMicrotest, InitTransportsRank_SingleHost_HostIndexIsZero) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  topo->hostIdx = 9;  // 0 is the zero-init value, so poison it to make the assertion below live
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, topo->nHosts);
  EXPECT_EQ(0, topo->hostIdx);
}

// :1633 attributes hostIdx by comparing HOST HASHES, not rank indices. Every other layout here has
// self leading its own host group, where the two are indistinguishable; here self is rank 3 on a host
// first seen at rank 2, so a rank-matching mutant yields 0 instead of 1.
TEST_F(InitMicrotest, InitTransportsRank_SelfNotFirstOnItsHost_HostIndexFollowsTheHostHash) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/3);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[0].node = 1;  // layout B B A A: ranks 0,1 on host B, ranks 2,3 (incl. self) on host A
  specs[1].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, topo->nHosts);
  EXPECT_EQ(1, topo->hostIdx);  // host A is the second distinct host encountered
}

// Interleaved hosts: the :1626 de-dup scan must look at every earlier rank, not just the adjacent one.
TEST_F(InitMicrotest, InitTransportsRank_InterleavedHosts_DeDupScanCoversAllEarlierRanks) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[1].node = 1;  // layout A B A B -- an adjacent-only scan would count 4 hosts, not 2
  specs[3].node = 1;
  InstallPeerInfoAllGather(c, specs);
  topo->hostIdx = 9;  // self leads host A, so the written value is 0 -- poison to tell it from zero-init
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, topo->nHosts);
  EXPECT_EQ(0, topo->hostIdx);
}

TEST_F(InitMicrotest, InitTransportsRank_UniformRanksPerHost_KeepsPresetTopoMatching) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[2].node = 1;  // 2 hosts x 2 ranks
  specs[3].node = 1;
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, topo->nHosts);  // live: skipPresetTopoMatching false is also the zero-init value
  EXPECT_FALSE(topo->skipPresetTopoMatching);
}

TEST_F(InitMicrotest, InitTransportsRank_NonUniformRanksPerHost_SkipsPresetTopoMatching) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::vector<PeerSpec> specs(4);
  specs[3].node = 1;  // 3 ranks on one host, 1 on the other
  InstallPeerInfoAllGather(c, specs);
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_TRUE(topo->skipPresetTopoMatching);
}

TEST_F(InitMicrotest, InitTransportsRank_TopoComputeFails_PropagatesAndRunsCleanup) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  g_ncclTopoCompute = [](ncclTopoSystem*, ncclTopoGraph*) { return ncclInvalidArgument; };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputeCalls);
  EXPECT_EQ(2, g_ncclOsCpuCountCalls);  // :1608 and exit::2403 -- unlike the :1618 bypass above
}

// --- The graphs[] alias table (init.cc:1401) ---
// `graphs[]` is indexed by ALGORITHM (7) but there are only 5 graphs, because two algorithms reuse
// another's: NVLS_TREE shares the NVLS graph and PAT shares the TREE graph. So graphs[5] == graphs[4]
// and graphs[6] == graphs[0] are the same objects, and a write through the alias silently edits the
// real graph. Upstream NCCL owns this (2.22.3-1 added the nvls duplicate, 2.23.4-1 the tree one);
// RCCL only reflowed the line, so the guard below is a canary, not a fix.

// Guards HEADER RENUMBERING only: NCCL_ALGO_* are #defines, so if plugin/nccl_tuner.h renumbers one,
// :1401 keeps compiling while mapping the wrong graph to the wrong algorithm, and these fail the build.
// It does NOT catch a positional REORDER of the initializers at :1401 -- verified by swapping two of
// them, which no test in this suite detects, because the local graphs[] is first read at :1875, past
// both terminators. Nothing here can observe that array; only its aliasing side effects (test below).
TEST_F(InitMicrotest, AlgorithmEnum_NumberingStillMatchesTheAliasTablePositions) {
  static_assert(NCCL_NUM_ALGORITHMS == 7, ":1401 lists exactly 7 initializers");
  static_assert(NCCL_ALGO_TREE == 0, "graphs[0] is treeGraph");
  static_assert(NCCL_ALGO_RING == 1, "graphs[1] is ringGraph");
  static_assert(NCCL_ALGO_COLLNET_DIRECT == 2, "graphs[2] is collNetDirectGraph");
  static_assert(NCCL_ALGO_COLLNET_CHAIN == 3, "graphs[3] is collNetChainGraph");
  static_assert(NCCL_ALGO_NVLS == 4, "graphs[4] is nvlsGraph");
  static_assert(NCCL_ALGO_NVLS_TREE == 5, "graphs[5] ALIASES graphs[4]");
  static_assert(NCCL_ALGO_PAT == 6, "graphs[6] ALIASES graphs[0]");
  SUCCEED() << "alias table assumptions hold; see init.cc:1401";
}

// Poisons every graph slot, runs to the :1648 terminator, and asserts that ONLY the ring graph was
// written. Watching the aliased-TO slots is the point: a stray write through the local graphs[5]
// lands on NVLS (slot 4) and through graphs[6] on TREE (slot 0), leaving slots 5 and 6 pristine --
// so a test that only poisoned the two unused slots would miss exactly the corruption it was for.
TEST_F(InitMicrotest, InitTransportsRank_ThroughRingCompute_WritesOnlyTheRingGraph) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  const std::vector<unsigned char> poison(sizeof(ncclTopoGraph), 0xA5);
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    std::memcpy(&c.get()->graphs[a], poison.data(), poison.size());
  }
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  // Ring IS seeded at :1643-1647 -- without this the test would pass on a run that did nothing.
  EXPECT_NE(0, std::memcmp(&c.get()->graphs[NCCL_ALGO_RING], poison.data(), poison.size()));
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    if (a == NCCL_ALGO_RING) continue;
    EXPECT_EQ(0, std::memcmp(&c.get()->graphs[a], poison.data(), poison.size()))
        << "graphs[" << a << "] was written before :1648; if a is 0 or 4 this may be a write through "
        << "the :1401 alias for PAT or NVLS_TREE";
  }
}

TEST_F(InitMicrotest, InitTransportsRank_RingGraphSeededBeforeTopoCompute) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclRemoteError, initTransportsRank(c.get(), nullptr, c.timers()));
  const ncclTopoGraph& ring = c.get()->graphs[NCCL_ALGO_RING];
  EXPECT_EQ(0, ring.id);
  EXPECT_EQ(NCCL_TOPO_PATTERN_RING, ring.pattern);
  EXPECT_EQ(1, ring.minChannels);
  EXPECT_EQ(MAXCHANNELS / 2, ring.maxChannels);
  // And that :1648 was handed THAT graph -- the seeded fields above are written by :1643-1647
  // regardless of which pointer the compute call then receives.
  ASSERT_EQ(1u, g_ncclTopoComputeGraphs.size());
  EXPECT_EQ(&c.get()->graphs[NCCL_ALGO_RING], g_ncclTopoComputeGraphs[0]);
}

// ===========================================================================
// initTransportsRank rung 3: the ring/tree/CollNet/NVLS graph block, the graph
// dump and the P2P peer cap (src :1649-1774). ncclTopoCompute now succeeds and
// stamps nChannels, and ncclTopoComputeP2pChannelsPerPeer (:1774) takes over as
// terminator -- with ncclTimeout, a DIFFERENT sentinel from the earlier rungs, so
// a test that forgot to arm the compute seam cannot satisfy a rung-3 oracle.
// ===========================================================================
namespace {
// Walks the ladder through the whole graph block. Stamping nChannels matters: :1671-1672 read
// ringGraph->nChannels back to size the tree graph, so a seam that only returned success would
// leave every downstream channel count at zero and hide that dependency.
void InstallTopoComputeSuccess(int nChannels) {
  g_ncclTopoCompute = [nChannels](ncclTopoSystem*, ncclTopoGraph* g) {
    g->nChannels = nChannels;
    return ncclSuccess;
  };
}
}  // namespace

TEST_F(InitMicrotest, InitTransportsRank_TreeGraphSeededFromRingChannelCount) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));  // reached :1774
  const ncclTopoGraph& tree = c.get()->graphs[NCCL_ALGO_TREE];
  EXPECT_EQ(1, tree.id);
  EXPECT_EQ(NCCL_TOPO_PATTERN_BALANCED_TREE, tree.pattern);
  EXPECT_EQ(5, tree.minChannels);  // both taken from ringGraph->nChannels, not from a constant
  EXPECT_EQ(5, tree.maxChannels);
}

TEST_F(InitMicrotest, InitTransportsRank_CollNetGraphsSeededWithDistinctIdsAndPatterns) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  const ncclTopoGraph& chain = c.get()->graphs[NCCL_ALGO_COLLNET_CHAIN];
  EXPECT_EQ(2, chain.id);
  EXPECT_EQ(NCCL_TOPO_PATTERN_TREE, chain.pattern);
  EXPECT_EQ(1, chain.collNet);
  EXPECT_EQ(5, chain.minChannels);  // ring-sized, unlike direct below
  EXPECT_EQ(5, chain.maxChannels);
  const ncclTopoGraph& direct = c.get()->graphs[NCCL_ALGO_COLLNET_DIRECT];
  EXPECT_EQ(4, direct.id);  // 4, not 3 -- nvls takes 3
  EXPECT_EQ(NCCL_TOPO_PATTERN_COLLNET_DIRECT, direct.pattern);
  EXPECT_EQ(1, direct.collNet);
  EXPECT_EQ(1, direct.minChannels);
  EXPECT_EQ(MAXCHANNELS, direct.maxChannels);
}

TEST_F(InitMicrotest, InitTransportsRank_NvlsGraphSeededWithFullChannelRange) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  const ncclTopoGraph& nvls = c.get()->graphs[NCCL_ALGO_NVLS];
  EXPECT_EQ(3, nvls.id);
  EXPECT_EQ(NCCL_TOPO_PATTERN_NVLS, nvls.pattern);
  EXPECT_EQ(1, nvls.minChannels);
  EXPECT_EQ(MAXCHANNELS, nvls.maxChannels);
}

// The compute ORDER is the oracle these share: the seam records every graph pointer it was handed,
// so a swapped or dropped compute is visible in a way a per-graph field check is not.
TEST_F(InitMicrotest, InitTransportsRank_CollNetAndNvlsOff_ComputesOnlyRingAndTree) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  ncclComm* comm = c.get();
  EXPECT_EQ(std::vector<ncclTopoGraph*>({&comm->graphs[NCCL_ALGO_RING], &comm->graphs[NCCL_ALGO_TREE]}),
            g_ncclTopoComputeGraphs);
}

TEST_F(InitMicrotest, InitTransportsRank_CollNetEnabled_ComputesChainThenDirect) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  ncclCollNet_t collNet{};
  c.get()->ncclCollNet = &collNet;  // keeps collnetEnable alive past :1614
  c.get()->config.collnetEnable = 1;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  ncclComm* comm = c.get();
  EXPECT_EQ(std::vector<ncclTopoGraph*>({&comm->graphs[NCCL_ALGO_RING], &comm->graphs[NCCL_ALGO_TREE],
                                         &comm->graphs[NCCL_ALGO_COLLNET_CHAIN],
                                         &comm->graphs[NCCL_ALGO_COLLNET_DIRECT]}),
            g_ncclTopoComputeGraphs);  // chain before direct, the reverse of the :1763 dump order
  // The collnet-enabled arm is the only place :1691/:1693 run, and the suite's other pairing oracle
  // has collnet disabled -- so without this a print handed the wrong graph here is unobservable.
  EXPECT_EQ(g_ncclTopoComputeGraphs, g_ncclTopoPrintGraphGraphs);
}

TEST_F(InitMicrotest, InitTransportsRank_NvlsSupported_ComputesNvlsGraphLast) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  c.get()->nvlsSupport = 1;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  ncclComm* comm = c.get();
  EXPECT_EQ(std::vector<ncclTopoGraph*>({&comm->graphs[NCCL_ALGO_RING], &comm->graphs[NCCL_ALGO_TREE],
                                         &comm->graphs[NCCL_ALGO_NVLS]}),
            g_ncclTopoComputeGraphs);
}

// Each compute is immediately followed by a print of the SAME graph; recording both is what makes a
// mismatched pair -- printing the tree graph after computing the ring one -- visible.
TEST_F(InitMicrotest, InitTransportsRank_EachComputedGraphIsPrintedInTheSameOrder) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  c.get()->nvlsSupport = 1;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(g_ncclTopoComputeGraphs, g_ncclTopoPrintGraphGraphs);
  EXPECT_EQ(3u, g_ncclTopoPrintGraphGraphs.size());  // and not vacuously empty
}

TEST_F(InitMicrotest, InitTransportsRank_TreeGraphComputeFails_PropagatesAfterRingSucceeded) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclTopoCompute = [](ncclTopoSystem*, ncclTopoGraph*) {
    return g_ncclTopoComputeCalls == 2 ? ncclInvalidUsage : ncclSuccess;  // fail the :1673 tree compute
  };
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclTopoComputeCalls);
  EXPECT_EQ(1u, g_ncclTopoPrintGraphGraphs.size());  // only the ring print ran
}

TEST_F(InitMicrotest, InitTransportsRank_PrintGraphFails_PropagatesBeforeTheTreeCompute) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  g_ncclTopoPrintGraphResult = ncclInvalidArgument;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidArgument, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoComputeCalls);  // :1649 stopped it before :1673
}

// gfx1151 overrides whatever ncclTopoCompute chose, clamped into the ring graph's own channel range.
TEST_F(InitMicrotest, InitTransportsRank_Gfx1151_OverridesRingChannelsWithParam) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx1151");
  SetParams({{"RCCL_INIT_CHANNELS", 3}});  // RCCL_PARAM prefixes; the bare name would silently miss
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(3, c.get()->graphs[NCCL_ALGO_RING].nChannels);  // 3, not the 5 the compute stamped
  // The override has to land BEFORE :1671-1672 size the tree from ringGraph->nChannels; without this
  // the whole block could move below the tree setup and nothing would notice.
  EXPECT_EQ(3, c.get()->graphs[NCCL_ALGO_TREE].minChannels);
}

TEST_F(InitMicrotest, InitTransportsRank_Gfx1151_UnsetParamFallsBackToSixChannels) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx1151");
  InstallTopoComputeSuccess(/*nChannels=*/5);  // INIT_CHANNELS defaults to -1, so the literal 6 wins
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(6, c.get()->graphs[NCCL_ALGO_RING].nChannels);
}

// Zero is the boundary the `> 0` test exists for: it means "not set", so the literal 6 still wins.
// A `>= 0` reading would take the 0 through the clamp at :1664 and land on minChannels, i.e. 1.
TEST_F(InitMicrotest, InitTransportsRank_Gfx1151_ZeroInitChannelsIsTreatedAsUnset) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  ncclTopoSystem* topo = c.installTopo();
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx1151");
  SetParams({{"RCCL_INIT_CHANNELS", 0}});
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(6, c.get()->graphs[NCCL_ALGO_RING].nChannels);
}

TEST_F(InitMicrotest, InitTransportsRank_NonGfx1151_KeepsTheComputedRingChannelCount) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();  // gcn stays empty, so IsArchMatch is false
  SetParams({{"RCCL_INIT_CHANNELS", 3}});  // RCCL_PARAM prefixes; the bare name would silently miss
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(5, c.get()->graphs[NCCL_ALGO_RING].nChannels);  // param ignored off gfx1151
}

// :1763 builds its own array, ordered direct BEFORE chain -- the reverse of the compute order above.
TEST_F(InitMicrotest, InitTransportsRank_DumpFileRankMatches_DumpsFiveGraphsDirectBeforeChain) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);  // GRAPH_DUMP_FILE_RANK defaults to 0
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoDumpGraphsCalls);
  EXPECT_EQ(5, g_ncclTopoDumpGraphsNgraphs);  // the hardcoded count at :1764, not the clamped vector
  ncclComm* comm = c.get();
  EXPECT_EQ(std::vector<ncclTopoGraph*>({&comm->graphs[NCCL_ALGO_RING], &comm->graphs[NCCL_ALGO_TREE],
                                         &comm->graphs[NCCL_ALGO_COLLNET_DIRECT],
                                         &comm->graphs[NCCL_ALGO_COLLNET_CHAIN],
                                         &comm->graphs[NCCL_ALGO_NVLS]}),
            g_ncclTopoDumpGraphsArray);
}

TEST_F(InitMicrotest, InitTransportsRank_DumpFileRankDiffers_SkipsTheDump) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/1);  // rank 1 != GRAPH_DUMP_FILE_RANK 0
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(0, g_ncclTopoDumpGraphsCalls);
  EXPECT_EQ(-1, g_ncclTopoDumpGraphsNgraphs);  // nor was it handed a count
  // The array recorder is .assign()-replaced rather than appended, so this only holds if TearDown
  // genuinely clears it -- which is what makes that reset line load-bearing rather than decorative.
  EXPECT_TRUE(g_ncclTopoDumpGraphsArray.empty());
}

TEST_F(InitMicrotest, InitTransportsRank_DumpGraphsFails_Propagates) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  g_ncclTopoDumpGraphsResult = ncclSystemError;
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclSystemError, initTransportsRank(c.get(), nullptr, c.timers()));
}

// The rung-3 terminator's own error path. Every other rung-3 test rides its ncclTimeout default, so
// without this nothing writes that knob -- and :1774 propagating whatever the seam returns, rather
// than a fixed code, would go unnoticed. Also what makes the knob's TearDown reset load-bearing.
TEST_F(InitMicrotest, InitTransportsRank_P2pChannelsPerPeerFails_PropagatesThatCode) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  InstallTopoComputeSuccess(/*nChannels=*/5);
  g_ncclTopoComputeP2pChannelsPerPeerResult = ncclInvalidUsage;  // not the ncclTimeout sentinel
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclInvalidUsage, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(1, g_ncclTopoDumpGraphsCalls);  // and it really did get as far as :1774
}

TEST_F(InitMicrotest, InitTransportsRank_MaxP2pPeersAboveRankCount_IsCappedToNRanks) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  c.get()->config.maxP2pPeers = 64;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(4, c.get()->config.maxP2pPeers);
}

TEST_F(InitMicrotest, InitTransportsRank_MaxP2pPeersBelowRankCount_IsLeftAlone) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  c.get()->config.maxP2pPeers = 2;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, c.get()->config.maxP2pPeers);  // positive anchor for the cap above
}

TEST_F(InitMicrotest, InitTransportsRank_P2pChannelsPerPeerFails_PropagatesAndRunsCleanup) {
  TransportsRankComm c(/*nRanks=*/4, /*rank=*/0);
  c.installTopo();
  g_ncclOsCpuCountValue = 1;
  InstallTopoComputeSuccess(/*nChannels=*/5);
  InstallPeerInfoAllGather(c, std::vector<PeerSpec>(4));
  EXPECT_EQ(ncclTimeout, initTransportsRank(c.get(), nullptr, c.timers()));
  EXPECT_EQ(2, g_ncclOsCpuCountCalls);  // :1608 and exit::2403 -- fail: fell through to exit:
}

// setCommAbortFlags writes four INDEPENDENT uint32_t* on comm; distinct non-zero sentinels make a
// dropped or mis-targeted store visible instead of coinciding with the value being stored.
namespace {
class AbortFlagsComm {
 public:
  static constexpr uint32_t kParentSentinel = 0xA1A1A1A1u;
  static constexpr uint32_t kParentDevSentinel = 0xB2B2B2B2u;
  static constexpr uint32_t kChildSentinel = 0xC3C3C3C3u;
  static constexpr uint32_t kChildDevSentinel = 0xD4D4D4D4u;

  AbortFlagsComm() : comm_(new ncclComm{}) {
    comm_->abortFlag = &slots_[0];
    comm_->abortFlagDev = &slots_[1];
    comm_->childAbortFlag = &slots_[2];
    comm_->childAbortFlagDev = &slots_[3];
  }
  ncclComm* get() { return comm_.get(); }
  void detachChild() { comm_->childAbortFlag = nullptr; }
  uint32_t parent() const { return slots_[0]; }
  uint32_t parentDev() const { return slots_[1]; }
  uint32_t child() const { return slots_[2]; }
  uint32_t childDev() const { return slots_[3]; }

 private:
  uint32_t slots_[4] = {kParentSentinel, kParentDevSentinel, kChildSentinel, kChildDevSentinel};
  std::unique_ptr<ncclComm> comm_;
};
}  // namespace

// Arm: childAbortFlag != nullptr TRUE -- all four flags written. The distinctness check is load-bearing:
// without it four equal reads could come from one slot being aliased four times.
TEST_F(InitMicrotest, SetCommAbortFlags_ChildAttached_WritesAllFourFlags) {
  AbortFlagsComm c;
  ASSERT_NE(c.get()->abortFlag, c.get()->abortFlagDev);
  ASSERT_NE(c.get()->abortFlag, c.get()->childAbortFlag);
  ASSERT_NE(c.get()->abortFlag, c.get()->childAbortFlagDev);
  ASSERT_NE(c.get()->abortFlagDev, c.get()->childAbortFlag);
  ASSERT_NE(c.get()->abortFlagDev, c.get()->childAbortFlagDev);
  ASSERT_NE(c.get()->childAbortFlag, c.get()->childAbortFlagDev);

  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 1));
  EXPECT_EQ(1u, c.parent());
  EXPECT_EQ(1u, c.parentDev());
  EXPECT_EQ(1u, c.child());
  EXPECT_EQ(1u, c.childDev());
}

// Arm: childAbortFlag == nullptr FALSE -- the child pair must stay untouched while the parent pair is
// still written; the parent assertions are the positive anchor for the two "unchanged" checks.
TEST_F(InitMicrotest, SetCommAbortFlags_ChildDetached_WritesOnlyParentFlags) {
  AbortFlagsComm c;
  c.detachChild();
  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 1));
  EXPECT_EQ(1u, c.parent());
  EXPECT_EQ(1u, c.parentDev());
  EXPECT_EQ(AbortFlagsComm::kChildSentinel, c.child());
  EXPECT_EQ(AbortFlagsComm::kChildDevSentinel, c.childDev());
}

// value 0 is the un-abort path (init.cc:3921, :4152): the stores must be unconditional, not "set only".
TEST_F(InitMicrotest, SetCommAbortFlags_ZeroAfterOne_ClearsAllFourFlags) {
  AbortFlagsComm c;
  ASSERT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 1));
  ASSERT_EQ(1u, c.parent());
  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 0));
  EXPECT_EQ(0u, c.parent());
  EXPECT_EQ(0u, c.parentDev());
  EXPECT_EQ(0u, c.child());
  EXPECT_EQ(0u, c.childDev());
}

// The int -> uint32_t cast is a value-preserving bit reinterpretation, not a truthiness collapse.
TEST_F(InitMicrotest, SetCommAbortFlags_NegativeValue_StoredAsTwosComplementBits) {
  AbortFlagsComm c;
  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), -1));
  EXPECT_EQ(0xFFFFFFFFu, c.parent());
  EXPECT_EQ(0xFFFFFFFFu, c.parentDev());
  EXPECT_EQ(0xFFFFFFFFu, c.child());
  EXPECT_EQ(0xFFFFFFFFu, c.childDev());

  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), INT32_MIN));
  EXPECT_EQ(0x80000000u, c.parent());
  EXPECT_EQ(0x80000000u, c.parentDev());
  EXPECT_EQ(0x80000000u, c.child());
  EXPECT_EQ(0x80000000u, c.childDev());
}

// A large positive value keeps all 31 payload bits: pins that no truncation or masking is applied.
TEST_F(InitMicrotest, SetCommAbortFlags_LargePositiveValue_StoredVerbatim) {
  AbortFlagsComm c;
  EXPECT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 0x7F00FF01));
  EXPECT_EQ(0x7F00FF01u, c.parent());
  EXPECT_EQ(0x7F00FF01u, c.parentDev());
  EXPECT_EQ(0x7F00FF01u, c.child());
  EXPECT_EQ(0x7F00FF01u, c.childDev());
}

// LATENT COUPLING (init.cc:3889): childAbortFlagDev is stored under childAbortFlag's null check, so a
// non-null childAbortFlag paired with a null childAbortFlagDev writes through NULL. Differential with
// SetCommAbortFlags_ChildAttached_WritesAllFourFlags: identical state but that pointer, which survives.
TEST_F(InitMicrotest, SetCommAbortFlags_NullChildDevUnderNonNullChildFlag_DiesOnNullDeref) {
  EXPECT_EXIT(
      {
        AbortFlagsComm c;
        c.get()->childAbortFlagDev = nullptr;
        fprintf(stderr, "setCommAbortFlags-reached-with-null-childAbortFlagDev\n");
        fflush(stderr);
        (void)setCommAbortFlags(c.get(), 1);
        fprintf(stderr, "setCommAbortFlags-returned-without-dereferencing-null\n");
        _exit(0);  // a non-crashing build exits 0, matching neither DEATH_BY_SEGV alternative
      },
      DEATH_BY_SEGV, "setCommAbortFlags-reached-with-null-childAbortFlagDev");
}
// ---------------------------------------------------------------------------
// ncclGetUniqueId_impl (init.cc:358) -- the standalone public-API id minter.
// ---------------------------------------------------------------------------
namespace {
// 0xAB, not 0x00: the memset's job is erasing pre-existing bytes, which a zero-filled buffer cannot show.
constexpr char kIdPoison = static_cast<char>(0xAB);

// Distinctive, byte-wise distinct from kIdPoison and from 0, so a short or long memcpy is visible in every byte.
ncclBootstrapHandle DistinctiveHandle() {
  ncclBootstrapHandle h{};
  h.magic = 0x0123456789ABCDEFULL;
  h.addr.sa.sa_family = AF_INET;
  std::memset(h.addr.sa.sa_data, 0x5C, sizeof(h.addr.sa.sa_data));
  h.nRanks = 0x11223344;
  return h;
}
}  // namespace

// Whole-buffer oracle: the first sizeof(handle) bytes must be the handle verbatim, every later byte must be zero.
TEST_F(InitMicrotest, GetUniqueId_HappyPath_CopiesHandleAndZeroesTheTail) {
  g_bootstrapHandleTemplate = DistinctiveHandle();
  g_bootstrapHandleMagic = g_bootstrapHandleTemplate.magic;

  ncclUniqueId id;
  std::memset(&id, kIdPoison, sizeof(id));
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId_impl(&id));

  EXPECT_EQ(1, g_bootstrapGetUniqueIdCalls);
  const ncclBootstrapHandle expect = DistinctiveHandle();
  EXPECT_EQ(0, std::memcmp(id.internal, &expect, sizeof(expect))) << "handle bytes were not copied verbatim";
  for (size_t i = sizeof(ncclBootstrapHandle); i < sizeof(ncclUniqueId); ++i) {
    ASSERT_EQ('\0', id.internal[i]) << "byte " << i << " past the handle was not zeroed";
  }
}

// The memset covers the WHOLE id, so a byte the handle does write must still come from the handle, not the poison.
TEST_F(InitMicrotest, GetUniqueId_HappyPath_ZeroValuedHandleFieldsOverwriteThePoison) {
  ncclBootstrapHandle h = DistinctiveHandle();
  h.nRanks = 0;  // a zero field inside the copied prefix: only memset-then-memcpy can make this read back as 0
  g_bootstrapHandleTemplate = h;
  g_bootstrapHandleMagic = h.magic;

  ncclUniqueId id;
  std::memset(&id, kIdPoison, sizeof(id));
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId_impl(&id));

  int nRanksOut = -1;
  std::memcpy(&nRanksOut, id.internal + offsetof(ncclBootstrapHandle, nRanks), sizeof(nRanksOut));
  EXPECT_EQ(0, nRanksOut);
  uint64_t magicOut = 0;
  std::memcpy(&magicOut, id.internal, sizeof(magicOut));
  EXPECT_EQ(h.magic, magicOut);  // positive anchor: the copy really ran
}

// The Recorder call is a line of the unit: it must see rrGetUniqueId, the sentinel ranks, and the caller's buffer.
TEST_F(InitMicrotest, GetUniqueId_HappyPath_RecordsGetUniqueIdWithSentinelRanks) {
  g_bootstrapHandleTemplate = DistinctiveHandle();
  ncclUniqueId id{};
  ASSERT_EQ(ncclSuccess, ncclGetUniqueId_impl(&id));
  EXPECT_EQ(1, g_recorderIdCalls);
  EXPECT_EQ(static_cast<int>(rccl::rrGetUniqueId), g_recorderLastIdCall);
  EXPECT_EQ(&id, g_recorderLastId);
  EXPECT_EQ(-1, g_recorderLastRank);
  EXPECT_EQ(-1, g_recorderLastNranks);
}

// PtrCheck's null arm. Full message, not "argument is NULL": every PtrCheck site shares that suffix.
TEST_F(InitMicrotest, GetUniqueId_NullOut_ReturnsInvalidArgumentBeforeMintingAHandle) {
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { res = ncclGetUniqueId_impl(nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "GetUniqueId : out argument is NULL")) << log;
  EXPECT_EQ(0, g_bootstrapGetUniqueIdCalls) << "must not mint an id it has nowhere to put";
  EXPECT_EQ(0, g_recorderIdCalls);
}

// bootstrapGetUniqueId's failure arm: the error propagates and the caller's buffer is left completely alone.
TEST_F(InitMicrotest, GetUniqueId_BootstrapFails_PropagatesAndLeavesOutUntouched) {
  g_bootstrapGetUniqueIdResult = ncclSystemError;
  ncclUniqueId id;
  std::memset(&id, kIdPoison, sizeof(id));
  EXPECT_EQ(ncclSystemError, ncclGetUniqueId_impl(&id));
  EXPECT_EQ(1, g_bootstrapGetUniqueIdCalls);
  for (size_t i = 0; i < sizeof(ncclUniqueId); ++i) {
    ASSERT_EQ(kIdPoison, id.internal[i]) << "byte " << i << " was written on a failed mint";
  }
  EXPECT_EQ(0, g_recorderIdCalls);
}

// ncclInit()'s per-call NCCLCHECK (init.cc:292) runs before its call_once, so this arm is reachable in-process.
TEST_F(InitMicrotest, GetUniqueId_NcclInitFails_PropagatesBeforePtrCheck) {
  g_ncclOsTopoGetStrFromSysResult = ncclRemoteError;
  ncclUniqueId id;
  std::memset(&id, kIdPoison, sizeof(id));
  EXPECT_EQ(ncclRemoteError, ncclGetUniqueId_impl(&id));
  EXPECT_EQ(1, g_ncclOsTopoGetStrFromSysCalls);
  EXPECT_EQ(0, g_bootstrapGetUniqueIdCalls);
  EXPECT_EQ(kIdPoison, id.internal[0]);
}

// Same failure with a null out: proves ncclInit() is checked BEFORE PtrCheck, which would otherwise report first.
TEST_F(InitMicrotest, GetUniqueId_NcclInitFails_OutrunsTheNullOutCheck) {
  g_ncclOsTopoGetStrFromSysResult = ncclRemoteError;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { res = ncclGetUniqueId_impl(nullptr); });
  EXPECT_EQ(ncclRemoteError, res);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "GetUniqueId : out argument is NULL")) << log;
  EXPECT_EQ(1, g_ncclOsTopoGetStrFromSysCalls);  // positive anchor: an empty log cannot pass the check above
}

// envInitOnceFlag latches for the process, so the failure arm is only reachable in a child that has never called it.
TEST_F(InitMicrotestIsolated, GetUniqueId_NcclInitEnvFails_PropagatesBeforeNcclInit) {
  RUN_ISOLATED_TEST(
      "Init_GetUniqueId_NcclInitEnvFails",
      []() {
        g_ncclEnvPluginInitResult = ncclInvalidUsage;
        ncclUniqueId id;
        std::memset(&id, kIdPoison, sizeof(id));
        ASSERT_EQ(ncclInvalidUsage, ncclGetUniqueId_impl(&id));
        ASSERT_EQ(0, g_ncclOsTopoGetStrFromSysCalls);  // ncclInit() never ran
        ASSERT_EQ(0, g_bootstrapGetUniqueIdCalls);
        ASSERT_EQ(kIdPoison, id.internal[0]);
      });
}

// LATENT BUG (init.cc:369): this is the only record(rcclCall_t,int,int,ncclUniqueId*,...) site that drops the
// result; :3467/:3483/:3709 all NCCLCHECK it. A recorder failure here is swallowed and the API reports success.
TEST_F(InitMicrotest, GetUniqueId_RecorderFails_StillReturnsSuccess) {
  g_bootstrapHandleTemplate = DistinctiveHandle();
  g_recorderResult = ncclInternalError;
  ncclUniqueId id{};
  EXPECT_EQ(ncclSuccess, ncclGetUniqueId_impl(&id));
  EXPECT_EQ(1, g_recorderIdCalls);  // positive anchor: the ignored call really happened
}

// Destructor / push-free family, and the small residual paths around it.

namespace {
// A heap ncclComm with a live memPermanent stack, which is what ncclCommPush*() allocates its nodes from.
class Dtor_PushComm {
 public:
  Dtor_PushComm() : comm_(new ncclComm{}) { ncclMemoryStackConstruct(&comm_->memPermanent); }
  ~Dtor_PushComm() { ncclMemoryStackDestruct(&comm_->memPermanent); }
  Dtor_PushComm(const Dtor_PushComm&) = delete;
  Dtor_PushComm& operator=(const Dtor_PushComm&) = delete;
  ncclComm* get() { return comm_.get(); }

 private:
  std::unique_ptr<ncclComm> comm_;
};

// ncclCudaFree() returns early on ncclMemEntryAlreadyReleased(manager, ptr), so this observes both args.
class Dtor_ReleasedManager {
 public:
  explicit Dtor_ReleasedManager(void* ptr) : mgr_(new ncclMemManager{}) {
    entry_.ptr = ptr;
    entry_.state = ncclDynMemStateReleased;
    entry_.next = nullptr;
    mgr_->entries = &entry_;
    mgr_->numEntries = 1;
    mgr_->released = 1;
  }
  ncclMemManager* get() { return mgr_.get(); }

 private:
  ncclDynMemEntry entry_{};
  std::unique_ptr<ncclMemManager> mgr_;
};

class Dtor_GdrCopyGuard {
 public:
  Dtor_GdrCopyGuard() : saved_(ncclGdrCopy) {}
  ~Dtor_GdrCopyGuard() { ncclGdrCopy = saved_; }
  Dtor_GdrCopyGuard(const Dtor_GdrCopyGuard&) = delete;
  Dtor_GdrCopyGuard& operator=(const Dtor_GdrCopyGuard&) = delete;

 private:
  gdr_t saved_;
};

class Dtor_LastErrorGuard {
 public:
  Dtor_LastErrorGuard() : saved_(ncclLastError) {}
  ~Dtor_LastErrorGuard() { std::snprintf(ncclLastError, 1024, "%s", saved_.c_str()); }
  Dtor_LastErrorGuard(const Dtor_LastErrorGuard&) = delete;
  Dtor_LastErrorGuard& operator=(const Dtor_LastErrorGuard&) = delete;

 private:
  std::string saved_;
};

const gdr_t Dtor_kGdrPoison = (gdr_t)0xBADF00DUL;
const gdr_t Dtor_kGdrSupportedHandle = (gdr_t)0x12345678L;
constexpr uintptr_t Dtor_kNotABaseAddress = 0x1000;

// The two push functions differ only in the destructor they stamp, so one body covers both.
void Dtor_ExpectPushChainsNewestFirst(void (*push)(struct ncclComm*, void*),
                                      ncclResult_t (*fn)(struct ncclDestructor*)) {
  Dtor_PushComm c;
  int first = 0;
  double second = 0;
  push(c.get(), &first);
  push(c.get(), &second);

  struct ncclDestructor* head = c.get()->destructorHead;
  ASSERT_NE(nullptr, head);
  EXPECT_EQ(&second, head->obj);
  EXPECT_EQ(c.get(), head->comm);
  EXPECT_EQ(fn, head->fn);
  ASSERT_NE(nullptr, head->next);
  EXPECT_NE(head, head->next);
  EXPECT_EQ(&first, head->next->obj);
  EXPECT_EQ(c.get(), head->next->comm);
  EXPECT_EQ(fn, head->next->fn);
  EXPECT_EQ(nullptr, head->next->next);
}
}  // namespace

TEST_F(InitMicrotest, CommPushFree_TwoPushes_ChainsNewestFirstWithFreeDestructor) {
  Dtor_ExpectPushChainsNewestFirst(ncclCommPushFree, &ncclDestructorFnFree);
}

TEST_F(InitMicrotest, CommPushCudaGdrFree_TwoPushes_ChainsNewestFirstWithGdrDestructor) {
  Dtor_ExpectPushChainsNewestFirst(ncclCommPushCudaGdrFree, &ncclDestructorFnCudaGdrFree);
}

TEST_F(InitMicrotest, CommPushFreeThenGdrFree_ShareOneList_KeepingEachEntrysOwnDestructor) {
  Dtor_PushComm c;
  int plain = 0;
  int gdr = 0;
  ncclCommPushFree(c.get(), &plain);
  ncclCommPushCudaGdrFree(c.get(), &gdr);

  struct ncclDestructor* head = c.get()->destructorHead;
  ASSERT_NE(nullptr, head);
  ASSERT_NE(nullptr, head->next);
  EXPECT_EQ(&ncclDestructorFnCudaGdrFree, head->fn);
  EXPECT_EQ(&gdr, head->obj);
  EXPECT_EQ(&ncclDestructorFnFree, head->next->fn);
  EXPECT_EQ(&plain, head->next->obj);
}

TEST_F(InitMicrotest, DestructorFnFree_FreesTheObjectPointer_NotTheComm) {
  Dtor_PushComm c;
  void* obj = std::malloc(64);
  ASSERT_NE(nullptr, obj);
  ncclCommPushFree(c.get(), obj);

  void* freed = nullptr;
  ScopedHook microFree(g_microFree, [&](void* p) { freed = p; });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnFree(c.get()->destructorHead));
  EXPECT_EQ(1, microFree.calls);
  EXPECT_EQ(obj, freed);
  std::free(obj);
}

TEST_F(InitMicrotest, DestructorFnCudaFree_ObjAlreadyReleasedByManager_SkipsTheDeviceFree) {
  Dtor_PushComm c;
  int obj = 0;
  Dtor_ReleasedManager mgr(&obj);
  c.get()->memManager = mgr.get();
  ncclCommPushCudaFree(c.get(), &obj);

  ScopedHook hipFree(g_hipFree, [](void*) { return hipSuccess; });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnCudaFree(c.get()->destructorHead));
  EXPECT_EQ(0, hipFree.calls);
}

TEST_F(InitMicrotest, DestructorFnCudaFree_ObjUntrackedByManager_FreesThatExactPointer) {
  Dtor_PushComm c;
  int other = 0;
  Dtor_ReleasedManager mgr(&other);
  c.get()->memManager = mgr.get();
  int obj = 0;
  ncclCommPushCudaFree(c.get(), &obj);

  g_hipAsyncOpsResult = hipSuccess;  // hipThreadExchangeStreamCaptureMode, on ncclCudaFree's real-free path
  ScopedHook memRange(g_hipMemGetAddressRange,
                      [](hipDeviceptr_t* base, std::size_t* size, hipDeviceptr_t) {
                        if (base) *base = reinterpret_cast<hipDeviceptr_t>(Dtor_kNotABaseAddress);
                        if (size) *size = 0;
                        return hipSuccess;
                      });
  void* freed = nullptr;
  ScopedHook hipFree(g_hipFree, [&](void* p) {
    freed = p;
    return hipSuccess;
  });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnCudaFree(c.get()->destructorHead));
  EXPECT_EQ(1, hipFree.calls);
  EXPECT_EQ(&obj, freed);
}

TEST_F(InitMicrotest, DestructorFnCudaHostFree_PassesTheObjectPointerToTheHostFree) {
  Dtor_PushComm c;
  int obj = 0;
  ncclCommPushCudaHostFree(c.get(), &obj);

  void* freed = nullptr;
  ScopedHook hostFree(g_hipHostFree, [&](void* p) {
    freed = p;
    return hipSuccess;
  });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnCudaHostFree(c.get()->destructorHead));
  EXPECT_EQ(1, hostFree.calls);
  EXPECT_EQ(&obj, freed);
}

TEST_F(InitMicrotest, DestructorFnCudaGdrFree_DevMemAlreadyReleased_SkipsTheDeviceFreeButFreesTheDescriptor) {
  Dtor_PushComm c;
  int devMem = 0;
  gdr_mem_desc_t* md = static_cast<gdr_mem_desc_t*>(std::malloc(sizeof(gdr_mem_desc_t)));
  ASSERT_NE(nullptr, md);
  *md = gdr_mem_desc_t{};
  md->gdrDevMem = &devMem;
  Dtor_ReleasedManager mgr(&devMem);
  c.get()->memManager = mgr.get();
  ncclCommPushCudaGdrFree(c.get(), md);

  ScopedHook hipFree(g_hipFree, [](void*) { return hipSuccess; });
  void* freed = nullptr;
  ScopedHook microFree(g_microFree, [&](void* p) { freed = p; });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnCudaGdrFree(c.get()->destructorHead));
  EXPECT_EQ(0, hipFree.calls);
  EXPECT_EQ(1, microFree.calls);
  EXPECT_EQ(md, freed);
  std::free(md);
}

TEST_F(InitMicrotest, DestructorFnCudaGdrFree_DevMemUntracked_FreesDevMemThroughTheCommsManager) {
  Dtor_PushComm c;
  int devMem = 0;
  int other = 0;
  gdr_mem_desc_t* md = static_cast<gdr_mem_desc_t*>(std::malloc(sizeof(gdr_mem_desc_t)));
  ASSERT_NE(nullptr, md);
  *md = gdr_mem_desc_t{};
  md->gdrDevMem = &devMem;
  Dtor_ReleasedManager mgr(&other);
  c.get()->memManager = mgr.get();
  ncclCommPushCudaGdrFree(c.get(), md);

  g_hipAsyncOpsResult = hipSuccess;  // hipThreadExchangeStreamCaptureMode, on ncclCudaFree's real-free path
  ScopedHook memRange(g_hipMemGetAddressRange,
                      [](hipDeviceptr_t* base, std::size_t* size, hipDeviceptr_t) {
                        if (base) *base = reinterpret_cast<hipDeviceptr_t>(Dtor_kNotABaseAddress);
                        if (size) *size = 0;
                        return hipSuccess;
                      });
  void* devFreed = nullptr;
  ScopedHook hipFree(g_hipFree, [&](void* p) {
    devFreed = p;
    return hipSuccess;
  });
  void* freed = nullptr;
  ScopedHook microFree(g_microFree, [&](void* p) { freed = p; });
  EXPECT_EQ(ncclSuccess, ncclDestructorFnCudaGdrFree(c.get()->destructorHead));
  EXPECT_EQ(1, hipFree.calls);
  EXPECT_EQ(&devMem, devFreed);
  EXPECT_EQ(md, freed);
  std::free(md);
}

TEST_F(InitMicrotest, InitGdrCopy_EnabledOnSupportedArch_PublishesTheGdrHandle) {
  Dtor_GdrCopyGuard guard;
  ncclGdrCopy = Dtor_kGdrPoison;
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "GDRCOPY_ENABLE") == 0 ? int64_t(1) : deft;
  };
  g_hipGetDeviceProperties = [](hipDeviceProp_t* prop, int) {
    *prop = hipDeviceProp_t{};
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx942:sramecc+:xnack-");
    return hipSuccess;
  };
  EXPECT_EQ(ncclSuccess, initGdrCopy());
  EXPECT_EQ(Dtor_kGdrSupportedHandle, ncclGdrCopy);
}

TEST_F(InitMicrotest, InitGdrCopy_EnabledOnUnsupportedArch_ClearsTheGdrHandle) {
  Dtor_GdrCopyGuard guard;
  ncclGdrCopy = Dtor_kGdrPoison;
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "GDRCOPY_ENABLE") == 0 ? int64_t(1) : deft;
  };
  g_hipGetDeviceProperties = [](hipDeviceProp_t* prop, int) {
    *prop = hipDeviceProp_t{};
    std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "gfx90a:sramecc+:xnack-");
    return hipSuccess;
  };
  EXPECT_EQ(ncclSuccess, initGdrCopy());
  EXPECT_EQ(nullptr, ncclGdrCopy);
}

TEST_F(InitMicrotest, InitGdrCopy_ParamNotExactlyOne_LeavesTheGdrHandleUntouched) {
  Dtor_GdrCopyGuard guard;
  ncclGdrCopy = Dtor_kGdrPoison;
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "GDRCOPY_ENABLE") == 0 ? int64_t(2) : deft;
  };
  g_hipGetDeviceProperties = [](hipDeviceProp_t*, int) -> hipError_t {
    ADD_FAILURE() << "ncclGdrInit must not run unless GDRCOPY_ENABLE is exactly 1";
    return hipErrorInvalidValue;
  };
  EXPECT_EQ(ncclSuccess, initGdrCopy());
  EXPECT_EQ(Dtor_kGdrPoison, ncclGdrCopy);
}

TEST_F(InitMicrotest, CommEnsureReady_AbortFlagSet_AbortsAndSkipsTheAsyncErrorQuery) {
  ReadyComm rc;
  uint32_t aborting = 1;
  rc.get()->abortFlag = &aborting;
  rc.get()->asyncResult = ncclSystemError;
  auto gj = std::make_unique<ncclGroupJob>();
  rc.get()->groupJob = gj.get();

  struct ncclGroupJob* aborted = nullptr;
  ScopedHook abort(g_ncclGroupJobAbort, [&](struct ncclGroupJob* j) {
    aborted = j;
    return ncclSuccess;
  });
  g_recorderLabels.clear();
  EXPECT_EQ(ncclSuccess, ncclCommEnsureReady(rc.get()));
  EXPECT_EQ(1, abort.calls);
  EXPECT_EQ(gj.get(), aborted);
  EXPECT_EQ(gj.get(), rc.get()->groupJob);
  // asyncResult is only ever read out (init.cc:4497), so it holds on both arms; the label is the discriminator.
  EXPECT_EQ(0, std::count(g_recorderLabels.begin(), g_recorderLabels.end(), "GetAsyncError"));
}

TEST_F(InitMicrotest, CommEnsureReady_NotAborting_ReportsTheCommsAsyncError) {
  ReadyComm rc;
  rc.get()->asyncResult = ncclSystemError;
  ScopedHook abort(g_ncclGroupJobAbort, [](struct ncclGroupJob*) { return ncclSuccess; });
  g_recorderLabels.clear();
  EXPECT_EQ(ncclSystemError, ncclCommEnsureReady(rc.get()));
  EXPECT_EQ(0, abort.calls);
  EXPECT_EQ(1, std::count(g_recorderLabels.begin(), g_recorderLabels.end(), "GetAsyncError"));
}

namespace {
class Dtor_GinComm {
 public:
  explicit Dtor_GinComm(bool needsProxyProgress) : sr_(new ncclSharedResources{}) {
    sr_->ginState.connected = true;
    sr_->ginState.needsProxyProgress = needsProxyProgress;
    rc_.get()->sharedRes = sr_.get();
  }
  ncclComm* get() { return rc_.get(); }
  struct ncclGinState* gin() { return &sr_->ginState; }

 private:
  ReadyComm rc_;
  std::unique_ptr<ncclSharedResources> sr_;
};
}  // namespace

TEST_F(InitMicrotest, GetAsyncError_GinProgressThreadFaulted_ReportsTheGinAsyncResult) {
  Dtor_GinComm c(/*needsProxyProgress=*/true);
  c.gin()->asyncResult = ncclInternalError;
  ncclResult_t e = ncclSuccess;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(c.get(), &e));
  EXPECT_EQ(ncclInternalError, e);
}

TEST_F(InitMicrotest, GetAsyncError_GinWithoutProgressThread_IgnoresTheGinAsyncResult) {
  Dtor_GinComm c(/*needsProxyProgress=*/false);
  c.gin()->asyncResult = ncclInternalError;
  ncclResult_t e = ncclInternalError;
  EXPECT_EQ(ncclSuccess, ncclCommGetAsyncError_impl(c.get(), &e));
  EXPECT_EQ(ncclSuccess, e);
}

TEST_F(InitMicrotest, CommFinalizeAsyncJobFree_ReturnsTheJobToTheHeap) {
  auto* job = new ncclCommFinalizeAsyncJob{};
  DeleteWatch watch(job);
  ncclCommFinalizeAsyncJobFree(job);
  EXPECT_EQ(1, watch.hits()) << "the job was not deleted";
}

TEST_F(InitMicrotest, CommInitJobFree_FreesTheCommIdThenReturnsTheJobToTheHeap) {
  auto* job = new ncclCommInitRankAsyncJob{};
  job->commId = static_cast<ncclUniqueId*>(std::malloc(sizeof(ncclUniqueId)));
  ASSERT_NE(nullptr, job->commId);
  void* commId = job->commId;

  void* freed = nullptr;
  DeleteWatch watch(job);
  int hits = 0;
  {
    ScopedHook microFree(g_microFree, [&](void* p) { freed = p; });
    ncclCommInitJobFree(job);
    hits = watch.hits();
    EXPECT_EQ(1, microFree.calls);
  }
  EXPECT_EQ(commId, freed);
  std::free(commId);
  EXPECT_EQ(1, hits) << "the job was not deleted";
}

TEST_F(InitMicrotest, ChildCommCleanupJob_ExcludeListPresent_FreesExactlyThatList) {
  auto* job = new ncclCommInitRankAsyncJob{};
  job->excludeRanksList = static_cast<int*>(std::malloc(4 * sizeof(int)));
  ASSERT_NE(nullptr, job->excludeRanksList);
  void* list = job->excludeRanksList;

  void* freed = nullptr;
  DeleteWatch watch(job);
  int hits = 0;
  {
    ScopedHook microFree(g_microFree, [&](void* p) { freed = p; });
    childCommCleanupJob(job);
    hits = watch.hits();
    EXPECT_EQ(1, microFree.calls);
  }
  EXPECT_EQ(list, freed);
  std::free(list);
  EXPECT_EQ(1, hits) << "the job was not deleted";
}

TEST_F(InitMicrotest, ChildCommCleanupJob_NoExcludeList_FreesNothing) {
  auto* job = new ncclCommInitRankAsyncJob{};
  job->excludeRanksList = nullptr;
  DeleteWatch watch(job);
  ScopedHook microFree(g_microFree, [](void*) {});
  childCommCleanupJob(job);
  EXPECT_EQ(0, microFree.calls);
  EXPECT_EQ(1, watch.hits()) << "the job was not deleted";
}

TEST_F(InitMicrotest, GetLastError_ReturnsTheGlobalErrorBuffer_AndRecordsTheCall) {
  Dtor_LastErrorGuard guard;
  const char* kMessage = "Dtor microtest last error";
  std::snprintf(ncclLastError, 1024, "%s", kMessage);
  g_recorderLabels.clear();

  const char* got = ncclGetLastError_impl(nullptr);
  EXPECT_EQ(static_cast<const char*>(ncclLastError), got);
  EXPECT_STREQ(kMessage, got);
  ASSERT_EQ(1u, g_recorderLabels.size());
  EXPECT_EQ("GetLastEror", g_recorderLabels[0]);
}

TEST_F(InitMicrotest, GetLastError_CommArgumentIsIgnored_SameBufferForAnyComm) {
  Dtor_LastErrorGuard guard;
  ReadyComm rc;
  std::snprintf(ncclLastError, 1024, "%s", "first message");
  const char* viaNull = ncclGetLastError_impl(nullptr);
  EXPECT_STREQ("first message", viaNull);
  // Comparing the two return values would be p == p; only a write between the calls shows the comm selects nothing.
  std::snprintf(ncclLastError, 1024, "%s", "second message");
  EXPECT_STREQ("second message", viaNull);
  EXPECT_STREQ("second message", ncclGetLastError_impl(rc.get()));
}

TEST_F(InitMicrotest, EnvInitOnceFunc_StoresThePluginResultInTheLatchedResult) {
  const ncclResult_t saved = envInitResult;
  {
    ScopedHook plugin(g_ncclEnvPluginInit, [] { return ncclSystemError; });
    envInitOnceFunc();
    EXPECT_EQ(1, plugin.calls);
  }
  EXPECT_EQ(ncclSystemError, envInitResult);
  {
    ScopedHook plugin(g_ncclEnvPluginInit, [] { return ncclSuccess; });
    envInitOnceFunc();
    EXPECT_EQ(1, plugin.calls);
  }
  EXPECT_EQ(ncclSuccess, envInitResult);
  envInitResult = saved;
}

namespace {
constexpr const char Dtor_kKernelVersionLine[] = "Linux version 6.8.0-microtest";
constexpr const char Dtor_kNumaBalancingWarning[] = "NUMA auto balancing enabled";
constexpr const char Dtor_kIommuWarning[] = "Missing \"iommu=pt\" from kernel command line";

// ncclInit() reads numa_balancing, /proc/version and bios_version through this one entry point.
ncclResult_t Dtor_SysFileText(const char* file, const char* numaBalancing, char* out, int maxLen) {
  if (!out || maxLen <= 0) return ncclSuccess;
  if (file && std::strcmp(file, "numa_balancing") == 0) {
    std::snprintf(out, maxLen, "%s", numaBalancing);
  } else {
    std::snprintf(out, maxLen, "%s", Dtor_kKernelVersionLine);
  }
  return ncclSuccess;
}

int Dtor_CpuidWithoutHypervisorBit(unsigned, unsigned* a, unsigned* b, unsigned* c, unsigned* d) {
  if (a) *a = 0;
  if (b) *b = 0;
  if (c) *c = 0;
  if (d) *d = 0;
  return 1;
}
}  // namespace

// Isolated: ncclInit latches initOnceFlag, so only a fresh process sees its result deterministically.
TEST_F(InitMicrotestIsolated, NcclInit_NumaBalancingOnAndIommuNotPassthrough_WarnsAboutBoth) {
  RUN_ISOLATED_TEST(
      "Init_NcclInit_NumaAndIommuWarnings",
      []() {
        ScopedHook sysText(g_microTopoGetStrFromSys,
                           [](const char*, const char* file, char* out, int maxLen) {
                             return Dtor_SysFileText(file, "1", out, maxLen);
                           });
        ScopedHook cpuid(g_microCpuid, Dtor_CpuidWithoutHypervisorBit);
        ScopedHook iommu(g_microIommuPassthroughOk, [](const char*) { return false; });
        std::string log = RcclUnitTesting::CaptureLog([] { ASSERT_EQ(ncclSuccess, ncclInit()); });
        ASSERT_TRUE(LogHas(log, Dtor_kNumaBalancingWarning)) << "actual log:\n" << log;
        ASSERT_TRUE(LogHas(log, Dtor_kIommuWarning)) << "actual log:\n" << log;
        ASSERT_EQ(1, iommu.calls);
      });
}

TEST_F(InitMicrotestIsolated, NcclInit_NumaBalancingOffAndIommuPassthrough_WarnsAboutNeither) {
  RUN_ISOLATED_TEST(
      "Init_NcclInit_NoNumaOrIommuWarnings",
      []() {
        ScopedHook sysText(g_microTopoGetStrFromSys,
                           [](const char*, const char* file, char* out, int maxLen) {
                             return Dtor_SysFileText(file, "0", out, maxLen);
                           });
        ScopedHook cpuid(g_microCpuid, Dtor_CpuidWithoutHypervisorBit);
        ScopedHook iommu(g_microIommuPassthroughOk, [](const char*) { return true; });
        ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
        std::string log = RcclUnitTesting::CaptureLog([] { ASSERT_EQ(ncclSuccess, ncclInit()); });
        ASSERT_TRUE(LogHas(log, "Kernel version: 6.8.0-microtest")) << "actual log:\n" << log;
        ASSERT_FALSE(LogHas(log, Dtor_kNumaBalancingWarning)) << "actual log:\n" << log;
        ASSERT_FALSE(LogHas(log, Dtor_kIommuWarning)) << "actual log:\n" << log;
        // Anchors the iommu leg: the probe sits inside the non-cray block (init.cc:310-336), the INFO above does not.
        ASSERT_EQ(1, iommu.calls);
      });
}

// Isolated: ncclInitEnv latches envInitOnceFlag, which no later test could then observe unlatched.
TEST_F(InitMicrotestIsolated, InitEnv_PluginFails_LatchesThatErrorAndCallsThePluginOnce) {
  RUN_ISOLATED_TEST(
      "Init_InitEnv_PluginFails",
      []() {
        ScopedHook plugin(g_ncclEnvPluginInit, [] { return ncclInternalError; });
        ASSERT_EQ(ncclInternalError, ncclInitEnv());
        ASSERT_EQ(1, plugin.calls);
        ASSERT_EQ(ncclInternalError, ncclInitEnv());
        ASSERT_EQ(1, plugin.calls);
      });
}

// --- parseCommConfig: version negotiation, per-field validation, defaulting (init.cc:3243-3462) ---

namespace {
constexpr char ParseCfg_kNetName[] = "microfake-net";

class ParseCfg_Scene {
 public:
  ParseCfg_Scene() : comm_(new ncclComm{}) {}
  // envConfigOverride re-mallocs config.netName and never frees it, so the scene owns that buffer.
  ~ParseCfg_Scene() { free(const_cast<char*>(comm_->config.netName)); }
  ncclConfig_t& config() { return cfg_; }
  const ncclConfig_t& result_config() const { return comm_->config; }
  ncclResult_t Run() { return parseCommConfig(comm_.get(), &cfg_); }
  std::string RunCapturingWarn(ncclResult_t* result) {
    return RcclUnitTesting::CaptureLog([&] { *result = Run(); });
  }
  std::string RunCapturingInfo(ncclResult_t* result) {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    return RcclUnitTesting::CaptureLog([&] { *result = Run(); });
  }

 private:
  ncclConfig_t cfg_ = NCCL_CONFIG_INITIALIZER;
  std::unique_ptr<ncclComm> comm_;
};

// The return code alone cannot tell which field's check fired, so the diagnostic carries the oracle.
void ParseCfg_ExpectRejected(const std::function<void(ncclConfig_t&)>& tweak, const char* warning) {
  ParseCfg_Scene s;
  tweak(s.config());
  ncclResult_t res = ncclSuccess;
  const std::string log = s.RunCapturingWarn(&res);
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_TRUE(LogHas(log, warning)) << "actual log:\n" << log;
}
}  // namespace

TEST_F(InitMicrotest, ParseCommConfig_BadSplitShare_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.splitShare = 7; },
                          "Invalid config splitShare attribute value 7");
}
TEST_F(InitMicrotest, ParseCommConfig_BadShrinkShare_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.shrinkShare = 9; },
                          "Invalid config shrinkShare attribute value 9");
}
TEST_F(InitMicrotest, ParseCommConfig_NonPositiveNvlsCTAs_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.nvlsCTAs = -3; },
                          "Invalid config nvlsCTAs attribute value -3");
}
TEST_F(InitMicrotest, ParseCommConfig_NChannelsPerNetPeerAboveMax_RejectsAndNamesTheField) {
  const std::string warning =
      "Invalid config nChannelsPerNetPeer attribute value " + std::to_string(MAXCHANNELS + 1);
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.nChannelsPerNetPeer = MAXCHANNELS + 1; },
                          warning.c_str());
}
TEST_F(InitMicrotest, ParseCommConfig_ZeroNChannelsPerNetPeer_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.nChannelsPerNetPeer = 0; },
                          "Invalid config nChannelsPerNetPeer attribute value 0");
}
TEST_F(InitMicrotest, ParseCommConfig_BadNvlinkCentricSched_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.nvlinkCentricSched = 5; },
                          "Invalid config nvlinkCentricSched attribute value 5");
}
TEST_F(InitMicrotest, ParseCommConfig_GraphUsageModeAboveTwo_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.graphUsageMode = 3; },
                          "Invalig config graphUsageMode attribute value 3");
}
TEST_F(InitMicrotest, ParseCommConfig_NegativeNumRmaCtx_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.numRmaCtx = -11; },
                          "Invalid config numRmaCtx attribute value -11");
}
TEST_F(InitMicrotest, ParseCommConfig_BadGraphStreamOrdering_RejectsAndNamesTheField) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.graphStreamOrdering = 13; },
                          "Invalid config graphStreamOrdering attribute value 13");
}

TEST_F(InitMicrotest, ParseCommConfig_ZeroNvlsCTAs_IsRejectedAtTheBoundary) {
  ParseCfg_ExpectRejected([](ncclConfig_t& c) { c.nvlsCTAs = 0; },
                          "Invalid config nvlsCTAs attribute value 0");
}
TEST_F(InitMicrotest, ParseCommConfig_OneNvlsCTA_IsAcceptedAndAssigned) {
  ParseCfg_Scene s;
  s.config().nvlsCTAs = 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().nvlsCTAs);
}
TEST_F(InitMicrotest, ParseCommConfig_ZeroNumRmaCtx_IsAcceptedAndAssigned) {
  ParseCfg_Scene s;
  s.config().numRmaCtx = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().numRmaCtx);
}
TEST_F(InitMicrotest, ParseCommConfig_NChannelsPerNetPeerAtMax_IsAcceptedAndAssigned) {
  ParseCfg_Scene s;
  s.config().nChannelsPerNetPeer = MAXCHANNELS;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(MAXCHANNELS, s.result_config().nChannelsPerNetPeer);
}
TEST_F(InitMicrotest, ParseCommConfig_GraphUsageModeTwo_IsAcceptedAndAssigned) {
  ParseCfg_Scene s;
  s.config().graphUsageMode = 2;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(2, s.result_config().graphUsageMode);
}
TEST_F(InitMicrotest, ParseCommConfig_BinaryFlagsAtTheirBounds_AreAcceptedAndAssignedIndependently) {
  ParseCfg_Scene s;
  s.config().splitShare = 1;
  s.config().shrinkShare = 0;
  s.config().nvlinkCentricSched = 1;
  s.config().graphStreamOrdering = 0;
  s.config().graphUsageMode = 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().splitShare);
  EXPECT_EQ(0, s.result_config().shrinkShare);
  EXPECT_EQ(1, s.result_config().nvlinkCentricSched);
  EXPECT_EQ(0, s.result_config().graphStreamOrdering);
  EXPECT_EQ(1, s.result_config().graphUsageMode);
}
TEST_F(InitMicrotest, ParseCommConfig_BinaryFlagsAtTheirOtherBound_AreAcceptedAndAssignedIndependently) {
  ParseCfg_Scene s;
  s.config().splitShare = 0;
  s.config().shrinkShare = 1;
  s.config().nvlinkCentricSched = 0;
  s.config().graphStreamOrdering = 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().splitShare);
  EXPECT_EQ(1, s.result_config().shrinkShare);
  EXPECT_EQ(0, s.result_config().nvlinkCentricSched);
  EXPECT_EQ(1, s.result_config().graphStreamOrdering);
}

// --- version gates: below the gate a field is reset to the initializer default, at the gate it survives ---

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow214_ResetsBlockingToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 14, 0) - 1;
  s.config().blocking = 2;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().blocking);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt214_KeepsBlocking) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 14, 0);
  s.config().blocking = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().blocking);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow217_ResetsCgaClusterSizeToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0) - 1;
  s.config().cgaClusterSize = -5;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(4, s.result_config().cgaClusterSize);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow217_ResetsMinCTAsToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0) - 1;
  s.config().minCTAs = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().minCTAs);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow217_ResetsMaxCTAsToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0) - 1;
  s.config().maxCTAs = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(MAXCHANNELS, s.result_config().maxCTAs);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow217_DropsNetNameBeforeDefaulting) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0) - 1;
  s.config().netName = ParseCfg_kNetName;
  s.config().blocking = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(nullptr, s.result_config().netName);
  EXPECT_EQ(0, s.result_config().blocking);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt217_KeepsCgaClusterSizeAndCtaBounds) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0);
  s.config().cgaClusterSize = 3;
  s.config().minCTAs = 2;
  s.config().maxCTAs = 5;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(3, s.result_config().cgaClusterSize);
  EXPECT_EQ(2, s.result_config().minCTAs);
  EXPECT_EQ(5, s.result_config().maxCTAs);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt217_KeepsNetName) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 17, 0);
  s.config().netName = ParseCfg_kNetName;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_STREQ(ParseCfg_kNetName, s.result_config().netName);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow225_ResetsTrafficClassToUndefined) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 25, 0) - 1;
  s.config().trafficClass = 42;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, s.result_config().trafficClass);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt225_KeepsTrafficClass) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 25, 0);
  s.config().trafficClass = 42;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(42, s.result_config().trafficClass);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow227_ResetsCollnetEnableToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 27, 0) - 1;
  s.config().collnetEnable = 2;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().collnetEnable);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow227_ResetsCTAPolicyToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 27, 0) - 1;
  s.config().CTAPolicy =
      (NCCL_CTA_POLICY_DEFAULT | NCCL_CTA_POLICY_EFFICIENCY | NCCL_CTA_POLICY_ZERO) + 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, s.result_config().CTAPolicy);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow227_ResetsShrinkShareToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 27, 0) - 1;
  s.config().shrinkShare = 9;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().shrinkShare);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow227_ResetsNvlsCTAsToUndefined) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 27, 0) - 1;
  s.config().nvlsCTAs = -3;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, s.result_config().nvlsCTAs);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt227_KeepsCollnetShrinkShareAndNvlsCTAs) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 27, 0);
  s.config().collnetEnable = 1;
  s.config().CTAPolicy = NCCL_CTA_POLICY_ZERO;
  s.config().shrinkShare = 0;
  s.config().nvlsCTAs = 6;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().collnetEnable);
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, s.result_config().CTAPolicy);
  EXPECT_EQ(0, s.result_config().shrinkShare);
  EXPECT_EQ(6, s.result_config().nvlsCTAs);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow228_ResetsNChannelsPerNetPeerToUndefined) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 28, 0) - 1;
  s.config().nChannelsPerNetPeer = MAXCHANNELS + 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, s.result_config().nChannelsPerNetPeer);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow228_ResetsNvlinkCentricSchedToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 28, 0) - 1;
  s.config().nvlinkCentricSched = 5;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().nvlinkCentricSched);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt228_KeepsNChannelsPerNetPeerAndNvlinkCentricSched) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 28, 0);
  s.config().nChannelsPerNetPeer = 7;
  s.config().nvlinkCentricSched = 1;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(7, s.result_config().nChannelsPerNetPeer);
  EXPECT_EQ(1, s.result_config().nvlinkCentricSched);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow229_ResetsGraphUsageModeToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 29, 0) - 1;
  s.config().graphUsageMode = 3;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().graphUsageMode);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionBelow229_ResetsNumRmaCtxToPlatformDefault) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 29, 0) - 1;
  s.config().numRmaCtx = -11;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().numRmaCtx);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt229_KeepsGraphUsageModeAndNumRmaCtx) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 29, 0);
  s.config().graphUsageMode = 2;
  s.config().numRmaCtx = 3;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(2, s.result_config().graphUsageMode);
  EXPECT_EQ(3, s.result_config().numRmaCtx);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow230_ResetsMaxP2pPeersToUndefined) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 30, 0) - 1;
  s.config().maxP2pPeers = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, s.result_config().maxP2pPeers);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt230_KeepsMaxP2pPeers) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 30, 0);
  s.config().maxP2pPeers = 12;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(12, s.result_config().maxP2pPeers);
}

TEST_F(InitMicrotest, ParseCommConfig_VersionBelow2305_ResetsGraphStreamOrderingToSerialize) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 30, 5) - 1;
  s.config().graphStreamOrdering = 13;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(1, s.result_config().graphStreamOrdering);
}
TEST_F(InitMicrotest, ParseCommConfig_VersionAt2305_KeepsGraphStreamOrdering) {
  ParseCfg_Scene s;
  s.config().version = NCCL_VERSION(2, 30, 5);
  s.config().graphStreamOrdering = 0;
  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(0, s.result_config().graphStreamOrdering);
}

// --- graphStreamOrdering=0 is unsupported with graph mixing and is forced back to 1 (init.cc:3452) ---

TEST_F(InitMicrotest, ParseCommConfig_StreamOrderingZeroWithGraphMixing_WarnsAndForcesSerialize) {
  ParseCfg_Scene s;
  s.config().graphStreamOrdering = 0;
  s.config().graphUsageMode = 2;
  ncclResult_t res = ncclInternalError;
  const std::string log = s.RunCapturingWarn(&res);
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_EQ(1, s.result_config().graphStreamOrdering);
  EXPECT_EQ(2, s.result_config().graphUsageMode);
  EXPECT_TRUE(LogHas(log, "graphStreamOrdering=0 with graphUsageMode=2")) << "actual log:\n" << log;
}
TEST_F(InitMicrotest, ParseCommConfig_StreamOrderingZeroWithoutGraphMixing_StaysZeroAndIsSilent) {
  ParseCfg_Scene s;
  s.config().graphStreamOrdering = 0;
  s.config().graphUsageMode = 1;
  ncclResult_t res = ncclInternalError;
  const std::string log = s.RunCapturingInfo(&res);
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_EQ(0, s.result_config().graphStreamOrdering);
  EXPECT_FALSE(LogHas(log, "graphStreamOrdering=0 with graphUsageMode=2")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "Comm config graphStreamOrdering set to 0")) << "actual log:\n" << log;
}
TEST_F(InitMicrotest, ParseCommConfig_GraphMixingWithStreamOrderingOne_StaysOneAndIsSilent) {
  ParseCfg_Scene s;
  s.config().graphStreamOrdering = 1;
  s.config().graphUsageMode = 2;
  ncclResult_t res = ncclInternalError;
  const std::string log = s.RunCapturingInfo(&res);
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_EQ(1, s.result_config().graphStreamOrdering);
  EXPECT_FALSE(LogHas(log, "graphStreamOrdering=0 with graphUsageMode=2")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "Comm config graphStreamOrdering set to 1")) << "actual log:\n" << log;
}

// --- computeBuffSizes: the multi-node, chunk-clamp and shared-resource arms (init.cc:1299-1315) ---

namespace {
class BuffSizes_Scene {
 public:
  BuffSizes_Scene() : comm_(new ncclComm{}), sr_(new ncclSharedResources{}) {
    comm_->sharedRes = sr_.get();
    sr_->owner = comm_.get();
  }
  ncclComm* comm() { return comm_.get(); }
  ncclSharedResources* shared() { return sr_.get(); }

 private:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclSharedResources> sr_;
};
}  // namespace

TEST_F(InitMicrotest, ComputeBuffSizes_MultiNode_UsesNetChunkSizeNotTheNvlinkOne) {
  BuffSizes_Scene s;
  s.comm()->nNodes = 2;
  s.comm()->isAllNvlink = true;  // would yield the 512 kB NVL size if the node count were misread
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(1 << 17, s.comm()->p2pChunkSize);
}
TEST_F(InitMicrotest, ComputeBuffSizes_SingleNodeAllNvlink_UsesTheNvlinkChunkSize) {
  BuffSizes_Scene s;
  s.comm()->nNodes = 1;
  s.comm()->isAllNvlink = true;
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(1 << 19, s.comm()->p2pChunkSize);
}
TEST_F(InitMicrotest, ComputeBuffSizes_ChunkExceedsSimpleBuffer_ClampsToOneStep) {
  BuffSizes_Scene s;
  s.comm()->nNodes = 1;
  s.comm()->isAllNvlink = false;
  SetParams({{"BUFFSIZE", 1 << 16}, {"P2P_PCI_CHUNKSIZE", 1 << 15}});
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(1 << 16, s.comm()->buffSizes[NCCL_PROTO_SIMPLE]);
  EXPECT_EQ((1 << 16) / NCCL_STEPS, s.comm()->p2pChunkSize);
}
TEST_F(InitMicrotest, ComputeBuffSizes_ChunkFitsSimpleBuffer_IsLeftAlone) {
  BuffSizes_Scene s;
  s.comm()->nNodes = 1;
  s.comm()->isAllNvlink = false;
  SetParams({{"BUFFSIZE", 1 << 22}, {"P2P_PCI_CHUNKSIZE", 1 << 15}});
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(1 << 15, s.comm()->p2pChunkSize);
}
TEST_F(InitMicrotest, ComputeBuffSizes_NotSharedResOwner_CapsToTheSharedChunkSize) {
  BuffSizes_Scene s;
  auto other = std::make_unique<ncclComm>();
  s.shared()->owner = other.get();
  s.shared()->tpP2pChunkSize = 4096;
  s.comm()->nNodes = 1;
  s.comm()->isAllNvlink = false;
  SetParams({{"P2P_PCI_CHUNKSIZE", 1 << 15}});
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(4096, s.comm()->p2pChunkSize);
  EXPECT_EQ(4096, s.shared()->tpP2pChunkSize);
}
TEST_F(InitMicrotest, ComputeBuffSizes_NotSharedResOwnerWithLargerShared_KeepsItsOwnChunkSize) {
  BuffSizes_Scene s;
  auto other = std::make_unique<ncclComm>();
  s.shared()->owner = other.get();
  s.shared()->tpP2pChunkSize = 1 << 20;
  s.comm()->nNodes = 1;
  s.comm()->isAllNvlink = false;
  SetParams({{"P2P_PCI_CHUNKSIZE", 1 << 15}});
  EXPECT_EQ(ncclSuccess, computeBuffSizes(s.comm()));
  EXPECT_EQ(1 << 15, s.comm()->p2pChunkSize);
  EXPECT_EQ(1 << 20, s.shared()->tpP2pChunkSize);
}

// --- fillInfo: the AMD SMI UALoE/MNNVL fabric probe (init.cc:1200-1219) ---

namespace {
constexpr uint32_t FillInfo_kFabricDeviceIndex = 3;

void FillInfo_FillFabricInfo(struct amdsmiFabricDeviceInfo* info) {
  info->fabricSupported = true;
  info->acceleratorId = 11;
  info->bandwidth = 400000;
  info->latency = 250;
  info->ppodSize = 8;
  info->cliqueId = 5;
  info->vpodSize = 4;
  for (std::size_t i = 0; i < sizeof(info->clusterUuid); ++i) {
    info->clusterUuid[i] = static_cast<uint8_t>(i + 1);
  }
}

std::string FillInfo_RunCapturingInfo(FillInfoComm* c, ncclPeerInfo* info, ncclResult_t* result) {
  ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
  return RcclUnitTesting::CaptureLog([&] { *result = fillInfo(c->get(), info, 0); });
}
}  // namespace

TEST_F(InitMicrotest, FillInfo_NoAmdSmiFabricDevice_SkipsTheProbeAndLeavesFabricInfoAlone) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  ncclPeerInfo info{};
  info.fabricInfo.fabricSupported = true;
  std::string probedBusId = "not-probed";
  ScopedHook index(g_amdSmiGetDeviceIndexByPciBusId, [&](const char* busId, uint32_t* out) {
    probedBusId = busId;
    *out = static_cast<uint32_t>(-1);
    return ncclSuccess;
  });
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo,
                    [](uint32_t, struct amdsmiFabricDeviceInfo*) { return ncclSuccess; });
  EXPECT_EQ(ncclSuccess, fillInfo(c.get(), &info, 0));
  EXPECT_EQ(1, index.calls);
  EXPECT_EQ(0, fabric.calls);
  EXPECT_TRUE(info.fabricInfo.fabricSupported);
  char expected[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  ASSERT_EQ(ncclSuccess, int64ToBusId(info.busId, expected));
  EXPECT_EQ(std::string(expected), probedBusId);
}

TEST_F(InitMicrotest, FillInfo_FabricDeviceWithoutFabricSupport_ClearsTheFlagAndLogsNothing) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  g_pciComputePartition = "CPX";  // makes fillInfo emit the MLOPart line, anchoring the log capture
  ncclPeerInfo info{};
  info.fabricInfo.fabricSupported = true;
  uint32_t handedIndex = 0;
  ScopedHook index(g_amdSmiGetDeviceIndexByPciBusId, [](const char*, uint32_t* out) {
    *out = FillInfo_kFabricDeviceIndex;
    return ncclSuccess;
  });
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo,
                    [&](uint32_t deviceIndex, struct amdsmiFabricDeviceInfo*) {
                      handedIndex = deviceIndex;
                      return ncclSuccess;
                    });
  ncclResult_t res = ncclInternalError;
  const std::string log = FillInfo_RunCapturingInfo(&c, &info, &res);
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_EQ(1, fabric.calls);
  EXPECT_EQ(FillInfo_kFabricDeviceIndex, handedIndex);
  EXPECT_FALSE(info.fabricInfo.fabricSupported);
  EXPECT_FALSE(LogHas(log, "UALoE-enabled")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "MLOPart: physical device")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, FillInfo_FabricSupported_LogsTopologyWithBothUuidHalves) {
  FillInfoComm c;
  c.get()->busId = kPhysGpuBusId;
  ncclPeerInfo info{};
  ScopedHook index(g_amdSmiGetDeviceIndexByPciBusId, [](const char*, uint32_t* out) {
    *out = FillInfo_kFabricDeviceIndex;
    return ncclSuccess;
  });
  ScopedHook fabric(g_amdSmiGetFabricDeviceInfo,
                    [](uint32_t, struct amdsmiFabricDeviceInfo* out) {
                      FillInfo_FillFabricInfo(out);
                      return ncclSuccess;
                    });
  ncclResult_t res = ncclInternalError;
  const std::string log = FillInfo_RunCapturingInfo(&c, &info, &res);
  EXPECT_EQ(ncclSuccess, res);
  EXPECT_TRUE(info.fabricInfo.fabricSupported);
  EXPECT_EQ(11u, info.fabricInfo.acceleratorId);
  EXPECT_TRUE(LogHas(log, "UALoE-enabled (aka MNNVL) device busId 0x11000")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "acceleratorId 11 bandwidth 400000 Mb/s latency 250 ns")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "UUID 807060504030201.100f0e0d0c0b0a09")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "ppodSize 8 cliqueId 5 clique size 4")) << "actual log:\n" << log;
}

// Comm teardown / lifecycle (init.cc:453-4100): commFree, setCommAbortFlags, destroy/finalize/revoke/abort.

namespace {

// commFree() ends in free(comm), so the comm must be malloc-backed exactly like production.
void Teardown_MakeFreeableComm(ncclComm** comm, uint32_t* abortFlag, int* abortRefCount) {
  InstallCommAllocSuccess();
  *comm = nullptr;
  ASSERT_EQ(ncclSuccess, ncclCalloc(comm, 1));
  ASSERT_EQ(ncclSuccess, commAlloc(*comm, /*parent=*/nullptr, /*ndev=*/8, /*rank=*/0));
  (*comm)->abortFlag = abortFlag;
  (*comm)->abortFlagRefCount = abortRefCount;
}

struct Teardown_DtorNode {
  struct ncclDestructor base;
  int id;
  std::vector<int>* log;
  ncclResult_t result;
};

ncclResult_t Teardown_LogDtor(struct ncclDestructor* me) {
  auto* node = reinterpret_cast<Teardown_DtorNode*>(me);
  node->log->push_back(node->id);
  return node->result;
}

// The destructor chain runs after the task-queue drain, so it is where "were the queues emptied?" is observable.
ncclResult_t Teardown_LogTaskQueueState(struct ncclDestructor* me) {
  auto* node = reinterpret_cast<Teardown_DtorNode*>(me);
  node->log->push_back(ncclIntruQueueEmpty(&me->comm->suspendTaskQueue) ? 1 : 0);
  node->log->push_back(ncclIntruQueueEmpty(&me->comm->resumeTaskQueue) ? 1 : 0);
  return ncclSuccess;
}

void Teardown_ChainDtors(Teardown_DtorNode* nodes, int n, std::vector<int>* log) {
  for (int i = 0; i < n; ++i) {
    nodes[i].base.fn = Teardown_LogDtor;
    nodes[i].base.next = (i + 1 < n) ? &nodes[i + 1].base : nullptr;
    nodes[i].log = log;
    nodes[i].result = ncclSuccess;
  }
}

}  // namespace

TEST_F(InitMicrotest, CommFree_DestructorChain_RunsEveryNodeHeadToTail) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  std::vector<int> ran;
  Teardown_DtorNode nodes[3]{};
  Teardown_ChainDtors(nodes, 3, &ran);
  nodes[0].id = 11;
  nodes[1].id = 22;
  nodes[2].id = 33;
  comm->destructorHead = &nodes[0].base;

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<int>({11, 22, 33}), ran);
}

TEST_F(InitMicrotest, CommFree_DestructorFails_PropagatesErrorAndStopsWalk) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  std::vector<int> ran;
  Teardown_DtorNode nodes[2]{};
  Teardown_ChainDtors(nodes, 2, &ran);
  nodes[0].id = 44;
  nodes[0].result = ncclSystemError;
  nodes[1].id = 55;
  comm->destructorHead = &nodes[0].base;

  EXPECT_EQ(ncclSystemError, commFree(comm));
  EXPECT_EQ(std::vector<int>({44}), ran);
  free(comm);
}

TEST_F(InitMicrotest, CommFree_SharedAbortFlagReference_DecrementsRefCountByOne) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 3;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(2, abortRef);
  EXPECT_EQ(0u, abortFlag);
}

TEST_F(InitMicrotest, CommFree_LastAbortFlagReference_ReleasesFlagStorage) {
  ncclComm* comm = nullptr;
  auto* abortFlag = static_cast<uint32_t*>(::calloc(1, sizeof(uint32_t)));
  auto* abortRefCount = static_cast<int*>(::malloc(sizeof(int)));
  *abortRefCount = 1;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, abortFlag, abortRefCount));
  comm->abortFlagDev = static_cast<uint32_t*>(::calloc(1, sizeof(uint32_t)));

  std::vector<void*> released;
  {
    ScopedHook microFree(g_microFree, [&](void* p) {
      released.push_back(p);
      ::free(p);
    });
    EXPECT_EQ(ncclSuccess, commFree(comm));
  }
  EXPECT_EQ(1, std::count(released.begin(), released.end(), static_cast<void*>(abortFlag)));
  EXPECT_EQ(1, std::count(released.begin(), released.end(), static_cast<void*>(abortRefCount)));
}

TEST_F(InitMicrotest, CommFree_AbortFlagSet_ReportsAbortCompletion) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 1;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
  const std::string log = RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, commFree(comm)); });
  EXPECT_TRUE(LogHas(log, "- Abort COMPLETE")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "- Destroy COMPLETE")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommFree_AbortFlagClear_ReportsDestroyCompletion) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
  const std::string log = RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, commFree(comm)); });
  EXPECT_TRUE(LogHas(log, "- Destroy COMPLETE")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "- Abort COMPLETE")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommFree_PendingSuspendAndResumeTasks_DrainsBothQueues) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  const int kSuspendTasks = 3;
  const int kResumeTasks = 2;
  for (int i = 0; i < kSuspendTasks; ++i) {
    ncclIntruQueueEnqueue(&comm->suspendTaskQueue,
                          static_cast<ncclMemManagerTask*>(::calloc(1, sizeof(ncclMemManagerTask))));
  }
  for (int i = 0; i < kResumeTasks; ++i) {
    ncclIntruQueueEnqueue(&comm->resumeTaskQueue,
                          static_cast<ncclMemManagerTask*>(::calloc(1, sizeof(ncclMemManagerTask))));
  }

  std::vector<int> emptied;
  Teardown_DtorNode probe{};
  probe.base.fn = Teardown_LogTaskQueueState;
  probe.base.comm = comm;
  probe.log = &emptied;
  comm->destructorHead = &probe.base;

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<int>({1, 1}), emptied);
}

TEST_F(InitMicrotest, CommFree_NodeRanksTable_FreesEveryNodeEntry) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));

  const int kNodes = 3;
  comm->nNodes = kNodes;
  comm->nodeRanks = static_cast<ncclNodeRanks*>(::calloc(kNodes, sizeof(ncclNodeRanks)));
  ncclNodeRanks* const table = comm->nodeRanks;
  void* perNode[kNodes];
  for (int n = 0; n < kNodes; ++n) {
    table[n].localRankToRank = static_cast<int*>(::calloc(4, sizeof(int)));
    perNode[n] = table[n].localRankToRank;
  }

  std::vector<void*> released;
  {
    ScopedHook microFree(g_microFree, [&](void* p) {
      released.push_back(p);
      ::free(p);
    });
    EXPECT_EQ(ncclSuccess, commFree(comm));
  }
  for (int n = 0; n < kNodes; ++n) {
    EXPECT_EQ(1, std::count(released.begin(), released.end(), perNode[n])) << "node " << n;
  }
  EXPECT_EQ(1, std::count(released.begin(), released.end(), static_cast<void*>(table)));
}

// --- commDestroySync / commCleanup / commReclaim and the public entry points ---
namespace {

// Magic-valid heap comm for tests that stop before commFree; Teardown_MakeFreeableComm is the malloc-backed one.
class Teardown_LifecycleComm {
 public:
  explicit Teardown_LifecycleComm(int cudaDev = 1, int rank = 0)
      : comm_(new ncclComm{}), sharedRes_(new ncclSharedResources{}) {
    comm_->startMagic = comm_->endMagic = NCCL_MAGIC;
    comm_->abortFlag = &abortFlag_;
    comm_->abortFlagDev = &abortFlagDev_;
    comm_->childAbortFlag = &childAbortFlag_;
    comm_->childAbortFlagDev = &childAbortFlagDev_;
    comm_->cudaDev = cudaDev;
    comm_->rank = rank;
    comm_->nRanks = 2;
    comm_->busId = 0x1000 + rank;
    comm_->config.blocking = 1;
    comm_->initState = ncclSuccess;
    comm_->sharedRes = sharedRes_.get();
  }
  ncclComm* get() { return comm_.get(); }
  uint32_t abortFlag() const { return abortFlag_; }
  uint32_t abortFlagDev() const { return abortFlagDev_; }
  uint32_t childAbortFlag() const { return childAbortFlag_; }
  uint32_t childAbortFlagDev() const { return childAbortFlagDev_; }

 private:
  uint32_t abortFlag_ = 0;
  uint32_t abortFlagDev_ = 0;
  uint32_t childAbortFlag_ = 0;
  uint32_t childAbortFlagDev_ = 0;
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclSharedResources> sharedRes_;
};

// The callback drain exchanges the stream-capture mode; the hip fake fails that by default.
void Teardown_AllowStreamCaptureExchange() { g_hipAsyncOpsResult = hipSuccess; }

ncclResult_t Teardown_RunDestroySync(ncclComm* comm) {
  ncclCommFinalizeAsyncJob job{};
  job.comm = comm;
  return commDestroySync(reinterpret_cast<ncclAsyncJob*>(&job));
}

ncclResult_t Teardown_RunReclaim(ncclComm* comm) {
  ncclCommFinalizeAsyncJob job{};
  job.comm = comm;
  return commReclaim(reinterpret_cast<ncclAsyncJob*>(&job));
}

ncclResult_t Teardown_RunRevokeAsync(ncclComm* comm) {
  ncclCommRevokeAsyncJob job{};
  job.comm = comm;
  return commRevokeAsync(reinterpret_cast<ncclAsyncJob*>(&job));
}

// Records the launched job instead of running it, so the entry point's own contract stays observable.
using Teardown_JobFn = ncclResult_t (*)(ncclAsyncJob*);

struct Teardown_LaunchRecord {
  int calls = 0;
  ncclAsyncJob* job = nullptr;
  ncclResult_t (*func)(ncclAsyncJob*) = nullptr;
  void (*destructor)(void*) = nullptr;
  ncclComm* comm = nullptr;
  ncclResult_t result = ncclSuccess;

  void Release() {
    if (job && destructor) destructor(job);
    job = nullptr;
  }
};

std::function<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(ncclAsyncJob*), void (*)(void*),
                           ncclComm*)>
Teardown_CaptureLaunch(Teardown_LaunchRecord* rec) {
  return [rec](ncclAsyncJob* job, ncclResult_t (*func)(ncclAsyncJob*), void (*)(ncclAsyncJob*),
               void (*destructor)(void*), ncclComm* comm) {
    rec->calls++;
    rec->job = job;
    rec->func = func;
    rec->destructor = destructor;
    rec->comm = comm;
    return rec->result;
  };
}

struct Teardown_CommCallback {
  struct ncclCommCallback base;
  ncclComm* owner;
  ncclResult_t result;
};

}  // namespace

TEST_F(InitMicrotest, CommDestroySync_InitStateSuccess_SetsDeviceStopsProxyAndDestroysCollTrace) {
  Teardown_LifecycleComm c(/*cudaDev=*/3);
  Teardown_AllowStreamCaptureExchange();
  int setDev = -1;
  ScopedHook setDevice(g_hipSetDevice, [&](int dev) {
    setDev = dev;
    return hipSuccess;
  });
  ncclComm* tracedComm = nullptr;
  ScopedHook collTrace(g_collTraceDestroy, [&](ncclComm* comm) {
    tracedComm = comm;
    return ncclSuccess;
  });
  ncclComm* stoppedComm = nullptr;
  ScopedHook proxyStop(g_ncclProxyStop, [&](ncclComm* comm) {
    stoppedComm = comm;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, Teardown_RunDestroySync(c.get()));
  EXPECT_EQ(3, setDev);
  EXPECT_EQ(c.get(), tracedComm);
  EXPECT_EQ(c.get(), stoppedComm);
  EXPECT_EQ(1, collTrace.calls);
  EXPECT_EQ(1, proxyStop.calls);
}

TEST_F(InitMicrotest, CommDestroySync_SetDeviceFails_ReturnsErrorWithoutDestroyingCollTrace) {
  Teardown_LifecycleComm c;
  ScopedHook setDevice(g_hipSetDevice, [](int) { return hipErrorInvalidDevice; });
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_NE(ncclSuccess, Teardown_RunDestroySync(c.get()));
  EXPECT_EQ(0, collTrace.calls);
  EXPECT_EQ(0, proxyStop.calls);
}

TEST_F(InitMicrotest, CommDestroySync_CollTraceDestroyFails_PropagatesAndSkipsProxyStop) {
  Teardown_LifecycleComm c;
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSystemError; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSystemError, Teardown_RunDestroySync(c.get()));
  EXPECT_EQ(1, collTrace.calls);
  EXPECT_EQ(0, proxyStop.calls);
}

TEST_F(InitMicrotest, CommDestroySync_InitStateNotReady_SkipsStreamSyncButStillStopsProxy) {
  Teardown_LifecycleComm c;
  c.get()->initState = ncclInProgress;
  c.get()->localPersistentRefs = 1;  // the poll loop is inside the skipped block; a hang means it ran
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, Teardown_RunDestroySync(c.get()));
  EXPECT_EQ(1, proxyStop.calls);
  EXPECT_EQ(1, c.get()->localPersistentRefs);
}

TEST_F(InitMicrotest, CommDestroySync_StreamSyncFails_WarnsAndStillStopsProxy) {
  Teardown_LifecycleComm c;
  Teardown_AllowStreamCaptureExchange();
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });
  g_ncclStrongStreamResult = ncclSystemError;

  const std::string log = RcclUnitTesting::CaptureLog([&] { Teardown_RunDestroySync(c.get()); });
  EXPECT_TRUE(LogHas(log, "commDestroySync: comm")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "sync hostStream error")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "sync deviceStream error")) << "actual log:\n" << log;
  EXPECT_EQ(1, proxyStop.calls);
}

TEST_F(InitMicrotest, CommDestroySync_ProxyStopFails_WarnsAndReturnsThatError) {
  Teardown_LifecycleComm c;
  Teardown_AllowStreamCaptureExchange();
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclInternalError; });

  ncclResult_t ret = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { ret = Teardown_RunDestroySync(c.get()); });
  EXPECT_EQ(ncclInternalError, ret);
  EXPECT_TRUE(LogHas(log, "ncclProxyStop: comm")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommDestroySync_PersistentRefsOutstanding_PollsUntilCallbacksClearThem) {
  Teardown_LifecycleComm c;
  Teardown_AllowStreamCaptureExchange();
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  // The first drain re-arms the counter and enqueues the second callback, so the waitSome poll cannot block.
  static Teardown_CommCallback second;
  static Teardown_CommCallback first;
  second = {};
  first = {};
  second.owner = c.get();
  second.base.fn = [](ncclComm* comm, ncclCommCallback*) {
    comm->localPersistentRefs = 0;
    return ncclSuccess;
  };
  first.owner = c.get();
  first.base.fn = [](ncclComm* comm, ncclCommCallback*) {
    comm->localPersistentRefs = 1;
    ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &second.base);
    return ncclSuccess;
  };
  ncclIntruQueueMpscEnqueue(&c.get()->callbackQueue, &first.base);
  c.get()->localPersistentRefs = 1;

  EXPECT_EQ(ncclSuccess, Teardown_RunDestroySync(c.get()));
  EXPECT_EQ(0, c.get()->localPersistentRefs);
}

TEST_F(InitMicrotest, CommDestroySync_LegacyCleanupCallbackFails_WarnsAndDrainsRemainder) {
  Teardown_LifecycleComm c;
  Teardown_AllowStreamCaptureExchange();
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclSuccess; });
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  static int ran;
  ran = 0;
  Teardown_CommCallback cbs[2]{};
  cbs[0].base.fn = [](ncclComm*, ncclCommCallback*) {
    ran++;
    return ncclSystemError;
  };
  cbs[1].base.fn = [](ncclComm*, ncclCommCallback*) {
    ran++;
    return ncclSuccess;
  };
  ncclIntruQueueEnqueue(&c.get()->legacyRegCleanupQueue, &cbs[0].base);
  ncclIntruQueueEnqueue(&c.get()->legacyRegCleanupQueue, &cbs[1].base);

  const std::string log =
      RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, Teardown_RunDestroySync(c.get())); });
  EXPECT_EQ(2, ran);
  EXPECT_TRUE(LogHas(log, "Legacy IPC cleanup callback failed comm")) << "actual log:\n" << log;
  EXPECT_TRUE(ncclIntruQueueEmpty(&c.get()->legacyRegCleanupQueue));
}

// --- shared tuner probe for commCleanup-driven teardown (init.cc:3772) ---
namespace {
constexpr uint64_t Teardown_kTunerMagic = 0xC0FFEE01u;

struct Teardown_TunerRecord {
  uint64_t magic = Teardown_kTunerMagic;
  int finalizeCalls = 0;
};

// Returning an error unless the magic survives is what makes "passed the wrong context" a failing test.
ncclResult_t Teardown_TunerFinalize(void* context) {
  auto* rec = static_cast<Teardown_TunerRecord*>(context);
  if (rec == nullptr || rec->magic != Teardown_kTunerMagic) return ncclInternalError;
  rec->finalizeCalls++;
  return ncclSuccess;
}

void Teardown_AttachTuner(ncclComm* comm, ncclTuner_t* tuner, Teardown_TunerRecord* rec) {
  *tuner = ncclTuner_t{};
  tuner->finalize = Teardown_TunerFinalize;
  comm->tuner = tuner;
  comm->tunerContext = rec;
}
}  // namespace

// --- commReclaim (init.cc:3833) ---
namespace {
// Carries a tuner probe so "commCleanup ran on this chain member" stays observable after the comm is freed.
struct Teardown_ChainMember {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRefCount = 2;
  ncclTuner_t tuner{};
  Teardown_TunerRecord rec;
};

void Teardown_BuildChain(Teardown_ChainMember* members, int n) {
  for (int i = 0; i < n; ++i) {
    ASSERT_NO_FATAL_FAILURE(
        Teardown_MakeFreeableComm(&members[i].comm, &members[i].abortFlag, &members[i].abortRefCount));
    members[i].comm->rank = i;
    members[i].comm->intraRanks = n;
    members[i].comm->intraComm0 = members[0].comm;
    Teardown_AttachTuner(members[i].comm, &members[i].tuner, &members[i].rec);
  }
  for (int i = 0; i < n; ++i) {
    members[i].comm->intraNext = (i + 1 < n) ? members[i + 1].comm : nullptr;
  }
}
}  // namespace

TEST_F(InitMicrotest, CommReclaim_NoIntraComm0_ReturnsSuccessWithoutTouchingTheComm) {
  Teardown_LifecycleComm c;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(c.get()));
  EXPECT_EQ(0, proxyStop.calls);
  EXPECT_EQ(0, c.get()->finalizeRankCnt);
}

TEST_F(InitMicrotest, CommReclaim_NotLastIntraRank_BumpsLeaderCounterAndDefersCleanup) {
  const int kChainLength = 2;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[1].comm));
  EXPECT_EQ(1, members[0].comm->finalizeRankCnt);
  EXPECT_EQ(0, proxyStop.calls);
  EXPECT_EQ(0, members[0].rec.finalizeCalls);
  EXPECT_EQ(0, members[1].rec.finalizeCalls);

  for (int i = 0; i < kChainLength; ++i) {
    members[i].comm->tuner = nullptr;
    EXPECT_EQ(ncclSuccess, commFree(members[i].comm));
  }
}

TEST_F(InitMicrotest, CommReclaim_LastIntraRank_SyncsAndCleansEveryCommInTheChain) {
  Teardown_AllowStreamCaptureExchange();
  const int kChainLength = 3;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeRankCnt = kChainLength - 1;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[kChainLength - 1].comm));
  EXPECT_EQ(kChainLength, proxyStop.calls);
  for (int i = 0; i < kChainLength; ++i) {
    EXPECT_EQ(1, members[i].rec.finalizeCalls) << "chain member " << i << " was not cleaned up";
    EXPECT_EQ(1, members[i].abortRefCount);
  }
}

TEST_F(InitMicrotest, CommReclaim_DestroySyncFails_WarnsAndStillCleansTheChain) {
  const int kChainLength = 2;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeRankCnt = kChainLength - 1;
  ScopedHook collTrace(g_collTraceDestroy, [](ncclComm*) { return ncclInternalError; });

  const std::string log =
      RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[1].comm)); });
  EXPECT_TRUE(LogHas(log, "commReclaim: comm")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "in commDestroySync, error")) << "actual log:\n" << log;
  for (int i = 0; i < kChainLength; ++i) EXPECT_EQ(1, members[i].rec.finalizeCalls);
}

TEST_F(InitMicrotest, CommReclaim_RevokedComm_DrainsLegacyQueueInsteadOfDestroySync) {
  const int kChainLength = 1;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeCalled = true;
  members[0].comm->revokedFlag = 1;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  static int drained;
  drained = 0;
  Teardown_CommCallback cb{};
  cb.base.fn = [](ncclComm*, ncclCommCallback*) {
    drained++;
    return ncclSuccess;
  };
  ncclIntruQueueEnqueue(&members[0].comm->legacyRegCleanupQueue, &cb.base);

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[0].comm));
  EXPECT_EQ(1, drained);
  EXPECT_EQ(0, proxyStop.calls);
  EXPECT_EQ(1, members[0].rec.finalizeCalls);
}

TEST_F(InitMicrotest, CommReclaim_FinalizedButNotRevoked_SkipsBothSyncAndDrain) {
  const int kChainLength = 1;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeCalled = true;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  static int drained;
  drained = 0;
  Teardown_CommCallback cb{};
  cb.base.fn = [](ncclComm*, ncclCommCallback*) {
    drained++;
    return ncclSuccess;
  };
  ncclIntruQueueEnqueue(&members[0].comm->legacyRegCleanupQueue, &cb.base);

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[0].comm));
  EXPECT_EQ(0, drained);
  EXPECT_EQ(0, proxyStop.calls);
  EXPECT_EQ(1, members[0].rec.finalizeCalls);
}

// --- commRevokeAsync (init.cc:3972) ---
TEST_F(InitMicrotest, CommRevokeAsync_ValidComm_StopsProxyAndLowersEveryAbortFlag) {
  Teardown_AllowStreamCaptureExchange();
  Teardown_LifecycleComm c;
  ASSERT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 1));
  ncclComm* stoppedComm = nullptr;
  ScopedHook proxyStop(g_ncclProxyStop, [&](ncclComm* comm) {
    stoppedComm = comm;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, Teardown_RunRevokeAsync(c.get()));
  EXPECT_EQ(c.get(), stoppedComm);
  EXPECT_EQ(0u, c.abortFlag());
  EXPECT_EQ(0u, c.abortFlagDev());
  EXPECT_EQ(0u, c.childAbortFlag());
  EXPECT_EQ(0u, c.childAbortFlagDev());
}

TEST_F(InitMicrotest, CommRevokeAsync_StreamSyncFails_LeavesAbortFlagsRaisedAndRecordsAsyncError) {
  Teardown_LifecycleComm c;
  ASSERT_EQ(ncclSuccess, setCommAbortFlags(c.get(), 1));
  g_ncclStrongStreamResult = ncclSystemError;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSystemError, Teardown_RunRevokeAsync(c.get()));
  EXPECT_EQ(0, proxyStop.calls);
  EXPECT_EQ(1u, c.abortFlag());
  EXPECT_EQ(1u, c.childAbortFlag());
  EXPECT_EQ(ncclSystemError, c.get()->asyncResult);
}

TEST_F(InitMicrotest, CommRevokeAsync_NullComm_ReportsTheArgumentCheckAndStopsNoProxy) {
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });
  ncclResult_t ret = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { ret = Teardown_RunRevokeAsync(nullptr); });
  EXPECT_EQ(ncclInvalidArgument, ret);
  EXPECT_TRUE(LogHas(log, "CommRevokeAsync : comm argument is NULL")) << "actual log:\n" << log;
  EXPECT_EQ(0, proxyStop.calls);
}

// --- ncclCommFinalize_impl (init.cc:3784) ---
TEST_F(InitMicrotest, CommFinalize_NullComm_ReturnsSuccessWithoutLaunching) {
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  EXPECT_EQ(ncclSuccess, ncclCommFinalize_impl(nullptr));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommFinalize_FirstCall_MarksFinalizedAndLaunchesDestroySync) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclSuccess, ncclCommFinalize_impl(c.get()));
  EXPECT_TRUE(c.get()->finalizeCalled);
  EXPECT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(commDestroySync), rec.func);
  EXPECT_EQ(c.get(), rec.comm);
  EXPECT_EQ(c.get(), reinterpret_cast<ncclCommFinalizeAsyncJob*>(rec.job)->comm);
  rec.Release();
}

TEST_F(InitMicrotest, CommFinalize_SecondCall_ReturnsInvalidArgumentAndLaunchesOnlyOnce) {
  Teardown_LifecycleComm c;
  c.get()->finalizeCalled = true;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclInvalidArgument, ncclCommFinalize_impl(c.get()));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommFinalize_RevokedComm_WarnsAndReturnsInvalidUsage) {
  Teardown_LifecycleComm c;
  c.get()->revokedFlag = 1;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  ncclResult_t ret = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { ret = ncclCommFinalize_impl(c.get()); });
  EXPECT_EQ(ncclInvalidUsage, ret);
  EXPECT_TRUE(LogHas(log, "has been revoked; use ncclCommDestroy instead")) << "actual log:\n" << log;
  EXPECT_EQ(0, rec.calls);
  EXPECT_FALSE(c.get()->finalizeCalled);
}

TEST_F(InitMicrotest, CommFinalize_NonBlockingRevokedComm_PublishesTheErrorAsAsyncResult) {
  Teardown_LifecycleComm c;
  c.get()->revokedFlag = 1;
  c.get()->config.blocking = 0;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  ncclResult_t ret = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] { ret = ncclCommFinalize_impl(c.get()); });
  EXPECT_EQ(ncclInvalidUsage, ret);
  EXPECT_EQ(ncclInvalidUsage, c.get()->asyncResult);
}

// --- ncclCommDestroy_impl (init.cc:3905) ---
TEST_F(InitMicrotest, CommDestroy_NullComm_ReturnsSuccessWithoutLaunching) {
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  EXPECT_EQ(ncclSuccess, ncclCommDestroy_impl(nullptr));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommDestroy_ValidComm_SetsDestroyFlagAndLaunchesReclaim) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclSuccess, ncclCommDestroy_impl(c.get()));
  EXPECT_EQ(1u, c.get()->destroyFlag);
  EXPECT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(commReclaim), rec.func);
  EXPECT_EQ(c.get(), rec.comm);
  EXPECT_EQ(c.get(), reinterpret_cast<ncclCommFinalizeAsyncJob*>(rec.job)->comm);
  EXPECT_EQ(0u, c.abortFlag());  // destroy must not raise the abort flags that abort does
  rec.Release();
}

TEST_F(InitMicrotest, CommDestroy_AlreadyDestroyedComm_WarnsAndReturnsInvalidArgument) {
  Teardown_LifecycleComm c;
  c.get()->rank = -1;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  ncclResult_t ret = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { ret = ncclCommDestroy_impl(c.get()); });
  EXPECT_EQ(ncclInvalidArgument, ret);
  EXPECT_TRUE(LogHas(log, "has already been destroyed")) << "actual log:\n" << log;
  EXPECT_EQ(0, rec.calls);
  EXPECT_EQ(0u, c.get()->destroyFlag);
}

TEST_F(InitMicrotest, CommDestroy_BusIdMarkedDestroyed_ReturnsInvalidArgument) {
  Teardown_LifecycleComm c;
  c.get()->busId = -1;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  ncclResult_t ret = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] { ret = ncclCommDestroy_impl(c.get()); });
  EXPECT_EQ(ncclInvalidArgument, ret);
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommDestroy_LaunchFails_PropagatesTheLaunchError) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  rec.result = ncclSystemError;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclSystemError, ncclCommDestroy_impl(c.get()));
  EXPECT_EQ(1, rec.calls);
  rec.Release();
}

// --- ncclCommAbort_impl (init.cc:4060) ---
TEST_F(InitMicrotest, CommAbort_NullComm_ReturnsSuccessWithoutLaunching) {
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  EXPECT_EQ(ncclSuccess, ncclCommAbort_impl(nullptr));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommAbort_ValidComm_RaisesEveryAbortFlagAndLaunchesReclaim) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclSuccess, ncclCommAbort_impl(c.get()));
  EXPECT_EQ(1u, c.abortFlag());
  EXPECT_EQ(1u, c.abortFlagDev());
  EXPECT_EQ(1u, c.childAbortFlag());
  EXPECT_EQ(1u, c.childAbortFlagDev());
  EXPECT_EQ(1u, c.get()->destroyFlag);
  EXPECT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(commReclaim), rec.func);
  EXPECT_EQ(c.get(), reinterpret_cast<ncclCommFinalizeAsyncJob*>(rec.job)->comm);
  rec.Release();
}

// --- ncclCommRevoke_impl (init.cc:4006) ---
TEST_F(InitMicrotest, CommRevoke_UnsupportedFlags_ReturnsInvalidArgumentWithoutRevoking) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclInvalidArgument, ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT + 1));
  EXPECT_EQ(0u, c.get()->revokedFlag);
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommRevoke_NullComm_ReturnsInvalidArgument) {
  EXPECT_EQ(ncclInvalidArgument, ncclCommRevoke_impl(nullptr, NCCL_REVOKE_DEFAULT));
}

TEST_F(InitMicrotest, CommRevoke_DestroyInProgress_ReturnsInvalidArgument) {
  Teardown_LifecycleComm c;
  c.get()->destroyFlag = 1;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclInvalidArgument, ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT));
  EXPECT_EQ(0u, c.get()->revokedFlag);
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommRevoke_FinalizeInProgress_ReturnsInvalidArgument) {
  Teardown_LifecycleComm c;
  c.get()->finalizeCalled = true;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclInvalidArgument, ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT));
  EXPECT_EQ(0u, c.get()->revokedFlag);
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommRevoke_AlreadyRevoked_ReturnsInvalidArgument) {
  Teardown_LifecycleComm c;
  c.get()->revokedFlag = 1;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclInvalidArgument, ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommRevoke_ValidComm_RaisesAbortFlagsMarksRevokedAndLaunchesRevokeAsync) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  ncclResult_t ret = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] { ret = ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT); });
  EXPECT_EQ(ncclSuccess, ret);
  EXPECT_EQ(1u, c.get()->revokedFlag);
  EXPECT_TRUE(c.get()->finalizeCalled);
  EXPECT_EQ(1u, c.abortFlag());
  EXPECT_EQ(1u, c.childAbortFlag());
  EXPECT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(commRevokeAsync), rec.func);
  EXPECT_EQ(c.get(), reinterpret_cast<ncclCommRevokeAsyncJob*>(rec.job)->comm);
  rec.Release();
}

TEST_F(InitMicrotest, CommRevoke_ValidComm_RunsRevokeAsyncWhichLowersTheAbortFlagsAgain) {
  Teardown_AllowStreamCaptureExchange();
  Teardown_LifecycleComm c;
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  ncclResult_t ret = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] { ret = ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT); });
  EXPECT_EQ(ncclSuccess, ret);
  EXPECT_EQ(1, proxyStop.calls);
  EXPECT_EQ(1u, c.get()->revokedFlag);
  EXPECT_EQ(0u, c.abortFlag());
  EXPECT_EQ(0u, c.childAbortFlag());
}

namespace {
// ncclCudaFree only reaches hipFree when the pointer is untracked and is not its allocation's base address.
class Teardown_DeviceFreeSpy {
 public:
  explicit Teardown_DeviceFreeSpy(ncclComm* comm)
      : memRange_(g_hipMemGetAddressRange,
                  [](hipDeviceptr_t* base, std::size_t* size, hipDeviceptr_t) {
                    if (base) {
                      *base = reinterpret_cast<hipDeviceptr_t>(Dtor_kNotABaseAddress);
                    }
                    if (size) {
                      *size = 0;
                    }
                    return hipSuccess;
                  }),
        deviceFree_(g_hipFree,
                    [this](void* p) {
                      freed_.push_back(p);
                      return hipSuccess;
                    }),
        hostFree_(g_microFree, [comm](void* p) {
          if (p != comm) {
            ::free(p);
          }
        }) {
    Teardown_AllowStreamCaptureExchange();
  }
  Teardown_DeviceFreeSpy(const Teardown_DeviceFreeSpy&) = delete;
  Teardown_DeviceFreeSpy& operator=(const Teardown_DeviceFreeSpy&) = delete;

  const std::vector<void*>& freed() const { return freed_; }

 private:
  std::vector<void*> freed_;
  ScopedHook<hipError_t(hipDeviceptr_t*, std::size_t*, hipDeviceptr_t)> memRange_;
  ScopedHook<hipError_t(void*)> deviceFree_;
  ScopedHook<void(void*)> hostFree_;
};
}  // namespace

// --- commFree's remaining conditional resources ---
TEST_F(InitMicrotest, CommFree_SingleNodeComm_ReleasesBothSizeArraysAndNullsThePointers) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  comm->nNodes = 1;
  // Distinct addresses: equal poisons would let a mutant storing one pointer into the other field pass both checks.
  int localSizes = 0;
  long gatheredSizes = 0;
  comm->localSizes = &localSizes;
  comm->gatheredSizes = &gatheredSizes;
  ASSERT_NE(comm->localSizes, comm->gatheredSizes);

  std::vector<void*> released;
  ScopedHook memFree(g_ncclMemFree, [&](void* p) {
    released.push_back(p);
    return ncclSuccess;
  });

  Teardown_DeviceFreeSpy spy(comm);
  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<void*>({&localSizes, &gatheredSizes}), released);
  EXPECT_EQ(nullptr, comm->localSizes);
  EXPECT_EQ(nullptr, comm->gatheredSizes);
  ::free(comm);
}

TEST_F(InitMicrotest, CommFree_MultiNodeComm_LeavesTheSizeArraysAlone) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  comm->nNodes = 2;
  int localSizes = 0;
  comm->localSizes = &localSizes;
  ScopedHook memFree(g_ncclMemFree, [](void*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(0, memFree.calls);
}


TEST_F(InitMicrotest, CommFree_TempBuff_ReleasesItThroughTheCommsMemManager) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  int untracked = 0;
  Dtor_ReleasedManager mgr(&untracked);
  comm->memManager = mgr.get();
  int buf = 0;
  comm->tempBuff = &buf;

  Teardown_DeviceFreeSpy spy(comm);
  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<void*>({&buf}), spy.freed());
  EXPECT_EQ(nullptr, comm->tempBuff);
  ::free(comm);
}

TEST_F(InitMicrotest, CommFree_HierarchicalTempBuffer_ReleasesItAndClearsThePointer) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  int untracked = 0;
  Dtor_ReleasedManager mgr(&untracked);
  comm->memManager = mgr.get();
  int buf = 0;
  comm->hierarchicalTempBuffer = &buf;

  Teardown_DeviceFreeSpy spy(comm);
  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<void*>({&buf}), spy.freed());
  EXPECT_EQ(nullptr, comm->hierarchicalTempBuffer);
  ::free(comm);
}

TEST_F(InitMicrotest, CommFree_BothTempBuffers_ReleasesTempBuffBeforeTheHierarchicalOne) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  int untracked = 0;
  Dtor_ReleasedManager mgr(&untracked);
  comm->memManager = mgr.get();
  int temp = 0;
  int hierarchical = 0;
  comm->tempBuff = &temp;
  comm->hierarchicalTempBuffer = &hierarchical;

  Teardown_DeviceFreeSpy spy(comm);
  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<void*>({&temp, &hierarchical}), spy.freed());
  EXPECT_EQ(nullptr, comm->tempBuff);
  EXPECT_EQ(nullptr, comm->hierarchicalTempBuffer);
  ::free(comm);
}

TEST_F(InitMicrotest, CommFree_HierarchicalSubComms_DestroysIntraThenInter) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  auto intra = std::make_unique<ncclComm>();
  auto inter = std::make_unique<ncclComm>();
  comm->hierarchicalIntraComm = intra.get();
  comm->hierarchicalInterComm = inter.get();

  std::vector<ncclComm*> destroyed;
  ScopedHook destroy(g_ncclCommDestroy, [&](ncclComm* c) {
    destroyed.push_back(c);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(std::vector<ncclComm*>({intra.get(), inter.get()}), destroyed);
}

TEST_F(InitMicrotest, CommFree_SymmetricSupport_FinalizesSymmetricResources) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  comm->symmetricSupport = true;
  ncclComm* finalized = nullptr;
  ScopedHook symk(g_ncclSymkFinalize, [&](ncclComm* c) {
    finalized = c;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(comm, finalized);
  EXPECT_EQ(1, symk.calls);
}

TEST_F(InitMicrotest, CommFree_NoSymmetricSupport_SkipsSymmetricFinalize) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  comm->symmetricSupport = false;
  ScopedHook symk(g_ncclSymkFinalize, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_EQ(0, symk.calls);
}

TEST_F(InitMicrotest, CommFree_SymmetricFinalizeFails_PropagatesAndStopsTeardown) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  comm->symmetricSupport = true;
  ScopedHook symk(g_ncclSymkFinalize, [](ncclComm*) { return ncclInternalError; });

  EXPECT_EQ(ncclInternalError, commFree(comm));
  EXPECT_EQ(1, symk.calls);
  EXPECT_EQ(2, abortRef) << "commFree bailed before the abort-flag refcount drop";
  ::free(comm);
}

namespace {
// The payload records that the join really happened, not merely that the pointer was non-null.
struct Teardown_ProxyThreads {
  std::unique_ptr<ncclProxyState> state{new ncclProxyState{}};
  bool mainRan = false;
  bool udsRan = false;

  Teardown_ProxyThreads() = default;
  Teardown_ProxyThreads(const Teardown_ProxyThreads&) = delete;
  Teardown_ProxyThreads& operator=(const Teardown_ProxyThreads&) = delete;

  // ncclProxyState holds both threads by value with no destructor, so an unjoined one terminates the binary.
  ~Teardown_ProxyThreads() {
    if (state == nullptr) {
      return;
    }
    if (state->thread.joinable()) {
      state->thread.join();
    }
    if (state->threadUDS.joinable()) {
      state->threadUDS.join();
    }
  }

  void Attach(ncclComm* comm) {
    state->thread = std::thread([this] { mainRan = true; });
    state->threadUDS = std::thread([this] { udsRan = true; });
    comm->proxyState = state.get();
    comm->proxyRefCountOld = 0;
  }
};
}  // namespace

TEST_F(InitMicrotest, CommFree_ProxyThreadsRunning_JoinsBothBeforeReturning) {
  ncclComm* comm = nullptr;
  uint32_t abortFlag = 0;
  int abortRef = 2;
  ASSERT_NO_FATAL_FAILURE(Teardown_MakeFreeableComm(&comm, &abortFlag, &abortRef));
  Teardown_ProxyThreads proxy;
  proxy.Attach(comm);

  EXPECT_EQ(ncclSuccess, commFree(comm));
  EXPECT_TRUE(proxy.mainRan);
  EXPECT_TRUE(proxy.udsRan);
  EXPECT_FALSE(proxy.state->thread.joinable());
  EXPECT_FALSE(proxy.state->threadUDS.joinable());
}

TEST_F(InitMicrotest, CommRevokeAsync_ProxyThreadsRunning_JoinsBothAfterStoppingProxy) {
  Teardown_AllowStreamCaptureExchange();
  Teardown_LifecycleComm c;
  Teardown_ProxyThreads proxy;
  proxy.Attach(c.get());
  ScopedHook proxyStop(g_ncclProxyStop, [](ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, Teardown_RunRevokeAsync(c.get()));
  EXPECT_TRUE(proxy.mainRan);
  EXPECT_TRUE(proxy.udsRan);
  EXPECT_FALSE(proxy.state->thread.joinable());
  EXPECT_FALSE(proxy.state->threadUDS.joinable());
}

// --- commReclaim's revoked-comm drain and its cleanup-failure report ---
TEST_F(InitMicrotest, CommReclaim_RevokedCommWithPersistentRefs_PollsUntilTheyClear) {
  const int kChainLength = 1;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  ncclComm* comm = members[0].comm;
  comm->finalizeCalled = true;
  comm->revokedFlag = 1;
  comm->localPersistentRefs = 1;

  static Teardown_CommCallback clearRefs;
  clearRefs = {};
  clearRefs.base.fn = [](ncclComm* c, ncclCommCallback*) {
    c->localPersistentRefs = 0;
    return ncclSuccess;
  };
  ncclIntruQueueMpscEnqueue(&comm->callbackQueue, &clearRefs.base);

  EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(comm));
  EXPECT_EQ(1, members[0].rec.finalizeCalls);
}

TEST_F(InitMicrotest, CommReclaim_RevokedCommLegacyCallbackFails_WarnsAndKeepsDraining) {
  const int kChainLength = 1;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeCalled = true;
  members[0].comm->revokedFlag = 1;

  static int drained;
  drained = 0;
  Teardown_CommCallback cbs[2]{};
  cbs[0].base.fn = [](ncclComm*, ncclCommCallback*) {
    drained++;
    return ncclSystemError;
  };
  cbs[1].base.fn = [](ncclComm*, ncclCommCallback*) {
    drained++;
    return ncclSuccess;
  };
  ncclIntruQueueEnqueue(&members[0].comm->legacyRegCleanupQueue, &cbs[0].base);
  ncclIntruQueueEnqueue(&members[0].comm->legacyRegCleanupQueue, &cbs[1].base);

  const std::string log =
      RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[0].comm)); });
  EXPECT_EQ(2, drained);
  EXPECT_TRUE(LogHas(log, "commReclaim: legacy IPC cleanup callback failed comm")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommReclaim_CommCleanupFails_WarnsAndStillCleansTheRestOfTheChain) {
  Teardown_AllowStreamCaptureExchange();
  const int kChainLength = 2;
  Teardown_ChainMember members[kChainLength];
  ASSERT_NO_FATAL_FAILURE(Teardown_BuildChain(members, kChainLength));
  members[0].comm->finalizeRankCnt = kChainLength - 1;
  ncclComm* firstComm = members[0].comm;
  ScopedHook unload(g_ncclTunerPluginUnload,
                    [firstComm](ncclComm* c) { return c == firstComm ? ncclSystemError : ncclSuccess; });

  const std::string log =
      RcclUnitTesting::CaptureLog([&] { EXPECT_EQ(ncclSuccess, Teardown_RunReclaim(members[1].comm)); });
  EXPECT_TRUE(LogHas(log, "commReclaim: cleanup comm")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "failed in destroy/abort, error")) << "actual log:\n" << log;
  EXPECT_EQ(1, members[1].rec.finalizeCalls);
  members[0].comm->tuner = nullptr;
  EXPECT_EQ(ncclSuccess, commFree(members[0].comm));
}

// --- the non-blocking failure paths of revoke and abort ---
TEST_F(InitMicrotest, CommRevoke_JobAllocationFails_PropagatesAndPublishesTheAsyncError) {
  Teardown_LifecycleComm c;
  c.get()->config.blocking = 0;
  g_callocFailAt = g_callocCallIndex;  // the job calloc is the next one the UUT reaches

  ncclResult_t ret = ncclSuccess;
  RcclUnitTesting::CaptureLog([&] { ret = ncclCommRevoke_impl(c.get(), NCCL_REVOKE_DEFAULT); });
  EXPECT_EQ(ncclSystemError, ret);
  EXPECT_EQ(ncclSystemError, c.get()->asyncResult);
  EXPECT_EQ(1u, c.get()->revokedFlag);
}

TEST_F(InitMicrotest, CommAbort_LaunchFails_PropagatesTheLaunchError) {
  Teardown_LifecycleComm c;
  Teardown_LaunchRecord rec;
  rec.result = ncclSystemError;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));

  EXPECT_EQ(ncclSystemError, ncclCommAbort_impl(c.get()));
  EXPECT_EQ(1, rec.calls);
  EXPECT_EQ(1u, c.abortFlag());
  rec.Release();
}

// envConfigOverride (init.cc:2963): the NCCL_* env and ncclConfig_t validation ladders.

namespace {
// Every NCCL_PARAM body in this TU routes through g_loadParam(env, deft); this maps an env name to a forced value.
using Env_ParamMap = std::map<std::string, int64_t>;

// NCCL_CONFIG_UNDEF_INT is INT_MIN, so an undefined config field always trips the 0/1 clamps at :3189 and :3194.
constexpr char Env_kUndefSplitShareLog[] = "splitShare -2147483648 is not a valid value 0/1, set it to 0";
constexpr char Env_kUndefCollnetLog[] = "collnetEnable -2147483648 is not a valid value 0/1, set it to 0";

// Large enough that a minCTAs case never trips the min > max clamp at :3183.
constexpr int Env_kAmpleMaxCTAs = 64;

class Env_ConfigComm {
 public:
  Env_ConfigComm() : comm_(new ncclComm{}) {
    const ncclConfig_t fresh = NCCL_CONFIG_INITIALIZER;
    comm_->config = fresh;
  }
  ~Env_ConfigComm() {
    if (ran_) {
      ::free(const_cast<char*>(comm_->config.netName));
    }
  }
  Env_ConfigComm(const Env_ConfigComm&) = delete;
  Env_ConfigComm& operator=(const Env_ConfigComm&) = delete;

  ncclConfig_t& config() { return comm_->config; }
  ncclComm* comm() { return comm_.get(); }
  ncclResult_t result() const { return result_; }

  ncclResult_t Run(const Env_ParamMap& params) {
    ScopedHook loadParam(g_loadParam, [&params](const char* env, int64_t deft) {
      const auto it = params.find(env);
      return it == params.end() ? deft : it->second;
    });
    ran_ = true;
    result_ = envConfigOverride(comm_.get());
    return result_;
  }

  std::string RunCapturingLog(const Env_ParamMap& params) {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    std::string log = RcclUnitTesting::CaptureLog([&] { Run(params); });
    EXPECT_EQ(ncclSuccess, result_) << "actual log:\n" << log;
    return log;
  }

 private:
  std::unique_ptr<ncclComm> comm_;
  ncclResult_t result_ = ncclNumResults;
  bool ran_ = false;
};
}  // namespace

TEST_F(InitMicrotest, EnvConfigOverride_CgaInRangeConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"CGA_CLUSTER_SIZE", 4}});
  EXPECT_EQ(ncclSuccess, c.result());
  EXPECT_EQ(4, c.config().cgaClusterSize);
  EXPECT_FALSE(LogHas(log, "cgaClusterSize reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CgaInRangeConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().cgaClusterSize = 2;
  const std::string log = c.RunCapturingLog({{"CGA_CLUSTER_SIZE", 4}});
  EXPECT_EQ(4, c.config().cgaClusterSize);
  EXPECT_TRUE(LogHas(log, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=4"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CgaAtUpperBound_AcceptedNotClamped) {
  Env_ConfigComm c;
  c.config().cgaClusterSize = 2;
  const std::string log = c.RunCapturingLog({{"CGA_CLUSTER_SIZE", NCCL_MAX_CGA_CLUSTER_SIZE}});
  EXPECT_EQ(NCCL_MAX_CGA_CLUSTER_SIZE, c.config().cgaClusterSize);
  EXPECT_TRUE(LogHas(log, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=8"))
      << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "is too big")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CgaAtLowerBound_AcceptedAndOverwritesConfig) {
  Env_ConfigComm c;
  c.config().cgaClusterSize = 2;
  EXPECT_EQ(ncclSuccess, c.Run({{"CGA_CLUSTER_SIZE", 0}}));
  EXPECT_EQ(0, c.config().cgaClusterSize);
}

TEST_F(InitMicrotest, EnvConfigOverride_CgaAboveMax_ClampsToMaxAndLogs) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"CGA_CLUSTER_SIZE", NCCL_MAX_CGA_CLUSTER_SIZE + 1}});
  EXPECT_EQ(NCCL_MAX_CGA_CLUSTER_SIZE, c.config().cgaClusterSize);
  EXPECT_TRUE(LogHas(log, "NCCL_CGA_CLUSTER_SIZE value 9 is too big. Limiting value to 8."))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CgaNegative_LeavesConfigUntouched) {
  Env_ConfigComm c;
  c.config().cgaClusterSize = 2;
  const std::string log = c.RunCapturingLog({{"CGA_CLUSTER_SIZE", -1}});
  EXPECT_EQ(2, c.config().cgaClusterSize);
  EXPECT_FALSE(LogHas(log, "cgaClusterSize")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "CGA_CLUSTER_SIZE")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsPositiveConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  c.config().maxCTAs = Env_kAmpleMaxCTAs;
  const std::string log = c.RunCapturingLog({{"MIN_CTAS", 7}});
  EXPECT_EQ(7, c.config().minCTAs);
  EXPECT_EQ(Env_kAmpleMaxCTAs, c.config().maxCTAs);
  EXPECT_FALSE(LogHas(log, "minCTAs reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsPositiveConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().minCTAs = 3;
  c.config().maxCTAs = Env_kAmpleMaxCTAs;
  const std::string log = c.RunCapturingLog({{"MIN_CTAS", 7}});
  EXPECT_EQ(7, c.config().minCTAs);
  EXPECT_TRUE(LogHas(log, "Comm config minCTAs reset to NCCL_MIN_CTAS=7")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsZero_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().minCTAs = 5;
  c.config().maxCTAs = Env_kAmpleMaxCTAs;
  const std::string log = c.RunCapturingLog({{"MIN_CTAS", 0}});
  EXPECT_EQ(5, c.config().minCTAs);
  EXPECT_TRUE(LogHas(log, "NCCL_MIN_CTAS 0 is too low, leaving it set at 5")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsNegative_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().minCTAs = 5;
  c.config().maxCTAs = Env_kAmpleMaxCTAs;
  const std::string log = c.RunCapturingLog({{"MIN_CTAS", -3}});
  EXPECT_EQ(5, c.config().minCTAs);
  EXPECT_TRUE(LogHas(log, "NCCL_MIN_CTAS -3 is too low, leaving it set at 5")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsPositiveConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"MAX_CTAS", 9}});
  EXPECT_EQ(9, c.config().maxCTAs);
  EXPECT_FALSE(LogHas(log, "maxCTAs reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsPositiveConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().maxCTAs = 3;
  const std::string log = c.RunCapturingLog({{"MAX_CTAS", 9}});
  EXPECT_EQ(9, c.config().maxCTAs);
  EXPECT_TRUE(LogHas(log, "Comm config maxCTAs reset to NCCL_MAX_CTAS=9")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsZero_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().maxCTAs = 5;
  const std::string log = c.RunCapturingLog({{"MAX_CTAS", 0}});
  EXPECT_EQ(5, c.config().maxCTAs);
  EXPECT_TRUE(LogHas(log, "NCCL_MAX_CTAS 0 is too low, leaving it set at 5")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsNegative_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().maxCTAs = 5;
  const std::string log = c.RunCapturingLog({{"MAX_CTAS", -3}});
  EXPECT_EQ(5, c.config().maxCTAs);
  EXPECT_TRUE(LogHas(log, "NCCL_MAX_CTAS -3 is too low, leaving it set at 5")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinAndMaxCTAsBothSet_EachTakesItsOwnValue) {
  Env_ConfigComm c;
  c.config().minCTAs = 1;
  c.config().maxCTAs = 2;
  EXPECT_EQ(ncclSuccess, c.Run({{"MIN_CTAS", 6}, {"MAX_CTAS", 11}}));
  EXPECT_EQ(6, c.config().minCTAs);
  EXPECT_EQ(11, c.config().maxCTAs);
}

TEST_F(InitMicrotest, EnvConfigOverride_NChannelsPerNetPeerPositiveConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"NCHANNELS_PER_NET_PEER", 3}});
  EXPECT_EQ(3, c.config().nChannelsPerNetPeer);
  EXPECT_FALSE(LogHas(log, "nChannelsPerNetPeer reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NChannelsPerNetPeerPositiveConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().nChannelsPerNetPeer = 1;
  const std::string log = c.RunCapturingLog({{"NCHANNELS_PER_NET_PEER", 3}});
  EXPECT_EQ(3, c.config().nChannelsPerNetPeer);
  EXPECT_TRUE(LogHas(log, "Comm config nChannelsPerNetPeer reset to NCCL_NCHANNELS_PER_NET_PEER=3"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NChannelsPerNetPeerZero_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().nChannelsPerNetPeer = 2;
  const std::string log = c.RunCapturingLog({{"NCHANNELS_PER_NET_PEER", 0}});
  EXPECT_EQ(2, c.config().nChannelsPerNetPeer);
  EXPECT_TRUE(LogHas(log, "NCCL_NCHANNELS_PER_NET_PEER 0 is too low, leaving it set at 2"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NChannelsPerNetPeerNegative_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().nChannelsPerNetPeer = 2;
  const std::string log = c.RunCapturingLog({{"NCHANNELS_PER_NET_PEER", -5}});
  EXPECT_EQ(2, c.config().nChannelsPerNetPeer);
  EXPECT_TRUE(LogHas(log, "NCCL_NCHANNELS_PER_NET_PEER -5 is too low, leaving it set at 2"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlinkCentricSchedOneConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"NVLINK_UTIL_CENTRIC_SCHED_ENABLE", 1}});
  EXPECT_EQ(1, c.config().nvlinkCentricSched);
  EXPECT_FALSE(LogHas(log, "nvlinkCentricSched reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlinkCentricSchedZeroConfigOne_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().nvlinkCentricSched = 1;
  const std::string log = c.RunCapturingLog({{"NVLINK_UTIL_CENTRIC_SCHED_ENABLE", 0}});
  EXPECT_EQ(0, c.config().nvlinkCentricSched);
  EXPECT_TRUE(LogHas(log, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=0"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlinkCentricSchedOneConfigZero_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().nvlinkCentricSched = 0;
  const std::string log = c.RunCapturingLog({{"NVLINK_UTIL_CENTRIC_SCHED_ENABLE", 1}});
  EXPECT_EQ(1, c.config().nvlinkCentricSched);
  EXPECT_TRUE(LogHas(log, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlinkCentricSchedTwo_KeepsConfigAndLogsNotValid) {
  Env_ConfigComm c;
  c.config().nvlinkCentricSched = 1;
  const std::string log = c.RunCapturingLog({{"NVLINK_UTIL_CENTRIC_SCHED_ENABLE", 2}});
  EXPECT_EQ(1, c.config().nvlinkCentricSched);
  EXPECT_TRUE(LogHas(log, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE 2 is not valid, leaving it set at 1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlinkCentricSchedNegative_KeepsConfigAndLogsNotValid) {
  Env_ConfigComm c;
  c.config().nvlinkCentricSched = 0;
  const std::string log = c.RunCapturingLog({{"NVLINK_UTIL_CENTRIC_SCHED_ENABLE", -1}});
  EXPECT_EQ(0, c.config().nvlinkCentricSched);
  EXPECT_TRUE(LogHas(log, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE -1 is not valid, leaving it set at 0"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphMixingSupportOneConfigUndef_SetsUsageModeTwoSilently) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"GRAPH_MIXING_SUPPORT", 1}});
  EXPECT_EQ(2, c.config().graphUsageMode);
  EXPECT_FALSE(LogHas(log, "graphUsageMode reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphMixingSupportOneConfigSet_SetsUsageModeTwoAndLogsReset) {
  Env_ConfigComm c;
  c.config().graphUsageMode = 7;
  const std::string log = c.RunCapturingLog({{"GRAPH_MIXING_SUPPORT", 1}});
  EXPECT_EQ(2, c.config().graphUsageMode);
  EXPECT_TRUE(LogHas(log, "Comm config graphUsageMode reset to 2 by NCCL_GRAPH_MIXING_SUPPORT=1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphMixingSupportZeroConfigSet_SetsUsageModeZeroAndLogsReset) {
  Env_ConfigComm c;
  c.config().graphUsageMode = 7;
  const std::string log = c.RunCapturingLog({{"GRAPH_MIXING_SUPPORT", 0}});
  EXPECT_EQ(0, c.config().graphUsageMode);
  EXPECT_TRUE(LogHas(log, "Comm config graphUsageMode reset to 0 by NCCL_GRAPH_MIXING_SUPPORT=0"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphMixingSupportInvalid_KeepsUsageModeAndLogsNotValid) {
  Env_ConfigComm c;
  c.config().graphUsageMode = 7;
  const std::string log = c.RunCapturingLog({{"GRAPH_MIXING_SUPPORT", 5}});
  EXPECT_EQ(7, c.config().graphUsageMode);
  EXPECT_TRUE(LogHas(log, "NCCL_GRAPH_MIXING_SUPPORT 5 is not valid, leaving it set at 7"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NumRmaCtxPositive_AssignsWithoutDisabledLog) {
  Env_ConfigComm c;
  c.config().numRmaCtx = 9;
  const std::string log = c.RunCapturingLog({{"NUM_RMA_CTX", 4}});
  EXPECT_EQ(4, c.config().numRmaCtx);
  EXPECT_FALSE(LogHas(log, "RMA disabled")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NumRmaCtxZero_AssignsZeroAndLogsRmaDisabled) {
  Env_ConfigComm c;
  c.config().numRmaCtx = 9;
  const std::string log = c.RunCapturingLog({{"NUM_RMA_CTX", 0}});
  EXPECT_EQ(0, c.config().numRmaCtx);
  EXPECT_TRUE(LogHas(log, "NCCL_NUM_RMA_CTX=0, RMA disabled for this communicator")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NumRmaCtxNegative_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().numRmaCtx = 9;
  const std::string log = c.RunCapturingLog({{"NUM_RMA_CTX", -1}});
  EXPECT_EQ(9, c.config().numRmaCtx);
  EXPECT_TRUE(LogHas(log, "NCCL_NUM_RMA_CTX -1 is too low, leaving it set at 9")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "RMA disabled")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxP2pPeersPositiveConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"P2P_MAX_PEERS", 6}});
  EXPECT_EQ(6, c.config().maxP2pPeers);
  EXPECT_FALSE(LogHas(log, "maxP2pPeers reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxP2pPeersPositiveConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().maxP2pPeers = 2;
  const std::string log = c.RunCapturingLog({{"P2P_MAX_PEERS", 6}});
  EXPECT_EQ(6, c.config().maxP2pPeers);
  EXPECT_TRUE(LogHas(log, "Comm config maxP2pPeers reset to NCCL_MAX_P2P_PEERS=6")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxP2pPeersZero_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().maxP2pPeers = 2;
  const std::string log = c.RunCapturingLog({{"P2P_MAX_PEERS", 0}});
  EXPECT_EQ(2, c.config().maxP2pPeers);
  EXPECT_TRUE(LogHas(log, "NCCL_MAX_P2P_PEERS 0 is too low, leaving it set at 2")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxP2pPeersNegative_KeepsConfigAndLogsTooLow) {
  Env_ConfigComm c;
  c.config().maxP2pPeers = 2;
  const std::string log = c.RunCapturingLog({{"P2P_MAX_PEERS", -4}});
  EXPECT_EQ(2, c.config().maxP2pPeers);
  EXPECT_TRUE(LogHas(log, "NCCL_MAX_P2P_PEERS -4 is too low, leaving it set at 2")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingParamUndefined_LeavesConfigUntouched) {
  Env_ConfigComm c;
  c.config().graphStreamOrdering = 1;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(1, c.config().graphStreamOrdering);
  EXPECT_FALSE(LogHas(log, "GRAPH_STREAM_ORDERING")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingZeroConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"GRAPH_STREAM_ORDERING", 0}});
  EXPECT_EQ(0, c.config().graphStreamOrdering);
  EXPECT_FALSE(LogHas(log, "graphStreamOrdering reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingOneConfigZero_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().graphStreamOrdering = 0;
  const std::string log = c.RunCapturingLog({{"GRAPH_STREAM_ORDERING", 1}});
  EXPECT_EQ(1, c.config().graphStreamOrdering);
  EXPECT_TRUE(LogHas(log, "Comm config graphStreamOrdering reset to NCCL_GRAPH_STREAM_ORDERING=1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingZeroConfigOne_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().graphStreamOrdering = 1;
  const std::string log = c.RunCapturingLog({{"GRAPH_STREAM_ORDERING", 0}});
  EXPECT_EQ(0, c.config().graphStreamOrdering);
  EXPECT_TRUE(LogHas(log, "Comm config graphStreamOrdering reset to NCCL_GRAPH_STREAM_ORDERING=0"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingTwo_KeepsConfigAndLogsNotValid) {
  Env_ConfigComm c;
  c.config().graphStreamOrdering = 1;
  const std::string log = c.RunCapturingLog({{"GRAPH_STREAM_ORDERING", 2}});
  EXPECT_EQ(1, c.config().graphStreamOrdering);
  EXPECT_TRUE(LogHas(log, "NCCL_GRAPH_STREAM_ORDERING 2 is not valid, leaving it set at 1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_GraphStreamOrderingNegative_KeepsConfigAndLogsNotValid) {
  Env_ConfigComm c;
  c.config().graphStreamOrdering = 0;
  const std::string log = c.RunCapturingLog({{"GRAPH_STREAM_ORDERING", -1}});
  EXPECT_EQ(0, c.config().graphStreamOrdering);
  EXPECT_TRUE(LogHas(log, "NCCL_GRAPH_STREAM_ORDERING -1 is not valid, leaving it set at 0"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvUnsetConfigUndef_LeavesNetNameNull) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(nullptr, c.config().netName);
  EXPECT_FALSE(LogHas(log, "netName reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvSetConfigUndef_CopiesEnvValueWithoutResetLog) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_NET", "Socket");
  const std::string log = c.RunCapturingLog({});
  ASSERT_NE(nullptr, c.config().netName);
  EXPECT_STREQ("Socket", c.config().netName);
  EXPECT_FALSE(LogHas(log, "netName reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvRocmIb_TranslatesToIbCast) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_NET", "ROCM-IB");
  EXPECT_EQ(ncclSuccess, c.Run({}));
  ASSERT_NE(nullptr, c.config().netName);
  EXPECT_STREQ("IB-CAST", c.config().netName);
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvRocmIbLowerCase_TranslatesToIbCast) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_NET", "rocm-ib");
  EXPECT_EQ(ncclSuccess, c.Run({}));
  ASSERT_NE(nullptr, c.config().netName);
  EXPECT_STREQ("IB-CAST", c.config().netName);
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvSetConfigSet_ReplacesConfigNameAndLogsReset) {
  Env_ConfigComm c;
  c.config().netName = "IB";
  SetMicroEnv("NCCL_NET", "Socket");
  const std::string log = c.RunCapturingLog({});
  ASSERT_NE(nullptr, c.config().netName);
  EXPECT_STREQ("Socket", c.config().netName);
  EXPECT_TRUE(LogHas(log, "Comm config netName reset to NCCL_NET=Socket")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NetEnvUnsetConfigSet_CopiesIntoFreshBufferAndNeverFreesTheIncumbent) {
  Env_ConfigComm c;
  const char* const configured = "IB";
  c.config().netName = configured;
  int incumbentFrees = 0;
  {
    ScopedHook microFree(g_microFree, [&](void* p) {
      if (p == static_cast<const void*>(configured)) {
        ++incumbentFrees;
        return;
      }
      ::free(p);
    });
    EXPECT_EQ(ncclSuccess, c.Run({}));
  }
  ASSERT_NE(nullptr, c.config().netName);
  EXPECT_STREQ("IB", c.config().netName);
  EXPECT_NE(configured, c.config().netName);
  EXPECT_EQ(0, incumbentFrees);
}

TEST_F(InitMicrotest, EnvConfigOverride_SplitShareConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"COMM_SPLIT_SHARE_RESOURCES", 1}});
  EXPECT_EQ(1, c.config().splitShare);
  EXPECT_FALSE(LogHas(log, "splitShare reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefCollnetLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_SplitShareConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().splitShare = 0;
  const std::string log = c.RunCapturingLog({{"COMM_SPLIT_SHARE_RESOURCES", 1}});
  EXPECT_EQ(1, c.config().splitShare);
  EXPECT_TRUE(LogHas(log, "Comm config splitShare reset to NCCL_COMM_SPLIT_SHARE_RESOURCES=1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_SplitShareOutOfRange_ClampedToZeroWithLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"COMM_SPLIT_SHARE_RESOURCES", 5}});
  EXPECT_EQ(0, c.config().splitShare);
  EXPECT_TRUE(LogHas(log, "splitShare 5 is not a valid value 0/1, set it to 0")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_ShrinkShareConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"COMM_SHRINK_SHARE_RESOURCES", 1}});
  EXPECT_EQ(1, c.config().shrinkShare);
  EXPECT_FALSE(LogHas(log, "shrinkShare reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_ShrinkShareConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().shrinkShare = 0;
  const std::string log = c.RunCapturingLog({{"COMM_SHRINK_SHARE_RESOURCES", 1}});
  EXPECT_EQ(1, c.config().shrinkShare);
  EXPECT_TRUE(LogHas(log, "Comm config shrinkShare reset to NCCL_COMM_SHRINK_SHARE_RESOURCES=1"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_ShrinkShareOutOfRange_KeptVerbatimUnlikeSplitShare) {
  Env_ConfigComm c;
  EXPECT_EQ(ncclSuccess, c.Run({{"COMM_SHRINK_SHARE_RESOURCES", 5}}));
  EXPECT_EQ(5, c.config().shrinkShare);
}

TEST_F(InitMicrotest, EnvConfigOverride_SplitAndShrinkShareBothSet_EachTakesItsOwnValue) {
  Env_ConfigComm c;
  c.config().splitShare = 9;
  c.config().shrinkShare = 9;
  EXPECT_EQ(ncclSuccess, c.Run({{"COMM_SPLIT_SHARE_RESOURCES", 1}, {"COMM_SHRINK_SHARE_RESOURCES", 0}}));
  EXPECT_EQ(1, c.config().splitShare);
  EXPECT_EQ(0, c.config().shrinkShare);
}

TEST_F(InitMicrotest, EnvConfigOverride_CollnetEnableEnvSetConfigUndef_AssignsAndLogsEnvironment) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_COLLNET_ENABLE", "1");
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(1, c.config().collnetEnable);
  EXPECT_TRUE(LogHas(log, "NCCL_COLLNET_ENABLE set by environment to 1.")) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "collnetEnable reset to")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CollnetEnableConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().collnetEnable = 0;
  SetMicroEnv("NCCL_COLLNET_ENABLE", "1");
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(1, c.config().collnetEnable);
  EXPECT_TRUE(LogHas(log, "Comm config collnetEnable reset to NCCL_COLLNET_ENABLE=1")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CollnetEnableHexValue_ParsedBaseZeroThenClampedToZero) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_COLLNET_ENABLE", "0x3");
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(0, c.config().collnetEnable);
  EXPECT_TRUE(LogHas(log, "NCCL_COLLNET_ENABLE set by environment to 3.")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "collnetEnable 3 is not a valid value 0/1, set it to 0")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CollnetEnableParsesToUndefSentinel_LeavesConfigUntouched) {
  Env_ConfigComm c;
  c.config().collnetEnable = 1;
  SetMicroEnv("NCCL_COLLNET_ENABLE", "-2147483648");
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(1, c.config().collnetEnable);
  EXPECT_FALSE(LogHas(log, "NCCL_COLLNET_ENABLE set by environment")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CtaPolicyEnvSetConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  SetMicroEnvAbsent("NCCL_CTA_POLICY");
  ctaPolicyEnv = NCCL_CTA_POLICY_ZERO;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, c.config().CTAPolicy);
  EXPECT_FALSE(LogHas(log, "CTAPolicy reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CtaPolicyEnvSetConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  SetMicroEnvAbsent("NCCL_CTA_POLICY");
  ctaPolicyEnv = NCCL_CTA_POLICY_ZERO;
  c.config().CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, c.config().CTAPolicy);
  const std::string needle =
      "Comm config CTAPolicy reset to NCCL_CTA_POLICY=" + std::to_string(NCCL_CTA_POLICY_ZERO);
  EXPECT_TRUE(LogHas(log, needle.c_str())) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CtaPolicyZeroAndEfficiency_UnsetsEfficiencyWithWarn) {
  Env_ConfigComm c;
  SetMicroEnvAbsent("NCCL_CTA_POLICY");
  ctaPolicyEnv = NCCL_CTA_POLICY_ZERO | NCCL_CTA_POLICY_EFFICIENCY;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, c.config().CTAPolicy);
  EXPECT_TRUE(LogHas(log, "Unsetting POLICY_EFFICIENCY")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlsCTAsPositiveConfigUndef_AssignsWithoutResetLog) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"NVLS_NCHANNELS", 4}});
  EXPECT_EQ(4, c.config().nvlsCTAs);
  EXPECT_FALSE(LogHas(log, "nvlsCTAs reset to")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlsCTAsPositiveConfigSet_OverwritesAndLogsReset) {
  Env_ConfigComm c;
  c.config().nvlsCTAs = 2;
  const std::string log = c.RunCapturingLog({{"NVLS_NCHANNELS", 4}});
  EXPECT_EQ(4, c.config().nvlsCTAs);
  EXPECT_TRUE(LogHas(log, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=4")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlsCTAsZero_AssignedThenRestoredToUndefined) {
  Env_ConfigComm c;
  c.config().nvlsCTAs = 2;
  const std::string log = c.RunCapturingLog({{"NVLS_NCHANNELS", 0}});
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, c.config().nvlsCTAs);
  EXPECT_TRUE(LogHas(log, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=0")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, "nvlsCTAs 0 is not a valid value")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NvlsCTAsNegative_AssignedThenRestoredToUndefined) {
  Env_ConfigComm c;
  c.config().nvlsCTAs = 2;
  const std::string log = c.RunCapturingLog({{"NVLS_NCHANNELS", -1}});
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, c.config().nvlsCTAs);
  EXPECT_TRUE(LogHas(log, "nvlsCTAs -1 is not a valid value")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsAboveChannelLimit_CappedToMaxChannels) {
  Env_ConfigComm c;
  c.config().minCTAs = MAXCHANNELS + 1;
  c.config().maxCTAs = MAXCHANNELS;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(MAXCHANNELS, c.config().minCTAs);
  EXPECT_EQ(MAXCHANNELS, c.config().maxCTAs);
  const std::string needle = "minCTAs " + std::to_string(MAXCHANNELS + 1) +
                             " is larger than #channels upper limit " + std::to_string(MAXCHANNELS) +
                             ", cap it to " + std::to_string(MAXCHANNELS);
  EXPECT_TRUE(LogHas(log, needle.c_str())) << "actual log:\n" << log;
  EXPECT_FALSE(LogHas(log, "is larger than maxCTAs")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsAtChannelLimit_NotCapped) {
  Env_ConfigComm c;
  c.config().minCTAs = MAXCHANNELS;
  c.config().maxCTAs = MAXCHANNELS;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(MAXCHANNELS, c.config().minCTAs);
  EXPECT_FALSE(LogHas(log, "#channels upper limit")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsAboveChannelLimit_CappedToMaxChannels) {
  Env_ConfigComm c;
  c.config().maxCTAs = MAXCHANNELS + 1;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(MAXCHANNELS, c.config().maxCTAs);
  const std::string needle = "maxCTAs " + std::to_string(MAXCHANNELS + 1) +
                             " is larger than #channels upper limit " + std::to_string(MAXCHANNELS) +
                             ", cap it to " + std::to_string(MAXCHANNELS);
  EXPECT_TRUE(LogHas(log, needle.c_str())) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MaxCTAsAtChannelLimit_NotCapped) {
  Env_ConfigComm c;
  c.config().maxCTAs = MAXCHANNELS;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(MAXCHANNELS, c.config().maxCTAs);
  EXPECT_FALSE(LogHas(log, "#channels upper limit")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsAboveMaxCTAs_LowersMinToMax) {
  Env_ConfigComm c;
  c.config().minCTAs = 8;
  c.config().maxCTAs = 4;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(4, c.config().minCTAs);
  EXPECT_EQ(4, c.config().maxCTAs);
  EXPECT_TRUE(LogHas(log, "minCTAs 8 is larger than maxCTAs 4, set both to 4")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsEqualsMaxCTAs_LeavesBothAlone) {
  Env_ConfigComm c;
  c.config().minCTAs = 4;
  c.config().maxCTAs = 4;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(4, c.config().minCTAs);
  EXPECT_EQ(4, c.config().maxCTAs);
  EXPECT_FALSE(LogHas(log, "is larger than maxCTAs")) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, Env_kUndefSplitShareLog)) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsAboveBothLimits_CapsToChannelLimitBeforeLoweringToMaxCTAs) {
  Env_ConfigComm c;
  c.config().minCTAs = MAXCHANNELS + 1;
  c.config().maxCTAs = 4;
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(4, c.config().minCTAs);
  EXPECT_EQ(4, c.config().maxCTAs);
  const std::string capped = "minCTAs " + std::to_string(MAXCHANNELS + 1) +
                             " is larger than #channels upper limit " + std::to_string(MAXCHANNELS);
  const std::string lowered = "minCTAs " + std::to_string(MAXCHANNELS) + " is larger than maxCTAs 4, set both to 4";
  EXPECT_TRUE(LogHas(log, capped.c_str())) << "actual log:\n" << log;
  EXPECT_TRUE(LogHas(log, lowered.c_str())) << "actual log:\n" << log;
}

// A MIN_CTAS override does apply, then :3183 lowers it to the still-undefined maxCTAs (AICOMRCCL-1685 TODO at :1163).
TEST_F(InitMicrotest, EnvConfigOverride_MinCTAsSetWithUndefinedMaxCTAs_LoweredBackToUndefined) {
  Env_ConfigComm c;
  const std::string log = c.RunCapturingLog({{"MIN_CTAS", 7}});
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, c.config().minCTAs);
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, c.config().maxCTAs);
  EXPECT_TRUE(LogHas(log, "minCTAs 7 is larger than maxCTAs -2147483648, set both to -2147483648"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_NoCheckEnv_ResetsCheckModeToDefault) {
  Env_ConfigComm c;
  c.comm()->checkMode = ncclCheckModeDebugGlobal;
  c.RunCapturingLog({});
  EXPECT_EQ(ncclCheckModeDefault, c.comm()->checkMode);
}

TEST_F(InitMicrotest, EnvConfigOverride_DeprecatedCheckPointers_SelectsDebugLocal) {
  Env_ConfigComm c;
  c.RunCapturingLog({{"CHECK_POINTERS", 1}});
  EXPECT_EQ(ncclCheckModeDebugLocal, c.comm()->checkMode);
}

TEST_F(InitMicrotest, EnvConfigOverride_CheckPointersNotOne_LeavesCheckModeDefault) {
  Env_ConfigComm c;
  c.RunCapturingLog({{"CHECK_POINTERS", 2}});
  EXPECT_EQ(ncclCheckModeDefault, c.comm()->checkMode);
}

TEST_F(InitMicrotest, EnvConfigOverride_CheckModeDebugGlobal_SelectsDebugGlobalCaseInsensitively) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_CHECK_MODE", "debug_global");
  const std::string log = c.RunCapturingLog({});
  EXPECT_EQ(ncclCheckModeDebugGlobal, c.comm()->checkMode);
  EXPECT_TRUE(LogHas(log, "NCCL_CHECK_MODE set by environment to debug_global")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, EnvConfigOverride_CheckModeDebugLocalOverridesCheckPointers_SelectsDebugLocal) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_CHECK_MODE", "DEBUG_LOCAL");
  c.RunCapturingLog({{"CHECK_POINTERS", 1}});
  EXPECT_EQ(ncclCheckModeDebugLocal, c.comm()->checkMode);
}

TEST_F(InitMicrotest, EnvConfigOverride_CheckModeUnrecognised_KeepsTheCheckPointersChoice) {
  Env_ConfigComm c;
  SetMicroEnv("NCCL_CHECK_MODE", "DEBUG_GLOBALLY");
  c.RunCapturingLog({{"CHECK_POINTERS", 1}});
  EXPECT_EQ(ncclCheckModeDebugLocal, c.comm()->checkMode);
}

// --- ncclCommInitRankFunc (init.cc:2631) and the entry points that funnel into it ---

namespace {

constexpr int Rank_kCudaDev = 5;
constexpr int Rank_kArchMajor = 9;
constexpr int Rank_kArchMinor = 4;
constexpr int Rank_kCudaArch = 100 * Rank_kArchMajor + 10 * Rank_kArchMinor;
constexpr int Rank_kMaxSharedMem = 65536;
constexpr int Rank_kCuCount = 304;
constexpr int Rank_kUntouchedNRanks = 77;
constexpr char Rank_kGcnArchName[] = "gfx942:sramecc+:xnack-";

// Distinct per-attribute values: with major == minor the 100* / 10* operands could be swapped unnoticed.
hipError_t Rank_DeviceAttribute(int* pi, hipDeviceAttribute_t attr, int) {
  if (!pi) {
    return hipErrorInvalidValue;
  }
  switch (attr) {
    case hipDeviceAttributeSharedMemPerBlockOptin: *pi = Rank_kMaxSharedMem; break;
    case hipDeviceAttributeComputeCapabilityMajor: *pi = Rank_kArchMajor; break;
    case hipDeviceAttributeComputeCapabilityMinor: *pi = Rank_kArchMinor; break;
    case hipDeviceAttributeWarpSize: *pi = g_hipWarpSize; break;
    case hipDeviceAttributeDirectManagedMemAccessFromHost: *pi = 1; break;
    default: *pi = 0; break;
  }
  return hipSuccess;
}

hipError_t Rank_DeviceProperties(hipDeviceProp_t* prop, int) {
  if (!prop) {
    return hipErrorInvalidValue;
  }
  *prop = hipDeviceProp_t{};
  std::snprintf(prop->gcnArchName, sizeof(prop->gcnArchName), "%s", Rank_kGcnArchName);
  prop->multiProcessorCount = Rank_kCuCount;
  prop->warpSize = 64;
  return hipSuccess;
}

hipError_t Rank_SetDeviceOk(int) { return hipSuccess; }

ncclResult_t Rank_KernelInitOk(int, int, size_t* maxStackSize) {
  if (maxStackSize) {
    *maxStackSize = 0;
  }
  return ncclSuccess;
}

// Heap comm and job: channels[MAXCHANNELS] is inline, so a stack ncclComm overflows.
class Rank_JobScene {
 public:
  Rank_JobScene(int nranks, int myrank)
      : comm_(new ncclComm{}),
        job_(new ncclCommInitRankAsyncJob{}),
        attr_(g_hipDeviceGetAttribute, Rank_DeviceAttribute),
        props_(g_hipGetDeviceProperties, Rank_DeviceProperties) {
    InstallCommAllocSuccess();
    job_->comm = comm_.get();
    job_->nranks = nranks;
    job_->myrank = myrank;
    job_->nId = 1;
    job_->cudaDev = Rank_kCudaDev;
    job_->newcomm = &published_;
    std::snprintf(job_->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", "Rank_JobScene");
  }
  ~Rank_JobScene() {
    free(comm_->archName);
    free(comm_->peerInfo);
  }
  Rank_JobScene(const Rank_JobScene&) = delete;
  Rank_JobScene& operator=(const Rank_JobScene&) = delete;

  ncclComm* comm() { return comm_.get(); }
  ncclCommInitRankAsyncJob* job() { return job_.get(); }
  ncclComm* published() const { return published_; }
  ncclResult_t Run() { return ncclCommInitRankFunc(reinterpret_cast<ncclAsyncJob*>(job_.get())); }

 private:
  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclCommInitRankAsyncJob> job_;
  ncclComm* published_ = nullptr;
  ScopedHook<hipError_t(int*, hipDeviceAttribute_t, int)> attr_;
  ScopedHook<hipError_t(hipDeviceProp_t*, int)> props_;
};

}  // namespace

TEST_F(InitMicrotest, CommInitRankFunc_SetDeviceFails_PublishesTheErrorAsInitStateAndStillAssignsNewcomm) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  int seenDev = -1;
  ScopedHook setDevice(g_hipSetDevice, [&](int dev) {
    seenDev = dev;
    return hipErrorInvalidDevice;
  });

  EXPECT_EQ(ncclUnhandledCudaError, s.Run());
  EXPECT_EQ(Rank_kCudaDev, seenDev);
  EXPECT_EQ(1, setDevice.calls);
  EXPECT_EQ(ncclUnhandledCudaError, s.comm()->initState);
  EXPECT_EQ(s.comm(), s.published()) << "exit: publishes the comm even when init failed";
}

TEST_F(InitMicrotest, CommInitRankFunc_SharedMemAttributeFails_ReturnsBeforeReadingTheArch) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook attr(g_hipDeviceGetAttribute, [](int* pi, hipDeviceAttribute_t, int) {
    if (pi) {
      *pi = 0;
    }
    return hipErrorInvalidValue;
  });

  EXPECT_EQ(ncclUnhandledCudaError, s.Run());
  EXPECT_EQ(1, attr.calls) << "the two arch reads must not run after the shared-mem read failed";
  EXPECT_EQ(ncclUnhandledCudaError, s.comm()->initState);
}

TEST_F(InitMicrotest, CommInitRankFunc_DevicePropertiesFail_ReturnsBeforeKernelInit) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook props(g_hipGetDeviceProperties, [](hipDeviceProp_t*, int) { return hipErrorInvalidValue; });
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclUnhandledCudaError, s.Run());
  EXPECT_EQ(0, kernels.calls);
  EXPECT_EQ(nullptr, s.comm()->archName);
  EXPECT_EQ(0, s.comm()->cuCount);
}

TEST_F(InitMicrotest, CommInitRankFunc_KernelInitFails_ForwardsArchAndSharedMemThenReturnsWithoutPublishing) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  int seenArch = -1;
  int seenSharedMem = -1;
  ScopedHook kernels(g_ncclInitKernelsForDevice, [&](int arch, int sharedMem, size_t* out) {
    seenArch = arch;
    seenSharedMem = sharedMem;
    if (out) {
      *out = 0;
    }
    return ncclInternalError;
  });

  EXPECT_EQ(ncclInternalError, s.Run());
  EXPECT_EQ(Rank_kCudaArch, seenArch);
  EXPECT_EQ(Rank_kMaxSharedMem, seenSharedMem);
  EXPECT_EQ(1, kernels.calls);
  EXPECT_EQ(nullptr, s.published()) << "a bare NCCLCHECK returns directly, skipping the exit: publish";
  EXPECT_EQ(ncclSuccess, s.comm()->initState) << "and skipping the fail: initState store";
}

TEST_F(InitMicrotest, CommInitRankFunc_KernelsWantStackSpace_RaisesTheDeviceStackLimitToThatSize) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  const size_t kMaxLocalSizeBytes = 3072;
  SetParams({{"SET_STACK_SIZE", 1}});
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, [kMaxLocalSizeBytes](int, int, size_t* out) {
    if (out) {
      *out = kMaxLocalSizeBytes;
    }
    return ncclSuccess;
  });
  size_t seenLimit = 0;
  ScopedHook setLimit(g_hipDeviceSetLimit, [&](hipLimit_t limit, size_t value) {
    if (limit == hipLimitStackSize) {
      seenLimit = value;
    }
    return hipSuccess;
  });
  ScopedHook bootInit(g_bootstrapInit, [](int, void*, ncclComm*, ncclComm*) { return ncclRemoteError; });
  ncclUniqueId id{};
  s.job()->commId = &id;

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(kMaxLocalSizeBytes, seenLimit);
  EXPECT_EQ(1, setLimit.calls);
}

TEST_F(InitMicrotest, CommInitRankFunc_KernelsWantNoStackSpace_LeavesTheDeviceStackLimitAlone) {
  Rank_JobScene s(/*nranks=*/4, /*myrank=*/1);
  SetParams({{"SET_STACK_SIZE", 1}});
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);
  ScopedHook setLimit(g_hipDeviceSetLimit, [](hipLimit_t, size_t) { return hipSuccess; });
  ScopedHook bootInit(g_bootstrapInit, [](int, void*, ncclComm*, ncclComm*) { return ncclRemoteError; });
  ncclUniqueId id{};
  s.job()->commId = &id;

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(0, setLimit.calls);
}

TEST_F(InitMicrotest, CommInitRankFunc_SplitNoColor_ReturnsSuccessWithoutAllocatingTheChild) {
  Rank_JobScene s(/*nranks=*/0, /*myrank=*/0);
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  s.job()->parent = parent.get();
  s.job()->color = NCCL_SPLIT_NOCOLOR;
  s.comm()->nRanks = Rank_kUntouchedNRanks;
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {NCCL_SPLIT_NOCOLOR, 0}, {7, 2}, {7, 3}});
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclSuccess, s.Run());
  EXPECT_EQ(Rank_kUntouchedNRanks, s.comm()->nRanks) << "commAlloc must not run for NCCL_SPLIT_NOCOLOR";
  EXPECT_EQ(nullptr, s.comm()->archName);
  EXPECT_EQ(0u, s.comm()->commHash);
  EXPECT_EQ(s.comm(), s.published());
}

TEST_F(InitMicrotest, CommInitRankFunc_SplitWithColor_DerivesChildHashAndHandsItToBootstrapSplit) {
  Rank_JobScene s(/*nranks=*/0, /*myrank=*/0);
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  parent->commHash = 0xABCDEF0123456789ULL;
  s.job()->parent = parent.get();
  s.job()->childCount = 3;
  s.job()->color = 7;
  s.job()->key = 2;
  s.comm()->isGrow = true;
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {7, 2}, {7, 4}, {7, 6}});

  uint64_t hacc[2] = {1, 1};
  eatHash(hacc, &parent->commHash);
  eatHash(hacc, &s.job()->childCount);
  eatHash(hacc, &s.job()->color);
  const uint64_t expectedHash = digestHash(hacc);

  uint64_t seenHash = 0;
  ncclComm* seenComm = nullptr;
  ncclComm* seenParent = nullptr;
  int seenColor = -1;
  int seenKey = -1;
  std::vector<int> seenRanks;
  ScopedHook split(g_bootstrapSplit,
                   [&](uint64_t hash, ncclComm* comm, ncclComm* parentComm, int color, int key, int* ranks) {
                     seenHash = hash;
                     seenComm = comm;
                     seenParent = parentComm;
                     seenColor = color;
                     seenKey = key;
                     seenRanks.assign(ranks, ranks + 4);
                     return ncclRemoteError;
                   });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(expectedHash, s.comm()->commHash);
  EXPECT_EQ(expectedHash, seenHash);
  EXPECT_EQ(s.comm(), seenComm);
  EXPECT_EQ(parent.get(), seenParent);
  EXPECT_EQ(7, seenColor);
  EXPECT_EQ(2, seenKey);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), seenRanks);
  EXPECT_EQ(4, s.job()->nranks);
  EXPECT_EQ(1, s.job()->myrank);
  EXPECT_EQ(4, s.comm()->nRanks);
  EXPECT_EQ(1, s.comm()->rank);
  EXPECT_FALSE(s.comm()->isGrow);
  EXPECT_EQ(ncclRemoteError, s.comm()->initState);
}

TEST_F(InitMicrotest, CommInitRankFunc_SplitBootstrapSucceeds_CarriesOnIntoTheSharedArchBringup) {
  Rank_JobScene s(/*nranks=*/0, /*myrank=*/0);
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/1);
  s.job()->parent = parent.get();
  s.job()->color = 7;
  s.job()->key = 2;
  InstallSplitInfoTable(/*selfRank=*/1, {{7, 0}, {7, 2}, {7, 4}, {7, 6}});
  ScopedHook split(g_bootstrapSplit, [](uint64_t, ncclComm*, ncclComm*, int, int, int*) {
    g_bootstrapAllGather = [](void*, void*, int) { return ncclInternalError; };
    return ncclSuccess;
  });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_NE(ncclSuccess, s.Run()) << "initTransportsRank cannot complete host-only";
  EXPECT_EQ(1, split.calls);
  EXPECT_EQ(Rank_kCudaArch, s.comm()->cudaArch) << "the split arm reaches the same arch bringup as a normal init";
  EXPECT_STREQ(Rank_kGcnArchName, s.comm()->archName);
}

TEST_F(InitMicrotest, CommInitRankFunc_ShrinkWithExcludeList_SkipsTheSplitInfoAllGather) {
  Rank_JobScene s(/*nranks=*/0, /*myrank=*/0);
  auto parent = MakeParentComm(/*nRanks=*/5, /*rank=*/4);
  s.job()->parent = parent.get();
  int exclude[2] = {1, 3};
  s.job()->excludeRanksList = exclude;
  s.job()->excludeRanksCount = 2;
  ScopedHook allGather(g_bootstrapAllGather, [](void*, void*, int) {
    ADD_FAILURE() << "the shrink arm must not consult commGetSplitInfo";
    return ncclInternalError;
  });
  std::vector<int> seenRanks;
  ScopedHook split(g_bootstrapSplit, [&](uint64_t, ncclComm*, ncclComm*, int, int, int* ranks) {
    seenRanks.assign(ranks, ranks + 3);
    return ncclRemoteError;
  });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(0, allGather.calls);
  EXPECT_EQ(3, s.job()->nranks);
  EXPECT_EQ(2, s.job()->myrank) << "rank 4 is the third survivor of {0,2,4}";
  EXPECT_EQ(std::vector<int>({0, 2, 4}), seenRanks);
  EXPECT_EQ(3, s.comm()->nRanks);
  EXPECT_EQ(2, s.comm()->rank);
}

TEST_F(InitMicrotest, CommInitRankFunc_NormalInit_DerivesCommHashFromTheCommIdAndClearsIsGrow) {
  Rank_JobScene s(/*nranks=*/8, /*myrank=*/3);
  ncclUniqueId id{};
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    id.internal[i] = static_cast<char>(i + 1);
  }
  s.job()->commId = &id;
  s.job()->nId = 2;
  s.comm()->isGrow = true;
  const uint64_t expectedHash = getHash(id.internal, NCCL_UNIQUE_ID_BYTES);

  int seenHandles = -1;
  void* seenHandle = nullptr;
  ncclComm* seenComm = nullptr;
  ncclComm* seenParent = s.comm();
  ScopedHook bootInit(g_bootstrapInit, [&](int nHandles, void* handle, ncclComm* comm, ncclComm* parent) {
    seenHandles = nHandles;
    seenHandle = handle;
    seenComm = comm;
    seenParent = parent;
    return ncclRemoteError;
  });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(expectedHash, s.comm()->commHash);
  EXPECT_FALSE(s.comm()->isGrow);
  EXPECT_EQ(2, seenHandles);
  EXPECT_EQ(static_cast<void*>(&id), seenHandle);
  EXPECT_EQ(s.comm(), seenComm);
  EXPECT_EQ(nullptr, seenParent) << "the normal arm allocates and bootstraps with no parent";
  EXPECT_EQ(8, s.comm()->nRanks);
  EXPECT_EQ(3, s.comm()->rank);
  EXPECT_EQ(ncclRemoteError, s.comm()->initState);
}

TEST_F(InitMicrotest, CommInitRankFunc_Grow_DerivesCommHashFromTheHandleMagicAndNewRankCount) {
  Rank_JobScene s(/*nranks=*/6, /*myrank=*/2);
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/0);
  parent->magic = 0xFEEDFACEULL;
  parent->childCount = 9;
  s.job()->parent = parent.get();
  s.job()->isGrow = true;
  ncclBootstrapHandle handle{};
  handle.magic = 0x5A5A12345678ULL;
  s.job()->commId = reinterpret_cast<ncclUniqueId*>(&handle);
  const uint64_t expectedHash = hashCombine(handle.magic, s.job()->nranks);

  ScopedHook bootInit(g_bootstrapInit, [](int, void*, ncclComm*, ncclComm*) { return ncclRemoteError; });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(expectedHash, s.comm()->commHash);
  EXPECT_TRUE(s.comm()->isGrow);
  EXPECT_EQ(6, s.comm()->nRanks);
  EXPECT_EQ(2, s.comm()->rank);
}

TEST_F(InitMicrotest, CommInitRankFunc_GrowWithoutHandle_FallsBackToTheParentMagicAndChildCount) {
  Rank_JobScene s(/*nranks=*/6, /*myrank=*/2);
  auto parent = MakeParentComm(/*nRanks=*/4, /*rank=*/0);
  parent->magic = 0xFEEDFACEULL;
  parent->childCount = 9;
  s.job()->parent = parent.get();
  s.job()->isGrow = true;
  s.job()->commId = nullptr;
  const uint64_t expectedHash = hashCombine(hashCombine(parent->magic, parent->childCount), s.job()->nranks);

  ScopedHook bootInit(g_bootstrapInit, [](int, void*, ncclComm*, ncclComm*) { return ncclRemoteError; });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  EXPECT_EQ(ncclRemoteError, s.Run());
  EXPECT_EQ(expectedHash, s.comm()->commHash);
  EXPECT_NE(hashCombine(parent->magic, s.job()->nranks), s.comm()->commHash)
      << "the childCount must take part in the fallback base magic";
}

TEST_F(InitMicrotest, CommInitRankFunc_TransportInitFails_StampsTheArchFieldsAndLeavesArchNameOwnedByTheComm) {
  Rank_JobScene s(/*nranks=*/8, /*myrank=*/3);
  ncclUniqueId id{};
  s.job()->commId = &id;
  ScopedHook bootInit(g_bootstrapInit, [](int, void*, ncclComm*, ncclComm*) { return ncclSuccess; });
  ScopedHook setDevice(g_hipSetDevice, Rank_SetDeviceOk);
  ScopedHook kernels(g_ncclInitKernelsForDevice, Rank_KernelInitOk);

  const ncclResult_t res = s.Run();
  EXPECT_NE(ncclSuccess, res) << "initTransportsRank cannot complete host-only";
  EXPECT_EQ(Rank_kCudaArch, s.comm()->cudaArch);
  EXPECT_EQ(Rank_kCuCount, s.comm()->cuCount);
  ASSERT_NE(nullptr, s.comm()->archName);
  EXPECT_STREQ(Rank_kGcnArchName, s.comm()->archName);
  EXPECT_EQ(rcclLL128LineElemsFromArch(Rank_kGcnArchName), s.comm()->ll128LineElems);
  EXPECT_EQ(rcclLL128DataElemsFromArch(Rank_kGcnArchName), s.comm()->ll128DataElems);
  EXPECT_EQ(res, s.comm()->initState);
}

namespace {

constexpr int Rank_kEntryNRanks = 8;
constexpr int Rank_kEntryMyRank = 3;
constexpr int Rank_kEntryCudaDev = 6;
constexpr int Rank_kEntryMinCTAs = 5;

void Rank_FillUniqueId(ncclUniqueId* id, unsigned char seed) {
  for (int i = 0; i < NCCL_UNIQUE_ID_BYTES; ++i) {
    id->internal[i] = static_cast<char>(seed + i);
  }
}

ncclCommInitRankAsyncJob* Rank_LaunchedJob(const Teardown_LaunchRecord& rec) {
  return reinterpret_cast<ncclCommInitRankAsyncJob*>(rec.job);
}

// The suppressed launch never runs commFree, so a success-path test owns everything the entry point allocated.
void Rank_ReleaseComm(ncclComm* c, bool ownsAbortResources = true) {
  if (!c) {
    return;
  }
  if (ownsAbortResources) {
    ::free(c->abortFlag);
    ::free((void*)c->abortFlagDev);
    ::free(c->abortFlagRefCount);
  }
  ::free((void*)c->config.netName);
  ::free(c);
}

}  // namespace

TEST_F(InitMicrotest, CommInitRankDev_HappyPath_LaunchesRankFuncWithTheJobItBuilt) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  Rank_FillUniqueId(&id, 0xA1);
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclSuccess, ncclCommInitRankDev(&nc, Rank_kEntryNRanks, /*nId=*/1, &id, Rank_kEntryMyRank,
                                             Rank_kEntryCudaDev, &cfg, "Rank_Dev"));
  ASSERT_NE(nullptr, nc);
  EXPECT_EQ(ncclInProgress, nc->initState);
  ASSERT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(ncclCommInitRankFunc), rec.func);
  EXPECT_EQ(ncclCommInitJobFree, rec.destructor);
  EXPECT_EQ(nc, rec.comm);
  ncclCommInitRankAsyncJob* job = Rank_LaunchedJob(rec);
  EXPECT_EQ(nc, job->comm);
  EXPECT_EQ(Rank_kEntryNRanks, job->nranks);
  EXPECT_EQ(Rank_kEntryMyRank, job->myrank);
  EXPECT_EQ(Rank_kEntryCudaDev, job->cudaDev);
  EXPECT_EQ(1, job->nId);
  EXPECT_STREQ("Rank_Dev", job->funcName);
  ASSERT_NE(nullptr, job->commId);
  EXPECT_NE(static_cast<const void*>(job->commId), static_cast<const void*>(&id)) << "the id must be copied";
  EXPECT_EQ(0, std::memcmp(job->commId, &id, NCCL_UNIQUE_ID_BYTES));
  rec.Release();
  Rank_ReleaseComm(nc);
}

TEST_F(InitMicrotest, CommInitRankDev_LaunchFails_ClearsNewcommAndReportsTheError) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  rec.result = ncclSystemError;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = reinterpret_cast<ncclComm_t>(0x1);
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclSystemError, ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/1, &id, /*myrank=*/0, 0, &cfg, "t"));
  EXPECT_EQ(nullptr, nc) << "a failed init must not hand back a comm";
  EXPECT_EQ(1, rec.calls);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankDev_CommIdEnvOnRankZero_ClampsNIdToOneAndStartsTheBootstrapRoot) {
  InstallDevCommSetupSuccess();
  SetMicroEnv("NCCL_COMM_ID", "127.0.0.1:23456");
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  bool seenIdFromEnv = false;
  ScopedHook root(g_bootstrapCreateRoot, [&](ncclBootstrapHandle*, bool idFromEnv) {
    seenIdFromEnv = idFromEnv;
    return ncclSuccess;
  });
  ncclComm_t nc = nullptr;
  ncclUniqueId ids[2] = {};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclSuccess, ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/2, ids, /*myrank=*/0, 0, &cfg, "t"));
  ASSERT_EQ(1, rec.calls);
  EXPECT_EQ(1, Rank_LaunchedJob(rec)->nId) << "NCCL_COMM_ID cannot be combined with several unique ids";
  EXPECT_EQ(1, root.calls);
  EXPECT_TRUE(seenIdFromEnv);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankDev_CommIdEnvOnNonZeroRank_KeepsNIdAndStartsNoRoot) {
  InstallDevCommSetupSuccess();
  SetMicroEnv("NCCL_COMM_ID", "127.0.0.1:23456");
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ScopedHook root(g_bootstrapCreateRoot, [](ncclBootstrapHandle*, bool) { return ncclSuccess; });
  ncclComm_t nc = nullptr;
  ncclUniqueId ids[2] = {};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclSuccess, ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/2, ids, /*myrank=*/1, 0, &cfg, "t"));
  ASSERT_EQ(1, rec.calls);
  EXPECT_EQ(2, Rank_LaunchedJob(rec)->nId);
  EXPECT_EQ(0, root.calls) << "only rank 0 starts the bootstrap root";
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankDev_BootstrapRootFails_ClearsNewcommAndSkipsTheLaunch) {
  InstallDevCommSetupSuccess();
  SetMicroEnv("NCCL_COMM_ID", "127.0.0.1:23456");
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ScopedHook root(g_bootstrapCreateRoot, [](ncclBootstrapHandle*, bool) { return ncclRemoteError; });
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclRemoteError, ncclCommInitRankDev(&nc, /*nranks=*/4, /*nId=*/1, &id, /*myrank=*/0, 0, &cfg, "t"));
  EXPECT_EQ(0, rec.calls);
  EXPECT_EQ(nullptr, nc);
}

TEST_F(InitMicrotest, CommInitRank_ForwardsOneIdAndTheCurrentDeviceToRankDev) {
  InstallDevCommSetupSuccess();
  const int currentDev = 4;
  ScopedHook getDevice(g_hipGetDevice, [currentDev](int* dev) {
    if (dev) {
      *dev = currentDev;
    }
    return hipSuccess;
  });
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  Rank_FillUniqueId(&id, 0xB2);

  EXPECT_EQ(ncclSuccess, ncclCommInitRank_impl(&nc, Rank_kEntryNRanks, id, Rank_kEntryMyRank));
  ASSERT_EQ(1, rec.calls);
  ncclCommInitRankAsyncJob* job = Rank_LaunchedJob(rec);
  EXPECT_EQ(Rank_kEntryNRanks, job->nranks);
  EXPECT_EQ(Rank_kEntryMyRank, job->myrank);
  EXPECT_EQ(1, job->nId);
  EXPECT_EQ(currentDev, job->cudaDev);
  EXPECT_EQ(0, std::memcmp(job->commId, &id, NCCL_UNIQUE_ID_BYTES));
  EXPECT_STREQ("ncclCommInitRank_impl", job->funcName);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankConfig_NullConfig_StillLaunchesWithTheInternalDefaultConfig) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};

  EXPECT_EQ(ncclSuccess, ncclCommInitRankConfig_impl(&nc, Rank_kEntryNRanks, id, Rank_kEntryMyRank,
                                                     /*config=*/nullptr));
  ASSERT_NE(nullptr, nc);
  ASSERT_EQ(1, rec.calls);
  EXPECT_NE(Rank_kEntryMinCTAs, nc->config.minCTAs) << "no user config was supplied to copy from";
  EXPECT_EQ(Rank_kEntryMyRank, Rank_LaunchedJob(rec)->myrank);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankConfig_UserConfig_IsTheConfigParsedIntoTheComm) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.minCTAs = Rank_kEntryMinCTAs;
  cfg.maxCTAs = Rank_kEntryMinCTAs;

  EXPECT_EQ(ncclSuccess, ncclCommInitRankConfig_impl(&nc, Rank_kEntryNRanks, id, Rank_kEntryMyRank, &cfg));
  ASSERT_NE(nullptr, nc);
  EXPECT_EQ(Rank_kEntryMinCTAs, nc->config.minCTAs);
  EXPECT_EQ(1, rec.calls);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankConfig_NonBlockingComm_ReportsTheCommsAsyncErrorInsteadOfTheReturnCode) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, [&rec](ncclAsyncJob* job, ncclResult_t (*func)(ncclAsyncJob*),
                                              void (*)(ncclAsyncJob*), void (*destructor)(void*), ncclComm* comm) {
    rec.calls++;
    rec.job = job;
    rec.func = func;
    rec.destructor = destructor;
    rec.comm = comm;
    comm->asyncResult = ncclInProgress;
    return ncclSuccess;
  });
  ncclComm_t nc = nullptr;
  ncclUniqueId id{};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.blocking = 0;

  EXPECT_EQ(ncclInProgress, ncclCommInitRankConfig_impl(&nc, Rank_kEntryNRanks, id, Rank_kEntryMyRank, &cfg));
  ASSERT_NE(nullptr, nc);
  EXPECT_EQ(0, nc->config.blocking);
  EXPECT_EQ(1, rec.calls);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankScalable_ForwardsNIdAndTheWholeCommIdArray) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  const int nId = 3;
  ncclUniqueId ids[nId] = {};
  for (int i = 0; i < nId; ++i) {
    Rank_FillUniqueId(&ids[i], static_cast<unsigned char>(0x21 + i));
  }
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclSuccess,
            ncclCommInitRankScalable(&nc, Rank_kEntryNRanks, Rank_kEntryMyRank, nId, ids, &cfg));
  ASSERT_EQ(1, rec.calls);
  ncclCommInitRankAsyncJob* job = Rank_LaunchedJob(rec);
  EXPECT_EQ(nId, job->nId);
  EXPECT_EQ(Rank_kEntryNRanks, job->nranks);
  EXPECT_EQ(Rank_kEntryMyRank, job->myrank);
  EXPECT_EQ(0, std::memcmp(job->commId, ids, nId * NCCL_UNIQUE_ID_BYTES));
  EXPECT_STREQ("ncclCommInitRankScalable", job->funcName);
  rec.Release();
}

TEST_F(InitMicrotest, CommInitRankScalable_NIdAboveNranks_ReturnsInvalidArgumentWithoutLaunching) {
  InstallDevCommSetupSuccess();
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t nc = nullptr;
  ncclUniqueId ids[2] = {};
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;

  EXPECT_EQ(ncclInvalidArgument,
            ncclCommInitRankScalable(&nc, /*nranks=*/1, /*myrank=*/0, /*nId=*/2, ids, &cfg));
  EXPECT_EQ(0, rec.calls);
  EXPECT_EQ(nullptr, nc);
}

TEST_F(InitMicrotest, CommInitAll_TwoDevices_LaunchesOncePerRankWithTheDeviceFromTheDevlist) {
  InstallDevCommSetupSuccess();
  std::vector<int> seenRanks;
  std::vector<int> seenDevs;
  std::vector<int> seenNranks;
  ScopedHook launch(g_ncclAsyncLaunch, [&](ncclAsyncJob* job, ncclResult_t (*)(ncclAsyncJob*),
                                           void (*)(ncclAsyncJob*), void (*destructor)(void*), ncclComm*) {
    auto* initJob = reinterpret_cast<ncclCommInitRankAsyncJob*>(job);
    seenRanks.push_back(initJob->myrank);
    seenDevs.push_back(initJob->cudaDev);
    seenNranks.push_back(initJob->nranks);
    destructor(job);
    return ncclSuccess;
  });
  ncclComm_t comms[2] = {};
  const int devlist[2] = {1, 0};

  EXPECT_EQ(ncclSuccess, ncclCommInitAll_impl(comms, 2, devlist));
  EXPECT_EQ(2, launch.calls);
  EXPECT_EQ(std::vector<int>({0, 1}), seenRanks);
  EXPECT_EQ(std::vector<int>({1, 0}), seenDevs) << "rank i takes devlist[i], not i";
  EXPECT_EQ(std::vector<int>({2, 2}), seenNranks);
  EXPECT_NE(nullptr, comms[0]);
  EXPECT_NE(nullptr, comms[1]);
  Rank_ReleaseComm(comms[0]);
  Rank_ReleaseComm(comms[1]);
}

TEST_F(InitMicrotest, CommSplit_NoColor_StillJoinsTheSplitCollectiveWithoutAChildComm) {
  ReadyComm rc;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t out = reinterpret_cast<ncclComm_t>(0x1);

  EXPECT_EQ(ncclSuccess, ncclCommSplit_impl(rc.get(), NCCL_SPLIT_NOCOLOR, /*key=*/0, &out, /*config=*/nullptr));
  EXPECT_EQ(nullptr, out);
  ASSERT_EQ(1, rec.calls);
  EXPECT_EQ(Teardown_JobFn(ncclCommInitRankFunc), rec.func);
  ncclCommInitRankAsyncJob* job = Rank_LaunchedJob(rec);
  EXPECT_EQ(nullptr, job->comm) << "no child comm is allocated for NCCL_SPLIT_NOCOLOR";
  EXPECT_EQ(rc.get(), job->parent);
  EXPECT_EQ(NCCL_SPLIT_NOCOLOR, job->color);
  rec.Release();
}

TEST_F(InitMicrotest, CommSplit_WithColor_ForwardsColourKeyAndTheIncrementedChildCount) {
  InstallDevCommSetupSuccess();
  ReadyComm rc;
  rc.get()->childCount = 4;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t out = nullptr;
  const int color = 7;
  const int key = 2;

  EXPECT_EQ(ncclSuccess, ncclCommSplit_impl(rc.get(), color, key, &out, /*config=*/nullptr));
  ASSERT_EQ(1, rec.calls);
  ncclCommInitRankAsyncJob* job = Rank_LaunchedJob(rec);
  EXPECT_EQ(color, job->color);
  EXPECT_EQ(key, job->key);
  EXPECT_EQ(5, job->childCount) << "every split must take a fresh child index";
  EXPECT_EQ(nullptr, job->excludeRanksList) << "a split is not a shrink";
  EXPECT_NE(nullptr, job->comm);
  EXPECT_EQ(rc.get(), job->parent);
  ncclComm* const child = job->comm;
  rec.Release();
  Rank_ReleaseComm(child);
}

TEST_F(InitMicrotest, CommSplit_NotReadyComm_ReturnsInvalidArgumentWithoutLaunching) {
  ReadyComm rc;
  rc.get()->asyncResult = ncclInProgress;
  Teardown_LaunchRecord rec;
  ScopedHook launch(g_ncclAsyncLaunch, Teardown_CaptureLaunch(&rec));
  ncclComm_t out = reinterpret_cast<ncclComm_t>(0x1);

  EXPECT_EQ(ncclInvalidArgument, ncclCommSplit_impl(rc.get(), /*color=*/1, /*key=*/0, &out, /*config=*/nullptr));
  EXPECT_EQ(0, rec.calls);
}

TEST_F(InitMicrotest, CommShrink_AbortRequestedAndStreamSyncFails_LeavesTheAbortFlagsRaised) {
  Teardown_LifecycleComm c(/*cudaDev=*/1, /*rank=*/1);
  g_ncclStrongStreamResult = ncclSystemError;
  ncclComm_t out = nullptr;
  int exclude[1] = {0};

  EXPECT_EQ(ncclSystemError, ncclCommShrink_impl(c.get(), exclude, /*excludeRanksCount=*/1, &out,
                                                 /*config=*/nullptr, NCCL_SHRINK_ABORT));
  EXPECT_EQ(1u, c.abortFlag());
  EXPECT_EQ(1u, c.abortFlagDev());
  EXPECT_EQ(nullptr, out);
}

TEST_F(InitMicrotest, CommShrink_AbortRequestedAndStreamSyncSucceeds_LowersTheAbortFlagsBeforeTheChildInit) {
  Teardown_LifecycleComm c(/*cudaDev=*/1, /*rank=*/1);
  ncclComm_t out = nullptr;
  int exclude[1] = {0};

  EXPECT_EQ(ncclInvalidArgument, ncclCommShrink_impl(c.get(), exclude, /*excludeRanksCount=*/0, &out,
                                                     /*config=*/nullptr, NCCL_SHRINK_ABORT));
  EXPECT_EQ(0u, c.abortFlag()) << "the abort window must close before the child comm is built";
  EXPECT_EQ(0u, c.abortFlagDev());
}

TEST_F(InitMicrotest, CommShrink_NoAbortFlag_NeverSynchronisesTheDeviceStream) {
  Teardown_LifecycleComm c(/*cudaDev=*/1, /*rank=*/1);
  g_ncclStrongStreamResult = ncclSystemError;
  ncclComm_t out = nullptr;
  int exclude[1] = {0};

  EXPECT_EQ(ncclInvalidArgument, ncclCommShrink_impl(c.get(), exclude, /*excludeRanksCount=*/0, &out,
                                                     /*config=*/nullptr, /*shrinkFlags=*/0));
  EXPECT_EQ(0u, c.abortFlag()) << "without NCCL_SHRINK_ABORT nothing raises the flags";
}


namespace {

constexpr uint64_t kGrow_ParentMagic = 0x5A5A0000BEEF0001ULL;
constexpr int kGrow_ParentMinCTAs = 6;
constexpr int kGrow_SuppliedMinCTAs = 3;
constexpr int kGrow_ParentRanks = 4;
constexpr int kGrow_ParentRank = 2;
constexpr int kGrow_ParentCudaDev = 5;
constexpr int kGrow_TargetRanks = 6;

// A distinguishable non-null value for *newcomm, so "the store happened" and "the store was deleted" differ.
ncclComm* const kGrow_NewcommPoison = reinterpret_cast<ncclComm*>(0xF00DULL);

class Grow_ParentComm {
 public:
  Grow_ParentComm() : comm_(new ncclComm{}) {
    comm_->startMagic = comm_->endMagic = NCCL_MAGIC;
    comm_->abortFlag = &abortFlag_;
    comm_->abortFlagDev = &abortFlagDev_;
    comm_->abortFlagRefCount = &abortFlagRefCount_;
    comm_->magic = kGrow_ParentMagic;
    comm_->nRanks = kGrow_ParentRanks;
    comm_->rank = kGrow_ParentRank;
    comm_->cudaDev = kGrow_ParentCudaDev;
    comm_->config.minCTAs = kGrow_ParentMinCTAs;
    comm_->config.maxCTAs = kGrow_ParentMinCTAs;
  }
  ncclComm* get() { return comm_.get(); }
  ncclComm* operator->() { return comm_.get(); }

 private:
  uint32_t abortFlag_ = 0;
  uint32_t abortFlagDev_ = 0;
  int abortFlagRefCount_ = 1;
  std::unique_ptr<ncclComm> comm_;
};

struct Grow_LaunchRecord {
  ncclResult_t (*func)(struct ncclAsyncJob*) = nullptr;
  void (*undo)(struct ncclAsyncJob*) = nullptr;
  void (*destructor)(void*) = nullptr;
  ncclComm* launchComm = nullptr;
  ncclCommInitRankAsyncJob* job = nullptr;
  ncclComm* jobComm = nullptr;
  ncclComm* jobParent = nullptr;
  ncclComm** jobNewcomm = nullptr;
  int cudaDev = -99, nranks = -99, myrank = -99, nId = -99;
  int color = -99, key = -99, childCount = -99, excludeRanksCount = -99;
  bool isGrow = true;
  bool hasCommId = false;
  ncclUniqueId commId{};
  std::vector<int> excludeRanks;
  std::string funcName;
};

// Suppresses ncclCommInitRankFunc and snapshots the job, so every field the caller computed stays observable.
class Grow_LaunchSpy {
 public:
  explicit Grow_LaunchSpy(ncclResult_t result = ncclSuccess)
      : hook_(g_ncclAsyncLaunch, [this, result](struct ncclAsyncJob* raw, ncclResult_t (*func)(struct ncclAsyncJob*),
                                                void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*),
                                                ncclComm* comm) {
          Capture(raw, func, undo, destructor, comm);
          if (result != ncclSuccess) return result;
          ReleaseCommId(raw);
          if (destructor) destructor(raw);
          return result;
        }) {}

  int calls() const { return hook_.calls; }
  const Grow_LaunchRecord& rec() const { return rec_; }

 private:
  void Capture(struct ncclAsyncJob* raw, ncclResult_t (*func)(struct ncclAsyncJob*),
               void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*), ncclComm* comm) {
    auto* job = reinterpret_cast<ncclCommInitRankAsyncJob*>(raw);
    rec_.job = job;
    rec_.func = func;
    rec_.undo = undo;
    rec_.destructor = destructor;
    rec_.launchComm = comm;
    rec_.jobComm = job->comm;
    rec_.jobParent = job->parent;
    rec_.jobNewcomm = job->newcomm;
    rec_.cudaDev = job->cudaDev;
    rec_.nranks = job->nranks;
    rec_.myrank = job->myrank;
    rec_.nId = job->nId;
    rec_.color = job->color;
    rec_.key = job->key;
    rec_.childCount = job->childCount;
    rec_.excludeRanksCount = job->excludeRanksCount;
    rec_.isGrow = job->isGrow;
    rec_.funcName = job->funcName;
    rec_.hasCommId = job->commId != nullptr;
    if (job->commId) {
      std::memcpy(&rec_.commId, job->commId, sizeof(ncclUniqueId));
    }
    if (job->excludeRanksList) {
      rec_.excludeRanks.assign(job->excludeRanksList, job->excludeRanksList + job->excludeRanksCount);
    }
  }

  // Only the launched job's destructor releases commId; on a failed launch the unit's own fail path must do it.
  static void ReleaseCommId(struct ncclAsyncJob* raw) {
    auto* job = reinterpret_cast<ncclCommInitRankAsyncJob*>(raw);
    ::free(job->commId);
    job->commId = nullptr;
  }

  Grow_LaunchRecord rec_;
  ScopedHook<ncclResult_t(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                          void (*)(struct ncclAsyncJob*), void (*)(void*), ncclComm*)>
      hook_;
};

// Records every free() the unit made and performs none of them, so the test can prove which release happened.
class Grow_FreeSpy {
 public:
  Grow_FreeSpy() : hook_(g_microFree, [this](void* p) { freed_.push_back(p); }) {}
  const std::vector<void*>& freed() const { return freed_; }

 private:
  std::vector<void*> freed_;
  ScopedHook<void(void*)> hook_;
};

void Grow_ReleaseComm(ncclComm* c, bool ownsAbortResources) {
  if (!c) {
    return;
  }
  if (ownsAbortResources) {
    ::free(c->abortFlag);
    ::free((void*)c->abortFlagDev);
    ::free(c->abortFlagRefCount);
  }
  ::free(c);
}

ncclResult_t Grow_RunExisting(ncclComm* parent, int nRanks, const ncclUniqueId* id, ncclComm_t* out,
                              ncclConfig_t* config) {
  return ncclCommGrow_impl(parent, nRanks, id, /*rank=*/-1, out, config);
}

// hipThreadExchangeStreamCaptureMode is fail-loud by default; the real one succeeds and ncclCudaHostCalloc needs it.
void Grow_AllowHostAlloc() { g_hipAsyncOpsResult = hipSuccess; }

ncclUniqueId Grow_MakeUniqueId(unsigned char fill) {
  ncclUniqueId id{};
  std::memset(&id, fill, sizeof(id));
  return id;
}

}  // namespace

TEST_F(InitMicrotest, CommGrow_NullNewcomm_ReturnsInvalidArgumentWithoutBroadcasting) {
  Grow_ParentComm parent;
  EXPECT_EQ(ncclInvalidArgument,
            ncclCommGrow_impl(parent.get(), kGrow_TargetRanks, nullptr, /*rank=*/-1, nullptr, nullptr));
  EXPECT_EQ(0, g_bcastGrowHandleCalls);
}

TEST_F(InitMicrotest, CommGrow_ZeroNRanks_WarnsAndLeavesNewcommUntouched) {
  Grow_ParentComm parent;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { res = Grow_RunExisting(parent.get(), 0, nullptr, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(kGrow_NewcommPoison, out) << "the nRanks guard returns before *newcomm is cleared";
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: total ranks must be positive, got 0")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NegativeNRanks_WarnsAndLeavesNewcommUntouched) {
  Grow_ParentComm parent;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { res = Grow_RunExisting(parent.get(), -3, nullptr, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(kGrow_NewcommPoison, out);
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: total ranks must be positive, got -3")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NRanksEqualsParentRanks_WarnsAndLeavesNewcommUntouched) {
  Grow_ParentComm parent;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = Grow_RunExisting(parent.get(), kGrow_ParentRanks, nullptr, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(kGrow_NewcommPoison, out);
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: total ranks 4 is less than current ranks 4")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NRanksBelowParentRanks_WarnsAndLeavesNewcommUntouched) {
  Grow_ParentComm parent;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = Grow_RunExisting(parent.get(), kGrow_ParentRanks - 1, nullptr, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(kGrow_NewcommPoison, out);
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: total ranks 3 is less than current ranks 4")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_CorruptedParentComm_ReturnsInvalidArgumentAndClearsNewcomm) {
  Grow_ParentComm parent;
  parent->startMagic = 0;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclInvalidArgument, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));
  EXPECT_EQ(nullptr, out) << "*newcomm is cleared before the CommCheck";
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_ExistingRankNotReady_ReturnsInvalidArgumentAndClearsNewcomm) {
  Grow_ParentComm parent;
  parent->asyncResult = ncclInProgress;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclInvalidArgument, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, parent->childCount) << "childCount must not advance for a comm that never grew";
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_ExistingRankPassesNonNegativeRank_WarnsAndLeavesChildCountUnchanged) {
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&]() {
    res = ncclCommGrow_impl(parent.get(), kGrow_TargetRanks, nullptr, /*rank=*/0, &out, nullptr);
  });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, parent->childCount);
  EXPECT_EQ(0, spy.calls());
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: existing ranks must pass rank=-1, got 0")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NewRankNegativeRank_WarnsAndReturnsInvalidArgument) {
  Grow_LaunchSpy spy;
  const ncclUniqueId id = Grow_MakeUniqueId(0x11);
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = ncclCommGrow_impl(nullptr, kGrow_TargetRanks, &id, /*rank=*/-2, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: new ranks must pass valid rank >= 0, got -2")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NewRankNullUniqueId_WarnsAndReturnsInvalidArgument) {
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = ncclCommGrow_impl(nullptr, kGrow_TargetRanks, nullptr, /*rank=*/1, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: new ranks must pass non-NULL uniqueId")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_NewRankAtNRanks_WarnsAndReturnsInvalidArgument) {
  Grow_LaunchSpy spy;
  const ncclUniqueId id = Grow_MakeUniqueId(0x11);
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = ncclCommGrow_impl(nullptr, kGrow_TargetRanks, &id, kGrow_TargetRanks, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: new rank 6 exceeds total ranks 6")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_BoundaryRankMagicMismatch_WarnsAndBroadcastsAsNonRoot) {
  Grow_ParentComm parent;
  parent->rank = 0;
  const uint64_t wrongMagic = hashCombine(kGrow_ParentMagic, 1) ^ 0xFFULL;
  ScopedHook bcast(g_bcastGrowHandle, [&](struct ncclBootstrapHandle* h, ncclComm*, bool) {
    h->magic = wrongMagic;
    return ncclSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { res = Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr); });
  EXPECT_EQ(ncclInvalidArgument, res);
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(1, bcast.calls);
  EXPECT_FALSE(g_bcastGrowHandleIsRoot) << "a growing member receives the handle, it does not mint it";
  EXPECT_EQ(1, parent->childCount) << "childCount advances before the handle is validated";
  EXPECT_EQ(0, spy.calls());
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: magic mismatch computed by the root")) << "actual log:\n" << log;
}

TEST_F(InitMicrotest, CommGrow_BoundaryRankMagicMatches_ReplacesUniqueIdWithReceivedHandle) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->rank = 0;
  ncclBootstrapHandle sent{};
  std::memset(&sent, 0x3C, sizeof(sent));
  sent.magic = hashCombine(kGrow_ParentMagic, 1);
  ScopedHook bcast(g_bcastGrowHandle, [&](struct ncclBootstrapHandle* h, ncclComm*, bool) {
    *h = sent;
    return ncclSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), kGrow_TargetRanks, /*uniqueId=*/nullptr, &out, nullptr));

  EXPECT_EQ(1, bcast.calls);
  ASSERT_EQ(1, spy.calls());
  EXPECT_EQ(1, spy.rec().nId) << "the received handle stands in for the caller's absent uniqueId";
  ASSERT_TRUE(spy.rec().hasCommId);
  EXPECT_EQ(0, std::memcmp(&spy.rec().commId, &sent, sizeof(sent)));
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_LastRankIsBoundary_ReceivesHandle) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->rank = kGrow_ParentRanks - 1;
  ScopedHook bcast(g_bcastGrowHandle, [&](struct ncclBootstrapHandle* h, ncclComm*, bool) {
    h->magic = hashCombine(kGrow_ParentMagic, 1);
    return ncclSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), kGrow_TargetRanks, /*uniqueId=*/nullptr, &out, nullptr));

  EXPECT_EQ(1, bcast.calls);
  EXPECT_EQ(1, spy.rec().nId);
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_MiddleRank_SkipsHandleBroadcastAndLaunchesWithoutCommId) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  ScopedHook bcast(g_bcastGrowHandle, [&](struct ncclBootstrapHandle*, ncclComm*, bool) {
    ADD_FAILURE() << "only ranks 0 and N-1 receive the grow handle";
    return ncclSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), kGrow_TargetRanks, /*uniqueId=*/nullptr, &out, nullptr));

  EXPECT_EQ(0, bcast.calls);
  EXPECT_EQ(0, spy.rec().nId);
  EXPECT_FALSE(spy.rec().hasCommId);
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_SingleRankParent_SkipsHandleBroadcast) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->nRanks = 1;
  parent->rank = 0;
  ScopedHook bcast(g_bcastGrowHandle, [&](struct ncclBootstrapHandle*, ncclComm*, bool) {
    ADD_FAILURE() << "a one-rank parent has no coordinator to receive from";
    return ncclSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), /*nRanks=*/2, /*uniqueId=*/nullptr, &out, nullptr));

  EXPECT_EQ(0, bcast.calls);
  EXPECT_EQ(1, parent->childCount);
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_BroadcastFails_PropagatesAndSkipsLaunch) {
  Grow_ParentComm parent;
  parent->rank = 0;
  g_bcastGrowHandleResult = ncclInternalError;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclInternalError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_CommAllocFails_ReturnsSystemErrorAndSkipsLaunch) {
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  g_callocFailAt = g_callocCallIndex;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclSystemError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_AbortFlagAllocFails_ReturnsSystemErrorAndReleasesTheComm) {
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  g_callocFailAt = g_callocCallIndex + 1;
  ncclComm_t out = kGrow_NewcommPoison;
  Grow_FreeSpy frees;

  EXPECT_EQ(ncclSystemError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  ASSERT_EQ(3u, frees.freed().size()) << "the half-built comm and its two unset abort slots are all released";
  for (void* p : frees.freed()) {
    ::free(p);
  }
}

TEST_F(InitMicrotest, CommGrow_AbortFlagDevAllocFails_ReturnsCudaErrorAndSkipsLaunch) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  ScopedHook hostMalloc(g_hipHostMalloc, [](void**, std::size_t, unsigned) { return hipErrorOutOfMemory; });
  ncclComm_t out = kGrow_NewcommPoison;
  Grow_FreeSpy frees;

  EXPECT_EQ(ncclUnhandledCudaError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, nullptr));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(1, hostMalloc.calls);
  EXPECT_EQ(0, spy.calls());
  ASSERT_EQ(3u, frees.freed().size());
  for (void* p : frees.freed()) {
    ::free(p);
  }
}

TEST_F(InitMicrotest, CommGrow_ParentDoesNotShare_FailPathReleasesAbortResourcesAndComm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->shareResources = false;
  ncclConfig_t bad = NCCL_CONFIG_INITIALIZER;
  bad.blocking = 7;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  Grow_FreeSpy frees;

  EXPECT_EQ(ncclInvalidArgument, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, &bad));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  ASSERT_EQ(3u, frees.freed().size()) << "abortFlag, abortFlagRefCount and the comm itself";
  for (void* p : frees.freed()) {
    ::free(p);
  }
}

TEST_F(InitMicrotest, CommGrow_ParentSharesResources_FailPathReleasesOnlyTheComm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->shareResources = true;
  ncclConfig_t bad = NCCL_CONFIG_INITIALIZER;
  bad.blocking = 7;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  Grow_FreeSpy frees;

  EXPECT_EQ(ncclInvalidArgument, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, &bad));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  ASSERT_EQ(1u, frees.freed().size()) << "a sharing parent keeps ownership of the abort resources";
  auto* leaked = static_cast<ncclComm*>(frees.freed()[0]);
  EXPECT_EQ(NCCL_MAGIC, leaked->startMagic);
  Grow_ReleaseComm(leaked, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_ExistingRankNullConfig_CopiesParentConfigIntoNewComm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, /*config=*/nullptr));

  ASSERT_NE(nullptr, out);
  EXPECT_EQ(kGrow_ParentMinCTAs, out->config.minCTAs);
  EXPECT_EQ(kGrow_ParentMinCTAs, out->config.maxCTAs);
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_ExistingRankWithConfig_ParsesSuppliedConfigOverParent) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.minCTAs = kGrow_SuppliedMinCTAs;
  cfg.maxCTAs = kGrow_SuppliedMinCTAs;
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, &cfg));

  ASSERT_NE(nullptr, out);
  EXPECT_EQ(kGrow_SuppliedMinCTAs, out->config.minCTAs);
  EXPECT_EQ(kGrow_ParentMinCTAs, parent->config.minCTAs) << "the parent's config is a source, never a destination";
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_InvalidConfig_ReturnsInvalidArgumentAndSkipsLaunch) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  ncclConfig_t bad = NCCL_CONFIG_INITIALIZER;
  bad.magic = 0;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclInvalidArgument, Grow_RunExisting(parent.get(), kGrow_TargetRanks, nullptr, &out, &bad));
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_ExistingRankHappyPath_LaunchesInitRankFuncOnTheNewComm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  const ncclUniqueId id = Grow_MakeUniqueId(0x7E);
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;
  ncclResult_t res = ncclSuccess;
  std::string log;
  {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    log = RcclUnitTesting::CaptureLog(
        [&]() { res = Grow_RunExisting(parent.get(), kGrow_TargetRanks, &id, &out, nullptr); });
  }

  ASSERT_EQ(ncclSuccess, res);
  ASSERT_EQ(1, spy.calls());
  const Grow_LaunchRecord& r = spy.rec();
  EXPECT_EQ(&ncclCommInitRankFunc, r.func);
  EXPECT_EQ(nullptr, r.undo);
  EXPECT_EQ(&childCommCleanupJob, r.destructor);
  EXPECT_EQ(out, r.launchComm) << "grow launches against the new comm, not the parent";
  EXPECT_EQ(out, r.jobComm);
  EXPECT_EQ(parent.get(), r.jobParent);
  EXPECT_EQ(&out, r.jobNewcomm);
  EXPECT_EQ(kGrow_ParentCudaDev, r.cudaDev);
  EXPECT_EQ(kGrow_TargetRanks, r.nranks) << "the grown size comes from the argument, not the parent";
  EXPECT_EQ(kGrow_ParentRank, r.myrank);
  EXPECT_EQ(kGrow_ParentRank, r.key);
  EXPECT_EQ(0, r.color);
  EXPECT_TRUE(r.isGrow);
  EXPECT_EQ(1, r.nId);
  EXPECT_EQ("ncclCommGrow", r.funcName);
  ASSERT_TRUE(r.hasCommId);
  EXPECT_EQ(0, std::memcmp(&r.commId, &id, sizeof(id)));
  EXPECT_EQ(ncclInProgress, out->initState);
  EXPECT_EQ(NCCL_MAGIC, out->startMagic);
  EXPECT_EQ(NCCL_MAGIC, out->endMagic);
  ASSERT_NE(nullptr, out->abortFlagRefCount);
  EXPECT_EQ(1, *out->abortFlagRefCount);
  EXPECT_NE(parent->abortFlag, out->abortFlag) << "a grown comm never shares the parent's abort flag";
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: existing rank creating new communicator with 6 total ranks"))
      << "actual log:\n" << log;
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_NewRankHappyPath_UsesCurrentDeviceAndCallerRank) {
  Grow_AllowHostAlloc();
  constexpr int kNewRank = 3;
  constexpr int kCurrentDevice = 6;
  g_currentDevice = kCurrentDevice;
  const ncclUniqueId id = Grow_MakeUniqueId(0x2D);
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.minCTAs = kGrow_SuppliedMinCTAs;
  cfg.maxCTAs = kGrow_SuppliedMinCTAs;
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;
  ncclResult_t res = ncclSuccess;
  std::string log;
  {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    log = RcclUnitTesting::CaptureLog([&]() {
      res = ncclCommGrow_impl(/*comm=*/nullptr, /*nRanks=*/8, &id, kNewRank, &out, &cfg);
    });
  }

  ASSERT_EQ(ncclSuccess, res);
  ASSERT_EQ(1, spy.calls());
  const Grow_LaunchRecord& r = spy.rec();
  EXPECT_EQ(nullptr, r.jobParent) << "a joining rank has no parent communicator";
  EXPECT_EQ(kCurrentDevice, r.cudaDev);
  EXPECT_EQ(kNewRank, r.myrank);
  EXPECT_EQ(kNewRank, r.key);
  EXPECT_EQ(8, r.nranks);
  EXPECT_TRUE(r.isGrow);
  EXPECT_EQ("ncclCommGrow", r.funcName);
  EXPECT_EQ(kGrow_SuppliedMinCTAs, out->config.minCTAs);
  EXPECT_TRUE(LogHas(log, "ncclCommGrow: new rank creating new communicator with 8 total ranks"))
      << "actual log:\n" << log;
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_NewRankNullConfig_UsesDefaultConfig) {
  Grow_AllowHostAlloc();
  const ncclUniqueId id = Grow_MakeUniqueId(0x2D);
  Grow_LaunchSpy spy;
  ncclComm_t out = nullptr;

  ASSERT_EQ(ncclSuccess, ncclCommGrow_impl(nullptr, /*nRanks=*/8, &id, /*rank=*/0, &out, /*config=*/nullptr));

  ASSERT_NE(nullptr, out);
  EXPECT_EQ(1, out->config.blocking) << "an absent config is parsed as defaults, never copied from a null parent";
  Grow_ReleaseComm(out, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, CommGrow_CommIdAllocFails_ReturnsSystemErrorAndClearsNewcomm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  const ncclUniqueId id = Grow_MakeUniqueId(0x7E);
  Grow_LaunchSpy spy;
  g_callocFailAt = g_callocCallIndex + 3;
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclSystemError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, &id, &out, nullptr));
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, CommGrow_AsyncLaunchFails_PropagatesAndClearsNewcomm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  const ncclUniqueId id = Grow_MakeUniqueId(0x7E);
  Grow_LaunchSpy spy(ncclInternalError);
  ncclComm_t out = nullptr;
  EXPECT_EQ(ncclInternalError, Grow_RunExisting(parent.get(), kGrow_TargetRanks, &id, &out, nullptr));
  EXPECT_EQ(1, spy.calls());
  EXPECT_EQ(nullptr, out) << "a failed launch must not publish a half-built comm";
}

namespace {

constexpr int kGrow_SplitColor = 3;
constexpr int kGrow_SplitKey = 9;
constexpr int kGrow_ParentChildCount = 7;
constexpr char kGrow_Caller[] = "microCaller";

ncclResult_t Grow_RunSplit(ncclComm* parent, int color, int key, ncclComm_t* out, ncclConfig_t* config) {
  return ncclCommInitChildComm(parent, out, /*isShrink=*/false, NCCL_SHRINK_DEFAULT, color, key,
                               /*excludeRanksList=*/nullptr, /*excludeRanksCount=*/0, config, kGrow_Caller);
}

ncclResult_t Grow_RunShrink(ncclComm* parent, int flags, int* excludeRanksList, int excludeRanksCount,
                            ncclComm_t* out) {
  return ncclCommInitChildComm(parent, out, /*isShrink=*/true, flags, /*color=*/0, parent->rank, excludeRanksList,
                               excludeRanksCount, /*config=*/nullptr, kGrow_Caller);
}

}  // namespace

TEST_F(InitMicrotest, InitChildComm_GetDeviceFails_ReturnsCudaErrorBeforeAnyValidation) {
  ScopedHook getDevice(g_hipGetDevice, [](int*) { return hipErrorInvalidValue; });
  ncclComm_t out = kGrow_NewcommPoison;
  EXPECT_EQ(ncclUnhandledCudaError, Grow_RunSplit(/*parent=*/nullptr, kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));
  EXPECT_EQ(kGrow_NewcommPoison, out) << "the device query precedes every write to *newcomm";
  EXPECT_EQ(1, getDevice.calls);
}

TEST_F(InitMicrotest, InitChildComm_SetDeviceFails_ReturnsCudaErrorAndRestoresDevice) {
  Grow_ParentComm parent;
  constexpr int kOldDevice = 1;
  g_currentDevice = kOldDevice;
  int setCalls = 0;
  ScopedHook setDevice(g_hipSetDevice, [&](int dev) -> hipError_t {
    if (++setCalls == 1) return hipErrorInvalidValue;
    g_currentDevice = dev;
    return hipSuccess;
  });
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  EXPECT_EQ(ncclUnhandledCudaError, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  EXPECT_EQ(kGrow_NewcommPoison, out) << "*newcomm is cleared only after the device switch succeeds";
  EXPECT_EQ(2, setDevice.calls);
  EXPECT_EQ(kOldDevice, g_currentDevice);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, InitChildComm_SplitNoColor_LaunchesWithoutAllocatingChild) {
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;
  ncclResult_t res = ncclSuccess;
  std::string log;
  {
    ScopedDebugLogging dbg(NCCL_LOG_INFO, NCCL_ALL);
    log = RcclUnitTesting::CaptureLog(
        [&]() { res = Grow_RunSplit(parent.get(), NCCL_SPLIT_NOCOLOR, kGrow_SplitKey, &out, nullptr); });
  }

  EXPECT_EQ(ncclSuccess, res);
  EXPECT_EQ(nullptr, out);
  ASSERT_EQ(1, spy.calls());
  EXPECT_EQ(nullptr, spy.rec().jobComm) << "a colourless rank still joins the job, with no child of its own";
  EXPECT_EQ(NCCL_SPLIT_NOCOLOR, spy.rec().color);
  EXPECT_EQ(parent.get(), spy.rec().launchComm);
  EXPECT_TRUE(LogHas(log, "Rank 2 has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator"))
      << "actual log:\n" << log;
}

TEST_F(InitMicrotest, InitChildComm_ShrinkIgnoresNoColor_StillAllocatesChild) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  int exclude[1] = {0};
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, ncclCommInitChildComm(parent.get(), &out, /*isShrink=*/true, NCCL_SHRINK_DEFAULT,
                                               NCCL_SPLIT_NOCOLOR, parent->rank, exclude, 1, nullptr, kGrow_Caller));

  ASSERT_EQ(1, spy.calls());
  ASSERT_NE(nullptr, spy.rec().jobComm) << "NCCL_SPLIT_NOCOLOR is a split concept and must not skip a shrink child";
  Grow_ReleaseComm(spy.rec().jobComm, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_Split_AllocatesChildAndLaunchesOnTheParent) {
  Grow_AllowHostAlloc();
  constexpr int kSplitOldDevice = 1;
  g_currentDevice = kSplitOldDevice;
  Grow_ParentComm parent;
  parent->childCount = kGrow_ParentChildCount;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  EXPECT_EQ(nullptr, out) << "*newcomm stays null until the async job completes";
  ASSERT_EQ(1, spy.calls());
  const Grow_LaunchRecord& r = spy.rec();
  ncclComm* child = r.jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_EQ(&ncclCommInitRankFunc, r.func);
  EXPECT_EQ(nullptr, r.undo);
  EXPECT_EQ(&childCommCleanupJob, r.destructor);
  EXPECT_EQ(parent.get(), r.launchComm) << "a split launches against the parent, not the child";
  EXPECT_EQ(parent.get(), r.jobParent);
  EXPECT_EQ(&out, r.jobNewcomm);
  EXPECT_EQ(kGrow_SplitColor, r.color);
  EXPECT_EQ(kGrow_SplitKey, r.key);
  EXPECT_EQ(kGrow_ParentChildCount + 1, r.childCount);
  EXPECT_EQ(kGrow_ParentChildCount + 1, parent->childCount);
  EXPECT_EQ(kGrow_ParentCudaDev, r.cudaDev);
  EXPECT_EQ(kGrow_Caller, r.funcName);
  EXPECT_FALSE(r.isGrow);
  EXPECT_EQ(NCCL_MAGIC, child->startMagic);
  EXPECT_EQ(NCCL_MAGIC, child->endMagic);
  EXPECT_EQ(ncclInternalError, child->initState);
  EXPECT_FALSE(parent->shareResources);
  ASSERT_NE(nullptr, child->abortFlag);
  EXPECT_NE(parent->abortFlag, child->abortFlag);
  EXPECT_EQ(child->abortFlag, parent->childAbortFlag);
  EXPECT_EQ(child->abortFlagDev, parent->childAbortFlagDev);
  ASSERT_NE(nullptr, child->abortFlagRefCount);
  EXPECT_EQ(1, *child->abortFlagRefCount);
  EXPECT_EQ(kGrow_ParentMinCTAs, child->config.minCTAs);
  EXPECT_EQ(kSplitOldDevice, g_currentDevice) << "the caller's device is restored before returning";
  Grow_ReleaseComm(child, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_SplitShareEnabled_ReusesParentAbortResources) {
  Grow_ParentComm parent;
  parent->config.splitShare = 1;
  parent->childAbortFlag = parent->abortFlag;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_TRUE(parent->shareResources);
  EXPECT_EQ(parent->abortFlag, child->abortFlag);
  EXPECT_EQ(parent->abortFlagDev, child->abortFlagDev);
  EXPECT_EQ(parent->abortFlagRefCount, child->abortFlagRefCount);
  EXPECT_EQ(nullptr, parent->childAbortFlag) << "a sharing parent has no separate child abort flag to watch";
  EXPECT_EQ(2, *parent->abortFlagRefCount);
  Grow_ReleaseComm(child, /*ownsAbortResources=*/false);
}

TEST_F(InitMicrotest, InitChildComm_RevokedParent_ForcesFreshAbortResources) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->config.splitShare = 1;
  parent->revokedFlag = 1;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_FALSE(parent->shareResources);
  EXPECT_NE(parent->abortFlag, child->abortFlag);
  EXPECT_EQ(1, *parent->abortFlagRefCount) << "a revoked parent's refcount must not be taken";
  Grow_ReleaseComm(child, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_ShrinkDefaultMode_SharesWhenShrinkShareIsSet) {
  Grow_ParentComm parent;
  parent->config.shrinkShare = 1;
  parent->config.splitShare = 0;
  int exclude[1] = {0};
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunShrink(parent.get(), NCCL_SHRINK_DEFAULT, exclude, 1, &out));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_TRUE(parent->shareResources) << "a shrink consults shrinkShare, never splitShare";
  EXPECT_EQ(parent->abortFlag, child->abortFlag);
  Grow_ReleaseComm(child, /*ownsAbortResources=*/false);
}

TEST_F(InitMicrotest, InitChildComm_ShrinkAbortMode_DoesNotShareEvenWithShrinkShareSet) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->config.shrinkShare = 1;
  int exclude[1] = {0};
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunShrink(parent.get(), NCCL_SHRINK_ABORT, exclude, 1, &out));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_FALSE(parent->shareResources);
  EXPECT_NE(parent->abortFlag, child->abortFlag);
  Grow_ReleaseComm(child, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_SplitShareUnsetWithShrinkShareSet_DoesNotShare) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->config.shrinkShare = 1;
  parent->config.splitShare = 0;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_FALSE(parent->shareResources) << "a split consults splitShare, never shrinkShare";
  Grow_ReleaseComm(child, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_Shrink_SortsCallerListAndCopiesItIntoTheJob) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  parent->childCount = kGrow_ParentChildCount;
  int exclude[3] = {5, 1, 3};
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunShrink(parent.get(), NCCL_SHRINK_DEFAULT, exclude, 3, &out));

  EXPECT_EQ(1, exclude[0]);
  EXPECT_EQ(3, exclude[1]);
  EXPECT_EQ(5, exclude[2]);
  const Grow_LaunchRecord& r = spy.rec();
  EXPECT_EQ(3, r.excludeRanksCount);
  EXPECT_EQ(std::vector<int>({1, 3, 5}), r.excludeRanks);
  EXPECT_EQ(0, r.childCount) << "a shrink does not consume a childCount slot";
  EXPECT_EQ(kGrow_ParentChildCount, parent->childCount);
  Grow_ReleaseComm(r.jobComm, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_ConfigProvided_ParsesInsteadOfCopyingParent) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  ncclConfig_t cfg = NCCL_CONFIG_INITIALIZER;
  cfg.minCTAs = kGrow_SuppliedMinCTAs;
  cfg.maxCTAs = kGrow_SuppliedMinCTAs;
  Grow_LaunchSpy spy;
  ncclComm_t out = kGrow_NewcommPoison;

  ASSERT_EQ(ncclSuccess, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, &cfg));

  ncclComm* child = spy.rec().jobComm;
  ASSERT_NE(nullptr, child);
  EXPECT_EQ(kGrow_SuppliedMinCTAs, child->config.minCTAs);
  EXPECT_EQ(kGrow_ParentMinCTAs, parent->config.minCTAs);
  Grow_ReleaseComm(child, /*ownsAbortResources=*/true);
}

TEST_F(InitMicrotest, InitChildComm_ChildAllocFails_ReturnsSystemErrorAndClearsNewcomm) {
  constexpr int kSplitOldDevice = 1;
  g_currentDevice = kSplitOldDevice;
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  g_callocFailAt = g_callocCallIndex;
  ncclComm_t out = kGrow_NewcommPoison;

  EXPECT_EQ(ncclSystemError, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  EXPECT_EQ(kSplitOldDevice, g_currentDevice);
}

TEST_F(InitMicrotest, InitChildComm_AbortFlagDevAllocFails_ReleasesChildAndClearsNewcomm) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  Grow_LaunchSpy spy;
  ScopedHook hostMalloc(g_hipHostMalloc, [](void**, std::size_t, unsigned) { return hipErrorOutOfMemory; });
  ncclComm_t out = kGrow_NewcommPoison;
  Grow_FreeSpy frees;

  EXPECT_EQ(ncclUnhandledCudaError, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
  ASSERT_EQ(2u, frees.freed().size()) << "the child's abort flag and the child itself";
  EXPECT_NE(frees.freed()[0], frees.freed()[1]);
  for (void* p : frees.freed()) {
    ::free(p);
  }
}

TEST_F(InitMicrotest, InitChildComm_ExcludeListAllocFails_ReturnsSystemErrorAndSkipsLaunch) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  int exclude[2] = {0, 1};
  Grow_LaunchSpy spy;
  g_callocFailAt = g_callocCallIndex + 3;
  ncclComm_t out = kGrow_NewcommPoison;

  EXPECT_EQ(ncclSystemError, Grow_RunShrink(parent.get(), NCCL_SHRINK_DEFAULT, exclude, 2, &out));

  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(0, spy.calls());
}

TEST_F(InitMicrotest, InitChildComm_AsyncLaunchFails_ClearsNewcommRestoresDeviceAndLeavesTheJobAllocated) {
  Grow_AllowHostAlloc();
  Grow_ParentComm parent;
  constexpr int kOldDevice = 1;
  g_currentDevice = kOldDevice;
  Grow_LaunchSpy spy(ncclInternalError);
  ncclComm_t out = kGrow_NewcommPoison;

  EXPECT_EQ(ncclInternalError, Grow_RunSplit(parent.get(), kGrow_SplitColor, kGrow_SplitKey, &out, nullptr));

  EXPECT_EQ(1, spy.calls());
  EXPECT_EQ(nullptr, out);
  EXPECT_EQ(kOldDevice, g_currentDevice);

  ASSERT_NE(nullptr, spy.rec().job);
  std::unique_ptr<ncclCommInitRankAsyncJob> orphan(spy.rec().job);
  auto* reused = new ncclCommInitRankAsyncJob{};
  const bool recycled = (reused == spy.rec().job);
  delete reused;
  if (Dtor_kHeapSlotRecycles) {
    EXPECT_FALSE(recycled) << "the fail path released the job: its heap slot was reused";
  }
}
