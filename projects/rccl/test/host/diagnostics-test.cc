/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only microtests for src/diagnostics.cc, #include-d via DIAGNOSTICS_CC_PATH. Its runners are renamed there so
 * they sit beside fakes/diagnostics_fakes.cc; popen/pclose, ncclCalloc/free and the two checks it drives are seams,
 * safe as macros only because every header diagnostics.cc includes is included first.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <regex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/nccl_fakes.h"

#include "alloc.h"
#include "bootstrap.h"
#include "checks.h"
#include "comm.h"
#include "diagnostics.h"
#include "diagnostics/ib_write_bw.h"
#include "diagnostics/p2p.h"
#include "diagnostics_log.h"
#include "graph.h"
#include "graph/topo.h"

#include "fakes/param_redirect.h"

using RcclUnitTesting::CaptureStdout;

static std::function<FILE*(const char*, const char*)> g_diagnosticsPopen = ::popen;
static std::function<int(FILE*)> g_diagnosticsPclose = ::pclose;
static std::function<void(ncclComm*)> g_diagnosticsIbWriteBw;
static std::function<ncclResult_t(ncclComm*)> g_diagnosticsP2pRun;

static int g_diagnosticsCallocFailAt = 0;
static std::vector<void*> g_diagnosticsHostLive;
template <typename T>
static ncclResult_t DiagnosticsCalloc(T** ptr, std::size_t nelem) {
  if (--g_diagnosticsCallocFailAt == 0) {
    return ncclSystemError;
  }
  const ncclResult_t ret = ncclCallocDebug(ptr, nelem, __FILE__, __LINE__, __func__, true);
  if (ret == ncclSuccess) {
    g_diagnosticsHostLive.push_back(*ptr);
  }
  return ret;
}

static void DiagnosticsFree(void* ptr) {
  auto it = std::find(g_diagnosticsHostLive.begin(), g_diagnosticsHostLive.end(), ptr);
  if (it != g_diagnosticsHostLive.end()) {
    g_diagnosticsHostLive.erase(it);
  } else if (ptr != nullptr) {
    ADD_FAILURE() << "free of untracked host pointer " << ptr;
    return;
  }
  std::free(ptr);
}

#undef ncclCalloc
#define ncclCalloc(...) DiagnosticsCalloc(__VA_ARGS__)
#define free(ptr) DiagnosticsFree(ptr)
#define popen(command, mode) g_diagnosticsPopen(command, mode)
#define pclose(stream) g_diagnosticsPclose(stream)
#define ncclDiagRunIbWriteBw(comm) g_diagnosticsIbWriteBw(comm)
#define ncclDiagP2pRun(comm) g_diagnosticsP2pRun(comm)
#define ncclDiagChildRunStream DiagnosticsChildRunStream
#define ncclDiagChildRun DiagnosticsChildRun

#include DIAGNOSTICS_CC_PATH

#undef ncclDiagChildRun
#undef ncclDiagChildRunStream
#undef ncclDiagP2pRun
#undef ncclDiagRunIbWriteBw
#undef pclose
#undef popen
#undef free
#undef ncclCalloc

namespace {

constexpr uintptr_t kBootstrapBits = 0xb007;
constexpr int kTimeoutSec = 7;
constexpr char kScrub[] =
    "for __v in $(env | sed -nE '/^(OMPI_|OPAL_|PMIX_|PMI_|PRTE_|PERFTEST_FORCE_MPI=)/s/=.*//p'); "
    "do unset \"$__v\"; done; ";

std::string Wrapped(const std::string& command, int timeoutSec = kTimeoutSec) {
  return kScrub + std::string("timeout -k 2 ") + std::to_string(timeoutSec) + " stdbuf -oL " + command +
         " </dev/null 2>&1";
}

std::string DiagPrefix() {
  diagLogInit();
  return std::string(diagLogHost) + ":" + std::to_string(diagLogPid) + " ";
}

// One DIAG_PRINT line: "<host>:<pid> <body>\n".
std::string DiagLine(const std::string& body) {
  return DiagPrefix() + body + "\n";
}

// The closing line's milliseconds, or -1 when `out` does not end with it.
double ElapsedMs(const std::string& out, int nRanks) {
  const std::string head = DiagPrefix() + "NCCL DIAG NCCL diagnostics completed in ";
  const std::size_t at = out.rfind(head);
  const std::string tail = at == std::string::npos ? "" : out.substr(at + head.size());
  const std::regex line("([0-9]+\\.[0-9]) ms across " + std::to_string(nRanks) + " ranks\n");
  std::smatch match;
  if (!std::regex_match(tail, match, line)) {
    return -1;
  }
  return std::stod(match[1]);
}

const std::string kBanner = DiagLine("NCCL DIAG === NCCL Diagnostics ===");
const std::string kNetSkipped = DiagLine("NCCL DIAG [OK] net bw: skipped (single host or no network transport)");

unsigned Bit(int path) {
  return 1u << path;
}

void RecordLine(const char* line, void* ctx) {
  static_cast<std::vector<std::string>*>(ctx)->push_back(line);
}

class DiagnosticsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_diagnosticsIbWriteBw = [this](ncclComm* comm) {
      EXPECT_EQ(comm, comm_.get());
      checks_.push_back("ib");
    };
    g_diagnosticsP2pRun = [this](ncclComm* comm) {
      EXPECT_EQ(comm, comm_.get());
      checks_.push_back("p2p");
      std::this_thread::sleep_for(p2pDelay_);
      return p2pResult_;
    };
    g_devrBootstrapAllGather = [this](void* bs, void* buf, int size) { return AllGather(bs, buf, size); };
  }

  void TearDown() override {
    EXPECT_TRUE(g_diagnosticsHostLive.empty()) << "host buffers leaked: " << g_diagnosticsHostLive.size();
    for (void* p : g_diagnosticsHostLive) {
      std::free(p);
    }
    g_diagnosticsHostLive.clear();
    g_diagnosticsCallocFailAt = 0;
    g_diagnosticsPopen = ::popen;
    g_diagnosticsPclose = ::pclose;
    g_diagnosticsIbWriteBw = nullptr;
    g_diagnosticsP2pRun = nullptr;
    ResetDevRuntimeMicroFakes();
    ResetNcclFakes();
  }

  // The child's whole output arrives through a real FILE*, so fgets chunks it exactly as a pipe would.
  void ScriptChild(const std::string& output, int status) {
    g_diagnosticsPopen = [this, output](const char* command, const char* mode) {
      EXPECT_STREQ(mode, "r");
      commands_.push_back(command);
      FILE* stream = std::tmpfile();
      std::fwrite(output.data(), 1, output.size(), stream);
      std::rewind(stream);
      return stream;
    };
    g_diagnosticsPclose = [status](FILE* stream) {
      std::fclose(stream);
      return status;
    };
  }

  int RunStream(const char* command, int timeoutSec, int outputSize) {
    output_.assign(std::max(outputSize, 1) + 1, 'Z');
    truncated_ = true;
    lines_.clear();
    return DiagnosticsChildRunStream(command, timeoutSec, output_.data(), outputSize, RecordLine, &lines_,
                                     &truncated_);
  }

  void BuildComm(int nRanks, int rank, int nNodes) {
    comm_ = std::make_unique<ncclComm>();
    comm_->rank = rank;
    comm_->nRanks = nRanks;
    comm_->nNodes = nNodes;
    comm_->bootstrap = reinterpret_cast<void*>(kBootstrapBits);
    peerMasks_.assign(nRanks, 0);
  }

  // GPU node i holds rank gpuRanks[i]; this rank's node, if any, reaches node j over paths[j]. Other rows stay
  // PATH_LOC, so reading any row but this rank's shows up in the mask.
  void BuildTopo(const std::vector<int>& gpuRanks, const std::vector<int>& paths) {
    topo_ = std::make_unique<ncclTopoSystem>();
    const int n = static_cast<int>(gpuRanks.size());
    links_.assign(n, std::vector<ncclTopoLinkList>(n));
    topo_->nodes[GPU].count = n;
    for (int i = 0; i < n; i++) {
      topo_->nodes[GPU].nodes[i].gpu.rank = gpuRanks[i];
      topo_->nodes[GPU].nodes[i].paths[GPU] = links_[i].data();
      if (gpuRanks[i] == comm_->rank) {
        for (int j = 0; j < n; j++) {
          links_[i][j].type = paths[j];
        }
      }
    }
    comm_->topo = topo_.get();
  }

  ncclResult_t AllGather(void* bs, void* buf, int size) {
    EXPECT_EQ(bs, comm_->bootstrap);
    EXPECT_EQ(size, static_cast<int>(sizeof(unsigned int)));
    gathers_++;
    unsigned int* masks = static_cast<unsigned int*>(buf);
    for (int r = 0; r < comm_->nRanks; r++) {
      if (r == comm_->rank) {
        myMask_ = masks[r];
      } else {
        EXPECT_EQ(masks[r], 0u) << "peer slot " << r << " not zeroed";
        masks[r] = peerMasks_[r];
      }
    }
    return gatherResult_;
  }

  unsigned int Detect(ncclResult_t* ret) {
    unsigned int mask = 0xdeadbeef;
    *ret = ncclDiagDetectTransportMask(comm_.get(), &mask);
    return mask;
  }

  std::string Run(ncclResult_t* ret) {
    checks_.clear();
    return CaptureStdout([&] { *ret = ncclRunDiagnostics(comm_.get()); });
  }

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclTopoSystem> topo_;
  std::vector<std::vector<ncclTopoLinkList>> links_;
  std::vector<unsigned int> peerMasks_;
  unsigned int myMask_ = 0;
  int gathers_ = 0;
  ncclResult_t gatherResult_ = ncclSuccess;
  std::vector<std::string> checks_;
  ncclResult_t p2pResult_ = ncclSuccess;
  std::chrono::milliseconds p2pDelay_{0};
  std::vector<std::string> commands_;
  std::vector<std::string> lines_;
  std::vector<char> output_;
  bool truncated_ = true;
};

TEST_F(DiagnosticsMicrotest, Param_RunDiagnosticsDefaultsOff) {
  ScopedHook param(g_loadParam, [](const char* env, int64_t deftVal) {
    EXPECT_STREQ(env, "RUN_DIAGNOSTICS");
    EXPECT_EQ(deftVal, 0);
    return int64_t{1};
  });
  EXPECT_EQ(ncclParamDiagnostics(), 1);
  EXPECT_EQ(param.calls, 1);
}

TEST_F(DiagnosticsMicrotest, ChildRun_RejectsNullCommandAndSubSecondTimeoutWithoutSpawning) {
  ScriptChild("never\n", 0);
  EXPECT_EQ(RunStream(nullptr, kTimeoutSec, 8), -1);
  EXPECT_EQ(output_[0], '\0');
  EXPECT_FALSE(truncated_);
  EXPECT_EQ(RunStream("true", 0, 8), -1);
  EXPECT_EQ(output_[0], '\0');
  EXPECT_FALSE(truncated_);
  EXPECT_EQ(RunStream("true", -1, 8), -1);
  EXPECT_TRUE(commands_.empty());
  EXPECT_TRUE(lines_.empty());
  EXPECT_EQ(RunStream("true", 1, 8), 0);
  EXPECT_EQ(commands_, (std::vector<std::string>{Wrapped("true", 1)}));
}

TEST_F(DiagnosticsMicrotest, ChildRun_WrapsCommandInMpiScrubTimeoutAndLineBuffering) {
  ScriptChild("", 0);
  EXPECT_EQ(DiagnosticsChildRun("ib_write_bw -d mlx5_0", 30, nullptr, 0, nullptr), 0);
  ASSERT_EQ(commands_.size(), 1u);
  EXPECT_EQ(commands_[0],
            "for __v in $(env | sed -nE '/^(OMPI_|OPAL_|PMIX_|PMI_|PRTE_|PERFTEST_FORCE_MPI=)/s/=.*//p'); do unset "
            "\"$__v\"; done; timeout -k 2 30 stdbuf -oL ib_write_bw -d mlx5_0 </dev/null 2>&1");
}

TEST_F(DiagnosticsMicrotest, ChildRun_RejectsCommandWhoseWrappedFormFillsTheBuffer) {
  ScriptChild("", 0);
  const std::size_t fits = 2047 - Wrapped("").size();
  const std::string longest(fits, 'c');
  EXPECT_EQ(RunStream(longest.c_str(), kTimeoutSec, 8), 0);
  ASSERT_EQ(commands_.size(), 1u);
  EXPECT_EQ(commands_[0], Wrapped(longest));
  EXPECT_EQ(commands_[0].size(), 2047u);
  const std::string tooLong(fits + 1, 'c');
  EXPECT_EQ(RunStream(tooLong.c_str(), kTimeoutSec, 8), -1);
  EXPECT_EQ(commands_.size(), 1u);
  EXPECT_EQ(output_[0], '\0');
  EXPECT_FALSE(truncated_);
}

TEST_F(DiagnosticsMicrotest, ChildRun_PopenFailureReturnsMinusOne) {
  ScopedHook spawn(g_diagnosticsPopen, [](const char*, const char*) { return static_cast<FILE*>(nullptr); });
  ScopedHook reap(g_diagnosticsPclose, [](FILE*) { return 0; });
  EXPECT_EQ(RunStream("true", kTimeoutSec, 8), -1);
  EXPECT_EQ(spawn.calls, 1);
  EXPECT_EQ(reap.calls, 0);
  EXPECT_EQ(output_[0], '\0');
  EXPECT_FALSE(truncated_);
}

TEST_F(DiagnosticsMicrotest, ChildRunStream_ReportsEachLineAndKeepsAllOutput) {
  ScriptChild("listening on 18515\nconnected\nlast-no-newline", 0);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 64), 0);
  EXPECT_EQ(lines_, (std::vector<std::string>{"listening on 18515\n", "connected\n", "last-no-newline"}));
  EXPECT_STREQ(output_.data(), "listening on 18515\nconnected\nlast-no-newline");
  EXPECT_FALSE(truncated_);
}

TEST_F(DiagnosticsMicrotest, ChildRunStream_SplitsLinesLongerThanTheReadBuffer) {
  ScriptChild(std::string(5000, 'x') + "\n", 0);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 8192), 0);
  EXPECT_EQ(lines_, (std::vector<std::string>{std::string(4095, 'x'), std::string(905, 'x') + "\n"}));
  EXPECT_EQ(std::strlen(output_.data()), 5001u);
  EXPECT_FALSE(truncated_);
}

TEST_F(DiagnosticsMicrotest, ChildRunStream_ClampsOutputToBufferAndFlagsTruncation) {
  ScriptChild("line1\nline2\n", 0);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 13), 0);
  EXPECT_STREQ(output_.data(), "line1\nline2\n");
  EXPECT_FALSE(truncated_);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 12), 0);
  EXPECT_STREQ(output_.data(), "line1\nline2");
  EXPECT_TRUE(truncated_);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 4), 0);
  EXPECT_STREQ(output_.data(), "lin");
  EXPECT_EQ(output_[4], 'Z');
  EXPECT_TRUE(truncated_);
  EXPECT_EQ(lines_, (std::vector<std::string>{"line1\n", "line2\n"}));
}

TEST_F(DiagnosticsMicrotest, ChildRunStream_NoRoomForOutputStillStreamsLines) {
  ScriptChild("one\n", 0);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 1), 0);
  EXPECT_EQ(output_[0], '\0');
  EXPECT_EQ(output_[1], 'Z');
  EXPECT_TRUE(truncated_);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 0), 0);
  EXPECT_EQ(output_[0], 'Z');
  EXPECT_TRUE(truncated_);
  EXPECT_EQ(lines_, (std::vector<std::string>{"one\n"}));
  bool truncated = false;
  EXPECT_EQ(DiagnosticsChildRunStream("srv", kTimeoutSec, nullptr, 64, nullptr, nullptr, &truncated), 0);
  EXPECT_TRUE(truncated);
  ScriptChild("", 0);
  EXPECT_EQ(RunStream("srv", kTimeoutSec, 0), 0);
  EXPECT_FALSE(truncated_);
}

TEST_F(DiagnosticsMicrotest, ChildRun_ReturnsExitCodeOnlyForNormalExit) {
  char output[16];
  ScriptChild("x\n", 3 << 8);
  EXPECT_EQ(DiagnosticsChildRun("tool", kTimeoutSec, output, sizeof(output), nullptr), 3);
  EXPECT_STREQ(output, "x\n");
  ScriptChild("", 124 << 8);
  EXPECT_EQ(DiagnosticsChildRun("tool", kTimeoutSec, output, sizeof(output), nullptr), 124);
  ScriptChild("", SIGKILL);
  EXPECT_EQ(DiagnosticsChildRun("tool", kTimeoutSec, output, sizeof(output), nullptr), -1);
  ScriptChild("", -1);
  EXPECT_EQ(DiagnosticsChildRun("tool", kTimeoutSec, output, sizeof(output), nullptr), -1);
}

// The only tests that spawn: the shell wrapper itself is what is under test.
TEST_F(DiagnosticsMicrotest, ChildRun_RealChildLosesMpiEnvAndStdinAndMergesStderr) {
  // Under mpirun these may already be set; put them back afterwards.
  const std::vector<std::pair<const char*, const char*>> vars = {{"OMPI_COMM_WORLD_RANK", "3"},
                                                                 {"PMI_RANK", "3"},
                                                                 {"PERFTEST_FORCE_MPI", "1"},
                                                                 {"PERFTEST_FORCE_MPIX", "kept"},
                                                                 {"KEEP_OMPI_X", "kept"}};
  std::vector<std::pair<std::string, bool>> saved;
  for (const auto& [name, value] : vars) {
    const char* old = std::getenv(name);
    saved.emplace_back(old != nullptr ? old : "", old != nullptr);
    setenv(name, value, 1);
  }
  char output[128];
  bool truncated = true;
  const int exitCode = DiagnosticsChildRun(
      "sh -c 'echo ${OMPI_COMM_WORLD_RANK-unset} ${PMI_RANK-unset} ${PERFTEST_FORCE_MPI-unset} "
      "${PERFTEST_FORCE_MPIX-unset} ${KEEP_OMPI_X-unset}; cat; echo err >&2; exit 5'",
      kTimeoutSec, output, sizeof(output), &truncated);
  for (std::size_t i = 0; i < vars.size(); i++) {
    if (saved[i].second) {
      setenv(vars[i].first, saved[i].first.c_str(), 1);
    } else {
      unsetenv(vars[i].first);
    }
  }
  EXPECT_EQ(exitCode, 5);
  EXPECT_STREQ(output, "unset unset unset kept kept\nerr\n");
  EXPECT_FALSE(truncated);
}

TEST_F(DiagnosticsMicrotest, ChildRun_RealTimeoutKillsAHungChild) {
  char output[16];
  EXPECT_EQ(DiagnosticsChildRun("sleep 30", 1, output, sizeof(output), nullptr), 124);
  EXPECT_STREQ(output, "");
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_WithoutTopoPeersAreUndetermined) {
  BuildComm(3, 1, 2);
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_DIS));
  EXPECT_EQ(ret, ncclSuccess);
  EXPECT_EQ(myMask_, Bit(PATH_LOC) | Bit(PATH_DIS));
  EXPECT_EQ(gathers_, 1);
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_OwnGpuMissingFromTopoIsNotNet) {
  BuildComm(3, 1, 2);
  BuildTopo({0, 2}, {PATH_LOC, PATH_NVL});
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_DIS));
  EXPECT_EQ(ret, ncclSuccess);
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_ClassifiesPeersByPathAndAbsentPeersAsNet) {
  BuildComm(4, 2, 2);
  BuildTopo({3, 2}, {PATH_NVL, PATH_LOC});
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_NVL) | Bit(PATH_NET));
  EXPECT_EQ(ret, ncclSuccess);
  BuildComm(2, 1, 1);
  BuildTopo({0, 1}, {PATH_SYS, PATH_LOC});
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_SYS));
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_OutOfRangePathCountsAsDisconnected) {
  BuildComm(3, 0, 1);
  BuildTopo({0, 1, 2}, {PATH_LOC, PATH_DIS + 1, -1});
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_DIS));
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_UnionsEveryRanksMask) {
  BuildComm(3, 0, 1);
  BuildTopo({0, 1, 2}, {PATH_LOC, PATH_NVL, PATH_NVL});
  peerMasks_ = {0, Bit(PATH_PIX), Bit(PATH_NET) | Bit(PATH_LOC)};
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), Bit(PATH_LOC) | Bit(PATH_NVL) | Bit(PATH_PIX) | Bit(PATH_NET));
  EXPECT_EQ(myMask_, Bit(PATH_LOC) | Bit(PATH_NVL));
  EXPECT_EQ(ncclDiagDetectTransportMask(comm_.get(), nullptr), ncclSuccess);
  EXPECT_EQ(gathers_, 2);
}

TEST_F(DiagnosticsMicrotest, DetectTransportMask_AllGatherFailureLeavesMaskZero) {
  BuildComm(2, 0, 2);
  peerMasks_ = {0, Bit(PATH_NET)};
  gatherResult_ = ncclRemoteError;
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), 0u);
  EXPECT_EQ(ret, ncclRemoteError);
}

// Peers enter the allgather this rank skips, so later diagnostics collectives pair up wrongly; a fix flips this pin.
TEST_F(DiagnosticsMicrotest, DetectTransportMask_CallocFailureStillJoinsAllGather) {
  BuildComm(2, 0, 2);
  g_diagnosticsCallocFailAt = 1;
  ncclResult_t ret;
  EXPECT_EQ(Detect(&ret), 0u);
  // The allocation failure is still surfaced...
  EXPECT_EQ(ret, ncclSystemError);
  // ...but the rank joins the all-gather via the fallback buffer, so peers do not hang.
  EXPECT_EQ(gathers_, 1);
}

TEST_F(DiagnosticsMicrotest, Run_SingleHostSkipsNetBwAndRunsP2p) {
  BuildComm(3, 0, 1);
  BuildTopo({0, 1, 2}, {PATH_LOC, PATH_NVL, PATH_NVL});
  ncclResult_t ret;
  const std::string out = Run(&ret);
  EXPECT_EQ(ret, ncclSuccess);
  EXPECT_EQ(checks_, (std::vector<std::string>{"p2p"}));
  EXPECT_EQ(out.rfind(kBanner + kNetSkipped, 0), 0u) << out;
  EXPECT_GE(ElapsedMs(out, 3), 0.0) << out;
  EXPECT_EQ(std::count(out.begin(), out.end(), '\n'), 3);
}

TEST_F(DiagnosticsMicrotest, Run_NetPathAcrossTwoHostsRunsNetBwBeforeP2p) {
  BuildComm(2, 0, 2);
  BuildTopo({0}, {PATH_LOC});
  ncclResult_t ret;
  const std::string out = Run(&ret);
  EXPECT_EQ(ret, ncclSuccess);
  EXPECT_EQ(checks_, (std::vector<std::string>{"ib", "p2p"}));
  EXPECT_EQ(out.find(kNetSkipped), std::string::npos) << out;
  EXPECT_EQ(out.rfind(kBanner, 0), 0u) << out;
}

TEST_F(DiagnosticsMicrotest, Run_NetBwNeedsBothANetPathAndTwoHosts) {
  ncclResult_t ret;
  BuildComm(2, 0, 1);
  BuildTopo({0}, {PATH_LOC});
  EXPECT_NE(Run(&ret).find(kNetSkipped), std::string::npos);
  EXPECT_EQ(checks_, (std::vector<std::string>{"p2p"}));
  BuildComm(2, 0, 3);
  BuildTopo({0, 1}, {PATH_LOC, PATH_SYS});
  EXPECT_NE(Run(&ret).find(kNetSkipped), std::string::npos);
  EXPECT_EQ(checks_, (std::vector<std::string>{"p2p"}));
  // A peer only reports NET: the allgathered union still sends this rank into the check.
  BuildComm(2, 0, 2);
  BuildTopo({0, 1}, {PATH_LOC, PATH_SYS});
  peerMasks_[1] = Bit(PATH_NET);
  Run(&ret);
  EXPECT_EQ(checks_, (std::vector<std::string>{"ib", "p2p"}));
}

TEST_F(DiagnosticsMicrotest, Run_NonZeroRankPrintsNothingEvenOnFailures) {
  BuildComm(2, 1, 1);
  gatherResult_ = ncclRemoteError;
  p2pResult_ = ncclSystemError;
  ncclResult_t ret;
  EXPECT_EQ(Run(&ret), "");
  EXPECT_EQ(ret, ncclSuccess);
  EXPECT_EQ(checks_, (std::vector<std::string>{"p2p"}));
  BuildComm(2, 1, 2);
  gatherResult_ = ncclSuccess;
  peerMasks_[0] = Bit(PATH_NET);
  EXPECT_EQ(Run(&ret), "");
  EXPECT_EQ(checks_, (std::vector<std::string>{"ib", "p2p"}));
}

TEST_F(DiagnosticsMicrotest, Run_ReportsDetectAndP2pFailuresButSucceeds) {
  BuildComm(2, 0, 2);
  peerMasks_[1] = Bit(PATH_NET);
  gatherResult_ = ncclRemoteError;
  p2pResult_ = ncclSystemError;
  ncclResult_t ret;
  const std::string out = Run(&ret);
  EXPECT_EQ(ret, ncclSuccess);
  EXPECT_EQ(checks_, (std::vector<std::string>{"p2p"}));
  const std::string expected = kBanner +
                               DiagLine("NCCL DIAG [INFO] transport detect returned " +
                                        std::to_string(ncclRemoteError)) +
                               kNetSkipped +
                               DiagLine("NCCL DIAG [INFO] p2p: check returned " + std::to_string(ncclSystemError));
  EXPECT_EQ(out.substr(0, expected.size()), expected);
}

TEST_F(DiagnosticsMicrotest, Run_ReportsElapsedMillisecondsAcrossRanks) {
  BuildComm(5, 0, 1);
  p2pDelay_ = std::chrono::milliseconds(25);
  ncclResult_t ret;
  const std::string out = Run(&ret);
  const double elapsedMs = ElapsedMs(out, 5);
  EXPECT_GE(elapsedMs, 25.0) << out;
  EXPECT_LT(elapsedMs, 10000.0);
}

}  // namespace
