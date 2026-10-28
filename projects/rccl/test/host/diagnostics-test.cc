/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/diagnostics.cc: ncclDiagChildRun, which the diagnostics checks use to run
// external tools (rdma_topo, lspci, dmesg, ib_write_bw). A tool must run outside the parent rank's MPI/PMI
// session, so the wrapper strips the session's environment before it starts the child.

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include DIAGNOSTICS_CC_PATH

// diagnostics.cc's driver calls these checks. No test here runs it, but coverage keeps it linked.
ncclResult_t ncclDiagP2pRun(struct ncclComm*) { return ncclSuccess; }
void ncclDiagRunIbWriteBw(struct ncclComm*) {}

namespace {

// One variable per prefix that MPI_ENV_PATTERN strips, plus the one exact name it lists.
const char* const kMpiSessionVars[] = {"OMPI_RCCL_DIAG_PROBE", "OPAL_RCCL_DIAG_PROBE", "PMIX_RCCL_DIAG_PROBE",
                                       "PMI_RCCL_DIAG_PROBE",  "PRTE_RCCL_DIAG_PROBE", "PERFTEST_FORCE_MPI"};
// Not a session variable: it must still reach the child.
const char* const kKeptVar = "RCCL_DIAG_PROBE_KEPT";

// Runs `env` through ncclDiagChildRun. Returns the child's environment as "\nNAME=value\n..." so a lookup of
// "\nNAME=" matches whole names only.
std::string DiagChildEnvironment() {
  std::vector<char> output(1 << 20);
  EXPECT_EQ(0, ncclDiagChildRun("env", 10, output.data(), static_cast<int>(output.size())));
  std::string env = "\n" + std::string(output.data());
  EXPECT_LT(env.size(), output.size()) << "the child's environment filled the output buffer";
  return env;
}

bool HasVar(const std::string& env, const char* name) {
  return env.find(std::string("\n") + name + "=") != std::string::npos;
}

}  // namespace

TEST(DiagnosticsMicrotest, ChildRun_NonMpiVariable_ReachesChildAndExitCodePropagates) {
  ASSERT_EQ(0, setenv(kKeptVar, "kept", 1));
  std::string env = DiagChildEnvironment();
  unsetenv(kKeptVar);
  EXPECT_NE(std::string::npos, env.find(std::string("\n") + kKeptVar + "=kept\n"))
      << kKeptVar << " did not reach the diagnostics child";

  char output[64];
  EXPECT_EQ(3, ncclDiagChildRun("sh -c 'exit 3'", 10, output, sizeof(output)));
}

TEST(DiagnosticsMicrotest, ChildRun_MpiSessionVariables_AreStripped) {
  for (const char* name : kMpiSessionVars) ASSERT_EQ(0, setenv(name, "leaked", 1));
  ASSERT_EQ(0, setenv(kKeptVar, "kept", 1));
  std::string env = DiagChildEnvironment();
  for (const char* name : kMpiSessionVars) unsetenv(name);
  unsetenv(kKeptVar);

  EXPECT_TRUE(HasVar(env, kKeptVar)) << kKeptVar << " did not reach the diagnostics child";
  for (const char* name : kMpiSessionVars) {
    EXPECT_FALSE(HasVar(env, name)) << name << " reached the diagnostics child";
  }
}
