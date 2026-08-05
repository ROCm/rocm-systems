/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamPriorityAqlLog hipStreamPriorityAqlLog
 * @{
 * @ingroup StreamTest
 * Validates that the AQL packet trace emitted by the runtime reports the stream
 * priority next to the HWq, for each distinct HW queue priority bucket
 * (Low/Normal/High). The priority is only observable in the AQL debug log
 * (AMD_LOG_LEVEL=5, AMD_LOG_MASK=LOG_AQL), so the test re-executes itself in a
 * child process with that logging enabled, captures the log, and verifies the
 * tokens. AMD + Linux only.
 */

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

// Env marker that tells a re-executed instance to run the kernel workload that
// produces AQL packets, instead of driving a child.
constexpr char kChildEnv[] = "HIP_TEST_AQL_PRIORITY_CHILD";

__global__ void noopKernel() {}

// Launch a kernel on a stream created at the given priority and synchronize, so
// the runtime emits AQL dispatch/barrier packets tagged with that priority.
void launchOnPriority(int priority) {
  hipStream_t stream{nullptr};
  HIP_CHECK(hipStreamCreateWithPriority(&stream, hipStreamDefault, priority));
  hipLaunchKernelGGL(noopKernel, dim3(1), dim3(1), 0, stream);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
}

std::string selfExePath() {
  char buf[4096];
  ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return std::string{};
  buf[n] = '\0';
  return std::string(buf);
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Internal child workload (only runs when HIP_TEST_AQL_PRIORITY_CHILD is set).
 *    Creates streams at High/Normal/Low priority and launches a kernel on each
 *    so the AQL packet logger emits priority-tagged trace lines.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipStreamPriorityAqlLog.cc
 */
HIP_TEST_CASE(Unit_hipStreamPriorityAqlLog_ChildWorkload) {
  if (std::getenv(kChildEnv) == nullptr) {
    // Not the child invocation; nothing to do when run directly.
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  int priorityLow = 0, priorityHigh = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));

  // priorityHigh/priorityLow are the extremes of the supported range; the
  // mid-range value maps to Normal (consistent with other stream-priority tests
  // that treat (low + high) / 2 as normal priority).
  launchOnPriority(priorityHigh);                       // -> priority=3 (High)
  launchOnPriority((priorityLow + priorityHigh) / 2);  // -> priority=1 (Normal)
  launchOnPriority(priorityLow);                        // -> priority=0 (Low)
}

/**
 * Test Description
 * ------------------------
 *  - Re-executes this test binary as a child with AQL logging enabled
 *    (AMD_LOG_LEVEL=5, AMD_LOG_MASK=LOG_AQL), running the child workload that
 *    launches kernels on High/Normal/Low priority streams. Captures the child's
 *    stderr log and verifies that every priority bucket is reported next to the
 *    HWq in the AQL trace.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipStreamPriorityAqlLog.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipStreamPriorityAqlLog_TraceReportsPriority) {
  // Guard against accidental recursion if invoked as a child.
  if (std::getenv(kChildEnv) != nullptr) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }

  int priorityLow = 0, priorityHigh = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
  if (priorityLow == priorityHigh) {
    WARN("Stream priority range not supported. Skipping test.");
    return;
  }

  const std::string self = selfExePath();
  REQUIRE_FALSE(self.empty());

  const std::string logFile = "hip_aql_priority_" + std::to_string(getpid()) + ".log";

  // Launch the child via fork()+execv() (no shell) so the executable path needs
  // no quoting and the child starts a fresh process. AMD_LOG_LEVEL must be set
  // before the child's runtime initializes, which an in-process change could not
  // achieve. AMD_LOG_LEVEL=5 (LOG_DETAIL_DEBUG) + AMD_LOG_MASK=8 (LOG_AQL) emit
  // the AQL packet trace, which now carries "priority=<bucket>" next to "HWq=".
  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    // Child: enable AQL logging, redirect stdout to /dev/null and stderr (where
    // the log is written) to the capture file, then re-exec the child workload.
    setenv(kChildEnv, "1", 1);
    setenv("AMD_LOG_LEVEL", "5", 1);
    setenv("AMD_LOG_MASK", "8", 1);
    const int logFd = open(logFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (logFd < 0) _exit(127);
    dup2(logFd, STDERR_FILENO);
    const int devNull = open("/dev/null", O_WRONLY);
    if (devNull >= 0) dup2(devNull, STDOUT_FILENO);
    const char* argv[] = {self.c_str(), "Unit_hipStreamPriorityAqlLog_ChildWorkload", nullptr};
    execv(self.c_str(), const_cast<char* const*>(argv));
    _exit(127);  // execv only returns on failure
  }

  int status = 0;
  REQUIRE(waitpid(pid, &status, 0) == pid);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);

  std::ifstream in(logFile);
  REQUIRE(in.good());
  std::stringstream ss;
  ss << in.rdbuf();
  const std::string log = ss.str();
  in.close();
  std::remove(logFile.c_str());

  // Require the priority to be reported on an actual AQL packet line, i.e. on
  // the same line as an "HWq=" record. Matching the tokens anywhere in the log
  // independently could pass on unrelated logging that happens to contain them.
  // The runtime prints the raw amd::CommandQueue::Priority enum value:
  // Low=0, Normal=1, High=3.
  auto hwqLineHasPriority = [&log](int priorityValue) {
    const std::string token = "priority=" + std::to_string(priorityValue);
    std::istringstream stream(log);
    std::string line;
    while (std::getline(stream, line)) {
      if (line.find("HWq=") != std::string::npos && line.find(token) != std::string::npos) {
        return true;
      }
    }
    return false;
  };

  // Each priority bucket we launched on must appear on an AQL packet line.
  REQUIRE(hwqLineHasPriority(3));  // High
  REQUIRE(hwqLineHasPriority(1));  // Normal
  REQUIRE(hwqLineHasPriority(0));  // Low
}

/**
 * End doxygen group StreamTest.
 * @}
 */

#endif  // HT_AMD && HT_LINUX
