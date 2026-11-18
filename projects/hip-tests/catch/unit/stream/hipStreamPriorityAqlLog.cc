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
#include <vector>

namespace {

// Env marker that tells a re-executed instance to run the kernel workload that
// produces AQL packets, instead of driving a child.
constexpr char kChildEnv[] = "HIP_TEST_AQL_PRIORITY_CHILD";
constexpr char kBarrierChildEnv[] = "HIP_TEST_AQL_BARRIER_CHILD";
constexpr char kBarrierScenarioEnv[] = "HIP_TEST_AQL_BARRIER_SCENARIO";

__global__ void noopKernel() {}

__global__ void delayedNoopKernel(uint64_t delay_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const uint64_t start_cycle = wall_clock64();
  while (wall_clock64() - start_cycle < delay_cycles) {
  }
}

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

void launchBarrierTestKernel(hipStream_t stream, uint32_t workgroup_size) {
  hipLaunchKernelGGL(noopKernel, dim3(1), dim3(workgroup_size), 0, stream);
  HIP_CHECK(hipGetLastError());
}

void runBarrierScenario(const std::string& scenario) {
  hipStream_t stream_a = nullptr;
  hipStream_t stream_b = nullptr;
  hipStream_t stream_c = nullptr;
  HIP_CHECK(hipStreamCreate(&stream_a));

  if (scenario == "same_stream") {
    launchBarrierTestKernel(stream_a, 17);
    launchBarrierTestKernel(stream_a, 18);
    launchBarrierTestKernel(stream_a, 19);
  } else {
    HIP_CHECK(hipStreamCreate(&stream_b));
    if (scenario == "interleaved_aba") {
      launchBarrierTestKernel(stream_a, 17);
      launchBarrierTestKernel(stream_b, 18);
      launchBarrierTestKernel(stream_a, 19);
    } else if (scenario == "interleaved_abb") {
      launchBarrierTestKernel(stream_a, 17);
      launchBarrierTestKernel(stream_b, 18);
      launchBarrierTestKernel(stream_b, 19);
    } else if (scenario == "three_streams") {
      HIP_CHECK(hipStreamCreate(&stream_c));
      launchBarrierTestKernel(stream_a, 17);
      launchBarrierTestKernel(stream_b, 18);
      launchBarrierTestKernel(stream_c, 19);
      launchBarrierTestKernel(stream_a, 20);
    } else if (scenario == "event_wait") {
      hipEvent_t event = nullptr;
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      constexpr uint64_t kEventWaitDelayCycles = 100000000;
      hipLaunchKernelGGL(delayedNoopKernel, dim3(1), dim3(17), 0, stream_a, kEventWaitDelayCycles);
      HIP_CHECK(hipGetLastError());
      launchBarrierTestKernel(stream_b, 18);
      HIP_CHECK(hipEventRecord(event, stream_a));
      HIP_CHECK(hipStreamWaitEvent(stream_b, event, 0));
      launchBarrierTestKernel(stream_b, 19);
      HIP_CHECK(hipEventDestroy(event));
    } else {
      FAIL("Unknown AQL barrier-bit scenario: " << scenario);
    }
  }

  HIP_CHECK(hipDeviceSynchronize());
  if (stream_c != nullptr) {
    HIP_CHECK(hipStreamDestroy(stream_c));
  }
  if (stream_b != nullptr) {
    HIP_CHECK(hipStreamDestroy(stream_b));
  }
  HIP_CHECK(hipStreamDestroy(stream_a));
}

std::string runBarrierScenarioChild(const std::string& scenario) {
  const std::string self = selfExePath();
  REQUIRE_FALSE(self.empty());
  const std::string log_file =
      "hip_aql_barrier_" + scenario + "_" + std::to_string(getpid()) + ".log";

  const pid_t child_pid = fork();
  REQUIRE(child_pid >= 0);
  if (child_pid == 0) {
    setenv(kBarrierChildEnv, "1", 1);
    setenv(kBarrierScenarioEnv, scenario.c_str(), 1);
    setenv("GPU_MAX_HW_QUEUES", "1", 1);
    setenv("DEBUG_CLR_AQL_BARRIER_OPT", "1", 1);
    setenv("AMD_LOG_LEVEL", "5", 1);
    setenv("AMD_LOG_MASK", "8", 1);

    const int log_fd = open(log_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (log_fd < 0) _exit(127);
    dup2(log_fd, STDERR_FILENO);
    const int dev_null = open("/dev/null", O_WRONLY);
    if (dev_null >= 0) dup2(dev_null, STDOUT_FILENO);

    const char* child_argv[] = {
        self.c_str(), "Unit_hipStreamAqlBarrierBit_ChildWorkload", nullptr};
    execv(self.c_str(), const_cast<char* const*>(child_argv));
    _exit(127);
  }

  int child_status = 0;
  REQUIRE(waitpid(child_pid, &child_status, 0) == child_pid);
  REQUIRE(WIFEXITED(child_status));
  REQUIRE(WEXITSTATUS(child_status) == 0);

  std::ifstream log_stream(log_file);
  REQUIRE(log_stream.good());
  std::stringstream log_contents;
  log_contents << log_stream.rdbuf();
  log_stream.close();
  std::remove(log_file.c_str());
  return log_contents.str();
}

int dispatchBarrierBitForWorkgroup(const std::string& log, uint32_t workgroup_size) {
  const std::string workgroup_token =
      "workgroup=[" + std::to_string(workgroup_size) + ", 1, 1]";
  std::istringstream log_lines(log);
  std::string log_line;
  while (std::getline(log_lines, log_line)) {
    if (log_line.find("Dispatch Header") == std::string::npos ||
        log_line.find(workgroup_token) == std::string::npos) {
      continue;
    }
    if (log_line.find("barrier=0") != std::string::npos) {
      return 0;
    }
    if (log_line.find("barrier=1") != std::string::npos) {
      return 1;
    }
  }
  return -1;
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

HIP_TEST_CASE(Unit_hipStreamAqlBarrierBit_ChildWorkload) {
  if (std::getenv(kBarrierChildEnv) == nullptr) {
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  const char* scenario = std::getenv(kBarrierScenarioEnv);
  REQUIRE(scenario != nullptr);
  runBarrierScenario(scenario);
}

HIP_TEST_CASE(Unit_hipStreamAqlBarrierBit_Scenarios) {
  if (std::getenv(kBarrierChildEnv) != nullptr) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }

  SECTION("Same stream") {
    const std::string log = runBarrierScenarioChild("same_stream");
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 18) == 1);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Interleaved streams A B A") {
    const std::string log = runBarrierScenarioChild("interleaved_aba");
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Interleaved streams A B B") {
    const std::string log = runBarrierScenarioChild("interleaved_abb");
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Three streams A B C A") {
    const std::string log = runBarrierScenarioChild("three_streams");
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 19) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 20) == 1);
  }

  SECTION("Event record and wait preserve stream ordering") {
    const std::string log = runBarrierScenarioChild("event_wait");
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(dispatchBarrierBitForWorkgroup(log, 19) == 1);

    // Runtime init also emits barrier packets before the scenario kernels run.
    // Scope collection to the event record/wait pair between wg18 and wg19.
    const std::string wg18_token = "workgroup=[18, 1, 1]";
    const std::string wg19_token = "workgroup=[19, 1, 1]";
    std::vector<int> barrier_bits;
    std::istringstream log_lines(log);
    std::string log_line;
    bool after_wg18_dispatch = false;
    while (std::getline(log_lines, log_line)) {
      if (!after_wg18_dispatch) {
        if (log_line.find("Dispatch Header") != std::string::npos &&
            log_line.find(wg18_token) != std::string::npos) {
          after_wg18_dispatch = true;
        }
        continue;
      }
      if (log_line.find("Dispatch Header") != std::string::npos &&
          log_line.find(wg19_token) != std::string::npos) {
        break;
      }
      if (log_line.find("Barrier-AND Header") == std::string::npos &&
          log_line.find("BarrierValue Header") == std::string::npos) {
        continue;
      }
      if (log_line.find("barrier=0") != std::string::npos) {
        barrier_bits.push_back(0);
      } else {
        REQUIRE(log_line.find("barrier=1") != std::string::npos);
        barrier_bits.push_back(1);
      }
    }
    REQUIRE(barrier_bits.size() == 2);
    REQUIRE(barrier_bits[0] == 1);
    REQUIRE(barrier_bits[1] == 0);
  }
}

/**
 * End doxygen group StreamTest.
 * @}
 */

#endif  // HT_AMD && HT_LINUX
