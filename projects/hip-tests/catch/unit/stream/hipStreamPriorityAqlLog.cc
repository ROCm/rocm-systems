/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamPriorityAqlLog hipStreamPriorityAqlLog
 * @{
 * @ingroup StreamTest
 * Validates two properties that are only observable in the AQL packet trace:
 * that the trace reports the stream priority next to the HWq for each priority
 * bucket (Low/Normal/High), and that the AQL barrier bit on a dispatch follows
 * stream ordering. Each workload runs in a fresh child because tracing is
 * configured during runtime initialization. AMD + Linux only.
 */

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX

#include <hip_test_aqllog.hh>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace aqllog = hip::aqllog;

constexpr char kScenarioEnv[] = "HIP_TEST_AQL_SCENARIO";

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

// The workgroup size identifies this dispatch in the trace.
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
  return aqllog::RunChildTest(
      "Unit_hipStreamAqlBarrierBit_ChildWorkload",
      {{kScenarioEnv, scenario}, {"GPU_MAX_HW_QUEUES", "1"}, {"DEBUG_CLR_AQL_BARRIER_OPT", "1"}});
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Internal child workload (only runs when driven by a parent case).
 *    Creates streams at High/Normal/Low priority and launches a kernel on each
 *    so the AQL packet logger emits priority-tagged trace lines.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipStreamPriorityAqlLog.cc
 */
HIP_TEST_CASE(Unit_hipStreamPriorityAqlLog_ChildWorkload) {
  if (!aqllog::RunningAsChild()) {
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  int priorityLow = 0, priorityHigh = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));

  // priorityHigh/priorityLow are the extremes of the supported range; the
  // mid-range value maps to Normal (consistent with other stream-priority tests
  // that treat (low + high) / 2 as normal priority).
  launchOnPriority(priorityHigh);                      // -> priority=3 (High)
  launchOnPriority((priorityLow + priorityHigh) / 2);  // -> priority=1 (Normal)
  launchOnPriority(priorityLow);                       // -> priority=0 (Low)
}

/**
 * Test Description
 * ------------------------
 *  - Re-executes this test binary as a child with AQL tracing enabled, running
 *    the child workload that launches kernels on High/Normal/Low priority
 *    streams. Captures the child's trace and verifies that every priority
 *    bucket is reported next to the HWq.
 * Test source
 * ------------------------
 *  - catch\unit\stream\hipStreamPriorityAqlLog.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipStreamPriorityAqlLog_TraceReportsPriority) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }

  int priorityLow = 0, priorityHigh = 0;
  HIP_CHECK(hipDeviceGetStreamPriorityRange(&priorityLow, &priorityHigh));
  if (priorityLow == priorityHigh) {
    WARN("Stream priority range not supported. Skipping test.");
    return;
  }

  const std::string log = aqllog::RunChildTest("Unit_hipStreamPriorityAqlLog_ChildWorkload");

  // Require the priority to be reported on an actual AQL packet line, i.e. on
  // the same line as an "HWq=" record. Matching the tokens anywhere in the log
  // independently could pass on unrelated logging that happens to contain them.
  // The runtime prints the raw amd::CommandQueue::Priority enum value:
  // Low=0, Normal=1, High=3.
  auto hwqLineHasPriority = [&log](int priorityValue) {
    const std::string token = "priority=" + std::to_string(priorityValue);
    for (const aqllog::Packet& packet : aqllog::ParsePackets(log)) {
      if (packet.line.find("HWq=") != std::string::npos &&
          packet.line.find(token) != std::string::npos) {
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
  if (!aqllog::RunningAsChild()) {
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  const char* scenario = std::getenv(kScenarioEnv);
  REQUIRE(scenario != nullptr);
  runBarrierScenario(scenario);
}

HIP_TEST_CASE(Unit_hipStreamAqlBarrierBit_Scenarios) {
  if (aqllog::RunningAsChild()) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }

  SECTION("Same stream") {
    const std::string log = runBarrierScenarioChild("same_stream");
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 18) == 1);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Interleaved streams A B A") {
    const std::string log = runBarrierScenarioChild("interleaved_aba");
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Interleaved streams A B B") {
    const std::string log = runBarrierScenarioChild("interleaved_abb");
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 19) == 1);
  }

  SECTION("Three streams A B C A") {
    const std::string log = runBarrierScenarioChild("three_streams");
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 19) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 20) == 1);
  }

  SECTION("Event record and wait preserve stream ordering") {
    const std::string log = runBarrierScenarioChild("event_wait");
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 17) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 18) == 0);
    REQUIRE(aqllog::DispatchBarrierBitForWorkgroup(log, 19) == 1);

    // Runtime init also emits barrier packets before the scenario kernels run,
    // so scope collection to the event record/wait pair between wg18 and wg19.
    const std::vector<aqllog::Packet> barriers = aqllog::BarriersBetweenDispatches(
        log, aqllog::WorkgroupToken(18), aqllog::WorkgroupToken(19));
    REQUIRE(barriers.size() == 2);
    REQUIRE(barriers[0].barrier == 1);
    REQUIRE(barriers[1].barrier == 0);
  }
}

/**
 * End doxygen group StreamTest.
 * @}
 */

#endif  // HT_AMD && HT_LINUX
