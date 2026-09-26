/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * Exercises ordered AQL ring doorbells with two producers on one hardware queue.
 * When the runtime orders them, a producer holding a slot behind an earlier
 * reservation must not ring the doorbell until that earlier reservation has rung
 * its own, so no doorbell can send the packet processor at a slot that is still
 * being written.
 *
 * A graph of many kernels reserves all of its slots with a single write-index
 * bump, so it holds a large reservation open for as long as it takes to write
 * the packets. A second thread reserving one slot behind it therefore blocks
 * inside its launch call, which makes the ordering observable as launch
 * latency. With ordering disabled that thread rings immediately.
 *
 * The stream-to-hardware-queue mapping and the ordering flag are both read when
 * the runtime initializes, so the test re-executes itself in a child process
 * with GPU_MAX_HW_QUEUES=1 and DEBUG_CLR_ORDER_DOORBELL set to 2 (order every
 * host) and then 0 (never order), and compares the two runs. AMD + Linux only.
 */

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

// Env marker that tells a re-executed instance to run the two-producer
// workload, instead of driving a child.
constexpr char kChildEnv[] = "HIP_TEST_ORDERED_DOORBELL_CHILD";

// DEBUG_CLR_ORDER_DOORBELL selects which hosts get ordered publication: 0 none,
// 1 Intel hosts only, 2 every host. Forcing 2 exercises the ordered path
// whatever the host CPU vendor is; 0 is the control run.
constexpr char kOrderDoorbellAllHosts[] = "2";
constexpr char kOrderDoorbellNever[] = "0";

// Kernels per graph launch. The reservation has to stay open long enough for
// the second thread to reserve behind it while packets are still being written,
// so the count needs to be large; it also has to fit the AQL ring.
constexpr int kGraphKernels = 4096;
constexpr int kIterations = 20;
constexpr int kWarmupIterations = 5;
constexpr int kSerialIterations = 3;
constexpr int kBaselineLaunches = 200;

// How long the single-kernel thread waits after the rendezvous so the graph
// thread wins the reservation. It must stay well below the time that thread
// needs to write its packets, or the graph has already published by the time
// this launch reserves and there is nothing left to wait for.
constexpr double kReservationDelayUs = 5.0;

// The wait the follower inherits is the leader's remaining packet-write time,
// which is orders of magnitude above an uncontended launch. Require only a
// modest factor, so the test reports the ordering, not the host's speed.
constexpr double kMinOrderedSlowdown = 3.0;

// Guards against a lost doorbell or a publication deadlock hanging CI. A live
// run of this workload completes in a fraction of this.
constexpr int kConcurrentPhaseTimeoutSeconds = 300;

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

__global__ void incrementKernel(unsigned int* counter) { atomicAdd(counter, 1u); }

void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

// Sense-reversing barrier. Spins rather than sleeping so the rendezvous does
// not add latency of its own to the measurement that follows it. A thread that
// hits a HIP error aborts the barrier, so the other thread stops waiting for a
// rendezvous that will never come.
class SpinBarrier {
 public:
  explicit SpinBarrier(int num_threads) : num_threads_(num_threads) {}

  // Returns false once any thread has aborted.
  bool arriveAndWait() {
    if (aborted_.load(std::memory_order_acquire)) return false;
    const int generation = generation_.load(std::memory_order_acquire);
    if (arrived_.fetch_add(1, std::memory_order_acq_rel) == num_threads_ - 1) {
      arrived_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
      return true;
    }
    while (generation_.load(std::memory_order_acquire) == generation) {
      if (aborted_.load(std::memory_order_acquire)) return false;
      cpuRelax();
    }
    return true;
  }

  void abort() { aborted_.store(true, std::memory_order_release); }

 private:
  const int num_threads_;
  std::atomic<int> arrived_{0};
  std::atomic<int> generation_{0};
  std::atomic<bool> aborted_{false};
};

// Catch2 assertions are not thread safe, so the worker threads record the first
// HIP failure here and the main thread turns it into an assertion.
struct WorkerStatus {
  hipError_t error = hipSuccess;
  std::string failed_call;
};

#define WORKER_CHECK(status, barrier, expr)                                                        \
  do {                                                                                             \
    const hipError_t error = (expr);                                                               \
    if (error != hipSuccess) {                                                                     \
      (status).error = error;                                                                      \
      (status).failed_call = #expr;                                                                \
      (barrier).abort();                                                                           \
      return;                                                                                      \
    }                                                                                              \
  } while (0)

void spinFor(double micros) {
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(Micros(micros));
  while (Clock::now() < deadline) {
    cpuRelax();
  }
}

struct Stats {
  double p50;
  double max;
};

Stats summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t median_index = (samples.size() - 1) / 2;
  return {samples[median_index], samples.back()};
}

// Records every kernel of the graph onto a capture stream, then instantiates it.
hipGraphExec_t buildGraph(hipStream_t capture_stream, unsigned int* counter) {
  HIP_CHECK(hipStreamBeginCapture(capture_stream, hipStreamCaptureModeThreadLocal));
  for (int kernel_idx = 0; kernel_idx < kGraphKernels; ++kernel_idx) {
    hipLaunchKernelGGL(incrementKernel, dim3(1), dim3(1), 0, capture_stream, counter);
  }
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipStreamEndCapture(capture_stream, &graph));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
  return graph_exec;
}

unsigned int readCounter(const unsigned int* counter) {
  unsigned int host_counter = 0;
  HIP_CHECK(hipMemcpy(&host_counter, counter, sizeof(host_counter), hipMemcpyDeviceToHost));
  return host_counter;
}

// One producer at a time on the shared queue: every reservation is published
// before the next one is taken, so the ordered path never has to wait.
void runSerialPhase(hipGraphExec_t graph_exec, hipStream_t graph_stream, hipStream_t single_stream,
                    unsigned int* counter) {
  HIP_CHECK(hipMemset(counter, 0, sizeof(unsigned int)));
  for (int iteration = 0; iteration < kSerialIterations; ++iteration) {
    HIP_CHECK(hipGraphLaunch(graph_exec, graph_stream));
    HIP_CHECK(hipStreamSynchronize(graph_stream));
    hipLaunchKernelGGL(incrementKernel, dim3(1), dim3(1), 0, single_stream, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(single_stream));
  }
  REQUIRE(readCounter(counter) ==
          static_cast<unsigned int>(kSerialIterations) * (kGraphKernels + 1));
}

// Both producers push to the shared queue at once, with the single-kernel
// thread aiming to reserve behind the graph's block. Returns the latency of the
// contended launches; the counter tells us no dispatch was lost on the way.
std::vector<double> runConcurrentPhase(hipGraphExec_t graph_exec, hipStream_t graph_stream,
                                       hipStream_t single_stream, unsigned int* counter,
                                       Stats* graph_launch_stats) {
  HIP_CHECK(hipMemset(counter, 0, sizeof(unsigned int)));

  std::vector<double> graph_launch_us(kIterations, 0.0);
  std::vector<double> contended_launch_us(kIterations, 0.0);
  WorkerStatus graph_status;
  WorkerStatus single_status;
  SpinBarrier barrier(2);

  const auto phase_start = Clock::now();

  // Reserves a large block of slots, then publishes it packet by packet.
  std::thread graph_thread([&]() {
    for (int warmup_idx = 0; warmup_idx < kWarmupIterations; ++warmup_idx) {
      WORKER_CHECK(graph_status, barrier, hipGraphLaunch(graph_exec, graph_stream));
      WORKER_CHECK(graph_status, barrier, hipStreamSynchronize(graph_stream));
    }
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      if (!barrier.arriveAndWait()) return;
      const auto start = Clock::now();
      WORKER_CHECK(graph_status, barrier, hipGraphLaunch(graph_exec, graph_stream));
      graph_launch_us[iteration] = Micros(Clock::now() - start).count();
      WORKER_CHECK(graph_status, barrier, hipStreamSynchronize(graph_stream));
      if (!barrier.arriveAndWait()) return;
    }
  });

  // Reserves one slot behind the graph, so ordered doorbells make it wait for
  // the graph's packets to be written and rung before it can ring its own.
  std::thread single_thread([&]() {
    for (int warmup_idx = 0; warmup_idx < kWarmupIterations; ++warmup_idx) {
      hipLaunchKernelGGL(incrementKernel, dim3(1), dim3(1), 0, single_stream, counter);
      WORKER_CHECK(single_status, barrier, hipGetLastError());
      WORKER_CHECK(single_status, barrier, hipStreamSynchronize(single_stream));
    }
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      if (!barrier.arriveAndWait()) return;
      // Let the graph thread win the reservation, so this slot sits behind its
      // whole block.
      spinFor(kReservationDelayUs);
      const auto start = Clock::now();
      hipLaunchKernelGGL(incrementKernel, dim3(1), dim3(1), 0, single_stream, counter);
      contended_launch_us[iteration] = Micros(Clock::now() - start).count();
      WORKER_CHECK(single_status, barrier, hipGetLastError());
      WORKER_CHECK(single_status, barrier, hipStreamSynchronize(single_stream));
      if (!barrier.arriveAndWait()) return;
    }
  });

  graph_thread.join();
  single_thread.join();

  const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - phase_start).count();

  INFO("graph thread failing call: " << graph_status.failed_call);
  REQUIRE(graph_status.error == hipSuccess);
  INFO("single-kernel thread failing call: " << single_status.failed_call);
  REQUIRE(single_status.error == hipSuccess);
  REQUIRE(elapsed_seconds < kConcurrentPhaseTimeoutSeconds);

  const int total_launches = kWarmupIterations + kIterations;
  REQUIRE(readCounter(counter) == static_cast<unsigned int>(total_launches) * (kGraphKernels + 1));

  *graph_launch_stats = summarize(graph_launch_us);
  return contended_launch_us;
}

std::string selfExePath() {
  char buf[4096];
  const ssize_t length = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (length <= 0) return std::string{};
  buf[length] = '\0';
  return std::string(buf);
}

struct ChildSummary {
  double baseline_p50 = 0.0;
  double contended_p50 = 0.0;
  double contended_max = 0.0;
  double graph_launch_p50 = 0.0;
};

// Reads back the single "SUMMARY key=value ..." line the child prints.
ChildSummary parseChildSummary(const std::string& output) {
  std::istringstream output_lines(output);
  std::string line;
  while (std::getline(output_lines, line)) {
    if (line.compare(0, 8, "SUMMARY ") != 0) continue;
    ChildSummary summary;
    std::istringstream tokens(line.substr(8));
    std::string token;
    while (tokens >> token) {
      const size_t separator = token.find('=');
      if (separator == std::string::npos) continue;
      const std::string key = token.substr(0, separator);
      const double value = std::atof(token.c_str() + separator + 1);
      if (key == "baseline_p50")
        summary.baseline_p50 = value;
      else if (key == "contended_p50")
        summary.contended_p50 = value;
      else if (key == "contended_max")
        summary.contended_max = value;
      else if (key == "graph_launch_p50")
        summary.graph_launch_p50 = value;
    }
    return summary;
  }
  return ChildSummary{};
}

// Runs the workload in a fresh process, so GPU_MAX_HW_QUEUES and the ordering
// flag are in place before that process initializes the runtime.
ChildSummary runChild(const char* order_doorbell) {
  const std::string self = selfExePath();
  REQUIRE_FALSE(self.empty());
  const std::string output_file = "hip_ordered_doorbell_" + std::string(order_doorbell) + "_" +
                                  std::to_string(getpid()) + ".log";

  const pid_t child_pid = fork();
  REQUIRE(child_pid >= 0);
  if (child_pid == 0) {
    setenv(kChildEnv, "1", 1);
    // Both streams have to land on the same hardware queue; separate rings have
    // no shared reserve index and therefore no ordering to observe.
    setenv("GPU_MAX_HW_QUEUES", "1", 1);
    setenv("DEBUG_CLR_ORDER_DOORBELL", order_doorbell, 1);

    const int output_fd = open(output_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (output_fd < 0) _exit(127);
    dup2(output_fd, STDOUT_FILENO);
    dup2(output_fd, STDERR_FILENO);

    const char* child_argv[] = {self.c_str(), "Unit_hipMultiThreadOrderedDoorbell_ChildWorkload",
                                nullptr};
    execv(self.c_str(), const_cast<char* const*>(child_argv));
    _exit(127);
  }

  int child_status = 0;
  REQUIRE(waitpid(child_pid, &child_status, 0) == child_pid);

  std::ifstream output_stream(output_file);
  REQUIRE(output_stream.good());
  std::stringstream output_contents;
  output_contents << output_stream.rdbuf();
  const std::string output = output_contents.str();
  output_stream.close();
  std::remove(output_file.c_str());

  INFO("child output:\n" << output);
  REQUIRE(WIFEXITED(child_status));
  REQUIRE(WEXITSTATUS(child_status) == 0);

  const ChildSummary summary = parseChildSummary(output);
  REQUIRE(summary.baseline_p50 > 0.0);
  REQUIRE(summary.contended_p50 > 0.0);
  return summary;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Internal child workload, only runs when HIP_TEST_ORDERED_DOORBELL_CHILD is
 *    set by the parent test. Two streams share one hardware queue; one thread
 *    launches a graph of many kernels and the other launches a single kernel
 *    just after it, both serially and concurrently. Checks that every dispatch
 *    ran, then prints the launch latencies for the parent to compare.
 * Test source
 * ------------------------
 *  - catch/unit/multiThread/hipMultiThreadOrderedDoorbell.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipMultiThreadOrderedDoorbell_ChildWorkload) {
  if (std::getenv(kChildEnv) == nullptr) {
    SUCCEED("Skipping child workload outside of parent-driven run");
    return;
  }

  hipStream_t graph_stream = nullptr;
  hipStream_t single_stream = nullptr;
  hipStream_t capture_stream = nullptr;
  HIP_CHECK(hipStreamCreate(&graph_stream));
  HIP_CHECK(hipStreamCreate(&single_stream));
  HIP_CHECK(hipStreamCreate(&capture_stream));

  unsigned int* counter = nullptr;
  HIP_CHECK(hipMalloc(&counter, sizeof(unsigned int)));

  hipGraphExec_t graph_exec = buildGraph(capture_stream, counter);

  runSerialPhase(graph_exec, graph_stream, single_stream, counter);

  // Uncontended cost of the single-kernel launch, with the graph thread idle.
  std::vector<double> baseline_launch_us;
  baseline_launch_us.reserve(kBaselineLaunches);
  for (int launch_idx = 0; launch_idx < kBaselineLaunches + kWarmupIterations; ++launch_idx) {
    const auto start = Clock::now();
    hipLaunchKernelGGL(incrementKernel, dim3(1), dim3(1), 0, single_stream, counter);
    const double elapsed_us = Micros(Clock::now() - start).count();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(single_stream));
    if (launch_idx >= kWarmupIterations) {
      baseline_launch_us.push_back(elapsed_us);
    }
  }

  Stats graph_launch{};
  const std::vector<double> contended_launch_us =
      runConcurrentPhase(graph_exec, graph_stream, single_stream, counter, &graph_launch);

  const Stats baseline = summarize(baseline_launch_us);
  const Stats contended = summarize(contended_launch_us);
  std::printf(
      "SUMMARY baseline_p50=%.1f contended_p50=%.1f contended_max=%.1f graph_launch_p50=%.1f\n",
      baseline.p50, contended.p50, contended.max, graph_launch.p50);
  std::fflush(stdout);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipFree(counter));
  HIP_CHECK(hipStreamDestroy(capture_stream));
  HIP_CHECK(hipStreamDestroy(single_stream));
  HIP_CHECK(hipStreamDestroy(graph_stream));
}

/**
 * Test Description
 * ------------------------
 *  - Runs the two-producer workload twice in child processes that share one
 *    hardware queue, once with DEBUG_CLR_ORDER_DOORBELL=2 and once with =0, and
 *    verifies that ordering is observable: the thread that reserves behind the
 *    graph's block waits for it to be published, which shows up as launch
 *    latency well above both its own uncontended launch and the same launch
 *    with ordering disabled.
 * Test source
 * ------------------------
 *  - catch/unit/multiThread/hipMultiThreadOrderedDoorbell.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipMultiThreadOrderedDoorbell_LateProducerWaitsForReservation) {
  // Guard against recursion if this case is picked up in a child invocation.
  if (std::getenv(kChildEnv) != nullptr) {
    SUCCEED("Skipping parent driver inside child invocation");
    return;
  }

  const ChildSummary ordered = runChild(kOrderDoorbellAllHosts);
  const ChildSummary unordered = runChild(kOrderDoorbellNever);

  INFO("ordered:   baseline p50 " << ordered.baseline_p50 << " us, behind graph p50 "
                                  << ordered.contended_p50 << " us (max " << ordered.contended_max
                                  << " us), graph launch p50 " << ordered.graph_launch_p50
                                  << " us");
  INFO("unordered: baseline p50 " << unordered.baseline_p50 << " us, behind graph p50 "
                                  << unordered.contended_p50 << " us (max "
                                  << unordered.contended_max << " us), graph launch p50 "
                                  << unordered.graph_launch_p50 << " us");

  // With ordering on, the launch behind the graph reservation carries the wait
  // for that reservation to be published.
  REQUIRE(ordered.contended_p50 > kMinOrderedSlowdown * ordered.baseline_p50);
  // And that wait is what the flag adds: without it the same launch publishes
  // straight away.
  REQUIRE(ordered.contended_p50 > kMinOrderedSlowdown * unordered.contended_p50);
}

#endif  // HT_AMD && HT_LINUX
