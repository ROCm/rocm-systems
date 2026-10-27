/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphNullStreamFence hipGraphNullStreamFence
 * @{
 * @ingroup GraphTest
 * Graph instantiation and exec updates form the nodes' AQL packets on the null stream without
 * submitting them there. That must leave the null stream's fence state as it was: every launch into
 * a blocking stream checks the null stream, and adds a cache-flushing barrier packet when the null
 * stream has work whose writes are not yet released to system scope.
 *
 * The cases count the AQL packets the runtime puts on a blocking stream's HW queue between two
 * kernels, from the ring slots of the two kernels' own dispatch packets. The first kernel holds the
 * stream until the second is enqueued, so both run from the same HW queue.
 */

#include <hip_test_common.hh>

#include <vector>

namespace {

constexpr int kElements = 1024;
constexpr unsigned long long kAqlPacketBytes = 64;

struct PacketSlot {
  unsigned long long packet;  // address of the kernel's dispatch packet in the HW queue ring
  unsigned long long queue;   // HW queue the packet was read from
};

__global__ void RecordPacketSlot(PacketSlot* slot, int* gate) {
  if (threadIdx.x == 0) {
    slot->packet = reinterpret_cast<unsigned long long>(__builtin_amdgcn_dispatch_ptr());
    slot->queue = reinterpret_cast<unsigned long long>(__builtin_amdgcn_queue_ptr());
    if (gate != nullptr) {
      while (__hip_atomic_load(gate, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) {
      }
    }
  }
}

__global__ void Fill(int* out, int value, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = value;
  }
}

class NullStreamFenceTest {
 public:
  NullStreamFenceTest() {
    HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamDefault));
    HIP_CHECK(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&no_fence_event_, hipEventDisableSystemFence));
    HIP_CHECK(hipHostMalloc(&slots_, 2 * sizeof(*slots_), hipHostMallocCoherent));
    HIP_CHECK(hipHostMalloc(&gate_, sizeof(*gate_), hipHostMallocCoherent));
    HIP_CHECK(hipMalloc(&src_, kElements * sizeof(int)));
    HIP_CHECK(hipMalloc(&dst_, kElements * sizeof(int)));
  }

  ~NullStreamFenceTest() {
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(dst_));
    HIP_CHECK(hipFree(src_));
    HIP_CHECK(hipHostFree(gate_));
    HIP_CHECK(hipHostFree(slots_));
    HIP_CHECK(hipEventDestroy(no_fence_event_));
    HIP_CHECK(hipEventDestroy(event_));
    HIP_CHECK(hipStreamDestroy(stream_));
  }

  // Leaves the null stream with a completed last command and nothing left to flush: an event
  // record releases at system scope.
  void SettleNullStream() {
    HIP_CHECK(hipEventRecord(event_, nullptr));
    HIP_CHECK(hipEventSynchronize(event_));
  }

  // Leaves the null stream with a kernel whose writes are not released to system scope, followed
  // by a completed event record that carries no system fence either. The record gives the null
  // stream a last command with a HW signal, so later waits on it need no extra null-stream packet.
  void LeaveNullStreamUnflushed() {
    hipLaunchKernelGGL(Fill, dim3(kElements / 256), dim3(256), 0, nullptr, src_, 5, kElements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(no_fence_event_, nullptr));
    HIP_CHECK(hipEventSynchronize(no_fence_event_));
  }

  // Runs |between| while the stream is held between two probe kernels and returns how many AQL
  // packets were put on the stream's HW queue for the second probe's launch, its own included.
  template <typename F> long long PacketsForNextLaunch(const F& between) {
    // Opens the gate on every exit, so a failing check cannot leave the first probe spinning.
    struct GateOpener {
      int* gate;
      ~GateOpener() { __atomic_store_n(gate, 1, __ATOMIC_RELEASE); }
    };
    for (int attempt = 0; attempt < 3; ++attempt) {
      *gate_ = 0;
      {
        GateOpener opener{gate_};
        hipLaunchKernelGGL(RecordPacketSlot, dim3(1), dim3(64), 0, stream_, &slots_[0], gate_);
        HIP_CHECK(hipGetLastError());
        between();
        hipLaunchKernelGGL(RecordPacketSlot, dim3(1), dim3(64), 0, stream_, &slots_[1], nullptr);
        HIP_CHECK(hipGetLastError());
      }
      HIP_CHECK(hipStreamSynchronize(stream_));
      REQUIRE(slots_[0].queue == slots_[1].queue);
      // Retry if the ring wrapped between the two packets.
      if (slots_[1].packet > slots_[0].packet) {
        return static_cast<long long>((slots_[1].packet - slots_[0].packet) / kAqlPacketBytes);
      }
    }
    FAIL("AQL ring wrapped on every attempt");
    return 0;
  }

  void CheckDst(int expected) {
    std::vector<int> host(kElements);
    HIP_CHECK(hipMemcpy(host.data(), dst_, kElements * sizeof(int), hipMemcpyDeviceToHost));
    for (int i = 0; i < kElements; ++i) {
      if (host[i] != expected) {
        INFO("dst[" << i << "] = " << host[i] << ", expected " << expected);
        REQUIRE(host[i] == expected);
      }
    }
  }

  hipStream_t stream() const { return stream_; }
  int* src() const { return src_; }
  int* dst() const { return dst_; }

 private:
  hipStream_t stream_;
  hipEvent_t event_;
  hipEvent_t no_fence_event_;
  PacketSlot* slots_;
  int* gate_;
  int* src_;
  int* dst_;
};

struct FillGraph {
  FillGraph(int* out, int value) : out(out), value(value) {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    params.func = reinterpret_cast<void*>(Fill);
    params.gridDim = dim3(kElements / 256);
    params.blockDim = dim3(256);
    params.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
  }
  ~FillGraph() { HIP_CHECK(hipGraphDestroy(graph)); }

  int* out;
  int value;
  int n = kElements;
  void* args[3] = {&out, &value, &n};
  hipKernelNodeParams params{};
  hipGraph_t graph;
  hipGraphNode_t node;
};

}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *  - With the null stream idle and nothing left to flush, a graph exec update (kernel, memset and
 *    memcpy node; per-node and whole-graph) or a graph instantiation must not add a barrier packet
 *    to the next launch into a blocking stream. The updated graphs must produce the updated output.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphNullStreamFence.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphNullStreamFence_UpdateLeavesNullStreamClean) {
  NullStreamFenceTest test;
  FillGraph fill(test.dst(), 1);
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, fill.graph, nullptr, nullptr, 0));
  test.SettleNullStream();
  const long long baseline = test.PacketsForNextLaunch([] {});
  REQUIRE(baseline >= 1);

  SECTION("hipGraphExecKernelNodeSetParams") {
    for (int value = 2; value < 5; ++value) {
      fill.value = value;
      CHECK(test.PacketsForNextLaunch([&] {
        HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, fill.node, &fill.params));
      }) == baseline);
      HIP_CHECK(hipGraphLaunch(exec, test.stream()));
      HIP_CHECK(hipStreamSynchronize(test.stream()));
      test.CheckDst(value);
      test.SettleNullStream();
    }
  }

  SECTION("hipGraphExecUpdate") {
    for (int value = 2; value < 5; ++value) {
      FillGraph updated(test.dst(), value);
      hipGraphNode_t error_node = nullptr;
      hipGraphExecUpdateResult result;
      CHECK(test.PacketsForNextLaunch([&] {
        HIP_CHECK(hipGraphExecUpdate(exec, updated.graph, &error_node, &result));
      }) == baseline);
      REQUIRE(result == hipGraphExecUpdateSuccess);
      HIP_CHECK(hipGraphLaunch(exec, test.stream()));
      HIP_CHECK(hipStreamSynchronize(test.stream()));
      test.CheckDst(value);
      test.SettleNullStream();
    }
  }

  SECTION("hipGraphInstantiate") {
    hipGraphExec_t second_exec = nullptr;
    CHECK(test.PacketsForNextLaunch([&] {
      HIP_CHECK(hipGraphInstantiate(&second_exec, fill.graph, nullptr, nullptr, 0));
    }) == baseline);
    HIP_CHECK(hipGraphLaunch(second_exec, test.stream()));
    HIP_CHECK(hipStreamSynchronize(test.stream()));
    test.CheckDst(1);
    HIP_CHECK(hipGraphExecDestroy(second_exec));
  }

  SECTION("hipGraphExecMemsetNodeSetParams") {
    hipGraph_t graph;
    hipGraphNode_t node;
    hipGraphExec_t memset_exec;
    hipMemsetParams params{};
    params.dst = test.dst();
    params.elementSize = sizeof(int);
    params.width = kElements;
    params.height = 1;
    params.value = 1;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphInstantiate(&memset_exec, graph, nullptr, nullptr, 0));
    test.SettleNullStream();
    for (int value = 2; value < 5; ++value) {
      params.value = value;
      CHECK(test.PacketsForNextLaunch([&] {
        HIP_CHECK(hipGraphExecMemsetNodeSetParams(memset_exec, node, &params));
      }) == baseline);
      HIP_CHECK(hipGraphLaunch(memset_exec, test.stream()));
      HIP_CHECK(hipStreamSynchronize(test.stream()));
      test.CheckDst(value);
      test.SettleNullStream();
    }
    HIP_CHECK(hipGraphExecDestroy(memset_exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  SECTION("hipGraphExecMemcpyNodeSetParams1D") {
    hipGraph_t graph;
    hipGraphNode_t node;
    hipGraphExec_t memcpy_exec;
    HIP_CHECK(hipMemset(test.src(), 0, kElements * sizeof(int)));
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, test.dst(), test.src(),
                                      kElements * sizeof(int), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipGraphInstantiate(&memcpy_exec, graph, nullptr, nullptr, 0));
    // Fill the source with a known value before the update moves the copy onto it.
    hipLaunchKernelGGL(Fill, dim3(kElements / 256), dim3(256), 0, test.stream(), test.src(), 7,
                       kElements);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(test.stream()));
    test.SettleNullStream();
    CHECK(test.PacketsForNextLaunch([&] {
      HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(memcpy_exec, node, test.dst(), test.src(),
                                                  kElements * sizeof(int),
                                                  hipMemcpyDeviceToDevice));
    }) == baseline);
    HIP_CHECK(hipGraphLaunch(memcpy_exec, test.stream()));
    HIP_CHECK(hipStreamSynchronize(test.stream()));
    test.CheckDst(7);
    HIP_CHECK(hipGraphExecDestroy(memcpy_exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
}

/**
 * Test Description
 * ------------------------
 *  - When the null stream does have work not yet released to system scope (a kernel followed by
 *    an event recorded with hipEventDisableSystemFence), an exec update must keep that state: the
 *    next launch into a blocking stream still gets its barrier packet.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphNullStreamFence.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphNullStreamFence_UpdateKeepsPendingNullStreamFlush) {
  NullStreamFenceTest test;
  FillGraph fill(test.dst(), 1);
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, fill.graph, nullptr, nullptr, 0));
  test.SettleNullStream();
  const long long settled = test.PacketsForNextLaunch([] {});
  test.LeaveNullStreamUnflushed();
  const long long unflushed = test.PacketsForNextLaunch([] {});
  REQUIRE(unflushed > settled);

  fill.value = 2;
  CHECK(test.PacketsForNextLaunch([&] {
    HIP_CHECK(hipGraphExecKernelNodeSetParams(exec, fill.node, &fill.params));
  }) == unflushed);
  HIP_CHECK(hipGraphLaunch(exec, test.stream()));
  HIP_CHECK(hipStreamSynchronize(test.stream()));
  test.CheckDst(2);
  HIP_CHECK(hipGraphExecDestroy(exec));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
