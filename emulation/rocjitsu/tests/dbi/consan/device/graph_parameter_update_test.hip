// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// ConSan publication behavior through the two HIP graph kernel-parameter
/// update paths exercised by Aorta's HRX launch probes. Both updates use the
/// pre-packed `extra` argument-buffer path rather than the argument array used
/// to create the node.

#include "consan_device_test_support.h"

namespace {

constexpr uint32_t kStates = 3u;
constexpr uint32_t kMaximumThreads = 128u;
constexpr uint32_t kInitialMarker = 0x00110000u;
constexpr uint32_t kGraphMarker = 0x00220000u;
constexpr uint32_t kExecutableMarker = 0x00330000u;

struct UpdateState {
  uint32_t observed;
  uint32_t control[kMaximumThreads];
};

struct KernelArguments {
  UpdateState *state;
  uint32_t marker;
};

template <bool Synchronize>
__global__
__launch_bounds__(kMaximumThreads) void graph_parameter_update_for_consan(UpdateState *state,
                                                                          uint32_t marker) {
  __shared__ uint32_t payload;
  const uint32_t tid = threadIdx.x;
  if (tid == 0u)
    consan_device_test::store_lds_word(&payload, marker | 0x5au);
  if constexpr (Synchronize)
    __syncthreads();
  if (tid == warpSize)
    state->observed = consan_device_test::load_lds_word(&payload);
  state->control[tid] = marker | tid;
}

hipKernelNodeParams make_node_params(void *function, uint32_t thread_count,
                                     void **kernel_parameters, void **extra) {
  hipKernelNodeParams params{};
  params.func = function;
  params.gridDim = dim3(1u, 1u, 1u);
  params.blockDim = dim3(thread_count, 1u, 1u);
  params.sharedMemBytes = 0u;
  params.kernelParams = kernel_parameters;
  params.extra = extra;
  return params;
}

void run_graph_parameter_update(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);

  UpdateState *device_states = nullptr;
  HIP_ASSERT(hipMalloc(&device_states, kStates * sizeof(UpdateState)));
  HIP_ASSERT(hipMemset(device_states, 0, kStates * sizeof(UpdateState)));

  void *function = synchronize ? (void *)graph_parameter_update_for_consan<true>
                               : (void *)graph_parameter_update_for_consan<false>;
  UpdateState *initial_state = &device_states[0];
  uint32_t initial_marker = kInitialMarker;
  void *initial_parameters[] = {&initial_state, &initial_marker};

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  HIP_ASSERT(hipGraphCreate(&graph, 0u));
  hipKernelNodeParams initial_params =
      make_node_params(function, thread_count, initial_parameters, nullptr);
  HIP_ASSERT(hipGraphAddKernelNode(&node, graph, nullptr, 0u, &initial_params));

  KernelArguments graph_arguments{&device_states[1], kGraphMarker};
  size_t graph_argument_size = sizeof(graph_arguments);
  void *graph_configuration[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER,
      &graph_arguments,
      HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &graph_argument_size,
      HIP_LAUNCH_PARAM_END,
  };
  hipKernelNodeParams graph_params =
      make_node_params(function, thread_count, nullptr, graph_configuration);
  HIP_ASSERT(hipGraphKernelNodeSetParams(node, &graph_params));

  hipGraphExec_t executable = nullptr;
  HIP_ASSERT(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0u));
  HIP_ASSERT(hipGraphLaunch(executable, nullptr));
  HIP_ASSERT(hipDeviceSynchronize());

  KernelArguments executable_arguments{&device_states[2], kExecutableMarker};
  size_t executable_argument_size = sizeof(executable_arguments);
  void *executable_configuration[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, &executable_arguments, HIP_LAUNCH_PARAM_BUFFER_SIZE,
      &executable_argument_size,       HIP_LAUNCH_PARAM_END,
  };
  hipKernelNodeParams executable_params =
      make_node_params(function, thread_count, nullptr, executable_configuration);
  HIP_ASSERT(hipGraphExecKernelNodeSetParams(executable, node, &executable_params));
  HIP_ASSERT(hipGraphLaunch(executable, nullptr));
  HIP_ASSERT(hipDeviceSynchronize());

  UpdateState results[kStates]{};
  HIP_ASSERT(hipMemcpy(results, device_states, sizeof(results), hipMemcpyDeviceToHost));

  // The node's original argument array must not receive either launch.
  EXPECT_EQ(results[0].observed, 0u);
  for (uint32_t tid = 0u; tid < thread_count; ++tid)
    EXPECT_EQ(results[0].control[tid], 0u) << "thread=" << tid;

  constexpr uint32_t markers[] = {kGraphMarker, kExecutableMarker};
  for (uint32_t update = 0u; update < 2u; ++update) {
    const UpdateState &result = results[update + 1u];
    for (uint32_t tid = 0u; tid < thread_count; ++tid) {
      EXPECT_EQ(result.control[tid], markers[update] | tid)
          << "update=" << update << " thread=" << tid;
    }
    if (synchronize)
      EXPECT_EQ(result.observed, markers[update] | 0x5au) << "update=" << update;
  }

  HIP_ASSERT(hipGraphExecDestroy(executable));
  HIP_ASSERT(hipGraphDestroy(graph));
  HIP_ASSERT(hipFree(device_states));
}

TEST(ConSanDeviceGraphParameterUpdateTest, Correct) { run_graph_parameter_update(true); }

TEST(ConSanDeviceGraphParameterUpdateTest, Incorrect) { run_graph_parameter_update(false); }

} // namespace
