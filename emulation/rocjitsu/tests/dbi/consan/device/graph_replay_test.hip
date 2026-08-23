// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Concurrent HIP graph replay distilled from Aorta's independent-subgraph
/// and multi-stream workloads. Two captured graphs repeatedly launch the same
/// code object with distinct state. The behavioral pair changes only the LDS
/// publication edge inside each graph node.

#include "consan_device_test_support.h"

#include <array>

namespace {

constexpr uint32_t kStreams = 2u;
constexpr uint32_t kReplays = 3u;
constexpr uint32_t kMaximumThreads = 128u;

struct StreamState {
  uint32_t launches;
  uint32_t observed[kReplays];
  uint32_t control[kReplays][kMaximumThreads];
};

template <bool Synchronize>
__global__ __launch_bounds__(kMaximumThreads) void graph_replay_for_consan(StreamState *state,
                                                                           uint32_t stream_id) {
  __shared__ uint32_t payload;
  __shared__ uint32_t replay;
  const uint32_t tid = threadIdx.x;
  const uint32_t wave_size = warpSize;

  if (tid == 0u)
    replay = atomicAdd(&state->launches, 1u);
  __syncthreads();

  const uint32_t replay_id = replay;
  if (tid == 0u) {
    consan_device_test::store_lds_word(&payload, 0xc7500000u | (stream_id << 8u) | replay_id);
  }
  if constexpr (Synchronize)
    __syncthreads();
  if (tid == wave_size)
    state->observed[replay_id] = consan_device_test::load_lds_word(&payload);
  state->control[replay_id][tid] = 0xc7000000u | (stream_id << 16u) | (replay_id << 8u) | tid;
}

struct CapturedGraph {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
};

void run_graph_replay(bool synchronize) {
  const uint32_t thread_count = 2u * consan_device_test::active_wave_size();
  ASSERT_LE(thread_count, kMaximumThreads);

  std::array<StreamState *, kStreams> device_states{};
  std::array<CapturedGraph, kStreams> graphs{};
  for (uint32_t stream_id = 0u; stream_id < kStreams; ++stream_id) {
    HIP_ASSERT(hipMalloc(&device_states[stream_id], sizeof(StreamState)));
    HIP_ASSERT(hipMemset(device_states[stream_id], 0, sizeof(StreamState)));
    HIP_ASSERT(hipStreamCreate(&graphs[stream_id].stream));
    HIP_ASSERT(hipStreamBeginCapture(graphs[stream_id].stream, hipStreamCaptureModeGlobal));
    if (synchronize) {
      graph_replay_for_consan<true>
          <<<1, thread_count, 0, graphs[stream_id].stream>>>(device_states[stream_id], stream_id);
    } else {
      graph_replay_for_consan<false>
          <<<1, thread_count, 0, graphs[stream_id].stream>>>(device_states[stream_id], stream_id);
    }
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipStreamEndCapture(graphs[stream_id].stream, &graphs[stream_id].graph));
    HIP_ASSERT(hipGraphInstantiate(&graphs[stream_id].executable, graphs[stream_id].graph, nullptr,
                                   nullptr, 0));
  }

  for (uint32_t replay = 0u; replay < kReplays; ++replay) {
    for (CapturedGraph &graph : graphs)
      HIP_ASSERT(hipGraphLaunch(graph.executable, graph.stream));
    for (CapturedGraph &graph : graphs)
      HIP_ASSERT(hipStreamSynchronize(graph.stream));
  }

  std::array<StreamState, kStreams> results{};
  for (uint32_t stream_id = 0u; stream_id < kStreams; ++stream_id) {
    HIP_ASSERT(hipMemcpy(&results[stream_id], device_states[stream_id], sizeof(StreamState),
                         hipMemcpyDeviceToHost));
    EXPECT_EQ(results[stream_id].launches, kReplays) << "stream=" << stream_id;
    for (uint32_t replay = 0u; replay < kReplays; ++replay) {
      for (uint32_t tid = 0u; tid < thread_count; ++tid) {
        EXPECT_EQ(results[stream_id].control[replay][tid],
                  0xc7000000u | (stream_id << 16u) | (replay << 8u) | tid)
            << "stream=" << stream_id << " replay=" << replay << " thread=" << tid;
      }
      if (synchronize) {
        EXPECT_EQ(results[stream_id].observed[replay], 0xc7500000u | (stream_id << 8u) | replay)
            << "stream=" << stream_id << " replay=" << replay;
      }
    }

    HIP_ASSERT(hipGraphExecDestroy(graphs[stream_id].executable));
    HIP_ASSERT(hipGraphDestroy(graphs[stream_id].graph));
    HIP_ASSERT(hipStreamDestroy(graphs[stream_id].stream));
    HIP_ASSERT(hipFree(device_states[stream_id]));
  }
}

TEST(ConSanDeviceGraphReplayTest, Correct) { run_graph_replay(true); }

TEST(ConSanDeviceGraphReplayTest, Incorrect) { run_graph_replay(false); }

} // namespace
