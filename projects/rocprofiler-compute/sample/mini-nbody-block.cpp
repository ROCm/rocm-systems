// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Compute-bound health workload. The run shape follows the mini-nbody block
// sample in ROCm/HIP-Examples:
// https://github.com/ROCm/HIP-Examples/blob/master/mini-nbody/hip/nbody-block.cpp
// This file is not a copy of that sample.
//
// Usage: mini-nbody-block <bodies> <iterations>
//   health: mini-nbody-block 81920 500

#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

constexpr int kBlockSize = 128;
constexpr float kSoftening = 1e-9f;

namespace {

struct timeval g_timer_start;

void start_timer() { gettimeofday(&g_timer_start, nullptr); }

double elapsed_ms() {
  struct timeval stop {};
  struct timeval elapsed {};
  gettimeofday(&stop, nullptr);
  timersub(&stop, &g_timer_start, &elapsed);
  return elapsed.tv_sec * 1000.0 + elapsed.tv_usec / 1000.0;
}

void fill_random(float* values, int count) {
  for (int index = 0; index < count; ++index) {
    values[index] = 2.0f * (rand() / static_cast<float>(RAND_MAX)) - 1.0f;
  }
}

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t error = (call);                                                 \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s:%d %s: %s\n", __FILE__, __LINE__, #call,   \
              hipGetErrorString(error));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

}  // namespace

__global__ void update_bodies(const float* pos_x, const float* pos_y,
                              const float* pos_z, float* vel_x, float* vel_y,
                              float* vel_z, float step, int body_count) {
  __shared__ float3 tile_positions[kBlockSize];

  int body = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  bool valid_body = body < body_count;
  float force_x = 0.0f;
  float force_y = 0.0f;
  float force_z = 0.0f;
  float origin_x = valid_body ? pos_x[body] : 0.0f;
  float origin_y = valid_body ? pos_y[body] : 0.0f;
  float origin_z = valid_body ? pos_z[body] : 0.0f;

  for (int tile_start = 0; tile_start < body_count;
       tile_start += kBlockSize) {
    int other = tile_start + hipThreadIdx_x;
    int tile_size = body_count - tile_start;
    tile_size = tile_size < kBlockSize ? tile_size : kBlockSize;
    if (hipThreadIdx_x < tile_size) {
      tile_positions[hipThreadIdx_x] =
          make_float3(pos_x[other], pos_y[other], pos_z[other]);
    }
    __syncthreads();

    if (valid_body) {
      for (int tile_index = 0; tile_index < tile_size; ++tile_index) {
        float delta_x = tile_positions[tile_index].x - origin_x;
        float delta_y = tile_positions[tile_index].y - origin_y;
        float delta_z = tile_positions[tile_index].z - origin_z;
        float distance_squared = delta_x * delta_x + delta_y * delta_y +
                                 delta_z * delta_z + kSoftening;
        float inverse_distance = rsqrtf(distance_squared);
        float scale = inverse_distance * inverse_distance * inverse_distance;
        force_x += delta_x * scale;
        force_y += delta_y * scale;
        force_z += delta_z * scale;
      }
    }
    __syncthreads();
  }

  if (valid_body) {
    vel_x[body] += step * force_x;
    vel_y[body] += step * force_y;
    vel_z[body] += step * force_z;
  }
}

int main(int argc, char** argv) {
  int body_count = 81920;
  int iteration_count = 10;
  if (argc > 1) {
    body_count = atoi(argv[1]);
  }
  if (argc > 2) {
    iteration_count = atoi(argv[2]);
  }
  if (body_count < 1) {
    body_count = 1;
  }
  if (iteration_count < 1) {
    iteration_count = 1;
  }

  const float step = 0.01f;
  const int coordinate_count = body_count * 6;
  float* host_state = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&host_state),
                          coordinate_count * sizeof(float)));
  fill_random(host_state, coordinate_count);

  float* device_state = nullptr;
  HIP_CHECK(hipMalloc(&device_state, coordinate_count * sizeof(float)));
  float* pos_x = device_state;
  float* pos_y = pos_x + body_count;
  float* pos_z = pos_y + body_count;
  float* vel_x = pos_z + body_count;
  float* vel_y = vel_x + body_count;
  float* vel_z = vel_y + body_count;

  int block_count = (body_count + kBlockSize - 1) / kBlockSize;
  double timed_seconds = 0.0;
  for (int iteration = 1; iteration <= iteration_count; ++iteration) {
    start_timer();
    HIP_CHECK(hipMemcpy(device_state, host_state,
                        coordinate_count * sizeof(float),
                        hipMemcpyHostToDevice));
    hipLaunchKernelGGL(update_bodies, dim3(block_count), dim3(kBlockSize), 0, 0,
                       pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, step,
                       body_count);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(host_state, device_state,
                        coordinate_count * sizeof(float),
                        hipMemcpyDeviceToHost));
    for (int body = 0; body < body_count; ++body) {
      host_state[body] += host_state[body + 3 * body_count] * step;
      host_state[body + body_count] +=
          host_state[body + 4 * body_count] * step;
      host_state[body + 2 * body_count] +=
          host_state[body + 5 * body_count] * step;
    }
    double seconds = elapsed_ms() / 1000.0;
    if (iteration > 1) {
      timed_seconds += seconds;
    }
    printf("Iteration %d: %.3f seconds\n", iteration, seconds);
  }

  int timed_iterations = iteration_count > 1 ? iteration_count - 1 : 1;
  double average_seconds = timed_seconds / timed_iterations;
  double interactions = average_seconds > 0.0
                            ? 1e-9 * body_count * body_count / average_seconds
                            : 0.0;
  printf("%d bodies: %.3f billion interactions/second\n", body_count,
         interactions);

  HIP_CHECK(hipHostFree(host_state));
  HIP_CHECK(hipFree(device_state));
  return 0;
}
