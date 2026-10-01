// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Vendored from ROCm/HIP-Examples mini-nbody/hip/nbody-block.cpp (Apache-2.0).
// Local changes for rocprofiler-compute health workloads (AIPROFCOMP-865):
//   - Accept argv[2] as iteration count (health: nbody-block 131072 500)
//   - Pin host staging buffer (hipHostMalloc); keep device alloc outside loop
//   - Guard avgTime when nIters < 2

#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define BLOCK_SIZE 256
#define SOFTENING 1e-9f

typedef struct { float4 *pos, *vel; } BodySystem;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(float4 *p, float4 *v, float dt, int n) {
  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int tile = 0; tile < hipGridDim_x; tile++) {
      __shared__ float3 spos[BLOCK_SIZE];
      float4 tpos = p[tile * hipBlockDim_x + hipThreadIdx_x];
      spos[hipThreadIdx_x] = make_float3(tpos.x, tpos.y, tpos.z);
      __syncthreads();

      for (int j = 0; j < BLOCK_SIZE; j++) {
        float dx = spos[j].x - p[i].x;
        float dy = spos[j].y - p[i].y;
        float dz = spos[j].z - p[i].z;
        float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        float invDist = 1.0f / sqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
      }
      __syncthreads();
    }

    v[i].x += dt*Fx; v[i].y += dt*Fy; v[i].z += dt*Fz;
  }
}

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t _err = (x);                                                     \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error at %s:%d in '%s': %s\n", __FILE__, __LINE__,  \
              #x, hipGetErrorString(_err));                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(const int argc, const char** argv) {

  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  int nIters = 10;        // simulation iterations (override with argv[2])
  if (argc > 2) nIters = atoi(argv[2]);
  if (nIters < 1) nIters = 1;

  int bytes = 2*nBodies*sizeof(float4);
  // Pinned host staging for H2D/D2H each iter; device buffer allocated once.
  float *buf = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&buf), bytes));
  BodySystem p = { (float4*)buf, ((float4*)buf) + nBodies };

  randomizeBodies(buf, 8*nBodies); // Init pos / vel data

  float *d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, bytes));
  BodySystem d_p = { (float4*)d_buf, ((float4*)d_buf) + nBodies };

  int nBlocks = (nBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    HIP_CHECK(hipMemcpy(d_buf, buf, bytes, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(bodyForce, dim3(nBlocks), dim3(BLOCK_SIZE), 0, 0,
                       d_p.pos, d_p.vel, dt, nBodies);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(buf, d_buf, bytes, hipMemcpyDeviceToHost));

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p.pos[i].x += p.vel[i].x*dt;
      p.pos[i].y += p.vel[i].y*dt;
      p.pos[i].z += p.vel[i].z*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed;
    }
#ifndef SHMOO
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
#endif
  }
  double avgTime = (nIters > 1) ? totalTime / (double)(nIters-1) : totalTime;

#ifdef SHMOO
  printf("%d, %0.3f\n", nBodies,
         (avgTime > 0.0) ? 1e-9 * nBodies * nBodies / avgTime : 0.0);
#else
  printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
         nIters, (avgTime > 0.0) ? 1.0 / avgTime : 0.0);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies,
         (avgTime > 0.0) ? 1e-9 * nBodies * nBodies / avgTime : 0.0);
#endif
  HIP_CHECK(hipHostFree(buf));
  HIP_CHECK(hipFree(d_buf));
  return 0;
}
