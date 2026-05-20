/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Workload (direct GPU implementations)
 * @{
 * @ingroup HRRTest
 * Direct GPU test implementations — hidden from the default CTest run with the
 * Catch2 [.] tag so they are NOT automatically discovered as CTest tests.
 *
 * They are invoked in two ways:
 *   1. As subprocesses by Unit_HRR_GpuWorkload / Unit_HRR_GraphWorkload /
 *      Unit_HRR_CaptureReplayRoundtrip / Unit_HRR_GraphRoundtrip in
 *      hrr_roundtrip.cc.  CreateProcess gives each subprocess a clean Windows
 *      environment, avoiding MSYS2/bash SEH-exception-handling interference.
 *   2. Directly from PowerShell / cmd.exe for manual validation:
 *        HrrTest.exe "Unit_HRR_GpuWorkload_Direct"
 *        HrrTest.exe "Unit_HRR_GraphWorkload_Direct"
 */

#include <hip_test_common.hh>

// ---------------------------------------------------------------------------
// Workload parameters
// ---------------------------------------------------------------------------

static constexpr int    N            = 1 << 12;   // 4K floats (16 KB)
static constexpr size_t SZ           = N * sizeof(float);
static constexpr int    KERNEL_ITERS = 4;
static constexpr int    GRAPH_ITERS  = 4;

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

__global__ void hrr_vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void hrr_vectorSaxpy(float alpha, const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = alpha * x[i] + y[i];
}

__global__ void hrr_vectorScale(const float* in, float* out, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * s;
}

__global__ void hrr_vectorFill(float* out, float val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

__global__ void hrr_memcpyKernel(const float* src, float* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

__global__ void hrr_dotPartial(const float* a, const float* b, float* partials, int n) {
  extern __shared__ float smem[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = (i < n) ? a[i] * b[i] : 0.f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];
}

// ---------------------------------------------------------------------------
// Direct GPU workload test — hidden ([.]) so CTest does not auto-discover it.
//
// H2D → vectorSaxpy → vectorAdd × KERNEL_ITERS → D2D → dotPartial → D2H
// Expected: hc[i] == 2.0f
//
// When called with HIP_HRR_CAPTURE_OUTPUT set the D2H memcpy is recorded as a
// blob; hrr-playback validates the replayed buffer matches it.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_GpuWorkload_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  hipStream_t s0, s1;
  HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N];
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  float *da, *db, *dc, *dd, *dp;
  HIP_CHECK(hipMalloc(&da, SZ));
  HIP_CHECK(hipMalloc(&db, SZ));
  HIP_CHECK(hipMalloc(&dc, SZ));
  HIP_CHECK(hipMalloc(&dd, SZ));
  HIP_CHECK(hipMalloc(&dp, SZ));

  dim3 block(256), grid((N + 255) / 256);
  int nblocks = static_cast<int>(grid.x);

  HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, s0));
  HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, s0));
  HIP_CHECK(hipMemsetAsync(dc, 0, SZ, s1));
  HIP_CHECK(hipStreamSynchronize(s0));
  HIP_CHECK(hipStreamSynchronize(s1));

  // saxpy: dc = 2*da + dc = 2*1 + 0 = 2
  hipLaunchKernelGGL(hrr_vectorSaxpy, grid, block, 0, s0, 2.0f, da, dc, N);
  HIP_CHECK(hipGetLastError());

  // vectorAdd overwrites dc each iter: dc = da + db = 1 + 1 = 2
  for (int iter = 0; iter < KERNEL_ITERS; ++iter) {
    hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, s0, da, db, dc, N);
    HIP_CHECK(hipGetLastError());
  }

  HIP_CHECK(hipStreamSynchronize(s0));
  // D2D copy exercises that event type in the capture stream
  HIP_CHECK(hipMemcpyAsync(dd, dc, SZ, hipMemcpyDeviceToDevice, s1));

  hipLaunchKernelGGL(hrr_dotPartial, grid, block, block.x * sizeof(float), s0,
                     da, db, dp, N);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(hrr_vectorScale, grid, block, 0, s0, dp, dp,
                     1.0f / nblocks, N);
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipStreamSynchronize(s1));
  // D2H — blob captured here when HIP_HRR_CAPTURE_OUTPUT is set
  HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, s0));
  HIP_CHECK(hipStreamSynchronize(s0));

  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
  HIP_CHECK(hipFree(dd)); HIP_CHECK(hipFree(dp));
  HIP_CHECK(hipStreamDestroy(s0));
  HIP_CHECK(hipStreamDestroy(s1));
  delete[] ha; delete[] hb; delete[] hc;
}

// ---------------------------------------------------------------------------
// Direct HIP graph workload test — hidden ([.]).
//
// Graph: fill → saxpy(3x) → add → scale(0.5x) → memcpyKernel → D2D →
//        add → saxpy(-1x) → add
// Expected: hc[i] == 2.0f
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_GraphWorkload_Direct", "[.][hrr-direct]") {
  HIP_CHECK(hipSetDevice(0));

  hipStream_t copyStream, execStream;
  HIP_CHECK(hipStreamCreateWithFlags(&copyStream, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&execStream, hipStreamNonBlocking));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N];
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  float *da, *db, *dc, *tmp, *dd;
  HIP_CHECK(hipMalloc(&da, SZ));
  HIP_CHECK(hipMalloc(&db, SZ));
  HIP_CHECK(hipMalloc(&dc, SZ));
  HIP_CHECK(hipMalloc(&tmp, SZ));
  HIP_CHECK(hipMalloc(&dd, SZ));

  HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, copyStream));
  HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, copyStream));
  HIP_CHECK(hipStreamSynchronize(copyStream));

  hipGraph_t     graph;
  hipGraphExec_t graphExec;
  dim3 block(256), grid((N + 255) / 256);

  // Math per graph execution:
  //   fill:   dc=0
  //   saxpy:  dc=3*1+0=3
  //   add:    tmp=1+1=2
  //   scale:  tmp=0.5*2=1
  //   copy:   dc=1
  //   D2D:    dd=1
  //   add:    dc=1+1=2
  //   saxpy:  dc=-1*1+2=1
  //   add:    dc=1+1=2  ✓
  HIP_CHECK(hipStreamBeginCapture(execStream, hipStreamCaptureModeThreadLocal));

  hipLaunchKernelGGL(hrr_vectorFill,   grid, block, 0, execStream, dc,  0.0f, N);
  hipLaunchKernelGGL(hrr_vectorSaxpy,  grid, block, 0, execStream, 3.0f, da, dc, N);
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, da, db, tmp, N);
  hipLaunchKernelGGL(hrr_vectorScale,  grid, block, 0, execStream, tmp, tmp, 0.5f, N);
  hipLaunchKernelGGL(hrr_memcpyKernel, grid, block, 0, execStream, tmp, dc, N);
  HIP_CHECK(hipMemcpyAsync(dd, dc, SZ, hipMemcpyDeviceToDevice, execStream));
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, da, dc, dc, N);
  hipLaunchKernelGGL(hrr_vectorSaxpy,  grid, block, 0, execStream, -1.0f, db, dc, N);
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, dc, dc, dc, N);

  HIP_CHECK(hipStreamEndCapture(execStream, &graph));
  HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  HIP_CHECK(hipGraphDestroy(graph));

  for (int iter = 0; iter < GRAPH_ITERS; ++iter)
    HIP_CHECK(hipGraphLaunch(graphExec, execStream));
  HIP_CHECK(hipStreamSynchronize(execStream));

  // D2H — blob captured here when HIP_HRR_CAPTURE_OUTPUT is set
  HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, copyStream));
  HIP_CHECK(hipStreamSynchronize(copyStream));

  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipFree(da)); HIP_CHECK(hipFree(db)); HIP_CHECK(hipFree(dc));
  HIP_CHECK(hipFree(tmp)); HIP_CHECK(hipFree(dd));
  HIP_CHECK(hipStreamDestroy(copyStream));
  HIP_CHECK(hipStreamDestroy(execStream));
  delete[] ha; delete[] hb; delete[] hc;
}
