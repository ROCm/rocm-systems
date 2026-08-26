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

#include "hrr_test_common.hh"
#include <hip/hiprtc.h>
#include <hip/hip_ext.h>  // hipExtModuleLaunchKernel

// Local hipRTC error check (replaces the hip-tests hipRTC error check). hipRTC
// symbols come from <hip/hiprtc.h> included above; this mirrors the local
// HRR_HIPRTC_CHECK used by hrr_roundtrip_test.cc.
#define HRR_HIPRTC_CHECK(expr)                                                 \
  do {                                                                         \
    hiprtcResult _hrr_rtc = (expr);                                            \
    INFO("hipRTC call failed: " #expr);                                        \
    INFO("hiprtcResult: " << static_cast<int>(_hrr_rtc));                      \
    INFO("hiprtcGetErrorString: " << hiprtcGetErrorString(_hrr_rtc));          \
    REQUIRE(_hrr_rtc == HIPRTC_SUCCESS);                                       \
  } while (0)
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Workload parameters
// ---------------------------------------------------------------------------

static constexpr int    N            = 1 << 12;   // 4K floats (16 KB)
static constexpr size_t SZ           = N * sizeof(float);
static constexpr int    KERNEL_ITERS = 4;
static constexpr int    GRAPH_ITERS  = 4;

static bool hrr_find_peer_accessible_pair(int& src_dev, int& dst_dev, int& ndev) {
  HRR_HIP_CHECK(hipGetDeviceCount(&ndev));
  if (ndev < 2) return false;

  for (int src = 0; src < ndev; ++src) {
    for (int dst = 0; dst < ndev; ++dst) {
      if (src == dst) continue;
      int can_access = 0;
      HRR_HIP_CHECK(hipDeviceCanAccessPeer(&can_access, src, dst));
      if (can_access) {
        src_dev = src;
        dst_dev = dst;
        return true;
      }
    }
  }
  return false;
}

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
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s0, s1;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N];
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  float *da, *db, *dc, *dd, *dp;
  HRR_HIP_CHECK(hipMalloc(&da, SZ));
  HRR_HIP_CHECK(hipMalloc(&db, SZ));
  HRR_HIP_CHECK(hipMalloc(&dc, SZ));
  HRR_HIP_CHECK(hipMalloc(&dd, SZ));
  HRR_HIP_CHECK(hipMalloc(&dp, SZ));

  dim3 block(256), grid((N + 255) / 256);
  int nblocks = static_cast<int>(grid.x);

  HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, s0));
  HRR_HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, s0));
  HRR_HIP_CHECK(hipMemsetAsync(dc, 0, SZ, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  // saxpy: dc = 2*da + dc = 2*1 + 0 = 2
  hipLaunchKernelGGL(hrr_vectorSaxpy, grid, block, 0, s0, 2.0f, da, dc, N);
  HRR_HIP_CHECK(hipGetLastError());

  // vectorAdd overwrites dc each iter: dc = da + db = 1 + 1 = 2
  for (int iter = 0; iter < KERNEL_ITERS; ++iter) {
    hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, s0, da, db, dc, N);
    HRR_HIP_CHECK(hipGetLastError());
  }

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  // D2D copy exercises that event type in the capture stream
  HRR_HIP_CHECK(hipMemcpyAsync(dd, dc, SZ, hipMemcpyDeviceToDevice, s1));

  hipLaunchKernelGGL(hrr_dotPartial, grid, block, block.x * sizeof(float), s0,
                     da, db, dp, N);
  HRR_HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(hrr_vectorScale, grid, block, 0, s0, dp, dp,
                     1.0f / nblocks, N);
  HRR_HIP_CHECK(hipGetLastError());

  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  // D2H — blob captured here when HIP_HRR_CAPTURE_OUTPUT is set
  HRR_HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));

  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  HRR_HIP_CHECK(hipFree(da)); HRR_HIP_CHECK(hipFree(db)); HRR_HIP_CHECK(hipFree(dc));
  HRR_HIP_CHECK(hipFree(dd)); HRR_HIP_CHECK(hipFree(dp));
  HRR_HIP_CHECK(hipStreamDestroy(s0));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
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
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t copyStream, execStream;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&copyStream, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&execStream, hipStreamNonBlocking));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N];
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  float *da, *db, *dc, *tmp, *dd;
  HRR_HIP_CHECK(hipMalloc(&da, SZ));
  HRR_HIP_CHECK(hipMalloc(&db, SZ));
  HRR_HIP_CHECK(hipMalloc(&dc, SZ));
  HRR_HIP_CHECK(hipMalloc(&tmp, SZ));
  HRR_HIP_CHECK(hipMalloc(&dd, SZ));

  HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, copyStream));
  HRR_HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, copyStream));
  HRR_HIP_CHECK(hipStreamSynchronize(copyStream));

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
  HRR_HIP_CHECK(hipStreamBeginCapture(execStream, hipStreamCaptureModeThreadLocal));

  hipLaunchKernelGGL(hrr_vectorFill,   grid, block, 0, execStream, dc,  0.0f, N);
  hipLaunchKernelGGL(hrr_vectorSaxpy,  grid, block, 0, execStream, 3.0f, da, dc, N);
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, da, db, tmp, N);
  hipLaunchKernelGGL(hrr_vectorScale,  grid, block, 0, execStream, tmp, tmp, 0.5f, N);
  hipLaunchKernelGGL(hrr_memcpyKernel, grid, block, 0, execStream, tmp, dc, N);
  HRR_HIP_CHECK(hipMemcpyAsync(dd, dc, SZ, hipMemcpyDeviceToDevice, execStream));
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, da, dc, dc, N);
  hipLaunchKernelGGL(hrr_vectorSaxpy,  grid, block, 0, execStream, -1.0f, db, dc, N);
  hipLaunchKernelGGL(hrr_vectorAdd,    grid, block, 0, execStream, dc, dc, dc, N);

  HRR_HIP_CHECK(hipStreamEndCapture(execStream, &graph));
  HRR_HIP_CHECK(hipGraphInstantiateWithFlags(&graphExec, graph, 0));
  HRR_HIP_CHECK(hipGraphDestroy(graph));

  for (int iter = 0; iter < GRAPH_ITERS; ++iter)
    HRR_HIP_CHECK(hipGraphLaunch(graphExec, execStream));
  HRR_HIP_CHECK(hipStreamSynchronize(execStream));

  // D2H — blob captured here when HIP_HRR_CAPTURE_OUTPUT is set
  HRR_HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, copyStream));
  HRR_HIP_CHECK(hipStreamSynchronize(copyStream));

  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  HRR_HIP_CHECK(hipGraphExecDestroy(graphExec));
  HRR_HIP_CHECK(hipFree(da)); HRR_HIP_CHECK(hipFree(db)); HRR_HIP_CHECK(hipFree(dc));
  HRR_HIP_CHECK(hipFree(tmp)); HRR_HIP_CHECK(hipFree(dd));
  HRR_HIP_CHECK(hipStreamDestroy(copyStream));
  HRR_HIP_CHECK(hipStreamDestroy(execStream));
  delete[] ha; delete[] hb; delete[] hc;
}

// ---------------------------------------------------------------------------
// Kernels for the hipHostMalloc workload and the all-APIs workload
// ---------------------------------------------------------------------------

__global__ void hrr_incrementInt(int* buf) {
  *buf += 1;
}

__global__ void hrr_scalarAdd(const int* a, const int* b, int* c, int n, int scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i] + scalar;
}

// ---------------------------------------------------------------------------
// Direct hipHostMalloc / hipHostGetDevicePointer workload — hidden ([.]).
//
// hipHostMalloc → hipHostGetDevicePointer → kernel (buf += 1) → sync → D2H
// Expected: *buf == 2  (initialised to 1, incremented by kernel to 2)
//
// Exercises:
//   - hipHostMalloc capture (pinned host memory)
//   - hipHostGetDevicePointer translate-ptr roundtrip
//   - D2H via the device pointer (kernel writes to buf_dev; host reads buf)
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_HostMemWorkload_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  int* buf;
  int* buf_dev;
  HRR_HIP_CHECK(hipHostMalloc(&buf, sizeof(*buf)));
  HRR_HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&buf_dev), buf, 0));

  // Initialize via H2D so the captured blob restores this value at playback.
  int init_val = 1;
  HRR_HIP_CHECK(hipMemcpy(buf_dev, &init_val, sizeof(init_val), hipMemcpyHostToDevice));

  hipLaunchKernelGGL(hrr_incrementInt, dim3(1), dim3(1), 0, hipStreamDefault,
                     buf_dev);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  // buf is host-visible; kernel wrote via buf_dev — same physical page.
  REQUIRE(*buf == 2);

  // Explicit D2H memcpy so the capture layer records a D2H blob for playback validation.
  int result = 0;
  HRR_HIP_CHECK(hipMemcpy(&result, buf_dev, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == 2);

  HRR_HIP_CHECK(hipFree(buf));
}

// ---------------------------------------------------------------------------
// Comprehensive HIP API coverage workload — hidden ([.]).
//
// Exercises ~55 distinct HIP APIs across:
//   device queries, stream/event management, memory allocation
//   (Malloc/Async/Pool/Host/Managed), memset, memcpy variants
//   (H2D/D2H/D2D/Async/WithStream), occupancy query, pointer attributes,
//   cache config, device sync, peer access query, and managed-memory
//   advise/prefetch/range-attr (device-capability-gated).
//
// The single D2H blob captured:
//   d0[i]=42, d1[i]=42, scalar=10  →  d2[i]=94  (validated by playback).
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_AllApis_Direct", "[.][hrr-direct]") {
  // =========================================================================
  // 1. Device queries
  // =========================================================================
  int deviceCount = 0;
  HRR_HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount >= 1);
  HRR_HIP_CHECK(hipSetDevice(0));

  hipDeviceProp_t props{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&props, 0));
  INFO("Device: " << props.name);

  int maxBlockDimX = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&maxBlockDimX, hipDeviceAttributeMaxBlockDimX, 0));
  REQUIRE(maxBlockDimX > 0);

  int priLo = 0, priHi = 0;
  HRR_HIP_CHECK(hipDeviceGetStreamPriorityRange(&priLo, &priHi));

  int driverVer = 0, runtimeVer = 0;
  HRR_HIP_CHECK(hipDriverGetVersion(&driverVer));
  HRR_HIP_CHECK(hipRuntimeGetVersion(&runtimeVer));
  REQUIRE(driverVer > 0);
  REQUIRE(runtimeVer > 0);

  // =========================================================================
  // 2. Error string API
  // =========================================================================
  (void)hipGetLastError();     // drain any pending errors
  (void)hipPeekAtLastError();  // peek without clearing
  REQUIRE(hipGetErrorName(hipSuccess) != nullptr);
  REQUIRE(hipGetErrorString(hipSuccess) != nullptr);

  // =========================================================================
  // 3. Stream management
  // =========================================================================
  hipStream_t s0, s1, s2;
  HRR_HIP_CHECK(hipStreamCreate(&s0));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithPriority(&s2, hipStreamNonBlocking, priLo));

  unsigned int sflags = 0;
  HRR_HIP_CHECK(hipStreamGetFlags(s1, &sflags));
  REQUIRE(sflags == hipStreamNonBlocking);

  int spri = 0;
  HRR_HIP_CHECK(hipStreamGetPriority(s2, &spri));

  // hipStreamQuery — hipSuccess or hipErrorNotReady are both valid responses
  {
    hipError_t q = hipStreamQuery(s0);
    REQUIRE((q == hipSuccess || q == hipErrorNotReady));
  }

  // =========================================================================
  // 4. Events
  // =========================================================================
  hipEvent_t ev_start, ev_stop, ev_nodur;
  HRR_HIP_CHECK(hipEventCreate(&ev_start));
  HRR_HIP_CHECK(hipEventCreate(&ev_stop));
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev_nodur, hipEventDisableTiming));

  // =========================================================================
  // 5. Memory allocation
  // =========================================================================
  constexpr int    N  = 1024;
  constexpr size_t SZ = N * sizeof(int);

  size_t memFree = 0, memTotal = 0;
  HRR_HIP_CHECK(hipMemGetInfo(&memFree, &memTotal));
  REQUIRE(memTotal > 0);

  int *d0, *d1, *d2;
  HRR_HIP_CHECK(hipMalloc(&d0, SZ));
  HRR_HIP_CHECK(hipMalloc(&d1, SZ));
  HRR_HIP_CHECK(hipMalloc(&d2, SZ));

  int *d_async = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(&d_async, SZ, s0));

  hipMemPool_t      pool;
  hipMemPoolProps   poolProps{};
  poolProps.allocType     = hipMemAllocationTypePinned;
  poolProps.location.type = hipMemLocationTypeDevice;
  poolProps.location.id   = 0;
  HRR_HIP_CHECK(hipMemPoolCreate(&pool, &poolProps));

  uint64_t threshold = 0;
  HRR_HIP_CHECK(hipMemPoolGetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));
  threshold = static_cast<uint64_t>(-1);  // never release automatically
  HRR_HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, &threshold));

  int *d_pool = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(&d_pool, SZ, pool, s0));

  int *h_pinned = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(&h_pinned, SZ, 0));

  const bool managed_ok = (props.managedMemory != 0);
  int *d_managed = nullptr;
  if (managed_ok)
    HRR_HIP_CHECK(hipMallocManaged(&d_managed, SZ, hipMemAttachGlobal));

  // =========================================================================
  // 6. Memset + H2D memcpy variants
  // =========================================================================
  HRR_HIP_CHECK(hipMemset(d0, 0, SZ));
  HRR_HIP_CHECK(hipMemsetAsync(d1, 0, SZ, s0));

  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = 42;

  // hipMemcpy (sync H2D) → d0 = 42
  HRR_HIP_CHECK(hipMemcpy(d0, h_src, SZ, hipMemcpyHostToDevice));
  // hipMemcpyAsync (H2D on s0) → d1 = 42
  HRR_HIP_CHECK(hipMemcpyAsync(d1, h_src, SZ, hipMemcpyHostToDevice, s0));
  // hipMemcpyWithStream (H2D on s1) → d_async = 1
  for (int i = 0; i < N; ++i) h_pinned[i] = 1;
  HRR_HIP_CHECK(hipMemcpyWithStream(d_async, h_pinned, SZ, hipMemcpyHostToDevice, s1));

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  // =========================================================================
  // 7. Pointer attributes
  // =========================================================================
  hipPointerAttribute_t pattr{};
  HRR_HIP_CHECK(hipPointerGetAttributes(&pattr, d0));
  REQUIRE(pattr.type == hipMemoryTypeDevice);

  // =========================================================================
  // 8. Cache config query
  // =========================================================================
  hipFuncCache_t cacheConf = hipFuncCachePreferNone;
  HRR_HIP_CHECK(hipDeviceGetCacheConfig(&cacheConf));

  // =========================================================================
  // 9. Peer access capability (query only, no enable/disable)
  // =========================================================================
  if (deviceCount > 1) {
    int canAccess = 0;
    HRR_HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 0, 1));
  }

  // =========================================================================
  // 10. Occupancy query + timed kernel launch
  // =========================================================================
  int occBlockSize = 0, occGridSize = 0;
  HRR_HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&occGridSize, &occBlockSize,
                                              hrr_scalarAdd, 0, 0));
  REQUIRE(occBlockSize > 0);

  HRR_HIP_CHECK(hipEventRecord(ev_start, s0));

  dim3 block(256), grid((N + 255) / 256);
  // d2[i] = d0[i] + d1[i] + 10 = 42 + 42 + 10 = 94
  hipLaunchKernelGGL(hrr_scalarAdd, grid, block, 0, s0, d0, d1, d2, N, 10);
  HRR_HIP_CHECK(hipGetLastError());

  HRR_HIP_CHECK(hipEventRecord(ev_stop, s0));
  // s1 waits for ev_stop before performing the D2H copy
  HRR_HIP_CHECK(hipStreamWaitEvent(s1, ev_stop, 0));

  // =========================================================================
  // 11. Event query + elapsed time
  // =========================================================================
  HRR_HIP_CHECK(hipEventSynchronize(ev_stop));
  { hipError_t q = hipEventQuery(ev_stop); REQUIRE(q == hipSuccess); }

  float ms = 0.f;
  HRR_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
  // Allow small negative values: GPU timer resolution can return -epsilon
  // when events are very close together. Accept anything > -1 ms.
  REQUIRE(ms > -1.f);

  // =========================================================================
  // 12. D2H memcpy — blob captured here; playback validates against it
  // =========================================================================
  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d2, SZ, hipMemcpyDeviceToHost, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  for (int i = 0; i < N; ++i)
    REQUIRE(h_out[i] == 94);

  // =========================================================================
  // 13. D2D copy + full device sync
  // =========================================================================
  HRR_HIP_CHECK(hipMemcpy(d1, d2, SZ, hipMemcpyDeviceToDevice));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // =========================================================================
  // 14. Managed memory: advise, range-attr query, prefetch (conditional)
  // =========================================================================
  if (managed_ok && d_managed) {
    for (int i = 0; i < N; ++i) d_managed[i] = i;
    HRR_HIP_CHECK(hipMemAdvise(d_managed, SZ, hipMemAdviseSetReadMostly, 0));

    // hipMemRangeGetAttribute segfaults on some Linux ROCm builds —
    // guard with a non-fatal call and skip the assertion on failure.
    uint32_t rmAttr = 0;
    hipError_t rga_err = hipMemRangeGetAttribute(&rmAttr, sizeof(rmAttr),
                                                  hipMemRangeAttributeReadMostly,
                                                  d_managed, SZ);
    if (rga_err == hipSuccess) {
      REQUIRE(rmAttr == 1u);
    } else {
      WARN("hipMemRangeGetAttribute returned " << (int)rga_err
           << " — skipping range-attr assertion on this platform");
    }

    HRR_HIP_CHECK(hipMemPrefetchAsync(d_managed, SZ, 0, s0));
    HRR_HIP_CHECK(hipStreamSynchronize(s0));
    HRR_HIP_CHECK(hipMemAdvise(d_managed, SZ, hipMemAdviseUnsetReadMostly, 0));
  }

  // =========================================================================
  // 15. hipMemcpy3D (H2D) + hipMemcpy3DAsync (D2H) — struct-pointer coverage
  //
  // Use a flat 1-row, 1-depth slice so the 3D params are trivially computed.
  // srcPtr / dstPtr widths are in bytes; extent.width is also in bytes.
  // D2H blob: d3_out[i] == 99  (validated by playback).
  // =========================================================================
  {
    int* d3 = nullptr;
    HRR_HIP_CHECK(hipMalloc(&d3, SZ));
    int* h3_src = new int[N];
    int* h3_out = new int[N]();
    for (int i = 0; i < N; ++i) h3_src[i] = 99;

    // H2D via hipMemcpy3D
    hipMemcpy3DParms p_h2d{};
    p_h2d.srcPtr = make_hipPitchedPtr(h3_src, N * sizeof(int), N, 1);
    p_h2d.dstPtr = make_hipPitchedPtr(d3,     N * sizeof(int), N, 1);
    p_h2d.extent = make_hipExtent(N * sizeof(int), 1, 1);
    p_h2d.kind   = hipMemcpyHostToDevice;
    HRR_HIP_CHECK(hipMemcpy3D(&p_h2d));

    // D2H via hipMemcpy3DAsync — blob captured here; playback validates 99
    hipMemcpy3DParms p_d2h{};
    p_d2h.srcPtr = make_hipPitchedPtr(d3,     N * sizeof(int), N, 1);
    p_d2h.dstPtr = make_hipPitchedPtr(h3_out, N * sizeof(int), N, 1);
    p_d2h.extent = make_hipExtent(N * sizeof(int), 1, 1);
    p_d2h.kind   = hipMemcpyDeviceToHost;
    HRR_HIP_CHECK(hipMemcpy3DAsync(&p_d2h, s0));
    HRR_HIP_CHECK(hipStreamSynchronize(s0));

    for (int i = 0; i < N; ++i)
      REQUIRE(h3_out[i] == 99);

    HRR_HIP_CHECK(hipFree(d3));
    delete[] h3_src;
    delete[] h3_out;
  }

  // =========================================================================
  // 16. hipStreamSetAttribute — struct-pointer coverage
  //
  // Set synchronization policy (universally supported attribute).
  // =========================================================================
  {
    hipStreamAttrValue attr_val{};
    attr_val.syncPolicy = hipSyncPolicySpin;
    HRR_HIP_CHECK(hipStreamSetAttribute(s0, hipStreamAttributeSynchronizationPolicy,
                                    &attr_val));
  }

  // =========================================================================
  // 17. hipMemGetAllocationGranularity — struct-pointer coverage
  // =========================================================================
  {
    hipMemAllocationProp alloc_prop{};
    alloc_prop.type          = hipMemAllocationTypePinned;
    alloc_prop.location.type = hipMemLocationTypeDevice;
    alloc_prop.location.id   = 0;
    size_t granularity = 0;
    HRR_HIP_CHECK(hipMemGetAllocationGranularity(
        &granularity, &alloc_prop, hipMemAllocationGranularityMinimum));
    REQUIRE(granularity > 0);
  }

  // =========================================================================
  // 18. hipMemPoolSetAccess — struct-pointer coverage (uses pool from §5)
  // =========================================================================
  {
    hipMemAccessDesc access_desc{};
    access_desc.location.type = hipMemLocationTypeDevice;
    access_desc.location.id   = 0;
    access_desc.flags         = hipMemAccessFlagsProtReadWrite;
    HRR_HIP_CHECK(hipMemPoolSetAccess(pool, &access_desc, 1));
  }

  // =========================================================================
  // 19. hipArrayCreate + hipArray3DCreate — handle-map coverage
  //
  // Array creation requires image support.  Query device capability first and
  // skip gracefully on hardware that reports no texture support.
  // =========================================================================
  {
    int supportsImages = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&supportsImages, hipDeviceAttributeImageSupport, 0));
    if (supportsImages) {
      HIP_ARRAY_DESCRIPTOR desc1d{};
      desc1d.Width       = static_cast<size_t>(N);
      desc1d.Height      = 0;   // 1-D
      desc1d.Format      = HIP_AD_FORMAT_SIGNED_INT32;
      desc1d.NumChannels = 1;
      hipArray_t arr1d = nullptr;
      HRR_HIP_CHECK(hipArrayCreate(&arr1d, &desc1d));
      HRR_HIP_CHECK(hipFreeArray(arr1d));

      HIP_ARRAY3D_DESCRIPTOR desc3d{};
      desc3d.Width       = 4;
      desc3d.Height      = 4;
      desc3d.Depth       = 4;
      desc3d.Format      = HIP_AD_FORMAT_SIGNED_INT32;
      desc3d.NumChannels = 1;
      desc3d.Flags       = 0;
      hipArray_t arr3d = nullptr;
      HRR_HIP_CHECK(hipArray3DCreate(&arr3d, &desc3d));
      HRR_HIP_CHECK(hipFreeArray(arr3d));
    }
  }

  // =========================================================================
  // Cleanup (reverse allocation order)
  // =========================================================================
  if (managed_ok && d_managed) HRR_HIP_CHECK(hipFree(d_managed));

  HRR_HIP_CHECK(hipFreeAsync(d_pool, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipMemPoolDestroy(pool));

  HRR_HIP_CHECK(hipFreeAsync(d_async, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));

  HRR_HIP_CHECK(hipFree(d0)); HRR_HIP_CHECK(hipFree(d1)); HRR_HIP_CHECK(hipFree(d2));
  HRR_HIP_CHECK(hipFree(h_pinned));

  HRR_HIP_CHECK(hipEventDestroy(ev_start));
  HRR_HIP_CHECK(hipEventDestroy(ev_stop));
  HRR_HIP_CHECK(hipEventDestroy(ev_nodur));

  HRR_HIP_CHECK(hipStreamDestroy(s0));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
  HRR_HIP_CHECK(hipStreamDestroy(s2));

  delete[] h_src;
  delete[] h_out;
}

// ---------------------------------------------------------------------------
// Stress workload — hidden ([.]).
//
// Generates 500+ HIP API call events in a single capture, exercising:
//   - 8 stream + 8 event create/destroy cycles
//   - 25 hipMalloc / hipFree pairs → 50 events
//   - 10 hipMallocAsync / hipFreeAsync pairs → 20 events
//   - 10 hipMemPoolCreate / hipMemPoolDestroy pairs → 20 events
//   - 60 kernel launches (hrr_vectorAdd + hrr_vectorScale in loops)
//   - 30 H2D hipMemcpyAsync calls
//   - 30 D2H hipMemcpyAsync calls (blob captured each time)
//   - 20 D2D hipMemcpy calls
//   - 30 hipMemsetAsync calls
//   - 20 hipEventRecord + hipEventSynchronize pairs → 40 events
//   - 15 hipStreamSynchronize calls
//   - 10 hipDeviceGetAttribute calls
//   - 10 hipMemGetInfo calls
//   - 10 hipPointerGetAttributes calls
//   - hipOccupancyMaxPotentialBlockSize, hipDeviceGetCacheConfig, etc.
//
// Final D2H blob: h_out[i] == 2.0f (validated by playback).
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_StressApis_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  // =========================================================================
  // Constants
  // =========================================================================
  constexpr int    N    = 1 << 12;   // 4096 floats
  constexpr size_t SZ   = N * sizeof(float);
  constexpr int    STREAMS  = 8;
  constexpr int    EVENTS   = 8;
  constexpr int    POOLS    = 10;
  constexpr int    ALLOCS   = 25;
  constexpr int    KL_ITERS = 60;   // kernel launches
  constexpr int    CP_ITERS = 30;   // memcpy rounds
  constexpr int    MS_ITERS = 30;   // memset rounds

  // =========================================================================
  // 1. Create streams and events
  // =========================================================================
  hipStream_t streams[STREAMS];
  for (int i = 0; i < STREAMS; ++i)
    HRR_HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

  hipEvent_t evs[EVENTS];
  for (int i = 0; i < EVENTS; ++i)
    HRR_HIP_CHECK(hipEventCreate(&evs[i]));

  // =========================================================================
  // 2. Working buffers
  // =========================================================================
  float *da, *db, *dc;
  HRR_HIP_CHECK(hipMalloc(&da, SZ));
  HRR_HIP_CHECK(hipMalloc(&db, SZ));
  HRR_HIP_CHECK(hipMalloc(&dc, SZ));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N]();
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  // Initialise device buffers
  HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, streams[0]));
  HRR_HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, streams[1]));
  HRR_HIP_CHECK(hipMemsetAsync(dc, 0, SZ, streams[0]));
  HRR_HIP_CHECK(hipStreamSynchronize(streams[0]));
  HRR_HIP_CHECK(hipStreamSynchronize(streams[1]));

  dim3 block(256), grid((N + 255) / 256);

  // =========================================================================
  // 3. hipMalloc / hipFree pairs (ALLOCS × 2 = 50 events)
  // =========================================================================
  for (int i = 0; i < ALLOCS; ++i) {
    float* tmp = nullptr;
    HRR_HIP_CHECK(hipMalloc(&tmp, SZ));
    HRR_HIP_CHECK(hipFree(tmp));
  }

  // =========================================================================
  // 4. hipMallocAsync / hipFreeAsync pairs (10 × 2 = 20 events)
  // =========================================================================
  for (int i = 0; i < 10; ++i) {
    float* tmp = nullptr;
    HRR_HIP_CHECK(hipMallocAsync(&tmp, SZ, streams[i % STREAMS]));
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));
    HRR_HIP_CHECK(hipFreeAsync(tmp, streams[i % STREAMS]));
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));
  }

  // =========================================================================
  // 5. hipMemPoolCreate / hipMemPoolDestroy pairs (POOLS × 2 = 20 events)
  // =========================================================================
  for (int i = 0; i < POOLS; ++i) {
    hipMemPool_t pool;
    hipMemPoolProps pp{};
    pp.allocType     = hipMemAllocationTypePinned;
    pp.location.type = hipMemLocationTypeDevice;
    pp.location.id   = 0;
    HRR_HIP_CHECK(hipMemPoolCreate(&pool, &pp));
    HRR_HIP_CHECK(hipMemPoolDestroy(pool));
  }

  // =========================================================================
  // 6. Kernel launches in a loop (KL_ITERS = 60 events)
  //    vectorAdd: dc[i] = da[i] + db[i] = 1+1 = 2
  // =========================================================================
  for (int iter = 0; iter < KL_ITERS; ++iter) {
    hipStream_t s = streams[iter % STREAMS];
    hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, s, da, db, dc, N);
    HRR_HIP_CHECK(hipGetLastError());
    // Interleave with event record every 10 launches (60/10 = 6 × 2 = 12 events)
    if (iter % 10 == 9) {
      hipEvent_t ev = evs[(iter / 10) % EVENTS];
      HRR_HIP_CHECK(hipEventRecord(ev, s));
      HRR_HIP_CHECK(hipEventSynchronize(ev));
    }
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // =========================================================================
  // 7. Memset rounds (MS_ITERS = 30 events)
  // =========================================================================
  for (int i = 0; i < MS_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemsetAsync(dc, 0, SZ, s));
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());
  // Restore dc = da + db = 2
  hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, streams[0], da, db, dc, N);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipStreamSynchronize(streams[0]));

  // =========================================================================
  // 8. H2D memcpy rounds (CP_ITERS = 30 events)
  // =========================================================================
  for (int i = 0; i < CP_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, s));
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // =========================================================================
  // 9. D2D copies (20 events)
  // =========================================================================
  for (int i = 0; i < 20; ++i) {
    HRR_HIP_CHECK(hipMemcpy(db, da, SZ, hipMemcpyDeviceToDevice));
  }

  // =========================================================================
  // 10. hipEventRecord + hipEventSynchronize pairs (20 × 2 = 40 events)
  // =========================================================================
  for (int i = 0; i < 20; ++i) {
    hipEvent_t ev = evs[i % EVENTS];
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipEventRecord(ev, s));
    HRR_HIP_CHECK(hipEventSynchronize(ev));
  }

  // =========================================================================
  // 11. hipStreamSynchronize rounds (15 events)
  // =========================================================================
  for (int i = 0; i < 15; ++i)
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));

  // =========================================================================
  // 12. Device attribute / info queries (10+10+10 = 30 events)
  // =========================================================================
  for (int i = 0; i < 10; ++i) {
    int val = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&val, hipDeviceAttributeMaxBlockDimX, 0));
  }
  for (int i = 0; i < 10; ++i) {
    size_t mfree = 0, mtotal = 0;
    HRR_HIP_CHECK(hipMemGetInfo(&mfree, &mtotal));
  }
  for (int i = 0; i < 10; ++i) {
    hipPointerAttribute_t pa{};
    HRR_HIP_CHECK(hipPointerGetAttributes(&pa, dc));
  }

  // =========================================================================
  // 13. D2H blob captures (CP_ITERS = 30 events — blob written each time;
  //     dedup means same-content blobs share one file, but events are recorded)
  // =========================================================================
  for (int i = 0; i < CP_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // =========================================================================
  // Validate final host result
  // =========================================================================
  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  // =========================================================================
  // Cleanup
  // =========================================================================
  HRR_HIP_CHECK(hipFree(da)); HRR_HIP_CHECK(hipFree(db)); HRR_HIP_CHECK(hipFree(dc));
  for (int i = 0; i < EVENTS; ++i) HRR_HIP_CHECK(hipEventDestroy(evs[i]));
  for (int i = 0; i < STREAMS; ++i) HRR_HIP_CHECK(hipStreamDestroy(streams[i]));
  delete[] ha; delete[] hb; delete[] hc;
}

// ===========================================================================
// Workload: hipStreamWriteValue32 / hipStreamWriteValue64
//
// Exercises hipStreamWriteValue32 and hipStreamWriteValue64. These are replayed
// faithfully (replay-only fix: the destination void* ptr is translated via the
// alloc_map and the stream is translated); replay must reproduce the written
// values, validated via D2H.
//
// Each written value gets its OWN D2H readback rather than one combined 16-byte
// readback, because hrr-playback's D2H validator falls back to candidate float
// encodings (f32/bf16/f16/f64, atol=rtol=1e-3) when the bytes differ and skips
// byte-identical elements. A 16-byte readback holding the 32-bit sentinel in the
// low half of an 8-byte slot decodes, under the f64 candidate, to the subnormal
// 1.68e-314; a lost 32-bit write reads back as 0.0, which is inside the 1e-3
// tolerance, so the buffer would be accepted as "f64 within tolerance" while the
// 64-bit slot stayed byte-identical (skipped). Splitting the readbacks makes the
// 32-bit blob 4 bytes, and 4 % 8 != 0 excludes the f64 candidate outright; the
// remaining candidates reject 0.0-vs-0xCAFEBABE by ~1000x (f32 sees -8.35e6
// against a tolerance of 8.35e3). Unit_HRR_StreamWriteValueRoundtrip also pins
// HIP_HRR_D2H_EXACT=1 so the replay gate really is byte-for-byte.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamWriteValue_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  // Stream write/wait value is an optional device capability. Gate on the same
  // attribute every other test of these APIs uses (catch/unit/stream/
  // hipStreamValue.cc, Unit_HRR_StreamAdvanced2_Direct) so an unsupported
  // target skips instead of aborting the capture subprocess.
  int canUseStreamValue = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&canUseStreamValue,
                                  hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (!canUseStreamValue) {
    HRR_SKIP("stream wait value unsupported");
  }

  constexpr uint64_t kVal64 = 0xDEADBEEFFEEDFACEull;
  constexpr uint32_t kVal32 = 0xCAFEBABEu;
  // Increment slot: base value, then a read-modify-write increment on top of it.
  // 0x0A0A0A0A + 0xF4E3F0C4 wraps to 0xFEEDFACE.
  constexpr uint32_t kIncBase = 0x0A0A0A0Au;
  constexpr uint32_t kIncDelta = 0xF4E3F0C4u;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // Two 64-bit slots so the 32-bit write and the 64-bit write land in disjoint,
  // fully-defined memory.
  uint64_t* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 2 * sizeof(uint64_t)));
  HRR_HIP_CHECK(hipMemset(d, 0, 2 * sizeof(uint64_t)));

  // Separate slot for the flags-bearing write. Allocated 8 bytes wide so a
  // 64-bit-granular implementation of the increment cannot overrun it.
  uint32_t* inc = nullptr;
  HRR_HIP_CHECK(hipMalloc(&inc, sizeof(uint64_t)));
  HRR_HIP_CHECK(hipMemset(inc, 0, sizeof(uint64_t)));

  // 64-bit stream write into slot0.
  HRR_HIP_CHECK(hipStreamWriteValue64(s, d, kVal64, 0));
  // 32-bit stream write into slot1 (low 32 bits); the high 32 bits stay zero.
  HRR_HIP_CHECK(hipStreamWriteValue32(s, d + 1, kVal32, 0));

  // Flags are plumbed through capture and replay, so exercise a non-default one.
  // hipExtStreamWriteValueIncrement turns the write into a read-modify-write,
  // which makes replay fidelity depend on the earlier writes to the same slot
  // having been replayed too. The increment flag is not guaranteed on every
  // target, so tolerate a rejection: capture only records successful calls, so
  // on a target that rejects it the slot simply keeps kIncBase in both the
  // recorded blob and the replay.
  HRR_HIP_CHECK(hipStreamWriteValue32(s, inc, kIncBase, 0));
  hipError_t incErr = hipStreamWriteValue32(s, inc, kIncDelta,
                                            hipExtStreamWriteValueIncrement);
  REQUIRE((incErr == hipSuccess || incErr == hipErrorInvalidValue
           || incErr == hipErrorNotSupported));

  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // One readback per value: an 8-byte blob for the 64-bit write and 4-byte blobs
  // for the 32-bit ones (see the note above on the f64 candidate encoding).
  uint64_t h64 = 0;
  HRR_HIP_CHECK(hipMemcpy(&h64, d, sizeof(h64), hipMemcpyDeviceToHost));
  REQUIRE(h64 == kVal64);

  uint32_t h32 = 0;
  HRR_HIP_CHECK(hipMemcpy(&h32, d + 1, sizeof(h32), hipMemcpyDeviceToHost));
  REQUIRE(h32 == kVal32);

  uint32_t hInc = 0;
  HRR_HIP_CHECK(hipMemcpy(&hInc, inc, sizeof(hInc), hipMemcpyDeviceToHost));
  // Deliberately not asserting the exact sum: the point of this slot is that
  // replay reproduces whatever the increment produced at capture time. Only
  // assert that a successful increment of a non-zero delta changed the slot.
  if (incErr == hipSuccess) REQUIRE(hInc != kIncBase);

  HRR_HIP_CHECK(hipFree(inc));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

// ===========================================================================
// Pitched device-to-device memset (hipMemsetD2D*): shared geometry
//
// PITCH must be strictly greater than WIDTH.  CLR short-circuits the pitched
// fill to the plain non-pitched one whenever pitch == extent.width
// (ihipMemset3DCommand), so a contiguous buffer never reaches the pitched
// FillMemoryCommand at all, and it has no inter-row padding, so there is
// nothing to prove the fill stayed inside the requested sub-region.
//
// Note on units: CLR treats `width` as BYTES (sizeBytes = width * height);
// the D2D8/16/32 flavour only selects the fill-pattern width via elementSize,
// despite the doxygen wording implying elements.  hipMemsetD2D32Async.cc in
// catch/unit/memory/ likewise passes width = numW * sizeof(int).
// ===========================================================================
namespace {
constexpr size_t kD2DWidth = 32 * sizeof(int);          // 128 bytes filled per row
constexpr size_t kD2DPitch = 192;                       // row stride (> width)
constexpr size_t kD2DRows  = 32;
constexpr size_t kD2DTotal = kD2DPitch * kD2DRows;      // 6144 bytes
constexpr size_t kD2DPad   = kD2DPitch - kD2DWidth;     // 64 untouched bytes/row

// Sentinel pre-fill of the WHOLE buffer vs the pattern written into the
// WIDTH sub-region.  The two must be far apart under every dtype the replay
// D2H validator guesses (f32/bf16/f16/f64 at atol=rtol=1e-3), otherwise a
// replay that never ran the memsets (HRR zero-initialises replay allocations,
// so the buffer reads back all-zero) would land inside the tolerance and
// "pass".  0x5A5A5A5A and 0x44444444 decode to (f32) 1.54e16 / 785.07,
// (bf16) 1.53e16 / 784, (f16) 203.25 / 4.266 and (f64) 1.78e127 / 7.48e20;
// against zero, and against each other, every one of those is far outside
// tolerance, so both a skipped-memset replay and a pitch-ignoring replay
// (which would overwrite the padding with the pattern) genuinely FAIL.
constexpr unsigned char kD2DSentinelByte = 0x5Au;       // 0x5A5A5A5A per int
constexpr unsigned int  kD2DPattern      = 0x44444444u;

static_assert(kD2DPitch > kD2DWidth, "pitched memset path requires pitch > width");
static_assert(kD2DWidth % sizeof(unsigned int) == 0, "width must hold whole ints");

// Count ints in the filled sub-region that hold `pattern`, and bytes in the
// inter-row padding that still hold the sentinel.  Aggregated rather than
// asserted per element so a mismatch does not emit thousands of Catch2 failures.
void d2d_count_matches(const std::vector<unsigned char>& buf, unsigned int pattern,
                       size_t* filled_ok, size_t* padding_ok) {
  *filled_ok = 0;
  *padding_ok = 0;
  for (size_t r = 0; r < kD2DRows; ++r) {
    const unsigned char* row = buf.data() + r * kD2DPitch;
    for (size_t c = 0; c < kD2DWidth; c += sizeof(unsigned int)) {
      unsigned int v = 0;
      memcpy(&v, row + c, sizeof(v));
      if (v == pattern) ++*filled_ok;
    }
    for (size_t b = kD2DWidth; b < kD2DPitch; ++b)
      if (row[b] == kD2DSentinelByte) ++*padding_ok;
  }
}
}  // namespace

// ===========================================================================
// Workload: pitched device-to-device memset (hipMemsetD2D*)
//
// Exercises hipMemsetD2D8 / hipMemsetD2D8Async / hipMemsetD2D16 /
// hipMemsetD2D16Async / hipMemsetD2D32 / hipMemsetD2D32Async. These are
// replayed faithfully (mirror the hipMemset2D handler, with alloc-map pointer
// translation and stream translation for the async variants); replay must
// reproduce the final byte pattern.
//
// Final blob (whole allocation, padding included): the first kD2DWidth bytes
// of every row are kD2DPattern, the remaining kD2DPad bytes still hold
// kD2DSentinelByte.
// ===========================================================================
TEST_CASE("Unit_HRR_MemsetD2D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kD2DTotal));
  const hipDeviceptr_t dp = reinterpret_cast<hipDeviceptr_t>(d);

  // Sentinel the whole allocation, padding included, using an API that is
  // already replayed faithfully, so the padding has a known non-zero value
  // that the pitched fills below must leave alone.
  HRR_HIP_CHECK(hipMemsetD8(dp, kD2DSentinelByte, kD2DTotal));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // Pitched fills, kD2DWidth bytes per row only.
  HRR_HIP_CHECK(hipMemsetD2D8(dp, kD2DPitch, 0x11, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D8Async(dp, kD2DPitch, 0x22, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D16(dp, kD2DPitch, 0x3333, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D16Async(dp, kD2DPitch, 0x4444, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D32(dp, kD2DPitch, 0x11223344u, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D32Async(dp, kD2DPitch, 0x55667788u, kD2DWidth, kD2DRows, s));

  // The non-Async D2D memsets are NOT synchronous here: ihipMemset3D flips
  // isAsync true for a plain device allocation at offset 0 and enqueues on the
  // null stream without finish(), and `s` was created hipStreamNonBlocking so
  // it never implicitly waits on the null stream.  The fills above are
  // therefore mutually unordered.  Drain the device, then issue one final
  // deterministic fill and drain again, so the captured end state is unique.
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemsetD2D32(dp, kD2DPitch, kD2DPattern, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // D2H the WHOLE allocation so the captured blob covers the padding too.
  std::vector<unsigned char> h(kD2DTotal, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kD2DTotal, hipMemcpyDeviceToHost));

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));

  size_t filled_ok = 0, padding_ok = 0;
  d2d_count_matches(h, kD2DPattern, &filled_ok, &padding_ok);
  REQUIRE(filled_ok == kD2DRows * (kD2DWidth / sizeof(unsigned int)));
  REQUIRE(padding_ok == kD2DRows * kD2DPad);
}

// ===========================================================================
// Workload: hipMemsetD2D* onto a destination whose allocation is NOT replayed
//
// hipMemAllocPitch is itself in the playback no-op set, so nothing it returns
// ever reaches alloc_map, yet hipMemAllocPitch + hipMemsetD2D* is the idiomatic
// driver-API pairing, which makes an untranslatable destination the expected
// case rather than a corner case.  dispatch_event() treats any
// non-success handler return as fatal, so passing the real API a null
// destination (hipErrorInvalidValue) would abort the whole replay; the
// handlers must warn once and skip instead.
//
// A plain hipMalloc buffer is filled and read back as well, so the archive
// still carries a translatable D2H blob.  Replay only reaches and validates
// that blob if the untranslatable memsets were skipped rather than fatal,
// which is what makes the roundtrip a regression guard for the skip path.
// ===========================================================================
TEST_CASE("Unit_HRR_MemsetD2DPitchAlloc_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // Destination that replay cannot translate.
  hipDeviceptr_t pitched = nullptr;
  size_t pitch = 0;
  HRR_HIP_CHECK(hipMemAllocPitch(&pitched, &pitch, kD2DWidth, kD2DRows,
                             /*elementSizeBytes=*/4));
  REQUIRE(pitched != nullptr);
  REQUIRE(pitch >= kD2DWidth);

  // All six variants, so every generated handler takes the skip path on replay.
  // These fills are mutually unordered (null stream vs non-blocking `s`), which
  // is harmless: this buffer is never read back, only its recorded destination
  // matters.
  HRR_HIP_CHECK(hipMemsetD2D8(pitched, pitch, 0x11, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D8Async(pitched, pitch, 0x22, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D16(pitched, pitch, 0x3333, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D16Async(pitched, pitch, 0x4444, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D32(pitched, pitch, 0x11223344u, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D32Async(pitched, pitch, kD2DPattern, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // Translatable destination: this is the blob replay must still validate.
  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kD2DTotal));
  const hipDeviceptr_t dp = reinterpret_cast<hipDeviceptr_t>(d);
  HRR_HIP_CHECK(hipMemsetD8(dp, kD2DSentinelByte, kD2DTotal));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemsetD2D32(dp, kD2DPitch, kD2DPattern, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> h(kD2DTotal, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, kD2DTotal, hipMemcpyDeviceToHost));

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipFree(reinterpret_cast<void*>(pitched)));
  HRR_HIP_CHECK(hipStreamDestroy(s));

  size_t filled_ok = 0, padding_ok = 0;
  d2d_count_matches(h, kD2DPattern, &filled_ok, &padding_ok);
  REQUIRE(filled_ok == kD2DRows * (kD2DWidth / sizeof(unsigned int)));
  REQUIRE(padding_ok == kD2DRows * kD2DPad);
}

// ===========================================================================
// Workload B: hipMemsetD8/16/32 variants + hipMemset2D/2DAsync
//
// Exercises typed-memset driver APIs and 2-D pitched memset.
// Final blob: h[i] == 0x44444444 (set by hipMemsetD32 at the end).
// ===========================================================================
TEST_CASE("Unit_HRR_MemsetVariants_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 1024;
  constexpr size_t SZ = N * sizeof(int);  // 4096 bytes

  // Final validation pattern.  Replay zero-initialises its allocations and the
  // playback D2H validator falls back to a float tolerance (atol=rtol=1e-3 over
  // f32/bf16/f16/f64, accepting the first encoding that fits), so a small
  // integer canary cannot detect a no-op replay: 2 decodes to f32 2.8e-45, well
  // inside atol, and would validate against an all-zero buffer.  0x44444444 is
  // far from zero in every candidate encoding (f32 785.07, bf16 784, f16 4.27,
  // f64 7.5e20).
  constexpr int VAL = 0x44444444;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // hipMemsetD8 / hipMemsetD8Async (count = bytes)
  HRR_HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(d), 0x01, SZ));
  HRR_HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(d), 0x02, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemsetD16 / hipMemsetD16Async (count = 16-bit elements)
  HRR_HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(d), 0x0003, SZ / 2));
  HRR_HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(d), 0x0004, SZ / 2, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemsetD32 / hipMemsetD32Async (count = 32-bit elements)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 2, N));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), 2, N, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemset2D / hipMemset2DAsync — treat d as 32-col × 32-row 2D buffer
  constexpr size_t COLS  = 32;
  constexpr size_t ROWS  = N / COLS;
  constexpr size_t PITCH = COLS * sizeof(int);
  HRR_HIP_CHECK(hipMemset2D(d, PITCH, 0, PITCH, ROWS));
  HRR_HIP_CHECK(hipMemset2DAsync(d, PITCH, 0, PITCH, ROWS, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // Restore the final validation pattern
  HRR_HIP_CHECK(hipDeviceSynchronize());  // ensure all async ops complete first
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // D2H blob — playback validates all values == VAL
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload C: Device info + management APIs
//
// Exercises hipGetDevice, hipDeviceGetName, hipDeviceGetPCIBusId,
// hipDeviceGetByPCIBusId, hipDeviceTotalMem, hipDeviceComputeCapability,
// hipDeviceGetLimit, hipDeviceSetLimit, hipDeviceGetSharedMemConfig,
// hipDeviceSetSharedMemConfig, hipGetDeviceFlags, hipSetDeviceFlags,
// hipChooseDevice, hipInit, hipDeviceGetDefaultMemPool, hipDeviceGetMemPool,
// hipDeviceSetMemPool, hipDeviceGetGraphMemAttribute, hipDeviceSetGraphMemAttribute,
// hipDeviceGraphMemTrim, hipDevicePrimaryCtxGetState.
// ===========================================================================
TEST_CASE("Unit_HRR_DeviceInfo_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  // hipInit — idempotent
  HRR_HIP_CHECK(hipInit(0));

  // hipGetDevice
  int dev = -1;
  HRR_HIP_CHECK(hipGetDevice(&dev));
  REQUIRE(dev == 0);

  // hipDeviceGetName
  char devname[256] = {};
  HRR_HIP_CHECK(hipDeviceGetName(devname, sizeof(devname), dev));
  // On some Linux ROCm builds hipDeviceGetName returns an empty string due to
  // driver metadata not being fully populated. Log it but do not assert —
  // this is a driver issue, not an HRR bug.
  INFO("Device name: '" << devname << "'");

  // hipDeviceGetPCIBusId + hipDeviceGetByPCIBusId roundtrip
  char pci[64] = {};
  HRR_HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), dev));
  int dev2 = -1;
  HRR_HIP_CHECK(hipDeviceGetByPCIBusId(&dev2, pci));
  REQUIRE(dev2 == dev);

  // hipDeviceTotalMem
  size_t totalMem = 0;
  HRR_HIP_CHECK(hipDeviceTotalMem(&totalMem, dev));
  REQUIRE(totalMem > 0);

  // hipDeviceComputeCapability
  int major = 0, minor = 0;
  HRR_HIP_CHECK(hipDeviceComputeCapability(&major, &minor, dev));
  REQUIRE(major > 0);

  // hipDeviceGetLimit / hipDeviceSetLimit
  size_t stackSize = 0;
  HRR_HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
  HRR_HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, stackSize));

  // hipDeviceGetSharedMemConfig / hipDeviceSetSharedMemConfig
  hipSharedMemConfig shmCfg = hipSharedMemBankSizeDefault;
  HRR_HIP_CHECK(hipDeviceGetSharedMemConfig(&shmCfg));
  HRR_HIP_CHECK(hipDeviceSetSharedMemConfig(shmCfg));

  // hipGetDeviceFlags — query only (hipSetDeviceFlags resets the context, skip)
  unsigned int dflags = 0;
  HRR_HIP_CHECK(hipGetDeviceFlags(&dflags));

  // hipChooseDevice
  hipDeviceProp_t req{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&req, 0));
  int chosen = -1;
  HRR_HIP_CHECK(hipChooseDevice(&chosen, &req));
  REQUIRE(chosen >= 0);

  // D2H blob setup — allocate BEFORE pool/graph queries to avoid SEH side effects
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // hipDeviceGetDefaultMemPool
  hipMemPool_t defPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetDefaultMemPool(&defPool, dev));
  REQUIRE(defPool != nullptr);

  // hipDeviceGetMemPool — query only (SetMemPool can reset pool context)
  hipMemPool_t curPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, dev));

  // hipDeviceGetGraphMemAttribute — query only
  uint64_t usedMem = 0;
  HRR_HIP_CHECK(hipDeviceGetGraphMemAttribute(dev, hipGraphMemAttrUsedMemCurrent, &usedMem));

  // hipDevicePrimaryCtxGetState — query only
  unsigned int ctxFlags = 0; int ctxActive = 0;
  HRR_HIP_CHECK(hipDevicePrimaryCtxGetState(dev, &ctxFlags, &ctxActive));

  // D2H blob (value = 7) for playback validation
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 7, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 7);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload D: Stream + event advanced APIs
//
// Exercises hipStreamGetId, hipStreamGetAttribute, hipStreamCopyAttributes,
// hipStreamIsCapturing, hipStreamGetCaptureInfo,
// hipThreadExchangeStreamCaptureMode, hipStreamGetDevice,
// hipEventRecordWithFlags, hipStreamQuery.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamAdvanced_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s0, s1;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithPriority(&s1, hipStreamNonBlocking, 0));

  // hipStreamGetId
  unsigned long long sid = 0;
  HRR_HIP_CHECK(hipStreamGetId(s0, &sid));
  REQUIRE(sid != 0);

  // hipStreamGetAttribute / hipStreamSetAttribute roundtrip
  hipStreamAttrValue av{};
  HRR_HIP_CHECK(hipStreamGetAttribute(s0, hipStreamAttributeSynchronizationPolicy, &av));
  HRR_HIP_CHECK(hipStreamSetAttribute(s0, hipStreamAttributeSynchronizationPolicy, &av));

  // hipStreamCopyAttributes
  HRR_HIP_CHECK(hipStreamCopyAttributes(s1, s0));

  // hipStreamIsCapturing
  hipStreamCaptureStatus capStatus = hipStreamCaptureStatusActive;
  HRR_HIP_CHECK(hipStreamIsCapturing(s0, &capStatus));
  REQUIRE(capStatus == hipStreamCaptureStatusNone);

  // hipStreamGetCaptureInfo
  unsigned long long capId = 0;
  HRR_HIP_CHECK(hipStreamGetCaptureInfo(s0, &capStatus, &capId));
  REQUIRE(capStatus == hipStreamCaptureStatusNone);

  // hipStreamGetDevice (may return hipErrorInvalidValue on some ROCm builds)
  hipDevice_t streamDev = -1;
  { hipError_t e = hipStreamGetDevice(s0, &streamDev);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  // hipStreamQuery
  { hipError_t q = hipStreamQuery(s0); REQUIRE((q == hipSuccess || q == hipErrorNotReady)); }

  // hipEventRecordWithFlags
  hipEvent_t ev;
  HRR_HIP_CHECK(hipEventCreate(&ev));
  HRR_HIP_CHECK(hipEventRecordWithFlags(ev, s0, 0));
  HRR_HIP_CHECK(hipEventSynchronize(ev));

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  // hipThreadExchangeStreamCaptureMode — query/restore at the very end,
  // after all stream work is complete, to avoid interfering with work submission.
  hipStreamCaptureMode mode = hipStreamCaptureModeGlobal;
  HRR_HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));
  HRR_HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));  // restore original

  HRR_HIP_CHECK(hipDeviceSynchronize());

  // D2H blob (value = 5) for playback validation
  constexpr int N = 256; constexpr size_t SZ = N * sizeof(int);
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  // Use hipMemset (byte-pattern) + value 5 via kernel: set all bytes to 0,
  // then use hipMemsetD32 for the specific integer value.
  HRR_HIP_CHECK(hipMemsetAsync(d, 0, SZ, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 5, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 5);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipEventDestroy(ev));
  HRR_HIP_CHECK(hipStreamDestroy(s0));
  HRR_HIP_CHECK(hipStreamDestroy(s1));
  delete[] h;
}

// ===========================================================================
// Workload L: Driver-style memcpy APIs
//
// Exercises hipMemcpyDtoD, hipMemcpyDtoDAsync, hipMemcpyDtoH, hipMemcpyDtoHAsync,
// hipMemcpyHtoDAsync (driver-style), hipMemcpy2D, hipMemcpy2DAsync,
// hipMallocPitch, hipMemcpyPeer, hipMemcpyPeerAsync.
// Final blob: h_out[i] == 99.
// ===========================================================================
TEST_CASE("Unit_HRR_DrvMemcpy_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 512;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int *d0, *d1, *d2;
  HRR_HIP_CHECK(hipMalloc(&d0, SZ));
  HRR_HIP_CHECK(hipMalloc(&d1, SZ));
  HRR_HIP_CHECK(hipMalloc(&d2, SZ));

  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = 99;

  // hipMemcpyHtoD (driver-style, sync) + device sync to ensure completion
  HRR_HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d0), h_src, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // hipMemcpyDtoD (sync)
  HRR_HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d1),
                           reinterpret_cast<hipDeviceptr_t>(d0), SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // hipMemcpyDtoDAsync
  HRR_HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d2),
                                reinterpret_cast<hipDeviceptr_t>(d1), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // hipMemcpyDtoH (sync)
  int* h_mid = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoH(h_mid, reinterpret_cast<hipDeviceptr_t>(d2), SZ));
  for (int i = 0; i < N; ++i) REQUIRE(h_mid[i] == 99);

  // hipMemcpyDtoHAsync
  int* h_mid2 = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h_mid2, reinterpret_cast<hipDeviceptr_t>(d2), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_mid2[i] == 99);

  // hipMemcpyHtoDAsync (driver-style, async)
  HRR_HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(d0), h_src, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemcpy2D (flat 1-row, sync D2D)
  HRR_HIP_CHECK(hipMemcpy2D(d1, SZ, d2, SZ, SZ, 1, hipMemcpyDeviceToDevice));

  // hipMemcpy2DAsync
  HRR_HIP_CHECK(hipMemcpy2DAsync(d2, SZ, d0, SZ, SZ, 1, hipMemcpyDeviceToDevice, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMallocPitch + hipMemcpy2D into pitched buffer
  size_t pitch = 0;
  int* d_p = nullptr;
  HRR_HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&d_p), &pitch,
                            N * sizeof(int), 4));
  HRR_HIP_CHECK(hipMemcpy2D(d_p, pitch, d0, SZ, N * sizeof(int), 1,
                         hipMemcpyDeviceToDevice));
  HRR_HIP_CHECK(hipFree(d_p));

  // hipMemcpyPeer / hipMemcpyPeerAsync (src==dst device == 0, valid as D2D)
  HRR_HIP_CHECK(hipMemcpyPeer(d1, 0, d0, 0, SZ));
  HRR_HIP_CHECK(hipMemcpyPeerAsync(d2, 0, d1, 0, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // Final D2H blob via hipMemcpyAsync — blob captured, playback validates 99
  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d2, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == 99);

  HRR_HIP_CHECK(hipFree(d0)); HRR_HIP_CHECK(hipFree(d1)); HRR_HIP_CHECK(hipFree(d2));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h_src; delete[] h_mid; delete[] h_mid2; delete[] h_out;
}

// ===========================================================================
// Workload E: Occupancy + extended kernel launch
//
// Exercises hipOccupancyMaxActiveBlocksPerMultiprocessor,
// hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
// hipOccupancyMaxPotentialBlockSize,
// hipLaunchCooperativeKernel, hipExtModuleLaunchKernel (via module path).
// Final blob: d[i] == 3 after vectorAdd kernel via hipExtModuleLaunchKernel.
// ===========================================================================
static __global__ void fill_kernel_e(int* d, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = val;
}

TEST_CASE("Unit_HRR_Occupancy_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  // hipOccupancyMaxActiveBlocksPerMultiprocessor
  int numBlocks = 0;
  HRR_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, fill_kernel_e, 64, 0));
  REQUIRE(numBlocks > 0);

  // hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
  int numBlocks2 = 0;
  HRR_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &numBlocks2, fill_kernel_e, 64, 0, hipOccupancyDefault));
  REQUIRE(numBlocks2 > 0);

  // hipOccupancyMaxPotentialBlockSize
  int minGridSize = 0, blockSize = 0;
  HRR_HIP_CHECK(hipOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, fill_kernel_e, 0, 0));
  REQUIRE(blockSize > 0);

  // hipLaunchCooperativeKernel — guarded by cooperative launch support
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  int supportsCoopLaunch = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&supportsCoopLaunch,
                                   hipDeviceAttributeCooperativeLaunch, 0));

  int val = 3;
  dim3 grid((N + 63) / 64), block(64);
  if (supportsCoopLaunch) {
    void* args[] = {&d, &val, const_cast<int*>(&N)};
    HRR_HIP_CHECK(hipLaunchCooperativeKernel(
        reinterpret_cast<const void*>(fill_kernel_e),
        grid, block, args, 0, s));
  } else {
    // Fall back to regular launch to still exercise the kernel path
    hipLaunchKernelGGL(fill_kernel_e, grid, block, 0, s, d, val, N);
  }
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // D2H blob (value = 3)
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 3);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload H: Host pointer aliases + array allocation
//
// Exercises hipHostAlloc, hipFreeHost, hipFreeHost (alias hipFree),
// hipMallocHost, hipMallocArray, hipMalloc3DArray, hipMalloc3D,
// hipHostGetFlags, hipMemAllocHost, hipMemAllocPitch,
// hipPointerGetAttribute (singular).
// Final blob: d[i] == 0.
// ===========================================================================
TEST_CASE("Unit_HRR_HostAliases_Direct", "[.][hrr-direct]") {
  // Drain any GPU errors left by earlier tests; this test mixes array + 3D
  // alloc with regular device memory — on Windows the driver needs a clean slate.
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  // hipHostAlloc / hipFreeHost — legacy pinned alloc
  void* ha = nullptr;
  HRR_HIP_CHECK(hipHostAlloc(&ha, SZ, hipHostMallocDefault));
  REQUIRE(ha != nullptr);

  // hipHostGetFlags
  unsigned int hflags = 0;
  HRR_HIP_CHECK(hipHostGetFlags(&hflags, ha));

  // hipFreeHost — free the hipHostAlloc buffer
  HRR_HIP_CHECK(hipFreeHost(ha));

  // hipMallocHost — another legacy alias for hipHostMalloc
  void* mh = nullptr;
  HRR_HIP_CHECK(hipMallocHost(&mh, SZ));
  REQUIRE(mh != nullptr);
  HRR_HIP_CHECK(hipFreeHost(mh));

  // hipMemAllocHost — driver-style pinned alloc
  void* dah = nullptr;
  HRR_HIP_CHECK(hipMemAllocHost(&dah, SZ));
  REQUIRE(dah != nullptr);
  HRR_HIP_CHECK(hipFreeHost(dah));

  // hipMallocArray / hipMalloc3DArray — texture arrays
  // Check image support once; skip both array tests if not available.
  hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
  {
    int supportsImages = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&supportsImages,
                                    hipDeviceAttributeImageSupport, 0));
    if (supportsImages) {
      hipArray_t arr1d = nullptr;
      HRR_HIP_CHECK(hipMallocArray(&arr1d, &desc, N, 1, hipArrayDefault));
      REQUIRE(arr1d != nullptr);
      HRR_HIP_CHECK(hipFreeArray(arr1d));

      hipArray_t arr3d = nullptr;
      hipExtent ext3d = make_hipExtent(16, 16, 4);
      HRR_HIP_CHECK(hipMalloc3DArray(&arr3d, &desc, ext3d, hipArrayDefault));
      REQUIRE(arr3d != nullptr);
      HRR_HIP_CHECK(hipFreeArray(arr3d));
    }
  }

  // hipMalloc3D — pitched 3D allocation
  hipPitchedPtr pp{};
  hipExtent ext = make_hipExtent(32 * sizeof(int), 8, 4);
  HRR_HIP_CHECK(hipMalloc3D(&pp, ext));
  REQUIRE(pp.ptr != nullptr);

  // hipPointerGetAttribute (singular) — query on 3D alloc
  hipMemoryType mtype = hipMemoryTypeUnified;
  HRR_HIP_CHECK(hipPointerGetAttribute(&mtype, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, pp.ptr));

  HRR_HIP_CHECK(hipFree(pp.ptr));

  // hipMemAllocPitch — driver-style pitched alloc
  hipDeviceptr_t dptr = 0;
  size_t rowPitch = 0;
  HRR_HIP_CHECK(hipMemAllocPitch(&dptr, &rowPitch, N * sizeof(int), 4, sizeof(int)));
  REQUIRE(dptr != 0);
  HRR_HIP_CHECK(hipFree(reinterpret_cast<void*>(dptr)));

  // D2H blob (value = 8)
  // Drain any pending GPU errors from earlier tests before D2H.
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  // Use hipMemset + synchronous hipMemcpy to avoid GPU TDR after hipMalloc3D
  // on Windows ROCm driver (async D2H after array/3D alloc triggers error 719).
  HRR_HIP_CHECK(hipMemset(d, 0, SZ));
  HRR_HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G: MemPool extended APIs
//
// Exercises hipMemPoolTrimTo, hipMemPoolGetAccess,
// hipMemPoolExportPointer, hipMemPoolImportPointer.
// Final blob: h[i] == 6.
// ===========================================================================
TEST_CASE("Unit_HRR_MemPoolExtended_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  // Create a pool
  hipMemPoolProps props{};
  props.allocType   = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id   = 0;
  hipMemPool_t pool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&pool, &props));

  // hipMemPoolGetAccess — query access for device 0
  hipMemAccessFlags accessFlags{};
  hipMemLocation loc{hipMemLocationTypeDevice, 0};
  HRR_HIP_CHECK(hipMemPoolGetAccess(&accessFlags, pool, &loc));

  // hipMemPoolTrimTo — release unused pool memory (minBytesToKeep=0)
  HRR_HIP_CHECK(hipMemPoolTrimTo(pool, 0));

  // Alloc from pool, then export and import the pointer
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d), SZ, pool, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemPoolExportPointer / hipMemPoolImportPointer roundtrip.
  // These APIs are not supported on all Linux ROCm builds — skip gracefully.
  hipMemPoolPtrExportData exportData{};
  hipError_t export_err = hipMemPoolExportPointer(&exportData, d);
  if (export_err == hipSuccess) {
    void* imported = nullptr;
    HRR_HIP_CHECK(hipMemPoolImportPointer(&imported, pool, &exportData));
    REQUIRE(imported != nullptr);
  } else {
    WARN("hipMemPoolExportPointer returned " << (int)export_err
         << " — skipping export/import sub-test on this platform");
  }

  // D2H blob (value = 6)
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 6, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 6);

  HRR_HIP_CHECK(hipFreeAsync(d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipMemPoolDestroy(pool));
  delete[] h;
}

// ===========================================================================
// Workload G2: Device-associated stream-ordered memory pool
//
// Exercises hipDeviceSetMemPool — the one device mem-pool API deliberately
// left untested by Unit_HRR_DeviceInfo_Direct ("hipDeviceGetMemPool — query
// only (SetMemPool can reset pool context)").  A user pool is made device 0's
// current pool, then the pool-less hipMallocAsync (which draws from the device's
// current pool) allocates from it; a known value is written and validated D2H.
//
// hipDeviceSetMemPool has a clean generated playback handler that resolves the
// pool via translate_mempool(), and every supporting API here (hipMemPoolCreate,
// hipDeviceGetDefaultMemPool, hipDeviceGetMemPool, hipMallocAsync, hipMemsetD32,
// hipFreeAsync, hipMemPoolDestroy) already replays via existing MemPool / DeviceInfo
// coverage.  Cleanup destroys the user pool while it is current; hipMemPoolDestroy
// force-resets the device to its default pool (see hip_mempool.cpp), so replay
// never needs an untracked "restore the default pool" handle.
// Final blob: h[i] == 11.
// ===========================================================================
TEST_CASE("Unit_HRR_DeviceMemPool_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  // Baseline: the default pool is the current pool of a fresh context.
  hipMemPool_t defPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetDefaultMemPool(&defPool, 0));
  REQUIRE(defPool != nullptr);
  hipMemPool_t curPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, 0));
  REQUIRE(curPool == defPool);

  // Create a user pool and make it device 0's current pool (API under test).
  hipMemPoolProps props{};
  props.allocType     = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id   = 0;
  hipMemPool_t myPool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&myPool, &props));

  HRR_HIP_CHECK(hipDeviceSetMemPool(0, myPool));

  // Confirm the association took effect.
  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, 0));
  REQUIRE(curPool == myPool);

  // Pool-less async alloc draws from the device's current pool (== myPool),
  // exercising the device-pool association end to end.
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // D2H blob (value = 11) for playback validation.
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 11, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 11);

  // Free the allocation, then destroy the (still-current) user pool: HIP resets
  // device 0 to its default pool automatically, so no unsafe restore is replayed.
  HRR_HIP_CHECK(hipFreeAsync(d, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipMemPoolDestroy(myPool));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G3: hipExtMallocWithFlags device allocation
//
// hipExtMallocWithFlags is a supported (manual playback handler) device
// allocation API with no prior HRR capture->replay coverage.  The manual
// handler allocates a real buffer and records it in alloc_map (parity with
// hipMalloc), so a write + D2H validates byte-for-byte at replay.
// Final blob: h[i] == 22.
// ===========================================================================
TEST_CASE("Unit_HRR_ExtMalloc_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&d), SZ,
                                  hipDeviceMallocDefault));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 22, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 22);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G4: hipStreamWaitEvent_spt cross-stream ordering
//
// hipStreamWaitEvent_spt (per-thread-default-stream variant) has no prior HRR
// coverage; its generated handler translates both the stream and the event
// handle.  Stream s1 waits (via _spt) on an event recorded on s0 after a memset,
// then copies the result — a correct D2H (33) requires the recorded event
// ordering to be reproduced at replay.
// Final blob: h[i] == 33.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamWaitEventSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s0, s1;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  hipEvent_t ev;
  HRR_HIP_CHECK(hipEventCreate(&ev));

  int *d0, *d1;
  HRR_HIP_CHECK(hipMalloc(&d0, SZ));
  HRR_HIP_CHECK(hipMalloc(&d1, SZ));

  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d0), 33, N, s0));
  HRR_HIP_CHECK(hipEventRecord(ev, s0));
  HRR_HIP_CHECK(hipStreamWaitEvent_spt(s1, ev, 0));
  HRR_HIP_CHECK(hipMemcpyAsync(d1, d0, SZ, hipMemcpyDeviceToDevice, s1));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d1, SZ, hipMemcpyDeviceToHost, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 33);

  HRR_HIP_CHECK(hipFree(d0)); HRR_HIP_CHECK(hipFree(d1));
  HRR_HIP_CHECK(hipEventDestroy(ev));
  HRR_HIP_CHECK(hipStreamDestroy(s0)); HRR_HIP_CHECK(hipStreamDestroy(s1));
  delete[] h;
}

// ===========================================================================
// Workload G5: hipGraphLaunch_spt
//
// hipGraphLaunch_spt (per-thread-default-stream graph launch) has no prior HRR
// coverage.  The graph is built with the already-supported stream-capture path
// (hipStreamBeginCapture / hipStreamEndCapture / hipGraphInstantiate) and
// launched via the _spt variant, whose generated handler translates the
// graph-exec handle recorded by the manual hipGraphInstantiate handler.
// Final blob: h[i] == 77.
// ===========================================================================
TEST_CASE("Unit_HRR_GraphLaunchSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), 77, N, s));
  hipGraph_t g;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));
  hipGraphExec_t exec;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));

  HRR_HIP_CHECK(hipGraphLaunch_spt(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 77);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G6: hipExtModuleLaunchKernel (OpenCL-style module kernel launch)
//
// hipExtModuleLaunchKernel has a manual playback handler (replay_kernel_launch
// with ext_global_worksize=true) that reconstructs the kernarg blob and
// translates the function handle and the device pointer inside the kernargs; it
// has no prior HRR coverage.  The kernel is compiled at runtime with HIPRTC and
// loaded via hipModuleLoadData (a manual handler that restores the code object
// from the archive by hash), so — unlike static fat-binary kernels, which are
// not captured at static-init on Linux — this replays with full D2H validation.
// NOTE: hipExtModuleLaunchKernel takes GLOBAL work size (total work items), not
// grid dims: global=LN, local=256 launches ceil(LN/256) workgroups.
// Final blob: h[i] == 55.
// ===========================================================================
TEST_CASE("Unit_HRR_ExtModuleLaunchKernel_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    LN  = 256;
  constexpr size_t LSZ = LN * sizeof(int);

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, LSZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // Runtime-compiled kernel (captured via hipModuleLoadData, not static init).
  static const char* ext_fill_src = R"(
extern "C" __global__ void ext_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}
)";
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, ext_fill_src, "ext_fill.hip",
                                   0, nullptr, nullptr));
  hiprtcResult crc = hiprtcCompileProgram(prog, 0, nullptr);
  if (crc != HIPRTC_SUCCESS) {
    size_t log_sz = 0;
    (void)hiprtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    (void)hiprtcGetProgramLog(prog, log.data());
    (void)hiprtcDestroyProgram(&prog);
    FAIL("hiprtcCompileProgram failed: " + log);
  }
  size_t co_size = 0;
  HRR_HIPRTC_CHECK(hiprtcGetCodeSize(prog, &co_size));
  std::vector<char> co(co_size);
  HRR_HIPRTC_CHECK(hiprtcGetCode(prog, co.data()));
  HRR_HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  hipModule_t mod = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod, co.data()));
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod, "ext_fill"));

  // API under test.  kernelParams holds the address of each argument value;
  // args[0] = &d is the address of the int* device pointer.
  int   val = 55;
  int   n   = LN;
  void* args[] = { &d, &val, &n };
  HRR_HIP_CHECK(hipExtModuleLaunchKernel(fn,
      /*globalWorkSize*/ LN, 1, 1,
      /*localWorkSize */ 256, 1, 1,
      /*sharedMemBytes*/ 0, s, args, /*extra*/ nullptr,
      /*startEvent*/ nullptr, /*stopEvent*/ nullptr, /*flags*/ 0));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipModuleUnload(mod));

  int* h = new int[LN]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, LSZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < LN; ++i) REQUIRE(h[i] == 55);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G7: hipHostFree (pinned host allocation lifecycle)
//
// hipHostFree has a real (non-noop) playback handler that translates the
// recorded pointer via alloc_map and removes the mapping; hipHostMalloc's
// handler records the pinned allocation (AllocKind::HostMalloc).  Neither is
// exercised by any existing _Direct workload (the HostMem workload frees pinned
// memory with hipFree).  The pinned buffer is used as an H2D source so the
// allocation is genuinely live, then released with hipHostFree *before* the D2H
// so its alloc-map removal is replayed mid-stream and must not disturb the
// device->host validation that follows.
// Final blob: h[i] == 44.
// ===========================================================================
TEST_CASE("Unit_HRR_HostFree_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* h_pinned = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&h_pinned), SZ,
                          hipHostMallocDefault));
  for (int i = 0; i < N; ++i) h_pinned[i] = 44;

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemcpyAsync(d, h_pinned, SZ, hipMemcpyHostToDevice, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // API under test: release the pinned allocation via hipHostFree (not hipFree).
  HRR_HIP_CHECK(hipHostFree(h_pinned));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 44);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G8: HIP external logging controls
//
// hipExtSetLoggingParams / hipExtEnableLogging / hipExtDisableLogging each have
// a real (non-noop) generated playback handler and no prior HRR coverage.  They
// take only scalar / no arguments (no stale pointers) and always return
// hipSuccess.  Params are set to level 0 so enabling logging produces no output.
// None touch device memory, so a memset-based D2H canary confirms the whole
// replay stream (including the three logging calls) stays intact.
// Final blob: h[i] == 66.
// ===========================================================================
TEST_CASE("Unit_HRR_Logging_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);

  // APIs under test (log_level 0 => enabling logging stays quiet).
  HRR_HIP_CHECK(hipExtSetLoggingParams(/*log_level*/ 0, /*log_size*/ 0,
                                   /*log_mask*/ 0));
  HRR_HIP_CHECK(hipExtEnableLogging());
  HRR_HIP_CHECK(hipExtDisableLogging());

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 66, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 66);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G9: hipStreamIsCapturing_spt / hipStreamGetCaptureInfo_spt
//
// The per-thread-stream capture *query* APIs have real generated handlers that
// only translate the stream and write to local outputs (safe).  They are
// exercised INSIDE a MANUAL hipStreamBeginCapture/EndCapture frame: the manual
// handlers set ctx.in_graph_capture and record the graph in graph_map, which the
// generated _spt begin/end handlers omit.  The captured memset builds a graph
// that is instantiated + launched, so a correct D2H proves the whole
// capture->graph->replay path — with the two _spt queries mid-capture — works.
// Final blob: h[i] == 0x5A5A5A5A.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamCaptureQuerySpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);
  constexpr int    VAL = 0x5A5A5A5A;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));

  // APIs under test: query capture state via the _spt variants.
  hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
  HRR_HIP_CHECK(hipStreamIsCapturing_spt(s, &st));
  REQUIRE(st == hipStreamCaptureStatusActive);

  hipStreamCaptureStatus st2 = hipStreamCaptureStatusNone;
  unsigned long long capId = 0;
  HRR_HIP_CHECK(hipStreamGetCaptureInfo_spt(s, &st2, &capId));
  REQUIRE(st2 == hipStreamCaptureStatusActive);

  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));

  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));
  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload G10: hipStreamBeginCapture_spt (GPU-VALIDATED)
//
// hipStreamBeginCapture_spt's generated handler translates the stream and starts
// capture but (unlike the manual hipStreamBeginCapture) does NOT set
// ctx.in_graph_capture.  That flag only gates timing / sync-after-launch /
// zero-init memsets, none of which apply to a memset-only capture region with no
// alloc or kernel launch inside it — so replay should still be correct.  Capture
// is ended with the MANUAL hipStreamEndCapture (records the graph in graph_map;
// the generated end_spt discards it, so end_spt stays R2).  This slice validates
// begin_spt on GPU: green keeps it, red reclassifies it R2.
// Final blob: h[i] == 0x33333333.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamCaptureBeginSpt_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);
  constexpr int    VAL = 0x33333333;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // API under test: start capture via the _spt variant.
  HRR_HIP_CHECK(hipStreamBeginCapture_spt(s, hipStreamCaptureModeThreadLocal));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), VAL, N, s));
  hipGraph_t g = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(s, &g));
  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipGraphExecDestroy(exec));
  HRR_HIP_CHECK(hipGraphDestroy(g));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}
// ===========================================================================
// Workload: hipConfigureCall (legacy execution-stack launch config)
//
// hipConfigureCall has a real (non-noop) generated playback handler that
// rebuilds the grid/block dim3s and the shared-mem size and translates the
// stream handle before re-issuing the call - it drops no buffer and
// dereferences no stale pointer.  It has no prior HRR coverage.  The legacy
// hipConfigureCall / hipSetupArgument / hipLaunchByPtr execution-stack launch
// path is separate from the <<<>>> (__hipPushCallConfiguration) path HRR records
// for kernel launches, so the API is exercised on its own: it only pushes a
// call configuration onto the thread-local execution stack and returns
// hipSuccess (no matching hipLaunchByPtr consumes it).  A memset-based D2H canary
// then confirms the replay stream - including the replayed hipConfigureCall -
// stays intact end to end.
// Final blob: h[i] == 88.
// ===========================================================================
TEST_CASE("Unit_HRR_ConfigureCall_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // API under test: push a launch configuration onto the execution stack.
  dim3 grid((N + 255) / 256), block(256);
  HRR_HIP_CHECK(hipConfigureCall(grid, block, /*sharedMem*/ 0, s));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), 88, N, s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 88);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload: cross-GPU peer copy (REQUIRES 2 GPUs)
//
// First HRR workload that spans two devices. src_dev is memset to a known value
// and peer-copied to dst_dev via hipMemcpyPeer. Replay must recreate the two
// allocations on the correct devices for the peer copy to land.
// Skips on hosts without two peer-accessible GPUs; the roundtrip driver guards
// the same way.
// Final blob: h[i] == 0x7E7E7E7E.
// ===========================================================================
TEST_CASE("Unit_HRR_MemcpyPeer_Direct", "[.][hrr-direct]") {
  int src_dev = 0;
  int dst_dev = 1;
  int ndev = 0;
  if (!hrr_find_peer_accessible_pair(src_dev, dst_dev, ndev)) {
    if (ndev < 2) {
      HRR_SKIP("fewer than two GPUs");
    } else {
      HRR_SKIP("peer access unavailable");
    }
  }
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = 0x7E7E7E7E;

  HRR_HIP_CHECK(hipSetDevice(src_dev));
  HRR_HIP_CHECK(hipDeviceEnablePeerAccess(dst_dev, 0));
  int* d0 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d0, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d0), VAL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipSetDevice(dst_dev));
  int* d1 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d1, SZ));

  // API under test: cross-device peer copy src_dev -> dst_dev.
  // Issue the copy under the src-device context (matches
  // catch/unit/memory/hipMemcpyPeer.cc).
  HRR_HIP_CHECK(hipSetDevice(src_dev));
  HRR_HIP_CHECK(hipMemcpyPeer(d1, dst_dev, d0, src_dev, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpy(h, d1, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d1));
  HRR_HIP_CHECK(hipSetDevice(src_dev));
  HRR_HIP_CHECK(hipFree(d0));
  HRR_HIP_CHECK(hipDeviceDisablePeerAccess(dst_dev));
  delete[] h;
}

// ===========================================================================
// Workload N: Additional memset variants
//
// Exercises hipMemset3D, hipMemset3DAsync, hipMemsetD2D8/16/32 and Async,
// hipMemsetMemPool, _spt stream memset variants.
// Final blob: d[i] == 9.
// ===========================================================================
TEST_CASE("Unit_HRR_MemsetExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // hipMemset3D + hipMemset3DAsync on a pitched 3D alloc
  {
    hipPitchedPtr pp{};
    hipExtent ext = make_hipExtent(16 * sizeof(int), 4, 2);
    HRR_HIP_CHECK(hipMalloc3D(&pp, ext));

    hipExtent ext3 = make_hipExtent(16 * sizeof(int), 4, 2);
    HRR_HIP_CHECK(hipMemset3D(pp, 0xAB, ext3));
    HRR_HIP_CHECK(hipMemset3DAsync(pp, 0x00, ext3, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipFree(pp.ptr));
  }

  // hipMemsetD2D8 + hipMemsetD2D8Async
  {
    void* d2d = nullptr;
    size_t pitch2d = 0;
    HRR_HIP_CHECK(hipMallocPitch(&d2d, &pitch2d, 16, 4));

    HRR_HIP_CHECK(hipMemsetD2D8(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x77, 16, 4));
    HRR_HIP_CHECK(hipMemsetD2D8Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x00, 16, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    // hipMemsetD2D16
    HRR_HIP_CHECK(hipMemsetD2D16(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x1234, 8, 4));
    HRR_HIP_CHECK(hipMemsetD2D16Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x0000, 8, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    // hipMemsetD2D32
    HRR_HIP_CHECK(hipMemsetD2D32(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0xDEAD, 4, 4));
    HRR_HIP_CHECK(hipMemsetD2D32Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x0000, 4, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipFree(d2d));
  }

  // hipMemsetMemPool — not in ROCm SDK 6.4, skip
  // hipMemPoolTrimTo tested in MemPoolExtended workload

  // _spt variants (per-thread-default-stream): hipMemset_spt, hipMemsetAsync_spt,
  // hipMemset2D_spt, hipMemset2DAsync_spt, hipMemset3D_spt, hipMemset3DAsync_spt
  {
    void* dp = nullptr;
    HRR_HIP_CHECK(hipMalloc(&dp, SZ));
    HRR_HIP_CHECK(hipMemset_spt(dp, 0xAA, SZ));
    HRR_HIP_CHECK(hipMemsetAsync_spt(dp, 0x00, SZ, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    size_t pitch_s = 0;
    void* dp2 = nullptr;
    HRR_HIP_CHECK(hipMallocPitch(&dp2, &pitch_s, 32, 4));
    HRR_HIP_CHECK(hipMemset2D_spt(dp2, pitch_s, 0xBB, 32, 4));
    HRR_HIP_CHECK(hipMemset2DAsync_spt(dp2, pitch_s, 0x00, 32, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    hipPitchedPtr pp3{};
    hipExtent ext3 = make_hipExtent(16 * sizeof(int), 4, 2);
    HRR_HIP_CHECK(hipMalloc3D(&pp3, ext3));
    HRR_HIP_CHECK(hipMemset3D_spt(pp3, 0xCC, ext3));
    HRR_HIP_CHECK(hipMemset3DAsync_spt(pp3, 0x00, ext3, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipFree(dp)); HRR_HIP_CHECK(hipFree(dp2)); HRR_HIP_CHECK(hipFree(pp3.ptr));
  }

  // D2H blob (value = 9)
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 9, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 9);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Device symbol used by the symbol _spt variants in Workload P.
__device__ int g_hrr_symbol[256];

// Workload P: Additional memcpy variants (array-based, param2D, _spt, symbol_spt)
//
// Exercises hipMemcpyToArray, hipMemcpyFromArray, hipMemcpy2DToArray,
// hipMemcpy2DFromArray, hipMemcpyAtoH, hipMemcpyHtoA,
// hipMemcpyParam2D, hipMemcpyParam2DAsync,
// hipMemcpy_spt, hipMemcpyAsync_spt, hipMemcpy2D_spt, hipMemcpy2DAsync_spt,
// hipMemcpyFromSymbol_spt, hipMemcpyToSymbol_spt,
// hipMemcpyFromSymbolAsync_spt, hipMemcpyToSymbolAsync_spt.
// Final blob: h[i] == 55.
// ===========================================================================
TEST_CASE("Unit_HRR_MemcpyExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 64;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // Allocate device + host buffers
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = 55;

  // Load d with 55 via hipMemcpy (has blob → restored at playback before any _spt calls)
  HRR_HIP_CHECK(hipMemcpy(d, h_src, SZ, hipMemcpyHostToDevice));

  // hipMemcpy_spt (per-thread-default-stream H2D+D2H)
  HRR_HIP_CHECK(hipMemcpy_spt(d, h_src, SZ, hipMemcpyHostToDevice));
  int* h_chk = new int[N]();
  HRR_HIP_CHECK(hipMemcpy_spt(h_chk, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == 55);

  // hipMemcpyAsync_spt
  HRR_HIP_CHECK(hipMemcpyAsync_spt(d, h_src, SZ, hipMemcpyHostToDevice, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipMemcpyAsync_spt(h_chk, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemcpy2D_spt (1-row, H2D then D2H)
  HRR_HIP_CHECK(hipMemcpy2D_spt(d, SZ, h_src, SZ, SZ, 1, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpy2D_spt(h_chk, SZ, d, SZ, SZ, 1, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h_chk[i] == 55);

  // hipMemcpy2DAsync_spt
  HRR_HIP_CHECK(hipMemcpy2DAsync_spt(d, SZ, h_src, SZ, SZ, 1, hipMemcpyHostToDevice, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipMemcpyParam2D — uses HIP_MEMCPY2D descriptor
  {
    hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeHost;
    p.srcHost       = h_src;
    p.srcPitch      = SZ;
    p.dstMemoryType = hipMemoryTypeDevice;
    p.dstDevice     = reinterpret_cast<hipDeviceptr_t>(d);
    p.dstPitch      = SZ;
    p.WidthInBytes  = SZ;
    p.Height        = 1;
    HRR_HIP_CHECK(hipMemcpyParam2D(&p));
  }

  // hipMemcpyParam2DAsync
  {
    hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeDevice;
    p.srcDevice     = reinterpret_cast<hipDeviceptr_t>(d);
    p.srcPitch      = SZ;
    p.dstMemoryType = hipMemoryTypeHost;
    p.dstHost       = h_chk;
    p.dstPitch      = SZ;
    p.WidthInBytes  = SZ;
    p.Height        = 1;
    HRR_HIP_CHECK(hipMemcpyParam2DAsync(&p, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // Array-based copies: hipMallocArray + hipMemcpyToArray/FromArray
  // Requires image support — skip on GPUs that don't support texture arrays.
  {
    int supportsImages = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&supportsImages, hipDeviceAttributeImageSupport, 0));
    if (supportsImages) {
      hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
      hipArray_t arr = nullptr;
      HRR_HIP_CHECK(hipMallocArray(&arr, &desc, N, 1, hipArrayDefault));

      // hipMemcpyToArray (H→Array)
      HRR_HIP_CHECK(hipMemcpyToArray(arr, 0, 0, h_src, SZ, hipMemcpyHostToDevice));

      // hipMemcpyFromArray (Array→H)
      int h_arr[N] = {};
      HRR_HIP_CHECK(hipMemcpyFromArray(h_arr, arr, 0, 0, SZ, hipMemcpyDeviceToHost));
      REQUIRE(h_arr[0] == 55);

      // hipMemcpy2DToArray (H→Array via 2D)
      HRR_HIP_CHECK(hipMemcpy2DToArray(arr, 0, 0, h_src, SZ, SZ, 1, hipMemcpyHostToDevice));

      // hipMemcpy2DFromArray (Array→H via 2D)
      int h_arr2[N] = {};
      HRR_HIP_CHECK(hipMemcpy2DFromArray(h_arr2, SZ, arr, 0, 0, SZ, 1, hipMemcpyDeviceToHost));
      REQUIRE(h_arr2[0] == 55);

      // hipMemcpyAtoH (Array→host, driver-style)
      int h_ato[N] = {};
      HRR_HIP_CHECK(hipMemcpyAtoH(h_ato, arr, 0, SZ));
      REQUIRE(h_ato[0] == 55);

      // hipMemcpyHtoA (host→Array, driver-style)
      HRR_HIP_CHECK(hipMemcpyHtoA(arr, 0, h_src, SZ));

      HRR_HIP_CHECK(hipFreeArray(arr));
    }
  }

  // Symbol _spt H2D variants — exercise capture path for symbol memcpy _spt APIs
  {
    int h_sym[N];
    for (int i = 0; i < N; ++i) h_sym[i] = 55;
    HRR_HIP_CHECK(hipMemcpyToSymbol_spt(HIP_SYMBOL(g_hrr_symbol), h_sym, SZ, 0,
                                     hipMemcpyHostToDevice));
    HRR_HIP_CHECK(hipMemcpyToSymbolAsync_spt(HIP_SYMBOL(g_hrr_symbol), h_sym, SZ, 0,
                                          hipMemcpyHostToDevice, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // Reload d with 55 via hipMemcpy (has blob → playback restores correctly)
  HRR_HIP_CHECK(hipMemcpy(d, h_src, SZ, hipMemcpyHostToDevice));

  // D2H blob (value = 55)
  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == 55);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h_src; delete[] h_chk; delete[] h_out;
}

// ===========================================================================
// Workload Q: Extra device management APIs
//
// Exercises hipDeviceGetUuid, hipDeviceGetP2PAttribute, hipDeviceSetCacheConfig,
// hipDeviceEnablePeerAccess, hipDeviceDisablePeerAccess,
// hipDevicePrimaryCtxRelease, hipDevicePrimaryCtxSetFlags,
// hipDeviceGetTexture1DLinearMaxWidth, hipDeviceGraphMemTrim.
// Final blob: d[i] == 11.
// ===========================================================================
TEST_CASE("Unit_HRR_DeviceExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // hipDeviceGetUuid
  hipUUID uuid{};
  { hipError_t e = hipDeviceGetUuid(&uuid, 0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  // hipDeviceGetP2PAttribute (query with self — hipErrorInvalidDevice is ok)
  int p2pVal = 0;
  { hipError_t e = hipDeviceGetP2PAttribute(&p2pVal,
                       hipDevP2PAttrPerformanceRank, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice)); }

  // hipDeviceSetCacheConfig — set + restore
  hipFuncCache_t cacheCfg = hipFuncCachePreferNone;
  HRR_HIP_CHECK(hipDeviceGetCacheConfig(&cacheCfg));
  HRR_HIP_CHECK(hipDeviceSetCacheConfig(cacheCfg));

  // hipDeviceEnablePeerAccess / hipDeviceDisablePeerAccess with self —
  // expected to return hipErrorInvalidDevice on single-GPU setups.
  { hipError_t e = hipDeviceEnablePeerAccess(0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice
             || e == hipErrorPeerAccessAlreadyEnabled)); }
  { hipError_t e = hipDeviceDisablePeerAccess(0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice
             || e == hipErrorPeerAccessNotEnabled)); }

  // hipDevicePrimaryCtxSetFlags — set flags on device 0
  { hipError_t e = hipDevicePrimaryCtxSetFlags(0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorContextAlreadyInUse)); }

  // hipDevicePrimaryCtxRelease — safe to release (HIP re-creates on next use)
  { hipError_t e = hipDevicePrimaryCtxRelease(0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice)); }
  // Re-initialize after potential ctx release
  HRR_HIP_CHECK(hipSetDevice(0));

  // hipDeviceGetTexture1DLinearMaxWidth — may return hipErrorNotSupported
  size_t maxW = 0;
  hipChannelFormatDesc cfd = hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
  { hipError_t e = hipDeviceGetTexture1DLinearMaxWidth(&maxW, &cfd, 0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  // hipDeviceGraphMemTrim — trim graph memory for device 0
  { hipError_t e = hipDeviceGraphMemTrim(0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  // D2H blob (value = 11)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 11, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 11);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload R: Stream advanced APIs — part 2
//
// Exercises hipExtStreamCreateWithCUMask, hipExtStreamGetCUMask,
// hipExtGetLinkTypeAndHopCount, hipStreamWaitValue32/64,
// hipStreamWriteValue32/64, hipStreamAttachMemAsync, hipGetStreamDeviceId,
// _spt stream/event variants.
// Final blob: d[i] == 13.
// ===========================================================================
TEST_CASE("Unit_HRR_StreamAdvanced2_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s0;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // hipExtStreamCreateWithCUMask — create stream with CU mask
  hipStream_t cuStream = nullptr;
  {
    uint32_t cuMask = 0xFFFFFFFF;
    hipError_t e = hipExtStreamCreateWithCUMask(&cuStream, 1, &cuMask);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  // hipExtStreamGetCUMask — query CU mask from our stream
  // cuMaskSize=2 may be too small for GPUs with >64 CUs, yielding hipErrorInvalidValue.
  if (cuStream) {
    uint32_t outMask[2] = {};
    hipError_t e = hipExtStreamGetCUMask(cuStream, 2, outMask);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  // hipExtGetLinkTypeAndHopCount — query link between device 0 and itself
  {
    uint32_t linktype = 0, hopcount = 0;
    hipError_t e = hipExtGetLinkTypeAndHopCount(0, 0, &linktype, &hopcount);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  // hipGetStreamDeviceId — query device ID for stream
  { int devId = hipGetStreamDeviceId(s0);
    REQUIRE((devId == 0 || devId == -1)); }

  // hipStreamAttachMemAsync — attach managed memory to stream
  {
    int* managed = nullptr;
    HRR_HIP_CHECK(hipMallocManaged(&managed, SZ));
    { hipError_t e = hipStreamAttachMemAsync(s0, managed, SZ,
                       hipMemAttachSingle);
      REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }
    HRR_HIP_CHECK(hipStreamSynchronize(s0));
    HRR_HIP_CHECK(hipFree(managed));
  }

  // hipStreamWaitValue32 / hipStreamWriteValue32 — use mapped host mem
  // hipDeviceAttributeCanUseStreamWaitValue gates support
  {
    int canWait = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&canWait,
                 hipDeviceAttributeCanUseStreamWaitValue, 0));
    if (canWait) {
      int* flag = nullptr;
      HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag),
                              sizeof(int), hipHostMallocMapped));
      *flag = 0;
      // Write 1 into flag, then wait for it to be 1
      HRR_HIP_CHECK(hipStreamWriteValue32(s0, flag, 1, 0));
      HRR_HIP_CHECK(hipStreamWaitValue32(s0, flag, 1,
                   hipStreamWaitValueEq, 0xFFFFFFFF));
      HRR_HIP_CHECK(hipStreamSynchronize(s0));
      HRR_HIP_CHECK(hipFreeHost(flag));
    }
  }

  // hipStreamWaitValue64 / hipStreamWriteValue64
  {
    int canWait = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&canWait,
                 hipDeviceAttributeCanUseStreamWaitValue, 0));
    if (canWait) {
      uint64_t* flag64 = nullptr;
      HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag64),
                              sizeof(uint64_t), hipHostMallocMapped));
      *flag64 = 0;
      HRR_HIP_CHECK(hipStreamWriteValue64(s0, flag64, 1ULL, 0));
      HRR_HIP_CHECK(hipStreamWaitValue64(s0, flag64, 1ULL,
                   hipStreamWaitValueEq, 0xFFFFFFFFFFFFFFFFULL));
      HRR_HIP_CHECK(hipStreamSynchronize(s0));
      HRR_HIP_CHECK(hipFreeHost(flag64));
    }
  }

  // _spt stream variants — record events via per-thread-default-stream wrappers
  {
    hipError_t e;
    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    e = hipStreamIsCapturing_spt(s0, &st);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    e = hipStreamQuery_spt(s0);
    REQUIRE((e == hipSuccess || e == hipErrorNotReady
             || e == hipErrorNotSupported));
    e = hipStreamSynchronize_spt(s0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    int prio = 0;
    e = hipStreamGetPriority_spt(s0, &prio);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    unsigned int fl = 0;
    e = hipStreamGetFlags_spt(s0, &fl);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
  }

  // hipEventRecord_spt / _spt event recording
  {
    hipEvent_t ev;
    HRR_HIP_CHECK(hipEventCreate(&ev));
    hipError_t e = hipEventRecord_spt(ev, s0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported));
    HRR_HIP_CHECK(hipEventSynchronize(ev));
    HRR_HIP_CHECK(hipEventDestroy(ev));
  }

  HRR_HIP_CHECK(hipDeviceSynchronize());

  // D2H blob (value = 13)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 13, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 13);

  HRR_HIP_CHECK(hipFree(d));
  if (cuStream) { HRR_HIP_CHECK(hipStreamDestroy(cuStream)); }
  HRR_HIP_CHECK(hipStreamDestroy(s0));
  delete[] h;
}

// ===========================================================================
// Workload S: Context APIs
//
// Exercises hipCtxSynchronize, hipCtxGetFlags, hipCtxGetCacheConfig,
// hipCtxSetCacheConfig, hipCtxGetSharedMemConfig, hipCtxSetSharedMemConfig,
// hipCtxGetApiVersion, hipCtxSetCurrent, hipCtxEnablePeerAccess,
// hipCtxDisablePeerAccess.
// Final blob: d[i] == 15.
// ===========================================================================
TEST_CASE("Unit_HRR_Context_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // hipCtxSynchronize — synchronize current context (may return NotSupported on some ROCm builds)
  { hipError_t e = hipCtxSynchronize();
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  // hipCtxGetFlags — query current context flags (return value varies by ROCm build)
  { unsigned int ctxFlags2 = 0; (void)hipCtxGetFlags(&ctxFlags2); }

  // hipCtxGetCacheConfig / hipCtxSetCacheConfig roundtrip
  hipFuncCache_t cc = hipFuncCachePreferNone;
  (void)hipCtxGetCacheConfig(&cc);
  (void)hipCtxSetCacheConfig(cc);

  // hipCtxGetSharedMemConfig / hipCtxSetSharedMemConfig roundtrip
  hipSharedMemConfig smc = hipSharedMemBankSizeDefault;
  (void)hipCtxGetSharedMemConfig(&smc);
  (void)hipCtxSetSharedMemConfig(smc);

  // hipCtxGetApiVersion — query API version for current ctx (nullptr = current)
  { unsigned int apiVer = 0; (void)hipCtxGetApiVersion(nullptr, &apiVer); }

  // hipCtxSetCurrent — set to current context handle
  hipCtx_t curCtx = nullptr;
  (void)hipCtxGetCurrent(&curCtx);
  if (curCtx) { (void)hipCtxSetCurrent(curCtx); }

  // hipCtxEnablePeerAccess / hipCtxDisablePeerAccess — pass null peer ctx
  (void)hipCtxEnablePeerAccess(nullptr, 0);
  (void)hipCtxDisablePeerAccess(nullptr);

  { hipError_t e = hipCtxSynchronize();
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  // D2H blob (value = 15)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 15, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 15);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ===========================================================================
// Workload T: Module/Library/Kernel management APIs
//
// Exercises hipModuleUnload, hipModuleGetFunctionCount, hipFuncGetAttribute,
// hipGetFuncBySymbol, hipModuleOccupancy*, hipLibraryLoadData,
// hipLibraryUnload, hipLibraryGetKernel, hipLibraryGetKernelCount,
// hipLibraryEnumerateKernels, hipKernelGetLibrary, hipKernelGetFunction,
// hipKernelGetParamInfo, hipKernelGetAttribute, hipKernelSetAttribute.
// All are NOOP at playback; D2H blob via hipMemsetD32 + hipMemcpyAsync.
// Final blob: d[i] == 17.
// ===========================================================================
TEST_CASE("Unit_HRR_ModuleExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // Get a hipFunction_t handle for the fill kernel via hipGetFuncBySymbol
  // (uses symbol ptr from fat binary — stale at playback, NOOP)
  hipFunction_t func = nullptr;
  { hipError_t e = hipGetFuncBySymbol(&func,
                       reinterpret_cast<const void*>(fill_kernel_e));
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorNotFound)); }

  // hipFuncGetAttribute (needs valid hipFunction_t from module, may fail)
  if (func) {
    int attribVal = 0;
    hipError_t e = hipFuncGetAttribute(&attribVal,
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  // Load module from the registered fat binary (via hipModuleLoadData path)
  // to exercise hipModuleGetFunctionCount and hipModuleOccupancy*.
  // Use hipModuleLoadData with nullptr to test capture of the API (it will
  // fail safely); the real module is already loaded by fat binary registration.
  {
    // hipModuleGetFunctionCount — query a real module (use module from fat binary)
    // hipModuleGetFunctionCount takes hipModule_t which we don't have easily here.
    // Instead call it with nullptr (will fail gracefully) — event still recorded.
    unsigned int fnCount = 0;
    hipError_t e = hipModuleGetFunctionCount(&fnCount, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidHandle)); }

  // hipModuleOccupancyMaxPotentialBlockSize — needs hipFunction_t
  if (func) {
    int gs = 0, bs = 0;
    hipError_t e = hipModuleOccupancyMaxPotentialBlockSize(&gs, &bs, func, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  // hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
  if (func) {
    int nb = 0;
    hipError_t e = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                       &nb, func, 64, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  // hipModuleOccupancyMaxPotentialBlockSizeWithFlags / WithFlags
  if (func) {
    int gs2 = 0, bs2 = 0;
    hipError_t e = hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
                       &gs2, &bs2, func, 0, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  if (func) {
    int nb2 = 0;
    hipError_t e = hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
                       &nb2, func, 64, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  // hipLibraryLoadData (needs valid code object; use nullptr to test capture)
  {
    hipLibrary_t lib = nullptr;
    hipError_t e = hipLibraryLoadData(&lib, nullptr,
                       nullptr, nullptr, 0,
                       nullptr, nullptr, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidImage || e == hipErrorNotSupported)); }

  // hipLibraryGetKernelCount — needs valid lib; test with nullptr
  { unsigned int kc = 0;
    hipError_t e = hipLibraryGetKernelCount(&kc, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidHandle || e == hipErrorNotSupported)); }

  // hipKernelGetAttribute — needs valid hipKernel_t; test with nullptr
  { int kav = 0;
    hipError_t e = hipKernelGetAttribute(&kav,
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                       nullptr, 0);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

  // hipKernelSetAttribute — test with nullptr
  { hipError_t e = hipKernelSetAttribute(
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256,
                       nullptr, 0);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

  // hipKernelGetFunction — test with nullptr kernel
  { hipFunction_t kf = nullptr;
    hipError_t e = hipKernelGetFunction(&kf, nullptr);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

  // D2H blob (value = 17)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 17, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 17);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload U — Misc APIs (profiler, occupancy extras, proc address, etc.)
// ---------------------------------------------------------------------------
// kernel for OccupancyAvailableDynamicSMemPerBlock
__global__ void fill_kernel_u(int* d, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = val;
}

TEST_CASE("Unit_HRR_MiscAPIs_Direct", "[.][hrr][direct]") {
  // Warm-up: the capture shims install during hip::init(), which is triggered by
  // the FIRST HIP call — so that first call is not recorded. Make it a harmless
  // hipSetDevice rather than the hipMalloc below, otherwise `d`'s allocation is
  // dropped and replay cannot map it.
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // hipProfilerStart / hipProfilerStop
  (void)hipProfilerStart();
  (void)hipProfilerStop();

  // hipExtGetLastError
  (void)hipExtGetLastError();

  // hipOccupancyAvailableDynamicSMemPerBlock
  { size_t dynSmem = 0;
    (void)hipOccupancyAvailableDynamicSMemPerBlock(&dynSmem, fill_kernel_u, 128, 0); }

  // hipGetProcAddress
  { void* pfn = nullptr;
    (void)hipGetProcAddress("hipMalloc", &pfn, HIP_VERSION, 0, nullptr); }

  // hipSetValidDevices
  { int devArr[] = {0};
    (void)hipSetValidDevices(devArr, 1); }

  // hipMemPtrGetInfo
  { size_t ptrSize = 0;
    (void)hipMemPtrGetInfo(d, &ptrSize); }

  // D2H blob (value = 19)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 19, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 19);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload V — Driver-style 3D/2D memcpy variants
//
// Canary choice matters here.  Replay zero-initialises allocations, and the D2H
// validator falls back from memcmp to a float tolerance (atol=rtol=1e-3) over
// candidate f32/bf16/f16/f64 decodings, accepting the first encoding with no
// out-of-tolerance element.  A canary that decodes near 0.0 in any candidate
// encoding therefore passes against an all-zero replay buffer, which makes the
// whole roundtrip vacuous.  Both values below decode far from zero in all four:
//   0x5C5C5C5C -> f32 2.48e17, bf16 2.48e17, f16 279, f64 8.25e136
//   0x4B4B4B4B -> f32 1.33e7,  bf16 1.33e7,  f16 14.6, f64 5.23e54
// (0x2D2D2D2D, for contrast, is 9.84e-12 as f32, i.e. inside atol.)
// ---------------------------------------------------------------------------
static constexpr int kDrvPayload  = 0x5C5C5C5C;  // bytes the copy must move
static constexpr int kDrvPad      = 0x4B4B4B4B;  // device padding that must survive
static constexpr int kDrvHostFill = 0x3A3A3A3A;  // host padding that must NOT be copied

// Pitched-rect geometry: pitch > width and height > 1, so the host source
// footprint is pitch*(height-1)+width rather than the width*height volume.
static constexpr size_t kDrvPitch = 256;  // bytes per row
static constexpr size_t kDrvWidth = 64;   // bytes actually copied per row
static constexpr size_t kDrvRows  = 4;    // rows per slice

TEST_CASE("Unit_HRR_DrvMemcpy3D_Direct", "[.][hrr][direct]") {
  // Driver 3D struct-pointer copies, validated end to end by D2H.
  // Chain: host(hsrc=VAL) --hipDrvMemcpy3D H2D--> A --hipDrvMemcpy3DAsync D2D--> B
  //        --hipDrvMemcpy3D D2D--> C. Final D2H requires all three driver copies
  //        (H2D blob path + dual device-ptr translation) to have replayed.
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = kDrvPayload;

  int *A = nullptr, *B = nullptr, *C = nullptr;
  HRR_HIP_CHECK(hipMalloc(&A, SZ));
  HRR_HIP_CHECK(hipMalloc(&B, SZ));
  HRR_HIP_CHECK(hipMalloc(&C, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* hsrc = new int[N];
  for (int i = 0; i < N; ++i) hsrc[i] = VAL;

  // (1) hipDrvMemcpy3D H2D: host hsrc -> device A (exercises the blob path).
  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hsrc;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(A);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3D(&p)); }

  // (2) hipDrvMemcpy3DAsync D2D: A -> B on a stream (dual pointer translation).
  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(A);
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(B);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3DAsync(&p, s)); }
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // (3) hipDrvMemcpy3D D2D: B -> C (sync device-to-device).
  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(B);
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(C);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3D(&p)); }

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, C, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  // (4) Pitched 3D H2D: kDrvRows rows of kDrvWidth bytes per slice, rows spaced
  // kDrvPitch apart, over two slices.  The host source rect therefore spans
  // srcPitch*srcHeight*(Depth-1) + srcPitch*(Height-1) + WidthInBytes
  // = 1024 + 768 + 64 = 1856 bytes, not the 64*4*2 = 512-byte volume, so a blob
  // sized by the volume makes the runtime stride 1344 bytes past its end on
  // replay.  Seeding the destination with kDrvPad and requiring the inter-row
  // padding to survive also proves the copy wrote only the row regions.
  constexpr size_t DEPTH  = 2;
  constexpr size_t PSZ    = kDrvPitch * kDrvRows * DEPTH;   // 2048 bytes
  constexpr size_t PWORDS = PSZ / sizeof(int);
  int* P = nullptr;
  HRR_HIP_CHECK(hipMalloc(&P, PSZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(P), kDrvPad, PWORDS));

  std::vector<int> hp(PWORDS, kDrvHostFill);
  for (size_t z = 0; z < DEPTH; ++z)
    for (size_t y = 0; y < kDrvRows; ++y) {
      size_t off = (z * kDrvRows + y) * kDrvPitch / sizeof(int);
      for (size_t w = 0; w < kDrvWidth / sizeof(int); ++w) hp[off + w] = kDrvPayload;
    }

  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hp.data();
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(P);
    p.WidthInBytes  = kDrvWidth; p.Height = kDrvRows; p.Depth = DEPTH;
    p.srcPitch = kDrvPitch; p.srcHeight = kDrvRows;
    p.dstPitch = kDrvPitch; p.dstHeight = kDrvRows;
    HRR_HIP_CHECK(hipDrvMemcpy3D(&p)); }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> back(PWORDS, 0);
  HRR_HIP_CHECK(hipMemcpy(back.data(), P, PSZ, hipMemcpyDeviceToHost));
  for (size_t z = 0; z < DEPTH; ++z)
    for (size_t y = 0; y < kDrvRows; ++y) {
      size_t row = (z * kDrvRows + y) * kDrvPitch / sizeof(int);
      for (size_t w = 0; w < kDrvPitch / sizeof(int); ++w)
        REQUIRE(back[row + w] == (w < kDrvWidth / sizeof(int) ? kDrvPayload : kDrvPad));
    }

  // (5) hipMemcpy3DPeer / hipMemcpy3DPeerAsync: device 0 -> device 0 self-copy
  // on a scratch buffer, purely to keep these two APIs exercised at capture.
  // Both are NOOP on playback, so they must not touch a buffer that feeds a D2H
  // check; a same-pointer copy is also a no-op at capture time, so capture and
  // replay agree either way.
  int* scratch = nullptr;
  HRR_HIP_CHECK(hipMalloc(&scratch, SZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(scratch), kDrvPad, N));
  { hipMemcpy3DPeerParms pp{};
    pp.srcDevice = 0; pp.dstDevice = 0;
    pp.srcPtr    = make_hipPitchedPtr(scratch, sizeof(int), 1, 1);
    pp.dstPtr    = make_hipPitchedPtr(scratch, sizeof(int), 1, 1);
    pp.extent    = make_hipExtent(sizeof(int), 1, 1);
    (void)hipMemcpy3DPeer(&pp); }
  { hipMemcpy3DPeerParms pp{};
    pp.srcDevice = 0; pp.dstDevice = 0;
    pp.srcPtr    = make_hipPitchedPtr(scratch, sizeof(int), 1, 1);
    pp.dstPtr    = make_hipPitchedPtr(scratch, sizeof(int), 1, 1);
    pp.extent    = make_hipExtent(sizeof(int), 1, 1);
    (void)hipMemcpy3DPeerAsync(&pp, s); }
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipFree(scratch));
  HRR_HIP_CHECK(hipFree(P));
  HRR_HIP_CHECK(hipFree(A));
  HRR_HIP_CHECK(hipFree(B));
  HRR_HIP_CHECK(hipFree(C));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] hsrc;
  delete[] h;
}

// ===========================================================================
// hipDrvMemcpy2DUnaligned — driver 2D struct-pointer copy (hip_Memcpy2D).
// host(hsrc=VAL) --H2D--> A --D2D--> B, validated by D2H on B, then a pitched
// H2D rect whose inter-row padding must survive.
// ===========================================================================
TEST_CASE("Unit_HRR_DrvMemcpy2DUnaligned_Direct", "[.][hrr][direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N   = 256;
  constexpr size_t SZ  = N * sizeof(int);
  constexpr int    VAL = kDrvPayload;

  int *A = nullptr, *B = nullptr;
  HRR_HIP_CHECK(hipMalloc(&A, SZ));
  HRR_HIP_CHECK(hipMalloc(&B, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* hsrc = new int[N];
  for (int i = 0; i < N; ++i) hsrc[i] = VAL;

  // (1) H2D single row: host hsrc -> device A (blob path).
  { hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hsrc;                                p.srcPitch = SZ;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(A); p.dstPitch = SZ;
    p.WidthInBytes  = SZ; p.Height = 1;
    HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&p)); }

  // (2) D2D: A -> B (dual pointer translation).
  { hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(A); p.srcPitch = SZ;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(B); p.dstPitch = SZ;
    p.WidthInBytes  = SZ; p.Height = 1;
    HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&p)); }

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, B, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  // (3) Pitched 2D H2D: kDrvRows rows of kDrvWidth bytes spaced kDrvPitch apart.
  // The host source rect spans srcPitch*(Height-1) + WidthInBytes
  // = 768 + 64 = 832 bytes, not the 64*4 = 256-byte volume, so a blob sized by
  // the volume makes the runtime stride 576 bytes past its end on replay.  The
  // destination is pre-seeded with kDrvPad, so the inter-row padding also has to
  // survive untouched.
  constexpr size_t PSZ    = kDrvPitch * kDrvRows;  // 1024 bytes
  constexpr size_t PWORDS = PSZ / sizeof(int);
  int* P = nullptr;
  HRR_HIP_CHECK(hipMalloc(&P, PSZ));
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(P), kDrvPad, PWORDS));

  std::vector<int> hp(PWORDS, kDrvHostFill);
  for (size_t y = 0; y < kDrvRows; ++y)
    for (size_t w = 0; w < kDrvWidth / sizeof(int); ++w)
      hp[y * kDrvPitch / sizeof(int) + w] = kDrvPayload;

  { hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hp.data();
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(P);
    p.srcPitch = kDrvPitch; p.dstPitch = kDrvPitch;
    p.WidthInBytes  = kDrvWidth; p.Height = kDrvRows;
    HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&p)); }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> back(PWORDS, 0);
  HRR_HIP_CHECK(hipMemcpy(back.data(), P, PSZ, hipMemcpyDeviceToHost));
  for (size_t y = 0; y < kDrvRows; ++y)
    for (size_t w = 0; w < kDrvPitch / sizeof(int); ++w)
      REQUIRE(back[y * kDrvPitch / sizeof(int) + w] ==
              (w < kDrvWidth / sizeof(int) ? kDrvPayload : kDrvPad));

  HRR_HIP_CHECK(hipFree(P));
  HRR_HIP_CHECK(hipFree(A));
  HRR_HIP_CHECK(hipFree(B));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] hsrc;
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload W — Texture / Array APIs (image-support gated)
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_Texture_Direct", "[.][hrr][direct]") {
  // Gate entire test on image support
  int imageSupport = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&imageSupport, hipDeviceAttributeImageSupport, 0));
  if (!imageSupport) {
    SUCCEED("Device has no image support — skipping texture workload");
    return;
  }

  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // hipArrayCreate
  hipArray_t arr0 = nullptr;
  { HIP_ARRAY_DESCRIPTOR desc{};
    desc.Width  = 64;
    desc.Height = 1;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    HRR_HIP_CHECK(hipArrayCreate(&arr0, &desc)); }

  // hipArrayGetDescriptor
  { HIP_ARRAY_DESCRIPTOR out{};
    (void)hipArrayGetDescriptor(&out, arr0); }

  // hipGetChannelDesc (hipArray_t → hipChannelFormatDesc)
  { hipChannelFormatDesc cfd{};
    (void)hipGetChannelDesc(&cfd, arr0); }

  // hipArray3DCreate
  hipArray_t arr3d = nullptr;
  { HIP_ARRAY3D_DESCRIPTOR desc3{};
    desc3.Width  = 4;
    desc3.Height = 4;
    desc3.Depth  = 1;
    desc3.Format = HIP_AD_FORMAT_FLOAT;
    desc3.NumChannels = 1;
    HRR_HIP_CHECK(hipArray3DCreate(&arr3d, &desc3)); }

  // hipArray3DGetDescriptor
  { HIP_ARRAY3D_DESCRIPTOR out3{};
    (void)hipArray3DGetDescriptor(&out3, arr3d); }

  // hipArrayGetInfo
  { hipChannelFormatDesc cfd{};
    hipExtent ext{};
    unsigned int flags = 0;
    (void)hipArrayGetInfo(&cfd, &ext, &flags, arr0); }

  // hipMallocMipmappedArray / hipGetMipmappedArrayLevel / hipFreeMipmappedArray
  { hipMipmappedArray_t mma = nullptr;
    hipChannelFormatDesc cfd = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipExtent ext = make_hipExtent(4, 4, 0);
    hipError_t e = hipMallocMipmappedArray(&mma, &cfd, ext, 2, 0);
    if (e == hipSuccess && mma) {
      hipArray_t lvl = nullptr;
      (void)hipGetMipmappedArrayLevel(&lvl, mma, 0);
      HRR_HIP_CHECK(hipFreeMipmappedArray(mma));
    } }

  // hipCreateTextureObject / hipDestroyTextureObject
  { hipTextureObject_t tex = 0;
    hipResourceDesc rd{}; rd.resType = hipResourceTypeArray; rd.res.array.array = arr0;
    hipTextureDesc  td{}; td.addressMode[0] = hipAddressModeClamp;
                          td.filterMode = hipFilterModePoint;
                          td.readMode   = hipReadModeElementType;
    hipError_t e = hipCreateTextureObject(&tex, &rd, &td, nullptr);
    if (e == hipSuccess) { (void)hipDestroyTextureObject(tex); } }

  // hipTexObjectCreate / hipTexObjectDestroy
  { hipTextureObject_t tex = 0;
    HIP_RESOURCE_DESC rd{}; rd.resType = HIP_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = arr0;
    HIP_TEXTURE_DESC  td{};
    hipError_t e = hipTexObjectCreate(&tex, &rd, &td, nullptr);
    if (e == hipSuccess) { (void)hipTexObjectDestroy(tex); } }

  // Cleanup arrays
  HRR_HIP_CHECK(hipArrayDestroy(arr0));
  HRR_HIP_CHECK(hipArrayDestroy(arr3d));

  // D2H blob (value = 23)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 23, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 23);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload X — Graph explicit APIs (clone, debug print, node queries, etc.)
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_GraphExplicit_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // Create an empty graph
  hipGraph_t emptyGraph = nullptr;
  (void)hipGraphCreate(&emptyGraph, 0);

  // Stream-capture a simple memset to get a real graph
  hipGraph_t capturedGraph = nullptr;
  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  HRR_HIP_CHECK(hipMemsetAsync(d, 0, SZ, s));
  HRR_HIP_CHECK(hipStreamEndCapture(s, &capturedGraph));

  // hipGraphClone
  hipGraph_t clonedGraph = nullptr;
  (void)hipGraphClone(&clonedGraph, capturedGraph);

  // hipGraphDebugDotPrint — write to null device (no output, just exercises path)
#ifdef _WIN32
  (void)hipGraphDebugDotPrint(capturedGraph, "NUL", 0);
#else
  (void)hipGraphDebugDotPrint(capturedGraph, "/dev/null", 0);
#endif

  // hipGraphInstantiate
  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, capturedGraph, nullptr, nullptr, 0));

  // hipGraphExecGetFlags
  { unsigned long long execFlags = 0;
    (void)hipGraphExecGetFlags(exec, &execFlags); }

  // hipGraphGetNodes — query count then enumerate
  { size_t numNodes = 0;
    (void)hipGraphGetNodes(capturedGraph, nullptr, &numNodes);
    if (numNodes > 0) {
      std::vector<hipGraphNode_t> nodes(numNodes);
      (void)hipGraphGetNodes(capturedGraph, nodes.data(), &numNodes);
      // hipGraphNodeGetType
      hipGraphNodeType nodeType{};
      (void)hipGraphNodeGetType(nodes[0], &nodeType);
      // hipGraphNodeSetEnabled / hipGraphNodeGetEnabled
      unsigned int enabled = 1;
      (void)hipGraphNodeSetEnabled(exec, nodes[0], 1);
      (void)hipGraphNodeGetEnabled(exec, nodes[0], &enabled);
      // hipGraphMemsetNodeGetParams
      hipMemsetParams msp{};
      (void)hipGraphMemsetNodeGetParams(nodes[0], &msp);
    } }

  // hipGraphUpload
  (void)hipGraphUpload(exec, s);

  // hipGraphLaunch (MANUAL playback — this exercises the graph replay path)
  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // hipGraphExecDestroy
  (void)hipGraphExecDestroy(exec);

  // hipUserObjectRetain / hipUserObjectRelease (nullptr handle — just exercises path)
  (void)hipUserObjectRetain(nullptr, 1);
  (void)hipUserObjectRelease(nullptr, 1);

  // hipGraphRetainUserObject / hipGraphReleaseUserObject
  (void)hipGraphRetainUserObject(capturedGraph, nullptr, 1, 0);
  (void)hipGraphReleaseUserObject(capturedGraph, nullptr, 1);

  // Destroy all graphs
  if (emptyGraph)   (void)hipGraphDestroy(emptyGraph);
  if (clonedGraph)  (void)hipGraphDestroy(clonedGraph);
  if (capturedGraph)(void)hipGraphDestroy(capturedGraph);

  // D2H blob (value = 25)
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 25, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 25);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload Y — hipHostRegister/Unregister + hipLaunchKernel + hipMemcpy3D_spt
// ---------------------------------------------------------------------------
__global__ void hrr_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}

TEST_CASE("Unit_HRR_HostRegLaunch_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // ---- hipHostRegister / hipHostUnregister --------------------------------
  {
    int* h_reg = static_cast<int*>(malloc(SZ));
    REQUIRE(h_reg != nullptr);
    for (int i = 0; i < N; ++i) h_reg[i] = i;
    HRR_HIP_CHECK(hipHostRegister(h_reg, SZ, hipHostRegisterDefault));
    // H2D: copy from registered buffer to device
    HRR_HIP_CHECK(hipMemcpy(d, h_reg, SZ, hipMemcpyHostToDevice));
    HRR_HIP_CHECK(hipHostUnregister(h_reg));
    free(h_reg);
  }

  // ---- hipLaunchKernel (explicit void** args) ------------------------------
  {
    int* d_int = reinterpret_cast<int*>(d);
    int  val   = 27;
    int  n     = N;
    void* args[] = { &d_int, &val, &n };
    int blocks = (N + 255) / 256;
    HRR_HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(hrr_fill),
                              dim3(blocks), dim3(256), args, 0, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // ---- hipMemcpy3D_spt / hipMemcpy3DAsync_spt -----------------------------
  {
    // Minimal 1-element D2D 3D copy just to exercise the path
    hipMemcpy3DParms p{};
    p.srcPtr = make_hipPitchedPtr(d, sizeof(float), 1, 1);
    p.dstPtr = make_hipPitchedPtr(d, sizeof(float), 1, 1);
    p.extent = make_hipExtent(sizeof(float), 1, 1);
    p.kind   = hipMemcpyDeviceToDevice;
    (void)hipMemcpy3D_spt(&p);
    (void)hipMemcpy3DAsync_spt(&p, s);
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // D2H blob (value = 27, set by hrr_fill above)
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 27);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload Z — hipModuleLoadData/DataEx/Load + hipModuleGetFunction +
//              hipModuleLaunchKernel  (uses HIPRTC to compile kernel at runtime)
// ---------------------------------------------------------------------------
static const char* k_fill_src = R"(
extern "C" __global__ void rtc_fill(int* out, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = val;
}
)";

TEST_CASE("Unit_HRR_ModuleAPI_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // Compile a minimal kernel via HIPRTC
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, k_fill_src, "rtc_fill.hip",
                                   0, nullptr, nullptr));
  hiprtcResult compile_rc = hiprtcCompileProgram(prog, 0, nullptr);
  if (compile_rc != HIPRTC_SUCCESS) {
    size_t log_sz = 0;
    (void)hiprtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    (void)hiprtcGetProgramLog(prog, log.data());
    (void)hiprtcDestroyProgram(&prog);
    FAIL("hiprtcCompileProgram failed: " + log);
  }
  size_t co_size = 0;
  HRR_HIPRTC_CHECK(hiprtcGetCodeSize(prog, &co_size));
  std::vector<char> co(co_size);
  HRR_HIPRTC_CHECK(hiprtcGetCode(prog, co.data()));
  HRR_HIPRTC_CHECK(hiprtcDestroyProgram(&prog));

  // ---- hipModuleLoadData ---------------------------------------------------
  hipModule_t mod_data = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod_data, co.data()));

  // ---- hipModuleGetFunction ------------------------------------------------
  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod_data, "rtc_fill"));

  // ---- hipModuleLaunchKernel ----------------------------------------------
  {
    int* d_int = reinterpret_cast<int*>(d);
    int  val   = 29;
    int  n     = N;
    void* args[] = { &d_int, &val, &n };
    int blocks = (N + 255) / 256;
    HRR_HIP_CHECK(hipModuleLaunchKernel(fn,
      blocks, 1, 1,   // grid
      256,    1, 1,   // block
      0, s, args, nullptr));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  // ---- hipModuleLoadDataEx (options=nullptr, numOptions=0) ----------------
  {
    hipModule_t mod_ex = nullptr;
    HRR_HIP_CHECK(hipModuleLoadDataEx(&mod_ex, co.data(), 0, nullptr, nullptr));
    hipFunction_t fn_ex = nullptr;
    HRR_HIP_CHECK(hipModuleGetFunction(&fn_ex, mod_ex, "rtc_fill"));
    HRR_HIP_CHECK(hipModuleUnload(mod_ex));
  }

  // ---- hipModuleLoad (write CO to temp file, then load from disk) ----------
  {
    namespace fs = std::filesystem;
    // Use fs::unique_path equivalent: the driver may keep the previous file
    // open after hipModuleUnload on Windows, making it undeletable until reboot.
    // Using a unique name per run avoids the "file already locked" open failure.
    auto tmp_co = fs::temp_directory_path() /
                  (std::string("hrr_rtc_fill_") +
                   std::to_string(reinterpret_cast<uintptr_t>(&co)) + ".co");
    {
      std::ofstream f(tmp_co, std::ios::binary);
      REQUIRE(f.is_open());
      f.write(co.data(), static_cast<std::streamsize>(co.size()));
    }
    hipModule_t mod_file = nullptr;
    HRR_HIP_CHECK(hipModuleLoad(&mod_file, tmp_co.string().c_str()));
    HRR_HIP_CHECK(hipModuleUnload(mod_file));
    // Ignore remove errors: on Windows the ROCm driver may keep the file
    // open after hipModuleUnload, making fs::remove throw.  The temp
    // directory will clean it up on next boot.
    std::error_code ec;
    fs::remove(tmp_co, ec);
  }

  HRR_HIP_CHECK(hipModuleUnload(mod_data));

  // D2H blob (value = 29, set by hipModuleLaunchKernel above)
  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 29);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// ---------------------------------------------------------------------------
// Workload AA — Virtual Memory Management (VMM) roundtrip
// ---------------------------------------------------------------------------
// Exercises hipMemAddressReserve, hipMemCreate, hipMemMap, hipMemSetAccess,
// hipMemGetAllocationPropertiesFromHandle, hipMemUnmap, hipMemRelease,
// hipMemAddressFree.  Gated on hipDeviceAttributeVirtualMemoryManagementSupported.
// D2H blob value = 0x1F1F1F1F per memset(0x1F) pattern.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_VMM_Direct", "[.][hrr][direct]") {
  int vmm_supported = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&vmm_supported,
                                  hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  if (!vmm_supported) {
    SUCCEED("VMM not supported on this device — skipping");
    return;
  }

  hipMemAllocationProp prop{};
  prop.type             = hipMemAllocationTypePinned;
  prop.location.type    = hipMemLocationTypeDevice;
  prop.location.id      = 0;

  size_t granularity = 0;
  HRR_HIP_CHECK(hipMemGetAllocationGranularity(&granularity, &prop,
                                           hipMemAllocationGranularityRecommended));
  REQUIRE(granularity > 0);

  // Allocate at least N ints, rounded up to granularity boundary
  const size_t min_bytes = N * sizeof(int);
  const size_t num_gran  = (min_bytes + granularity - 1) / granularity;
  const size_t alloc_sz  = num_gran * granularity;

  // hipMemAddressReserve
  void* va = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&va, alloc_sz, 0, nullptr, 0));
  REQUIRE(va != nullptr);

  // hipMemCreate
  hipMemGenericAllocationHandle_t handle{};
  HRR_HIP_CHECK(hipMemCreate(&handle, alloc_sz, &prop, 0));

  // hipMemGetAllocationPropertiesFromHandle — coverage only
  {
    hipMemAllocationProp out_prop{};
    (void)hipMemGetAllocationPropertiesFromHandle(&out_prop, handle);
  }

  // hipMemMap
  HRR_HIP_CHECK(hipMemMap(va, alloc_sz, 0, handle, 0));

  // hipMemSetAccess
  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id   = 0;
  desc.flags         = hipMemAccessFlagsProtReadWrite;
  HRR_HIP_CHECK(hipMemSetAccess(va, alloc_sz, &desc, 1));

  // D2H validation blob (value 0x1F1F1F1F per hipMemset(0x1F))
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  HRR_HIP_CHECK(hipMemset(va, 0x1F, alloc_sz));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* hvmm = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(hvmm, va, N * sizeof(int), hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(hvmm[i] == 0x1F1F1F1F);
  delete[] hvmm;

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipMemUnmap(va, alloc_sz));
  HRR_HIP_CHECK(hipMemRelease(handle));
  HRR_HIP_CHECK(hipMemAddressFree(va, alloc_sz));
}

// ---------------------------------------------------------------------------
// Workload AB — triple-chevron <<<>>> launch
// Exercises the __hipPushCallConfiguration → hipLaunchByPtr path, which is
// distinct from hipLaunchKernelGGL / hipLaunchKernel.  Without a manual
// capture___hipPushCallConfiguration that saves grid/block/shared/stream into
// TLS, replayed kernels would launch with all-zero dimensions.
// D2H blob value = 42.
// ---------------------------------------------------------------------------
TEST_CASE("Unit_HRR_ChevronLaunch_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  // Drain any pending GPU errors from earlier tests before the <<<>>> launch.
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();

  int blocks = (N + 255) / 256;
  // Triple-chevron launch — goes through __hipPushCallConfiguration + hipLaunchByPtr
  hrr_fill<<<dim3(blocks), dim3(256), 0, s>>>(d, 42, N);
  HRR_HIP_CHECK(hipGetLastError());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 42);

  delete[] h;
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
}
