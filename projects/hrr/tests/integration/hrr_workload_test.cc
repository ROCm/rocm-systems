/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Workload (direct GPU implementations)
 * @{
 * @ingroup HRRTest
 * Direct GPU tests, hidden from the default CTest run by the Catch2 [.] tag.
 * Subprocesses in hrr_roundtrip.cc and manual runs invoke them by name.
 */

#include "hrr_test_common.hh"
#include "hrr_clock_hook.hh"
#include <hip/hiprtc.h>
#include <hip/hip_ext.h>  // hipExtModuleLaunchKernel

#define HRR_HIPRTC_CHECK(expr)                                                 \
  do {                                                                         \
    hiprtcResult _hrr_rtc = (expr);                                            \
    INFO("hipRTC call failed: " #expr);                                        \
    INFO("hiprtcResult: " << static_cast<int>(_hrr_rtc));                      \
    INFO("hiprtcGetErrorString: " << hiprtcGetErrorString(_hrr_rtc));          \
    REQUIRE(_hrr_rtc == HIPRTC_SUCCESS);                                       \
  } while (0)
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#endif


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

  hipLaunchKernelGGL(hrr_vectorSaxpy, grid, block, 0, s0, 2.0f, da, dc, N);
  HRR_HIP_CHECK(hipGetLastError());

  for (int iter = 0; iter < KERNEL_ITERS; ++iter) {
    hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, s0, da, db, dc, N);
    HRR_HIP_CHECK(hipGetLastError());
  }

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipMemcpyAsync(dd, dc, SZ, hipMemcpyDeviceToDevice, s1));

  hipLaunchKernelGGL(hrr_dotPartial, grid, block, block.x * sizeof(float), s0,
                     da, db, dp, N);
  HRR_HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(hrr_vectorScale, grid, block, 0, s0, dp, dp,
                     1.0f / nblocks, N);
  HRR_HIP_CHECK(hipGetLastError());

  HRR_HIP_CHECK(hipStreamSynchronize(s1));
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


__global__ void hrr_incrementInt(int* buf) {
  *buf += 1;
}

__global__ void hrr_scalarAdd(const int* a, const int* b, int* c, int n, int scalar) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i] + scalar;
}

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

  REQUIRE(*buf == 2);

  int result = 0;
  HRR_HIP_CHECK(hipMemcpy(&result, buf_dev, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == 2);

  HRR_HIP_CHECK(hipFree(buf));
}

TEST_CASE("Unit_HRR_AllApis_Direct", "[.][hrr-direct]") {
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

  (void)hipGetLastError();     // drain any pending errors
  (void)hipPeekAtLastError();  // peek without clearing
  REQUIRE(hipGetErrorName(hipSuccess) != nullptr);
  REQUIRE(hipGetErrorString(hipSuccess) != nullptr);

  hipStream_t s0, s1, s2;
  HRR_HIP_CHECK(hipStreamCreate(&s0));
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithPriority(&s2, hipStreamNonBlocking, priLo));

  unsigned int sflags = 0;
  HRR_HIP_CHECK(hipStreamGetFlags(s1, &sflags));
  REQUIRE(sflags == hipStreamNonBlocking);

  int spri = 0;
  HRR_HIP_CHECK(hipStreamGetPriority(s2, &spri));

  {
    hipError_t q = hipStreamQuery(s0);
    REQUIRE((q == hipSuccess || q == hipErrorNotReady));
  }

  hipEvent_t ev_start, ev_stop, ev_nodur;
  HRR_HIP_CHECK(hipEventCreate(&ev_start));
  HRR_HIP_CHECK(hipEventCreate(&ev_stop));
  HRR_HIP_CHECK(hipEventCreateWithFlags(&ev_nodur, hipEventDisableTiming));

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
  // The reuse policies take an int, so capture records 4 bytes rather than 8.
  int32_t opportunistic = 1;
  HRR_HIP_CHECK(hipMemPoolSetAttribute(pool, hipMemPoolReuseAllowOpportunistic, &opportunistic));

  int *d_pool = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(&d_pool, SZ, pool, s0));

  int *h_pinned = nullptr;
  HRR_HIP_CHECK(hipHostMalloc(&h_pinned, SZ, 0));

  const bool managed_ok = (props.managedMemory != 0);
  int *d_managed = nullptr;
  if (managed_ok)
    HRR_HIP_CHECK(hipMallocManaged(&d_managed, SZ, hipMemAttachGlobal));

  HRR_HIP_CHECK(hipMemset(d0, 0, SZ));
  HRR_HIP_CHECK(hipMemsetAsync(d1, 0, SZ, s0));

  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = 42;

  HRR_HIP_CHECK(hipMemcpy(d0, h_src, SZ, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemcpyAsync(d1, h_src, SZ, hipMemcpyHostToDevice, s0));
  for (int i = 0; i < N; ++i) h_pinned[i] = 1;
  HRR_HIP_CHECK(hipMemcpyWithStream(d_async, h_pinned, SZ, hipMemcpyHostToDevice, s1));

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  hipPointerAttribute_t pattr{};
  HRR_HIP_CHECK(hipPointerGetAttributes(&pattr, d0));
  REQUIRE(pattr.type == hipMemoryTypeDevice);

  hipFuncCache_t cacheConf = hipFuncCachePreferNone;
  HRR_HIP_CHECK(hipDeviceGetCacheConfig(&cacheConf));

  if (deviceCount > 1) {
    int canAccess = 0;
    HRR_HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 0, 1));
  }

  int occBlockSize = 0, occGridSize = 0;
  HRR_HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&occGridSize, &occBlockSize,
                                              hrr_scalarAdd, 0, 0));
  REQUIRE(occBlockSize > 0);

  HRR_HIP_CHECK(hipEventRecord(ev_start, s0));

  dim3 block(256), grid((N + 255) / 256);
  hipLaunchKernelGGL(hrr_scalarAdd, grid, block, 0, s0, d0, d1, d2, N, 10);
  HRR_HIP_CHECK(hipGetLastError());

  HRR_HIP_CHECK(hipEventRecord(ev_stop, s0));
  HRR_HIP_CHECK(hipStreamWaitEvent(s1, ev_stop, 0));

  HRR_HIP_CHECK(hipEventSynchronize(ev_stop));
  { hipError_t q = hipEventQuery(ev_stop); REQUIRE(q == hipSuccess); }

  float ms = 0.f;
  HRR_HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
  REQUIRE(ms > -1.f);

  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d2, SZ, hipMemcpyDeviceToHost, s1));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  for (int i = 0; i < N; ++i)
    REQUIRE(h_out[i] == 94);

  HRR_HIP_CHECK(hipMemcpy(d1, d2, SZ, hipMemcpyDeviceToDevice));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  if (managed_ok && d_managed) {
    for (int i = 0; i < N; ++i) d_managed[i] = i;
    HRR_HIP_CHECK(hipMemAdvise(d_managed, SZ, hipMemAdviseSetReadMostly, 0));

    // hipMemRangeGetAttribute segfaults on some Linux ROCm builds, so the call is non-fatal.
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

  {
    int* d3 = nullptr;
    HRR_HIP_CHECK(hipMalloc(&d3, SZ));
    int* h3_src = new int[N];
    int* h3_out = new int[N]();
    for (int i = 0; i < N; ++i) h3_src[i] = 99;

    hipMemcpy3DParms p_h2d{};
    p_h2d.srcPtr = make_hipPitchedPtr(h3_src, N * sizeof(int), N, 1);
    p_h2d.dstPtr = make_hipPitchedPtr(d3,     N * sizeof(int), N, 1);
    p_h2d.extent = make_hipExtent(N * sizeof(int), 1, 1);
    p_h2d.kind   = hipMemcpyHostToDevice;
    HRR_HIP_CHECK(hipMemcpy3D(&p_h2d));

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

  {
    hipStreamAttrValue attr_val{};
    attr_val.syncPolicy = hipSyncPolicySpin;
    HRR_HIP_CHECK(hipStreamSetAttribute(s0, hipStreamAttributeSynchronizationPolicy,
                                    &attr_val));
  }

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

  {
    hipMemAccessDesc access_desc{};
    access_desc.location.type = hipMemLocationTypeDevice;
    access_desc.location.id   = 0;
    access_desc.flags         = hipMemAccessFlagsProtReadWrite;
    HRR_HIP_CHECK(hipMemPoolSetAccess(pool, &access_desc, 1));
  }

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

      // hipMemcpy3D and hipMemcpy3DAsync with an array on either end, going through device buffers.
      // The host readback of those buffers is the D2H blob replay validates.
      constexpr size_t W = 16, H = 4, D = 2;  // elements; one int each
      constexpr size_t ROW = W * sizeof(int);
      constexpr size_t BYTES3 = ROW * H * D;
      constexpr size_t BYTES2 = ROW * H;

      std::vector<int> h3(W * H * D), h2(W * H);
      for (size_t i = 0; i < h3.size(); ++i) h3[i] = static_cast<int>(i * 2654435761u) ^ 0x3d3d3d3d;
      for (size_t i = 0; i < h2.size(); ++i) h2[i] = static_cast<int>(i * 2654435761u) ^ 0x2d2d2d2d;

      HIP_ARRAY3D_DESCRIPTOR d3{};
      d3.Width = W;
      d3.Height = H;
      d3.Depth = D;
      d3.Format = HIP_AD_FORMAT_SIGNED_INT32;
      d3.NumChannels = 1;
      hipArray_t a3 = nullptr, a3_copy = nullptr;
      HRR_HIP_CHECK(hipArray3DCreate(&a3, &d3));
      HRR_HIP_CHECK(hipArray3DCreate(&a3_copy, &d3));

      HIP_ARRAY_DESCRIPTOR d2{};
      d2.Width = W;
      d2.Height = H;
      d2.Format = HIP_AD_FORMAT_SIGNED_INT32;
      d2.NumChannels = 1;
      hipArray_t a2 = nullptr;
      HRR_HIP_CHECK(hipArrayCreate(&a2, &d2));

      void *in3 = nullptr, *in2 = nullptr, *out3 = nullptr, *out3_copy = nullptr, *out2 = nullptr;
      HRR_HIP_CHECK(hipMalloc(&in3, BYTES3));
      HRR_HIP_CHECK(hipMalloc(&in2, BYTES2));
      HRR_HIP_CHECK(hipMalloc(&out3, BYTES3));
      HRR_HIP_CHECK(hipMalloc(&out3_copy, BYTES3));
      HRR_HIP_CHECK(hipMalloc(&out2, BYTES2));
      HRR_HIP_CHECK(hipMemcpy(in3, h3.data(), BYTES3, hipMemcpyHostToDevice));
      HRR_HIP_CHECK(hipMemcpy(in2, h2.data(), BYTES2, hipMemcpyHostToDevice));
      HRR_HIP_CHECK(hipMemset(out3, 0, BYTES3));
      HRR_HIP_CHECK(hipMemset(out3_copy, 0, BYTES3));
      HRR_HIP_CHECK(hipMemset(out2, 0, BYTES2));

      {
        hipMemcpy3DParms p{};
        p.srcPtr = make_hipPitchedPtr(in3, ROW, W, H);
        p.dstArray = a3;
        p.extent = make_hipExtent(W, H, D);
        p.kind = hipMemcpyDeviceToDevice;
        HRR_HIP_CHECK(hipMemcpy3D(&p));
      }
      {
        hipMemcpy3DParms p{};
        p.srcPtr = make_hipPitchedPtr(in2, ROW, W, H);
        p.dstArray = a2;
        p.extent = make_hipExtent(W, H, 1);
        p.kind = hipMemcpyDeviceToDevice;
        HRR_HIP_CHECK(hipMemcpy3DAsync(&p, nullptr));
        HRR_HIP_CHECK(hipDeviceSynchronize());
      }
      {
        hipMemcpy3DParms p{};
        p.srcArray = a3;
        p.dstArray = a3_copy;
        p.extent = make_hipExtent(W, H, D);
        p.kind = hipMemcpyDeviceToDevice;
        HRR_HIP_CHECK(hipMemcpy3D(&p));
      }
      auto array_to_device = [&](hipArray_t src, void* dst, size_t depth) {
        hipMemcpy3DParms p{};
        p.srcArray = src;
        p.dstPtr = make_hipPitchedPtr(dst, ROW, W, H);
        p.extent = make_hipExtent(W, H, depth);
        p.kind = hipMemcpyDeviceToDevice;
        HRR_HIP_CHECK(hipMemcpy3D(&p));
      };
      array_to_device(a3, out3, D);
      array_to_device(a3_copy, out3_copy, D);
      array_to_device(a2, out2, 1);
      HRR_HIP_CHECK(hipDeviceSynchronize());

      std::vector<int> r3(W * H * D), r3c(W * H * D), r2(W * H);
      HRR_HIP_CHECK(hipMemcpy(r3.data(), out3, BYTES3, hipMemcpyDeviceToHost));
      HRR_HIP_CHECK(hipMemcpy(r3c.data(), out3_copy, BYTES3, hipMemcpyDeviceToHost));
      HRR_HIP_CHECK(hipMemcpy(r2.data(), out2, BYTES2, hipMemcpyDeviceToHost));

      // Fails if the capture's copies moved nothing: zero blobs would still agree in the roundtrip.
      REQUIRE(r3 == h3);
      REQUIRE(r3c == h3);
      REQUIRE(r2 == h2);

      HRR_HIP_CHECK(hipFree(in3));
      HRR_HIP_CHECK(hipFree(in2));
      HRR_HIP_CHECK(hipFree(out3));
      HRR_HIP_CHECK(hipFree(out3_copy));
      HRR_HIP_CHECK(hipFree(out2));
      HRR_HIP_CHECK(hipFreeArray(a3));
      HRR_HIP_CHECK(hipFreeArray(a3_copy));
      HRR_HIP_CHECK(hipFreeArray(a2));
    }
  }

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

TEST_CASE("Unit_HRR_StressApis_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  constexpr int    N    = 1 << 12;   // 4096 floats
  constexpr size_t SZ   = N * sizeof(float);
  constexpr int    STREAMS  = 8;
  constexpr int    EVENTS   = 8;
  constexpr int    POOLS    = 10;
  constexpr int    ALLOCS   = 25;
  constexpr int    KL_ITERS = 60;   // kernel launches
  constexpr int    CP_ITERS = 30;   // memcpy rounds
  constexpr int    MS_ITERS = 30;   // memset rounds

  hipStream_t streams[STREAMS];
  for (int i = 0; i < STREAMS; ++i)
    HRR_HIP_CHECK(hipStreamCreateWithFlags(&streams[i], hipStreamNonBlocking));

  hipEvent_t evs[EVENTS];
  for (int i = 0; i < EVENTS; ++i)
    HRR_HIP_CHECK(hipEventCreate(&evs[i]));

  float *da, *db, *dc;
  HRR_HIP_CHECK(hipMalloc(&da, SZ));
  HRR_HIP_CHECK(hipMalloc(&db, SZ));
  HRR_HIP_CHECK(hipMalloc(&dc, SZ));

  float* ha = new float[N];
  float* hb = new float[N];
  float* hc = new float[N]();
  for (int i = 0; i < N; ++i) { ha[i] = 1.0f; hb[i] = 1.0f; }

  HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, streams[0]));
  HRR_HIP_CHECK(hipMemcpyAsync(db, hb, SZ, hipMemcpyHostToDevice, streams[1]));
  HRR_HIP_CHECK(hipMemsetAsync(dc, 0, SZ, streams[0]));
  HRR_HIP_CHECK(hipStreamSynchronize(streams[0]));
  HRR_HIP_CHECK(hipStreamSynchronize(streams[1]));

  dim3 block(256), grid((N + 255) / 256);

  for (int i = 0; i < ALLOCS; ++i) {
    float* tmp = nullptr;
    HRR_HIP_CHECK(hipMalloc(&tmp, SZ));
    HRR_HIP_CHECK(hipFree(tmp));
  }

  for (int i = 0; i < 10; ++i) {
    float* tmp = nullptr;
    HRR_HIP_CHECK(hipMallocAsync(&tmp, SZ, streams[i % STREAMS]));
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));
    HRR_HIP_CHECK(hipFreeAsync(tmp, streams[i % STREAMS]));
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));
  }

  for (int i = 0; i < POOLS; ++i) {
    hipMemPool_t pool;
    hipMemPoolProps pp{};
    pp.allocType     = hipMemAllocationTypePinned;
    pp.location.type = hipMemLocationTypeDevice;
    pp.location.id   = 0;
    HRR_HIP_CHECK(hipMemPoolCreate(&pool, &pp));
    HRR_HIP_CHECK(hipMemPoolDestroy(pool));
  }

  for (int iter = 0; iter < KL_ITERS; ++iter) {
    hipStream_t s = streams[iter % STREAMS];
    hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, s, da, db, dc, N);
    HRR_HIP_CHECK(hipGetLastError());
    if (iter % 10 == 9) {
      hipEvent_t ev = evs[(iter / 10) % EVENTS];
      HRR_HIP_CHECK(hipEventRecord(ev, s));
      HRR_HIP_CHECK(hipEventSynchronize(ev));
    }
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < MS_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemsetAsync(dc, 0, SZ, s));
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());
  hipLaunchKernelGGL(hrr_vectorAdd, grid, block, 0, streams[0], da, db, dc, N);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipStreamSynchronize(streams[0]));

  for (int i = 0; i < CP_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemcpyAsync(da, ha, SZ, hipMemcpyHostToDevice, s));
  }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  for (int i = 0; i < 20; ++i) {
    HRR_HIP_CHECK(hipMemcpy(db, da, SZ, hipMemcpyDeviceToDevice));
  }

  for (int i = 0; i < 20; ++i) {
    hipEvent_t ev = evs[i % EVENTS];
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipEventRecord(ev, s));
    HRR_HIP_CHECK(hipEventSynchronize(ev));
  }

  for (int i = 0; i < 15; ++i)
    HRR_HIP_CHECK(hipStreamSynchronize(streams[i % STREAMS]));

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

  for (int i = 0; i < CP_ITERS; ++i) {
    hipStream_t s = streams[i % STREAMS];
    HRR_HIP_CHECK(hipMemcpyAsync(hc, dc, SZ, hipMemcpyDeviceToHost, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
  }

  for (int i = 0; i < N; ++i)
    REQUIRE(hc[i] == 2.0f);

  HRR_HIP_CHECK(hipFree(da)); HRR_HIP_CHECK(hipFree(db)); HRR_HIP_CHECK(hipFree(dc));
  for (int i = 0; i < EVENTS; ++i) HRR_HIP_CHECK(hipEventDestroy(evs[i]));
  for (int i = 0; i < STREAMS; ++i) HRR_HIP_CHECK(hipStreamDestroy(streams[i]));
  delete[] ha; delete[] hb; delete[] hc;
}

TEST_CASE("Unit_HRR_StreamWriteValue_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  // Stream write/wait value is an optional device capability. Gate on the same attribute as
  // hipStreamValue.cc and Unit_HRR_StreamAdvanced2_Direct, so an unsupported device skips.
  int canUseStreamValue = 0;
  HRR_HIP_CHECK(hipDeviceGetAttribute(&canUseStreamValue,
                                  hipDeviceAttributeCanUseStreamWaitValue, 0));
  if (!canUseStreamValue) {
    HRR_SKIP("stream wait value unsupported");
  }

  constexpr uint64_t kVal64 = 0xDEADBEEFFEEDFACEull;
  constexpr uint32_t kVal32 = 0xCAFEBABEu;
  constexpr uint32_t kIncBase = 0x0A0A0A0Au;
  [[maybe_unused]] constexpr uint32_t kIncDelta = 0xF4E3F0C4u;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  // Two 64-bit slots keep the 32-bit and 64-bit writes in disjoint, defined memory.
  uint64_t* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, 2 * sizeof(uint64_t)));
  HRR_HIP_CHECK(hipMemsetAsync(d, 0, 2 * sizeof(uint64_t), s));

  // Separate slot for the flags-bearing write, 8 bytes wide so a 64-bit increment cannot overrun it.
  uint32_t* inc = nullptr;
  HRR_HIP_CHECK(hipMalloc(&inc, sizeof(uint64_t)));
  HRR_HIP_CHECK(hipMemsetAsync(inc, 0, sizeof(uint64_t), s));

  HRR_HIP_CHECK(hipStreamWriteValue64(s, d, kVal64, 0));
  HRR_HIP_CHECK(hipStreamWriteValue32(s, d + 1, kVal32, 0));

  // Flags are plumbed through capture and replay, so exercise a non-default one.
  HRR_HIP_CHECK(hipStreamWriteValue32(s, inc, kIncBase, 0));
  hipError_t incErr = hipErrorNotSupported;
#ifdef hipExtStreamWriteValueIncrement
  incErr = hipStreamWriteValue32(s, inc, kIncDelta, hipExtStreamWriteValueIncrement);
  REQUIRE((incErr == hipSuccess || incErr == hipErrorInvalidValue
           || incErr == hipErrorNotSupported));
#endif

  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  // One readback per value: an 8-byte blob for the 64-bit write, 4-byte blobs for the 32-bit ones.
  uint64_t h64 = 0;
  HRR_HIP_CHECK(hipMemcpy(&h64, d, sizeof(h64), hipMemcpyDeviceToHost));
  REQUIRE(h64 == kVal64);

  uint32_t h32 = 0;
  HRR_HIP_CHECK(hipMemcpy(&h32, d + 1, sizeof(h32), hipMemcpyDeviceToHost));
  REQUIRE(h32 == kVal32);

  uint32_t hInc = 0;
  HRR_HIP_CHECK(hipMemcpy(&hInc, inc, sizeof(hInc), hipMemcpyDeviceToHost));
  // The exact sum is not asserted: replay reproduces whatever capture produced.
  // A successful increment of a non-zero delta must change the slot.
  if (incErr == hipSuccess) REQUIRE(hInc != kIncBase);

  HRR_HIP_CHECK(hipFree(inc));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

// Pitched device-to-device memset (hipMemsetD2D*): shared geometry
// PITCH must be strictly greater than WIDTH.  CLR short-circuits the pitched
namespace {
constexpr size_t kD2DWidth = 32 * sizeof(int);          // 128 bytes filled per row
constexpr size_t kD2DPitch = 192;                       // row stride (> width)
constexpr size_t kD2DRows  = 32;
constexpr size_t kD2DTotal = kD2DPitch * kD2DRows;      // 6144 bytes
constexpr size_t kD2DPad   = kD2DPitch - kD2DWidth;     // 64 untouched bytes/row

// The sentinel pre-fills the whole buffer and the pattern fills the width sub-region.
// They must differ far beyond HIP_HRR_D2H_TOL (atol=rtol=1e-3 for f32/bf16/f16/f64), or a no-op replay would pass.
constexpr unsigned char kD2DSentinelByte = 0x5Au;       // 0x5A5A5A5A per int
constexpr unsigned int  kD2DPattern      = 0x44444444u;

static_assert(kD2DPitch > kD2DWidth, "pitched memset path requires pitch > width");
static_assert(kD2DWidth % sizeof(unsigned int) == 0, "width must hold whole ints");

// Counts ints in the filled sub-region equal to `pattern` and padding bytes still holding the sentinel.
// Aggregated so a mismatch does not emit thousands of Catch2 failures.
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

// Replay mirrors the hipMemset2D handler, with alloc-map pointer translation and stream translation
// for the async variants, and must reproduce the final byte pattern.
TEST_CASE("Unit_HRR_MemsetD2D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  void* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, kD2DTotal));
  const hipDeviceptr_t dp = reinterpret_cast<hipDeviceptr_t>(d);

  // Sentinel the whole allocation, padding included, with an already-replayed API,
  // so the padding has a known non-zero value that the pitched fills must leave alone.
  HRR_HIP_CHECK(hipMemsetD8(dp, kD2DSentinelByte, kD2DTotal));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipMemsetD2D8(dp, kD2DPitch, 0x11, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D8Async(dp, kD2DPitch, 0x22, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D16(dp, kD2DPitch, 0x3333, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D16Async(dp, kD2DPitch, 0x4444, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D32(dp, kD2DPitch, 0x11223344u, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D32Async(dp, kD2DPitch, 0x55667788u, kD2DWidth, kD2DRows, s));

  // The non-Async D2D memsets are not synchronous: ihipMemset3D enqueues on the null stream without finish(),
  // and `s` is hipStreamNonBlocking, so the device must be synchronized before the final fill.
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

// hipMemAllocPitch is a playback no-op, so the pitched destination it returns cannot be translated at replay.
TEST_CASE("Unit_HRR_MemsetD2DPitchAlloc_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  hipDeviceptr_t pitched = nullptr;
  size_t pitch = 0;
  HRR_HIP_CHECK(hipMemAllocPitch(&pitched, &pitch, kD2DWidth, kD2DRows,
                             /*elementSizeBytes=*/4));
  REQUIRE(pitched != nullptr);
  REQUIRE(pitch >= kD2DWidth);

  // All six variants, so every generated handler takes the skip path on replay.
  HRR_HIP_CHECK(hipMemsetD2D8(pitched, pitch, 0x11, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D8Async(pitched, pitch, 0x22, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D16(pitched, pitch, 0x3333, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D16Async(pitched, pitch, 0x4444, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipMemsetD2D32(pitched, pitch, 0x11223344u, kD2DWidth, kD2DRows));
  HRR_HIP_CHECK(hipMemsetD2D32Async(pitched, pitch, kD2DPattern, kD2DWidth, kD2DRows, s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

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

TEST_CASE("Unit_HRR_MemsetVariants_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 1024;
  constexpr size_t SZ = N * sizeof(int);  // 4096 bytes

  // Final validation pattern. Replay zero-initialises allocations, and under a float tolerance (HIP_HRR_D2H_TOL)
  // a small integer canary such as 2 decodes to 2.8e-45 and passes against zeros, so use a large pattern.
  constexpr int VAL = 0x44444444;

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  HRR_HIP_CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(d), 0x01, SZ));
  HRR_HIP_CHECK(hipMemsetD8Async(reinterpret_cast<hipDeviceptr_t>(d), 0x02, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(d), 0x0003, SZ / 2));
  HRR_HIP_CHECK(hipMemsetD16Async(reinterpret_cast<hipDeviceptr_t>(d), 0x0004, SZ / 2, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 2, N));
  HRR_HIP_CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(d), 2, N, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  constexpr size_t COLS  = 32;
  constexpr size_t ROWS  = N / COLS;
  constexpr size_t PITCH = COLS * sizeof(int);
  HRR_HIP_CHECK(hipMemset2D(d, PITCH, 0, PITCH, ROWS));
  HRR_HIP_CHECK(hipMemset2DAsync(d, PITCH, 0, PITCH, ROWS, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipDeviceSynchronize());  // ensure all async ops complete first
  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), VAL, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

TEST_CASE("Unit_HRR_DeviceInfo_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  HRR_HIP_CHECK(hipInit(0));

  int dev = -1;
  HRR_HIP_CHECK(hipGetDevice(&dev));
  REQUIRE(dev == 0);

  char devname[256] = {};
  HRR_HIP_CHECK(hipDeviceGetName(devname, sizeof(devname), dev));
  // hipDeviceGetName returns an empty string on some Linux ROCm builds, so log it without asserting.
  INFO("Device name: '" << devname << "'");

  char pci[64] = {};
  HRR_HIP_CHECK(hipDeviceGetPCIBusId(pci, sizeof(pci), dev));
  int dev2 = -1;
  HRR_HIP_CHECK(hipDeviceGetByPCIBusId(&dev2, pci));
  REQUIRE(dev2 == dev);

  size_t totalMem = 0;
  HRR_HIP_CHECK(hipDeviceTotalMem(&totalMem, dev));
  REQUIRE(totalMem > 0);

  int major = 0, minor = 0;
  HRR_HIP_CHECK(hipDeviceComputeCapability(&major, &minor, dev));
  REQUIRE(major > 0);

  size_t stackSize = 0;
  HRR_HIP_CHECK(hipDeviceGetLimit(&stackSize, hipLimitStackSize));
  HRR_HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, stackSize));

  hipSharedMemConfig shmCfg = hipSharedMemBankSizeDefault;
  HRR_HIP_CHECK(hipDeviceGetSharedMemConfig(&shmCfg));
  HRR_HIP_CHECK(hipDeviceSetSharedMemConfig(shmCfg));

  unsigned int dflags = 0;
  HRR_HIP_CHECK(hipGetDeviceFlags(&dflags));

  hipDeviceProp_t req{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&req, 0));
  int chosen = -1;
  HRR_HIP_CHECK(hipChooseDevice(&chosen, &req));
  REQUIRE(chosen >= 0);

  // Allocate before the pool and graph queries to avoid SEH side effects.
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  hipMemPool_t defPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetDefaultMemPool(&defPool, dev));
  REQUIRE(defPool != nullptr);

  hipMemPool_t curPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, dev));

  uint64_t usedMem = 0;
  HRR_HIP_CHECK(hipDeviceGetGraphMemAttribute(dev, hipGraphMemAttrUsedMemCurrent, &usedMem));

  unsigned int ctxFlags = 0; int ctxActive = 0;
  HRR_HIP_CHECK(hipDevicePrimaryCtxGetState(dev, &ctxFlags, &ctxActive));

  HRR_HIP_CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(d), 7, N));
  HRR_HIP_CHECK(hipDeviceSynchronize());
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 7);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

TEST_CASE("Unit_HRR_StreamAdvanced_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));

  hipStream_t s0, s1;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));
  HRR_HIP_CHECK(hipStreamCreateWithPriority(&s1, hipStreamNonBlocking, 0));

  unsigned long long sid = 0;
  HRR_HIP_CHECK(hipStreamGetId(s0, &sid));
  REQUIRE(sid != 0);

  hipStreamAttrValue av{};
  HRR_HIP_CHECK(hipStreamGetAttribute(s0, hipStreamAttributeSynchronizationPolicy, &av));
  HRR_HIP_CHECK(hipStreamSetAttribute(s0, hipStreamAttributeSynchronizationPolicy, &av));

  HRR_HIP_CHECK(hipStreamCopyAttributes(s1, s0));

  hipStreamCaptureStatus capStatus = hipStreamCaptureStatusActive;
  HRR_HIP_CHECK(hipStreamIsCapturing(s0, &capStatus));
  REQUIRE(capStatus == hipStreamCaptureStatusNone);

  unsigned long long capId = 0;
  HRR_HIP_CHECK(hipStreamGetCaptureInfo(s0, &capStatus, &capId));
  REQUIRE(capStatus == hipStreamCaptureStatusNone);

  hipDevice_t streamDev = -1;
  { hipError_t e = hipStreamGetDevice(s0, &streamDev);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  { hipError_t q = hipStreamQuery(s0); REQUIRE((q == hipSuccess || q == hipErrorNotReady)); }

  hipEvent_t ev;
  HRR_HIP_CHECK(hipEventCreate(&ev));
  HRR_HIP_CHECK(hipEventRecordWithFlags(ev, s0, 0));
  HRR_HIP_CHECK(hipEventSynchronize(ev));

  HRR_HIP_CHECK(hipStreamSynchronize(s0));
  HRR_HIP_CHECK(hipStreamSynchronize(s1));

  // Run last, after all stream work completes, so it cannot interfere with submission.
  hipStreamCaptureMode mode = hipStreamCaptureModeGlobal;
  HRR_HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));
  HRR_HIP_CHECK(hipThreadExchangeStreamCaptureMode(&mode));  // restore original

  HRR_HIP_CHECK(hipDeviceSynchronize());

  constexpr int N = 256; constexpr size_t SZ = N * sizeof(int);
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
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

  HRR_HIP_CHECK(hipMemcpyHtoD(reinterpret_cast<hipDeviceptr_t>(d0), h_src, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipMemcpyDtoD(reinterpret_cast<hipDeviceptr_t>(d1),
                           reinterpret_cast<hipDeviceptr_t>(d0), SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  HRR_HIP_CHECK(hipMemcpyDtoDAsync(reinterpret_cast<hipDeviceptr_t>(d2),
                                reinterpret_cast<hipDeviceptr_t>(d1), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h_mid = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoH(h_mid, reinterpret_cast<hipDeviceptr_t>(d2), SZ));
  for (int i = 0; i < N; ++i) REQUIRE(h_mid[i] == 99);

  int* h_mid2 = new int[N]();
  HRR_HIP_CHECK(hipMemcpyDtoHAsync(h_mid2, reinterpret_cast<hipDeviceptr_t>(d2), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_mid2[i] == 99);

  HRR_HIP_CHECK(hipMemcpyHtoDAsync(reinterpret_cast<hipDeviceptr_t>(d0), h_src, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  HRR_HIP_CHECK(hipMemcpy2D(d1, SZ, d2, SZ, SZ, 1, hipMemcpyDeviceToDevice));

  HRR_HIP_CHECK(hipMemcpy2DAsync(d2, SZ, d0, SZ, SZ, 1, hipMemcpyDeviceToDevice, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  size_t pitch = 0;
  int* d_p = nullptr;
  HRR_HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&d_p), &pitch,
                            N * sizeof(int), 4));
  HRR_HIP_CHECK(hipMemcpy2D(d_p, pitch, d0, SZ, N * sizeof(int), 1,
                         hipMemcpyDeviceToDevice));
  HRR_HIP_CHECK(hipFree(d_p));

  HRR_HIP_CHECK(hipMemcpyPeer(d1, 0, d0, 0, SZ));
  HRR_HIP_CHECK(hipMemcpyPeerAsync(d2, 0, d1, 0, SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d2, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == 99);

  HRR_HIP_CHECK(hipFree(d0)); HRR_HIP_CHECK(hipFree(d1)); HRR_HIP_CHECK(hipFree(d2));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h_src; delete[] h_mid; delete[] h_mid2; delete[] h_out;
}

static __global__ void fill_kernel_e(int* d, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = val;
}

TEST_CASE("Unit_HRR_Occupancy_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  int numBlocks = 0;
  HRR_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocks, fill_kernel_e, 64, 0));
  REQUIRE(numBlocks > 0);

  int numBlocks2 = 0;
  HRR_HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &numBlocks2, fill_kernel_e, 64, 0, hipOccupancyDefault));
  REQUIRE(numBlocks2 > 0);

  int minGridSize = 0, blockSize = 0;
  HRR_HIP_CHECK(hipOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, fill_kernel_e, 0, 0));
  REQUIRE(blockSize > 0);

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
    hipLaunchKernelGGL(fill_kernel_e, grid, block, 0, s, d, val, N);
  }
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 3);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

TEST_CASE("Unit_HRR_HostAliases_Direct", "[.][hrr-direct]") {
  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  void* ha = nullptr;
  HRR_HIP_CHECK(hipHostAlloc(&ha, SZ, hipHostMallocDefault));
  REQUIRE(ha != nullptr);

  unsigned int hflags = 0;
  HRR_HIP_CHECK(hipHostGetFlags(&hflags, ha));

  HRR_HIP_CHECK(hipFreeHost(ha));

  void* mh = nullptr;
  HRR_HIP_CHECK(hipMallocHost(&mh, SZ));
  REQUIRE(mh != nullptr);
  HRR_HIP_CHECK(hipFreeHost(mh));

  void* dah = nullptr;
  HRR_HIP_CHECK(hipMemAllocHost(&dah, SZ));
  REQUIRE(dah != nullptr);
  HRR_HIP_CHECK(hipFreeHost(dah));

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

  hipPitchedPtr pp{};
  hipExtent ext = make_hipExtent(32 * sizeof(int), 8, 4);
  HRR_HIP_CHECK(hipMalloc3D(&pp, ext));
  REQUIRE(pp.ptr != nullptr);

  hipMemoryType mtype = hipMemoryTypeUnified;
  HRR_HIP_CHECK(hipPointerGetAttribute(&mtype, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, pp.ptr));

  HRR_HIP_CHECK(hipFree(pp.ptr));

  hipDeviceptr_t dptr = 0;
  size_t rowPitch = 0;
  HRR_HIP_CHECK(hipMemAllocPitch(&dptr, &rowPitch, N * sizeof(int), 4, sizeof(int)));
  REQUIRE(dptr != 0);
  HRR_HIP_CHECK(hipFree(reinterpret_cast<void*>(dptr)));

  (void)hipDeviceSynchronize();
  (void)hipGetLastError();
  int* d = nullptr; int* h = new int[N]();
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  // hipMemset plus a synchronous hipMemcpy avoids a GPU TDR on the Windows ROCm driver:
  // an async D2H after array or 3D allocation triggers error 719.
  HRR_HIP_CHECK(hipMemset(d, 0, SZ));
  HRR_HIP_CHECK(hipMemcpy(h, d, SZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 0);
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

TEST_CASE("Unit_HRR_MemPoolExtended_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipMemPoolProps props{};
  props.allocType   = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id   = 0;
  hipMemPool_t pool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&pool, &props));

  hipMemAccessFlags accessFlags{};
  hipMemLocation loc{hipMemLocationTypeDevice, 0};
  HRR_HIP_CHECK(hipMemPoolGetAccess(&accessFlags, pool, &loc));

  HRR_HIP_CHECK(hipMemPoolTrimTo(pool, 0));

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMallocFromPoolAsync(reinterpret_cast<void**>(&d), SZ, pool, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  // Not supported on all Linux ROCm builds, so skip gracefully.
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

// Cleanup destroys the user pool while it is current. hipMemPoolDestroy then resets the device
// to its default pool (see hip_mempool.cpp), so replay needs no restore.
TEST_CASE("Unit_HRR_DeviceMemPool_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipMemPool_t defPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetDefaultMemPool(&defPool, 0));
  REQUIRE(defPool != nullptr);
  hipMemPool_t curPool = nullptr;
  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, 0));
  REQUIRE(curPool == defPool);

  hipMemPoolProps props{};
  props.allocType     = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id   = 0;
  hipMemPool_t myPool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&myPool, &props));

  HRR_HIP_CHECK(hipDeviceSetMemPool(0, myPool));

  HRR_HIP_CHECK(hipDeviceGetMemPool(&curPool, 0));
  REQUIRE(curPool == myPool);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMallocAsync(reinterpret_cast<void**>(&d), SZ, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

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

// hipExtMallocWithFlags: the manual handler allocates a real buffer and records it in alloc_map, as hipMalloc does,
// so a write plus D2H validates byte-for-byte at replay.
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

// hipExtModuleLaunchKernel has a manual playback handler (replay_kernel_launch).
TEST_CASE("Unit_HRR_ExtModuleLaunchKernel_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    LN  = 256;
  constexpr size_t LSZ = LN * sizeof(int);

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, LSZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

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

  // The runtime marks start and stop inside the launch; the app never records
  // them, so replay needs the markers capture stores around the launch record.
  hipEvent_t start = nullptr, stop = nullptr;
  HRR_HIP_CHECK(hipEventCreate(&start));
  HRR_HIP_CHECK(hipEventCreate(&stop));

  int   val = 55;
  int   n   = LN;
  void* args[] = { &d, &val, &n };
  HRR_HIP_CHECK(hipExtModuleLaunchKernel(fn,
      /*globalWorkSize*/ LN, 1, 1,
      /*localWorkSize */ 256, 1, 1,
      /*sharedMemBytes*/ 0, s, args, /*extra*/ nullptr,
      start, stop, /*flags*/ 0));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  float ms = -1.f;
  HRR_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  REQUIRE(ms >= 0.f);
  HRR_HIP_CHECK(hipEventDestroy(start));
  HRR_HIP_CHECK(hipEventDestroy(stop));

  HRR_HIP_CHECK(hipModuleUnload(mod));

  int* h = new int[LN]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, LSZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < LN; ++i) REQUIRE(h[i] == 55);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// The pinned buffer is an H2D source so the allocation is live, then hipHostFree releases it before the D2H.
// Replay's alloc-map removal happens mid-stream and must not disturb the pending copy.
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

  HRR_HIP_CHECK(hipHostFree(h_pinned));

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 44);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// The logging calls touch no device memory, so a memset-based D2H canary confirms the replay stream stays intact.
TEST_CASE("Unit_HRR_Logging_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);

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

// hipConfigureCall only pushes a call configuration onto the thread-local execution stack and returns hipSuccess.
// No hipLaunchByPtr consumes it, so a memset-based D2H canary checks the stream stays intact.
TEST_CASE("Unit_HRR_ConfigureCall_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    N  = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

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

// A buffer on src_dev is peer-copied to dst_dev via hipMemcpyPeer.
// Replay must recreate both allocations on the correct devices for the copy to land.
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

TEST_CASE("Unit_HRR_MemsetExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

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

  {
    void* d2d = nullptr;
    size_t pitch2d = 0;
    HRR_HIP_CHECK(hipMallocPitch(&d2d, &pitch2d, 16, 4));

    HRR_HIP_CHECK(hipMemsetD2D8(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x77, 16, 4));
    HRR_HIP_CHECK(hipMemsetD2D8Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x00, 16, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipMemsetD2D16(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x1234, 8, 4));
    HRR_HIP_CHECK(hipMemsetD2D16Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x0000, 8, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipMemsetD2D32(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0xDEAD, 4, 4));
    HRR_HIP_CHECK(hipMemsetD2D32Async(reinterpret_cast<hipDeviceptr_t>(d2d), pitch2d, 0x0000, 4, 4, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));

    HRR_HIP_CHECK(hipFree(d2d));
  }


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

TEST_CASE("Unit_HRR_MemcpyExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 64;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  int* h_src = new int[N];
  for (int i = 0; i < N; ++i) h_src[i] = 55;
  int* h_chk = new int[N]();

  HRR_HIP_CHECK(hipMemcpy(d, h_src, SZ, hipMemcpyHostToDevice));

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

  {
    int supportsImages = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&supportsImages, hipDeviceAttributeImageSupport, 0));
    if (supportsImages) {
      hipChannelFormatDesc desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
      hipArray_t arr = nullptr;
      HRR_HIP_CHECK(hipMallocArray(&arr, &desc, N, 1, hipArrayDefault));

      HRR_HIP_CHECK(hipMemcpyToArray(arr, 0, 0, h_src, SZ, hipMemcpyHostToDevice));

      int h_arr[N] = {};
      HRR_HIP_CHECK(hipMemcpyFromArray(h_arr, arr, 0, 0, SZ, hipMemcpyDeviceToHost));
      REQUIRE(h_arr[0] == 55);

      HRR_HIP_CHECK(hipMemcpy2DToArray(arr, 0, 0, h_src, SZ, SZ, 1, hipMemcpyHostToDevice));

      int h_arr2[N] = {};
      HRR_HIP_CHECK(hipMemcpy2DFromArray(h_arr2, SZ, arr, 0, 0, SZ, 1, hipMemcpyDeviceToHost));
      REQUIRE(h_arr2[0] == 55);

      int h_ato[N] = {};
      HRR_HIP_CHECK(hipMemcpyAtoH(h_ato, arr, 0, SZ));
      REQUIRE(h_ato[0] == 55);

      HRR_HIP_CHECK(hipMemcpyHtoA(arr, 0, h_src, SZ));

      HRR_HIP_CHECK(hipFreeArray(arr));
    }
  }

  HRR_HIP_CHECK(hipMemcpy(d, h_src, SZ, hipMemcpyHostToDevice));

  int* h_out = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h_out, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == 55);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h_src; delete[] h_chk; delete[] h_out;
}

TEST_CASE("Unit_HRR_DeviceExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  hipUUID uuid{};
  { hipError_t e = hipDeviceGetUuid(&uuid, 0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  int p2pVal = 0;
  { hipError_t e = hipDeviceGetP2PAttribute(&p2pVal,
                       hipDevP2PAttrPerformanceRank, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice)); }

  hipFuncCache_t cacheCfg = hipFuncCachePreferNone;
  HRR_HIP_CHECK(hipDeviceGetCacheConfig(&cacheCfg));
  HRR_HIP_CHECK(hipDeviceSetCacheConfig(cacheCfg));

  { hipError_t e = hipDeviceEnablePeerAccess(0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice
             || e == hipErrorPeerAccessAlreadyEnabled)); }
  { hipError_t e = hipDeviceDisablePeerAccess(0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice
             || e == hipErrorPeerAccessNotEnabled)); }

  { hipError_t e = hipDevicePrimaryCtxSetFlags(0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorContextAlreadyInUse)); }

  { hipError_t e = hipDevicePrimaryCtxRelease(0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidDevice)); }
  HRR_HIP_CHECK(hipSetDevice(0));

  size_t maxW = 0;
  hipChannelFormatDesc cfd = hipCreateChannelDesc(8, 0, 0, 0, hipChannelFormatKindUnsigned);
  { hipError_t e = hipDeviceGetTexture1DLinearMaxWidth(&maxW, &cfd, 0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  { hipError_t e = hipDeviceGraphMemTrim(0);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

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

TEST_CASE("Unit_HRR_StreamAdvanced2_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s0;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s0, hipStreamNonBlocking));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  hipStream_t cuStream = nullptr;
  {
    uint32_t cuMask = 0xFFFFFFFF;
    hipError_t e = hipExtStreamCreateWithCUMask(&cuStream, 1, &cuMask);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  if (cuStream) {
    uint32_t outMask[2] = {};
    hipError_t e = hipExtStreamGetCUMask(cuStream, 2, outMask);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  {
    uint32_t linktype = 0, hopcount = 0;
    hipError_t e = hipExtGetLinkTypeAndHopCount(0, 0, &linktype, &hopcount);
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported
             || e == hipErrorInvalidValue));
  }

  { int devId = hipGetStreamDeviceId(s0);
    REQUIRE((devId == 0 || devId == -1)); }

  {
    int* managed = nullptr;
    HRR_HIP_CHECK(hipMallocManaged(&managed, SZ));
    { hipError_t e = hipStreamAttachMemAsync(s0, managed, SZ,
                       hipMemAttachSingle);
      REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }
    HRR_HIP_CHECK(hipStreamSynchronize(s0));
    HRR_HIP_CHECK(hipFree(managed));
  }

  {
    int canWait = 0;
    HRR_HIP_CHECK(hipDeviceGetAttribute(&canWait,
                 hipDeviceAttributeCanUseStreamWaitValue, 0));
    if (canWait) {
      int* flag = nullptr;
      HRR_HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag),
                              sizeof(int), hipHostMallocMapped));
      *flag = 0;
      HRR_HIP_CHECK(hipStreamWriteValue32(s0, flag, 1, 0));
      HRR_HIP_CHECK(hipStreamWaitValue32(s0, flag, 1,
                   hipStreamWaitValueEq, 0xFFFFFFFF));
      HRR_HIP_CHECK(hipStreamSynchronize(s0));
      HRR_HIP_CHECK(hipFreeHost(flag));
    }
  }

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

  HRR_HIP_CHECK(hipDeviceSynchronize());

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

TEST_CASE("Unit_HRR_Context_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  { hipError_t e = hipCtxSynchronize();
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

  { unsigned int ctxFlags2 = 0; (void)hipCtxGetFlags(&ctxFlags2); }

  hipFuncCache_t cc = hipFuncCachePreferNone;
  (void)hipCtxGetCacheConfig(&cc);
  (void)hipCtxSetCacheConfig(cc);

  hipSharedMemConfig smc = hipSharedMemBankSizeDefault;
  (void)hipCtxGetSharedMemConfig(&smc);
  (void)hipCtxSetSharedMemConfig(smc);

  { unsigned int apiVer = 0; (void)hipCtxGetApiVersion(nullptr, &apiVer); }

  hipCtx_t curCtx = nullptr;
  (void)hipCtxGetCurrent(&curCtx);
  if (curCtx) { (void)hipCtxSetCurrent(curCtx); }

  (void)hipCtxEnablePeerAccess(nullptr, 0);
  (void)hipCtxDisablePeerAccess(nullptr);

  { hipError_t e = hipCtxSynchronize();
    REQUIRE((e == hipSuccess || e == hipErrorNotSupported)); }

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

// Resolved with dlsym after the capture library loads, so the symbol is present at run time.
static hipError_t (*hrr_hip_library_get_module())(hipModule_t*, hipLibrary_t) {
#ifndef _WIN32
  using Fn = hipError_t (*)(hipModule_t*, hipLibrary_t);
  static Fn fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "hipLibraryGetModule"));
  return fn;
#else
  return &hipLibraryGetModule;
#endif
}

TEST_CASE("Unit_HRR_ModuleExtra_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int N = 256;
  constexpr size_t SZ = N * sizeof(int);

  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  hipFunction_t func = nullptr;
  { hipError_t e = hipGetFuncBySymbol(&func,
                       reinterpret_cast<const void*>(fill_kernel_e));
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorNotFound)); }

  if (func) {
    int attribVal = 0;
    hipError_t e = hipFuncGetAttribute(&attribVal,
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, func);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  {
    unsigned int fnCount = 0;
    hipError_t e = hipModuleGetFunctionCount(&fnCount, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidHandle)); }

  if (func) {
    int gs = 0, bs = 0;
    hipError_t e = hipModuleOccupancyMaxPotentialBlockSize(&gs, &bs, func, 0, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

  if (func) {
    int nb = 0;
    hipError_t e = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                       &nb, func, 64, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue)); }

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

  {
    hipLibrary_t lib = nullptr;
    hipError_t e = hipLibraryLoadData(&lib, nullptr,
                       nullptr, nullptr, 0,
                       nullptr, nullptr, 0);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidImage || e == hipErrorNotSupported)); }

  { unsigned int kc = 0;
    hipError_t e = hipLibraryGetKernelCount(&kc, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidHandle || e == hipErrorNotSupported)); }

  // hipLibraryGetModule needs a valid library; nullptr drives only the capture shim's rejection path.
  // Generated shims record under `if (r == hipSuccess)`, so a failing call is deliberately not written.
  if (auto hipLibraryGetModuleFn = hrr_hip_library_get_module()) {
    hipModule_t lm = nullptr;
    hipError_t e = hipLibraryGetModuleFn(&lm, nullptr);
    REQUIRE((e == hipSuccess || e == hipErrorInvalidValue
             || e == hipErrorInvalidHandle || e == hipErrorNotSupported)); }

  { int kav = 0;
    hipError_t e = hipKernelGetAttribute(&kav,
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                       nullptr, 0);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

  { hipError_t e = hipKernelSetAttribute(
                       HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256,
                       nullptr, 0);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

  { hipFunction_t kf = nullptr;
    hipError_t e = hipKernelGetFunction(&kf, nullptr);
    (void)e; /* any error is acceptable with nullptr kernel handle */ }

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

__global__ void fill_kernel_u(int* d, int val, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = val;
}

TEST_CASE("Unit_HRR_MiscAPIs_Direct", "[.][hrr][direct]") {
  // Warm-up: capture shims install during hip::init() on the first HIP call, which is not recorded.
  // Make it a harmless hipSetDevice, or the hipMalloc below would go unrecorded.
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  (void)hipProfilerStart();
  (void)hipProfilerStop();

  (void)hipExtGetLastError();

  { size_t dynSmem = 0;
    (void)hipOccupancyAvailableDynamicSMemPerBlock(&dynSmem, fill_kernel_u, 128, 0); }

  { void* pfn = nullptr;
    (void)hipGetProcAddress("hipMalloc", &pfn, HIP_VERSION, 0, nullptr); }

  { int devArr[] = {0};
    (void)hipSetValidDevices(devArr, 1); }

  { size_t ptrSize = 0;
    (void)hipMemPtrGetInfo(d, &ptrSize); }

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

// Canaries must stay distinguishable from zero under a float tolerance, since replay zero-initialises allocations.
static constexpr int kDrvPayload  = 0x5C5C5C5C;  // bytes the copy must move
static constexpr int kDrvPad      = 0x4B4B4B4B;  // device padding that must survive
static constexpr int kDrvHostFill = 0x3A3A3A3A;  // host padding that must NOT be copied

// Pitched-rect geometry: pitch > width and height > 1, so the host source
// footprint is pitch*(height-1)+width rather than the width*height volume.
static constexpr size_t kDrvPitch = 256;  // bytes per row
static constexpr size_t kDrvWidth = 64;   // bytes actually copied per row
static constexpr size_t kDrvRows  = 4;    // rows per slice

TEST_CASE("Unit_HRR_DrvMemcpy3D_Direct", "[.][hrr][direct]") {
  // The copies below need the H2D blob path and dual device-pointer translation at replay.
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

  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hsrc;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(A);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3D(&p)); }

  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(A);
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(B);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3DAsync(&p, s)); }
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  { HIP_MEMCPY3D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(B);
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(C);
    p.WidthInBytes  = SZ; p.Height = 1; p.Depth = 1;
    p.srcPitch = SZ; p.srcHeight = 1;
    p.dstPitch = SZ; p.dstHeight = 1;
    HRR_HIP_CHECK(hipDrvMemcpy3D(&p)); }
  // The non-blocking stream never joins the null stream, so drain the device or the readback is unordered against the copy.
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, C, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

  // Host footprint is pitch*(height-1)+width per slice: 1024 + 768 + 64 = 1856 bytes, not the 512-byte volume.
  // A blob sized by volume makes the runtime stride 1344 bytes past its end on replay.
  // Seeding the destination with kDrvPad and requiring the inter-row padding intact catches it.
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

  // Both are no-ops on playback, so they must not touch a buffer that feeds a D2H check.
  // A same-pointer copy is also a no-op at capture time, so capture and replay agree either way.
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

  { hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeHost;   p.srcHost   = hsrc;                                p.srcPitch = SZ;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(A); p.dstPitch = SZ;
    p.WidthInBytes  = SZ; p.Height = 1;
    HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&p)); }

  { hip_Memcpy2D p{};
    p.srcMemoryType = hipMemoryTypeDevice; p.srcDevice = reinterpret_cast<hipDeviceptr_t>(A); p.srcPitch = SZ;
    p.dstMemoryType = hipMemoryTypeDevice; p.dstDevice = reinterpret_cast<hipDeviceptr_t>(B); p.dstPitch = SZ;
    p.WidthInBytes  = SZ; p.Height = 1;
    HRR_HIP_CHECK(hipDrvMemcpy2DUnaligned(&p)); }
  HRR_HIP_CHECK(hipDeviceSynchronize());

  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, B, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == VAL);

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

TEST_CASE("Unit_HRR_Texture_Direct", "[.][hrr][direct]") {
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

  hipArray_t arr0 = nullptr;
  { HIP_ARRAY_DESCRIPTOR desc{};
    desc.Width  = 64;
    desc.Height = 1;
    desc.Format = HIP_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    HRR_HIP_CHECK(hipArrayCreate(&arr0, &desc)); }

  { HIP_ARRAY_DESCRIPTOR out{};
    (void)hipArrayGetDescriptor(&out, arr0); }

  { hipChannelFormatDesc cfd{};
    (void)hipGetChannelDesc(&cfd, arr0); }

  hipArray_t arr3d = nullptr;
  { HIP_ARRAY3D_DESCRIPTOR desc3{};
    desc3.Width  = 4;
    desc3.Height = 4;
    desc3.Depth  = 1;
    desc3.Format = HIP_AD_FORMAT_FLOAT;
    desc3.NumChannels = 1;
    HRR_HIP_CHECK(hipArray3DCreate(&arr3d, &desc3)); }

  { HIP_ARRAY3D_DESCRIPTOR out3{};
    (void)hipArray3DGetDescriptor(&out3, arr3d); }

  { hipChannelFormatDesc cfd{};
    hipExtent ext{};
    unsigned int flags = 0;
    (void)hipArrayGetInfo(&cfd, &ext, &flags, arr0); }

  { hipMipmappedArray_t mma = nullptr;
    hipChannelFormatDesc cfd = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipExtent ext = make_hipExtent(4, 4, 0);
    hipError_t e = hipMallocMipmappedArray(&mma, &cfd, ext, 2, 0);
    if (e == hipSuccess && mma) {
      hipArray_t lvl = nullptr;
      (void)hipGetMipmappedArrayLevel(&lvl, mma, 0);
      HRR_HIP_CHECK(hipFreeMipmappedArray(mma));
    } }

  { hipTextureObject_t tex = 0;
    hipResourceDesc rd{}; rd.resType = hipResourceTypeArray; rd.res.array.array = arr0;
    hipTextureDesc  td{}; td.addressMode[0] = hipAddressModeClamp;
                          td.filterMode = hipFilterModePoint;
                          td.readMode   = hipReadModeElementType;
    hipError_t e = hipCreateTextureObject(&tex, &rd, &td, nullptr);
    if (e == hipSuccess) { (void)hipDestroyTextureObject(tex); } }

  { hipTextureObject_t tex = 0;
    HIP_RESOURCE_DESC rd{}; rd.resType = HIP_RESOURCE_TYPE_ARRAY; rd.res.array.hArray = arr0;
    HIP_TEXTURE_DESC  td{};
    hipError_t e = hipTexObjectCreate(&tex, &rd, &td, nullptr);
    if (e == hipSuccess) { (void)hipTexObjectDestroy(tex); } }

  HRR_HIP_CHECK(hipArrayDestroy(arr0));
  HRR_HIP_CHECK(hipArrayDestroy(arr3d));

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

TEST_CASE("Unit_HRR_GraphExplicit_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  hipGraph_t emptyGraph = nullptr;
  (void)hipGraphCreate(&emptyGraph, 0);

  hipGraph_t capturedGraph = nullptr;
  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  HRR_HIP_CHECK(hipMemsetAsync(d, 0, SZ, s));
  HRR_HIP_CHECK(hipStreamEndCapture(s, &capturedGraph));

  hipGraph_t clonedGraph = nullptr;
  (void)hipGraphClone(&clonedGraph, capturedGraph);

#ifdef _WIN32
  (void)hipGraphDebugDotPrint(capturedGraph, "NUL", 0);
#else
  (void)hipGraphDebugDotPrint(capturedGraph, "/dev/null", 0);
#endif

  hipGraphExec_t exec = nullptr;
  HRR_HIP_CHECK(hipGraphInstantiate(&exec, capturedGraph, nullptr, nullptr, 0));

  { unsigned long long execFlags = 0;
    (void)hipGraphExecGetFlags(exec, &execFlags); }

  { size_t numNodes = 0;
    (void)hipGraphGetNodes(capturedGraph, nullptr, &numNodes);
    if (numNodes > 0) {
      std::vector<hipGraphNode_t> nodes(numNodes);
      (void)hipGraphGetNodes(capturedGraph, nodes.data(), &numNodes);
      hipGraphNodeType nodeType{};
      (void)hipGraphNodeGetType(nodes[0], &nodeType);
      unsigned int enabled = 1;
      (void)hipGraphNodeSetEnabled(exec, nodes[0], 1);
      (void)hipGraphNodeGetEnabled(exec, nodes[0], &enabled);
      hipMemsetParams msp{};
      (void)hipGraphMemsetNodeGetParams(nodes[0], &msp);
    } }

  (void)hipGraphUpload(exec, s);

  HRR_HIP_CHECK(hipGraphLaunch(exec, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));

  (void)hipGraphExecDestroy(exec);

  (void)hipUserObjectRetain(nullptr, 1);
  (void)hipUserObjectRelease(nullptr, 1);

  (void)hipGraphRetainUserObject(capturedGraph, nullptr, 1, 0);
  (void)hipGraphReleaseUserObject(capturedGraph, nullptr, 1);

  if (emptyGraph)   (void)hipGraphDestroy(emptyGraph);
  if (clonedGraph)  (void)hipGraphDestroy(clonedGraph);
  if (capturedGraph)(void)hipGraphDestroy(capturedGraph);

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

// A fork onto a second stream that never joins back makes EndCapture return hipErrorStreamCaptureUnjoined,
// which still ends the capture. The direct work that follows runs on the same stream.
static void failed_endcapture_direct_work(hipStream_t s, float* d_in, float* d_out) {
  std::vector<float> h_in(N), h_out(N, 0.0f);
  for (int i = 0; i < N; ++i) h_in[i] = static_cast<float>(i + 1);
  HRR_HIP_CHECK(hipMemcpyAsync(d_in, h_in.data(), SZ, hipMemcpyHostToDevice, s));
  hipLaunchKernelGGL(hrr_vectorScale, dim3((N + 255) / 256), dim3(256), 0, s,
                     d_in, d_out, 2.0f, N);
  HRR_HIP_CHECK(hipGetLastError());
  HRR_HIP_CHECK(hipMemcpyAsync(h_out.data(), d_out, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h_out[i] == 2.0f * static_cast<float>(i + 1));
}

TEST_CASE("Unit_HRR_FailedEndCapture_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d_in = nullptr;
  float* d_out = nullptr;
  float* d_side = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_in, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_out, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_side, SZ));
  hipStream_t s, side;
  HRR_HIP_CHECK(hipStreamCreate(&s));
  HRR_HIP_CHECK(hipStreamCreate(&side));
  hipEvent_t fork;
  HRR_HIP_CHECK(hipEventCreate(&fork));

  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  HRR_HIP_CHECK(hipEventRecord(fork, s));
  HRR_HIP_CHECK(hipStreamWaitEvent(side, fork, 0));
  HRR_HIP_CHECK(hipMemsetAsync(d_side, 0, SZ, side));
  REQUIRE(hipStreamEndCapture(s, &graph) == hipErrorStreamCaptureUnjoined);
  (void)hipGetLastError();
  REQUIRE(graph == nullptr);

  failed_endcapture_direct_work(s, d_in, d_out);

  HRR_HIP_CHECK(hipEventDestroy(fork));
  HRR_HIP_CHECK(hipStreamDestroy(side));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d_side));
  HRR_HIP_CHECK(hipFree(d_out));
  HRR_HIP_CHECK(hipFree(d_in));
}

// hipMalloc during a global-mode capture fails and invalidates it. The failed call is not recorded,
// so replay's own capture stays valid.
TEST_CASE("Unit_HRR_FailedEndCaptureInvalidated_Direct", "[.][hrr][direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  float* d_in = nullptr;
  float* d_out = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d_in, SZ));
  HRR_HIP_CHECK(hipMalloc(&d_out, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  hipGraph_t graph = nullptr;
  void* prohibited = nullptr;
  HRR_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  HRR_HIP_CHECK(hipMemsetAsync(d_out, 0, SZ, s));
  REQUIRE(hipMalloc(&prohibited, SZ) == hipErrorStreamCaptureUnsupported);
  (void)hipGetLastError();
  REQUIRE(hipStreamEndCapture(s, &graph) == hipErrorStreamCaptureInvalidated);
  (void)hipGetLastError();
  REQUIRE(graph == nullptr);

  failed_endcapture_direct_work(s, d_in, d_out);

  HRR_HIP_CHECK(hipStreamDestroy(s));
  HRR_HIP_CHECK(hipFree(d_out));
  HRR_HIP_CHECK(hipFree(d_in));
}

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

  {
    int* h_reg = static_cast<int*>(malloc(SZ));
    REQUIRE(h_reg != nullptr);
    for (int i = 0; i < N; ++i) h_reg[i] = i;
    HRR_HIP_CHECK(hipHostRegister(h_reg, SZ, hipHostRegisterDefault));
    HRR_HIP_CHECK(hipMemcpy(d, h_reg, SZ, hipMemcpyHostToDevice));
    HRR_HIP_CHECK(hipHostUnregister(h_reg));
    free(h_reg);
  }

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

  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 27);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

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

  hipModule_t mod_data = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod_data, co.data()));

  hipFunction_t fn = nullptr;
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod_data, "rtc_fill"));

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

  {
    hipModule_t mod_ex = nullptr;
    HRR_HIP_CHECK(hipModuleLoadDataEx(&mod_ex, co.data(), 0, nullptr, nullptr));
    hipFunction_t fn_ex = nullptr;
    HRR_HIP_CHECK(hipModuleGetFunction(&fn_ex, mod_ex, "rtc_fill"));
    HRR_HIP_CHECK(hipModuleUnload(mod_ex));
  }

  {
    namespace fs = std::filesystem;
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
    {
      // Regression guard for a launch from a file-loaded module. Replay
      // resolves the launch by its own code-object hash, so this does not
      // check what the hipModuleLoad event itself loaded.
      hipFunction_t fn_file = nullptr;
      HRR_HIP_CHECK(hipModuleGetFunction(&fn_file, mod_file, "rtc_fill"));
      int* d_file = nullptr;
      HRR_HIP_CHECK(hipMalloc(&d_file, SZ));
      // Under a HIP_HRR_D2H_TOL float tolerance a small integer would also pass
      // against a buffer the kernel never wrote. 0x3F800000 is 1.0f, which is
      // outside the tolerance of an unwritten buffer.
      int  val   = 0x3F800000;
      int  n     = N;
      void* args[] = { &d_file, &val, &n };
      int blocks = (N + 255) / 256;
      HRR_HIP_CHECK(hipModuleLaunchKernel(fn_file,
        blocks, 1, 1,   // grid
        256,    1, 1,   // block
        0, s, args, nullptr));
      std::vector<int> h_file(N);
      HRR_HIP_CHECK(hipMemcpyAsync(h_file.data(), d_file, SZ, hipMemcpyDeviceToHost, s));
      HRR_HIP_CHECK(hipStreamSynchronize(s));
      for (int i = 0; i < N; ++i) REQUIRE(h_file[i] == val);
      HRR_HIP_CHECK(hipFree(d_file));
    }
    HRR_HIP_CHECK(hipModuleUnload(mod_file));
    std::error_code ec;
    fs::remove(tmp_co, ec);
  }

  // hipLibraryLoadData needs a real library here, not the nullptr probes of the ModuleExtra workload.
  // The function lookup proves the recorded handle is usable. Playback no-op.
  {
    hipLibrary_t lib = nullptr;
    HRR_HIP_CHECK(hipLibraryLoadData(&lib, co.data(), nullptr, nullptr, 0,
                                     nullptr, nullptr, 0));
    if (auto hipLibraryGetModuleFn = hrr_hip_library_get_module()) {
      hipModule_t mod_lib = nullptr;
      HRR_HIP_CHECK(hipLibraryGetModuleFn(&mod_lib, lib));
      REQUIRE(mod_lib != nullptr);

      hipFunction_t fn_lib = nullptr;
      HRR_HIP_CHECK(hipModuleGetFunction(&fn_lib, mod_lib, "rtc_fill"));
      REQUIRE(fn_lib != nullptr);
    }

    HRR_HIP_CHECK(hipLibraryUnload(lib));
  }

  HRR_HIP_CHECK(hipModuleUnload(mod_data));

  HRR_HIP_CHECK(hipDeviceSynchronize());
  int* h = new int[N]();
  HRR_HIP_CHECK(hipMemcpyAsync(h, d, SZ, hipMemcpyDeviceToHost, s));
  HRR_HIP_CHECK(hipStreamSynchronize(s));
  for (int i = 0; i < N; ++i) REQUIRE(h[i] == 29);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
  delete[] h;
}

// Both events record the same code object, the device ELF, not the bundle file.
static void hrr_append_u64_le(std::vector<char>& bytes, uint64_t value) {
  for (unsigned i = 0; i < sizeof(value); ++i)
    bytes.push_back(static_cast<char>(value >> (i * 8)));
}

TEST_CASE("Unit_HRR_ModuleLoadBundle_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

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
  REQUIRE(co.size() >= 4);
  REQUIRE(std::string(co.data(), 4) == "\x7f" "ELF");

  hipModule_t mod_data = nullptr;
  HRR_HIP_CHECK(hipModuleLoadData(&mod_data, co.data()));

  hipDeviceProp_t props{};
  HRR_HIP_CHECK(hipGetDeviceProperties(&props, 0));
  constexpr uint64_t kElfOffset = 4096;
  static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  std::vector<char> bundle(kMagic, kMagic + sizeof(kMagic) - 1);
  auto add_entry = [&bundle](const std::string& id, uint64_t size) {
    hrr_append_u64_le(bundle, kElfOffset);
    hrr_append_u64_le(bundle, size);
    hrr_append_u64_le(bundle, id.size());
    bundle.insert(bundle.end(), id.begin(), id.end());
  };
  hrr_append_u64_le(bundle, 2);
  add_entry("host-x86_64-unknown-linux-gnu-", 0);
  add_entry(std::string("hipv4-amdgcn-amd-amdhsa--") + props.gcnArchName, co.size());
  REQUIRE(bundle.size() <= kElfOffset);
  bundle.resize(kElfOffset, '\0');
  bundle.insert(bundle.end(), co.begin(), co.end());

  {
    namespace fs = std::filesystem;
    auto tmp_bundle = fs::temp_directory_path() /
                      (std::string("hrr_rtc_fill_bundle_") +
                       std::to_string(reinterpret_cast<uintptr_t>(&bundle)) + ".co");
    {
      std::ofstream f(tmp_bundle, std::ios::binary);
      REQUIRE(f.is_open());
      f.write(bundle.data(), static_cast<std::streamsize>(bundle.size()));
    }
    hipModule_t mod_file = nullptr;
    HRR_HIP_CHECK(hipModuleLoad(&mod_file, tmp_bundle.string().c_str()));
    hipFunction_t fn = nullptr;
    HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod_file, "rtc_fill"));
    // 1.0f as an int, so an unwritten buffer fails replay's float-tolerant
    // D2H comparison (see the ModuleAPI workload).
    int  val   = 0x3F800000;
    int  n     = N;
    void* args[] = { &d, &val, &n };
    int blocks = (N + 255) / 256;
    HRR_HIP_CHECK(hipModuleLaunchKernel(fn,
      blocks, 1, 1,   // grid
      256,    1, 1,   // block
      0, s, args, nullptr));
    std::vector<int> h(N);
    HRR_HIP_CHECK(hipMemcpyAsync(h.data(), d, SZ, hipMemcpyDeviceToHost, s));
    HRR_HIP_CHECK(hipStreamSynchronize(s));
    for (int i = 0; i < N; ++i) REQUIRE(h[i] == val);
    HRR_HIP_CHECK(hipModuleUnload(mod_file));
    std::error_code ec;
    fs::remove(tmp_bundle, ec);
  }

  HRR_HIP_CHECK(hipModuleUnload(mod_data));
  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipStreamDestroy(s));
}

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

  const size_t min_bytes = N * sizeof(int);
  const size_t num_gran  = (min_bytes + granularity - 1) / granularity;
  const size_t alloc_sz  = num_gran * granularity;

  void* va = nullptr;
  HRR_HIP_CHECK(hipMemAddressReserve(&va, alloc_sz, 0, nullptr, 0));
  REQUIRE(va != nullptr);

  hipMemGenericAllocationHandle_t handle{};
  HRR_HIP_CHECK(hipMemCreate(&handle, alloc_sz, &prop, 0));

  {
    hipMemAllocationProp out_prop{};
    (void)hipMemGetAllocationPropertiesFromHandle(&out_prop, handle);
  }

  HRR_HIP_CHECK(hipMemMap(va, alloc_sz, 0, handle, 0));

  hipMemAccessDesc desc{};
  desc.location.type = hipMemLocationTypeDevice;
  desc.location.id   = 0;
  desc.flags         = hipMemAccessFlagsProtReadWrite;
  HRR_HIP_CHECK(hipMemSetAccess(va, alloc_sz, &desc, 1));

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

// Without the launch configuration saved in TLS, replayed kernels would launch with all-zero dimensions.
TEST_CASE("Unit_HRR_ChevronLaunch_Direct", "[.][hrr][direct]") {
  // Warm-up first HIP call so the hipMalloc below is captured (see MiscAPIs).
  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  hipStream_t s;
  HRR_HIP_CHECK(hipStreamCreate(&s));

  (void)hipDeviceSynchronize();
  (void)hipGetLastError();

  int blocks = (N + 255) / 256;
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

#ifndef _WIN32
// CLR's crash handler runs on the faulting thread's own stack, so the emergency manifest must be built off it:
// the 129 KiB buffer it needs cannot fit there. The 1 MiB guard makes an overflow fault.
static void* hrr_raise_segv(void*) {
  raise(SIGSEGV);
  return nullptr;
}

TEST_CASE("Unit_HRR_CaptureCrashSmallStack_Direct", "[.][hrr-direct]") {
  // The crash is the point; a core file is not.
  struct rlimit no_core{0, 0};
  (void)setrlimit(RLIMIT_CORE, &no_core);

  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemset(d, 0, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  pthread_attr_t attr;
  REQUIRE(pthread_attr_init(&attr) == 0);
  REQUIRE(pthread_attr_setstacksize(&attr, 64 * 1024) == 0);
  REQUIRE(pthread_attr_setguardsize(&attr, 1024 * 1024) == 0);
  pthread_t t;
  REQUIRE(pthread_create(&t, &attr, hrr_raise_segv, nullptr) == 0);
  pthread_join(t, nullptr);
  FAIL("the process outlived its own SIGSEGV");
}

// that inherited a mutex locked would block there forever. Every other child
// records hipGetLastError(), which needs no device work, and exits; the rest
// exit as soon as fork returns. The parent waits for each under a deadline, so
TEST_CASE("Unit_HRR_ForkWhileRecording_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  // Nothing below may REQUIRE before the join: unwinding past a joinable
  // std::thread terminates the process.
  std::atomic<bool> stop{false};
  std::atomic<bool> recorder_ok{true};
  std::thread recorder([&] {
    std::vector<int> h(N);
    for (int iter = 0; !stop.load(std::memory_order_relaxed); ++iter) {
      h[0] = iter;
      if (hipMemcpy(d, h.data(), SZ, hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpy(h.data(), d, SZ, hipMemcpyDeviceToHost) != hipSuccess) {
        recorder_ok = false;
        return;
      }
    }
  });

  int forked = 0, hung = 0, failed = 0;
  for (; forked < kHrrForkWhileRecordingForks; ++forked) {
    pid_t pid = fork();
    if (pid == 0) {
      if (forked % 2 == 0) (void)hipGetLastError();
      _exit(0);
    }
    if (pid < 0) break;

    int status = 0;
    pid_t got = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while ((got = waitpid(pid, &status, WNOHANG)) == 0 &&
           std::chrono::steady_clock::now() < deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (got == 0) {
      // One hang is the failure. Waiting out every later child as well would
      // run past the roundtrip case's own timeout.
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      ++hung;
      break;
    } else if (got != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ++failed;
    }
  }

  stop = true;
  recorder.join();
  HRR_HIP_CHECK(hipFree(d));

  INFO("forked " << forked << ", hung " << hung << ", failed " << failed);
  REQUIRE(recorder_ok);
  REQUIRE(hung == 0);
  REQUIRE(forked == kHrrForkWhileRecordingForks);
  REQUIRE(failed == 0);
}

// Called from the fsync() interposed for this binary. With no hook set it does nothing.
static std::atomic<void (*)(int)> g_hrr_fsync_hook{nullptr};

extern "C" void hrr_workload_fsync_hook(int fd) {
  if (auto hook = g_hrr_fsync_hook.load(std::memory_order_acquire)) hook(fd);
}

static void hrr_sleep_1ms() {
  struct timespec ts{0, 1000 * 1000};
  nanosleep(&ts, nullptr);
}

static int hrr_wait_child(pid_t pid, int seconds) {
  int status = 0;
  pid_t got = 0;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  while ((got = waitpid(pid, &status, WNOHANG)) == 0 &&
         std::chrono::steady_clock::now() < deadline)
    hrr_sleep_1ms();
  if (got == pid) return status;
  if (got == 0) {
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
  }
  return -1;
}

// The recorder loops hipGetLastError(), and the fork happens when another thread could take the events mutex
// if the writer's prepare handler released it before fork(). A handler registered before hipSetDevice runs first.
namespace {
thread_local bool t_fork_lock_recorder = false;
std::atomic<int>  g_fork_lock_fsyncs{0};    // checkpoint fsyncs the recorder made
std::atomic<bool> g_fork_lock_hold{false};  // hold the recorder in its next one
std::atomic<bool> g_fork_lock_held{false};  // the recorder is held, mutex locked
bool              g_fork_lock_held_at_fork = false;

void fork_lock_fsync_hook(int) {
  if (!t_fork_lock_recorder) return;
  g_fork_lock_fsyncs.fetch_add(1);
  if (!g_fork_lock_hold.load()) return;
  g_fork_lock_held = true;
  // Bounded, in case the parent handler never runs.
  for (int ms = 0; ms < 10000 && g_fork_lock_hold.load(); ++ms) hrr_sleep_1ms();
  g_fork_lock_held = false;
}

void fork_lock_prepare() {
  g_fork_lock_hold = true;
  for (int ms = 0; ms < 2000 && !g_fork_lock_held.load(); ++ms) hrr_sleep_1ms();
  g_fork_lock_held_at_fork = g_fork_lock_held.load();
}

void fork_lock_parent() { g_fork_lock_hold = false; }

void fork_lock_child() {
  g_fork_lock_hold = false;
  g_hrr_fsync_hook = nullptr;
}
}  // namespace

TEST_CASE("Unit_HRR_ForkWhileWriterHoldsLock_Direct", "[.][hrr-direct]") {
  REQUIRE(pthread_atfork(fork_lock_prepare, fork_lock_parent, fork_lock_child) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  g_hrr_fsync_hook = fork_lock_fsync_hook;

  // Nothing below may REQUIRE before the join.
  std::atomic<bool> stop{false};
  std::thread recorder([&] {
    t_fork_lock_recorder = true;
    while (!stop.load(std::memory_order_relaxed)) (void)hipGetLastError();
  });

  // The recorder reaches a checkpoint every 4096 records. Seeing one proves
  // fsync() is interposed, without which the case could not fail.
  for (int ms = 0; ms < 10000 && g_fork_lock_fsyncs.load() == 0; ++ms) hrr_sleep_1ms();
  const bool interposed = g_fork_lock_fsyncs.load() > 0;

  int status = -1;
  if (interposed) {
    pid_t pid = fork();
    if (pid == 0) {
      (void)hipGetLastError();
      _exit(0);
    }
    if (pid > 0) status = hrr_wait_child(pid, 10);
  }

  stop = true;
  recorder.join();
  g_hrr_fsync_hook = nullptr;

  INFO("checkpoint fsyncs " << g_fork_lock_fsyncs.load() << ", child wait status "
                            << status);
  REQUIRE(interposed);
  REQUIRE_FALSE(g_fork_lock_held_at_fork);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

// A crash callback skipped for a thread the child lacks would skip its own manifest. The parent's callback
// is released once the child has gone, and the parent dies of its SIGSEGV.
// Unit_HRR_CaptureCrashDuringFork checks both manifests.
namespace {
std::atomic<bool> g_crash_fork_in_manifest{false};
std::atomic<bool> g_crash_fork_release{false};

bool hrr_fd_is_manifest(int fd) {
  char link[32] = "/proc/self/fd/";
  size_t n = strlen(link);
  char digits[12];
  size_t k = 0;
  do {
    digits[k++] = static_cast<char>('0' + fd % 10);
    fd /= 10;
  } while (fd > 0);
  while (k > 0) link[n++] = digits[--k];
  link[n] = '\0';
  char target[4096];
  const ssize_t len = readlink(link, target, sizeof(target));
  static constexpr char kSuffix[] = "/manifest.json";
  constexpr ssize_t kSuffixLen = sizeof(kSuffix) - 1;
  return len >= kSuffixLen && memcmp(target + len - kSuffixLen, kSuffix, kSuffixLen) == 0;
}

void crash_fork_fsync_hook(int fd) {
  if (!hrr_fd_is_manifest(fd)) return;
  g_crash_fork_in_manifest = true;
  for (int ms = 0; ms < 60000 && !g_crash_fork_release.load(); ++ms) hrr_sleep_1ms();
}
}  // namespace

TEST_CASE("Unit_HRR_CaptureCrashDuringFork_Direct", "[.][hrr-direct]") {
  // The crashes are the point; core files are not.
  struct rlimit no_core{0, 0};
  (void)setrlimit(RLIMIT_CORE, &no_core);

  HRR_HIP_CHECK(hipSetDevice(0));
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));
  HRR_HIP_CHECK(hipMemset(d, 0, SZ));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  g_hrr_fsync_hook = crash_fork_fsync_hook;
  std::thread([] {
    for (int ms = 0; ms < 30000 && !g_crash_fork_in_manifest.load(); ++ms) hrr_sleep_1ms();
    if (g_crash_fork_in_manifest.load()) {
      pid_t pid = fork();
      if (pid == 0) {
        g_hrr_fsync_hook = nullptr;
        (void)hipGetLastError();
        raise(SIGABRT);
        _exit(1);
      }
      if (pid > 0) (void)hrr_wait_child(pid, 30);
    }
    g_crash_fork_release = true;
  }).detach();

  raise(SIGSEGV);
  FAIL("the process outlived its own SIGSEGV");
}

// The child must not run the fat-binary destructors and runtime teardown that follow.
// In the parent the handler does nothing.
static void (*g_hrr_after_capture_shutdown)() = nullptr;

static void hrr_after_capture_shutdown() {
  if (!g_hrr_after_capture_shutdown) return;
  g_hrr_after_capture_shutdown();
  _exit(0);
}

// Interposed strlen() for this test binary, for the same reason as fsync(): CLR resolves it here first.
// The capture writer calls it with the API name once note_unreplayable() holds its mutex.
// With no hook set it is a plain strlen.
static std::atomic<void (*)(const char*)> g_hrr_strlen_hook{nullptr};

extern "C" size_t strlen(const char* s) noexcept {
  if (auto hook = g_hrr_strlen_hook.load(std::memory_order_acquire)) hook(s);
  // volatile, so the compiler cannot turn the loop back into a strlen() call.
  const volatile char* p = s;
  while (*p) ++p;
  return static_cast<size_t>(p - static_cast<const volatile char*>(s));
}

// Unit_HRR_ForkWhileNotingUnreplayable_Direct
// Forks while another thread is inside note_unreplayable(), under the mutex of
namespace {
std::atomic<bool> g_unreplayable_hold{false};      // hold the next match
std::atomic<bool> g_unreplayable_held{false};      // the recorder is held
std::atomic<bool> g_unreplayable_release{false};   // let it go
std::atomic<bool> g_unreplayable_ran_out{false};   // the hold ran out first

void unreplayable_strlen_hook(const char* s) {
  static constexpr char kApi[] = "hipUserObjectCreate";
  const volatile char* p = s;
  for (size_t i = 0; i < sizeof(kApi); ++i)
    if (p[i] != kApi[i]) return;
  if (!g_unreplayable_hold.exchange(false)) return;
  g_unreplayable_held = true;
  int ms = 0;
  for (; ms < 2000 && !g_unreplayable_release.load(); ++ms) hrr_sleep_1ms();
  g_unreplayable_ran_out = ms == 2000;
}

int  g_unreplayable_payload = 0;
void unreplayable_destroy(void*) {}
}  // namespace

TEST_CASE("Unit_HRR_ForkWhileNotingUnreplayable_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  g_hrr_strlen_hook = unreplayable_strlen_hook;
  g_unreplayable_hold = true;

  // Nothing below may REQUIRE before the join.
  hipError_t created = hipErrorUnknown;
  std::thread recorder([&] {
    hipUserObject_t obj = nullptr;
    created = hipUserObjectCreate(&obj, &g_unreplayable_payload, unreplayable_destroy, 1,
                                  hipUserObjectNoDestructorSync);
    if (created == hipSuccess) (void)hipUserObjectRelease(obj, 1);
  });

  for (int ms = 0; ms < 10000 && !g_unreplayable_held.load(); ++ms) hrr_sleep_1ms();
  const bool held = g_unreplayable_held.load();

  int status = -1;
  if (held) {
    pid_t pid = fork();
    if (pid == 0) {
      g_hrr_strlen_hook = nullptr;
      g_hrr_after_capture_shutdown = [] {};
      (void)hipGetLastError();  // opens the child's archive
      exit(0);
    }
    g_unreplayable_release = true;
    if (pid > 0) status = hrr_wait_child(pid, 10);
  }

  g_unreplayable_release = true;
  recorder.join();
  g_hrr_strlen_hook = nullptr;

  INFO("hipUserObjectCreate " << created << ", child wait status " << status);
  REQUIRE(created == hipSuccess);
  REQUIRE(held);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
  // Clear if fork() did not wait for the recorder to leave the mutex.
  REQUIRE(g_unreplayable_ran_out.load());
}

TEST_CASE("Unit_HRR_ForkedChildStaysIncomplete_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  hipStream_t stream = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&stream));
  void* dst = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dst, 1024));
  std::vector<char> host(1024, 1);
  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(host.data(), 256, 256, 4);
  p.dstPtr = make_hipPitchedPtr(dst, 256, 256, 4);
  p.extent = make_hipExtent(256, (size_t{1} << 56) + 1, 1);  // 2^64 + 256 bytes
  p.kind = hipMemcpyHostToDevice;
  HRR_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  const pid_t pid = fork();
  if (pid == 0) {
    g_hrr_after_capture_shutdown = [] {};
    if (hipMemcpy3DAsync(&p, stream) != hipSuccess) _exit(3);
    (void)hipGetLastError();  // opens the child's archive
    exit(0);
  }
  const int status = pid > 0 ? hrr_wait_child(pid, 30) : -1;
  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HRR_HIP_CHECK(hipGraphDestroy(graph));
  HRR_HIP_CHECK(hipFree(dst));
  HRR_HIP_CHECK(hipStreamDestroy(stream));
  INFO("child wait status " << status);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

static void hrr_push_pop_launch_config() {
  dim3 grid, block;
  size_t shared = 0;
  hipStream_t stream = nullptr;
  (void)__hipPushCallConfiguration(dim3(1), dim3(1), 0, nullptr);
  (void)__hipPopCallConfiguration(&grid, &block, &shared, &stream);
}

// A forked child whose archive was not opened before its capture shutdown has nothing to finalize.
// Its fat-binary destructors run after that shutdown and still record through the compiler dispatch table.
// A record there must not open an archive.
TEST_CASE("Unit_HRR_ForkedChildRecordsAfterShutdown_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();

  pid_t pid = fork();
  if (pid == 0) {
    g_hrr_after_capture_shutdown = hrr_push_pop_launch_config;
    exit(0);
  }
  REQUIRE(pid > 0);
  const int status = hrr_wait_child(pid, 30);
  INFO("child wait status " << status);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

// The child must not open an archive on its first record: nothing in the child would finalize it.
// The parent forks after its shutdown, and the child records through the compiler dispatch table.
TEST_CASE("Unit_HRR_ForkAfterCaptureShutdown_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();

  g_hrr_after_capture_shutdown = [] {
    pid_t pid = fork();
    if (pid == 0) {
      hrr_push_pop_launch_config();
      _exit(0);
    }
    const int status = pid > 0 ? hrr_wait_child(pid, 30) : -1;
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) _exit(3);
  };
}

static std::atomic<void (*)(const char*)> g_hrr_mkdir_hook{nullptr};

extern "C" int mkdir(const char* path, mode_t mode) noexcept {
  if (auto hook = g_hrr_mkdir_hook.load(std::memory_order_acquire)) hook(path);
  return static_cast<int>(syscall(SYS_mkdirat, AT_FDCWD, path, mode));
}

extern "C" int mkdirat(int dirfd, const char* path, mode_t mode) noexcept {
  if (auto hook = g_hrr_mkdir_hook.load(std::memory_order_acquire)) hook(path);
  return static_cast<int>(syscall(SYS_mkdirat, dirfd, path, mode));
}

// The child's archive open can race its capture shutdown on another thread. The shutdown must wait for that open,
// or a manifest or trailer is written for an archive that is not there yet. A second thread holds the open here.
namespace {
std::atomic<bool> g_child_open_held{false};

void child_open_mkdir_hook(const char* path) {
  static constexpr char kSuffix[] = "code_objects";
  constexpr size_t kSuffixLen = sizeof(kSuffix) - 1;
  const size_t len = strlen(path);
  if (len < kSuffixLen || memcmp(path + len - kSuffixLen, kSuffix, kSuffixLen) != 0) return;
  if (g_child_open_held.exchange(true)) return;
  for (int ms = 0; ms < 2000; ++ms) hrr_sleep_1ms();
}
}  // namespace

TEST_CASE("Unit_HRR_ShutdownWhileChildOpensArchive_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();

  pid_t pid = fork();
  if (pid == 0) {
    g_hrr_after_capture_shutdown = [] {};
    g_hrr_mkdir_hook = child_open_mkdir_hook;
    std::thread([] { (void)hipGetLastError(); }).detach();
    for (int ms = 0; ms < 10000 && !g_child_open_held.load(); ++ms) hrr_sleep_1ms();
    if (!g_child_open_held.load()) _exit(4);
    exit(0);
  }
  REQUIRE(pid > 0);
  const int status = hrr_wait_child(pid, 30);
  INFO("child wait status " << status);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}

static std::atomic<void (*)(const char*)> g_hrr_rename_hook{nullptr};

extern "C" int rename(const char* oldpath, const char* newpath) noexcept {
  if (auto hook = g_hrr_rename_hook.load(std::memory_order_acquire)) hook(newpath);
  return static_cast<int>(syscall(SYS_renameat2, AT_FDCWD, oldpath, AT_FDCWD, newpath, 0));
}

extern "C" int renameat(int olddirfd, const char* oldpath, int newdirfd,
                        const char* newpath) noexcept {
  if (auto hook = g_hrr_rename_hook.load(std::memory_order_acquire)) hook(newpath);
  return static_cast<int>(syscall(SYS_renameat2, olddirfd, oldpath, newdirfd, newpath, 0));
}

// The shutdown must not land a record after the trailer. A second thread is held inside hipGetLastError()
// until the shutdown reaches its first manifest, and the shutdown is held there until that call returns.
namespace {
std::atomic<bool> g_late_in_call{false};
std::atomic<bool> g_shutdown_held{false};
std::atomic<bool> g_late_recorded{false};

void late_record_clock_hook() {
  t_hrr_clock_hook = nullptr;
  g_late_in_call = true;
  for (int ms = 0; ms < 10000 && !g_shutdown_held.load(); ++ms) hrr_sleep_1ms();
}

void shutdown_rename_hook(const char* path) {
  static constexpr char kSuffix[] = "/manifest.json";
  constexpr size_t kSuffixLen = sizeof(kSuffix) - 1;
  const size_t len = strlen(path);
  if (len < kSuffixLen || memcmp(path + len - kSuffixLen, kSuffix, kSuffixLen) != 0) return;
  if (g_shutdown_held.exchange(true)) return;
  for (int ms = 0; ms < 10000 && !g_late_recorded.load(); ++ms) hrr_sleep_1ms();
}
}  // namespace

TEST_CASE("Unit_HRR_RecordAfterCaptureShutdown_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();

  g_hrr_after_capture_shutdown = [] {
    g_hrr_rename_hook = nullptr;
    if (!g_late_recorded.load()) _exit(4);
  };
  g_hrr_rename_hook = shutdown_rename_hook;
  std::thread([] {
    t_hrr_clock_hook = late_record_clock_hook;
    (void)hipGetLastError();
    g_late_recorded = true;
  }).detach();
  for (int ms = 0; ms < 10000 && !g_late_in_call.load(); ++ms) hrr_sleep_1ms();
}

static std::atomic<void (*)()> g_hrr_mutex_lock_hook{nullptr};

extern "C" int pthread_mutex_lock(pthread_mutex_t* mutex) noexcept {
  if (auto hook = g_hrr_mutex_lock_hook.load(std::memory_order_acquire)) hook();
  using Fn = int (*)(pthread_mutex_t*);
  static std::atomic<Fn> real{nullptr};
  Fn fn = real.load(std::memory_order_relaxed);
  if (fn == nullptr) {
    fn = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "pthread_mutex_lock"));
    real.store(fn, std::memory_order_relaxed);
  }
  return fn(mutex);
}

// A thread that forks between the flush and close() must not leave the child to open an archive on its first record:
// the shutdown that would finalize it runs on the parent's exiting thread.
namespace {
std::atomic<bool> g_flush_done{false};
std::atomic<bool> g_forked_after_flush{false};
std::atomic<bool> g_close_hold_ran_out{false};
std::atomic<int>  g_flush_fork_status{-1};
thread_local bool t_hold_next_lock = false;

void hold_close_lock_hook() {
  if (!t_hold_next_lock) return;
  t_hold_next_lock = false;
  g_hrr_mutex_lock_hook = nullptr;
  g_flush_done = true;
  int ms = 0;
  for (; ms < 10000 && !g_forked_after_flush.load(); ++ms) hrr_sleep_1ms();
  // fork() takes the lock close() is about to take, so a fork that returned
  // while this waited ran before close().
  g_close_hold_ran_out = ms == 10000;
}

// flush() publishes the root manifest last, after the trailer and the
// process manifest. Nothing after it on this thread takes a mutex until
// close() does.
void root_manifest_rename_hook(const char* path) {
  static constexpr char kSuffix[] = "/manifest.json";
  constexpr size_t kSuffixLen = sizeof(kSuffix) - 1;
  const size_t len = strlen(path);
  if (len < kSuffixLen || memcmp(path + len - kSuffixLen, kSuffix, kSuffixLen) != 0) return;
  const std::string dir = std::filesystem::path(path).parent_path().filename().string();
  if (dir.rfind("pid-", 0) == 0) return;
  g_hrr_rename_hook = nullptr;
  t_hold_next_lock = true;
  g_hrr_mutex_lock_hook = hold_close_lock_hook;
}
}  // namespace

TEST_CASE("Unit_HRR_ForkBetweenCaptureFlushAndClose_Direct", "[.][hrr-direct]") {
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();

  g_hrr_after_capture_shutdown = [] {
    if (!g_flush_done.load() || g_close_hold_ran_out.load()) _exit(4);
    const int status = g_flush_fork_status.load();
    if (status == -1 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) _exit(3);
  };
  g_hrr_rename_hook = root_manifest_rename_hook;
  std::thread([] {
    for (int ms = 0; ms < 20000 && !g_flush_done.load(); ++ms) hrr_sleep_1ms();
    if (!g_flush_done.load()) return;
    pid_t pid = fork();
    if (pid == 0) {
      hrr_push_pop_launch_config();
      _exit(0);
    }
    g_flush_fork_status = pid > 0 ? hrr_wait_child(pid, 30) : -1;
    g_forked_after_flush = true;
  }).detach();
}

// The shutdown must wait for a crash callback that holds the process number, or another thread could reuse it
// and the callback write its manifest into that directory.
namespace {
std::atomic<bool> g_crash_close_hold{false};
std::atomic<bool> g_crash_close_held{false};
std::atomic<bool> g_crash_close_exiting{false};

void crash_close_fsync_hook(int fd) {
  if (!hrr_fd_is_manifest(fd) || !g_crash_close_hold.exchange(false)) return;
  g_crash_close_held = true;
  for (int ms = 0; ms < 10000 && !g_crash_close_exiting.load(); ++ms) hrr_sleep_1ms();
  for (int ms = 0; ms < 1000; ++ms) hrr_sleep_1ms();
  if (g_crash_close_exiting.load()) {
    static constexpr char kLine[] = "capture shutdown started while the crash callback was held\n";
    (void)!write(2, kLine, sizeof(kLine) - 1);
  }
  g_crash_close_held = false;
}

void crash_close_exiting() { g_crash_close_exiting = true; }

void crash_close_after_shutdown() {
  if (g_crash_close_held.load()) _exit(3);
  for (;;) pause();  // for the SIGSEGV the callback raises again
}
}  // namespace

TEST_CASE("Unit_HRR_CloseDuringCrashManifest_Direct", "[.][hrr-direct]") {
  // The crash is the point; a core file is not.
  struct rlimit no_core{0, 0};
  (void)setrlimit(RLIMIT_CORE, &no_core);
  REQUIRE(std::atexit(hrr_after_capture_shutdown) == 0);
  HRR_HIP_CHECK(hipSetDevice(0));
  (void)hipGetLastError();
  REQUIRE(std::atexit(crash_close_exiting) == 0);

  g_hrr_after_capture_shutdown = crash_close_after_shutdown;
  g_hrr_fsync_hook = crash_close_fsync_hook;
  g_crash_close_hold = true;
  std::thread([] { raise(SIGSEGV); }).detach();
  for (int ms = 0; ms < 10000 && !g_crash_close_held.load(); ++ms) hrr_sleep_1ms();
}

// writing thread, which a forked child does not have. Here the rename hook
// holds a second thread inside a blob write, with that directory open, for two
// seconds, and the main thread forks meanwhile. The child exits 3 if it holds a
namespace {
std::atomic<bool> g_blob_write_hold{false};
std::atomic<bool> g_blob_write_held{false};

void blob_write_rename_hook(const char* path) {
  static constexpr char kSuffix[] = ".blob";
  constexpr size_t kSuffixLen = sizeof(kSuffix) - 1;
  const size_t len = strlen(path);
  if (len < kSuffixLen || memcmp(path + len - kSuffixLen, kSuffix, kSuffixLen) != 0) return;
  if (!g_blob_write_hold.exchange(false)) return;
  g_blob_write_held = true;
  for (int ms = 0; ms < 2000; ++ms) hrr_sleep_1ms();
  g_blob_write_held = false;
}

int hrr_fds_inside(const char* dir) {
  const size_t dir_len = strlen(dir);
  int found = 0;
  for (int fd = 0; fd < 1024; ++fd) {
    char link[32] = "/proc/self/fd/";
    size_t n = strlen(link);
    char digits[12];
    size_t k = 0;
    int v = fd;
    do {
      digits[k++] = static_cast<char>('0' + v % 10);
      v /= 10;
    } while (v > 0);
    while (k > 0) link[n++] = digits[--k];
    link[n] = '\0';
    char target[4096];
    const ssize_t len = readlink(link, target, sizeof(target));
    if (len > static_cast<ssize_t>(dir_len) && memcmp(target, dir, dir_len) == 0 &&
        target[dir_len] == '/')
      ++found;
  }
  return found;
}
}  // namespace

TEST_CASE("Unit_HRR_ForkDuringBlobWrite_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  const char* base = std::getenv("HIP_HRR_CAPTURE_OUTPUT");
  REQUIRE(base != nullptr);
  const std::string blobs_path =
      std::string(base) + "/pid-" + std::to_string(::getpid()) + "/blobs";
  char blobs[4096];
  REQUIRE(::realpath(blobs_path.c_str(), blobs) != nullptr);
  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, SZ));

  g_hrr_rename_hook = blob_write_rename_hook;
  g_blob_write_hold = true;
  // Nothing below may REQUIRE before the join.
  std::thread writer([d] {
    const std::vector<int> h(N, 0x5eed);
    (void)hipMemcpy(d, h.data(), SZ, hipMemcpyHostToDevice);
  });
  for (int ms = 0; ms < 10000 && !g_blob_write_held.load(); ++ms) hrr_sleep_1ms();
  const bool held = g_blob_write_held.load();
  int status = -1;
  if (held) {
    const pid_t pid = fork();
    if (pid == 0) _exit(hrr_fds_inside(blobs) == 0 ? 0 : 3);
    if (pid > 0) status = hrr_wait_child(pid, 10);
  }
  writer.join();
  g_hrr_rename_hook = nullptr;
  HRR_HIP_CHECK(hipFree(d));

  INFO("child wait status " << status);
  REQUIRE(held);
  REQUIRE(status != -1);
  REQUIRE(WIFEXITED(status));
  REQUIRE(WEXITSTATUS(status) == 0);
}
#endif  // !_WIN32

// A rejected copy, with an extent far larger than both buffers, through each of the four hipMemcpy3D entry points.
// Each has its own capture success gate. Unit_HRR_FailedMemcpy3DNotRecorded checks that capture does not record them.
TEST_CASE("Unit_HRR_FailedMemcpy3D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipPitchedPtr dst{};
  HRR_HIP_CHECK(hipMalloc3D(&dst, make_hipExtent(64, 4, 1)));
  std::vector<char> host(4096, 1);

  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(host.data(), size_t{1} << 20, size_t{1} << 20, size_t{1} << 12);
  p.dstPtr = dst;
  p.extent = make_hipExtent(size_t{1} << 20, size_t{1} << 12, 1);  // 4 GiB
  p.kind = hipMemcpyHostToDevice;
  REQUIRE(hipMemcpy3D(&p) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipMemcpy3DAsync(&p, nullptr) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipMemcpy3D_spt(&p) == hipErrorInvalidValue);
  (void)hipGetLastError();
  REQUIRE(hipMemcpy3DAsync_spt(&p, hipStreamPerThread) == hipErrorInvalidValue);
  (void)hipGetLastError();

  HRR_HIP_CHECK(hipFree(dst.ptr));
}

// A hipModuleLaunchKernel of a hipRTC kernel whose name is 70,000 characters, longer than the uint16_t wire name length.
// The launch succeeds and its output is checked here; Unit_HRR_LongKernelNameNotRecorded checks the capture.
TEST_CASE("Unit_HRR_LongKernelName_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr int    LN  = 256;
  constexpr size_t LSZ = LN * sizeof(int);

  const std::string name = "k" + std::string(70000, 'a');
  const std::string src = "extern \"C\" __global__ void " + name +
                          "(int* out, int val, int n) {\n"
                          "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                          "  if (i < n) out[i] = val;\n"
                          "}\n";
  hiprtcProgram prog = nullptr;
  HRR_HIPRTC_CHECK(hiprtcCreateProgram(&prog, src.c_str(), "long_name.hip",
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
  HRR_HIP_CHECK(hipModuleGetFunction(&fn, mod, name.c_str()));

  int* d = nullptr;
  HRR_HIP_CHECK(hipMalloc(&d, LSZ));
  int   val = 77;
  int   n   = LN;
  void* args[] = { &d, &val, &n };
  HRR_HIP_CHECK(hipModuleLaunchKernel(fn, 1, 1, 1, LN, 1, 1, 0, nullptr, args, nullptr));
  HRR_HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> h(LN, 0);
  HRR_HIP_CHECK(hipMemcpy(h.data(), d, LSZ, hipMemcpyDeviceToHost));
  for (int i = 0; i < LN; ++i) REQUIRE(h[i] == 77);

  HRR_HIP_CHECK(hipFree(d));
  HRR_HIP_CHECK(hipModuleUnload(mod));
}

// A host-to-device hipMemcpy3DAsync in a stream capture whose width * height *
// depth overflows size_t. The runtime accepts it: it validates with the same
// wrapping product and only adds a graph node, which is never launched.
TEST_CASE("Unit_HRR_OverflowingMemcpy3D_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipStream_t stream = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&stream));
  void* dst = nullptr;
  HRR_HIP_CHECK(hipMalloc(&dst, 1024));
  std::vector<char> host(1024, 1);

  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(host.data(), 256, 256, 4);
  p.dstPtr = make_hipPitchedPtr(dst, 256, 256, 4);
  p.extent = make_hipExtent(256, (size_t{1} << 56) + 1, 1);  // 2^64 + 256 bytes
  p.kind = hipMemcpyHostToDevice;
  HRR_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HRR_HIP_CHECK(hipMemcpy3DAsync(&p, stream));
  hipGraph_t graph = nullptr;
  HRR_HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HRR_HIP_CHECK(hipGraphDestroy(graph));

  HRR_HIP_CHECK(hipFree(dst));
  HRR_HIP_CHECK(hipStreamDestroy(stream));
}

// One rejected call per hand-written shim that inlines a struct, next to accepted calls on the same pool and stream.
// One accepted call sets a 4-byte reuse-policy attribute. Unit_HRR_FailedShimCallsNotRecorded checks capture.
TEST_CASE("Unit_HRR_FailedShimCalls_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  hipMemPoolProps props{};
  props.allocType = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = 0;
  hipMemPool_t pool = nullptr;
  HRR_HIP_CHECK(hipMemPoolCreate(&pool, &props));
  // The reuse policy is a 4-byte int. The word after it is a sentinel that an
  // 8-byte copy in capture would record as the value's high half.
  int32_t opportunistic[2] = {1, 0x5a5a5a5a};
  HRR_HIP_CHECK(
      hipMemPoolSetAttribute(pool, hipMemPoolReuseAllowOpportunistic, &opportunistic[0]));
  hipStream_t stream = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&stream));

  hipMemPoolProps bad_props = props;
  bad_props.allocType = hipMemAllocationTypeInvalid;
  hipMemPool_t bad_pool = nullptr;
  REQUIRE(hipMemPoolCreate(&bad_pool, &bad_props) == hipErrorInvalidValue);

  uint64_t used_high = 1;  // only 0 is accepted
  REQUIRE(hipMemPoolSetAttribute(pool, hipMemPoolAttrUsedMemHigh, &used_high) ==
          hipErrorInvalidValue);

  hipMemAccessDesc bad_access{};  // location.type is hipMemLocationTypeInvalid
  bad_access.flags = hipMemAccessFlagsProtReadWrite;
  REQUIRE(hipMemPoolSetAccess(pool, &bad_access, 1) == hipErrorInvalidValue);
  REQUIRE(hipMemSetAccess(nullptr, 4096, &bad_access, 1) == hipErrorInvalidValue);

  HIP_ARRAY_DESCRIPTOR array_desc{};
  array_desc.Width = 64;
  array_desc.Height = 64;
  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.NumChannels = 3;  // only 1, 2 or 4
  hipArray_t array = nullptr;
  REQUIRE(hipArrayCreate(&array, &array_desc) == hipErrorInvalidValue);

  HIP_ARRAY3D_DESCRIPTOR array3d_desc{};
  array3d_desc.Width = 16;
  array3d_desc.Height = 16;
  array3d_desc.Depth = 16;
  array3d_desc.Format = HIP_AD_FORMAT_FLOAT;
  array3d_desc.NumChannels = 3;
  REQUIRE(hipArray3DCreate(&array, &array3d_desc) == hipErrorInvalidValue);

  hipStreamAttrValue sync{};
  sync.syncPolicy = static_cast<hipSynchronizationPolicy>(hipSyncPolicyBlockingSync + 1);
  REQUIRE(hipStreamSetAttribute(stream, hipStreamAttributeSynchronizationPolicy, &sync) ==
          hipErrorInvalidValue);

  hipMemAllocationProp alloc_prop{};  // type is hipMemAllocationTypeInvalid
  alloc_prop.location.type = hipMemLocationTypeDevice;
  alloc_prop.location.id = 0;
  size_t granularity = 0;
  REQUIRE(hipMemGetAllocationGranularity(&granularity, &alloc_prop,
                                         hipMemAllocationGranularityMinimum) ==
          hipErrorInvalidValue);
  (void)hipGetLastError();

  HRR_HIP_CHECK(hipStreamDestroy(stream));
  HRR_HIP_CHECK(hipMemPoolDestroy(pool));
}

TEST_CASE("Unit_HRR_ExtMemcpyBatch_Direct", "[.][hrr-direct]") {
  HRR_HIP_CHECK(hipSetDevice(0));
  constexpr size_t kBytes = 4096;
  unsigned char *src = nullptr, *dst0 = nullptr, *dst1 = nullptr;
  HRR_HIP_CHECK(hipMalloc(&src, kBytes));
  HRR_HIP_CHECK(hipMalloc(&dst0, kBytes));
  HRR_HIP_CHECK(hipMalloc(&dst1, kBytes));
  std::vector<unsigned char> h(kBytes);
  for (size_t i = 0; i < kBytes; ++i) h[i] = static_cast<unsigned char>(i * 7 + 1);
  HRR_HIP_CHECK(hipMemcpy(src, h.data(), kBytes, hipMemcpyHostToDevice));
  HRR_HIP_CHECK(hipMemset(dst0, 0, kBytes));
  HRR_HIP_CHECK(hipMemset(dst1, 0, kBytes));
  hipStream_t stream = nullptr;
  HRR_HIP_CHECK(hipStreamCreate(&stream));

  void* dsts[] = {dst0, dst1 + 128};
  void* srcs[] = {src + 64, src};
  size_t sizes[] = {1024, 2048};
  size_t sizes_dst[] = {0, 0};
  hipExtMemcpyAttributes attrs[2] = {};
  attrs[0].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].srcAccessOrder = hipMemcpySrcAccessOrderStream;
  attrs[1].flags = hipMemcpyFlagExtPreferLinear;
  size_t attr_idxs[] = {0, 1};
  HRR_HIP_CHECK(hipExtMemcpyBatchAsync(dsts, srcs, sizes, sizes_dst, nullptr, nullptr, 2,
                                       attrs, attr_idxs, 2, stream));
  HRR_HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<unsigned char> out0(kBytes), out1(kBytes);
  HRR_HIP_CHECK(hipMemcpy(out0.data(), dst0, kBytes, hipMemcpyDeviceToHost));
  HRR_HIP_CHECK(hipMemcpy(out1.data(), dst1, kBytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < kBytes; ++i) {
    REQUIRE(out0[i] == (i < 1024 ? h[i + 64] : 0));
    REQUIRE(out1[i] == (i >= 128 && i < 128 + 2048 ? h[i - 128] : 0));
  }

  HRR_HIP_CHECK(hipMemcpy(dst0, out0.data(), kBytes, hipMemcpyHostToDevice));
  void* swap_dsts[] = {dst0};
  void* swap_srcs[] = {src};
  size_t swap_sizes[] = {2048};
  size_t swap_sizes_dst[] = {1024};
  hipExtMemcpyAttributes swap_attr{};
  swap_attr.srcAccessOrder = hipMemcpySrcAccessOrderStream;
  swap_attr.flags = hipMemcpyFlagExtOpSwap;
  size_t swap_attr_idx[] = {0};
  const hipError_t swap_err = hipExtMemcpyBatchAsync(swap_dsts, swap_srcs, swap_sizes,
                                                     swap_sizes_dst, nullptr, nullptr, 1,
                                                     &swap_attr, swap_attr_idx, 1, stream);
  if (swap_err == hipErrorNotSupported) {
    (void)hipGetLastError();
  } else {
    HRR_HIP_CHECK(swap_err);
    HRR_HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<unsigned char> swapped_dst(kBytes), swapped_src(kBytes);
    HRR_HIP_CHECK(hipMemcpy(swapped_dst.data(), dst0, kBytes, hipMemcpyDeviceToHost));
    HRR_HIP_CHECK(hipMemcpy(swapped_src.data(), src, kBytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < kBytes; ++i) {
      REQUIRE(swapped_dst[i] == (i < 2048 ? h[i] : out0[i]));
      REQUIRE(swapped_src[i] == (i < 1024 ? out0[i] : h[i]));
    }
  }

  HRR_HIP_CHECK(hipStreamDestroy(stream));
  HRR_HIP_CHECK(hipFree(src));
  HRR_HIP_CHECK(hipFree(dst0));
  HRR_HIP_CHECK(hipFree(dst1));
}
