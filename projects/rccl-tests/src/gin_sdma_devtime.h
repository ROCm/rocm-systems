/*************************************************************************
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared device-side timing scaffold for the GIN-SDMA collectives (rocSHMEM
// methodology). A collective provides (1) a persistent "*TimedKernel" that runs
// skip+loop back-to-back collective bodies in ONE launch and brackets only the
// timed region with the GPU fixed-frequency wall clock (wall_clock64()) -- every
// CTA stamps start_time[blockIdx.x] at i==skip and end_time[blockIdx.x] after the
// final iteration -- and (2) a thin "*DeviceTime" hook that calls gin_devtime::measure
// with a lambda that launches that kernel. measure() owns everything mechanical and
// identical across collectives: sample each GPU's wall-clock rate, allocate the
// per-CTA start/end stamp buffers, run the launch, reduce the grid busy window
// (min(start)..max(end) over CTAs) and the slowest rank (MPI MAX), and return the
// per-iteration device latency in microseconds. This is the pure device-function
// execution time -- it excludes host launch, stream/graph, and teardown overhead.
//
// The reported span is driven by BenchTime (see common.cu) via --device_timing:
// 1=augment (each hook prints its own extra "#[*-devtime]" line next to the graph
// numbers), 2=device-time-only (the hook returns per-iteration seconds via
// outDeltaSec and BenchTime reports it AS the out-of-place metric).
#ifndef GIN_SDMA_DEVTIME_H_
#define GIN_SDMA_DEVTIME_H_

#include <vector>
#include "common.h"
#include "gin_sdma_devtime_host.h"

namespace gin_devtime {

// Shared skip/loop tier selection for gin_devtime hooks (per-peer/per-rank bytes).
static inline void resolveLoopSkip(size_t perPeerBytes, int& loop, int& skip) {
  loop = devtimeLoop;
  skip = devtimeSkip < 0 ? 0 : devtimeSkip;
  if (devtimeLoopLarge > 0 && perPeerBytes >= (size_t)64 * 1024 * 1024) {
    loop = devtimeLoopLarge;
    if (devtimeSkipLarge >= 0) skip = devtimeSkipLarge;
    else skip = (skip < 1) ? skip : 1;
  } else if (devtimeLoopMid > 0 && perPeerBytes >= (size_t)8 * 1024 * 1024) {
    loop = devtimeLoopMid;
    if (devtimeSkipMid >= 0) skip = devtimeSkipMid;
    else skip = (skip < 2) ? skip : 2;
  }
}

// GPU fixed-frequency wall-clock rate (kHz) for a specific device ordinal. Sampled
// per-GPU (rather than once from the current device) so each GPU's cycle delta is
// converted with its own rate on mixed-clock nodes.
static inline bool wallClockRateKhz(int dev, int* outKhz) {
  int khz = 0;
  if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0)
    return false;
  *outKhz = khz;
  return true;
}

// Run the device-timing measurement. `launch` is a callable
//   void(int i, long long* d_start, long long* d_end)
// that launches the collective's persistent timed kernel on GPU i's stream with a
// <<<gridCtas, ...>>> grid (loop/skip captured by the caller), writing per-CTA
// start/end wall-clock stamps into d_start/d_end. measure() sets the device before
// each launch, sizes the stamp buffers to gridCtas, synchronizes, reduces
// min(start)..max(end) over CTAs into a per-iteration latency (divided by loop),
// takes the slowest local GPU, then MPI-MAX across ranks. Cross-rank barriers before
// and after isolate the timing run from the surrounding measurement flow so it can
// never race the next size's buffer re-init. Returns the per-iteration latency (us)
// in *outUs.
template <typename LaunchFn>
static inline testResult_t measure(struct threadArgs* args, int gridCtas, int loop,
                                   LaunchFn&& launch, double* outUs) {
  if (gridCtas < 1) gridCtas = 1;
  double localMaxUs = 0.0;

  std::vector<long long*> d_start(args->nGpus, nullptr);
  std::vector<long long*> d_end(args->nGpus, nullptr);

  Barrier(args);

  // Pass 1: allocate stamp buffers on every local GPU, then launch the timed
  // kernel on every local GPU, before synchronizing any of them. Malloc-first
  // keeps the error path safe: if GPU i's hipMalloc fails, no kernel is in
  // flight yet. Launch-all-then-sync avoids the -g N>1 device-barrier deadlock
  // (GPU 0 would spin waiting for GPUs never launched). Mirrors startColl()/
  // completeColl().
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(hipSetDevice(args->gpus[i]));
    CUDACHECK(hipMalloc(&d_start[i], (size_t)gridCtas * sizeof(long long)));
    CUDACHECK(hipMalloc(&d_end[i], (size_t)gridCtas * sizeof(long long)));
  }
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(hipSetDevice(args->gpus[i]));
    launch(i, d_start[i], d_end[i]);
  }

  // Pass 2: synchronize each GPU, copy stamps back, reduce the grid busy window.
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(hipSetDevice(args->gpus[i]));
    TESTCHECK(testStreamSynchronize(1, &args->streams[i], &args->comms[i]));

    // Sample THIS GPU's wall-clock rate to convert its own cycle delta; on a
    // mixed-clock node a single reference rate would skew the non-reference GPUs.
    int rate = 0;
    if (!wallClockRateKhz(args->gpus[i], &rate)) {
      CUDACHECK(hipFree(d_start[i]));
      CUDACHECK(hipFree(d_end[i]));
      d_start[i] = d_end[i] = nullptr;
      for (int j = i + 1; j < args->nGpus; j++) {
        if (d_start[j]) CUDACHECK(hipFree(d_start[j]));
        if (d_end[j]) CUDACHECK(hipFree(d_end[j]));
        d_start[j] = d_end[j] = nullptr;
      }
      if (args->proc == 0 && args->thread == 0) {
        fprintf(stderr, "ERROR: hipDeviceAttributeWallClockRate query failed for GPU %d\n",
                args->gpus[i]);
      }
      return testInternalError;
    }

    std::vector<long long> h_start(gridCtas), h_end(gridCtas);
    CUDACHECK(hipMemcpy(h_start.data(), d_start[i], (size_t)gridCtas * sizeof(long long), hipMemcpyDeviceToHost));
    CUDACHECK(hipMemcpy(h_end.data(), d_end[i], (size_t)gridCtas * sizeof(long long), hipMemcpyDeviceToHost));
    CUDACHECK(hipFree(d_start[i]));
    CUDACHECK(hipFree(d_end[i]));

    const long long mn = rccl_tests_devtime::reduceMinStart(h_start.data(), gridCtas);
    const long long mx = rccl_tests_devtime::reduceMaxEnd(h_end.data(), gridCtas);
    const double perIterUs =
        rccl_tests_devtime::gridBusyWindowPerIterUs(mn, mx, rate, loop);
    if (perIterUs > localMaxUs) localMaxUs = perIterUs;
  }
  Barrier(args);

  // Combine across threads AND ranks through the harness helper. A direct
  // MPI_Allreduce on MPI_COMM_WORLD here would be wrong under -t N>1: MPI is
  // initialized MPI_THREAD_SINGLE, and all N BenchTime threads would issue
  // concurrent collectives on the same communicator (can hang). Allreduce()
  // accumulates across the process's threads, has only the last thread touch MPI,
  // and broadcasts the reduced value back -- which also fixes localMaxUs (per-GPU
  // max on this thread) not being combined across threads.
  double devUs = localMaxUs;
  Allreduce(args, &devUs, /*max*/3);
  *outUs = devUs;
  return testSuccess;
}

}  // namespace gin_devtime

#endif  // GIN_SDMA_DEVTIME_H_
