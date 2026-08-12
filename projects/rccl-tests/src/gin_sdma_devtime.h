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
// identical across collectives: query the wall-clock rate once, allocate the per-CTA
// start/end stamp buffers, run the launch, reduce the grid busy window
// (min(start)..max(end) over CTAs) and the slowest rank (MPI MAX), and return the
// per-iteration device latency in microseconds. This is the pure device-function
// execution time -- it excludes host launch, stream/graph, and teardown overhead.
//
// The reported span is driven by BenchTime (see common.cu) via
// NCCL_GIN_ANVIL_DEVICE_TIMING (legacy NCCL_GIN_ANVIL_A2A_DEVICE_TIMING):
// 1=augment (each hook prints its own extra "#[*-devtime]" line next to the graph
// numbers), 2=device-time-only (the hook returns per-iteration seconds via
// outDeltaSec and BenchTime reports it AS the out-of-place metric).
#ifndef GIN_SDMA_DEVTIME_H_
#define GIN_SDMA_DEVTIME_H_

#include <vector>
#include "common.h"

namespace gin_devtime {

// Parse a positive int env var, else the default.
static inline int envInt(const char* name, int def) {
  const char* e = getenv(name);
  return (e && *e) ? atoi(e) : def;
}

// GPU fixed-frequency wall-clock rate (kHz), queried once for the current device.
static inline int wallClockRateKhz() {
  static int khz = 0;
  if (khz == 0) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev) != hipSuccess || khz <= 0)
      khz = 1;
  }
  return khz;
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
  const int rate = wallClockRateKhz();
  double localMaxUs = 0.0;

  Barrier(args);
  for (int i = 0; i < args->nGpus; i++) {
    CUDACHECK(cudaSetDevice(args->gpus[i]));

    long long* d_start = nullptr;
    long long* d_end = nullptr;
    CUDACHECK(cudaMalloc(&d_start, (size_t)gridCtas * sizeof(long long)));
    CUDACHECK(cudaMalloc(&d_end, (size_t)gridCtas * sizeof(long long)));

    launch(i, d_start, d_end);
    CUDACHECK(cudaStreamSynchronize(args->streams[i]));

    std::vector<long long> h_start(gridCtas), h_end(gridCtas);
    CUDACHECK(cudaMemcpy(h_start.data(), d_start, (size_t)gridCtas * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaMemcpy(h_end.data(), d_end, (size_t)gridCtas * sizeof(long long), cudaMemcpyDeviceToHost));
    CUDACHECK(cudaFree(d_start));
    CUDACHECK(cudaFree(d_end));

    long long mn = h_start[0], mx = h_end[0];
    for (int c = 1; c < gridCtas; c++) {
      if (h_start[c] < mn) mn = h_start[c];
      if (h_end[c] > mx) mx = h_end[c];
    }
    double totalUs = (double)(mx - mn) / (double)rate * 1.0e3;  // cycles/kHz -> ms -> us
    double perIterUs = (loop > 0) ? totalUs / (double)loop : totalUs;
    if (perIterUs > localMaxUs) localMaxUs = perIterUs;
  }
  Barrier(args);

  double devUs = localMaxUs;
#ifdef MPI_SUPPORT
  MPI_Allreduce(MPI_IN_PLACE, &devUs, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  *outUs = devUs;
  return testSuccess;
}

}  // namespace gin_devtime

#endif  // GIN_SDMA_DEVTIME_H_
