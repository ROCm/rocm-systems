/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TESTS_GIN_SDMA_DEVTIME_HOST_H_
#define RCCL_TESTS_GIN_SDMA_DEVTIME_HOST_H_

#include <cstdio>
#include <string>

// Pure host helpers shared by BenchTime (common.cu), gin_sdma_devtime.h, util.cu,
// and gin_sdma_devtime_host_test.cpp. Keeps zero-time guards, stamp-reduction
// math, and skipped-row output contract testable without GPU/MPI.

namespace rccl_tests_devtime {

// Out-of-place column filler when BenchTime skipMetricRow is set (matches
// writeBenchMarkLineNullBody() in util.cu). Preserves row layout so the
// in-place pass lands in the in-place column, not the out-of-place slot.
inline constexpr char kSkippedOutOfPlaceColumnFiller[] =
    "                                ";
inline constexpr char kSkippedOutOfPlaceJsonKey[] = "out_of_place";
inline constexpr std::size_t kSkippedOutOfPlaceColumnFillerLen =
    sizeof(kSkippedOutOfPlaceColumnFiller) - 1;

inline void emitSkippedOutOfPlaceColumnToStdout() {
  std::fputs(kSkippedOutOfPlaceColumnFiller, stdout);
}

inline std::string skippedOutOfPlaceJsonNullFragment() {
  return std::string("\"") + kSkippedOutOfPlaceJsonKey + "\":null";
}

inline std::string goldenSkippedRowStdoutFragment(const char* afterPreamble,
                                                  const char* inPlaceBody) {
  return std::string(afterPreamble) + kSkippedOutOfPlaceColumnFiller + inPlaceBody;
}

inline double normalizeElapsedPerIter(double elapsedSec, int iters, int aggIters) {
  const double iterScale = (double)iters * aggIters;
  return (iterScale > 0.0) ? elapsedSec / iterScale : 0.0;
}

inline double applyCudaGraphLaunchesScale(double deltaSec, int cudaGraphLaunches) {
  if (cudaGraphLaunches >= 1) {
    return deltaSec / (double)cudaGraphLaunches;
  }
  return deltaSec;
}

inline bool shouldSkipBenchTimeRow(double deltaSec) { return deltaSec <= 0.0; }

inline bool shouldComputeIterStats(int iters, int perIterSkip) {
  return (iters - perIterSkip) > 0;
}

inline long long reduceMinStart(const long long* starts, int gridCtas) {
  long long mn = starts[0];
  for (int c = 1; c < gridCtas; c++)
    if (starts[c] < mn) mn = starts[c];
  return mn;
}

inline long long reduceMaxEnd(const long long* ends, int gridCtas) {
  long long mx = ends[0];
  for (int c = 1; c < gridCtas; c++)
    if (ends[c] > mx) mx = ends[c];
  return mx;
}

// Convert a grid busy window (min start .. max end wall-clock cycles) to
// per-iteration latency in microseconds. Matches gin_devtime::measure().
inline double gridBusyWindowPerIterUs(long long minStart, long long maxEnd,
                                      int wallClockRateKhz, int loop) {
  if (wallClockRateKhz <= 0) return 0.0;
  const double totalUs =
      (double)(maxEnd - minStart) / (double)wallClockRateKhz * 1.0e3;
  return (loop > 0) ? totalUs / (double)loop : totalUs;
}

}  // namespace rccl_tests_devtime

#endif  // RCCL_TESTS_GIN_SDMA_DEVTIME_HOST_H_
