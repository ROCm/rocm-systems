// Interface between gather_chain.hip and the fixed harness.
#pragma once
#include <hip/hip_runtime.h>

constexpr int THREADS = 304 * 64;  // outputs
constexpr int STEPS   = 256;       // gathers per output
constexpr int TABLE   = 1 << 24;   // entries in the data table

// out[t] = sum over j < STEPS of data[index[j * THREADS + t]], with every index entry in
// [0, TABLE). All pointers are device memory on the current device; the result must be
// in out when the device has synchronized.
void
gather_launch(const int* index, const float* data, float* out);
