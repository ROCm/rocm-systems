// Interface between leader_scale.hip and the fixed harness.
#pragma once
#include <hip/hip_runtime.h>

constexpr int BLOCKS = 304 * 4;
constexpr int ROUNDS = 64;
constexpr int SERIES = 512;  // terms in the per-round scale series

// For b < BLOCKS, r < ROUNDS, x < 256, with i = (b * ROUNDS + r) * 256 + x:
//   out[i] = in[i] * scale(r),  scale(r) = 1 + 1e-3 * sum over k = 1..SERIES of decay^k / (k + r).
// decay changes from call to call. Device memory on the current device; the result must be
// in out when the device has synchronized.
void
leader_scale_launch(const float* in, float* out, float decay);
