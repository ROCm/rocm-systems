#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void high_lane_kernel(const uint8_t *input, uint32_t *output) {
  const uint32_t lane = threadIdx.x & 63u;
  const uint32_t selector = input[lane] & 1u;

  if (lane < 32u) {
    const uint32_t value = lane + 1u;
    output[lane] = value * 3u + 7u;
    return;
  }

  uint32_t value = lane + 1u;
  if (selector != 0u) {
    for (uint32_t i = 0; i < 4u; ++i) {
      value = (value * 5u) ^ (17u + input[(lane + i) & 63u]);
    }
  } else {
    for (uint32_t i = 0; i < 3u; ++i) {
      value = (value * 7u) ^ (23u + input[(lane + i) & 63u]);
    }
  }
  output[lane] = value;
}
