#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ __launch_bounds__(64)
void entry_liveness_kernel(uint32_t *out) {
  const uint32_t tid = threadIdx.x;
  uint32_t value0 = tid * 17u + 3u;
  uint32_t value1 = value0 ^ 0x9e3779b9u;
  uint32_t value2 = value1 + (tid << 4);
  uint32_t value3 = value2 ^ (value0 >> 1);

  for (uint32_t i = 0; i < 4; ++i) {
    value0 = value0 * 33u + i;
    value1 ^= value0 + (i << 8);
    value2 += value1 ^ tid;
    value3 = (value3 << 1) ^ (value2 >> 3);
  }

  out[tid] = value0 ^ value1 ^ value2 ^ value3;
}
