#include <hip/hip_runtime.h>

#include <stdint.h>

extern "C" __global__ void branchy_kernel_a(const uint8_t *input,
                                            uint32_t *scratch,
                                            uint32_t n) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }

  uint32_t value = input[tid & 255u] + tid;
  for (uint32_t i = 0; i < 8; ++i) {
    if (((value + i) & 1u) == 0) {
      value = value * 3u + i;
    } else {
      value = (value ^ (17u + i)) + (tid & 7u);
    }
  }

  if ((value & 0x20u) != 0) {
    value ^= 0xa5a5u;
  } else if ((value & 0x7u) == 3u) {
    value += 0x123u;
  } else {
    value ^= tid << 1;
  }

  scratch[tid] = value;
}

extern "C" __global__ void branchy_kernel_b(const uint32_t *scratch,
                                            uint32_t *result,
                                            uint32_t n,
                                            uint32_t selector) {
  const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) {
    return;
  }

  uint32_t value = scratch[tid] ^ selector;
  for (uint32_t i = 0; i < 5; ++i) {
    if ((selector & (1u << i)) != 0) {
      value += (tid + 1u) * (i + 3u);
    } else {
      value = (value >> 1) ^ (value << 3) ^ i;
    }
  }

  if ((value & 0x40u) != 0) {
    value ^= 0x5au;
  } else if ((value & 1u) != 0) {
    value += selector;
  } else {
    value += 7u;
  }
  result[tid] = value;
}
