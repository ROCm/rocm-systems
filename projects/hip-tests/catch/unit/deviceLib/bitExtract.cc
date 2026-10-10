/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/device_functions.h>

#include <stdint.h>
#include <vector>

// The intrinsics take the offset from src1 and the width from src2, both masked to the low bits of
// the source width, so every input the intrinsics can distinguish is covered by sweeping offset and
// width over [0, bits) for each source pattern below. A width of bits is not expressible: it masks
// to 0.
__host__ __device__ static inline unsigned long long int source_pattern(unsigned int index) {
  constexpr unsigned long long int kPatterns[] = {
      0ull,
      ~0ull,
      0x5555555555555555ull,
      0xaaaaaaaaaaaaaaaaull,
      0x0123456789abcdefull,
      0xdeadbeefcafef00dull,
  };
  return kPatterns[index];
}

static constexpr unsigned int kNumPatterns = 6;

__device__ static inline unsigned int bitextract(unsigned int src0, unsigned int src1,
                                                 unsigned int src2) {
  return __bitextract_u32(src0, src1, src2);
}

__device__ static inline unsigned long long int bitextract(unsigned long long int src0,
                                                           unsigned int src1, unsigned int src2) {
  return __bitextract_u64(src0, src1, src2);
}

// One thread per (pattern, offset, width) case. The inputs come from the index, so nothing has to
// be staged to the device.
template <typename T> __global__ void bitextract_kernel(T* out, T* aliased_out,
                                                        unsigned int num_cases) {
  constexpr unsigned int kBits = sizeof(T) * 8;

  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_cases) return;

  const unsigned int width = index % kBits;
  const unsigned int offset = (index / kBits) % kBits;
  const T src0 = static_cast<T>(source_pattern(index / (kBits * kBits)));

  out[index] = bitextract(src0, offset, width);
  // Operands past the source width must mask back to the same field.
  aliased_out[index] = bitextract(src0, offset + kBits, width + kBits);
}

// Reference model: assembles the field one bit at a time so that it shares no structure with the
// shift-and-mask form the intrinsics use.
template <typename T> static T bit_extract_ref(T src0, unsigned int offset, unsigned int width) {
  constexpr unsigned int kBits = sizeof(T) * 8;

  offset &= kBits - 1;
  width &= kBits - 1;

  T result = 0;
  for (unsigned int i = 0; i < width; ++i) {
    result |= static_cast<T>((src0 >> (offset + i)) & 1) << i;
  }
  return result;
}

template <typename T> static void run_bitextract_cases() {
  constexpr unsigned int kBits = sizeof(T) * 8;
  constexpr unsigned int kNumCases = kNumPatterns * kBits * kBits;

  LinearAllocGuard<T> out(LinearAllocs::hipMalloc, kNumCases * sizeof(T));
  LinearAllocGuard<T> aliased_out(LinearAllocs::hipMalloc, kNumCases * sizeof(T));

  constexpr unsigned int kThreadsPerBlock = 256;
  const unsigned int num_blocks = (kNumCases + kThreadsPerBlock - 1) / kThreadsPerBlock;
  bitextract_kernel<T><<<dim3(num_blocks), dim3(kThreadsPerBlock)>>>(out.ptr(), aliased_out.ptr(),
                                                                     kNumCases);
  HIP_CHECK(hipGetLastError());

  std::vector<T> host_out(kNumCases);
  std::vector<T> host_aliased_out(kNumCases);
  HIP_CHECK(hipMemcpy(host_out.data(), out.ptr(), kNumCases * sizeof(T), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(host_aliased_out.data(), aliased_out.ptr(), kNumCases * sizeof(T),
                      hipMemcpyDeviceToHost));

  unsigned int checked_cases = 0;
  for (unsigned int index = 0; index < kNumCases; ++index) {
    const unsigned int width = index % kBits;
    const unsigned int offset = (index / kBits) % kBits;
    const T src0 = static_cast<T>(source_pattern(index / (kBits * kBits)));

    // The intrinsics are only defined while the field fits in the source.
    if (offset + width > kBits) continue;

    const T expected = bit_extract_ref<T>(src0, offset, width);
    if (host_out[index] != expected || host_aliased_out[index] != host_out[index]) {
      CAPTURE(kBits, index, offset, width, src0, expected, host_out[index],
              host_aliased_out[index]);
      REQUIRE(host_out[index] == expected);
      REQUIRE(host_aliased_out[index] == host_out[index]);
    }
    ++checked_cases;
  }

  // Guards the sweep itself: the pairs with offset + width <= bits, width < bits.
  REQUIRE(checked_cases == kNumPatterns * (kBits + kBits * (kBits + 1) / 2 - 1));
}

HIP_TEST_CASE(Unit_bitExtract) {
  run_bitextract_cases<unsigned int>();
  run_bitextract_cases<unsigned long long int>();
}
