// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "util/data_types.h"

#define HIP_CHECK(e)                                                                               \
  do {                                                                                             \
    hipError_t e_ = (e);                                                                           \
    if (e_ != hipSuccess) {                                                                        \
      printf("HIP error %d: %s at %s:%d\n", e_, hipGetErrorString(e_), __FILE__, __LINE__);        \
      return 1;                                                                                    \
    }                                                                                              \
  } while (0)

// Vector types for builtins (ext_vector_type, not HIP vector types).
using v2f = float __attribute__((ext_vector_type(2)));
using v2s = short __attribute__((ext_vector_type(2)));
using v16f = float __attribute__((ext_vector_type(16)));
using v32f = float __attribute__((ext_vector_type(32)));
using v6u = unsigned __attribute__((ext_vector_type(6)));

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

static void pass(const char *name) {
  printf("[  PASS  ] %s\n", name);
  g_pass++;
}
static void fail(const char *name, const char *msg) {
  printf("[  FAIL  ] %s: %s\n", name, msg);
  g_fail++;
}
static void skip(const char *name) {
  printf("[  SKIP  ] %s\n", name);
  g_skip++;
}

static bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

// ---- Non-scaled FP8 kernels ----

__global__ void k_pk_fp8_f32(float a, float b, int *out) {
  *out = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
}

__global__ void k_pk_bf8_f32(float a, float b, int *out) {
  *out = __builtin_amdgcn_cvt_pk_bf8_f32(a, b, 0, false);
}

__global__ void k_pk_f32_fp8(int src, float *out) {
  v2f r = __builtin_amdgcn_cvt_pk_f32_fp8(src, false);
  out[0] = r[0];
  out[1] = r[1];
}

__global__ void k_pk_f32_bf8(int src, float *out) {
  v2f r = __builtin_amdgcn_cvt_pk_f32_bf8(src, false);
  out[0] = r[0];
  out[1] = r[1];
}

__global__ void k_sr_fp8_f32(float val, int rnd, int *out) {
  *out = __builtin_amdgcn_cvt_sr_fp8_f32(val, rnd, 0, 0);
}

__global__ void k_sr_bf8_f32(float val, int rnd, int *out) {
  *out = __builtin_amdgcn_cvt_sr_bf8_f32(val, rnd, 0, 0);
}

// ---- Scaled FP8/BF8 kernels ----

__global__ void k_scalef32_pk_fp8_f32(float a, float b, float scale, unsigned *out) {
  v2s zero = {0, 0};
  v2s r = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(zero, a, b, scale, false);
  unsigned packed;
  __builtin_memcpy(&packed, &r, sizeof(packed));
  *out = packed;
}

__global__ void k_scalef32_pk_bf8_f32(float a, float b, float scale, unsigned *out) {
  v2s zero = {0, 0};
  v2s r = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(zero, a, b, scale, false);
  unsigned packed;
  __builtin_memcpy(&packed, &r, sizeof(packed));
  *out = packed;
}

__global__ void k_scalef32_pk_f32_fp8(unsigned src, float scale, float *out) {
  v2f r = __builtin_amdgcn_cvt_scalef32_pk_f32_fp8(src, scale, false);
  out[0] = r[0];
  out[1] = r[1];
}

__global__ void k_scalef32_pk_f32_bf8(unsigned src, float scale, float *out) {
  v2f r = __builtin_amdgcn_cvt_scalef32_pk_f32_bf8(src, scale, false);
  out[0] = r[0];
  out[1] = r[1];
}

template <int BYTE> __global__ void k_scalef32_f32_fp8(unsigned src, float scale, float *out) {
  out[0] = __builtin_amdgcn_cvt_scalef32_f32_fp8(src, scale, BYTE);
}

__global__ void k_scalef32_sr_fp8_f32(float val, unsigned rnd, float scale, int *out) {
  *out = __builtin_amdgcn_cvt_scalef32_sr_fp8_f32(0, val, rnd, scale, 0);
}

__global__ void k_scalef32_sr_bf8_f32(float val, unsigned rnd, float scale, int *out) {
  *out = __builtin_amdgcn_cvt_scalef32_sr_bf8_f32(0, val, rnd, scale, 0);
}

// ---- Scaled FP4 kernels ----

__global__ void k_scalef32_pk_fp4_f32(float a, float b, float scale, int *out) {
  *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(0, a, b, scale, 0);
}

__global__ void k_scalef32_pk_f32_fp4(unsigned src, float scale, float *out) {
  v2f r = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(src, scale, 0);
  out[0] = r[0];
  out[1] = r[1];
}

__global__ void k_scalef32_sr_pk_fp4_f32(float a, float b, unsigned seed, float scale, int *out) {
  v2f data = {a, b};
  *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(0, data, seed, scale, 0);
}

// ---- Wide FP6/BF6 kernels ----

__global__ void k_scalef32_2xpk16_fp6_f32(const float *in, unsigned *out, float scale) {
  v16f lo, hi;
  for (int i = 0; i < 16; ++i) {
    lo[i] = in[i];
    hi[i] = in[16 + i];
  }
  v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(lo, hi, scale);
  for (int i = 0; i < 6; ++i)
    out[i] = r[i];
}

__global__ void k_scalef32_pk32_f32_fp6(const unsigned *in, float *out, float scale) {
  v6u src;
  for (int i = 0; i < 6; ++i)
    src[i] = in[i];
  v32f r = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(src, scale);
  for (int i = 0; i < 32; ++i)
    out[i] = r[i];
}

__global__ void k_scalef32_2xpk16_bf6_f32(const float *in, unsigned *out, float scale) {
  v16f lo, hi;
  for (int i = 0; i < 16; ++i) {
    lo[i] = in[i];
    hi[i] = in[16 + i];
  }
  v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(lo, hi, scale);
  for (int i = 0; i < 6; ++i)
    out[i] = r[i];
}

__global__ void k_scalef32_pk32_f32_bf6(const unsigned *in, float *out, float scale) {
  v6u src;
  for (int i = 0; i < 6; ++i)
    src[i] = in[i];
  v32f r = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(src, scale);
  for (int i = 0; i < 32; ++i)
    out[i] = r[i];
}

// ---- Device memory helpers ----

template <class T> static T *alloc_dev(size_t n) {
  T *d = nullptr;
  (void)hipMalloc(&d, n * sizeof(T));
  return d;
}

template <class T> static T read_dev(const T *d) {
  T h;
  (void)hipMemcpy(&h, d, sizeof(T), hipMemcpyDeviceToHost);
  return h;
}

template <class T> static void write_dev(T *d, const T &val) {
  (void)hipMemcpy(d, &val, sizeof(T), hipMemcpyHostToDevice);
}

template <class T> static std::vector<T> read_dev_vec(const T *d, size_t n) {
  std::vector<T> h(n);
  (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  return h;
}

template <class T> static T *to_dev(const std::vector<T> &h) {
  T *d = alloc_dev<T>(h.size());
  (void)hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

// ---- Test implementations ----

static int test_nonscaled_pk_fp8_f32() {
  int *dO = alloc_dev<int>(1);
  k_pk_fp8_f32<<<1, 1>>>(1.0f, 2.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t lo = result & 0xFF;
  uint8_t hi = (result >> 8) & 0xFF;
  uint8_t exp_lo = util::f32_to_fp8_e4m3_rne(1.0f);
  uint8_t exp_hi = util::f32_to_fp8_e4m3_rne(2.0f);
  (void)hipFree(dO);
  if (lo != exp_lo || hi != exp_hi) {
    printf("  got lo=%02x hi=%02x, expected lo=%02x hi=%02x\n", lo, hi, exp_lo, exp_hi);
    return 1;
  }
  return 0;
}

static int test_nonscaled_pk_bf8_f32() {
  int *dO = alloc_dev<int>(1);
  k_pk_bf8_f32<<<1, 1>>>(1.0f, 2.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t lo = result & 0xFF;
  uint8_t hi = (result >> 8) & 0xFF;
  uint8_t exp_lo = util::f32_to_bf8_e5m2_rne(1.0f);
  uint8_t exp_hi = util::f32_to_bf8_e5m2_rne(2.0f);
  (void)hipFree(dO);
  if (lo != exp_lo || hi != exp_hi) {
    printf("  got lo=%02x hi=%02x, expected lo=%02x hi=%02x\n", lo, hi, exp_lo, exp_hi);
    return 1;
  }
  return 0;
}

static int test_nonscaled_pk_f32_fp8() {
  float *dO = alloc_dev<float>(2);
  uint8_t fp8_1 = util::f32_to_fp8_e4m3_rne(1.0f);
  uint8_t fp8_2 = util::f32_to_fp8_e4m3_rne(2.0f);
  int packed = fp8_1 | (fp8_2 << 8);
  k_pk_f32_fp8<<<1, 1>>>(packed, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = read_dev_vec(dO, 2);
  (void)hipFree(dO);
  if (got[0] != 1.0f || got[1] != 2.0f) {
    printf("  got [%f, %f], expected [1.0, 2.0]\n", got[0], got[1]);
    return 1;
  }
  return 0;
}

static int test_nonscaled_pk_f32_bf8() {
  float *dO = alloc_dev<float>(2);
  uint8_t bf8_1 = util::f32_to_bf8_e5m2_rne(1.0f);
  uint8_t bf8_2 = util::f32_to_bf8_e5m2_rne(2.0f);
  int packed = bf8_1 | (bf8_2 << 8);
  k_pk_f32_bf8<<<1, 1>>>(packed, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = read_dev_vec(dO, 2);
  (void)hipFree(dO);
  if (got[0] != 1.0f || got[1] != 2.0f) {
    printf("  got [%f, %f], expected [1.0, 2.0]\n", got[0], got[1]);
    return 1;
  }
  return 0;
}

static int test_nonscaled_sr_fp8_f32() {
  int *dO = alloc_dev<int>(1);
  k_sr_fp8_f32<<<1, 1>>>(1.0f, 0, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t got = result & 0xFF;
  uint8_t expected = util::f32_to_fp8_e4m3_sr(1.0f, 0);
  (void)hipFree(dO);
  if (got != expected) {
    printf("  got %02x, expected %02x\n", got, expected);
    return 1;
  }
  return 0;
}

static int test_nonscaled_sr_bf8_f32() {
  int *dO = alloc_dev<int>(1);
  k_sr_bf8_f32<<<1, 1>>>(1.0f, 0, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t got = result & 0xFF;
  uint8_t expected = util::f32_to_bf8_e5m2_sr(1.0f, 0);
  (void)hipFree(dO);
  if (got != expected) {
    printf("  got %02x, expected %02x\n", got, expected);
    return 1;
  }
  return 0;
}

// ---- Scaled FP8 tests ----

static int test_scaled_pk_fp8_f32() {
  unsigned *dO = alloc_dev<unsigned>(1);
  k_scalef32_pk_fp8_f32<<<1, 1>>>(1.0f, 2.0f, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned packed = read_dev(dO);
  uint8_t lo = packed & 0xFF;
  uint8_t hi = (packed >> 8) & 0xFF;
  uint8_t exp_lo = util::f32_to_fp8_e4m3_rne(1.0f);
  uint8_t exp_hi = util::f32_to_fp8_e4m3_rne(2.0f);
  (void)hipFree(dO);
  if (lo != exp_lo || hi != exp_hi) {
    printf("  got lo=%02x hi=%02x, expected lo=%02x hi=%02x\n", lo, hi, exp_lo, exp_hi);
    return 1;
  }
  return 0;
}

static int test_scaled_pk_bf8_f32() {
  unsigned *dO = alloc_dev<unsigned>(1);
  k_scalef32_pk_bf8_f32<<<1, 1>>>(1.0f, 2.0f, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned packed = read_dev(dO);
  uint8_t lo = packed & 0xFF;
  uint8_t hi = (packed >> 8) & 0xFF;
  uint8_t exp_lo = util::f32_to_bf8_e5m2_rne(1.0f);
  uint8_t exp_hi = util::f32_to_bf8_e5m2_rne(2.0f);
  (void)hipFree(dO);
  if (lo != exp_lo || hi != exp_hi) {
    printf("  got lo=%02x hi=%02x, expected lo=%02x hi=%02x\n", lo, hi, exp_lo, exp_hi);
    return 1;
  }
  return 0;
}

static int test_scaled_pk_f32_fp8() {
  float *dO = alloc_dev<float>(2);
  uint8_t fp8_1 = util::f32_to_fp8_e4m3_rne(1.0f);
  uint8_t fp8_2 = util::f32_to_fp8_e4m3_rne(2.0f);
  unsigned packed = fp8_1 | (fp8_2 << 8);
  k_scalef32_pk_f32_fp8<<<1, 1>>>(packed, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = read_dev_vec(dO, 2);
  (void)hipFree(dO);
  if (got[0] != 1.0f || got[1] != 2.0f) {
    printf("  got [%f, %f], expected [1.0, 2.0]\n", got[0], got[1]);
    return 1;
  }
  return 0;
}

static int test_scaled_pk_f32_bf8() {
  float *dO = alloc_dev<float>(2);
  uint8_t bf8_1 = util::f32_to_bf8_e5m2_rne(1.0f);
  uint8_t bf8_2 = util::f32_to_bf8_e5m2_rne(2.0f);
  unsigned packed = bf8_1 | (bf8_2 << 8);
  k_scalef32_pk_f32_bf8<<<1, 1>>>(packed, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = read_dev_vec(dO, 2);
  (void)hipFree(dO);
  if (got[0] != 1.0f || got[1] != 2.0f) {
    printf("  got [%f, %f], expected [1.0, 2.0]\n", got[0], got[1]);
    return 1;
  }
  return 0;
}

static int test_scaled_f32_fp8_byte_sel() {
  float *dO = alloc_dev<float>(1);
  uint8_t fp8_vals[4] = {
      util::f32_to_fp8_e4m3_rne(1.0f),
      util::f32_to_fp8_e4m3_rne(2.0f),
      util::f32_to_fp8_e4m3_rne(3.0f),
      util::f32_to_fp8_e4m3_rne(4.0f),
  };
  unsigned packed = fp8_vals[0] | (fp8_vals[1] << 8) | (fp8_vals[2] << 16) | (fp8_vals[3] << 24);
  float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  auto run_byte = [&](int byte) -> int {
    switch (byte) {
    case 0:
      k_scalef32_f32_fp8<0><<<1, 1>>>(packed, 1.0f, dO);
      break;
    case 1:
      k_scalef32_f32_fp8<1><<<1, 1>>>(packed, 1.0f, dO);
      break;
    case 2:
      k_scalef32_f32_fp8<2><<<1, 1>>>(packed, 1.0f, dO);
      break;
    case 3:
      k_scalef32_f32_fp8<3><<<1, 1>>>(packed, 1.0f, dO);
      break;
    }
    return hipDeviceSynchronize();
  };
  for (int byte = 0; byte < 4; ++byte) {
    HIP_CHECK(static_cast<hipError_t>(run_byte(byte)));
    float got = read_dev(dO);
    if (got != expected[byte]) {
      printf("  byte_sel=%d: got %f, expected %f\n", byte, got, expected[byte]);
      (void)hipFree(dO);
      return 1;
    }
  }
  (void)hipFree(dO);
  return 0;
}

static int test_scaled_sr_fp8_f32() {
  int *dO = alloc_dev<int>(1);
  k_scalef32_sr_fp8_f32<<<1, 1>>>(1.0f, 0u, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t got = result & 0xFF;
  uint8_t expected = util::f32_to_fp8_e4m3_sr(1.0f, 0);
  (void)hipFree(dO);
  if (got != expected) {
    printf("  got %02x, expected %02x\n", got, expected);
    return 1;
  }
  return 0;
}

static int test_scaled_sr_bf8_f32() {
  int *dO = alloc_dev<int>(1);
  k_scalef32_sr_bf8_f32<<<1, 1>>>(1.0f, 0u, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int result = read_dev(dO);
  uint8_t got = result & 0xFF;
  uint8_t expected = util::f32_to_bf8_e5m2_sr(1.0f, 0);
  (void)hipFree(dO);
  if (got != expected) {
    printf("  got %02x, expected %02x\n", got, expected);
    return 1;
  }
  return 0;
}

// ---- FP4 tests ----

static int test_scaled_fp4_roundtrip() {
  // Hardcoded truth table for positive FP4 E2M1 values.
  static constexpr float kFp4Truth[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  int *dNarrow = alloc_dev<int>(1);
  float *dWide = alloc_dev<float>(2);
  int errs = 0;
  for (int ca = 0; ca < 8; ++ca) {
    for (int cb = 0; cb < 8; ++cb) {
      float a = kFp4Truth[ca];
      float b = kFp4Truth[cb];
      k_scalef32_pk_fp4_f32<<<1, 1>>>(a, b, 1.0f, dNarrow);
      HIP_CHECK(hipDeviceSynchronize());
      int packed = read_dev(dNarrow);
      uint8_t nibble0 = packed & 0xF;
      uint8_t nibble1 = (packed >> 4) & 0xF;
      if (nibble0 != ca || nibble1 != cb) {
        printf("  narrow: a=%d b=%d -> nib0=%d nib1=%d (expected %d %d)\n", ca, cb, nibble0,
               nibble1, ca, cb);
        errs++;
        if (errs > 5)
          break;
      }
      k_scalef32_pk_f32_fp4<<<1, 1>>>(static_cast<unsigned>(packed), 1.0f, dWide);
      HIP_CHECK(hipDeviceSynchronize());
      auto widened = read_dev_vec(dWide, 2);
      if (widened[0] != a || widened[1] != b) {
        printf("  widen: a=%d b=%d -> [%f, %f] (expected [%f, %f])\n", ca, cb, widened[0],
               widened[1], a, b);
        errs++;
        if (errs > 5)
          break;
      }
    }
    if (errs > 5)
      break;
  }
  (void)hipFree(dNarrow);
  (void)hipFree(dWide);
  return errs > 0 ? 1 : 0;
}

static int test_scaled_sr_pk_fp4_f32() {
  static constexpr float kFp4Truth[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  int *dO = alloc_dev<int>(1);
  int errs = 0;
  for (int ca = 0; ca < 8; ++ca) {
    for (int cb = 0; cb < 8; ++cb) {
      float a = kFp4Truth[ca];
      float b = kFp4Truth[cb];
      k_scalef32_sr_pk_fp4_f32<<<1, 1>>>(a, b, 0xdeadbeefu, 1.0f, dO);
      HIP_CHECK(hipDeviceSynchronize());
      int packed = read_dev(dO);
      uint8_t nibble0 = packed & 0xF;
      uint8_t nibble1 = (packed >> 4) & 0xF;
      if (nibble0 != ca || nibble1 != cb) {
        printf("  sr_pk_fp4: a=%d b=%d -> nib0=%d nib1=%d (expected %d %d)\n", ca, cb, nibble0,
               nibble1, ca, cb);
        errs++;
        if (errs > 5)
          break;
      }
    }
    if (errs > 5)
      break;
  }
  (void)hipFree(dO);
  return errs > 0 ? 1 : 0;
}

// ---- Wide FP6/BF6 tests ----
// Round-trip: narrow 32 floats to 6-bit packed, widen back, compare against
// the original values (which are exact FP6/BF6 representable values).

static int test_wide_fp6_roundtrip() {
  float *dIn = alloc_dev<float>(32);
  unsigned *dPacked = alloc_dev<unsigned>(6);
  float *dOut = alloc_dev<float>(32);
  int errs = 0;
  for (int base = 0; base < 64; base += 32) {
    std::vector<float> in(32);
    for (int c = 0; c < 32; ++c)
      in[c] = util::fp6_e2m3_to_f32(static_cast<uint8_t>(base + c));
    (void)hipMemcpy(dIn, in.data(), 32 * sizeof(float), hipMemcpyHostToDevice);
    k_scalef32_2xpk16_fp6_f32<<<1, 1>>>(dIn, dPacked, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    k_scalef32_pk32_f32_fp6<<<1, 1>>>(dPacked, dOut, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto widened = read_dev_vec(dOut, 32);
    // 2xpk16 interleaves: field[2k]=lo[k], field[2k+1]=hi[k]
    // pk32_f32 reads contiguously, so out[2k]=in[k], out[2k+1]=in[16+k]
    for (int k = 0; k < 16; ++k) {
      if (widened[2 * k] != in[k]) {
        printf("  fp6 rt base=%d lo[%d]: got %f, expected %f\n", base, k, widened[2 * k], in[k]);
        errs++;
      }
      if (widened[2 * k + 1] != in[16 + k]) {
        printf("  fp6 rt base=%d hi[%d]: got %f, expected %f\n", base, k, widened[2 * k + 1],
               in[16 + k]);
        errs++;
      }
    }
  }
  (void)hipFree(dIn);
  (void)hipFree(dPacked);
  (void)hipFree(dOut);
  return errs > 0 ? 1 : 0;
}

static int test_wide_bf6_roundtrip() {
  float *dIn = alloc_dev<float>(32);
  unsigned *dPacked = alloc_dev<unsigned>(6);
  float *dOut = alloc_dev<float>(32);
  int errs = 0;
  for (int base = 0; base < 64; base += 32) {
    std::vector<float> in(32);
    for (int c = 0; c < 32; ++c)
      in[c] = util::bf6_e3m2_to_f32(static_cast<uint8_t>(base + c));
    (void)hipMemcpy(dIn, in.data(), 32 * sizeof(float), hipMemcpyHostToDevice);
    k_scalef32_2xpk16_bf6_f32<<<1, 1>>>(dIn, dPacked, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    k_scalef32_pk32_f32_bf6<<<1, 1>>>(dPacked, dOut, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto widened = read_dev_vec(dOut, 32);
    // 2xpk16 interleaves: field[2k]=lo[k], field[2k+1]=hi[k]
    // pk32_f32 reads contiguously, so out[2k]=in[k], out[2k+1]=in[16+k]
    for (int k = 0; k < 16; ++k) {
      if (widened[2 * k] != in[k]) {
        printf("  bf6 rt base=%d lo[%d]: got %f, expected %f\n", base, k, widened[2 * k], in[k]);
        errs++;
      }
      if (widened[2 * k + 1] != in[16 + k]) {
        printf("  bf6 rt base=%d hi[%d]: got %f, expected %f\n", base, k, widened[2 * k + 1],
               in[16 + k]);
        errs++;
      }
    }
  }
  (void)hipFree(dIn);
  (void)hipFree(dPacked);
  (void)hipFree(dOut);
  return errs > 0 ? 1 : 0;
}

// ---- Scale edge cases ----

static int test_scale_variation() {
  unsigned *dO = alloc_dev<unsigned>(1);
  k_scalef32_pk_fp8_f32<<<1, 1>>>(1.0f, 2.0f, 2.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  unsigned packed = read_dev(dO);
  uint8_t lo = packed & 0xFF;
  uint8_t hi = (packed >> 8) & 0xFF;
  uint8_t exp_lo = util::f32_to_fp8_e4m3_rne(1.0f / 2.0f);
  uint8_t exp_hi = util::f32_to_fp8_e4m3_rne(2.0f / 2.0f);
  (void)hipFree(dO);
  if (lo != exp_lo || hi != exp_hi) {
    printf("  scale=2: got lo=%02x hi=%02x, expected lo=%02x hi=%02x\n", lo, hi, exp_lo, exp_hi);
    return 1;
  }
  return 0;
}

static int test_nan_inf_saturation() {
  int *dO = alloc_dev<int>(1);
  float nan_val = std::numeric_limits<float>::quiet_NaN();
  float inf_val = std::numeric_limits<float>::infinity();
  k_scalef32_pk_fp4_f32<<<1, 1>>>(nan_val, inf_val, 1.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  int packed = read_dev(dO);
  uint8_t nib0 = packed & 0xF;
  uint8_t nib1 = (packed >> 4) & 0xF;
  (void)hipFree(dO);
  if (nib0 != 0) {
    printf("  NaN->fp4: got %d, expected 0\n", nib0);
    return 1;
  }
  if (nib1 != 0x7) {
    printf("  Inf->fp4: got %d, expected 7 (max)\n", nib1);
    return 1;
  }
  return 0;
}

// ---- Main ----

struct TestCase {
  const char *name;
  int (*func)();
};

int main() {
  if (!have_device()) {
    printf("No HIP device found, skipping all tests\n");
    return 0;
  }

  TestCase tests[] = {
      {"NonScaled.PkFp8F32", test_nonscaled_pk_fp8_f32},
      {"NonScaled.PkBf8F32", test_nonscaled_pk_bf8_f32},
      {"NonScaled.PkF32Fp8", test_nonscaled_pk_f32_fp8},
      {"NonScaled.PkF32Bf8", test_nonscaled_pk_f32_bf8},
      {"NonScaled.SrFp8F32", test_nonscaled_sr_fp8_f32},
      {"NonScaled.SrBf8F32", test_nonscaled_sr_bf8_f32},
      {"Scaled.PkFp8F32", test_scaled_pk_fp8_f32},
      {"Scaled.PkBf8F32", test_scaled_pk_bf8_f32},
      {"Scaled.PkF32Fp8", test_scaled_pk_f32_fp8},
      {"Scaled.PkF32Bf8", test_scaled_pk_f32_bf8},
      {"Scaled.F32Fp8ByteSel", test_scaled_f32_fp8_byte_sel},
      {"Scaled.SrFp8F32", test_scaled_sr_fp8_f32},
      {"Scaled.SrBf8F32", test_scaled_sr_bf8_f32},
      {"Scaled.Fp4RoundTrip", test_scaled_fp4_roundtrip},
      {"Scaled.SrPkFp4F32", test_scaled_sr_pk_fp4_f32},
      {"Wide.Fp6RoundTrip", test_wide_fp6_roundtrip},
      {"Wide.Bf6RoundTrip", test_wide_bf6_roundtrip},
      {"Scale.Variation", test_scale_variation},
      {"Edge.NanInfSaturation", test_nan_inf_saturation},
  };

  for (auto &tc : tests) {
    printf("[ RUN    ] %s\n", tc.name);
    int result = tc.func();
    if (result == 0)
      pass(tc.name);
    else
      fail(tc.name, "see above");
  }

  printf("\n%d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
  return g_fail > 0 ? 1 : 0;
}
