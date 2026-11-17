/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

/*
 * Unit tests for the OCP FP8 (E4M3 / E5M2) types defined in hip_ext_ocp.h:
 *   __hipext_ocp_fp8_e4m3, __hipext_ocp_fp8_e5m2,
 *   __hipext_ocp_fp8x2_e4m3, __hipext_ocp_fp8x2_e5m2
 *
 * These tests are hardware agnostic: they only assert that "value in ->
 * expected value out". They cover round-trip idempotence over all representable
 * bit patterns, documented known values, scaled conversions, stochastic
 * rounding bounds, and fp16/bf16 sources. A final section runs the same
 * conversion on device and requires it to match the host result.
 */

namespace {

// Is a raw fp8 bit pattern NaN / Inf for the given interpretation?
bool is_e4m3_special(unsigned i) { return (i & 0x7fu) == 0x7fu; }  // 0x7f, 0xff are NaN
// 0x7c, 0xfc are Inf; 0x7d-0x7f, 0xfd-0xff are NaN
bool is_e5m2_special(unsigned i) { return (i & 0x7fu) >= 0x7cu; }

// All representable (finite, non-NaN) fp8 values for an interpretation.
template <bool is_e4m3> std::vector<float> all_representable_fp8() {
  std::vector<float> ret;
  for (unsigned i = 0; i <= 0xffu; ++i) {
    if (is_e4m3 ? is_e4m3_special(i) : is_e5m2_special(i)) continue;
    if (is_e4m3) {
      __hipext_ocp_fp8_e4m3 t;
      t.__x = static_cast<__amd_fp8_storage_t>(i);
      ret.push_back(static_cast<float>(t));
    } else {
      __hipext_ocp_fp8_e5m2 t;
      t.__x = static_cast<__amd_fp8_storage_t>(i);
      ret.push_back(static_cast<float>(t));
    }
  }
  return ret;
}

// Nearest representable values bracketing v: {lo <= v, hi >= v}.
std::pair<float, float> bracket(const std::vector<float>& reps, float v) {
  float lo = -INFINITY, hi = INFINITY;
  for (float r : reps) {
    if (r <= v && r > lo) lo = r;
    if (r >= v && r < hi) hi = r;
  }
  return {lo, hi};
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Every representable fp8 bit pattern must survive a fp8 -> float -> fp8
 *    round-trip unchanged (encode is the exact inverse of decode).
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_roundtrip_host) {
  SECTION("e4m3") {
    for (unsigned i = 0; i <= 0xffu; ++i) {
      if (is_e4m3_special(i)) continue;
      __hipext_ocp_fp8_e4m3 in;
      in.__x = static_cast<__amd_fp8_storage_t>(i);
      float f = in;
      __hipext_ocp_fp8_e4m3 back(f);
      INFO("pattern 0x" << std::hex << i << " -> " << f << " -> 0x" << (unsigned)back.__x);
      REQUIRE(back.__x == in.__x);
    }
  }
  SECTION("e5m2") {
    for (unsigned i = 0; i <= 0xffu; ++i) {
      if (is_e5m2_special(i)) continue;
      __hipext_ocp_fp8_e5m2 in;
      in.__x = static_cast<__amd_fp8_storage_t>(i);
      float f = in;
      __hipext_ocp_fp8_e5m2 back(f);
      INFO("pattern 0x" << std::hex << i << " -> " << f << " -> 0x" << (unsigned)back.__x);
      REQUIRE(back.__x == in.__x);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Assert documented reference values (min/max normal, unit, halves) from the
 *    format tables in amd_hip_ocp_fp.hpp decode to exact floats.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_known_values_host) {
  SECTION("e4m3") {
    // {bit pattern, expected float}
    std::vector<std::pair<unsigned, float>> table = {
        {0x00, 0.0f},        {0x38, 1.0f},  {0x40, 2.0f},
        {0x30, 0.5f},        {0x7E, 448.0f} /*max normal*/,
        {0x08, 0.015625f} /*min normal 2^-6*/};
    for (auto [pat, exp] : table) {
      __hipext_ocp_fp8_e4m3 t;
      t.__x = static_cast<__amd_fp8_storage_t>(pat);
      float f = t;
      INFO("e4m3 pattern 0x" << std::hex << pat);
      REQUIRE(f == exp);
    }
  }
  SECTION("e5m2") {
    std::vector<std::pair<unsigned, float>> table = {{0x3C, 1.0f},
                                                     {0x40, 2.0f},
                                                     {0x38, 0.5f},
                                                     {0x04, 6.10352e-05f} /*min normal 2^-14*/,
                                                     {0x7B, 57344.0f} /*max normal*/,
                                                     {0xFB, -57344.0f}};
    for (auto [pat, exp] : table) {
      __hipext_ocp_fp8_e5m2 t;
      t.__x = static_cast<__amd_fp8_storage_t>(pat);
      float f = t;
      INFO("e5m2 pattern 0x" << std::hex << pat);
      REQUIRE(f == Catch::Approx(exp));
    }
    // 0x7C / 0xFC are +-Inf, one step above the largest finite value.
    __hipext_ocp_fp8_e5m2 inf;
    inf.__x = 0x7C;
    REQUIRE(static_cast<float>(inf) == INFINITY);
    inf.__x = 0xFC;
    REQUIRE(static_cast<float>(inf) == -INFINITY);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Decoding with an E8M0 scale multiplies the base value by 2^scale.
 *    get_scaled_float/fp16/bf16 must reflect that exactly.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_scaled_decode_host) {
  // e4m3 pattern 0x38 == 1.0
  __hipext_ocp_fp8_e4m3 one;
  one.__x = 0x38;
  for (int s : {-2, -1, 0, 1, 2, 3}) {
    float expected = std::ldexp(1.0f, s);  // 1.0 * 2^s
    INFO("scale " << s);
    REQUIRE(one.get_scaled_float(static_cast<__amd_scale_t>(s)) == expected);
    REQUIRE(static_cast<float>(one.get_scaled_fp16(static_cast<__amd_scale_t>(s))) == expected);
    REQUIRE(static_cast<float>(one.get_scaled_bf16(static_cast<__amd_scale_t>(s))) == expected);
  }
}

/**
 * Test Description
 * ------------------------
 *  - fp8x2 packs two lanes and supports a non-SR scaled encode. Encoding
 *    (v0,v1) with scale s and decoding with the same scale is a round-trip when
 *    v/2^s is representable.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8x2_scaled_roundtrip_host) {
  const int s = 1;
  // base values are representable in e4m3/e5m2; v = base * 2^s.
  std::vector<float2> vals = {float2(2.0f, 4.0f), float2(0.5f, 1.0f), float2(8.0f, 16.0f)};
  SECTION("e4m3") {
    for (auto v : vals) {
      __hipext_ocp_fp8x2_e4m3 packed(__amd_floatx2_storage_t{v.x, v.y},
                                     static_cast<__amd_scale_t>(s));
      __amd_floatx2_storage_t out = packed.get_scaled_floatx2(static_cast<__amd_scale_t>(s));
      INFO("in (" << v.x << "," << v.y << ") out (" << out[0] << "," << out[1] << ")");
      REQUIRE(out[0] == v.x);
      REQUIRE(out[1] == v.y);
    }
  }
  SECTION("e5m2") {
    for (auto v : vals) {
      __hipext_ocp_fp8x2_e5m2 packed(__amd_floatx2_storage_t{v.x, v.y},
                                     static_cast<__amd_scale_t>(s));
      __amd_floatx2_storage_t out = packed.get_scaled_floatx2(static_cast<__amd_scale_t>(s));
      INFO("in (" << v.x << "," << v.y << ") out (" << out[0] << "," << out[1] << ")");
      REQUIRE(out[0] == v.x);
      REQUIRE(out[1] == v.y);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Stochastic rounding: exactly representable inputs must be unchanged for any
 *    seed; non-representable inputs must round to one of the two nearest
 *    representable neighbours.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_stochastic_rounding_host) {
  std::vector<unsigned> seeds = {0u, 1u, 0x1234u, 0x80000000u, 0xdeadbeefu, 0xffffffffu};
  auto reps = all_representable_fp8<true>();  // e4m3

  SECTION("representable is stable under SR") {
    for (float v : {0.5f, 1.0f, 2.0f, 4.0f}) {
      __hipext_ocp_fp8_e4m3 exact(v);
      for (unsigned seed : seeds) {
        __hipext_ocp_fp8_e4m3 sr(v, seed);
        INFO("v " << v << " seed 0x" << std::hex << seed);
        REQUIRE(sr.__x == exact.__x);
      }
    }
  }

  SECTION("non-representable stays within neighbours") {
    for (float v : {1.0625f, 3.3f, 5.7f, 0.6f}) {
      auto [lo, hi] = bracket(reps, v);
      for (unsigned seed : seeds) {
        __hipext_ocp_fp8_e4m3 sr(v, seed);
        float f = sr;
        INFO("v " << v << " seed 0x" << std::hex << seed << " -> " << f << " lo " << lo << " hi "
                  << hi);
        REQUIRE((f == lo || f == hi));
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Constructing an fp8 from fp16/bf16 sources yields the same encoding as the
 *    float source for representable values (SR is a no-op there).
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_fp16_bf16_source_host) {
  const unsigned seed = 0u;
  const __amd_scale_t scale = 0;
  std::vector<float> vals = {0.5f, 1.0f, 1.5f, 2.0f, 4.0f};
  for (float v : vals) {
    __hipext_ocp_fp8_e4m3 from_float_v(v);

    __amd_fp16_storage_t hv = static_cast<__amd_fp16_storage_t>(v);
    __amd_bf16_storage_t bv = static_cast<__amd_bf16_storage_t>(v);
    __hipext_ocp_fp8_e4m3 from_fp16(hv, seed, scale);
    __hipext_ocp_fp8_e4m3 from_bf16(bv, seed, scale);

    INFO("v " << v);
    REQUIRE(from_fp16.__x == from_float_v.__x);
    REQUIRE(from_bf16.__x == from_float_v.__x);
  }
}

/**
 * Test Description
 * ------------------------
 *  - A binary16 subnormal has no implicit leading 1. Converting it to fp8 must
 *    give the same encoding as converting the same value from float, for every
 *    binary16 subnormal (both signs). Known values: 2^-24 -> 0x00 and
 *    1.5 * 2^-15 -> 0x03 for E5M2.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_fp16_subnormal_source_host) {
  auto fp16_from_bits = [](unsigned bits) {
    uint16_t u = static_cast<uint16_t>(bits);
    __amd_fp16_storage_t h;
    std::memcpy(&h, &u, sizeof(h));
    return h;
  };
  auto encode_fp16 = [&](unsigned bits, __amd_fp8_interpretation_t interp) {
    __amd_fp16_storage_t h = fp16_from_bits(bits);
    return __amd_cvt_fp16x2_to_fp8x2(__amd_fp16x2_storage_t{h, h}, interp) & 0xffu;
  };
  auto encode_float = [&](unsigned bits, __amd_fp8_interpretation_t interp) {
    float f = static_cast<float>(fp16_from_bits(bits));
    return __amd_cvt_floatx2_to_fp8x2(__amd_floatx2_storage_t{f, f}, interp) & 0xffu;
  };

  for (auto interp : {__AMD_OCP_E4M3, __AMD_OCP_E5M2}) {
    for (unsigned sign : {0x0000u, 0x8000u}) {
      for (unsigned bits = 1; bits <= 0x3ffu; ++bits) {
        INFO("interpretation " << interp << " binary16 0x" << std::hex << (sign | bits));
        REQUIRE(encode_fp16(sign | bits, interp) == encode_float(sign | bits, interp));
      }
    }
  }

  REQUIRE(encode_fp16(0x0001, __AMD_OCP_E5M2) == 0x00);  // 2^-24
  REQUIRE(encode_fp16(0x0300, __AMD_OCP_E5M2) == 0x03);  // 1.5 * 2^-15
  REQUIRE(encode_fp16(0x03ff, __AMD_OCP_E5M2) == 0x04);  // rounds up to 2^-14, the min normal
  REQUIRE(encode_fp16(0x8300, __AMD_OCP_E5M2) == 0x83);
}

/**
 * Test Description
 * ------------------------
 *  - The host conversion saturates: a value that rounds past the largest finite
 *    value, and +-Inf, give +-max (E4M3: 448 = 0x7e, since 0x7f is NaN; E5M2:
 *    57344 = 0x7b), with round-to-nearest-even and with stochastic rounding.
 *    NaN stays NaN.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_overflow_saturates_host) {
  SECTION("e4m3") {
    // 464 is the tie between 448 and 480 and rounds to the even 448; anything above overflows
    for (float v : {448.0f, 449.0f, 464.0f, 465.0f, 500.0f, 1e5f, 3.0e38f, INFINITY}) {
      INFO("v " << v);
      REQUIRE(__hipext_ocp_fp8_e4m3(v).__x == 0x7e);
      REQUIRE(__hipext_ocp_fp8_e4m3(-v).__x == 0xfe);
      for (unsigned seed : {0u, 0x12345678u, 0xffffffffu}) {
        REQUIRE(__hipext_ocp_fp8_e4m3(v, seed).__x == 0x7e);
        REQUIRE(__hipext_ocp_fp8_e4m3(-v, seed).__x == 0xfe);
      }
    }
    REQUIRE((__hipext_ocp_fp8_e4m3(NAN).__x & 0x7f) == 0x7f);

    __amd_fp8x2_storage_t packed =
        __amd_cvt_floatx2_to_fp8x2(__amd_floatx2_storage_t{500.0f, INFINITY}, __AMD_OCP_E4M3);
    REQUIRE((packed & 0xff) == 0x7e);
    REQUIRE((packed >> 8) == 0x7e);
  }
  SECTION("e5m2") {
    // 61440 is the tie between 57344 and 65536 and rounds up, so it overflows
    for (float v : {57344.0f, 60000.0f, 61440.0f, 1e5f, 3.0e38f, INFINITY}) {
      INFO("v " << v);
      REQUIRE(__hipext_ocp_fp8_e5m2(v).__x == 0x7b);
      REQUIRE(__hipext_ocp_fp8_e5m2(-v).__x == 0xfb);
      for (unsigned seed : {0u, 0x12345678u, 0xffffffffu}) {
        REQUIRE(__hipext_ocp_fp8_e5m2(v, seed).__x == 0x7b);
        REQUIRE(__hipext_ocp_fp8_e5m2(-v, seed).__x == 0xfb);
      }
    }
    REQUIRE((__hipext_ocp_fp8_e5m2(NAN).__x & 0x7f) > 0x7c);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Every fp8 value is exactly representable in binary16, so decoding to fp16
 *    must give the same value as decoding to float for all 256 bit patterns.
 *    E5M2 subnormals (0x01-0x03) lie in the binary16 subnormal range.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_to_fp16_matches_float_host) {
  for (auto interp : {__AMD_OCP_E4M3, __AMD_OCP_E5M2}) {
    for (unsigned c = 0; c <= 0xffu; ++c) {
      float f = __amd_cvt_fp8_to_float(static_cast<__amd_fp8_storage_t>(c), interp);
      float h =
          static_cast<float>(__amd_cvt_fp8_to_fp16(static_cast<__amd_fp8_storage_t>(c), interp));
      INFO("interpretation " << interp << " pattern 0x" << std::hex << c << " float " << f
                             << " fp16 " << h);
      if (std::isnan(f)) {
        REQUIRE(std::isnan(h));
      } else {
        REQUIRE(h == f);
        REQUIRE(std::signbit(h) == std::signbit(f));
      }
    }
  }

  // E5M2 min subnormal 2^-16 and 0x03 = 3 * 2^-16
  REQUIRE(static_cast<float>(__amd_cvt_fp8_to_fp16(0x01, __AMD_OCP_E5M2)) == 0x1p-16f);
  REQUIRE(static_cast<float>(__amd_cvt_fp8_to_fp16(0x83, __AMD_OCP_E5M2)) == -3 * 0x1p-16f);
}

// ---- Device consistency -----------------------------------------------------

__global__ void fp8_e4m3_roundtrip_kernel(const float* in, float* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    __hipext_ocp_fp8_e4m3 t(in[i]);
    out[i] = t;
  }
}

/**
 * Test Description
 * ------------------------
 *  - The device conversion path must return the same result as the host for the
 *    full set of representable e4m3 values.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp8.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp8_device_matches_host) {
  auto reps = all_representable_fp8<true>();  // decoded floats, all representable
  size_t n = reps.size();

  float* d_in = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * n));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * n));
  HIP_CHECK(hipMemcpy(d_in, reps.data(), sizeof(float) * n, hipMemcpyHostToDevice));

  fp8_e4m3_roundtrip_kernel<<<(n / 256) + 1, 256>>>(d_in, d_out, n);
  HIP_CHECK(hipGetLastError());

  std::vector<float> dev(n, 0.0f);
  HIP_CHECK(hipMemcpy(dev.data(), d_out, sizeof(float) * n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (size_t i = 0; i < n; ++i) {
    __hipext_ocp_fp8_e4m3 h(reps[i]);
    float host = h;
    INFO("in " << reps[i] << " host " << host << " dev " << dev[i]);
    REQUIRE(dev[i] == host);
    REQUIRE(dev[i] == reps[i]);  // representable -> exact
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
