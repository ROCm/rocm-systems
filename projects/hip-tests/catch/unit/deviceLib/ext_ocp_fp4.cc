/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <utility>
#include <vector>

/*
 * Unit tests for the OCP FP4 (E2M1) packed type in hip_ext_ocp.h:
 *   __hipext_ocp_fp4x2_e2m1
 *
 * Hardware agnostic: assert "value in -> expected value out". Covers round-trip
 * over all representable values, documented known values, scaled conversions,
 * stochastic rounding bounds, fp16/bf16 sources, and host/device consistency.
 */

namespace {

// The 8 non-negative E2M1 magnitudes (bit patterns 0..7); 8..15 are negatives.
const std::vector<float> kFp4Magnitudes = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

std::vector<float> all_representable_fp4() {
  std::vector<float> v = kFp4Magnitudes;
  for (float m : kFp4Magnitudes)
    if (m != 0.0f) v.push_back(-m);
  return v;
}

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
 *  - Every representable E2M1 value packed into an fp4x2 and decoded back
 *    (scale 0) must be recovered exactly.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_roundtrip_host) {
  auto reps = all_representable_fp4();
  const __amd_scale_t scale = 0;
  for (float a : reps) {
    for (float b : reps) {
      __hipext_ocp_fp4x2_e2m1 packed(a, b, scale);
      __amd_floatx2_storage_t out = packed.get_scaled_floatx2(scale);
      INFO("in (" << a << "," << b << ") out (" << out[0] << "," << out[1] << ")");
      REQUIRE(out[0] == a);
      REQUIRE(out[1] == b);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Documented E2M1 values (max normal 6.0, min normal 1.0, subnormal 0.5)
 *    decode exactly.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_known_values_host) {
  const __amd_scale_t scale = 0;
  std::vector<std::pair<float, float>> pairs = {
      {6.0f, -6.0f}, {1.0f, -1.0f}, {0.5f, -0.5f}, {0.0f, 3.0f}};
  for (auto [a, b] : pairs) {
    __hipext_ocp_fp4x2_e2m1 packed(a, b, scale);
    __amd_floatx2_storage_t out = packed.get_scaled_floatx2(scale);
    INFO("in (" << a << "," << b << ")");
    REQUIRE(out[0] == a);
    REQUIRE(out[1] == b);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Encoding v = base * 2^s with scale s and decoding with the same scale
 *    recovers v when base is representable.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_scaled_roundtrip_host) {
  for (int s : {-2, -1, 0, 1, 2}) {
    __amd_scale_t scale = static_cast<__amd_scale_t>(s);
    for (float base : kFp4Magnitudes) {
      float v = std::ldexp(base, s);  // base * 2^s
      __hipext_ocp_fp4x2_e2m1 packed(v, -v, scale);
      __amd_floatx2_storage_t out = packed.get_scaled_floatx2(scale);
      INFO("scale " << s << " base " << base << " v " << v);
      REQUIRE(out[0] == v);
      REQUIRE(out[1] == -v);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Stochastic rounding: representable inputs are stable for any seed;
 *    non-representable inputs round to a nearest neighbour.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_stochastic_rounding_host) {
  std::vector<unsigned> seeds = {0u, 1u, 0x1234u, 0x80000000u, 0xdeadbeefu, 0xffffffffu};
  auto reps = all_representable_fp4();
  const __amd_scale_t scale = 0;

  SECTION("representable is stable under SR") {
    for (float v : kFp4Magnitudes) {
      __hipext_ocp_fp4x2_e2m1 exact(v, v, scale);
      __amd_floatx2_storage_t e = exact.get_scaled_floatx2(scale);
      for (unsigned seed : seeds) {
        __hipext_ocp_fp4x2_e2m1 sr(__amd_floatx2_storage_t{v, v}, seed, scale);
        __amd_floatx2_storage_t o = sr.get_scaled_floatx2(scale);
        INFO("v " << v << " seed 0x" << std::hex << seed);
        REQUIRE(o[0] == e[0]);
        REQUIRE(o[1] == e[1]);
      }
    }
  }

  SECTION("non-representable stays within neighbours") {
    for (float v : {2.5f, 0.75f, 3.7f, 5.1f}) {
      auto [lo, hi] = bracket(reps, v);
      for (unsigned seed : seeds) {
        __hipext_ocp_fp4x2_e2m1 sr(__amd_floatx2_storage_t{v, -v}, seed, scale);
        __amd_floatx2_storage_t o = sr.get_scaled_floatx2(scale);
        INFO("v " << v << " seed 0x" << std::hex << seed << " -> " << o[0] << " lo " << lo << " hi "
                  << hi);
        REQUIRE((o[0] == lo || o[0] == hi));
        REQUIRE((o[1] == -hi || o[1] == -lo));
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - fp16 and bf16 sources produce the same encoding as the float source for
 *    representable values.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_fp16_bf16_source_host) {
  const __amd_scale_t scale = 0;
  std::vector<std::pair<float, float>> pairs = {{0.5f, 1.0f}, {1.5f, 2.0f}, {3.0f, 6.0f}};
  for (auto [a, b] : pairs) {
    __hipext_ocp_fp4x2_e2m1 from_f(a, b, scale);
    __amd_floatx2_storage_t ref = from_f.get_scaled_floatx2(scale);

    __amd_fp16x2_storage_t hv{static_cast<__amd_fp16_storage_t>(a),
                              static_cast<__amd_fp16_storage_t>(b)};
    __amd_bf16x2_storage_t bv{static_cast<__amd_bf16_storage_t>(a),
                              static_cast<__amd_bf16_storage_t>(b)};
    __hipext_ocp_fp4x2_e2m1 from_h(hv, scale);
    __hipext_ocp_fp4x2_e2m1 from_b(bv, scale);
    __amd_floatx2_storage_t oh = from_h.get_scaled_floatx2(scale);
    __amd_floatx2_storage_t ob = from_b.get_scaled_floatx2(scale);

    INFO("in (" << a << "," << b << ")");
    REQUIRE(oh[0] == ref[0]);
    REQUIRE(oh[1] == ref[1]);
    REQUIRE(ob[0] == ref[0]);
    REQUIRE(ob[1] == ref[1]);
  }
}

// ---- Device consistency -----------------------------------------------------

__global__ void fp4_roundtrip_kernel(const float2* in, float2* out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    __hipext_ocp_fp4x2_e2m1 t(in[i].x, in[i].y, 0);
    __amd_floatx2_storage_t r = t.get_scaled_floatx2(0);
    out[i] = float2(r[0], r[1]);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Device conversion matches host for all representable value pairs.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp4.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp4_device_matches_host) {
  auto reps = all_representable_fp4();
  std::vector<float2> in;
  for (float a : reps)
    for (float b : reps) in.push_back(float2(a, b));
  size_t n = in.size();

  float2* d_in = nullptr;
  float2* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float2) * n));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float2) * n));
  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float2) * n, hipMemcpyHostToDevice));

  fp4_roundtrip_kernel<<<(n / 256) + 1, 256>>>(d_in, d_out, n);
  HIP_CHECK(hipGetLastError());

  std::vector<float2> dev(n, float2(0.0f, 0.0f));
  HIP_CHECK(hipMemcpy(dev.data(), d_out, sizeof(float2) * n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (size_t i = 0; i < n; ++i) {
    INFO("in (" << in[i].x << "," << in[i].y << ") dev (" << dev[i].x << "," << dev[i].y << ")");
    REQUIRE(dev[i].x == in[i].x);
    REQUIRE(dev[i].y == in[i].y);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
