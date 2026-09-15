/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <vector>

/*
 * Unit tests for the OCP FP6 (E2M3 / E3M2) packed-32 types in hip_ext_ocp.h:
 *   __hipext_ocp_fp6x32_e2m3, __hipext_ocp_fp6x32_e3m2
 *
 * These types only exist in the 32-wide packed form, so representable values are
 * tiled across the 32 lanes. Hardware agnostic: assert "value in -> expected
 * value out". Covers round-trip, documented known values, scaled conversions,
 * fp16/bf16 sources, and host/device consistency.
 */

namespace {

// Representable non-negative magnitudes.
// E2M3: bias 1, 3 mantissa bits. Subnormals in [0,0.875] step 0.125, then
// normals up to max 7.5.
const std::vector<float> kE2M3 = {0.0f, 0.125f, 0.5f, 0.875f, 1.0f, 1.5f,
                                  2.0f, 3.0f,   4.0f, 6.0f,    7.5f};
// E3M2: bias 3, 2 mantissa bits. Min normal 0.25, max normal 28.0.
const std::vector<float> kE3M2 = {0.0f, 0.0625f, 0.25f, 0.5f,  1.0f,  2.0f,
                                  4.0f, 8.0f,    16.0f, 24.0f, 28.0f};

// Fill a 32-wide float vector by tiling `src`.
__amd_floatx32_storage_t tile32(const std::vector<float>& src) {
  __amd_floatx32_storage_t v;
  for (int i = 0; i < 32; ++i) v[i] = src[i % src.size()];
  return v;
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Representable E2M3/E3M2 values encoded into fp6x32 (scale 0) and decoded
 *    back must be recovered exactly across all 32 lanes.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp6.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp6_roundtrip_host) {
  const __amd_scale_t scale = 0;
  SECTION("e2m3") {
    __amd_floatx32_storage_t in = tile32(kE2M3);
    __hipext_ocp_fp6x32_e2m3 packed(in, 0u /*round*/, scale);
    __amd_floatx32_storage_t out = packed.get_scaled_floatx32(scale);
    for (int i = 0; i < 32; ++i) {
      INFO("lane " << i << " in " << in[i] << " out " << out[i]);
      REQUIRE(out[i] == in[i]);
    }
  }
  SECTION("e3m2") {
    __amd_floatx32_storage_t in = tile32(kE3M2);
    __hipext_ocp_fp6x32_e3m2 packed(in, 0u /*round*/, scale);
    __amd_floatx32_storage_t out = packed.get_scaled_floatx32(scale);
    for (int i = 0; i < 32; ++i) {
      INFO("lane " << i << " in " << in[i] << " out " << out[i]);
      REQUIRE(out[i] == in[i]);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Documented boundary values (max/min normal) decode exactly.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp6.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp6_known_values_host) {
  const __amd_scale_t scale = 0;
  SECTION("e2m3 max/min normal") {
    std::vector<float> want = {7.5f, 1.0f, -7.5f, -1.0f, 0.125f, -0.125f};
    __amd_floatx32_storage_t in = tile32(want);
    __hipext_ocp_fp6x32_e2m3 packed(in, 0u /*round*/, scale);
    __amd_floatx32_storage_t out = packed.get_scaled_floatx32(scale);
    for (int i = 0; i < 32; ++i) REQUIRE(out[i] == in[i]);
  }
  SECTION("e3m2 max/min normal") {
    std::vector<float> want = {28.0f, 0.25f, -28.0f, -0.25f, 0.0625f, -0.0625f};
    __amd_floatx32_storage_t in = tile32(want);
    __hipext_ocp_fp6x32_e3m2 packed(in, 0u /*round*/, scale);
    __amd_floatx32_storage_t out = packed.get_scaled_floatx32(scale);
    for (int i = 0; i < 32; ++i) REQUIRE(out[i] == in[i]);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Encoding v = base * 2^s with scale s and decoding with the same scale
 *    recovers v when base is representable.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp6.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp6_scaled_roundtrip_host) {
  for (int s : {-2, -1, 0, 1, 2}) {
    __amd_scale_t scale = static_cast<__amd_scale_t>(s);
    std::vector<float> scaled;
    for (float base : kE2M3) scaled.push_back(std::ldexp(base, s));
    __amd_floatx32_storage_t in = tile32(scaled);
    __hipext_ocp_fp6x32_e2m3 packed(in, 0u /*round*/, scale);
    __amd_floatx32_storage_t out = packed.get_scaled_floatx32(scale);
    for (int i = 0; i < 32; ++i) {
      INFO("scale " << s << " lane " << i << " in " << in[i] << " out " << out[i]);
      REQUIRE(out[i] == in[i]);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - fp16 and bf16 sources produce the same fp6 decode as the float source for
 *    representable values.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp6.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp6_fp16_bf16_source_host) {
  const __amd_scale_t scale = 0;

  __amd_floatx32_storage_t fin = tile32(kE2M3);
  __hipext_ocp_fp6x32_e2m3 from_f(fin, 0u /*round*/, scale);
  __amd_floatx32_storage_t ref = from_f.get_scaled_floatx32(scale);

  __amd_fp16x32_storage_t hin;
  __amd_bf16x32_storage_t bin;
  for (int i = 0; i < 32; ++i) {
    hin[i] = static_cast<__amd_fp16_storage_t>(fin[i]);
    bin[i] = static_cast<__amd_bf16_storage_t>(fin[i]);
  }
  __hipext_ocp_fp6x32_e2m3 from_h(hin, scale);
  __hipext_ocp_fp6x32_e2m3 from_b(bin, scale);
  __amd_floatx32_storage_t oh = from_h.get_scaled_floatx32(scale);
  __amd_floatx32_storage_t ob = from_b.get_scaled_floatx32(scale);

  for (int i = 0; i < 32; ++i) {
    INFO("lane " << i << " ref " << ref[i] << " fp16 " << oh[i] << " bf16 " << ob[i]);
    REQUIRE(oh[i] == ref[i]);
    REQUIRE(ob[i] == ref[i]);
  }
}

// ---- Device consistency -----------------------------------------------------

__global__ void fp6_e2m3_roundtrip_kernel(const float* in, float* out) {
  __amd_floatx32_storage_t v;
  for (int i = 0; i < 32; ++i) v[i] = in[i];
  __hipext_ocp_fp6x32_e2m3 t(v, 0u /*round*/, 0 /*scale*/);
  __amd_floatx32_storage_t r = t.get_scaled_floatx32(0);
  for (int i = 0; i < 32; ++i) out[i] = r[i];
}

/**
 * Test Description
 * ------------------------
 *  - Device fp6 conversion matches host for representable E2M3 values.
 * Test source
 * ------------------------
 *  - /unit/deviceLib/ext_ocp_fp6.cc
 */
HIP_TEST_CASE(Unit_ext_ocp_fp6_device_matches_host) {
  std::vector<float> in(32);
  for (int i = 0; i < 32; ++i) in[i] = kE2M3[i % kE2M3.size()];

  float* d_in = nullptr;
  float* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_in, sizeof(float) * 32));
  HIP_CHECK(hipMalloc(&d_out, sizeof(float) * 32));
  HIP_CHECK(hipMemcpy(d_in, in.data(), sizeof(float) * 32, hipMemcpyHostToDevice));

  fp6_e2m3_roundtrip_kernel<<<1, 1>>>(d_in, d_out);
  HIP_CHECK(hipGetLastError());

  std::vector<float> dev(32, 0.0f);
  HIP_CHECK(hipMemcpy(dev.data(), d_out, sizeof(float) * 32, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  __amd_floatx32_storage_t hv = tile32(kE2M3);
  __hipext_ocp_fp6x32_e2m3 h(hv, 0u /*round*/, 0 /*scale*/);
  __amd_floatx32_storage_t host = h.get_scaled_floatx32(0);

  for (int i = 0; i < 32; ++i) {
    INFO("lane " << i << " in " << in[i] << " host " << host[i] << " dev " << dev[i]);
    REQUIRE(dev[i] == host[i]);
    REQUIRE(dev[i] == in[i]);
  }

  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
}
