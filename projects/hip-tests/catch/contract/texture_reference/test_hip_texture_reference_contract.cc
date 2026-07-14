/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

// Deprecated texture-reference (hipTexRef*) scalar-state contracts. These APIs
// are deprecated; the build sets -Wno-deprecated-declarations globally, so no
// local pragma is required.
//
// Every test is image-gated via CHECK_IMAGE_SUPPORT: on devices/runtime paths
// without image/array support each test skips (as on the local WSL2 iGPU) and
// runs on image-capable datacenter GPUs. Coverage is limited to the scalar
// per-reference state that a stack textureReference round-trips reliably on the
// AMD backend (address mode, filter mode, flags, format, max anisotropy). The
// bound-resource and bind-texture entry points require a texture bound to a
// device texture symbol rather than a stack reference (they return
// hipErrorInvalidSymbol on a stack reference), and the border-color and
// mipmap-parameter getters do not read the value back on this backend, so those
// are intentionally not covered here. Each test also treats hipErrorNotSupported
// at its key call as a graceful skip.

namespace {
// True when the runtime reports the specific hipTexRef* entry point is not
// implemented on this backend (as opposed to a genuine contract violation).
bool IsUnsupported(hipError_t status) { return status == hipErrorNotSupported; }
}  // namespace

// ---------------------------------------------------------------------------
// Pure set/get round-trips on a stack textureReference. These exercise the
// runtime's per-reference state store: set a value, read it back, assert it
// round-trips. Each guards its set call against hipErrorNotSupported.
// ---------------------------------------------------------------------------

HIP_TEST_CASE(Contract_TextureReference_SetGetAddressMode_RoundTrips) {
  CHECK_IMAGE_SUPPORT;

  textureReference tex_ref{};
  const hipError_t status =
      hipTexRefSetAddressMode(&tex_ref, 0, hipAddressModeMirror);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetAddressMode is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  hipTextureAddressMode returned = hipAddressModeWrap;
  const hipError_t get_status = hipTexRefGetAddressMode(&returned, &tex_ref, 0);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetAddressMode is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(returned == hipAddressModeMirror);
}

HIP_TEST_CASE(Contract_TextureReference_SetGetFilterMode_RoundTrips) {
  CHECK_IMAGE_SUPPORT;

  textureReference tex_ref{};
  const hipError_t status =
      hipTexRefSetFilterMode(&tex_ref, hipFilterModeLinear);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetFilterMode is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  hipTextureFilterMode returned = hipFilterModePoint;
  const hipError_t get_status = hipTexRefGetFilterMode(&returned, &tex_ref);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetFilterMode is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(returned == hipFilterModeLinear);
}

HIP_TEST_CASE(Contract_TextureReference_SetGetFlags_RoundTrips) {
  CHECK_IMAGE_SUPPORT;

  textureReference tex_ref{};
  const unsigned int flags = HIP_TRSF_READ_AS_INTEGER;
  const hipError_t status = hipTexRefSetFlags(&tex_ref, flags);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetFlags is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  unsigned int returned = 0;
  const hipError_t get_status = hipTexRefGetFlags(&returned, &tex_ref);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetFlags is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(returned == flags);
}

HIP_TEST_CASE(Contract_TextureReference_SetGetFormat_RoundTrips) {
  CHECK_IMAGE_SUPPORT;

  textureReference tex_ref{};
  const hipArray_Format format = HIP_AD_FORMAT_UNSIGNED_INT8;
  const int num_channels = 1;
  const hipError_t status = hipTexRefSetFormat(&tex_ref, format, num_channels);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetFormat is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  hipArray_Format returned_format = HIP_AD_FORMAT_UNSIGNED_INT8;
  int returned_channels = 0;
  const hipError_t get_status =
      hipTexRefGetFormat(&returned_format, &returned_channels, &tex_ref);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetFormat is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(returned_format == format);
  REQUIRE(returned_channels == num_channels);
}

HIP_TEST_CASE(Contract_TextureReference_SetGetMaxAnisotropy_RoundTrips) {
  CHECK_IMAGE_SUPPORT;

  textureReference tex_ref{};
  const unsigned int max_aniso = 4;
  const hipError_t status = hipTexRefSetMaxAnisotropy(&tex_ref, max_aniso);
  if (IsUnsupported(status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefSetMaxAnisotropy is not supported by this runtime path.");
  }
  HIP_CHECK(status);

  int returned = 0;
  const hipError_t get_status = hipTexRefGetMaxAnisotropy(&returned, &tex_ref);
  if (IsUnsupported(get_status)) {
    (void)hipGetLastError();
    HIP_SKIP_TEST("hipTexRefGetMaxAnisotropy is not supported by this runtime path.");
  }
  HIP_CHECK(get_status);

  REQUIRE(returned == static_cast<int>(max_aniso));
}
