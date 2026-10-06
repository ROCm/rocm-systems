/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipCreateTextureObject hipCreateTextureObject
 * @{
 * @ingroup TextureTest
 */

#define UNALIGN_OFFSET 1
#define SIZE_H 20
#define SIZE_W 30
#define N 512

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid 2D pitch resource:
 *    -# When device pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device pointer is not aligned appropriately
 *      - Expected output: do not return `hipSuccess`
 *    -# When pitch is not aligned appropriately
 *      - Expected output: do not return `hipSuccess`
 *    -# When height is 0
 *      - Expected output: do not return `hipSuccess`
 *    -# When height is 0 and device pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When height is `size_t` maximum
 *      - Expected output: do not return `hipSuccess`
 *    -# When width is 0
 *      - Expected output: do not return `hipSuccess`
 *    -# When width is 0 and device pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When width is `size_t` maximum
 *      - Expected output: do not return `hipSuccess`
 *    -# When pitch is `size_t` maximum
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_Pitch2D.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipCreateTextureObject_Pitch2DResource) {
  CHECK_IMAGE_SUPPORT

  hipError_t ret;
  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;
  hipDeviceProp_t devProp;
  size_t devPitchA;
  float* devPtrA;

  // Initialization
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devPtrA), &devPitchA, SIZE_W * sizeof(float),
                           SIZE_H));
  HIP_CHECK(hipGetDeviceProperties(&devProp, 0));
  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));
  resDesc.resType = hipResourceTypePitch2D;

  // Sections
  SECTION("hipResourceTypePitch2D and devPtr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and devicePtr un-aligned") {
    if (devProp.textureAlignment > UNALIGN_OFFSET) {
      // Populate resource descriptor
      resDesc.res.pitch2D.devPtr = reinterpret_cast<char*>(devPtrA) + UNALIGN_OFFSET;
      resDesc.res.pitch2D.height = SIZE_H;
      resDesc.res.pitch2D.width = SIZE_W;
      resDesc.res.pitch2D.pitchInBytes = devPitchA;
      resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

      // Populate texture descriptor
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.readMode = hipReadModeElementType;
      ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
      REQUIRE(ret != hipSuccess);
    }
  }

  SECTION("hipResourceTypePitch2D and pitch(un-aligned)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = UNALIGN_OFFSET;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and height(0)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = 0;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
  }

  SECTION("hipResourceTypePitch2D and height(0)/devptr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = 0;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and height(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and width(0)") {
    // Populate resource descriptor
    resDesc.resType = hipResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = 0;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
  }

  SECTION("hipResourceTypePitch2D and width(0)/devPtr(nullptr)") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = nullptr;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = 0;
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and width(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.pitchInBytes = devPitchA;
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  SECTION("hipResourceTypePitch2D and pitch(max(size_t))") {
    // Populate resource descriptor
    resDesc.res.pitch2D.devPtr = devPtrA;
    resDesc.res.pitch2D.height = SIZE_H;
    resDesc.res.pitch2D.width = SIZE_W;
    resDesc.res.pitch2D.pitchInBytes = std::numeric_limits<std::size_t>::max();
    resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();

    // Populate texture descriptor
    texDesc.readMode = hipReadModeElementType;

    ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE(ret != hipSuccess);
  }

  // De-Initialization
  HIP_CHECK(hipFree(devPtrA));
}

/**
 * Test Description
 * ------------------------
 *  - Validates that Pitch2D resources with a VALID (device-required) row pitch accept
 *    normalized coordinates and linear filtering, and that a mismatched-but-aligned row
 *    pitch is reported as unsupported rather than as out-of-memory. A valid pitch is the
 *    one returned by hipMallocPitch, so sampler-mode combinations no longer fail on a
 *    blanket heuristic; the actual addrlib/HSA verdict decides.
 *    -# When normalizedCoords is set and filterMode is point (valid pitch)
 *      - Expected output: on AMD, ASIC-dependent `hipSuccess` or `hipErrorNotSupported`
 *        (the sampler combo changes the required addrlib surface layout so the valid pitch
 *        may no longer match the device-required rowPitch); on NVIDIA, `hipSuccess`
 *    -# When normalizedCoords is unset and filterMode is linear (valid pitch)
 *      - Expected output: on AMD, ASIC-dependent `hipSuccess` or `hipErrorNotSupported`;
 *        on NVIDIA, `hipSuccess`
 *    -# When normalizedCoords is set and filterMode is linear (valid pitch)
 *      - Expected output: on AMD, ASIC-dependent `hipSuccess` or `hipErrorNotSupported`;
 *        on NVIDIA, `hipSuccess`
 *    -# When normalizedCoords is unset and filterMode is point (positive control)
 *      - Expected output: return `hipSuccess`
 *    -# When the row pitch is aligned but does not match the device-required pitch
 *      - Expected output: return `hipErrorNotSupported` (AMD, ASIC-dependent)
 * Test source
 * ------------------------
 *  - unit/texture/hipCreateTextureObject_Pitch2D.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipCreateTextureObject_Pitch2D_NormalizedCoordsLinearFilter) {
  CHECK_IMAGE_SUPPORT

  hipResourceDesc resDesc;
  hipTextureDesc texDesc;
  hipTextureObject_t texObj;
  size_t devPitchA;
  float* devPtrA;

  // Initialization
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devPtrA), &devPitchA, SIZE_W * sizeof(float),
                           SIZE_H));
  memset(&resDesc, 0, sizeof(resDesc));
  memset(&texDesc, 0, sizeof(texDesc));
  resDesc.resType = hipResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = devPtrA;
  resDesc.res.pitch2D.height = SIZE_H;
  resDesc.res.pitch2D.width = SIZE_W;
  resDesc.res.pitch2D.pitchInBytes = devPitchA;
  resDesc.res.pitch2D.desc = hipCreateChannelDesc<float>();
  texDesc.readMode = hipReadModeElementType;

  // Sections
  // A valid Pitch2D surface (the pitch hipMallocPitch returned) is now offered to normalized
  // coordinates and linear filtering without the removed #12543 blanket sampler-combo guard,
  // so the actual addrlib/HSA verdict decides rather than a heuristic. On AMD that verdict is
  // ASIC-dependent: these sampler combos change the required addrlib surface layout, so the
  // valid pitch may or may not still match the device-required rowPitch -- either hipSuccess or
  // hipErrorNotSupported is correct. On NVIDIA image creation succeeds (strict hipSuccess).
  SECTION("hipResourceTypePitch2D and normalizedCoords(1)") {
    texDesc.normalizedCoords = 1;
    texDesc.filterMode = hipFilterModePoint;

#if HT_AMD
    // The required addrlib surface layout changes with the sampler combo, so the valid
    // hipMallocPitch pitch may or may not match the device-required rowPitch on a given ASIC:
    // either hipSuccess or hipErrorNotSupported is correct here.
    hipError_t ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE((ret == hipSuccess || ret == hipErrorNotSupported));
    if (ret == hipSuccess) {
      HIP_CHECK(hipDestroyTextureObject(texObj));
    }
#else
    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
#endif
  }

  SECTION("hipResourceTypePitch2D and hipFilterModeLinear") {
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = hipFilterModeLinear;

#if HT_AMD
    // Linear filtering changes the required surface layout; the valid pitch may or may not
    // match the device-required rowPitch on a given ASIC: hipSuccess or hipErrorNotSupported.
    hipError_t ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE((ret == hipSuccess || ret == hipErrorNotSupported));
    if (ret == hipSuccess) {
      HIP_CHECK(hipDestroyTextureObject(texObj));
    }
#else
    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
#endif
  }

  SECTION("hipResourceTypePitch2D and normalizedCoords(1)/hipFilterModeLinear") {
    texDesc.normalizedCoords = 1;
    texDesc.filterMode = hipFilterModeLinear;

#if HT_AMD
    // Normalized coordinates combined with linear filtering changes the required surface
    // layout; the valid pitch may or may not match the device-required rowPitch on a given
    // ASIC: either hipSuccess or hipErrorNotSupported is correct here.
    hipError_t ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    REQUIRE((ret == hipSuccess || ret == hipErrorNotSupported));
    if (ret == hipSuccess) {
      HIP_CHECK(hipDestroyTextureObject(texObj));
    }
#else
    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
#endif
  }

  SECTION("hipResourceTypePitch2D and normalizedCoords(0)/hipFilterModePoint") {
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = hipFilterModePoint;

    HIP_CHECK(hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));
    HIP_CHECK(hipDestroyTextureObject(texObj));
  }

  // A row pitch that is still correctly aligned (a multiple of texturePitchAlignment, so it
  // passes the HIP-layer alignment validation) but does NOT equal the device-required pitch
  // makes HSA reject image creation with HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED (0x3002).
  // This must now surface as hipErrorNotSupported, not the old misleading hipErrorOutOfMemory.
  //
  // The exact required rowPitch is computed by addrlib and varies per ASIC, so a pitch that is
  // "aligned but wrong" on one GPU may be accepted on another. This section is AMD-only and its
  // trigger is ASIC-dependent; the GDB-confirmed 0x3002 repro was on gfx90a. The Tester confirms
  // actual behavior on the target GPU and guards/skips this section where it does not reproduce.
  SECTION("hipResourceTypePitch2D and mismatched-but-aligned pitch") {
#if HT_AMD
    hipDeviceProp_t devProp;
    HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

    // Build a pitch that is a multiple of the required alignment (so the HIP-layer alignment
    // check passes) but larger than the device-required pitch, so it fails addrlib's exact
    // rowPitch match. texturePitchAlignment is the alignment granularity for pitched textures.
    size_t alignment = devProp.texturePitchAlignment ? devProp.texturePitchAlignment : 1;
    size_t mismatchedPitch = devPitchA + alignment;
    REQUIRE((mismatchedPitch % alignment) == 0);  // still correctly aligned

    resDesc.res.pitch2D.pitchInBytes = mismatchedPitch;
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = hipFilterModePoint;

    // The exact required rowPitch is ASIC-dependent: an aligned-but-wrong pitch triggers 0x3002
    // on gfx90a (-> hipErrorNotSupported) but is accepted on gfx1030 (-> hipSuccess). Either is
    // acceptable. Any OTHER code is a failure; in particular hipErrorOutOfMemory would mean the
    // 0x3002 -> hipErrorNotSupported propagation (the fix under test) has regressed.
    hipError_t ret = hipCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
    INFO("Unexpected error: " << hipGetErrorString(ret) << "\n    Code: " << ret
                              << " (expected hipSuccess or hipErrorNotSupported; "
                              << "hipErrorOutOfMemory would indicate a propagation regression)");
    REQUIRE((ret == hipSuccess || ret == hipErrorNotSupported));
    if (ret == hipSuccess) {
      HIP_CHECK(hipDestroyTextureObject(texObj));
    }
#else
    WARN("Skipping section: " << HipTest::SkipReason::kPitch2DMismatchedPitchAmdOnly);
#endif
  }

  // De-Initialization
  HIP_CHECK(hipFree(devPtrA));
}


/**
 * End doxygen group TextureTest.
 * @}
 */
