/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <limits>
#include "hip/driver_types.h"

static constexpr size_t memsetVal{0x42};
static constexpr hipExtent validExtent{184, 57, 16};
static constexpr size_t height{validExtent.height};
static constexpr size_t width{validExtent.width};
static constexpr hipStream_t nullStream{nullptr};

inline void testHipMemsetApis(void* dst, int value, size_t sizeBytes,
                              hipError_t expectedReturn = hipErrorInvalidValue) {
  HIP_CHECK_ERROR(hipMemset(dst, value, sizeBytes), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetAsync(dst, value, sizeBytes, nullStream), expectedReturn);

  hipDeviceptr_t devicePtrDst = reinterpret_cast<hipDeviceptr_t>(dst);
  HIP_CHECK_ERROR(hipMemsetD32(devicePtrDst, value, sizeBytes), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetD32Async(devicePtrDst, value, sizeBytes, nullStream), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetD16(devicePtrDst, value, sizeBytes), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetD16Async(devicePtrDst, value, sizeBytes, nullStream), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetD8(devicePtrDst, value, sizeBytes), expectedReturn);
  HIP_CHECK_ERROR(hipMemsetD8Async(devicePtrDst, value, sizeBytes, nullStream), expectedReturn);
}

inline void testHipMemset2DApis(void* dst, size_t pitch, int value, size_t width, size_t height,
                                hipError_t expectedReturn = hipErrorInvalidValue) {
  HIP_CHECK_ERROR(hipMemset2D(dst, pitch, value, width, height), expectedReturn);
  HIP_CHECK_ERROR(hipMemset2DAsync(dst, pitch, value, width, height, nullStream), expectedReturn);
}


inline void testHipMemset3DApis(hipPitchedPtr& pitchedDevPtr, int value, const hipExtent& extent,
                                hipError_t expectedReturn = hipErrorInvalidValue) {
  HIP_CHECK_ERROR(hipMemset3D(pitchedDevPtr, value, extent), expectedReturn);
  HIP_CHECK_ERROR(hipMemset3DAsync(pitchedDevPtr, value, extent, nullStream), expectedReturn);
}

HIP_TEST_CASE(Unit_hipMemset_Negative_InvalidPtr) {
  void* dst;

  SECTION("Uninitialized Dst") {}
  SECTION("Nullptr as dst") { dst = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[width]);
    dst = hostPtr.get();
  }

  testHipMemsetApis(dst, memsetVal, width);
}


HIP_TEST_CASE(Unit_hipMemset_Negative_OutOfBoundsSize) {
#if !HT_AMD
  void* dst;
  constexpr size_t outOfBoundsSize{width + 1};
  HIP_CHECK(hipMalloc(&dst, width));

  testHipMemsetApis(dst, memsetVal, outOfBoundsSize);
  HIP_CHECK(hipFree(dst));
#endif
}

HIP_TEST_CASE(Unit_hipMemset_Negative_OutOfBoundsPtr) {
  void* dst;
  HIP_CHECK(hipMalloc(&dst, width));
  void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + width + 1};
  testHipMemsetApis(outOfBoundsPtr, memsetVal, width);
  HIP_CHECK(hipFree(dst));
}

HIP_TEST_CASE(Unit_hipMemset2D_Negative_InvalidPtr) {
  void* dst;
  SECTION("Uninitialized Dst") {}
  SECTION("Nullptr as Dst") { dst = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[height * width]);
    dst = hostPtr.get();
  }

  void* A_d;
  size_t pitch_A;
  HIP_CHECK(hipMallocPitch(&A_d, &pitch_A, width, height));
  testHipMemset2DApis(dst, pitch_A, memsetVal, width, height);
  HIP_CHECK(hipFree(A_d));
}

HIP_TEST_CASE(Unit_hipMemset2D_Negative_InvalidSizes) {
  void* dst;
  size_t realPitch;
  HIP_CHECK(hipMallocPitch(&dst, &realPitch, width, height));

  SECTION("Invalid Pitch") {
    size_t invalidPitch = 1;
    testHipMemset2DApis(dst, invalidPitch, memsetVal, width, height);
  }

  SECTION("Invalid Width") {
    size_t invalidWidth = realPitch + 1;
    testHipMemset2DApis(dst, realPitch, memsetVal, invalidWidth, height);
  }

#if !HT_AMD /* EXSWCPHIPT-52 */
  SECTION("Invalid height") {
    size_t invalidHeight = height + 1;
    testHipMemset2DApis(dst, realPitch, memsetVal, width, invalidHeight);
  }
#endif
  HIP_CHECK(hipFree(dst));
}

HIP_TEST_CASE(Unit_hipMemset2D_Negative_OutOfBoundsPtr) {
  void* dst;
  size_t realPitch;

  HIP_CHECK(hipMallocPitch(&dst, &realPitch, width, height));
  void* outOfBoundsPtr{reinterpret_cast<char*>(dst) + realPitch * height + 1};
  testHipMemset2DApis(outOfBoundsPtr, realPitch, memsetVal, width, height);
  HIP_CHECK(hipFree(dst));
}


HIP_TEST_CASE(Unit_hipMemset3D_Negative_InvalidPtr) {
  hipPitchedPtr pitchedDevPtr;

  SECTION("Uninitialized PitchedDevPtr") {}
  SECTION("Zero Initialized PitchedDevPtr") { pitchedDevPtr = {}; }

  testHipMemset3DApis(pitchedDevPtr, memsetVal, validExtent);
}

HIP_TEST_CASE(Unit_hipMemset3D_Negative_ModifiedPtr) {
  hipPitchedPtr pitchedDevPtr;

  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  void* allocatedMemory{pitchedDevPtr.ptr};

  SECTION("Nullptr Dst") { pitchedDevPtr.ptr = nullptr; }

  std::unique_ptr<char[]> hostPtr;
  SECTION("Host Pointer as Dst") {
    hostPtr.reset(new char[validExtent.width * validExtent.height * validExtent.depth]);
    pitchedDevPtr.ptr = hostPtr.get();
  }

  SECTION("Invalid Pitch") { pitchedDevPtr.pitch = 1; }

  CAPTURE(pitchedDevPtr.ptr, pitchedDevPtr.pitch, pitchedDevPtr.xsize, pitchedDevPtr.ysize);
  testHipMemset3DApis(pitchedDevPtr, memsetVal, validExtent);
  HIP_CHECK(hipFree(allocatedMemory));
}

HIP_TEST_CASE(Unit_hipMemset3D_Negative_InvalidSizes) {
  hipPitchedPtr pitchedDevPtr;
  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  hipExtent invalidExtent{validExtent};

  SECTION("Max Width") { invalidExtent.width = std::numeric_limits<std::size_t>::max(); }

  SECTION("Max Height") { invalidExtent.height = std::numeric_limits<std::size_t>::max(); }

#if !HT_NVIDIA /* This case hangs on Nvidia */
  SECTION("Max Depth") { invalidExtent.depth = std::numeric_limits<std::size_t>::max(); }
#endif

  SECTION("Invalid Width") { invalidExtent.width = pitchedDevPtr.pitch + 1; }

#if !HT_AMD /* EXSWCPHIPT-52 */
  SECTION("Invalid height") { invalidExtent.height += 1; }

  SECTION("Invalid depth") { invalidExtent.depth += 1; }
#endif

  CAPTURE(invalidExtent.width, invalidExtent.height, invalidExtent.depth);
  testHipMemset3DApis(pitchedDevPtr, memsetVal, invalidExtent);
  HIP_CHECK(hipFree(pitchedDevPtr.ptr));
}

HIP_TEST_CASE(Unit_hipMemset3D_Negative_OutOfBounds) {
  hipPitchedPtr pitchedDevPtr;

  HIP_CHECK(hipMalloc3D(&pitchedDevPtr, validExtent));
  hipPitchedPtr outOfBoundsPtr{pitchedDevPtr};
  outOfBoundsPtr.ptr = reinterpret_cast<char*>(pitchedDevPtr.ptr) +
                       pitchedDevPtr.pitch * validExtent.height * validExtent.depth + 1;

  SECTION("Extent Equal to 0") {
    hipExtent zeroExtent{0, 0, 0};
    testHipMemset3DApis(outOfBoundsPtr, memsetVal, zeroExtent, hipSuccess);
  }

  SECTION("Valid Extent") { testHipMemset3DApis(outOfBoundsPtr, memsetVal, validExtent); }

  HIP_CHECK(hipFree(pitchedDevPtr.ptr));
}
