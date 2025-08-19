/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <string.h>

#include <cstring>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <resource_guards.hh>

static bool testPassed1D = false;
static bool testPassed2D = false;
static constexpr auto NUM_ELM{1024};
/**
 * @addtogroup hipArrayGetDescriptor
 * @{
 * @ingroup MemoryTest
 * hipError_t hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray* array) -
 * Gets a 1D or 2D array descriptor
 */

// Create 1D array
hipArray_t arrayCreate1D(int format, int channel) {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  // Number of channels would be 1, 2, 4.
  switch (channel) {
    case 1:
      desc.NumChannels = channel;
      break;
    case 2:
      desc.NumChannels = channel;
      break;
    case 4:
      desc.NumChannels = channel;
      break;
  }
  desc.Width = 16;
  desc.Height = 0;
  // Number of Formats would be 8 as per enum format
  switch (format) {
    case 1:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
      break;
    case 2:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
      break;
    case 3:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
      break;
    case 8:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT8;
      break;
    case 9:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT16;
      break;
    case 10:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT32;
      break;
    case 16:
      desc.Format = HIP_AD_FORMAT_HALF;
      break;
    case 32:
      desc.Format = HIP_AD_FORMAT_FLOAT;
      break;
    default:
      desc.Format = HIP_AD_FORMAT_FLOAT;
  }
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}

// Create 2D array
hipArray_t arrayCreate2D(int format, int channel) {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  // Number of channels would be 1, 2, 4.
  switch (channel) {
    case 1:
      desc.NumChannels = channel;
      break;
    case 2:
      desc.NumChannels = channel;
      break;
    case 4:
      desc.NumChannels = channel;
      break;
  }
  desc.Width = 4;
  desc.Height = 4;
  // Number of Formats would be 8 as per enum format
  switch (format) {
    case 1:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
      break;
    case 2:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT16;
      break;
    case 3:
      desc.Format = HIP_AD_FORMAT_UNSIGNED_INT32;
      break;
    case 8:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT8;
      break;
    case 9:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT16;
      break;
    case 10:
      desc.Format = HIP_AD_FORMAT_SIGNED_INT32;
      break;
    case 16:
      desc.Format = HIP_AD_FORMAT_HALF;
      break;
    case 32:
      desc.Format = HIP_AD_FORMAT_FLOAT;
      break;
    default:
      desc.Format = HIP_AD_FORMAT_FLOAT;
  }
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}
// Create a simple 1D array for multithread scenario
static hipArray_t arrayCreate1D_Thread() {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.NumChannels = 2;
  desc.Width = 16;
  desc.Height = 0;
  desc.Format = HIP_AD_FORMAT_HALF;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}
// Create a simple 2D array for multithread scenario
static hipArray_t arrayCreate2D_Thread() {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.NumChannels = 1;
  desc.Width = 4;
  desc.Height = 4;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}
// Thread function for 1D Array
void thread_funct1D(hipArray_t array) {
  HIP_ARRAY_DESCRIPTOR desc;
  HIP_CHECK(hipArrayGetDescriptor(&desc, array));
  // Verify array parameters
  if ((desc.NumChannels == 2) && (desc.Width == 16) && (desc.Height == 0) &&
      (desc.Format == HIP_AD_FORMAT_HALF)) {
    testPassed1D = true;
  } else {
    testPassed1D = false;
  }
}
// Thread function for 2D Array
void thread_funct2D(hipArray_t array) {
  HIP_ARRAY_DESCRIPTOR desc;
  HIP_CHECK(hipArrayGetDescriptor(&desc, array));
  // Verify array parameters
  if ((desc.NumChannels == 1) && (desc.Width == 4) && (desc.Height == 4) &&
      (desc.Format == HIP_AD_FORMAT_FLOAT)) {
    testPassed2D = true;
  } else {
    testPassed2D = false;
  }
}
// 1D Array of type float
static hipArray_t arrayCreateSimple1D() {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.NumChannels = 1;
  desc.Width = 1024;
  desc.Height = 0;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}
// 2D Array of type Float
static hipArray_t arrayCreateSimple2D() {
  hipArray_t array;
  HIP_ARRAY_DESCRIPTOR desc;
  desc.NumChannels = 1;
  desc.Width = 1024;
  desc.Height = 1024;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  HIP_CHECK(hipArrayCreate(&array, &desc));
  return array;
}
// Function to verify data type and assign back memory from Array to Host
float* funcToChkArray(hipArray_t array) {
  HIP_ARRAY_DESCRIPTOR desc;
  HIP_CHECK(hipArrayGetDescriptor(&desc, array));
  float* A_h = nullptr;
  static constexpr auto NUM_ELM{1024};
  size_t mem_size = NUM_ELM * sizeof(float);
  if (desc.Format == HIP_AD_FORMAT_FLOAT) {
    A_h = reinterpret_cast<float*>(malloc(mem_size));
    for (int i = 0; i < NUM_ELM; i++) {
      A_h[i] = 2.0;
    }
  }
  HIP_CHECK(hipMemcpyAtoH(A_h, array, 0, mem_size));
  return A_h;
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify 1D, 2D Array parameters by using hipArrayGetDescriptor API.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipArrayGetDescriptor_1D_2D_ArrayParameterChk") {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    CHECK_IMAGE_SUPPORT
#if HT_NVIDIA
    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, i));
#endif
    // 1D array parameters verification
    SECTION("1D Array parameters verification") {
      hipArray_t array1D = arrayCreate1D(1, 1);
      HIP_ARRAY_DESCRIPTOR desc1;
      HIP_CHECK(hipArrayGetDescriptor(&desc1, array1D));
      // Verify width of Array
      REQUIRE(desc1.Width == 16);
      // Verify Height of Array
      REQUIRE(desc1.Height == 0);
      HIP_CHECK(hipArrayDestroy(array1D));
      for (int j = 1; j < 5; j++) {
        for (int i = 1; i < 33; i++) {
          hipArray_t array1D1 = arrayCreate1D(i, j);
          HIP_ARRAY_DESCRIPTOR desc1;
          HIP_CHECK(hipArrayGetDescriptor(&desc1, array1D1));

          // Verify Num Of Channels
          REQUIRE(desc1.NumChannels == j);
          // Verify format of Array
          REQUIRE(desc1.Format == i);
          if (i == 3) i = 7;
          if (i == 10) i = 15;
          if (i == 16) i = 31;
          HIP_CHECK(hipArrayDestroy(array1D1));
        }
        if (j == 2) j = 3;
      }
    }

    SECTION("2D Array parameters verification") {
      hipArray_t array2D = arrayCreate2D(1, 1);
      HIP_ARRAY_DESCRIPTOR desc;
      HIP_CHECK(hipArrayGetDescriptor(&desc, array2D));
      // Verify width of Array
      REQUIRE(desc.Width == 4);
      // Verify Height of Array
      REQUIRE(desc.Height == 4);
      HIP_CHECK(hipArrayDestroy(array2D));
      for (int j = 1; j < 5; j++) {
        for (int i = 1; i < 33; i++) {
          hipArray_t array2D1 = arrayCreate2D(i, j);
          HIP_ARRAY_DESCRIPTOR desc;
          HIP_CHECK(hipArrayGetDescriptor(&desc, array2D1));
          // Num Of Channels
          REQUIRE(desc.NumChannels == j);
          // Verify format of Array
          REQUIRE(desc.Format == i);
          if (i == 3) i = 7;
          if (i == 10) i = 15;
          if (i == 16) i = 31;
          HIP_CHECK(hipArrayDestroy(array2D1));
        }
        if (j == 2) j = 3;
      }
    }
#if HT_NVIDIA
    HIP_CHECK(hipCtxDestroy(ctx));
#endif
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify 1D, 2D Array parameters in a thread function while passing
     array address to the thread.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipArrayGetDescriptor_MultiThreadScenarioFor1D_2D_Array") {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipSetDevice(i));
    CHECK_IMAGE_SUPPORT
#if HT_NVIDIA
    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, i));
#endif
    hipArray_t array_t1 = arrayCreate1D_Thread();
    hipArray_t array_t2 = arrayCreate2D_Thread();
    std::vector<std::thread> ThreadVector1D;
    std::vector<std::thread> ThreadVector2D;
    for (int j = 0; j < 10; j++) {
      ThreadVector1D.emplace_back([&]() { thread_funct1D(array_t1); });
      ThreadVector2D.emplace_back([&]() { thread_funct2D(array_t2); });
    }
    for (auto& t : ThreadVector1D) {
      t.join();
    }
    for (auto& t : ThreadVector2D) {
      t.join();
    }
    // Validation
    REQUIRE(testPassed1D);
    REQUIRE(testPassed2D);
    HIP_CHECK(hipArrayDestroy(array_t1));
    HIP_CHECK(hipArrayDestroy(array_t2));
#if HT_NVIDIA
    HIP_CHECK(hipCtxDestroy(ctx));
#endif
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case to create 1D, 2D host memory and transfer that to Array and verify
     hipArrayGetDescriptor API functionality in the outside function while passing
         array as parameter to function.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Host2Array_Array2Host") {
  CHECK_IMAGE_SUPPORT

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int k = 0; k < numDevices; k++) {
    HIP_CHECK(hipSetDevice(k));
    CHECK_IMAGE_SUPPORT
#if HT_NVIDIA
    HIP_CHECK(hipInit(0));
    hipCtx_t ctx;
    HIP_CHECK(hipCtxCreate(&ctx, 0, k));
#endif
    int count_1D = 0;
    size_t mem_size = NUM_ELM * sizeof(float);
    float* A_h;
    A_h = reinterpret_cast<float*>(malloc(mem_size));
    for (int i = 0; i < NUM_ELM; i++) {
      A_h[i] = 2.0;
    }
    hipArray_t arraySimple1D = arrayCreateSimple1D();
    HIP_CHECK(hipMemcpyHtoA(arraySimple1D, 0, A_h, mem_size));
    float* A_h1 = funcToChkArray(arraySimple1D);
    for (int i = 0; i < NUM_ELM; i++) {
      if (A_h[i] == A_h1[i]) {
        count_1D += 1;
      }
    }
    // Validation
    REQUIRE(count_1D == NUM_ELM);
    free(A_h);
    HIP_CHECK(hipArrayDestroy(arraySimple1D));
    SECTION("2D Array Verification") {
      int count_2D = 0;
      size_t mem_size1 = NUM_ELM * sizeof(float);
      float* A_h2;
      A_h2 = reinterpret_cast<float*>(malloc(mem_size1));
      for (int i = 0; i < NUM_ELM; i++) {
        A_h2[i] = 2.0;
      }

      hipArray_t arraySimple2D = arrayCreateSimple2D();
      HIP_CHECK(hipMemcpyHtoA(arraySimple2D, 0, A_h2, mem_size1));
      float* A_h3 = funcToChkArray(arraySimple2D);
      for (int i = 0; i < NUM_ELM; i++) {
        if (A_h2[i] == A_h3[i]) {
          count_2D += 1;
        }
      }
      // Validation
      REQUIRE(count_2D == NUM_ELM);
      free(A_h2);
      HIP_CHECK(hipArrayDestroy(arraySimple2D));
    }
#if HT_NVIDIA
    HIP_CHECK(hipCtxDestroy(ctx));
#endif
  }
}
/**
 * Test Description
 * ------------------------
 * - Test case to verify negative scenarios of hipArrayGetDescriptor API.
 * Test source
 * ------------------------
 * - catch/unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Negative_Scenarios") {
  hipError_t error;
  HIP_ARRAY_DESCRIPTOR desc_Neg;
  SECTION("Array Address As Nullptr") {
    error = hipArrayGetDescriptor(&desc_Neg, nullptr);
    REQUIRE(error != hipSuccess);
  }
// Enable it for AMD only as it is failing on NVIDIA
#if HT_AMD
  void* dptr;
  SECTION("Invalid Array Address") {
    HIP_CHECK(hipMalloc(&dptr, 1024));
    error = hipArrayGetDescriptor(&desc_Neg, reinterpret_cast<hipArray_t>(dptr));
    REQUIRE(error != hipSuccess);
    HIP_CHECK(hipFree(dptr));
  }
#endif
}

/**
 * @addtogroup hipArrayGetDescriptor hipArrayGetDescriptor
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetDescriptor(HIP_ARRAY_DESCRIPTOR* pArrayDescriptor, hipArray* array)` -
 * Gets a 1D or 2D array descriptor.
 */

/**
 * Test Description
 * ------------------------
 *  - Basic sanity test for `hipArrayGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

  HIP_ARRAY_DESCRIPTOR expected_desc{};
  using vec_info = vector_info<float>;
  expected_desc.Format = vec_info::format;
  expected_desc.NumChannels = vec_info::size;
  expected_desc.Width = 1024 / sizeof(float);
  expected_desc.Height = 4;

  hipArray_t ptr;
  HIP_CHECK(hipArrayCreate(&ptr, &expected_desc));

  HIP_ARRAY_DESCRIPTOR desc;
  HIP_CHECK(hipArrayGetDescriptor(&desc, ptr));

  REQUIRE(desc.Format == expected_desc.Format);
  REQUIRE(desc.NumChannels == expected_desc.NumChannels);
  REQUIRE(desc.Width == expected_desc.Width);
  REQUIRE(desc.Height == expected_desc.Height);

  HIP_CHECK(hipArrayDestroy(ptr));
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetDescriptor`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetDescriptor.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipArrayGetDescriptor_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  HIP_ARRAY_DESCRIPTOR expected_desc{};
  using vec_info = vector_info<float>;
  expected_desc.Format = vec_info::format;
  expected_desc.NumChannels = vec_info::size;
  expected_desc.Width = 1024 / sizeof(float);
  expected_desc.Height = 4;

  hipArray_t ptr;
  HIP_CHECK(hipArrayCreate(&ptr, &expected_desc));

  HIP_ARRAY_DESCRIPTOR desc;

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetDescriptor(nullptr, ptr), hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetDescriptor(&desc, nullptr), hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    HIP_CHECK(hipArrayDestroy(ptr));
    HIP_CHECK_ERROR(hipArrayGetDescriptor(&desc, ptr), hipErrorInvalidHandle);
  }

  static_cast<void>(hipArrayDestroy(ptr));
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
