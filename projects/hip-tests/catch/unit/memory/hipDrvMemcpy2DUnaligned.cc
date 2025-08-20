/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

#include "memcpy2d_tests_common.hh"

/**
 * @addtogroup hipDrvMemcpy2DUnaligned hipDrvMemcpy2DUnaligned
 * @{
 * @ingroup MemoryTest
 * `hipError_t hipDrvMemcpy2DUnaligned(const hip_Memcpy2D * pCopy)`	-
 * Copies memory for 2D arrays.
 */

/**
 * Test Description
 * ------------------------
 *  - Negative test cases for hipDrvMemcpy2DUnaligned api
 * Test source
 * ------------------------
 *  - unit/memory/hipDrvMemcpy2DUnaligned.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvMemcpy2DUnaligned_NegTst") {
  // declare host and device arrays
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(3, 4, 100);
  int *srcD, *srcH;
  int *dstD, *dstH;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&srcD), sizeof(int) * rows * cols));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dstD), sizeof(int) * rows * cols));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&srcH), sizeof(int) * rows * cols));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&dstH), sizeof(int) * rows * cols));

  // initialise array with corresponding index values
  for (int i = 0; i < rows * cols; i++) {
    srcH[i] = i;
  }
  HIP_CHECK(hipMemcpyHtoD(srcD, srcH, sizeof(int) * rows * cols));

  hip_Memcpy2D pCopy;

  SECTION(
      "srcY(second argument) + non-zero WidthInBytes(15th argument)\
           * Height(16th argument) points to unallocated memory") {
    pCopy.srcXInBytes = 0;
    pCopy.srcY = rows;
    pCopy.srcMemoryType = hipMemoryTypeDevice;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = 0;
    pCopy.dstMemoryType = hipMemoryTypeDevice;
    pCopy.srcDevice = srcD;
    pCopy.dstDevice = dstD;
    pCopy.WidthInBytes = sizeof(int);
    pCopy.Height = 1;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);
    HIP_CHECK_ERROR(hipDrvMemcpy2DUnaligned(&pCopy), hipErrorInvalidValue);
  }
  SECTION(
      "srcHost(4th argument), srcDevice(5th argument), srcArray(6th\
          argument) passed nullptr") {
    pCopy.srcXInBytes = 0;
    pCopy.srcY = 0;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = 0;
    pCopy.dstMemoryType = hipMemoryTypeDevice;
    pCopy.dstDevice = dstD;
    pCopy.WidthInBytes = sizeof(int) * cols;
    pCopy.Height = rows;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);
    SECTION("srcHost passed nullptr") {
      pCopy.srcMemoryType = hipMemoryTypeHost;
      pCopy.srcHost = nullptr;
    }
    SECTION("srcDevice passed nullptr") {
      pCopy.srcMemoryType = hipMemoryTypeDevice;
      pCopy.srcDevice = nullptr;
    }
    SECTION("srcArray passed nullptr") {
      pCopy.srcMemoryType = hipMemoryTypeArray;
      pCopy.srcArray = nullptr;
    }
    HIP_CHECK_ERROR(hipDrvMemcpy2DUnaligned(&pCopy), hipErrorInvalidValue);
  }
  SECTION(
      "dstY(second argument) + non-zero WidthInBytes(15th argument)\
           * Height(16th argument) points to unallocated memory") {
    pCopy.srcXInBytes = 0;
    pCopy.srcY = 0;
    pCopy.srcMemoryType = hipMemoryTypeDevice;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = rows;
    pCopy.dstMemoryType = hipMemoryTypeDevice;
    pCopy.srcDevice = srcD;
    pCopy.dstDevice = dstD;
    pCopy.WidthInBytes = sizeof(int);
    pCopy.Height = 1;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);
    HIP_CHECK_ERROR(hipDrvMemcpy2DUnaligned(&pCopy), hipErrorInvalidValue);
  }
  SECTION(
      "dstHost(4th argument), dstDevice(5th argument), dstArray(6th\
          argument) passed nullptr") {
    pCopy.srcXInBytes = 0;
    pCopy.srcY = 0;
    pCopy.srcMemoryType = hipMemoryTypeDevice;
    pCopy.srcDevice = srcD;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = 0;
    pCopy.WidthInBytes = sizeof(int) * cols;
    pCopy.Height = rows;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);
    SECTION("dstHost passed nullptr") {
      pCopy.dstMemoryType = hipMemoryTypeHost;
      pCopy.dstHost = nullptr;
    }
    SECTION("dstDevice passed nullptr") {
      pCopy.dstMemoryType = hipMemoryTypeDevice;
      pCopy.dstDevice = nullptr;
    }
    SECTION("dstArray passed nullptr") {
      pCopy.dstMemoryType = hipMemoryTypeArray;
      pCopy.dstArray = nullptr;
    }
    HIP_CHECK_ERROR(hipDrvMemcpy2DUnaligned(&pCopy), hipErrorInvalidValue);
  }
  SECTION(
      "WidthInBytes * Height greater than allocated memory(both src \
          and dst)") {
    pCopy.srcXInBytes = 0;
    pCopy.srcY = 0;
    pCopy.srcMemoryType = hipMemoryTypeDevice;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = 0;
    pCopy.dstMemoryType = hipMemoryTypeDevice;
    pCopy.srcDevice = srcD;
    pCopy.dstDevice = dstD;
    pCopy.WidthInBytes = sizeof(int) * cols;
    pCopy.Height = rows + 1;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);
    HIP_CHECK_ERROR(hipDrvMemcpy2DUnaligned(&pCopy), hipErrorInvalidValue);
  }
  HIP_CHECK(hipFree(srcH));
  HIP_CHECK(hipFree(srcD));
  HIP_CHECK(hipFree(dstD));
  HIP_CHECK(hipFree(dstH));
}

/**
 * Test Description
 * ------------------------
 *  - Functional test cases for hipDrvMemcpy2DUnaligned api
 * Test source
 * ------------------------
 *  - unit/memory/hipDrvMemcpy2DUnaligned.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvMemcpy2DUnaligned_FuncTst") {
  SECTION(
      "Different types of memory transfers functional tests to check if\
          copied array contains correct values") {
    // declare host and device arrays
    int rows, cols;
    rows = GENERATE(3, 4, 100);
    cols = GENERATE(3, 4, 100);
    int *srcD, *srcH;
    int *dstD, *dstH;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&srcD), sizeof(int) * rows * cols));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dstD), sizeof(int) * rows * cols));
    srcH = reinterpret_cast<int*>(malloc(sizeof(int) * rows * cols));
    dstH = reinterpret_cast<int*>(malloc(sizeof(int) * rows * cols));

    // initialise array with corresponding index values
    for (int i = 0; i < rows * cols; i++) {
      srcH[i] = i;
    }

    hip_Memcpy2D pCopy;
    pCopy.srcXInBytes = 0;
    pCopy.srcY = 0;
    pCopy.dstXInBytes = 0;
    pCopy.dstY = 0;
    pCopy.WidthInBytes = sizeof(int) * cols;
    pCopy.Height = rows;
    pCopy.srcPitch = cols * sizeof(int);
    pCopy.dstPitch = cols * sizeof(int);

    SECTION("Device to Device") {
      HIP_CHECK(hipMemcpyHtoD(srcD, srcH, sizeof(int) * rows * cols));
      pCopy.srcMemoryType = hipMemoryTypeDevice;
      pCopy.dstMemoryType = hipMemoryTypeDevice;
      pCopy.srcDevice = srcD;
      pCopy.dstDevice = dstD;
      HIP_CHECK(hipDrvMemcpy2DUnaligned(&pCopy));
      HIP_CHECK(hipMemcpyDtoH(dstH, dstD, sizeof(int) * rows * cols));
    }
    SECTION("Device to Host") {
      HIP_CHECK(hipMemcpyHtoD(srcD, srcH, sizeof(int) * rows * cols));
      pCopy.srcMemoryType = hipMemoryTypeDevice;
      pCopy.dstMemoryType = hipMemoryTypeHost;
      pCopy.srcDevice = srcD;
      pCopy.dstHost = dstH;
      HIP_CHECK(hipDrvMemcpy2DUnaligned(&pCopy));
    }
    SECTION("Host to Device") {
      pCopy.srcMemoryType = hipMemoryTypeHost;
      pCopy.dstMemoryType = hipMemoryTypeDevice;
      pCopy.srcHost = srcH;
      pCopy.dstDevice = dstD;
      HIP_CHECK(hipDrvMemcpy2DUnaligned(&pCopy));
      HIP_CHECK(hipMemcpyDtoH(dstH, dstD, sizeof(int) * rows * cols));
    }
    SECTION("Host to Host") {
      pCopy.srcMemoryType = hipMemoryTypeHost;
      pCopy.dstMemoryType = hipMemoryTypeHost;
      pCopy.srcHost = srcH;
      pCopy.dstHost = dstH;
      HIP_CHECK(hipDrvMemcpy2DUnaligned(&pCopy));
    }

    for (int i = 0; i < rows * cols; i++) {
      REQUIRE(dstH[i] == i);
    }

    HIP_CHECK(hipFree(srcD));
    HIP_CHECK(hipFree(dstD));
    free(srcH);
    free(dstH);
  }
}


/**
 * Test Description
 * ------------------------
 *  - Basic test that copies and verifies copied data
 * Test source
 * ------------------------
 *  - unit/memory/hipDrvMemcpy2DUnaligned.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvMemcpy2DUnaligned_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

  SECTION("Device to Device") {
    SECTION("Peer access enabled") {
      Memcpy2DDeviceToDeviceShell<false, true, true>(DrvMemcpy2DUnalignedAdapter());
    }
  }

  SECTION("Host to Device") {
    Memcpy2DHostToDeviceShell<false, true>(DrvMemcpy2DUnalignedAdapter());
  }

  SECTION("Device to Host") {
    Memcpy2DDeviceToHostShell<false, true>(DrvMemcpy2DUnalignedAdapter());
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test that verifies synchronization behaviour
 * Test source
 * ------------------------
 *  - unit/memory/hipDrvMemcpy2DUnaligned.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvMemcpy2DUnaligned_Positive_Synchronization_Behavior") {
  CHECK_IMAGE_SUPPORT

  HIP_CHECK(hipDeviceSynchronize());

  SECTION("Host to Device") { Memcpy2DHtoDSyncBehavior<true>(DrvMemcpy2DUnalignedAdapter(), true); }

  SECTION("Device to Pinned Host") {
    Memcpy2DDtoHPinnedSyncBehavior<true>(DrvMemcpy2DUnalignedAdapter(), true);
  }

  SECTION("Device to Pageable Host") {
    Memcpy2DDtoHPageableSyncBehavior<true>(DrvMemcpy2DUnalignedAdapter(), true);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Basic test that copies and verifies copied data with zero width or height.
 * Test source
 * ------------------------
 *  - unit/memory/hipDrvMemcpy2DUnaligned.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvMemcpy2DUnaligned_Positive_Parameters") {
  Memcpy2DZeroWidthHeight<false, true>(DrvMemcpy2DUnalignedAdapter());
}
