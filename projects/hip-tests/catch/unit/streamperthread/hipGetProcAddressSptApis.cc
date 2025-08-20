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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <utils.hh>
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (memory copy) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_MemCpy") {
  void* hipMemcpy_spt_ptr = nullptr;
  void* hipMemcpyAsync_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemcpy_spt", &hipMemcpy_spt_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyAsync_spt", &hipMemcpyAsync_spt_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipMemcpy_spt_ptr)(void*, const void*, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, const void*, size_t, hipMemcpyKind)>(
          hipMemcpy_spt_ptr);

  hipError_t (*dyn_hipMemcpyAsync_spt_ptr)(void*, const void*, size_t, hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpyAsync_spt_ptr);

  // Validating hipMemcpy_spt API
  {
    int N = 64;
    int Nbytes = N * sizeof(int);
    int value = 2;
    // With flag hipMemcpyHostToHost
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(dstHostMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(dstHostMem, hostMem, Nbytes, hipMemcpyHostToHost));
      REQUIRE(validateHostArray(dstHostMem, N, value) == true);

      free(hostMem);
      free(dstHostMem);
    }
    // With flag hipMemcpyHostToDevice
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(devMem, hostMem, Nbytes, hipMemcpyHostToDevice));
      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDeviceToHost
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDeviceToDevice
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDeviceToDevice));
      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDeviceToDeviceNoCU));
      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
    // With flag hipMemcpyDefault - Host To Host
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(dstHostMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(dstHostMem, hostMem, Nbytes, hipMemcpyDefault));
      REQUIRE(validateHostArray(dstHostMem, N, value) == true);

      free(hostMem);
      free(dstHostMem);
    }
    // With flag hipMemcpyDefault - Host To Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(devMem, hostMem, Nbytes, hipMemcpyDefault));
      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDefault - Device To Host
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(hostMem, devMem, Nbytes, hipMemcpyDefault));
      REQUIRE(validateHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_spt_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDefault));
      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
  }

  // Validating hipMemcpyAsync_spt API
  {
    int N = 4096;
    const int Ns = 4;
    int Nbytes = N * sizeof(int);
    int value = 2;

    // With flag hipMemcpyHostToHost
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(dstHostMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(dstHostMem + startIndex, hostMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyHostToHost, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateHostArray(dstHostMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }

      free(hostMem);
      free(dstHostMem);
    }
    // With flag hipMemcpyHostToDevice
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(devMem + startIndex, hostMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyHostToDevice, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }
      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDeviceToHost
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(hostMem + startIndex, devMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDeviceToHost, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateHostArray(hostMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }
      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDeviceToDevice
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(dstDevMem + startIndex, devMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDeviceToDevice, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(dstDevMem + startIndex, devMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDeviceToDeviceNoCU,
                                             stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
    // With flag hipMemcpyDefault - Host To Host
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(dstHostMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(dstHostMem + startIndex, hostMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDefault, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateHostArray(dstHostMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }

      free(hostMem);
      free(dstHostMem);
    }
    // With flag hipMemcpyDefault - Host To Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(devMem + startIndex, hostMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDefault, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }
      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDefault - Device To Host
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(hostMem + startIndex, devMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDefault, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateHostArray(hostMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }
      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, Nbytes));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      hipStream_t stream[Ns];
      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamCreate(&stream[s]));
      }

      int* dstDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
      REQUIRE(dstDevMem != nullptr);

      for (int s = 0; s < Ns; s++) {
        int startIndex = s * (N / Ns);
        HIP_CHECK(dyn_hipMemcpyAsync_spt_ptr(dstDevMem + startIndex, devMem + startIndex,
                                             (Nbytes / Ns), hipMemcpyDefault, stream[s]));
      }

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamSynchronize(stream[s]));
      }

      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      for (int s = 0; s < Ns; s++) {
        HIP_CHECK(hipStreamDestroy(stream[s]));
      }

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (memset) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Memset") {
  void* hipMemset_spt_ptr = nullptr;
  void* hipMemsetAsync_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemset_spt", &hipMemset_spt_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemsetAsync_spt", &hipMemsetAsync_spt_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipMemset_spt_ptr)(void*, int, size_t) =
      reinterpret_cast<hipError_t (*)(void*, int, size_t)>(hipMemset_spt_ptr);
  hipError_t (*dyn_hipMemsetAsync_spt_ptr)(void*, int, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, int, size_t, hipStream_t)>(hipMemsetAsync_spt_ptr);

  // Validating hipMemset_spt API
  {
    int N = 16;
    int Nbytes = N * sizeof(char);
    int value = 10;

    void* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemset_spt_ptr(devMem, value, Nbytes));

    char* hostMem = reinterpret_cast<char*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemsetAsync_spt API
  {
    int N = 16;
    int Nbytes = N * sizeof(char);
    int value = 126;
    const int Ns = 4;

    char* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      HIP_CHECK(dyn_hipMemsetAsync_spt_ptr(devMem + startIndex, value, N / Ns, stream[s]));
    }
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamSynchronize(stream[s]));
    }

    char* hostMem = reinterpret_cast<char*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (memset 2D and 3D) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Memset2D3D") {
  CHECK_IMAGE_SUPPORT

  void* hipMemset2D_spt_ptr = nullptr;
  void* hipMemset2DAsync_spt_ptr = nullptr;
  void* hipMemset3D_spt_ptr = nullptr;
  void* hipMemset3DAsync_spt_ptr = nullptr;
  hipDriverProcAddressQueryResult symbolStatus = HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemset2D_spt", &hipMemset2D_spt_ptr, currentHipVersion, 0,
                              &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(hipGetProcAddress("hipMemset2DAsync_spt", &hipMemset2DAsync_spt_ptr, currentHipVersion,
                              0, &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(hipGetProcAddress("hipMemset3D_spt", &hipMemset3D_spt_ptr, currentHipVersion, 0,
                              &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(hipGetProcAddress("hipMemset3DAsync_spt", &hipMemset3DAsync_spt_ptr, currentHipVersion,
                              0, &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*dyn_hipMemset2D_spt_ptr)(void*, size_t, int, size_t, size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, int, size_t, size_t)>(hipMemset2D_spt_ptr);
  hipError_t (*dyn_hipMemset2DAsync_spt_ptr)(void*, size_t, int, size_t, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, int, size_t, size_t, hipStream_t)>(
          hipMemset2DAsync_spt_ptr);
  hipError_t (*dyn_hipMemset3D_spt_ptr)(hipPitchedPtr, int, hipExtent) =
      reinterpret_cast<hipError_t (*)(hipPitchedPtr, int, hipExtent)>(hipMemset3D_spt_ptr);
  hipError_t (*dyn_hipMemset3DAsync_spt_ptr)(hipPitchedPtr, int, hipExtent, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipPitchedPtr, int, hipExtent, hipStream_t)>(
          hipMemset3DAsync_spt_ptr);
  size_t width = 1024;
  size_t height = 1024;
  size_t depth = 1024;
  const int Ns = 4;
  int value = 10;
  size_t pitch;

  // Validating hipMemset2D_spt API
  {
    const int N = width * height;

    char* devMem = nullptr;
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
    REQUIRE(devMem != nullptr);

    HIP_CHECK(dyn_hipMemset2D_spt_ptr(devMem, pitch, value, width, height));

    char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy2D(hostMem, width, devMem, pitch, width, height, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemset2DAsync_spt API
  {
    const int N = width * height;
    char* devMem = nullptr;
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
    REQUIRE(devMem != nullptr);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    // set the whole matrix first to something different than 'value'
    HIP_CHECK(dyn_hipMemset2DAsync_spt_ptr(devMem, pitch, 5, width, height, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      int row = startIndex / width;
      HIP_CHECK(dyn_hipMemset2DAsync_spt_ptr(devMem + row * pitch, pitch, value, width, height / Ns,
                                             stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamSynchronize(stream[s]));
    }

    char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy2D(hostMem, width, devMem, pitch, width, height, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemset3D_spt API
  {
    const int N = width * height * depth;

    hipPitchedPtr devMem;
    hipExtent extent3d{width, height, depth};
    HIP_CHECK(hipMalloc3D(&devMem, extent3d));
    REQUIRE(devMem.ptr != nullptr);

    HIP_CHECK(dyn_hipMemset3D_spt_ptr(devMem, value, extent3d));

    char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
    REQUIRE(hostMem != nullptr);

    hipMemcpy3DParms myparms{};
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
    myparms.srcPtr = devMem;
    myparms.extent = extent3d;
    myparms.kind = hipMemcpyDeviceToHost;
    HIP_CHECK(hipMemcpy3D(&myparms));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem.ptr));
    free(hostMem);
  }

  // Validating hipMemset3DAsync_spt API
  {
    const int N = width * height * depth;

    hipPitchedPtr devMem;
    hipExtent extent3d{width, height, depth};
    HIP_CHECK(hipMalloc3D(&devMem, extent3d));
    REQUIRE(devMem.ptr != nullptr);

    HIP_CHECK(dyn_hipMemset3DAsync_spt_ptr(devMem, value, extent3d, NULL));

    char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
    REQUIRE(hostMem != nullptr);

    hipMemcpy3DParms myparms{};
    myparms.srcPos = make_hipPos(0, 0, 0);
    myparms.dstPos = make_hipPos(0, 0, 0);
    myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
    myparms.srcPtr = devMem;
    myparms.extent = extent3d;
    myparms.kind = hipMemcpyDeviceToHost;
    HIP_CHECK(hipMemcpy3D(&myparms));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem.ptr));
    free(hostMem);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (memory copy 2D) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Memcpy2D") {
  CHECK_IMAGE_SUPPORT

  void* hipMemcpy2D_spt_ptr = nullptr;
  void* hipMemcpy2DAsync_spt_ptr = nullptr;
  void* hipMemcpy2DToArray_spt_ptr = nullptr;
  void* hipMemcpy2DToArrayAsync_spt_ptr = nullptr;
  void* hipMemcpy2DFromArray_spt_ptr = nullptr;
  void* hipMemcpy2DFromArrayAsync_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMemcpy2D_spt", &hipMemcpy2D_spt_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DAsync_spt", &hipMemcpy2DAsync_spt_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DToArray_spt", &hipMemcpy2DToArray_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DToArrayAsync_spt", &hipMemcpy2DToArrayAsync_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DFromArray_spt", &hipMemcpy2DFromArray_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DFromArrayAsync_spt", &hipMemcpy2DFromArrayAsync_spt_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemcpy2D_spt_ptr)(void*, size_t, const void*, size_t, size_t, size_t,
                                        hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, size_t, const void*, size_t, size_t, size_t,
                                      hipMemcpyKind)>(hipMemcpy2D_spt_ptr);
  hipError_t (*dyn_hipMemcpy2DAsync_spt_ptr)(void*, size_t, const void*, size_t, size_t, size_t,
                                             hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, const void*, size_t, size_t, size_t,
                                      hipMemcpyKind, hipStream_t)>(hipMemcpy2DAsync_spt_ptr);
  hipError_t (*dyn_hipMemcpy2DToArray_spt_ptr)(hipArray_t, size_t, size_t, const void* src, size_t,
                                               size_t, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, size_t, const void* src, size_t, size_t,
                                      size_t, hipMemcpyKind)>(hipMemcpy2DToArray_spt_ptr);
  hipError_t (*dyn_hipMemcpy2DToArrayAsync_spt_ptr)(hipArray_t, size_t, size_t, const void* src,
                                                    size_t, size_t, size_t, hipMemcpyKind,
                                                    hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, size_t, const void* src, size_t, size_t,
                                      size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpy2DToArrayAsync_spt_ptr);
  hipError_t (*dyn_hipMemcpy2DFromArray_spt_ptr)(void*, size_t, hipArray_const_t, size_t, size_t,
                                                 size_t, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, size_t, hipArray_const_t, size_t, size_t, size_t,
                                      size_t, hipMemcpyKind)>(hipMemcpy2DFromArray_spt_ptr);
  hipError_t (*dyn_hipMemcpy2DFromArrayAsync_spt_ptr)(
      void*, size_t, hipArray_const_t, size_t, size_t, size_t, size_t, hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, hipArray_const_t, size_t, size_t, size_t,
                                      size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpy2DFromArrayAsync_spt_ptr);

  // Validating hipMemcpy2D_spt API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;
    size_t pitch;

    // With flag hipMemcpyHostToHost
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(dHostMem, width, sHostMem, width, width, height,
                                        hipMemcpyHostToHost));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyHostToDevice
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(devMem, pitch, hostMem, pitch, width, height,
                                        hipMemcpyHostToDevice));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // With flag hipMemcpyDeviceToHost
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(hostMem, width, devMem, pitch, width, height,
                                        hipMemcpyDeviceToHost));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToDevice
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                        hipMemcpyDeviceToDevice));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                        hipMemcpyDeviceToDeviceNoCU));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // With flag hipMemcpyDefault - Host To Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(dHostMem, width, sHostMem, width, width, height,
                                        hipMemcpyDefault));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyDefault - Host To Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(
          dyn_hipMemcpy2D_spt_ptr(devMem, pitch, hostMem, pitch, width, height, hipMemcpyDefault));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // With flag hipMemcpyDefault - Device To Host
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      HIP_CHECK(
          dyn_hipMemcpy2D_spt_ptr(hostMem, width, devMem, pitch, width, height, hipMemcpyDefault));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Device
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2D_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                        hipMemcpyDefault));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipMemcpy2DAsync_spt API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;
    size_t pitch;

    // With flag hipMemcpyHostToHost
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(dHostMem, width, sHostMem, width, width, height,
                                             hipMemcpyHostToHost, NULL));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyHostToDevice
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(devMem, pitch, hostMem, pitch, width, height,
                                             hipMemcpyHostToDevice, NULL));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // With flag hipMemcpyDeviceToHost
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(hostMem, width, devMem, pitch, width, height,
                                             hipMemcpyDeviceToHost, NULL));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToDevice
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                             hipMemcpyDeviceToDevice, NULL));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                             hipMemcpyDeviceToDeviceNoCU, NULL));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // With flag hipMemcpyDefault - Host To Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(dHostMem, width, sHostMem, width, width, height,
                                             hipMemcpyDefault, NULL));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyDefault - Host To Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(devMem, pitch, hostMem, pitch, width, height,
                                             hipMemcpyDefault, NULL));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // With flag hipMemcpyDefault - Device To Host
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(hostMem, width, devMem, pitch, width, height,
                                             hipMemcpyDefault, NULL));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Device
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DAsync_spt_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                             hipMemcpyDefault, NULL));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipMemcpy2DToArray_spt API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;

    // With flag hipMemcpyHostToDevice
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArray_spt_ptr(array, 0, 0, hostMem, width, width, height,
                                               hipMemcpyHostToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDevice
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArray_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                               hipMemcpyDeviceToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArray_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                               hipMemcpyDeviceToDeviceNoCU));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Host To Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArray_spt_ptr(array, 0, 0, hostMem, width, width, height,
                                               hipMemcpyDefault));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArray_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                               hipMemcpyDefault));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpy2DToArrayAsync_spt API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;

    // With flag hipMemcpyHostToDevice
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_spt_ptr(array, 0, 0, hostMem, width, width, height,
                                                    hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDevice
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                                    hipMemcpyDeviceToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                                    hipMemcpyDeviceToDeviceNoCU, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Host To Device
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_spt_ptr(array, 0, 0, hostMem, width, width, height,
                                                    hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_spt_ptr(array, 0, 0, sDevMem, sPitch, width, height,
                                                    hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(hipMemcpy2DFromArray(hostMemory, width, array, 0, 0, width, height,
                                     hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpy2DFromArray API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;

    // With flag hipMemcpyDeviceToHost
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, hostMem, width, width, height, hipMemcpyHostToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArray_spt_ptr(hostMemory, width, array, 0, 0, width, height,
                                                 hipMemcpyDeviceToHost));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDevice
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArray_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                 hipMemcpyDeviceToDevice));
      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArray_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                 hipMemcpyDeviceToDeviceNoCU));
      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Host
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, hostMem, width, width, height, hipMemcpyHostToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArray_spt_ptr(hostMemory, width, array, 0, 0, width, height,
                                                 hipMemcpyDefault));
      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArray_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                 hipMemcpyDefault));
      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpy2DFromArrayAsync API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;

    // With flag hipMemcpyDeviceToHost
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, hostMem, width, width, height, hipMemcpyHostToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_spt_ptr(hostMemory, width, array, 0, 0, width, height,
                                                      hipMemcpyDeviceToHost, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDevice
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                      hipMemcpyDeviceToDevice, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                      hipMemcpyDeviceToDeviceNoCU, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Host
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, hostMem, width, width, height, hipMemcpyHostToDevice));

      char* hostMemory = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_spt_ptr(hostMemory, width, array, 0, 0, width, height,
                                                      hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<char>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);
      HIP_CHECK(
          hipMemcpy2DToArray(array, 0, 0, sDevMem, sPitch, width, height, hipMemcpyDeviceToDevice));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_spt_ptr(dDevMem, width, array, 0, 0, width, height,
                                                      hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (memory copy 3D) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Memcpy3D") {
  CHECK_IMAGE_SUPPORT

  void* hipMemcpy3D_spt_ptr = nullptr;
  void* hipMemcpy3DAsync_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMemcpy3D_spt", &hipMemcpy3D_spt_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy3DAsync_spt", &hipMemcpy3DAsync_spt_ptr, currentHipVersion,
                              0, nullptr));

  hipError_t (*dyn_hipMemcpy3D_spt_ptr)(const struct hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(const struct hipMemcpy3DParms*)>(hipMemcpy3D_spt_ptr);
  hipError_t (*dyn_hipMemcpy3DAsync_spt_ptr)(const struct hipMemcpy3DParms*, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const struct hipMemcpy3DParms*, hipStream_t)>(
          hipMemcpy3DAsync_spt_ptr);

  // Validating hipMemcpy3D_spt API
  {
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;
    const int N = width * height * depth;
    int value = 10;
    hipExtent extent3d{width, height, depth};

    // With flag hipMemcpyHostToHost
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(sHostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(dHostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyHostToHost;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyHostToDevice
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = devMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyHostToDevice;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(devMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToHost
    {
      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(devMem, value, extent3d));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = devMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToHost;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToDevice
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToDevice;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }

    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToDeviceNoCU;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }

    // With flag hipMemcpyDefault - Host To Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(sHostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(dHostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyDefault - Host To Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = devMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(devMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Host
    {
      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(devMem, value, extent3d));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = devMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Device
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;
      HIP_CHECK(dyn_hipMemcpy3D_spt_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }
  }

  // Validating hipMemcpy3DAsync_spt API
  {
    size_t width = 256;
    size_t height = 256;
    size_t depth = 256;
    const int N = width * height * depth;
    int value = 10;
    hipExtent extent3d{width, height, depth};

    // With flag hipMemcpyHostToHost
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(sHostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(dHostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyHostToHost;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyHostToDevice
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = devMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyHostToDevice;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(devMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToHost
    {
      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(devMem, value, extent3d));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = devMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToHost;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDeviceToDevice
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToDevice;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }

    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDeviceToDeviceNoCU;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }

    // With flag hipMemcpyDefault - Host To Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(sHostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(dHostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(sHostMem);
      free(dHostMem);
    }

    // With flag hipMemcpyDefault - Host To Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = devMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(devMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Host
    {
      hipPitchedPtr devMem;
      HIP_CHECK(hipMalloc3D(&devMem, extent3d));
      REQUIRE(devMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(devMem, value, extent3d));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = devMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = make_hipPitchedPtr(hostMem, width, height, depth);
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(devMem.ptr));
      free(hostMem);
    }

    // With flag hipMemcpyDefault - Device To Device
    {
      hipPitchedPtr sDevMem;
      HIP_CHECK(hipMalloc3D(&sDevMem, extent3d));
      REQUIRE(sDevMem.ptr != nullptr);
      HIP_CHECK(hipMemset3D(sDevMem, value, extent3d));

      hipPitchedPtr dDevMem;
      HIP_CHECK(hipMalloc3D(&dDevMem, extent3d));
      REQUIRE(dDevMem.ptr != nullptr);

      hipMemcpy3DParms myparms{};
      myparms.srcPtr = sDevMem;
      myparms.srcPos = make_hipPos(0, 0, 0);
      myparms.dstPtr = dDevMem;
      myparms.dstPos = make_hipPos(0, 0, 0);
      myparms.extent = extent3d;
      myparms.kind = hipMemcpyDefault;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipMemcpy3DAsync_spt_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (kernel launch) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_LaunchKernel") {
  void* hipLaunchKernel_spt_ptr = nullptr;
  void* hipLaunchHostFunc_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipLaunchKernel_spt", &hipLaunchKernel_spt_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipLaunchHostFunc_spt", &hipLaunchHostFunc_spt_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipLaunchKernel_spt_ptr)(const void*, dim3, dim3, void**, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const void*, dim3, dim3, void**, size_t, hipStream_t)>(
          hipLaunchKernel_spt_ptr);
  hipError_t (*dyn_hipLaunchHostFunc_spt_ptr)(hipStream_t, hipHostFn_t, void*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipHostFn_t, void*)>(hipLaunchHostFunc_spt_ptr);

  const int N = 10;
  const int Nbytes = 10 * sizeof(int);

  int* hostArr = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostArr != nullptr);
  fillHostArray(hostArr, N, 10);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);
  HIP_CHECK(hipMemcpy(devArr, hostArr, Nbytes, hipMemcpyHostToDevice));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, N);

  struct kernelArgs {
    void* arr;
    int size;
  };
  kernelArgs kernelArg;
  kernelArg.arr = devArr;
  kernelArg.size = N;
  void* kernel_args[] = {&kernelArg.arr, &kernelArg.size};

  // Validating hipLaunchKernel_spt API
  {
    HIP_CHECK(dyn_hipLaunchKernel_spt_ptr(reinterpret_cast<void*>(addOneKernel), blocksPerGrid,
                                          threadsPerBlock, kernel_args, 0, nullptr));

    HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
    REQUIRE(validateHostArray(hostArr, N, 11) == true);
  }

  // Validating hipLaunchHostFunc_spt API
  {
    int data = 30;
    hipHostFn_t fn = addTen;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(dyn_hipLaunchHostFunc_spt_ptr(stream, fn, reinterpret_cast<void*>(&data)));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(data == 40);
    HIP_CHECK(hipStreamDestroy(stream));
  }
  free(hostArr);
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of
 *  - hipLaunchCooperativeKernel_spt from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_LaunchCooperativeKernel") {
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDeviceProperties(&device_properties, 0));

  if (!device_properties.cooperativeLaunch) {
    HipTest::HIP_SKIP_TEST("Skipping since cooperative launch not supported");
    return;
  }

  void* hipLaunchCooperativeKernel_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipLaunchCooperativeKernel_spt", &hipLaunchCooperativeKernel_spt_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipLaunchCooperativeKernel_spt_ptr)(const void*, dim3, dim3, void**,
                                                       unsigned int, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const void*, dim3, dim3, void**, unsigned int, hipStream_t)>(
          hipLaunchCooperativeKernel_spt_ptr);

  const int N = 10;
  const int Nbytes = 10 * sizeof(int);

  int* hostArr = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostArr != nullptr);
  fillHostArray(hostArr, N, 10);

  int* devArr = nullptr;
  HIP_CHECK(hipMalloc(&devArr, Nbytes));
  REQUIRE(devArr != nullptr);
  HIP_CHECK(hipMemcpy(devArr, hostArr, Nbytes, hipMemcpyHostToDevice));

  dim3 blocksPerGrid(1, 1, 1);
  dim3 threadsPerBlock(1, 1, N);

  struct kernelArgs {
    void* arr;
    int size;
  };
  kernelArgs kernelArg;
  kernelArg.arr = devArr;
  kernelArg.size = N;
  void* kernel_args[] = {&kernelArg.arr, &kernelArg.size};

  // Validating hipLaunchCooperativeKernel_spt API
  HIP_CHECK(dyn_hipLaunchCooperativeKernel_spt_ptr(
      reinterpret_cast<void*>(addOneKernel), blocksPerGrid, threadsPerBlock, kernel_args, 0, 0));
  HIP_CHECK(hipMemcpy(hostArr, devArr, Nbytes, hipMemcpyDeviceToHost));
  REQUIRE(validateHostArray(hostArr, N, 11) == true);

  free(hostArr);
  HIP_CHECK(hipFree(devArr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (streams) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Stream") {
  void* hipStreamGetFlags_spt_ptr = nullptr;
  void* hipStreamGetPriority_spt_ptr = nullptr;
  void* hipStreamSynchronize_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipStreamGetFlags_spt", &hipStreamGetFlags_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamGetPriority_spt", &hipStreamGetPriority_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamSynchronize_spt", &hipStreamSynchronize_spt_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipStreamGetFlags_spt_ptr)(hipStream_t, unsigned int*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, unsigned int*)>(hipStreamGetFlags_spt_ptr);
  hipError_t (*dyn_hipStreamGetPriority_spt_ptr)(hipStream_t, int*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, int*)>(hipStreamGetPriority_spt_ptr);
  hipError_t (*dyn_hipStreamSynchronize_spt_ptr)(hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipStream_t)>(hipStreamSynchronize_spt_ptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Validating hipStreamGetFlags_spt API
  {
    unsigned int flagsSpt, flagsSptWithPtr;
    HIP_CHECK(hipStreamGetFlags_spt(stream, &flagsSpt));
    HIP_CHECK(dyn_hipStreamGetFlags_spt_ptr(stream, &flagsSptWithPtr));
    REQUIRE(flagsSptWithPtr == flagsSpt);
  }

  // Validating hipStreamGetPriority_spt API
  {
    int prioritySpt, prioritySptWithPtr;
    HIP_CHECK(hipStreamGetPriority_spt(stream, &prioritySpt));
    HIP_CHECK(dyn_hipStreamGetPriority_spt_ptr(stream, &prioritySptWithPtr));
    REQUIRE(prioritySptWithPtr == prioritySpt);
  }

  HIP_CHECK(hipStreamDestroy(stream));


  // Validating hipStreamSynchronize_spt API
  {
    int N = 1024 * 1024;
    const int Ns = 4;
    int Nbytes = N * sizeof(int);
    int value = 10;

    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    fillHostArray(hostMem, N, value);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(hipMemcpy(devMem, hostMem, Nbytes, hipMemcpyHostToDevice));

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      addOneKernel<<<1, 1, 0, stream[s]>>>(devMem + startIndex, N / Ns);
    }

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(dyn_hipStreamSynchronize_spt_ptr(stream[s]));
    }

    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateHostArray(hostMem, N, 11) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }
    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (Graph) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Graph") {
  void* hipStreamBeginCapture_spt_ptr = nullptr;
  void* hipStreamIsCapturing_spt_ptr = nullptr;
  void* hipStreamEndCapture_spt_ptr = nullptr;
  void* hipStreamQuery_spt_ptr = nullptr;
  void* hipStreamGetCaptureInfo_spt_ptr = nullptr;
  void* hipStreamGetCaptureInfo_v2_spt_ptr = nullptr;
  void* hipStreamAddCallback_spt_ptr = nullptr;
  void* hipGraphLaunch_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipStreamBeginCapture_spt", &hipStreamBeginCapture_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamIsCapturing_spt", &hipStreamIsCapturing_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamEndCapture_spt", &hipStreamEndCapture_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamQuery_spt", &hipStreamQuery_spt_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamGetCaptureInfo_spt", &hipStreamGetCaptureInfo_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamGetCaptureInfo_v2_spt", &hipStreamGetCaptureInfo_v2_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamAddCallback_spt", &hipStreamAddCallback_spt_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGraphLaunch_spt", &hipGraphLaunch_spt_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipStreamBeginCapture_spt_ptr)(hipStream_t, hipStreamCaptureMode) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureMode)>(
          hipStreamBeginCapture_spt_ptr);

  hipError_t (*dyn_hipStreamIsCapturing_spt_ptr)(hipStream_t, hipStreamCaptureStatus*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus*)>(
          hipStreamIsCapturing_spt_ptr);

  hipError_t (*dyn_hipStreamEndCapture_spt_ptr)(hipStream_t, hipGraph_t*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipGraph_t*)>(hipStreamEndCapture_spt_ptr);

  hipError_t (*dyn_hipStreamQuery_spt_ptr)(hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipStream_t)>(hipStreamQuery_spt_ptr);

  hipError_t (*dyn_hipStreamGetCaptureInfo_spt_ptr)(hipStream_t, hipStreamCaptureStatus*,
                                                    unsigned long long*) =  // NOLINT
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus*,
                                      unsigned long long*)>  // NOLINT
      (hipStreamGetCaptureInfo_spt_ptr);

  hipError_t (*dyn_hipStreamGetCaptureInfo_v2_spt_ptr)(hipStream_t, hipStreamCaptureStatus*,
                                                       unsigned long long*,  // NOLINT
                                                       hipGraph_t*, const hipGraphNode_t**,
                                                       size_t*) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCaptureStatus*,
                                      unsigned long long*,  // NOLINT
                                      hipGraph_t*, const hipGraphNode_t**, size_t*)>(
          hipStreamGetCaptureInfo_v2_spt_ptr);

  hipError_t (*dyn_hipStreamAddCallback_spt_ptr)(hipStream_t, hipStreamCallback_t, void*,
                                                 unsigned int) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipStreamCallback_t, void*, unsigned int)>(
          hipStreamAddCallback_spt_ptr);

  hipError_t (*dyn_hipGraphLaunch_spt_ptr)(hipGraphExec_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipGraphExec_t, hipStream_t)>(hipGraphLaunch_spt_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  hipGraph_t graph = nullptr;
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Validating hipStreamBeginCapture_spt API
  HIP_CHECK(dyn_hipStreamBeginCapture_spt_ptr(stream, hipStreamCaptureModeGlobal));

  // Validating hipStreamIsCapturing_spt API
  hipStreamCaptureStatus pCaptureStatus = hipStreamCaptureStatusNone;
  HIP_CHECK(dyn_hipStreamIsCapturing_spt_ptr(stream, &pCaptureStatus));
  REQUIRE(pCaptureStatus == hipStreamCaptureStatusActive);

  HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice, stream));
  addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
  HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost, stream));

  // Validating hipStreamGetCaptureInfo_spt API
  hipStreamCaptureStatus pCaptureStatusWithOrgApi = hipStreamCaptureStatusNone,
                         pCaptureStatusWithFuncPtr = hipStreamCaptureStatusNone;

  unsigned long long pIdWithOrgApi = 0, pIdWithFuncPtr = 0;  // NOLINT

  HIP_CHECK(hipStreamGetCaptureInfo_spt(stream, &pCaptureStatusWithOrgApi, &pIdWithOrgApi));
  HIP_CHECK(
      dyn_hipStreamGetCaptureInfo_spt_ptr(stream, &pCaptureStatusWithFuncPtr, &pIdWithFuncPtr));

  REQUIRE(pCaptureStatusWithFuncPtr == pCaptureStatusWithOrgApi);
  REQUIRE(pIdWithFuncPtr == pIdWithOrgApi);

  // Validating hipStreamGetCaptureInfo_v2_spt API
  hipStreamCaptureStatus captureStatus_out_org, captureStatus_out_ptr;
  unsigned long long id_out_org, id_out_ptr;  // NOLINT
  hipGraph_t graph_out_org, graph_out_ptr;
  const hipGraphNode_t* dependencies_out_org{};
  const hipGraphNode_t* dependencies_out_ptr{};
  size_t numDependencies_out_org, numDependencies_out_ptr;

  HIP_CHECK(hipStreamGetCaptureInfo_v2_spt(stream, &captureStatus_out_org, &id_out_org,
                                           &graph_out_org, &dependencies_out_org,
                                           &numDependencies_out_org));
  HIP_CHECK(dyn_hipStreamGetCaptureInfo_v2_spt_ptr(stream, &captureStatus_out_ptr, &id_out_ptr,
                                                   &graph_out_ptr, &dependencies_out_ptr,
                                                   &numDependencies_out_ptr));

  REQUIRE(captureStatus_out_ptr == captureStatus_out_org);
  REQUIRE(id_out_ptr == id_out_org);
  REQUIRE(graph_out_ptr == graph_out_org);

  // Validating hipStreamEndCapture_spt API
  HIP_CHECK(dyn_hipStreamEndCapture_spt_ptr(stream, &graph));

  // Validating hipStreamAddCallback_spt API
  int data = 200;
  HIP_CHECK(dyn_hipStreamAddCallback_spt_ptr(stream, callBackFunction,
                                             reinterpret_cast<void*>(&data), 0));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  // Validating hipGraphLaunch_spt API
  HIP_CHECK(dyn_hipGraphLaunch_spt_ptr(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Validating hipStreamQuery_spt API
  REQUIRE(dyn_hipStreamQuery_spt_ptr(stream) == hipSuccess);

  REQUIRE(validateHostArray(hostMem, N, 11) == true);
  REQUIRE(data == 300);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different
 *  - spt (Events) APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/streamperthread/hipGetProcAddress_SPT_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_spt_Event") {
  void* hipEventRecord_spt_ptr = nullptr;
  void* hipStreamWaitEvent_spt_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipEventRecord_spt", &hipEventRecord_spt_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamWaitEvent_spt", &hipStreamWaitEvent_spt_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipEventRecord_spt_ptr)(hipEvent_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipEvent_t, hipStream_t)>(hipEventRecord_spt_ptr);

  hipError_t (*dyn_hipStreamWaitEvent_spt_ptr)(hipStream_t, hipEvent_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipStream_t, hipEvent_t, unsigned int)>(
          hipStreamWaitEvent_spt_ptr);

  int N = 40;
  int Nbytes = N * sizeof(int);

  int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hostMem != nullptr);
  fillHostArray(hostMem, N, 10);

  int* devMem = nullptr;
  HIP_CHECK(hipMalloc(&devMem, Nbytes));
  REQUIRE(devMem != nullptr);

  // Validating hipEventRecord_spt API with null stream
  {
    hipEvent_t start = nullptr, stop = nullptr;

    HIP_CHECK(hipEventCreate(&start));
    REQUIRE(start != nullptr);

    HIP_CHECK(hipEventCreate(&stop));
    REQUIRE(stop != nullptr);

    HIP_CHECK(dyn_hipEventRecord_spt_ptr(start, NULL));

    HIP_CHECK(hipMemcpy(devMem, hostMem, Nbytes, hipMemcpyHostToDevice));
    addOneKernel<<<1, 1>>>(devMem, N);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    HIP_CHECK(dyn_hipEventRecord_spt_ptr(stop, NULL));
    HIP_CHECK(hipEventSynchronize(stop));

    REQUIRE(validateHostArray(hostMem, N, 11) == true);

    float time = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&time, start, stop));
    REQUIRE(time > 0.0f);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
  }

  // Validating hipEventRecord_spt API with user defined stream
  {
    hipEvent_t start = nullptr, stop = nullptr;

    HIP_CHECK(hipEventCreate(&start));
    REQUIRE(start != nullptr);

    HIP_CHECK(hipEventCreate(&stop));
    REQUIRE(stop != nullptr);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(dyn_hipEventRecord_spt_ptr(start, stream));

    HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice, stream));
    addOneKernel<<<1, 1, 0, stream>>>(devMem, N);
    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost, stream));

    HIP_CHECK(dyn_hipEventRecord_spt_ptr(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));

    REQUIRE(validateHostArray(hostMem, N, 12) == true);

    float time = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&time, start, stop));
    REQUIRE(time > 0.0f);

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Validating hipStreamWaitEvent_spt API
  {
    hipEvent_t waitEvent = nullptr;

    HIP_CHECK(hipEventCreate(&waitEvent));
    REQUIRE(waitEvent != nullptr);

    hipStream_t stream1, stream2;
    HIP_CHECK(hipStreamCreate(&stream1));
    HIP_CHECK(hipStreamCreate(&stream2));

    HIP_CHECK(hipMemcpyAsync(devMem, hostMem, Nbytes, hipMemcpyHostToDevice, stream1));

    HIP_CHECK(hipEventRecord(waitEvent, stream1));
    HIP_CHECK(dyn_hipStreamWaitEvent_spt_ptr(stream2, waitEvent, 0));

    addOneKernel<<<1, 1, 0, stream2>>>(devMem, N);
    HIP_CHECK(hipMemcpyAsync(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost, stream2));

    HIP_CHECK(hipStreamSynchronize(stream2));

    REQUIRE(validateHostArray(hostMem, N, 13) == true);

    HIP_CHECK(hipEventDestroy(waitEvent));
    HIP_CHECK(hipStreamDestroy(stream1));
    HIP_CHECK(hipStreamDestroy(stream2));
  }
  free(hostMem);
  HIP_CHECK(hipFree(devMem));
}
