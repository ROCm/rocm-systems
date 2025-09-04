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
#include <utils.hh>
#include "hipMallocManagedCommon.hh"
#include "../device/hipGetProcAddressHelpers.hh"

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Allocation and free) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMallocFree") {
  void* hipMalloc_ptr = nullptr;
  void* hipFree_ptr = nullptr;
  void* hipExtMallocWithFlags_ptr = nullptr;
  void* hipMallocHost_ptr = nullptr;
  void* hipMemAllocHost_ptr = nullptr;
  void* hipHostMalloc_ptr = nullptr;
  void* hipHostAlloc_ptr = nullptr;
  void* hipHostGetDevicePointer_ptr = nullptr;
  void* hipHostGetFlags_ptr = nullptr;
  void* hipMallocPitch_ptr = nullptr;
  void* hipMemAllocPitch_ptr = nullptr;
  void* hipFreeHost_ptr = nullptr;
  void* hipHostFree_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMalloc", &hipMalloc_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFree", &hipFree_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipExtMallocWithFlags", &hipExtMallocWithFlags_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMallocHost", &hipMallocHost_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemAllocHost", &hipMemAllocHost_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHostMalloc", &hipHostMalloc_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHostAlloc", &hipHostAlloc_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHostGetDevicePointer", &hipHostGetDevicePointer_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipHostGetFlags", &hipHostGetFlags_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMallocPitch", &hipMallocPitch_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemAllocPitch", &hipMemAllocPitch_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFreeHost", &hipFreeHost_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHostFree", &hipHostFree_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMalloc_ptr)(void**, size_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t)>(hipMalloc_ptr);
  hipError_t (*dyn_hipFree_ptr)(void*) = reinterpret_cast<hipError_t (*)(void*)>(hipFree_ptr);
  hipError_t (*dyn_hipExtMallocWithFlags_ptr)(void**, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(void**, size_t, unsigned int)>(hipExtMallocWithFlags_ptr);
  hipError_t (*dyn_hipMallocHost_ptr)(void**, size_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t)>(hipMallocHost_ptr);
  hipError_t (*dyn_hipMemAllocHost_ptr)(void**, size_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t)>(hipMemAllocHost_ptr);
  hipError_t (*dyn_hipHostMalloc_ptr)(void**, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(void**, size_t, unsigned int)>(hipHostMalloc_ptr);
  hipError_t (*dyn_hipHostAlloc_ptr)(void**, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(void**, size_t, unsigned int)>(hipHostAlloc_ptr);
  hipError_t (*dyn_hipHostGetDevicePointer_ptr)(void**, void*, unsigned int) =
      reinterpret_cast<hipError_t (*)(void**, void*, unsigned int)>(hipHostGetDevicePointer_ptr);
  hipError_t (*dyn_hipHostGetFlags_ptr)(unsigned int*, void*) =
      reinterpret_cast<hipError_t (*)(unsigned int*, void*)>(hipHostGetFlags_ptr);
  hipError_t (*dyn_hipMallocPitch_ptr)(void**, size_t*, size_t, size_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t*, size_t, size_t)>(hipMallocPitch_ptr);
  hipError_t (*dyn_hipMemAllocPitch_ptr)(hipDeviceptr_t*, size_t*, size_t, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t*, size_t*, size_t, size_t, unsigned int)>(
          hipMemAllocPitch_ptr);
  hipError_t (*dyn_hipFreeHost_ptr)(void*) =
      reinterpret_cast<hipError_t (*)(void*)>(hipFreeHost_ptr);
  hipError_t (*dyn_hipHostFree_ptr)(void*) =
      reinterpret_cast<hipError_t (*)(void*)>(hipHostFree_ptr);

  // Validating hipMalloc and hipFree APIs
  {
    void* d_ptr = nullptr;
    HIP_CHECK(dyn_hipMalloc_ptr(&d_ptr, 256));

    REQUIRE(d_ptr != nullptr);
    size_t d_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(d_ptr, &d_ptr_size));
    REQUIRE(d_ptr_size == 256);

    HIP_CHECK(dyn_hipFree_ptr(d_ptr));
    REQUIRE(hipMemPtrGetInfo(d_ptr, &d_ptr_size) == hipErrorInvalidValue);
  }

  // Validating hipExtMallocWithFlags API
  {
    void* ext_d_ptr = nullptr;
    size_t ext_d_ptr_size;

    ::std::vector<unsigned int> ext_mlc_flags;
    ext_mlc_flags.push_back(hipDeviceMallocDefault);
    ext_mlc_flags.push_back(hipDeviceMallocUncached);
    ext_mlc_flags.push_back(hipMallocSignalMemory);

    if (DeviceAttributesSupport(0, hipDeviceAttributeFineGrainSupport)) {
      ext_mlc_flags.push_back(hipDeviceMallocFinegrained);
    }

    for (unsigned int flag : ext_mlc_flags) {
      ext_d_ptr = nullptr;
      HIP_CHECK(dyn_hipExtMallocWithFlags_ptr(&ext_d_ptr, 8, flag));
      REQUIRE(ext_d_ptr != nullptr);

      ext_d_ptr_size = -1;
      HIP_CHECK(hipMemPtrGetInfo(ext_d_ptr, &ext_d_ptr_size));
      REQUIRE(ext_d_ptr_size == 8);
      HIP_CHECK(hipFree(ext_d_ptr));
    }
  }

  // Validating hipMallocHost API
  {
    void* h_ptr = nullptr;
    HIP_CHECK(dyn_hipMallocHost_ptr(&h_ptr, 128));
    REQUIRE(h_ptr != nullptr);
    size_t h_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
    REQUIRE(h_ptr_size == 128);
    HIP_CHECK(hipFree(h_ptr));
  }

  // Validating hipMemAllocHost API
  {
    void* h_ptr = nullptr;
    HIP_CHECK(dyn_hipMemAllocHost_ptr(&h_ptr, 256));
    REQUIRE(h_ptr != nullptr);
    size_t h_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
    REQUIRE(h_ptr_size == 256);
    HIP_CHECK(hipFree(h_ptr));
  }

  // Validating hipHostMalloc API
  {
    void* h_ptr = nullptr;
    size_t h_ptr_size = -1;

    unsigned int h3_flags[] = {hipHostMallocCoherent, hipHostMallocNonCoherent, hipHostMallocMapped,
                               hipHostMallocNumaUser};

    for (unsigned int flag : h3_flags) {
      h_ptr = nullptr;

      HIP_CHECK(dyn_hipHostMalloc_ptr(&h_ptr, 256, flag));
      REQUIRE(h_ptr != nullptr);

      h_ptr_size = -1;
      HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
      REQUIRE(h_ptr_size == 256);
      HIP_CHECK(hipFree(h_ptr));
    }
  }

  // Validating hipHostAlloc API
  {
    void* h_ptr = nullptr;
    size_t h_ptr_size = -1;

    HIP_CHECK(dyn_hipHostAlloc_ptr(&h_ptr, 256, 0));
    REQUIRE(h_ptr != nullptr);

    h_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
    REQUIRE(h_ptr_size == 256);
    HIP_CHECK(hipFree(h_ptr));
  }

  // Validating hipHostGetDevicePointer API
  {
    void* hostPtr = nullptr;
    void* devicePtrForhostPtr = nullptr;
    void* devicePtrForhostPtrWithFuncPtr = nullptr;
    size_t hostPtr_size = -1;
    size_t hostPtr_ptr_size = -1;

    unsigned int flags[] = {hipHostMallocCoherent, hipHostMallocNonCoherent, hipHostMallocMapped,
                            hipHostMallocNumaUser};

    for (unsigned int flag : flags) {
      hostPtr = nullptr;

      HIP_CHECK(hipHostMalloc(&hostPtr, 1024, flag));
      REQUIRE(hostPtr != nullptr);

      devicePtrForhostPtr = nullptr;
      devicePtrForhostPtrWithFuncPtr = nullptr;

      HIP_CHECK(hipHostGetDevicePointer(&devicePtrForhostPtr, hostPtr, 0));
      HIP_CHECK(dyn_hipHostGetDevicePointer_ptr(&devicePtrForhostPtrWithFuncPtr, hostPtr, 0));

      REQUIRE(devicePtrForhostPtr != nullptr);
      REQUIRE(devicePtrForhostPtrWithFuncPtr != nullptr);

      REQUIRE(devicePtrForhostPtrWithFuncPtr == devicePtrForhostPtr);

      hostPtr_size = -1;
      hostPtr_ptr_size = -1;
      HIP_CHECK(hipMemPtrGetInfo(devicePtrForhostPtr, &hostPtr_size));
      REQUIRE(hostPtr_size == 1024);
      HIP_CHECK(hipMemPtrGetInfo(devicePtrForhostPtrWithFuncPtr, &hostPtr_ptr_size));
      REQUIRE(hostPtr_ptr_size == 1024);
      REQUIRE(hostPtr_size == hostPtr_ptr_size);

      HIP_CHECK(hipFree(hostPtr));
    }
  }

  // Validating hipHostGetFlags API
  {
    void* h = nullptr;
    unsigned int expect_flags = -1;
    unsigned int expect_flags_with_ptr = -1;

    unsigned int flags[] = {hipHostMallocCoherent, hipHostMallocNonCoherent, hipHostMallocMapped,
                            hipHostMallocNumaUser};

    for (unsigned int flag : flags) {
      h = nullptr;
      HIP_CHECK(hipHostMalloc(&h, 512, flag));
      REQUIRE(h != nullptr);

      expect_flags = -1;
      expect_flags_with_ptr = -1;
      HIP_CHECK(hipHostGetFlags(&expect_flags, h));
      HIP_CHECK(dyn_hipHostGetFlags_ptr(&expect_flags_with_ptr, h));
      REQUIRE(expect_flags == flag);
      REQUIRE(expect_flags_with_ptr == flag);

      REQUIRE(expect_flags_with_ptr == expect_flags);

      HIP_CHECK(hipFree(h));
    }
  }

  // Validating hipMallocPitch API
  {
    void* pitchedMem_ptr = nullptr;
    size_t pitch_ptr = -1;
    int width1 = 260;
    int height1 = 2;

    HIP_CHECK(dyn_hipMallocPitch_ptr(&pitchedMem_ptr, &pitch_ptr, width1, height1));

    size_t pitchedMem_sizeptr = -1;
    HIP_CHECK(hipMemPtrGetInfo(pitchedMem_ptr, &pitchedMem_sizeptr));
    REQUIRE(pitchedMem_sizeptr == 1024);

    HIP_CHECK(hipFree(pitchedMem_ptr));
  }

  // Validating hipMemAllocPitch API
  {
    hipDeviceptr_t pitchedMem_ptr = nullptr;
    size_t pitch_ptr = -1;
    int width = 260;
    int height = 2;
    unsigned int groupOfElementSizeBytes[] = {4, 8, 16};
    size_t pitchedMem_sizeptr = -1;

    for (auto elementSizeBytes : groupOfElementSizeBytes) {
      pitchedMem_ptr = nullptr;
      pitch_ptr = -1;

      HIP_CHECK(
          dyn_hipMemAllocPitch_ptr(&pitchedMem_ptr, &pitch_ptr, width, height, elementSizeBytes));

      REQUIRE(pitch_ptr == 512);
      pitchedMem_sizeptr = -1;
      HIP_CHECK(hipMemPtrGetInfo(pitchedMem_ptr, &pitchedMem_sizeptr));
      REQUIRE(pitchedMem_sizeptr == 1024);

      HIP_CHECK(hipFree(pitchedMem_ptr));
    }
  }

  // Validating hipFreeHost API
  {
    void* h_ptr = nullptr;
    HIP_CHECK(hipMallocHost(&h_ptr, 128));
    REQUIRE(h_ptr != nullptr);

    size_t h_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
    HIP_CHECK(dyn_hipFreeHost_ptr(h_ptr));
    REQUIRE(hipMemPtrGetInfo(h_ptr, &h_ptr_size) == hipErrorInvalidValue);
  }

  // Validating hipHostFree API
  {
    void* h_ptr = nullptr;
    HIP_CHECK(hipMallocHost(&h_ptr, 128));
    REQUIRE(h_ptr != nullptr);

    size_t h_ptr_size = -1;
    HIP_CHECK(hipMemPtrGetInfo(h_ptr, &h_ptr_size));
    HIP_CHECK(dyn_hipHostFree_ptr(h_ptr));
    REQUIRE(hipMemPtrGetInfo(h_ptr, &h_ptr_size) == hipErrorInvalidValue);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Register and unregister) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisRegisterUnReg") {
  void* hipHostRegister_ptr = nullptr;
  void* hipHostUnregister_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipHostRegister", &hipHostRegister_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipHostUnregister", &hipHostUnregister_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipHostRegister_ptr)(void*, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(void*, size_t, unsigned int)>(hipHostRegister_ptr);

  hipError_t (*dyn_hipHostUnregister_ptr)(void*) =
      reinterpret_cast<hipError_t (*)(void*)>(hipHostUnregister_ptr);

  // Validating hipHostRegister API
  void* reg_h_ptr = nullptr;
  reg_h_ptr = malloc(1024);
  REQUIRE(reg_h_ptr != nullptr);

  unsigned int reg_flags = hipHostRegisterDefault;
  HIP_CHECK(dyn_hipHostRegister_ptr(reg_h_ptr, 1024, reg_flags));

  void* devicePtrReg_ptr = nullptr;
  HIP_CHECK(hipHostGetDevicePointer(&devicePtrReg_ptr, reg_h_ptr, 0));
  REQUIRE(devicePtrReg_ptr != nullptr);

  size_t reg_ptr_size = -1;
  HIP_CHECK(hipMemPtrGetInfo(devicePtrReg_ptr, &reg_ptr_size));
  REQUIRE(reg_ptr_size == 1024);

  // Validating hipHostUnregister API
  HIP_CHECK(dyn_hipHostUnregister_ptr(reg_h_ptr));

  REQUIRE(hipHostGetDevicePointer(&devicePtrReg_ptr, reg_h_ptr, 0) == hipErrorInvalidValue);
  free(reg_h_ptr);
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Arrays) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisArrayRelated") {
  CHECK_IMAGE_SUPPORT

  void* hipMallocArray_ptr = nullptr;
  void* hipArrayCreate_ptr = nullptr;
  void* hipFreeArray_ptr = nullptr;
  void* hipArrayDestroy_ptr = nullptr;
  void* hipArrayGetInfo_ptr = nullptr;
  void* hipArray3DCreate_ptr = nullptr;
  void* hipArrayGetDescriptor_ptr = nullptr;
  void* hipArray3DGetDescriptor_ptr = nullptr;
  void* hipMalloc3DArray_ptr = nullptr;
  void* hipMalloc3D_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMallocArray", &hipMallocArray_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipArrayCreate", &hipArrayCreate_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFreeArray", &hipFreeArray_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipArrayDestroy", &hipArrayDestroy_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipArrayGetInfo", &hipArrayGetInfo_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipArray3DCreate", &hipArray3DCreate_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipArrayGetDescriptor", &hipArrayGetDescriptor_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipArray3DGetDescriptor", &hipArray3DGetDescriptor_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMalloc3DArray", &hipMalloc3DArray_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMalloc3D", &hipMalloc3D_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMallocArray_ptr)(hipArray_t*, const hipChannelFormatDesc*, size_t, size_t,
                                       unsigned int) =
      reinterpret_cast<hipError_t (*)(hipArray_t*, const hipChannelFormatDesc*, size_t, size_t,
                                      unsigned int)>(hipMallocArray_ptr);

  hipError_t (*dyn_hipArrayCreate_ptr)(hipArray_t*, const HIP_ARRAY_DESCRIPTOR*) =
      reinterpret_cast<hipError_t (*)(hipArray_t*, const HIP_ARRAY_DESCRIPTOR*)>(
          hipArrayCreate_ptr);

  hipError_t (*dyn_hipFreeArray_ptr)(hipArray_t) =
      reinterpret_cast<hipError_t (*)(hipArray_t)>(hipFreeArray_ptr);

  hipError_t (*dyn_hipArrayDestroy_ptr)(hipArray_t) =
      reinterpret_cast<hipError_t (*)(hipArray_t)>(hipArrayDestroy_ptr);

  hipError_t (*dyn_hipArrayGetInfo_ptr)(hipChannelFormatDesc*, hipExtent*, unsigned int*,
                                        hipArray_t) =
      reinterpret_cast<hipError_t (*)(hipChannelFormatDesc*, hipExtent*, unsigned int*,
                                      hipArray_t)>(hipArrayGetInfo_ptr);

  hipError_t (*dyn_hipArray3DCreate_ptr)(hipArray_t*, const HIP_ARRAY3D_DESCRIPTOR*) =
      reinterpret_cast<hipError_t (*)(hipArray_t*, const HIP_ARRAY3D_DESCRIPTOR*)>(
          hipArray3DCreate_ptr);

  hipError_t (*dyn_hipArrayGetDescriptor_ptr)(HIP_ARRAY_DESCRIPTOR*, hipArray_t) =
      reinterpret_cast<hipError_t (*)(HIP_ARRAY_DESCRIPTOR*, hipArray_t)>(
          hipArrayGetDescriptor_ptr);

  hipError_t (*dyn_hipArray3DGetDescriptor_ptr)(HIP_ARRAY3D_DESCRIPTOR*, hipArray_t) =
      reinterpret_cast<hipError_t (*)(HIP_ARRAY3D_DESCRIPTOR*, hipArray_t)>(
          hipArray3DGetDescriptor_ptr);

  hipError_t (*dyn_hipMalloc3DArray_ptr)(hipArray_t*, const struct hipChannelFormatDesc*,
                                         struct hipExtent, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipArray_t*, const struct hipChannelFormatDesc*,
                                      struct hipExtent, unsigned int)>(hipMalloc3DArray_ptr);

  hipError_t (*dyn_hipMalloc3D_ptr)(hipPitchedPtr*, hipExtent) =
      reinterpret_cast<hipError_t (*)(hipPitchedPtr*, hipExtent)>(hipMalloc3D_ptr);

  // Validating hipMallocArray API
  hipArray_t m_array = nullptr;
  hipArray_t m_array_ptr = nullptr;
  hipChannelFormatDesc m_desc = hipCreateChannelDesc<float>();
  size_t m_width = 16;
  size_t m_height = 16;
  unsigned int m_flags = hipArrayDefault;

  HIP_CHECK(hipMallocArray(&m_array, &m_desc, m_width, m_height, m_flags));
  HIP_CHECK(dyn_hipMallocArray_ptr(&m_array_ptr, &m_desc, m_width, m_height, m_flags));

  REQUIRE(m_array != nullptr);
  REQUIRE(m_array_ptr != nullptr);

  HIP_ARRAY_DESCRIPTOR m_array_desc;
  HIP_CHECK(hipArrayGetDescriptor(&m_array_desc, m_array));
  REQUIRE(m_array_desc.Width == m_width);
  REQUIRE(m_array_desc.Height == m_height);

  HIP_ARRAY_DESCRIPTOR m_array_ptr_desc;
  HIP_CHECK(hipArrayGetDescriptor(&m_array_ptr_desc, m_array_ptr));
  REQUIRE(m_array_ptr_desc.Width == m_width);
  REQUIRE(m_array_ptr_desc.Height == m_height);

  REQUIRE(m_array_ptr_desc.Width == m_array_desc.Width);
  REQUIRE(m_array_ptr_desc.Height == m_array_desc.Height);

  HIP_CHECK(hipFreeArray(m_array));
  HIP_CHECK(hipFreeArray(m_array_ptr));

  // Validating hipArrayCreate API
  hipArray_t array = nullptr;
  hipArray_t array_ptr = nullptr;

  HIP_ARRAY_DESCRIPTOR desc;
  desc.Format = HIP_AD_FORMAT_FLOAT;
  desc.NumChannels = 1;
  desc.Width = 8;
  desc.Height = 8;

  HIP_CHECK(hipArrayCreate(&array, &desc));
  HIP_CHECK(dyn_hipArrayCreate_ptr(&array_ptr, &desc));

  REQUIRE(array != nullptr);
  REQUIRE(array_ptr != nullptr);

  HIP_ARRAY_DESCRIPTOR array_desc;
  HIP_CHECK(hipArrayGetDescriptor(&array_desc, array));
  REQUIRE(array_desc.Width == desc.Width);
  REQUIRE(array_desc.Height == desc.Height);

  HIP_ARRAY_DESCRIPTOR array_ptr_desc;
  HIP_CHECK(hipArrayGetDescriptor(&array_ptr_desc, array_ptr));
  REQUIRE(array_ptr_desc.Width == desc.Width);
  REQUIRE(array_ptr_desc.Height == desc.Height);

  REQUIRE(array_ptr_desc.Width == array_desc.Width);
  REQUIRE(array_ptr_desc.Height == array_desc.Height);

  HIP_CHECK(hipFreeArray(array));
  HIP_CHECK(hipFreeArray(array_ptr));

  // Validating hipFreeArray API
  hipArray_t f_array = nullptr;
  hipArray_t f_array_ptr = nullptr;

  HIP_CHECK(hipArrayCreate(&f_array, &desc));
  HIP_CHECK(hipArrayCreate(&f_array_ptr, &desc));

  REQUIRE(f_array != nullptr);
  REQUIRE(f_array_ptr != nullptr);

  HIP_CHECK(hipFreeArray(f_array));
  HIP_CHECK(dyn_hipFreeArray_ptr(f_array_ptr));

  HIP_ARRAY_DESCRIPTOR f_array_desc;
  REQUIRE(hipArrayGetDescriptor(&f_array_desc, f_array) == hipErrorInvalidHandle);
  REQUIRE(hipArrayGetDescriptor(&f_array_desc, f_array_ptr) == hipErrorInvalidHandle);

  // Validating hipArrayDestroy API
  hipArray_t d_array = nullptr;
  hipArray_t d_array_ptr = nullptr;

  HIP_CHECK(hipArrayCreate(&d_array, &desc));
  HIP_CHECK(hipArrayCreate(&d_array_ptr, &desc));

  REQUIRE(d_array != nullptr);
  REQUIRE(d_array_ptr != nullptr);

  HIP_CHECK(hipArrayDestroy(d_array));
  HIP_CHECK(dyn_hipArrayDestroy_ptr(d_array_ptr));

  HIP_ARRAY_DESCRIPTOR d_array_desc;
  REQUIRE(hipArrayGetDescriptor(&d_array_desc, d_array) == hipErrorInvalidHandle);
  REQUIRE(hipArrayGetDescriptor(&d_array_desc, d_array_ptr) == hipErrorInvalidHandle);

  // Validating hipArrayGetInfo API
  hipArray_t gi_array = nullptr;
  hipArray_t gi_array_ptr = nullptr;

  HIP_ARRAY_DESCRIPTOR gi_desc;
  gi_desc.Format = HIP_AD_FORMAT_FLOAT;
  gi_desc.NumChannels = 1;
  gi_desc.Width = 64;
  gi_desc.Height = 64;

  HIP_CHECK(hipArrayCreate(&gi_array, &gi_desc));
  HIP_CHECK(hipArrayCreate(&gi_array_ptr, &gi_desc));

  REQUIRE(gi_array != nullptr);
  REQUIRE(gi_array_ptr != nullptr);

  hipChannelFormatDesc gi_array_desc;
  hipExtent gi_array_extent;
  unsigned int gi_array_flags;
  HIP_CHECK(hipArrayGetInfo(&gi_array_desc, &gi_array_extent, &gi_array_flags, gi_array));
  REQUIRE(gi_array_desc.x == 32);
  REQUIRE(gi_array_desc.y == 0);
  REQUIRE(gi_array_desc.z == 0);
  REQUIRE(gi_array_desc.w == 0);
  REQUIRE(gi_array_desc.f == hipChannelFormatKindFloat);
  REQUIRE(gi_array_extent.width == 64);
  REQUIRE(gi_array_extent.height == 64);
  REQUIRE(gi_array_extent.depth == 0);
  REQUIRE(gi_array_flags == 0);

  hipChannelFormatDesc gi_array_desc_ptr;
  hipExtent gi_array_extent_ptr;
  unsigned int gi_array_flags_ptr;
  HIP_CHECK(dyn_hipArrayGetInfo_ptr(&gi_array_desc_ptr, &gi_array_extent_ptr, &gi_array_flags_ptr,
                                    gi_array_ptr));
  REQUIRE(gi_array_desc_ptr.x == 32);
  REQUIRE(gi_array_desc_ptr.y == 0);
  REQUIRE(gi_array_desc_ptr.z == 0);
  REQUIRE(gi_array_desc_ptr.w == 0);
  REQUIRE(gi_array_desc_ptr.f == hipChannelFormatKindFloat);
  REQUIRE(gi_array_extent_ptr.width == 64);
  REQUIRE(gi_array_extent_ptr.height == 64);
  REQUIRE(gi_array_extent_ptr.depth == 0);
  REQUIRE(gi_array_flags_ptr == 0);

  REQUIRE(gi_array_desc_ptr.x == gi_array_desc.x);
  REQUIRE(gi_array_desc_ptr.y == gi_array_desc.y);
  REQUIRE(gi_array_desc_ptr.z == gi_array_desc.z);
  REQUIRE(gi_array_desc_ptr.w == gi_array_desc.w);
  REQUIRE(gi_array_desc_ptr.f == gi_array_desc.f);
  REQUIRE(gi_array_extent_ptr.width == gi_array_extent.width);
  REQUIRE(gi_array_extent_ptr.height == gi_array_extent.height);
  REQUIRE(gi_array_extent_ptr.depth == gi_array_extent.depth);
  REQUIRE(gi_array_flags_ptr == gi_array_flags);

  HIP_CHECK(hipFreeArray(gi_array));
  HIP_CHECK(hipFreeArray(gi_array_ptr));

  // Validating hipArrayGetDescriptor API
  hipArray_t gd_array = nullptr;
  hipArray_t gd_array_ptr = nullptr;

  HIP_ARRAY_DESCRIPTOR gd_desc;
  gd_desc.Format = HIP_AD_FORMAT_FLOAT;
  gd_desc.NumChannels = 1;
  gd_desc.Width = 32;
  gd_desc.Height = 32;

  HIP_CHECK(hipArrayCreate(&gd_array, &gd_desc));
  HIP_CHECK(hipArrayCreate(&gd_array_ptr, &gd_desc));

  REQUIRE(gd_array != nullptr);
  REQUIRE(gd_array_ptr != nullptr);

  HIP_ARRAY_DESCRIPTOR gd_array_desc;
  HIP_CHECK(hipArrayGetDescriptor(&gd_array_desc, gd_array));
  REQUIRE(gd_array_desc.Format == HIP_AD_FORMAT_FLOAT);
  REQUIRE(gd_array_desc.NumChannels == 1);
  REQUIRE(gd_array_desc.Width == 32);
  REQUIRE(gd_array_desc.Height == 32);

  HIP_ARRAY_DESCRIPTOR gd_array_desc_ptr;
  HIP_CHECK(dyn_hipArrayGetDescriptor_ptr(&gd_array_desc_ptr, gd_array_ptr));
  REQUIRE(gd_array_desc_ptr.Format == HIP_AD_FORMAT_FLOAT);
  REQUIRE(gd_array_desc_ptr.NumChannels == 1);
  REQUIRE(gd_array_desc_ptr.Width == 32);
  REQUIRE(gd_array_desc_ptr.Height == 32);

  REQUIRE(gd_array_desc_ptr.Format == gd_array_desc.Format);
  REQUIRE(gd_array_desc_ptr.NumChannels == gd_array_desc.NumChannels);
  REQUIRE(gd_array_desc_ptr.Width == gd_array_desc.Width);
  REQUIRE(gd_array_desc_ptr.Height == gd_array_desc.Height);

  HIP_CHECK(hipFreeArray(gd_array));
  HIP_CHECK(hipFreeArray(gd_array_ptr));

  // Validating hipArray3DCreate API
  hipArray_t array3d = nullptr;
  hipArray_t array3d_ptr = nullptr;

  HIP_ARRAY3D_DESCRIPTOR desc3d;
  desc3d.Format = HIP_AD_FORMAT_FLOAT;
  desc3d.NumChannels = 1;
  desc3d.Width = 8;
  desc3d.Height = 4;
  desc3d.Depth = 2;

  HIP_CHECK(hipArray3DCreate(&array3d, &desc3d));
  HIP_CHECK(dyn_hipArray3DCreate_ptr(&array3d_ptr, &desc3d));

  REQUIRE(array3d != nullptr);
  REQUIRE(array3d_ptr != nullptr);

  HIP_ARRAY3D_DESCRIPTOR array_desc3d;
  HIP_CHECK(hipArray3DGetDescriptor(&array_desc3d, array3d));
  REQUIRE(array_desc3d.Width == desc3d.Width);
  REQUIRE(array_desc3d.Height == desc3d.Height);
  REQUIRE(array_desc3d.Depth == desc3d.Depth);

  HIP_ARRAY3D_DESCRIPTOR array_ptr_desc3d;
  HIP_CHECK(hipArray3DGetDescriptor(&array_ptr_desc3d, array3d_ptr));
  REQUIRE(array_ptr_desc3d.Width == desc3d.Width);
  REQUIRE(array_ptr_desc3d.Height == desc3d.Height);
  REQUIRE(array_ptr_desc3d.Depth == desc3d.Depth);

  REQUIRE(array_ptr_desc3d.Width == array_desc3d.Width);
  REQUIRE(array_ptr_desc3d.Height == array_desc3d.Height);
  REQUIRE(array_ptr_desc3d.Depth == array_desc3d.Depth);

  HIP_CHECK(hipArrayDestroy(array3d));
  HIP_CHECK(hipArrayDestroy(array3d_ptr));

  // Validating hipArray3DGetDescriptor API
  hipArray_t gd_array3d = nullptr;
  hipArray_t gd_array3d_ptr = nullptr;

  HIP_ARRAY3D_DESCRIPTOR gd_desc3d;
  gd_desc3d.Format = HIP_AD_FORMAT_FLOAT;
  gd_desc3d.NumChannels = 1;
  gd_desc3d.Width = 16;
  gd_desc3d.Height = 4;
  gd_desc3d.Depth = 8;

  HIP_CHECK(hipArray3DCreate(&gd_array3d, &gd_desc3d));
  HIP_CHECK(hipArray3DCreate(&gd_array3d_ptr, &gd_desc3d));

  REQUIRE(gd_array3d != nullptr);
  REQUIRE(gd_array3d_ptr != nullptr);

  HIP_ARRAY3D_DESCRIPTOR gd_array_desc3d;
  HIP_CHECK(hipArray3DGetDescriptor(&gd_array_desc3d, gd_array3d));
  REQUIRE(gd_array_desc3d.Width == gd_desc3d.Width);
  REQUIRE(gd_array_desc3d.Height == gd_desc3d.Height);
  REQUIRE(gd_array_desc3d.Depth == gd_desc3d.Depth);

  HIP_ARRAY3D_DESCRIPTOR gd_array_ptr_desc3d;
  HIP_CHECK(dyn_hipArray3DGetDescriptor_ptr(&gd_array_ptr_desc3d, gd_array3d_ptr));
  REQUIRE(gd_array_ptr_desc3d.Width == gd_desc3d.Width);
  REQUIRE(gd_array_ptr_desc3d.Height == gd_desc3d.Height);
  REQUIRE(gd_array_ptr_desc3d.Depth == gd_desc3d.Depth);

  REQUIRE(gd_array_ptr_desc3d.Width == gd_array_desc3d.Width);
  REQUIRE(gd_array_ptr_desc3d.Height == gd_array_desc3d.Height);
  REQUIRE(gd_array_ptr_desc3d.Depth == gd_array_desc3d.Depth);

  HIP_CHECK(hipArrayDestroy(gd_array3d));
  HIP_CHECK(hipArrayDestroy(gd_array3d_ptr));

  // Validating hipMalloc3DArray API
  hipArray_t m_array3d = nullptr;
  hipArray_t m_array3d_ptr = nullptr;

  hipChannelFormatDesc m_desc3d = hipCreateChannelDesc<float>();
  hipExtent m_extent3d{12, 16, 8};
  unsigned int m_flags3d = hipArrayDefault;

  HIP_CHECK(hipMalloc3DArray(&m_array3d, &m_desc3d, m_extent3d, m_flags3d));
  HIP_CHECK(dyn_hipMalloc3DArray_ptr(&m_array3d_ptr, &m_desc3d, m_extent3d, m_flags3d));

  REQUIRE(m_array3d != nullptr);
  REQUIRE(m_array3d_ptr != nullptr);

  HIP_ARRAY3D_DESCRIPTOR m_array_desc3d;
  HIP_CHECK(hipArray3DGetDescriptor(&m_array_desc3d, m_array3d));
  REQUIRE(m_array_desc3d.Width == 12);
  REQUIRE(m_array_desc3d.Height == 16);
  REQUIRE(m_array_desc3d.Depth == 8);

  HIP_ARRAY3D_DESCRIPTOR m_array_ptr_desc3d;
  HIP_CHECK(hipArray3DGetDescriptor(&m_array_ptr_desc3d, m_array3d_ptr));
  REQUIRE(m_array_ptr_desc3d.Width == 12);
  REQUIRE(m_array_ptr_desc3d.Height == 16);
  REQUIRE(m_array_ptr_desc3d.Depth == 8);

  REQUIRE(m_array_ptr_desc3d.Width == m_array_desc3d.Width);
  REQUIRE(m_array_ptr_desc3d.Height == m_array_desc3d.Height);
  REQUIRE(m_array_ptr_desc3d.Depth == m_array_desc3d.Depth);

  HIP_CHECK(hipFreeArray(m_array3d));
  HIP_CHECK(hipFreeArray(m_array3d_ptr));

  // Validating hipMalloc3D API
  hipPitchedPtr p_array3d;
  hipPitchedPtr p_array3d_ptr;

  hipExtent p_extent3d{260, 16, 8};
  HIP_CHECK(hipMalloc3D(&p_array3d, p_extent3d));
  HIP_CHECK(dyn_hipMalloc3D_ptr(&p_array3d_ptr, p_extent3d));

  REQUIRE(p_array3d.ptr != nullptr);
  size_t p_size = -1;
  HIP_CHECK(hipMemPtrGetInfo(p_array3d.ptr, &p_size));
  REQUIRE(p_size == 65536);
  REQUIRE(p_array3d.pitch == 512);
  REQUIRE(p_array3d.xsize == 260);
  REQUIRE(p_array3d.ysize == 16);

  REQUIRE(p_array3d_ptr.ptr != nullptr);
  size_t p_size_ptr = -1;
  HIP_CHECK(hipMemPtrGetInfo(p_array3d_ptr.ptr, &p_size_ptr));
  REQUIRE(p_size_ptr == 65536);
  REQUIRE(p_array3d_ptr.pitch == 512);
  REQUIRE(p_array3d_ptr.xsize == 260);
  REQUIRE(p_array3d_ptr.ysize == 16);

  REQUIRE(p_size_ptr == p_size);
  REQUIRE(p_array3d_ptr.pitch == p_array3d.pitch);
  REQUIRE(p_array3d_ptr.xsize == p_array3d.xsize);
  REQUIRE(p_array3d_ptr.ysize == p_array3d.ysize);

  HIP_CHECK(hipFree(p_array3d.ptr));
  HIP_CHECK(hipFree(p_array3d_ptr.ptr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Set and Get Attributes) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisSetAndGetAttributes") {
  void* hipPointerGetAttribute_ptr = nullptr;
  void* hipPointerGetAttributes_ptr = nullptr;
  void* hipDrvPointerGetAttributes_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipPointerGetAttribute", &hipPointerGetAttribute_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipPointerGetAttributes", &hipPointerGetAttributes_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDrvPointerGetAttributes", &hipDrvPointerGetAttributes_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipPointerGetAttribute_ptr)(void*, hipPointer_attribute, hipDeviceptr_t) =
      reinterpret_cast<hipError_t (*)(void*, hipPointer_attribute, hipDeviceptr_t)>(
          hipPointerGetAttribute_ptr);
  hipError_t (*dyn_hipPointerGetAttributes_ptr)(hipPointerAttribute_t*, const void*) =
      reinterpret_cast<hipError_t (*)(hipPointerAttribute_t*, const void*)>(
          hipPointerGetAttributes_ptr);
  hipError_t (*dyn_hipDrvPointerGetAttributes_ptr)(unsigned int, hipPointer_attribute*, void**,
                                                   hipDeviceptr_t) =
      reinterpret_cast<hipError_t (*)(unsigned int, hipPointer_attribute*, void**, hipDeviceptr_t)>(
          hipDrvPointerGetAttributes_ptr);

#if __linux__
  // Validating hipPointerSetAttribute API
  {
    void* hipPointerSetAttribute_ptr = nullptr;
    HIP_CHECK(hipGetProcAddress("hipPointerSetAttribute", &hipPointerSetAttribute_ptr,
                                currentHipVersion, 0, nullptr));

    hipError_t (*dyn_hipPointerSetAttribute_ptr)(const void*, hipPointer_attribute,
                                                 hipDeviceptr_t) =
        reinterpret_cast<hipError_t (*)(const void*, hipPointer_attribute, hipDeviceptr_t)>(
            hipPointerSetAttribute_ptr);

    void* devPtr = nullptr;
    HIP_CHECK(hipMalloc(&devPtr, 1024));
    REQUIRE(devPtr != nullptr);

    // HIP_POINTER_ATTRIBUTE_CONTEXT
    int attrDataContext = 10;
    if (hipPointerSetAttribute(&attrDataContext, HIP_POINTER_ATTRIBUTE_CONTEXT,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataContext, HIP_POINTER_ATTRIBUTE_CONTEXT,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    int attrDataMemoryType = 1;
    if (hipPointerSetAttribute(&attrDataMemoryType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataMemoryType,
                                               HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    uint64_t attrDataDevicePointerUl = (uint64_t)devPtr;
    if (hipPointerSetAttribute(&attrDataDevicePointerUl, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataDevicePointerUl,
                                               HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_HOST_POINTER
    uint64_t attrDataHostPointer = (uint64_t)devPtr;
    if (hipPointerSetAttribute(&attrDataHostPointer, HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataHostPointer,
                                               HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_P2P_TOKENS
    int attrDataP2pTokens = 1;
    if (hipPointerSetAttribute(&attrDataP2pTokens, HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataP2pTokens, HIP_POINTER_ATTRIBUTE_P2P_TOKENS,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    int attrDataSyncMemops = 1;
    if (hipPointerSetAttribute(&attrDataSyncMemops, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataSyncMemops,
                                               HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_BUFFER_ID
    int attrDataBufferId = 1;
    if (hipPointerSetAttribute(&attrDataBufferId, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataBufferId, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_IS_MANAGED
    int attrDataIsManaged = 1;
    if (hipPointerSetAttribute(&attrDataIsManaged, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataIsManaged, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    int attrDataDeviceOrdinal = 1;
    if (hipPointerSetAttribute(&attrDataDeviceOrdinal, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataDeviceOrdinal,
                                               HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE
    int attrDataIsLegacyHipIpcCapable = 1;
    if (hipPointerSetAttribute(&attrDataIsLegacyHipIpcCapable,
                               HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataIsLegacyHipIpcCapable,
                                               HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    uint64_t attrDataRangeStartAddrUl = (uint64_t)devPtr;
    if (hipPointerSetAttribute(&attrDataRangeStartAddrUl, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataRangeStartAddrUl,
                                               HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    int attrDataRangeSize = 1024;
    if (hipPointerSetAttribute(&attrDataRangeSize, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataRangeSize, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_MAPPED
    int attributeDataMapped = 1;
    if (hipPointerSetAttribute(&attributeDataMapped, HIP_POINTER_ATTRIBUTE_MAPPED,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attributeDataMapped, HIP_POINTER_ATTRIBUTE_MAPPED,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
    int attrDataAllowedHandleTypes = 1;
    if (hipPointerSetAttribute(&attrDataAllowedHandleTypes,
                               HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataAllowedHandleTypes,
                                               HIP_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
    int attrDataIsGpuDirectRdmaCapable = 1;
    if (hipPointerSetAttribute(&attrDataIsGpuDirectRdmaCapable,
                               HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataIsGpuDirectRdmaCapable,
                                               HIP_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    int attrDataAccessFlags = 1;
    if (hipPointerSetAttribute(&attrDataAccessFlags, HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataAccessFlags,
                                               HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    // HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE
    int attrDataMempoolHandle = 1;
    if (hipPointerSetAttribute(&attrDataMempoolHandle, HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                               reinterpret_cast<hipDeviceptr_t>(devPtr)) == hipSuccess) {
      HIP_CHECK(dyn_hipPointerSetAttribute_ptr(&attrDataMempoolHandle,
                                               HIP_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                                               reinterpret_cast<hipDeviceptr_t>(devPtr)));
    }
    HIP_CHECK(hipFree(devPtr));
  }
#endif

  // Validating hipPointerGetAttribute API
  {
    void* devPtr1 = nullptr;
    HIP_CHECK(hipMalloc(&devPtr1, 1024));
    REQUIRE(devPtr1 != nullptr);

    // HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
    hipMemoryType memType;
    HIP_CHECK(hipPointerGetAttribute(&memType, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    hipMemoryType memTypeWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&memTypeWithPtr, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(memTypeWithPtr == memType);

    // HIP_POINTER_ATTRIBUTE_DEVICE_POINTER
    hipDeviceptr_t devPointer = nullptr;
    HIP_CHECK(hipPointerGetAttribute(&devPointer, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    hipDeviceptr_t devPointerWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&devPointerWithPtr,
                                             HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(devPointerWithPtr == devPointer);

    // HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS
    bool syncMemOps;
    HIP_CHECK(hipPointerGetAttribute(&syncMemOps, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    bool syncMemOpsWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&syncMemOpsWithPtr, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(syncMemOpsWithPtr == syncMemOps);

    // HIP_POINTER_ATTRIBUTE_BUFFER_ID
    int bufferId;
    HIP_CHECK(hipPointerGetAttribute(&bufferId, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    int bufferIdWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&bufferIdWithPtr, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(bufferIdWithPtr == bufferId);

    // HIP_POINTER_ATTRIBUTE_IS_MANAGED
    bool isManaged;
    HIP_CHECK(hipPointerGetAttribute(&isManaged, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    bool isManagedWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&isManagedWithPtr, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(isManagedWithPtr == isManaged);

    // HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
    int deviceOrdinal;
    HIP_CHECK(hipPointerGetAttribute(&deviceOrdinal, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    int deviceOrdinalWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&deviceOrdinalWithPtr,
                                             HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(deviceOrdinalWithPtr == deviceOrdinal);

    // HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
    hipDeviceptr_t startAddr = nullptr;
    HIP_CHECK(hipPointerGetAttribute(&startAddr, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    hipDeviceptr_t startAddrWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&startAddrWithPtr,
                                             HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(startAddrWithPtr == startAddr);

    // HIP_POINTER_ATTRIBUTE_RANGE_SIZE
    int rangeSizeVal;
    HIP_CHECK(hipPointerGetAttribute(&rangeSizeVal, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    int rangeSizeValWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&rangeSizeValWithPtr, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(rangeSizeValWithPtr == rangeSizeVal);

    // HIP_POINTER_ATTRIBUTE_MAPPED
    bool isMapped;
    HIP_CHECK(hipPointerGetAttribute(&isMapped, HIP_POINTER_ATTRIBUTE_MAPPED,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    bool isMappedWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&isMappedWithPtr, HIP_POINTER_ATTRIBUTE_MAPPED,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(isMappedWithPtr == isMapped);

    // HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS
    int accessFlags;
    HIP_CHECK(hipPointerGetAttribute(&accessFlags, HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,
                                     reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    int accessFlagsWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttribute_ptr(&accessFlagsWithPtr,
                                             HIP_POINTER_ATTRIBUTE_ACCESS_FLAGS,
                                             reinterpret_cast<hipDeviceptr_t>(devPtr1)));
    REQUIRE(accessFlagsWithPtr == accessFlags);

    HIP_CHECK(hipFree(devPtr1));
  }

  // Validating hipPointerGetAttributes API
  {
    void* devPtr2 = nullptr;
    HIP_CHECK(hipMalloc(&devPtr2, 1024));
    REQUIRE(devPtr2 != nullptr);

    hipPointerAttribute_t allAttributesData;
    HIP_CHECK(hipPointerGetAttributes(&allAttributesData, devPtr2));

    hipPointerAttribute_t allAttributesDataWithPtr;
    HIP_CHECK(dyn_hipPointerGetAttributes_ptr(&allAttributesDataWithPtr, devPtr2));

    REQUIRE(allAttributesDataWithPtr.type == allAttributesData.type);
    REQUIRE(allAttributesDataWithPtr.device == allAttributesData.device);
    REQUIRE(allAttributesDataWithPtr.devicePointer == allAttributesData.devicePointer);
    REQUIRE(allAttributesDataWithPtr.hostPointer == allAttributesData.hostPointer);
    REQUIRE(allAttributesDataWithPtr.isManaged == allAttributesData.isManaged);
    REQUIRE(allAttributesDataWithPtr.allocationFlags == allAttributesData.allocationFlags);
    HIP_CHECK(hipFree(devPtr2));
  }

  // Validating hipDrvPointerGetAttributes API
  {
    void* devPtr3 = nullptr;
    HIP_CHECK(hipMalloc(&devPtr3, 1024));
    REQUIRE(devPtr3 != nullptr);

    hipPointer_attribute requiredAttributes[] = {
        HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
        HIP_POINTER_ATTRIBUTE_RANGE_SIZE, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR};

    unsigned int memoryType = -1;
    void* devicePointer = nullptr;
    unsigned int rangeSize = -1;
    void* startAddress = nullptr;
    void* requiredData[] = {&memoryType, &devicePointer, &rangeSize, &startAddress};

    HIP_CHECK(hipDrvPointerGetAttributes(4, requiredAttributes, requiredData, devPtr3));

    unsigned int memoryTypeWithPtr = -1;
    void* devicePointerWithPtr = nullptr;
    unsigned int rangeSizeWithPtr = -1;
    void* startAddressWithPtr = nullptr;
    void* requiredDataWithPtr[] = {&memoryTypeWithPtr, &devicePointerWithPtr, &rangeSizeWithPtr,
                                   &startAddressWithPtr};

    HIP_CHECK(
        dyn_hipDrvPointerGetAttributes_ptr(4, requiredAttributes, requiredDataWithPtr, devPtr3));

    REQUIRE(memoryTypeWithPtr == memoryType);
    REQUIRE(devicePointerWithPtr == devicePointer);
    REQUIRE(rangeSizeWithPtr == rangeSize);
    REQUIRE(startAddressWithPtr == startAddress);

    HIP_CHECK(hipFree(devPtr3));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Memory copy) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemCopy") {
  void* hipMemcpyHtoD_ptr = nullptr;
  void* hipMemcpyDtoH_ptr = nullptr;
  void* hipMemcpyDtoD_ptr = nullptr;
  void* hipMemcpy_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemcpyHtoD", &hipMemcpyHtoD_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyDtoH", &hipMemcpyDtoH_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyDtoD", &hipMemcpyDtoD_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy", &hipMemcpy_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemcpyHtoD_ptr)(hipDeviceptr_t, void*, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, void*, size_t)>(hipMemcpyHtoD_ptr);
  hipError_t (*dyn_hipMemcpyDtoH_ptr)(void*, hipDeviceptr_t, size_t) =
      reinterpret_cast<hipError_t (*)(void*, hipDeviceptr_t, size_t)>(hipMemcpyDtoH_ptr);
  hipError_t (*dyn_hipMemcpyDtoD_ptr)(hipDeviceptr_t, hipDeviceptr_t, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, hipDeviceptr_t, size_t)>(hipMemcpyDtoD_ptr);
  hipError_t (*dyn_hipMemcpy_ptr)(void*, const void*, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, const void*, size_t, hipMemcpyKind)>(hipMemcpy_ptr);
  int N = 128;
  int Nbytes = N * sizeof(int);
  int value = 15;

  // Validating hipMemcpyHtoD API
  {
    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    fillHostArray(hostMem, N, value);

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemcpyHtoD_ptr(devMem, hostMem, Nbytes));
    REQUIRE(validateDeviceArray(devMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemcpyDtoH API
  {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    fillDeviceArray(devMem, N, value);

    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(dyn_hipMemcpyDtoH_ptr(hostMem, devMem, Nbytes));
    REQUIRE(validateHostArray(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemcpyDtoD API
  {
    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    fillDeviceArray(devMem, N, value);

    int* dstDevMem = nullptr;
    HIP_CHECK(hipMalloc(&dstDevMem, Nbytes));
    REQUIRE(dstDevMem != nullptr);
    HIP_CHECK(dyn_hipMemcpyDtoD_ptr(dstDevMem, devMem, Nbytes));
    REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    HIP_CHECK(hipFree(dstDevMem));
  }

  // Validating hipMemcpy API
  {
    // With flag hipMemcpyHostToHost
    {
      int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* dstHostMem = reinterpret_cast<int*>(malloc(Nbytes));
      REQUIRE(dstHostMem != nullptr);
      HIP_CHECK(dyn_hipMemcpy_ptr(dstHostMem, hostMem, Nbytes, hipMemcpyHostToHost));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(devMem, hostMem, Nbytes, hipMemcpyHostToDevice));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDeviceToDevice));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDeviceToDeviceNoCU));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(dstHostMem, hostMem, Nbytes, hipMemcpyDefault));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(devMem, hostMem, Nbytes, hipMemcpyDefault));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(hostMem, devMem, Nbytes, hipMemcpyDefault));
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
      HIP_CHECK(dyn_hipMemcpy_ptr(dstDevMem, devMem, Nbytes, hipMemcpyDefault));
      REQUIRE(validateDeviceArray(dstDevMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      HIP_CHECK(hipFree(dstDevMem));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Memory copy with stream) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemCopyWithStreams") {
  void* hipMemcpyHtoDAsync_ptr = nullptr;
  void* hipMemcpyDtoHAsync_ptr = nullptr;
  void* hipMemcpyDtoDAsync_ptr = nullptr;
  void* hipMemcpyAsync_ptr = nullptr;
  void* hipMemcpyWithStream_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemcpyHtoDAsync", &hipMemcpyHtoDAsync_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyDtoHAsync", &hipMemcpyDtoHAsync_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyDtoDAsync", &hipMemcpyDtoDAsync_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemcpyAsync", &hipMemcpyAsync_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyWithStream", &hipMemcpyWithStream_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipMemcpyHtoDAsync_ptr)(hipDeviceptr_t, void*, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, void*, size_t, hipStream_t)>(
          hipMemcpyHtoDAsync_ptr);
  hipError_t (*dyn_hipMemcpyDtoHAsync_ptr)(void*, hipDeviceptr_t, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, hipDeviceptr_t, size_t, hipStream_t)>(
          hipMemcpyDtoHAsync_ptr);
  hipError_t (*dyn_hipMemcpyDtoDAsync_ptr)(hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, hipDeviceptr_t, size_t, hipStream_t)>(
          hipMemcpyDtoDAsync_ptr);
  hipError_t (*dyn_hipMemcpyAsync_ptr)(void*, const void*, size_t, hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpyAsync_ptr);
  hipError_t (*dyn_hipMemcpyWithStream_ptr)(void*, const void*, size_t, hipMemcpyKind,
                                            hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, const void*, size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpyWithStream_ptr);
  int N = 4096;
  const int Ns = 4;
  int Nbytes = N * sizeof(int);
  int value = 2;
  // Validating hipMemcpyHtoDAsync API
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
      HIP_CHECK(dyn_hipMemcpyHtoDAsync_ptr(devMem + startIndex, hostMem + startIndex, (Nbytes / Ns),
                                           stream[s]));
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
  // Validating hipMemcpyDtoHAsync API
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
      HIP_CHECK(dyn_hipMemcpyDtoHAsync_ptr(hostMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                           stream[s]));
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

  // Validating hipMemcpyDtoDAsync API
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
      HIP_CHECK(dyn_hipMemcpyDtoDAsync_ptr(dstDevMem + startIndex, devMem + startIndex,
                                           (Nbytes / Ns), stream[s]));
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

  // Validating hipMemcpyAsync API
  {
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(dstHostMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(devMem + startIndex, hostMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyHostToDevice, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(hostMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDeviceToHost, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(dstDevMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDeviceToDevice, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(dstDevMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDeviceToDeviceNoCU, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(dstHostMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(devMem + startIndex, hostMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDefault, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(hostMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDefault, stream[s]));
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
        HIP_CHECK(dyn_hipMemcpyAsync_ptr(dstDevMem + startIndex, devMem + startIndex, (Nbytes / Ns),
                                         hipMemcpyDefault, stream[s]));
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

  // Validating hipMemcpyWithStream API
  {
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(dstHostMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(devMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(hostMem + startIndex, devMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(dstDevMem + startIndex, devMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(dstDevMem + startIndex, devMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(dstHostMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(devMem + startIndex, hostMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(hostMem + startIndex, devMem + startIndex,
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
        HIP_CHECK(dyn_hipMemcpyWithStream_ptr(dstDevMem + startIndex, devMem + startIndex,
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
 *  - This test will get the function pointer of different Memory management
 *  - (Memset) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemset") {
  void* hipMemsetD8_ptr = nullptr;
  void* hipMemsetD16_ptr = nullptr;
  void* hipMemsetD32_ptr = nullptr;
  void* hipMemsetD8Async_ptr = nullptr;
  void* hipMemsetD16Async_ptr = nullptr;
  void* hipMemsetD32Async_ptr = nullptr;
  void* hipMemset_ptr = nullptr;
  void* hipMemsetAsync_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemsetD8", &hipMemsetD8_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemsetD16", &hipMemsetD16_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemsetD32", &hipMemsetD32_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemsetD8Async", &hipMemsetD8Async_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemsetD16Async", &hipMemsetD16Async_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemsetD32Async", &hipMemsetD32Async_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemset", &hipMemset_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemsetAsync", &hipMemsetAsync_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemsetD8_ptr)(hipDeviceptr_t, unsigned char, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, unsigned char, size_t)>(hipMemsetD8_ptr);
  hipError_t (*dyn_hipMemsetD16_ptr)(hipDeviceptr_t, uint16_t, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, uint16_t, size_t)>(hipMemsetD16_ptr);
  hipError_t (*dyn_hipMemsetD32_ptr)(hipDeviceptr_t, int, size_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, int, size_t)>(hipMemsetD32_ptr);
  hipError_t (*dyn_hipMemsetD8Async_ptr)(hipDeviceptr_t, unsigned char, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, unsigned char, size_t, hipStream_t)>(
          hipMemsetD8Async_ptr);
  hipError_t (*dyn_hipMemsetD16Async_ptr)(hipDeviceptr_t, uint16_t, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, uint16_t, size_t, hipStream_t)>(
          hipMemsetD16Async_ptr);
  hipError_t (*dyn_hipMemsetD32Async_ptr)(hipDeviceptr_t, int, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t, int, size_t, hipStream_t)>(
          hipMemsetD32Async_ptr);
  hipError_t (*dyn_hipMemset_ptr)(void*, int, size_t) =
      reinterpret_cast<hipError_t (*)(void*, int, size_t)>(hipMemset_ptr);
  hipError_t (*dyn_hipMemsetAsync_ptr)(void*, int, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, int, size_t, hipStream_t)>(hipMemsetAsync_ptr);

  // Validating hipMemsetD8 API
  {
    int N = 16;
    int Nbytes = N * sizeof(char);
    unsigned char value = 255;

    void* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemsetD8_ptr(devMem, value, N));

    unsigned char* hostMem = (unsigned char*)malloc(Nbytes);
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<unsigned char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }
  // Validating hipMemsetD16 API
  {
    int N = 16;
    int Nbytes = N * sizeof(uint16_t);
    uint16_t value = 65535;

    void* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemsetD16_ptr(devMem, value, N));

    uint16_t* hostMem = reinterpret_cast<uint16_t*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<uint16_t>(hostMem, N, value) == true);
    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }
  // Validating hipMemsetD32 API
  {
    int N = 16;
    int Nbytes = N * sizeof(int);
    int value = 2147483647;

    void* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemsetD32_ptr(devMem, value, N));

    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<int>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }
  // Validating hipMemsetD8Async API
  {
    int N = 16;
    int Nbytes = N * sizeof(char);
    unsigned char value = 255;
    const int Ns = 4;

    unsigned char* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      HIP_CHECK(dyn_hipMemsetD8Async_ptr(devMem + startIndex, value, N / Ns, stream[s]));
    }
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamSynchronize(stream[s]));
    }

    unsigned char* hostMem = (unsigned char*)malloc(Nbytes);
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<unsigned char>(hostMem, N, value) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemsetD16Async API
  {
    int N = 16;
    int Nbytes = N * sizeof(uint16_t);
    uint16_t value = 65535;
    const int Ns = 4;

    uint16_t* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      HIP_CHECK(dyn_hipMemsetD16Async_ptr(devMem + startIndex, value, N / Ns, stream[s]));
    }
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamSynchronize(stream[s]));
    }

    uint16_t* hostMem = reinterpret_cast<uint16_t*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<uint16_t>(hostMem, N, value) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemsetD32Async API
  {
    int N = 16;
    int Nbytes = N * sizeof(int);
    int value = 2147483647;
    const int Ns = 4;

    int* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      HIP_CHECK(dyn_hipMemsetD32Async_ptr(devMem + startIndex, value, N / Ns, stream[s]));
    }
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamSynchronize(stream[s]));
    }

    int* hostMem = reinterpret_cast<int*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<int>(hostMem, N, value) == true);

    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamDestroy(stream[s]));
    }

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemset API
  {
    int N = 16;
    int Nbytes = N * sizeof(char);
    int value = 10;

    void* devMem = nullptr;
    HIP_CHECK(hipMalloc(&devMem, Nbytes));
    REQUIRE(devMem != nullptr);
    HIP_CHECK(dyn_hipMemset_ptr(devMem, value, Nbytes));

    char* hostMem = reinterpret_cast<char*>(malloc(Nbytes));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy(hostMem, devMem, Nbytes, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemsetAsync API
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
      HIP_CHECK(dyn_hipMemsetAsync_ptr(devMem + startIndex, value, N / Ns, stream[s]));
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
 *  - This test will get the function pointer of different Memory management
 *  - (Memset 2D and 3D) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemset2D3D") {
  CHECK_IMAGE_SUPPORT

  void* hipMemset2D_ptr = nullptr;
  void* hipMemset2DAsync_ptr = nullptr;
  void* hipMemset3D_ptr = nullptr;
  void* hipMemset3DAsync_ptr = nullptr;
  hipDriverProcAddressQueryResult symbolStatus = HIP_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMemset2D", &hipMemset2D_ptr, currentHipVersion, 0, &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(hipGetProcAddress("hipMemset2DAsync", &hipMemset2DAsync_ptr, currentHipVersion, 0,
                              &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(
      hipGetProcAddress("hipMemset3D", &hipMemset3D_ptr, currentHipVersion, 0, &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);
  HIP_CHECK(hipGetProcAddress("hipMemset3DAsync", &hipMemset3DAsync_ptr, currentHipVersion, 0,
                              &symbolStatus));
  REQUIRE(symbolStatus == HIP_GET_PROC_ADDRESS_SUCCESS);

  hipError_t (*dyn_hipMemset2D_ptr)(void*, size_t, int, size_t, size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, int, size_t, size_t)>(hipMemset2D_ptr);
  hipError_t (*dyn_hipMemset2DAsync_ptr)(void*, size_t, int, size_t, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, int, size_t, size_t, hipStream_t)>(
          hipMemset2DAsync_ptr);
  hipError_t (*dyn_hipMemset3D_ptr)(hipPitchedPtr, int, hipExtent) =
      reinterpret_cast<hipError_t (*)(hipPitchedPtr, int, hipExtent)>(hipMemset3D_ptr);
  hipError_t (*dyn_hipMemset3DAsync_ptr)(hipPitchedPtr, int, hipExtent, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipPitchedPtr, int, hipExtent, hipStream_t)>(
          hipMemset3DAsync_ptr);
  size_t width = 1024;
  size_t height = 1024;
  size_t depth = 1024;
  int value = 10;
  const int Ns = 4;
  size_t pitch;

  // Validating hipMemset2D API
  {
    const int N = width * height;
    char* devMem = nullptr;
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
    REQUIRE(devMem != nullptr);

    HIP_CHECK(dyn_hipMemset2D_ptr(devMem, pitch, value, width, height));

    char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
    REQUIRE(hostMem != nullptr);
    HIP_CHECK(hipMemcpy2D(hostMem, width, devMem, pitch, width, height, hipMemcpyDeviceToHost));

    REQUIRE(validateArrayT<char>(hostMem, N, value) == true);

    HIP_CHECK(hipFree(devMem));
    free(hostMem);
  }

  // Validating hipMemset2DAsync API
  {
    const int N = width * height;
    char* devMem = nullptr;
    HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
    REQUIRE(devMem != nullptr);

    // set the whole matrix first to something different than 'value'
    HIP_CHECK(dyn_hipMemset2DAsync_ptr(devMem, pitch, 5, width, height, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    hipStream_t stream[Ns];
    for (int s = 0; s < Ns; s++) {
      HIP_CHECK(hipStreamCreate(&stream[s]));
    }

    for (int s = 0; s < Ns; s++) {
      int startIndex = s * (N / Ns);
      int row = startIndex / width;
      HIP_CHECK(dyn_hipMemset2DAsync_ptr(devMem + row * pitch, pitch, value, width, height / Ns,
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

  // Validating hipMemset3D API
  {
    const int N = width * height * depth;

    hipPitchedPtr devMem;
    hipExtent extent3d{width, height, depth};
    HIP_CHECK(hipMalloc3D(&devMem, extent3d));
    REQUIRE(devMem.ptr != nullptr);

    HIP_CHECK(dyn_hipMemset3D_ptr(devMem, value, extent3d));

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

  // Validating hipMemset3DAsync API
  {
    size_t width = 64;
    size_t height = 64;
    size_t depth = 64;
    const int N = width * height * depth;
    int value = 10;

    hipPitchedPtr devMem;
    hipExtent extent3d{width, height, depth};
    HIP_CHECK(hipMalloc3D(&devMem, extent3d));
    REQUIRE(devMem.ptr != nullptr);

    HIP_CHECK(dyn_hipMemset3DAsync_ptr(devMem, value, extent3d, NULL));

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
 *  - This test will get the function pointer of different Memory management
 *  - (Memory Info) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisGetMemInfoRelated") {
  void* hipMemGetInfo_ptr = nullptr;
  void* hipMemPtrGetInfo_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemGetInfo", &hipMemGetInfo_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemPtrGetInfo", &hipMemPtrGetInfo_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemGetInfo_ptr)(size_t*, size_t*) =
      reinterpret_cast<hipError_t (*)(size_t*, size_t*)>(hipMemGetInfo_ptr);
  hipError_t (*dyn_hipMemPtrGetInfo_ptr)(void*, size_t*) =
      reinterpret_cast<hipError_t (*)(void*, size_t*)>(hipMemPtrGetInfo_ptr);

  // Validating hipMemGetInfo API
  size_t freeMem = 0, totalMem = 0, freeMemWithPtr = 0, totalMemWithPtr = 0;

  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  HIP_CHECK(dyn_hipMemGetInfo_ptr(&freeMemWithPtr, &totalMemWithPtr));

  REQUIRE(freeMemWithPtr == freeMem);
  REQUIRE(totalMemWithPtr == totalMem);

  // Validating hipMemPtrGetInfo API
  void* devPtr = nullptr;
  HIP_CHECK(hipMalloc(&devPtr, 128));
  REQUIRE(devPtr != nullptr);

  size_t devMemsize = -1;
  HIP_CHECK(dyn_hipMemPtrGetInfo_ptr(devPtr, &devMemsize));
  REQUIRE(devMemsize == 128);

  HIP_CHECK(hipFree(devPtr));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Memory copy 2D) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemcpy2DRelated") {
  CHECK_IMAGE_SUPPORT

  void* hipMemcpy2D_ptr = nullptr;
  void* hipMemcpy2DAsync_ptr = nullptr;
  void* hipMemcpyParam2D_ptr = nullptr;
  void* hipMemcpyParam2DAsync_ptr = nullptr;
  void* hipMemcpy2DToArray_ptr = nullptr;
  void* hipMemcpy2DToArrayAsync_ptr = nullptr;
  void* hipMemcpy2DFromArray_ptr = nullptr;
  void* hipMemcpy2DFromArrayAsync_ptr = nullptr;
  void* hipMemcpyToArray_ptr = nullptr;
  void* hipMemcpyFromArray_ptr = nullptr;
  void* hipMemcpyAtoH_ptr = nullptr;
  void* hipMemcpyHtoA_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemcpy2D", &hipMemcpy2D_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemcpy2DAsync", &hipMemcpy2DAsync_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemcpyParam2D", &hipMemcpyParam2D_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyParam2DAsync", &hipMemcpyParam2DAsync_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DToArray", &hipMemcpy2DToArray_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DToArrayAsync", &hipMemcpy2DToArrayAsync_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DFromArray", &hipMemcpy2DFromArray_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpy2DFromArrayAsync", &hipMemcpy2DFromArrayAsync_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemcpyToArray", &hipMemcpyToArray_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyFromArray", &hipMemcpyFromArray_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyAtoH", &hipMemcpyAtoH_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyHtoA", &hipMemcpyHtoA_ptr, currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMemcpy2D_ptr)(void*, size_t, const void*, size_t, size_t, size_t,
                                    hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, size_t, const void*, size_t, size_t, size_t,
                                      hipMemcpyKind)>(hipMemcpy2D_ptr);
  hipError_t (*dyn_hipMemcpy2DAsync_ptr)(void*, size_t, const void*, size_t, size_t, size_t,
                                         hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, const void*, size_t, size_t, size_t,
                                      hipMemcpyKind, hipStream_t)>(hipMemcpy2DAsync_ptr);
  hipError_t (*dyn_hipMemcpyParam2D_ptr)(const hip_Memcpy2D*) =
      reinterpret_cast<hipError_t (*)(const hip_Memcpy2D*)>(hipMemcpyParam2D_ptr);
  hipError_t (*dyn_hipMemcpyParam2DAsync_ptr)(const hip_Memcpy2D*, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const hip_Memcpy2D*, hipStream_t)>(hipMemcpyParam2DAsync_ptr);
  hipError_t (*dyn_hipMemcpy2DToArray_ptr)(hipArray_t, size_t, size_t, const void* src, size_t,
                                           size_t, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, size_t, const void* src, size_t, size_t,
                                      size_t, hipMemcpyKind)>(hipMemcpy2DToArray_ptr);
  hipError_t (*dyn_hipMemcpy2DToArrayAsync_ptr)(hipArray_t, size_t, size_t, const void* src, size_t,
                                                size_t, size_t, hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, size_t, const void* src, size_t, size_t,
                                      size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpy2DToArrayAsync_ptr);
  hipError_t (*dyn_hipMemcpy2DFromArray_ptr)(void*, size_t, hipArray_const_t, size_t, size_t,
                                             size_t, size_t, hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, size_t, hipArray_const_t, size_t, size_t, size_t,
                                      size_t, hipMemcpyKind)>(hipMemcpy2DFromArray_ptr);
  hipError_t (*dyn_hipMemcpy2DFromArrayAsync_ptr)(void*, size_t, hipArray_const_t, size_t, size_t,
                                                  size_t, size_t, hipMemcpyKind, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, hipArray_const_t, size_t, size_t, size_t,
                                      size_t, hipMemcpyKind, hipStream_t)>(
          hipMemcpy2DFromArrayAsync_ptr);
  hipError_t (*dyn_hipMemcpyToArray_ptr)(hipArray_t, size_t, size_t, const void*, size_t,
                                         hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, size_t, const void*, size_t,
                                      hipMemcpyKind)>(hipMemcpyToArray_ptr);
  hipError_t (*dyn_hipMemcpyFromArray_ptr)(void*, hipArray_const_t, size_t, size_t, size_t,
                                           hipMemcpyKind) =
      reinterpret_cast<hipError_t (*)(void*, hipArray_const_t, size_t, size_t, size_t,
                                      hipMemcpyKind)>(hipMemcpyFromArray_ptr);
  hipError_t (*dyn_hipMemcpyAtoH_ptr)(void*, hipArray_t, size_t, size_t) =
      reinterpret_cast<hipError_t (*)(void*, hipArray_t, size_t, size_t)>(hipMemcpyAtoH_ptr);
  hipError_t (*dyn_hipMemcpyHtoA_ptr)(hipArray_t, size_t, const void*, size_t) =
      reinterpret_cast<hipError_t (*)(hipArray_t, size_t, const void*, size_t)>(hipMemcpyHtoA_ptr);

  // Validating hipMemcpy2D API
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

      HIP_CHECK(dyn_hipMemcpy2D_ptr(dHostMem, width, sHostMem, width, width, height,
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

      HIP_CHECK(
          dyn_hipMemcpy2D_ptr(devMem, pitch, hostMem, pitch, width, height, hipMemcpyHostToDevice));

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

      HIP_CHECK(
          dyn_hipMemcpy2D_ptr(hostMem, width, devMem, pitch, width, height, hipMemcpyDeviceToHost));

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

      HIP_CHECK(dyn_hipMemcpy2D_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2D_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
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

      HIP_CHECK(
          dyn_hipMemcpy2D_ptr(dHostMem, width, sHostMem, width, width, height, hipMemcpyDefault));

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
          dyn_hipMemcpy2D_ptr(devMem, pitch, hostMem, pitch, width, height, hipMemcpyDefault));

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
          dyn_hipMemcpy2D_ptr(hostMem, width, devMem, pitch, width, height, hipMemcpyDefault));

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

      HIP_CHECK(
          dyn_hipMemcpy2D_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height, hipMemcpyDefault));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipMemcpy2DAsync API
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(dHostMem, width, sHostMem, width, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(devMem, pitch, hostMem, pitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(hostMem, width, devMem, pitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(dHostMem, width, sHostMem, width, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(devMem, pitch, hostMem, pitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(hostMem, width, devMem, pitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DAsync_ptr(dDevMem, dPitch, sDevMem, sPitch, width, height,
                                         hipMemcpyDefault, NULL));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipMemcpyParam2D API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;
    size_t pitch;

    // Host to Host
    {
      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = sHostMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = dHostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2D_ptr(&desc));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }
    // Host to Device
    {
      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = hostMem;
      desc.srcPitch = width;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(devMem);
      desc.dstPitch = pitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2D_ptr(&desc));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Host
    {
      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)devMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = hostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2D_ptr(&desc));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // Device To Device - single GPU
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

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = hipDeviceptr_t(sDevMem);
      desc.srcPitch = sPitch;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(dDevMem);
      desc.dstPitch = dPitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2D_ptr(&desc));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // Device To Device - Two GPU's
    {
      int deviceCount = 0;
      HIP_CHECK(hipGetDeviceCount(&deviceCount));

      if (deviceCount > 1) {
        HIP_CHECK(hipSetDevice(0));
        int can_access_peer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
        if (can_access_peer) {
          char* sDevMem = nullptr;
          size_t sPitch;
          HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
          REQUIRE(sDevMem != nullptr);
          HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

          HIP_CHECK(hipSetDevice(1));

          char* dDevMem = nullptr;
          size_t dPitch;
          HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
          REQUIRE(dDevMem != nullptr);
          hip_Memcpy2D desc = {};
          desc.srcMemoryType = hipMemoryTypeDevice;
          desc.srcDevice = hipDeviceptr_t(sDevMem);
          desc.srcPitch = sPitch;
          desc.dstMemoryType = hipMemoryTypeDevice;
          desc.dstDevice = hipDeviceptr_t(dDevMem);
          desc.dstPitch = dPitch;
          desc.WidthInBytes = width * sizeof(char);
          desc.Height = height;
          HIP_CHECK(dyn_hipMemcpyParam2D_ptr(&desc));

          REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

          HIP_CHECK(hipFree(sDevMem));
          HIP_CHECK(hipFree(dDevMem));
        }
      }
    }
  }

  // Validating hipMemcpyParam2DAsync API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;
    size_t pitch;

    // Host to Host
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(sHostMem != nullptr);
      fillCharHostArray(sHostMem, N, value);

      char* dHostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(dHostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = sHostMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = dHostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2DAsync_ptr(&desc, stream));

      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(dHostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(sHostMem);
      free(dHostMem);
    }

    // Host to Device
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);
      fillCharHostArray(hostMem, N, value);

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = hostMem;
      desc.srcPitch = width;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(devMem);
      desc.dstPitch = pitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(devMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Host
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* devMem = nullptr;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMem), &pitch, width, height));
      REQUIRE(devMem != nullptr);
      HIP_CHECK(hipMemset2D(devMem, pitch, value, width, height));

      char* hostMem = reinterpret_cast<char*>(malloc(N * sizeof(char)));
      REQUIRE(hostMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)devMem;
      desc.srcPitch = width;

      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = hostMem;
      desc.dstPitch = width;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(devMem));
      free(hostMem);
    }

    // Device To Device - single GPU
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      char* sDevMem = nullptr;
      size_t sPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
      REQUIRE(sDevMem != nullptr);
      HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

      char* dDevMem = nullptr;
      size_t dPitch;
      HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
      REQUIRE(dDevMem != nullptr);

      hip_Memcpy2D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = hipDeviceptr_t(sDevMem);
      desc.srcPitch = sPitch;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = hipDeviceptr_t(dDevMem);
      desc.dstPitch = dPitch;
      desc.WidthInBytes = width * sizeof(char);
      desc.Height = height;

      HIP_CHECK(dyn_hipMemcpyParam2DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }

    // Device To Device - Two GPU's
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      int deviceCount = 0;
      HIP_CHECK(hipGetDeviceCount(&deviceCount));

      if (deviceCount > 1) {
        HIP_CHECK(hipSetDevice(0));
        int can_access_peer = 0;
        HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, 0, 1));
        if (can_access_peer) {
          char* sDevMem = nullptr;
          size_t sPitch;
          HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&sDevMem), &sPitch, width, height));
          REQUIRE(sDevMem != nullptr);
          HIP_CHECK(hipMemset2D(sDevMem, sPitch, value, width, height));

          HIP_CHECK(hipSetDevice(1));

          char* dDevMem = nullptr;
          size_t dPitch;
          HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dDevMem), &dPitch, width, height));
          REQUIRE(dDevMem != nullptr);
          hip_Memcpy2D desc = {};
          desc.srcMemoryType = hipMemoryTypeDevice;
          desc.srcDevice = hipDeviceptr_t(sDevMem);
          desc.srcPitch = sPitch;
          desc.dstMemoryType = hipMemoryTypeDevice;
          desc.dstDevice = hipDeviceptr_t(dDevMem);
          desc.dstPitch = dPitch;
          desc.WidthInBytes = width * sizeof(char);
          desc.Height = height;
          HIP_CHECK(dyn_hipMemcpyParam2DAsync_ptr(&desc, stream));

          HIP_CHECK(hipStreamSynchronize(stream));

          REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

          HIP_CHECK(hipStreamDestroy(stream));
          HIP_CHECK(hipFree(sDevMem));
          HIP_CHECK(hipFree(dDevMem));
        }
      }
    }
  }

  // Validating hipMemcpy2DToArray API
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

      HIP_CHECK(dyn_hipMemcpy2DToArray_ptr(array, 0, 0, hostMem, width, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DToArray_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DToArray_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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

      HIP_CHECK(
          dyn_hipMemcpy2DToArray_ptr(array, 0, 0, hostMem, width, width, height, hipMemcpyDefault));

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

      HIP_CHECK(dyn_hipMemcpy2DToArray_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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

  // Validating hipMemcpy2DToArrayAsync API
  {
    size_t width = 256;
    size_t height = 256;
    const int N = width * height;
    int value = 10;

    // With flags hipMemcpyHostToDevice
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

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_ptr(array, 0, 0, hostMem, width, width, height,
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
    // With flags hipMemcpyDeviceToDevice
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

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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
    // With flags hipMemcpyDeviceToDeviceNoCU
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

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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
    // With flags hipMemcpyDefault - Host To Device
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

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_ptr(array, 0, 0, hostMem, width, width, height,
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
    // With flags hipMemcpyDefault - Device To Device
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

      HIP_CHECK(dyn_hipMemcpy2DToArrayAsync_ptr(array, 0, 0, sDevMem, sPitch, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArray_ptr(hostMemory, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArray_ptr(dDevMem, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArray_ptr(dDevMem, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArray_ptr(hostMemory, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArray_ptr(dDevMem, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_ptr(hostMemory, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_ptr(dDevMem, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_ptr(dDevMem, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_ptr(hostMemory, width, array, 0, 0, width, height,
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

      HIP_CHECK(dyn_hipMemcpy2DFromArrayAsync_ptr(dDevMem, width, array, 0, 0, width, height,
                                                  hipMemcpyDefault, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpyToArray API
  {
    size_t width = 64;
    size_t height = 1;
    const int N = width * height;
    int value = 10;
    // With flag hipMemcpyHostToDevice
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(
          dyn_hipMemcpyToArray_ptr(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          hipMemcpyFromArray(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDevice
    {
      int* sDevMem = nullptr;
      HIP_CHECK(hipMalloc(&sDevMem, N * sizeof(int)));
      REQUIRE(sDevMem != nullptr);
      fillDeviceArray(sDevMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(
          dyn_hipMemcpyToArray_ptr(array, 0, 0, sDevMem, N * sizeof(int), hipMemcpyDeviceToDevice));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          hipMemcpyFromArray(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      int* sDevMem = nullptr;
      HIP_CHECK(hipMalloc(&sDevMem, N * sizeof(int)));
      REQUIRE(sDevMem != nullptr);
      fillDeviceArray(sDevMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpyToArray_ptr(array, 0, 0, sDevMem, N * sizeof(int),
                                         hipMemcpyDeviceToDeviceNoCU));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          hipMemcpyFromArray(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Host To Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpyToArray_ptr(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyDefault));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          hipMemcpyFromArray(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
    // With flag hipMemcpyDefault - Device To Device
    {
      int* sDevMem = nullptr;
      HIP_CHECK(hipMalloc(&sDevMem, N * sizeof(int)));
      REQUIRE(sDevMem != nullptr);
      fillDeviceArray(sDevMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(dyn_hipMemcpyToArray_ptr(array, 0, 0, sDevMem, N * sizeof(int), hipMemcpyDefault));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          hipMemcpyFromArray(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpyFromArray API
  {
    size_t width = 64;
    size_t height = 1;
    const int N = width * height;
    int value = 10;

    // With flag hipMemcpyDeviceToHost
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpyFromArray_ptr(hostMemory, array, 0, 0, N * sizeof(int),
                                           hipMemcpyDeviceToHost));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }

    // With flag hipMemcpyDeviceToDevice
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* deviceMemory = nullptr;
      HIP_CHECK(hipMalloc(&deviceMemory, N * sizeof(int)));
      REQUIRE(deviceMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpyFromArray_ptr(deviceMemory, array, 0, 0, N * sizeof(int),
                                           hipMemcpyDeviceToDevice));
      REQUIRE(validateDeviceArray(deviceMemory, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(deviceMemory));
      HIP_CHECK(hipFreeArray(array));
    }

    // With flag hipMemcpyDeviceToDeviceNoCU
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* deviceMemory = nullptr;
      HIP_CHECK(hipMalloc(&deviceMemory, N * sizeof(int)));
      REQUIRE(deviceMemory != nullptr);

      HIP_CHECK(dyn_hipMemcpyFromArray_ptr(deviceMemory, array, 0, 0, N * sizeof(int),
                                           hipMemcpyDeviceToDeviceNoCU));
      REQUIRE(validateDeviceArray(deviceMemory, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(deviceMemory));
      HIP_CHECK(hipFreeArray(array));
    }

    // With flag hipMemcpyDefault - Device To Host
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMemory != nullptr);

      HIP_CHECK(
          dyn_hipMemcpyFromArray_ptr(hostMemory, array, 0, 0, N * sizeof(int), hipMemcpyDefault));
      REQUIRE(validateHostArray(hostMemory, N, value) == true);

      free(hostMem);
      free(hostMemory);
      HIP_CHECK(hipFreeArray(array));
    }

    // With flag hipMemcpyDefault - Device To Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      hipArray_t array = nullptr;
      hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
      unsigned int flags = hipArrayDefault;
      HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
      REQUIRE(array != nullptr);

      HIP_CHECK(hipMemcpyToArray(array, 0, 0, hostMem, N * sizeof(int), hipMemcpyHostToDevice));

      int* deviceMemory = nullptr;
      HIP_CHECK(hipMalloc(&deviceMemory, N * sizeof(int)));
      REQUIRE(deviceMemory != nullptr);

      HIP_CHECK(
          dyn_hipMemcpyFromArray_ptr(deviceMemory, array, 0, 0, N * sizeof(int), hipMemcpyDefault));
      REQUIRE(validateDeviceArray(deviceMemory, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(deviceMemory));
      HIP_CHECK(hipFreeArray(array));
    }
  }

  // Validating hipMemcpyAtoH & hipMemcpyHtoA API's
  {
    size_t width = 256;
    size_t height = 1;
    const int N = width * height;
    hipArray_t array = nullptr;
    hipChannelFormatDesc desc = hipCreateChannelDesc<int>();
    unsigned int flags = hipArrayDefault;
    HIP_CHECK(hipMallocArray(&array, &desc, width, height, flags));
    REQUIRE(array != nullptr);
    int value = 10;
    int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
    REQUIRE(hostMem != nullptr);
    fillHostArray(hostMem, N, value);
    HIP_CHECK(dyn_hipMemcpyHtoA_ptr(array, 0, hostMem, N * sizeof(int)));

    int* hostMemory = reinterpret_cast<int*>(malloc(N * sizeof(int)));
    REQUIRE(hostMemory != nullptr);
    HIP_CHECK(dyn_hipMemcpyAtoH_ptr(hostMemory, array, 0, N * sizeof(int)));
    REQUIRE(validateHostArray(hostMemory, N, value) == true);

    free(hostMem);
    free(hostMemory);
    HIP_CHECK(hipFreeArray(array));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (Memory copy 3D) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisMemcpy3DRelated") {
  CHECK_IMAGE_SUPPORT

  void* hipMemcpy3D_ptr = nullptr;
  void* hipMemcpy3DAsync_ptr = nullptr;
  void* hipDrvMemcpy3D_ptr = nullptr;
  void* hipDrvMemcpy3DAsync_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemcpy3D", &hipMemcpy3D_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemcpy3DAsync", &hipMemcpy3DAsync_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipDrvMemcpy3D", &hipDrvMemcpy3D_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDrvMemcpy3DAsync", &hipDrvMemcpy3DAsync_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipMemcpy3D_ptr)(const struct hipMemcpy3DParms*) =
      reinterpret_cast<hipError_t (*)(const struct hipMemcpy3DParms*)>(hipMemcpy3D_ptr);
  hipError_t (*dyn_hipMemcpy3DAsync_ptr)(const struct hipMemcpy3DParms*, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const struct hipMemcpy3DParms*, hipStream_t)>(
          hipMemcpy3DAsync_ptr);
  hipError_t (*dyn_hipDrvMemcpy3D_ptr)(const HIP_MEMCPY3D*) =
      reinterpret_cast<hipError_t (*)(const HIP_MEMCPY3D*)>(hipDrvMemcpy3D_ptr);
  hipError_t (*dyn_hipDrvMemcpy3DAsync_ptr)(const HIP_MEMCPY3D*, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const HIP_MEMCPY3D*, hipStream_t)>(hipDrvMemcpy3DAsync_ptr);

  // Validating hipMemcpy3D API
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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

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
      HIP_CHECK(dyn_hipMemcpy3D_ptr(&myparms));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }
  }

  // Validating hipMemcpy3DAsync API
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
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

      HIP_CHECK(dyn_hipMemcpy3DAsync_ptr(&myparms, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateCharDeviceArray(reinterpret_cast<char*>(dDevMem.ptr), N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem.ptr));
      HIP_CHECK(hipFree(dDevMem.ptr));
    }
  }

  // Validating hipDrvMemcpy3D API
  {
    size_t width = 16;
    size_t height = 16;
    size_t depth = 16;
    const int N = width * height * depth;
    int value = 10;

    // Host to Host
    {
      int* sHostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(sHostMem != nullptr);
      fillHostArray(sHostMem, N, value);

      int* dHostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(dHostMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = sHostMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = dHostMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      HIP_CHECK(dyn_hipDrvMemcpy3D_ptr(&desc));

      REQUIRE(validateHostArray(dHostMem, N, value) == true);

      free(sHostMem);
      free(dHostMem);
    }

    // Host to Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
      REQUIRE(devMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = hostMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = (hipDeviceptr_t)devMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      HIP_CHECK(dyn_hipDrvMemcpy3D_ptr(&desc));

      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Host
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)devMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = hostMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      HIP_CHECK(dyn_hipDrvMemcpy3D_ptr(&desc));

      REQUIRE(validateHostArray(hostMem, N, value) == true);

      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Device
    {
      int* sDevMem = nullptr;
      HIP_CHECK(hipMalloc(&sDevMem, N * sizeof(int)));
      REQUIRE(sDevMem != nullptr);
      fillDeviceArray(sDevMem, N, value);

      int* dDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dDevMem, N * sizeof(int)));
      REQUIRE(dDevMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)sDevMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = (hipDeviceptr_t)dDevMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      HIP_CHECK(dyn_hipDrvMemcpy3D_ptr(&desc));

      REQUIRE(validateDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }

  // Validating hipDrvMemcpy3DAsync API
  {
    size_t width = 16;
    size_t height = 16;
    size_t depth = 16;
    const int N = width * height * depth;
    int value = 10;

    // Host to Host
    {
      int* sHostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(sHostMem != nullptr);
      fillHostArray(sHostMem, N, value);

      int* dHostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(dHostMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = sHostMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = dHostMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipDrvMemcpy3DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateHostArray(dHostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(sHostMem);
      free(dHostMem);
    }

    // Host to Device
    {
      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);
      fillHostArray(hostMem, N, value);

      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
      REQUIRE(devMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeHost;
      desc.srcHost = hostMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = (hipDeviceptr_t)devMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipDrvMemcpy3DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateDeviceArray(devMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Host
    {
      int* devMem = nullptr;
      HIP_CHECK(hipMalloc(&devMem, N * sizeof(int)));
      REQUIRE(devMem != nullptr);
      fillDeviceArray(devMem, N, value);

      int* hostMem = reinterpret_cast<int*>(malloc(N * sizeof(int)));
      REQUIRE(hostMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)devMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeHost;
      desc.dstHost = hostMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipDrvMemcpy3DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateHostArray(hostMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      free(hostMem);
      HIP_CHECK(hipFree(devMem));
    }

    // Device To Device
    {
      int* sDevMem = nullptr;
      HIP_CHECK(hipMalloc(&sDevMem, N * sizeof(int)));
      REQUIRE(sDevMem != nullptr);
      fillDeviceArray(sDevMem, N, value);

      int* dDevMem = nullptr;
      HIP_CHECK(hipMalloc(&dDevMem, N * sizeof(int)));
      REQUIRE(dDevMem != nullptr);

      HIP_MEMCPY3D desc = {};
      desc.srcMemoryType = hipMemoryTypeDevice;
      desc.srcDevice = (hipDeviceptr_t)sDevMem;
      desc.srcPitch = width * sizeof(int);
      desc.srcHeight = height;
      desc.dstMemoryType = hipMemoryTypeDevice;
      desc.dstDevice = (hipDeviceptr_t)dDevMem;
      desc.dstPitch = width * sizeof(int);
      desc.dstHeight = height;
      desc.WidthInBytes = width * sizeof(int);
      desc.Height = height;
      desc.Depth = depth;

      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipDrvMemcpy3DAsync_ptr(&desc, stream));
      HIP_CHECK(hipStreamSynchronize(stream));

      REQUIRE(validateDeviceArray(dDevMem, N, value) == true);

      HIP_CHECK(hipStreamDestroy(stream));
      HIP_CHECK(hipFree(sDevMem));
      HIP_CHECK(hipFree(dDevMem));
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Memory management
 *  - (API address) related APIs from the hipGetProcAddress API
 *  - and then validates the basic functionality of that particular API
 *  - using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisAddressRelated") {
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  void* hipGetProcAddress_ptr = nullptr;
  HIP_CHECK(hipGetProcAddress("hipGetProcAddress", &hipGetProcAddress_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipGetProcAddress_ptr)(const char*, void**, int, uint64_t,
                                          hipDriverProcAddressQueryResult*) =
      reinterpret_cast<hipError_t (*)(const char*, void**, int, uint64_t,
                                      hipDriverProcAddressQueryResult*)>(hipGetProcAddress_ptr);

  // Validating hipGetProcAddress API
  {
    void* hipMallocPtrWithFunction = nullptr;
    void* hipMallocPtrWithFunctionPtr = nullptr;

    HIP_CHECK(
        hipGetProcAddress("hipMalloc", &hipMallocPtrWithFunction, currentHipVersion, 0, nullptr));
    HIP_CHECK(dyn_hipGetProcAddress_ptr("hipMalloc", &hipMallocPtrWithFunctionPtr,
                                        currentHipVersion, 0, nullptr));

    REQUIRE(hipMallocPtrWithFunction != nullptr);
    REQUIRE(hipMallocPtrWithFunctionPtr != nullptr);
    REQUIRE(hipMallocPtrWithFunctionPtr == hipMallocPtrWithFunction);
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Managed Memory
 *  - APIs from the hipGetProcAddress API and then validates the basic
 *  - functionality of that particular API using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisManagedMemory") {
  if (HmmAttrPrint() != 1) {
    HipTest::HIP_SKIP_TEST("Skipping test since managed memory not supported");
    return;
  }

  void* hipMallocManaged_ptr = nullptr;
  void* hipMemPrefetchAsync_ptr = nullptr;
  void* hipMemAdvise_ptr = nullptr;
  void* hipMemRangeGetAttribute_ptr = nullptr;
  void* hipMemRangeGetAttributes_ptr = nullptr;
  void* hipStreamAttachMemAsync_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMallocManaged", &hipMallocManaged_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPrefetchAsync", &hipMemPrefetchAsync_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemAdvise", &hipMemAdvise_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemRangeGetAttribute", &hipMemRangeGetAttribute_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemRangeGetAttributes", &hipMemRangeGetAttributes_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipStreamAttachMemAsync", &hipStreamAttachMemAsync_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMallocManaged_ptr)(void**, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(void**, size_t, unsigned int)>(hipMallocManaged_ptr);
  hipError_t (*dyn_hipMemPrefetchAsync_ptr)(const void*, size_t, int, hipStream_t) =
      reinterpret_cast<hipError_t (*)(const void*, size_t, int, hipStream_t)>(
          hipMemPrefetchAsync_ptr);
  hipError_t (*dyn_hipMemAdvise_ptr)(const void*, size_t, hipMemoryAdvise, int) =
      reinterpret_cast<hipError_t (*)(const void*, size_t, hipMemoryAdvise, int)>(hipMemAdvise_ptr);
  hipError_t (*dyn_hipMemRangeGetAttribute_ptr)(void*, size_t, hipMemRangeAttribute, const void*,
                                                size_t) =
      reinterpret_cast<hipError_t (*)(void*, size_t, hipMemRangeAttribute, const void*, size_t)>(
          hipMemRangeGetAttribute_ptr);
  hipError_t (*dyn_hipMemRangeGetAttributes_ptr)(void**, size_t*, hipMemRangeAttribute*, size_t,
                                                 const void*, size_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t*, hipMemRangeAttribute*, size_t, const void*,
                                      size_t)>(hipMemRangeGetAttributes_ptr);
  hipError_t (*dyn_hipStreamAttachMemAsync_ptr)(hipStream_t, void*, size_t, unsigned int) =
      reinterpret_cast<hipError_t (*)(hipStream_t, void*, size_t, unsigned int)>(
          hipStreamAttachMemAsync_ptr);

  const int N = 16;
  const int Nbytes = N * sizeof(int);
  int value = 10;

  // Validating hipMallocManaged API
  {
    int* memPtr = nullptr;
    size_t size;
    unsigned int flags[] = {hipMemAttachGlobal, hipMemAttachHost};

    for (unsigned int flag : flags) {
      memPtr = nullptr;
      size = -1;

      HIP_CHECK(dyn_hipMallocManaged_ptr(reinterpret_cast<void**>(&memPtr), Nbytes, flag));
      REQUIRE(memPtr != nullptr);

      HIP_CHECK(hipMemPtrGetInfo(memPtr, &size));
      REQUIRE(size == Nbytes);

      fillDeviceArray(memPtr, N, value);
      validateHostArray(memPtr, N, value);

      fillHostArray(memPtr, N, value + 1);
      validateHostArray(memPtr, N, value + 1);

      HIP_CHECK(hipFree(memPtr));
    }
  }

  // Validating hipMemPrefetchAsync API
  {
    hipDevice_t device = hipCpuDeviceId;

    HIP_CHECK(hipSetDevice(0));

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    int* memPtr = nullptr;
    HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
    REQUIRE(memPtr != nullptr);

    fillDeviceArray(memPtr, N, value);

    HIP_CHECK(dyn_hipMemPrefetchAsync_ptr(memPtr, Nbytes, device, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    fillHostArray(memPtr, N, value + 1);
    validateHostArray(memPtr, N, value + 1);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(memPtr));
  }

  // Validating hipMemAdvise and hipMemRangeGetAttribute APIs
  {
    HIP_CHECK(hipSetDevice(0));

    int attrData;
    int* memPtr = nullptr;
    HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
    REQUIRE(memPtr != nullptr);

    // With flag hipMemAdviseSetReadMostly
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetReadMostly, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeReadMostly, memPtr, Nbytes));
    REQUIRE(attrData == 1);

    // With flag hipMemAdviseUnsetReadMostly
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseUnsetReadMostly, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeReadMostly, memPtr, Nbytes));
    REQUIRE(attrData == 0);

    // With flag hipMemAdviseSetPreferredLocation
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetPreferredLocation, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(
        &attrData, sizeof(int), hipMemRangeAttributePreferredLocation, memPtr, Nbytes));
    REQUIRE(attrData == 0);

    // With flag hipMemAdviseUnsetPreferredLocation
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseUnsetPreferredLocation, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(
        &attrData, sizeof(int), hipMemRangeAttributePreferredLocation, memPtr, Nbytes));
    REQUIRE(attrData != 0);

    // With flag hipMemAdviseSetAccessedBy
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetAccessedBy, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeAccessedBy, memPtr, Nbytes));
    REQUIRE(attrData == 0);

    // With flag hipMemAdviseUnsetAccessedBy
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseUnsetAccessedBy, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeAccessedBy, memPtr, Nbytes));
    REQUIRE(attrData != 0);

    // With flag hipMemAdviseSetCoarseGrain
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetCoarseGrain, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeCoherencyMode, memPtr, Nbytes));
    REQUIRE(attrData == hipMemRangeCoherencyModeCoarseGrain);

    // With flag hipMemAdviseUnsetCoarseGrain
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseUnsetCoarseGrain, 0));
    attrData = -2;
    HIP_CHECK(dyn_hipMemRangeGetAttribute_ptr(&attrData, sizeof(int),
                                              hipMemRangeAttributeCoherencyMode, memPtr, Nbytes));
    REQUIRE(attrData == hipMemRangeCoherencyModeFineGrain);

    // With flag hipMemRangeAttributeLastPrefetchLocation
    // Prefetch the location and get the prefetched location
    hipDevice_t device = hipCpuDeviceId;
    HIP_CHECK(hipMemPrefetchAsync(memPtr, Nbytes, device, NULL));

    attrData = -2;
    HIP_CHECK(hipMemRangeGetAttribute(&attrData, sizeof(int),
                                      hipMemRangeAttributeLastPrefetchLocation, memPtr, Nbytes));
    REQUIRE(attrData == device);

    HIP_CHECK(hipFree(memPtr));
  }

  // Validating hipMemRangeGetAttributes API
  {
    int devCount = 0;
    HIP_CHECK(hipGetDeviceCount(&devCount));

    HIP_CHECK(hipSetDevice(0));

    int* memPtr = nullptr;
    HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
    REQUIRE(memPtr != nullptr);

    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetPreferredLocation, 0));
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(dyn_hipMemAdvise_ptr(memPtr, Nbytes, hipMemAdviseSetCoarseGrain, 0));

    const size_t num_attributes = 5;

    int* data[num_attributes];
    data[0] = new int;
    data[1] = new int;
    data[2] = new int[devCount];
    data[3] = new int;
    data[4] = new int;

    int* dataWithFuncPtr[num_attributes];
    dataWithFuncPtr[0] = new int;
    dataWithFuncPtr[1] = new int;
    dataWithFuncPtr[2] = new int[devCount];
    dataWithFuncPtr[3] = new int;
    dataWithFuncPtr[4] = new int;

    size_t data_sizes[num_attributes] = {sizeof(int), sizeof(int), (devCount * sizeof(int)),
                                         sizeof(int), sizeof(int)};

    hipMemRangeAttribute attributes[num_attributes] = {
        hipMemRangeAttributeReadMostly, hipMemRangeAttributePreferredLocation,
        hipMemRangeAttributeAccessedBy, hipMemRangeAttributeLastPrefetchLocation,
        hipMemRangeAttributeCoherencyMode};

    HIP_CHECK(hipMemRangeGetAttributes(reinterpret_cast<void**>(data),
                                       reinterpret_cast<size_t*>(data_sizes), attributes,
                                       num_attributes, memPtr, Nbytes));
    HIP_CHECK(dyn_hipMemRangeGetAttributes_ptr(reinterpret_cast<void**>(dataWithFuncPtr),
                                               reinterpret_cast<size_t*>(data_sizes), attributes,
                                               num_attributes, memPtr, Nbytes));

    for (int i = 0; i < num_attributes; i++) {
      if (i != 2) {
        REQUIRE(*(dataWithFuncPtr[i]) == *(data[i]));
      } else {
        for (int dev = 0; dev < devCount; dev++) {
          REQUIRE(dataWithFuncPtr[i][dev] == data[i][dev]);
        }
      }
    }

    for (int i = 0; i < num_attributes; i++) {
      delete data[i];
      delete dataWithFuncPtr[i];
    }

    HIP_CHECK(hipFree(memPtr));
  }

  // Validating hipStreamAttachMemAsync API
  {
    HIP_CHECK(hipSetDevice(0));

    int* memPtr = nullptr;
    HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
    REQUIRE(memPtr != nullptr);

    unsigned int flags[] = {hipMemAttachGlobal, hipMemAttachHost, hipMemAttachSingle};

    for (unsigned int flag : flags) {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      HIP_CHECK(dyn_hipStreamAttachMemAsync_ptr(stream, memPtr, Nbytes, flag));
      HIP_CHECK(hipStreamSynchronize(stream));

      HIP_CHECK(hipStreamDestroy(stream));
    }

    HIP_CHECK(hipFree(memPtr));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different stream ordered
 *  - Memory APIs from the hipGetProcAddress API and then validates the basic
 *  - functionality of that particular API using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisStreamOrderedMemory") {
  HIP_CHECK(hipSetDevice(0));
  int mem_pool_support = 0;

  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));

  if (mem_pool_support != 1) {
    HipTest::HIP_SKIP_TEST("Skipping test since Memory Pool is not supported");
    return;
  }

  void* hipMallocAsync_ptr = nullptr;
  void* hipFreeAsync_ptr = nullptr;
  void* hipMemPoolCreate_ptr = nullptr;
  void* hipMallocFromPoolAsync_ptr = nullptr;
  void* hipMemPoolDestroy_ptr = nullptr;
  void* hipMemPoolTrimTo_ptr = nullptr;
  void* hipMemPoolSetAccess_ptr = nullptr;
  void* hipMemPoolGetAccess_ptr = nullptr;
  void* hipMemPoolSetAttribute_ptr = nullptr;
  void* hipMemPoolGetAttribute_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(
      hipGetProcAddress("hipMallocAsync", &hipMallocAsync_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipFreeAsync", &hipFreeAsync_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemPoolCreate", &hipMemPoolCreate_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMallocFromPoolAsync", &hipMallocFromPoolAsync_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPoolDestroy", &hipMemPoolDestroy_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipMemPoolTrimTo", &hipMemPoolTrimTo_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPoolSetAccess", &hipMemPoolSetAccess_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPoolGetAccess", &hipMemPoolGetAccess_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPoolSetAttribute", &hipMemPoolSetAttribute_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemPoolGetAttribute", &hipMemPoolGetAttribute_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipMallocAsync_ptr)(void**, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t, hipStream_t)>(hipMallocAsync_ptr);
  hipError_t (*dyn_hipFreeAsync_ptr)(void*, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, hipStream_t)>(hipFreeAsync_ptr);
  hipError_t (*dyn_hipMemPoolCreate_ptr)(hipMemPool_t*, const hipMemPoolProps*) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t*, const hipMemPoolProps*)>(hipMemPoolCreate_ptr);
  hipError_t (*dyn_hipMallocFromPoolAsync_ptr)(void**, size_t, hipMemPool_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void**, size_t, hipMemPool_t, hipStream_t)>(
          hipMallocFromPoolAsync_ptr);
  hipError_t (*dyn_hipMemPoolDestroy_ptr)(hipMemPool_t) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t)>(hipMemPoolDestroy_ptr);
  hipError_t (*dyn_hipMemPoolTrimTo_ptr)(hipMemPool_t, size_t) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t, size_t)>(hipMemPoolTrimTo_ptr);
  hipError_t (*dyn_hipMemPoolSetAccess_ptr)(hipMemPool_t, const hipMemAccessDesc*, size_t) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t, const hipMemAccessDesc*, size_t)>(
          hipMemPoolSetAccess_ptr);
  hipError_t (*dyn_hipMemPoolGetAccess_ptr)(hipMemAccessFlags*, hipMemPool_t, hipMemLocation*) =
      reinterpret_cast<hipError_t (*)(hipMemAccessFlags*, hipMemPool_t, hipMemLocation*)>(
          hipMemPoolGetAccess_ptr);
  hipError_t (*dyn_hipMemPoolSetAttribute_ptr)(hipMemPool_t, hipMemPoolAttr, void*) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t, hipMemPoolAttr, void*)>(
          hipMemPoolSetAttribute_ptr);
  hipError_t (*dyn_hipMemPoolGetAttribute_ptr)(hipMemPool_t, hipMemPoolAttr, void*) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t, hipMemPoolAttr, void*)>(
          hipMemPoolGetAttribute_ptr);

  // Validating hipMallocAsync, hipFreeAsync API's
  {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    void* dPtr = nullptr;
    HIP_CHECK(dyn_hipMallocAsync_ptr(&dPtr, 256, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(dPtr != nullptr);

    size_t size = -1;
    HIP_CHECK(hipMemPtrGetInfo(dPtr, &size));
    REQUIRE(size == 256);

    HIP_CHECK(dyn_hipFreeAsync_ptr(dPtr, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    REQUIRE(hipMemPtrGetInfo(dPtr, &size) == hipErrorInvalidValue);

    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Validating hipMemPoolCreate , hipMallocFromPoolAsync API's
  {
    // hipMemPoolCreate
    hipMemPoolProps pool_props;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.handleTypes = hipMemHandleTypeNone;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.location.id = 0;
    pool_props.win32SecurityAttributes = nullptr;
    pool_props.maxSize = 1024;

    hipMemPool_t mem_pool = nullptr;
    HIP_CHECK(dyn_hipMemPoolCreate_ptr(&mem_pool, &pool_props));
    REQUIRE(mem_pool != nullptr);

    // hipMallocFromPoolAsync
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    void* dPtr = nullptr;
    HIP_CHECK(dyn_hipMallocFromPoolAsync_ptr(&dPtr, 1024, mem_pool, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(dPtr != nullptr);
    size_t size = -1;
    HIP_CHECK(hipMemPtrGetInfo(dPtr, &size));
    REQUIRE(size == 1024);
    void* dPtr2 = nullptr;
    REQUIRE(dyn_hipMallocFromPoolAsync_ptr(&dPtr2, 1, mem_pool, stream) == hipErrorOutOfMemory);

    HIP_CHECK(dyn_hipFreeAsync_ptr(dPtr, stream));
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Validating hipMemPoolDestroy API
  {
    hipMemPoolProps pool_props;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.handleTypes = hipMemHandleTypeNone;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.location.id = 0;
    pool_props.win32SecurityAttributes = nullptr;
    pool_props.maxSize = 1024;
    hipMemPool_t mem_pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
    REQUIRE(mem_pool != nullptr);

    REQUIRE(dyn_hipMemPoolDestroy_ptr(mem_pool) == hipSuccess);
    REQUIRE(dyn_hipMemPoolDestroy_ptr(mem_pool) == hipErrorInvalidValue);
  }

  // Validating hipMemPoolTrimTo API
  {
    hipMemPoolProps pool_props;
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.handleTypes = hipMemHandleTypeNone;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.location.id = 0;
    pool_props.win32SecurityAttributes = nullptr;
    pool_props.maxSize = 1024 * 1024;

    hipMemPool_t mem_pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
    REQUIRE(mem_pool != nullptr);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    void* dPtr1 = nullptr;
    HIP_CHECK(hipMallocFromPoolAsync(&dPtr1, 1024 * 1024, mem_pool, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(dPtr1 != nullptr);

    HIP_CHECK(hipFreeAsync(dPtr1, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(dyn_hipMemPoolTrimTo_ptr(mem_pool, 1024));

    void* dPtr2 = nullptr;
    REQUIRE(hipMallocFromPoolAsync(&dPtr2, 1024 * 1024, mem_pool, stream) == hipErrorOutOfMemory);
    HIP_CHECK(hipStreamSynchronize(stream));
    REQUIRE(dPtr2 == nullptr);

    HIP_CHECK(hipMemPoolDestroy(mem_pool));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  // Validating hipMemPoolSetAccess, hipMemPoolGetAccess API's
  {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount > 1) {
      HIP_CHECK(hipSetDevice(0));

      hipMemPoolProps pool_props;

      pool_props.allocType = hipMemAllocationTypePinned;
      pool_props.handleTypes = hipMemHandleTypeNone;
      pool_props.location.type = hipMemLocationTypeDevice;
      pool_props.location.id = 0;
      pool_props.win32SecurityAttributes = nullptr;
      pool_props.maxSize = 1024;

      hipMemPool_t mem_pool = nullptr;
      HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
      REQUIRE(mem_pool != nullptr);

      hipMemAccessFlags flagsList[] = {hipMemAccessFlagsProtRead, hipMemAccessFlagsProtReadWrite};

      for (hipMemAccessFlags flag : flagsList) {
        hipMemAccessDesc desc;
        hipMemLocation location = {hipMemLocationTypeDevice, 1};
        desc.location = location;
        desc.flags = flag;
        HIP_CHECK(dyn_hipMemPoolSetAccess_ptr(mem_pool, &desc, 1));

        hipMemAccessFlags flags;
        HIP_CHECK(dyn_hipMemPoolGetAccess_ptr(&flags, mem_pool, &location));
        REQUIRE(flags == flag);
      }
      HIP_CHECK(hipMemPoolDestroy(mem_pool));
    }
  }

  // Validating hipMemPoolSetAttribute, hipMemPoolGetAttribute API's
  {
    HIP_CHECK(hipSetDevice(0));

    hipMemPoolProps pool_props{};
    pool_props.allocType = hipMemAllocationTypePinned;
    pool_props.handleTypes = hipMemHandleTypeNone;
    pool_props.location.type = hipMemLocationTypeDevice;
    pool_props.location.id = 0;
    pool_props.win32SecurityAttributes = nullptr;
    pool_props.maxSize = 1024 * 1024;
    hipMemPool_t mem_pool = nullptr;
    HIP_CHECK(hipMemPoolCreate(&mem_pool, &pool_props));
    REQUIRE(mem_pool != nullptr);

    // Attribute - hipMemPoolReuseFollowEventDependencies
    {
      hipMemPoolAttr attr = hipMemPoolReuseFollowEventDependencies;

      int valueToSet = 0;
      HIP_CHECK(dyn_hipMemPoolSetAttribute_ptr(mem_pool, attr, &valueToSet));

      int value;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, attr, &value));
      REQUIRE(value == 0);
    }

    // Attribute - hipMemPoolReuseAllowOpportunistic
    {
      hipMemPoolAttr attr = hipMemPoolReuseAllowOpportunistic;

      int valueToSet = 0;
      HIP_CHECK(dyn_hipMemPoolSetAttribute_ptr(mem_pool, attr, &valueToSet));

      int value;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, attr, &value));
      REQUIRE(value == 0);
    }

    // Attribute - hipMemPoolReuseAllowInternalDependencies
    {
      hipMemPoolAttr attr = hipMemPoolReuseAllowInternalDependencies;

      int valueToSet = 0;
      HIP_CHECK(dyn_hipMemPoolSetAttribute_ptr(mem_pool, attr, &valueToSet));

      int value;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, attr, &value));
      REQUIRE(value == 0);
    }

    // Attribute - hipMemPoolAttrReleaseThreshold
    {
      hipMemPoolAttr attr = hipMemPoolAttrReleaseThreshold;

      uint64_t valueToSet = 1024;
      HIP_CHECK(dyn_hipMemPoolSetAttribute_ptr(mem_pool, attr, &valueToSet));

      uint64_t value;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, attr, &value));
      REQUIRE(value == 1024);
    }
    /*
       Attribute's are,
       hipMemPoolAttrReservedMemCurrent
       hipMemPoolAttrReservedMemHigh
       hipMemPoolAttrUsedMemCurrent
       hipMemPoolAttrUsedMemHigh
    */
    {
      hipStream_t stream;
      HIP_CHECK(hipStreamCreate(&stream));

      void* dPtr1 = nullptr;
      HIP_CHECK(hipMallocFromPoolAsync(&dPtr1, 1024, mem_pool, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      REQUIRE(dPtr1 != nullptr);

      uint64_t value = 0;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrReservedMemCurrent, &value));
      REQUIRE(value >= 1024);

      value = 0;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrReservedMemHigh, &value));
      REQUIRE(value >= 1024);

      value = 0;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrUsedMemCurrent, &value));
      REQUIRE(value >= 1024);

      value = 0;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrUsedMemHigh, &value));
      REQUIRE(value >= 1024);

      HIP_CHECK(hipFreeAsync(dPtr1, stream));

      uint64_t valueToSet = 0;
      HIP_CHECK(
          dyn_hipMemPoolSetAttribute_ptr(mem_pool, hipMemPoolAttrReservedMemHigh, &valueToSet));
      HIP_CHECK(dyn_hipMemPoolSetAttribute_ptr(mem_pool, hipMemPoolAttrUsedMemHigh, &valueToSet));

      value = -1;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrReservedMemHigh, &value));
      REQUIRE(value == 0);

      value = -1;
      HIP_CHECK(dyn_hipMemPoolGetAttribute_ptr(mem_pool, hipMemPoolAttrUsedMemHigh, &value));
      REQUIRE(value == 0);

      HIP_CHECK(hipStreamDestroy(stream));
    }
    HIP_CHECK(hipMemPoolDestroy(mem_pool));
  }
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of different Peer to peer Memory
 *  - APIs from the hipGetProcAddress API and then validates the basic
 *  - functionality of that particular API using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/memory/hipGetProcAddress_Memory_APIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_MemoryApisPeerToPeer") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");
    return;
  }

  void* hipMemGetAddressRange_ptr = nullptr;
  void* hipMemcpyPeer_ptr = nullptr;
  void* hipMemcpyPeerAsync_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipMemGetAddressRange", &hipMemGetAddressRange_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyPeer", &hipMemcpyPeer_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipMemcpyPeerAsync", &hipMemcpyPeerAsync_ptr, currentHipVersion, 0,
                              nullptr));

  hipError_t (*dyn_hipMemGetAddressRange_ptr)(hipDeviceptr_t*, size_t*, hipDeviceptr_t) =
      reinterpret_cast<hipError_t (*)(hipDeviceptr_t*, size_t*, hipDeviceptr_t)>(
          hipMemGetAddressRange_ptr);
  hipError_t (*dyn_hipMemcpyPeer_ptr)(void*, int, const void*, int, size_t) =
      reinterpret_cast<hipError_t (*)(void*, int, const void*, int, size_t)>(hipMemcpyPeer_ptr);
  hipError_t (*dyn_hipMemcpyPeerAsync_ptr)(void*, int, const void*, int, size_t, hipStream_t) =
      reinterpret_cast<hipError_t (*)(void*, int, const void*, int, size_t, hipStream_t)>(
          hipMemcpyPeerAsync_ptr);

  int deviceId = 0;
  int peerDeviceId = 1;

  int canAccessPeer = 0;
  HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, deviceId, peerDeviceId));
  if (!canAccessPeer) {
    std::string msg = "Skipped as peer access cannot be enabled between devices " +
                      std::to_string(deviceId) + " " + std::to_string(peerDeviceId);
    HipTest::HIP_SKIP_TEST(msg.c_str());
    return;
  }

  const int N = 16;
  const int Nbytes = N * sizeof(int);
  int value = 10;

  // Validating hipMemGetAddressRange API
  {
    int* devPtr = nullptr;
    HIP_CHECK(hipMalloc(&devPtr, 4 * sizeof(int)));
    REQUIRE(devPtr != nullptr);

    size_t size = -1;
    hipDeviceptr_t basePtr = nullptr;

    HIP_CHECK(dyn_hipMemGetAddressRange_ptr(&basePtr, &size, devPtr + 3));

    REQUIRE(basePtr == devPtr);
    REQUIRE(size == (4 * sizeof(int)));
  }

  // Validating hipMemcpyPeer API
  {
    HIP_CHECK(hipSetDevice(deviceId));

    int* srcDevPtr = nullptr;
    HIP_CHECK(hipMalloc(&srcDevPtr, Nbytes));
    REQUIRE(srcDevPtr != nullptr);
    fillDeviceArray(srcDevPtr, N, value);

    HIP_CHECK(hipSetDevice(peerDeviceId));

    int* dstDevPtr = nullptr;
    HIP_CHECK(hipMalloc(&dstDevPtr, Nbytes));
    REQUIRE(dstDevPtr != nullptr);

    HIP_CHECK(dyn_hipMemcpyPeer_ptr(dstDevPtr, peerDeviceId, srcDevPtr, deviceId, Nbytes));

    validateHostArray(dstDevPtr, N, value);

    HIP_CHECK(hipFree(srcDevPtr));
    HIP_CHECK(hipFree(dstDevPtr));
  }

  // Validating hipMemcpyPeerAsync API
  {
    HIP_CHECK(hipSetDevice(deviceId));

    int* srcDevPtr = nullptr;
    HIP_CHECK(hipMalloc(&srcDevPtr, Nbytes));
    REQUIRE(srcDevPtr != nullptr);
    fillDeviceArray(srcDevPtr, N, value);

    HIP_CHECK(hipSetDevice(peerDeviceId));

    int* dstDevPtr = nullptr;
    HIP_CHECK(hipMalloc(&dstDevPtr, Nbytes));
    REQUIRE(dstDevPtr != nullptr);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    HIP_CHECK(
        dyn_hipMemcpyPeerAsync_ptr(dstDevPtr, peerDeviceId, srcDevPtr, deviceId, Nbytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    validateHostArray(dstDevPtr, N, value);

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(srcDevPtr));
    HIP_CHECK(hipFree(dstDevPtr));
  }
}
