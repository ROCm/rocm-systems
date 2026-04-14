/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 Run through few sanity tests to verify different attributes of hipPointerGetAttribute
*/
#include <hip_test_common.hh>
#include <vector>
#include <iostream>
#include <string>

// Run few simple cases including  host pointer arithmetic:
TEST_CASE(Unit_hipPtrGetAttribute_Simple) {
  HIP_CHECK(hipSetDevice(0));
  size_t Nbytes = 0;
  constexpr size_t N{1000000};
  Nbytes = N * sizeof(char);
  printf("\n");

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  char* A_d;
  char* A_Pinned_h;
  char* A_Hmm;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_Pinned_h), Nbytes, hipHostMallocDefault));
  HIP_CHECK(hipMallocManaged(&A_Hmm, Nbytes));

  size_t free, total;
  HIP_CHECK(hipMemGetInfo(&free, &total));
  printf("hipMemGetInfo: free=%zu (%4.2f) Nbytes=%lu total=%zu (%4.2f)\n", free,
         (free / 1024.0 / 1024.0), static_cast<unsigned long>(Nbytes), total,
         (total / 1024.0 / 1024.0));
  REQUIRE(free + Nbytes <= total);

  hipDeviceptr_t data = 0;

  // Device memory
  printf("\nDevice memory (hipMalloc)\n");
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  char* ptr1 = reinterpret_cast<char*>(data);

  // Check pointer arithmetic cases:
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_d + 100)));
  char* ptr2 = reinterpret_cast<char*>(data);
  REQUIRE(ptr2 == ptr1 + 100);

  // Corner case at end of array:
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_d + Nbytes - 1)));
  ptr2 = reinterpret_cast<char*>(data);
  REQUIRE(ptr2 == (ptr1 + Nbytes - 1));

  // Device-visible host memory
  printf("\nDevice-visible host memory (hipHostMalloc)\n");
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h)));
  ptr1 = reinterpret_cast<char*>(data);

  // Check pointer arithmetic cases:
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h + 100)));
  ptr2 = reinterpret_cast<char*>(data);
  REQUIRE(ptr2 == ptr1 + 100);

  // Corner case at end of array:
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_HOST_POINTER,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h + Nbytes - 1)));
  ptr2 = reinterpret_cast<char*>(data);
  REQUIRE(ptr2 == (ptr1 + Nbytes - 1));

  // HIP_POINTER_ATTRIBUTE_MEMORY_TYPE
  unsigned int datatype;
  HIP_CHECK(hipPointerGetAttribute(&datatype, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  REQUIRE(datatype == hipMemoryTypeDevice);

  HIP_CHECK(hipPointerGetAttribute(&datatype, HIP_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h)));
  REQUIRE(datatype == hipMemoryTypeHost);

  // HIP_POINTER_ATTRIBUTE_IS_MANAGED
  bool isHmm;
  HIP_CHECK(hipPointerGetAttribute(&isHmm, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                                   reinterpret_cast<hipDeviceptr_t>(A_Hmm)));
  REQUIRE(isHmm == 1);

  HIP_CHECK(hipPointerGetAttribute(&isHmm, HIP_POINTER_ATTRIBUTE_IS_MANAGED,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h)));
  REQUIRE(isHmm == 0);

  // HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL
  if (numDevices > 1) {
    int canAccess = -1;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, 1, 0));
    if (canAccess == 1) {
      HIP_CHECK(hipSetDevice(1));
      HIP_CHECK(hipPointerGetAttribute(&datatype, HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                       reinterpret_cast<hipDeviceptr_t>(A_d)));
      REQUIRE(datatype == 0);
    }
  }

  // HIP_POINTER_ATTRIBUTE_MAPPED
  bool isMapped;
  HIP_CHECK(hipPointerGetAttribute(&isMapped, HIP_POINTER_ATTRIBUTE_MAPPED,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  REQUIRE(isMapped == 1);

  // HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
  HIP_CHECK(hipPointerGetAttribute(&data, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                   reinterpret_cast<hipDeviceptr_t>(A_d + 2)));
  char* ptr3 = reinterpret_cast<char*>(data);
  REQUIRE(ptr3 == A_d);

  // HIP_POINTER_ATTRIBUTE_RANGE_SIZE
  HIP_CHECK(hipPointerGetAttribute(&datatype, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  REQUIRE(datatype == Nbytes);

  unsigned int bufId1, bufId2;
  // HIP_POINTER_ATTRIBUTE_BUFFER_ID
  HIP_CHECK(hipPointerGetAttribute(&bufId1, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                   reinterpret_cast<hipDeviceptr_t>(A_d)));
  HIP_CHECK(hipPointerGetAttribute(&bufId2, HIP_POINTER_ATTRIBUTE_BUFFER_ID,
                                   reinterpret_cast<hipDeviceptr_t>(A_Pinned_h)));
  REQUIRE(bufId1 != bufId2);

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipHostFree(A_Pinned_h));
  HIP_CHECK(hipFree(A_Hmm));
}
