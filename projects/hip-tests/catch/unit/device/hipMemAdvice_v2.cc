/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <numa.h>

/**
 * Kernel to fill value for each element in the given array
 */
static __global__ void fillDataKernel(int *arr, size_t size, int value) {
  size_t offset = blockDim.x * blockIdx.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < size; i += stride) {
    arr[i] = value;
  }
}

TEST_CASE("Unit_hipMemAdvise_v2_Basic_Positive") {
  const int N = 1024;
  const int Nbytes = N * sizeof(int);
  int value = 10;

  int* memPtr = nullptr;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMallocManaged(reinterpret_cast<void **>(&memPtr), Nbytes, hipMemAttachGlobal));
  REQUIRE(memPtr != nullptr);

  SECTION("With Device") {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    for (int deviceId = 0; deviceId < deviceCount; deviceId++) {
	  hipMemLocation location;
      location.type = hipMemLocationTypeDevice;	
      location.id = deviceId;

      HIP_CHECK(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

      fillDataKernel<<<1, N/2>>>(memPtr, N, value);
	  HIP_CHECK(hipDeviceSynchronize());
	}
  }

  SECTION("With Host") {
	hipMemLocation location;
    location.type = hipMemLocationTypeHost;

    HIP_CHECK(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

	for(int i=0; i<N; i++) {
	  memPtr[i] = value;
	}
  }

  SECTION("With Host NUMA") {
	hipMemLocation location;
    location.type = hipMemLocationTypeHostNuma;
    
	int num_numa_nodes = numa_num_configured_nodes();
    std::cout << "Number of NUMA nodes: " << num_numa_nodes << std::endl;

    int maxNodes = numa_num_possible_nodes();
    std::cout << "maxNodes : " << maxNodes << std::endl;

	
	location.id = GENERATE(0, 1);

    HIP_CHECK(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

	for(int i=0; i<N; i++) {
	  memPtr[i] = value;
	}
  }

  SECTION("With Host Numa Current") {
	hipMemLocation location;
    location.type = hipMemLocationTypeHostNumaCurrent;

    HIP_CHECK(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location));

	for(int i=0; i<N; i++) {
	  memPtr[i] = value;
	}
  }

  for(int i=0; i<N; i++) {
    REQUIRE(memPtr[i] == value);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(memPtr));
}

/**
 * Test Description
 * ------------------------
 *  - This test case checks the following Negative scenarios
 *  - for the hipStreamGetAttribute,
 *  - 1) With dev_ptr as nullptr
 *  - 2) With count 0
 *  - 3) With count larger than actual size
 *  - 4) With invalid device
 *  - 5) With Invalid location
 *  - 6) With Invalid location -1
 *  - 7) With invalid numa node 
 *  - 8) With Invalid Advise
 * Test source
 * ------------------------
 *  - unit/memory/hipMemPrefetchAsync_v2.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipMemAdvise_v2_Negative") {
  const int N = 16;
  const int Nbytes = N * sizeof(int);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  void* memPtr = nullptr;
  HIP_CHECK(hipMallocManaged(&memPtr, Nbytes, hipMemAttachGlobal));
  
  hipMemLocation location;
  location.type = hipMemLocationTypeHost;

  SECTION("With dev_ptr as nullptr") {
    HIP_CHECK_ERROR(hipMemAdvise_v2(nullptr, Nbytes, hipMemAdviseSetReadMostly, location), hipErrorInvalidValue);
  }

  SECTION("With count 0") {
    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, 0, hipMemAdviseSetReadMostly, location), hipErrorInvalidValue);
  }

  SECTION("With count larger than actual size") {
    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes+10, hipMemAdviseSetReadMostly, location), hipErrorInvalidValue);
  }

  #if 0
  SECTION("With invalid device") {
    hipMemLocation dstLocation;
    dstLocation.type = hipMemLocationTypeDevice;
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    dstLocation.id = deviceCount;
    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location),
	                hipErrorInvalidDevice);  // No error
  }
  #endif
  
  SECTION("With Invalid location") {
    hipMemLocation location;
    location.type = GENERATE(hipMemLocationTypeInvalid, hipMemLocationTypeNone);

    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location),
                    hipErrorInvalidValue);
  }

  SECTION("With Invalid location -1") {
    hipMemLocation location;
    location.type = static_cast<hipMemLocationType>(-1);

    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, location),
                    hipErrorInvalidValue);
  }
  
  #if 0
  SECTION("With invalid numa node") {
    hipMemLocation dstLocation;
    dstLocation.type = hipMemLocationTypeHostNuma;
	// get numa nodes count
    dstLocation.id = 2;
    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, hipMemAdviseSetReadMostly, dstLocation),
	                hipErrorInvalidDevice); // No error
  }
  #endif

  // with in valid Advise
  SECTION("With Invalid Advise") {
    hipMemLocation location;
    location.type = hipMemLocationTypeDevice;
    location.id = 0;

	hipMemoryAdvise advice = static_cast<hipMemoryAdvise>(-1);

    HIP_CHECK_ERROR(hipMemAdvise_v2(memPtr, Nbytes, advice, location),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(memPtr));
}
