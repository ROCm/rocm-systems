/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/*
hipMalloc3D API test scenarios
1. Basic Functionality
2. Negative Scenarios
3. Allocating Small and big chunk data
4. Multithreaded scenario
*/

#include <hip_test_common.hh>
static constexpr auto SMALL_SIZE{4};
static constexpr auto CHUNK_LOOP{100};
static constexpr auto BIG_SIZE{100};
/*
This API verifies hipMalloc3D API by allocating memory in smaller chunks for
CHUNK_LOOP iterations
*/
static void MemoryAlloc3DDiffSizes(int gpu) {
  HIPCHECK(hipSetDevice(gpu));
  std::vector<size_t> array_size;
  array_size.push_back(SMALL_SIZE);
  array_size.push_back(BIG_SIZE);
  for (auto& sizes : array_size) {
    size_t width = sizes * sizeof(float);
    size_t height{sizes}, depth{sizes};
    hipPitchedPtr devPitchedPtr[CHUNK_LOOP];
    hipExtent extent = make_hipExtent(width, height, depth);
    size_t ptot, pavail;
    HIPCHECK(hipMemGetInfo(&pavail, &ptot));
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIPCHECK(hipMalloc3D(&devPitchedPtr[i], extent));
    }
    for (int i = 0; i < CHUNK_LOOP; i++) {
      HIPCHECK(hipFree(devPitchedPtr[i].ptr));
    }
  }
}

static void Malloc3DThreadFunc(int gpu) { MemoryAlloc3DDiffSizes(gpu); }

/*
 * This test allocates via hipMalloc3D and deallotes via hipFree().
 * To verify that the returned memory is again available, we use hipMemGetInfo().
 * On Windows, allocating/deallocating small amounts of device memory will not make a difference in
 * the reported available memory because they go into subheaps of size
 * PalSettings::subAllocationChunkSize_ and only whenever we trigger a whole subheap
 *  allocation/deallocation we will see a difference in the reported value.
 * To make sure we make a difference in the reported available memory; we perform small several
 * allocation whose combined size goes over PalSettings::subAllocationChunkSize_;
 * hipMemGetInfo() should indicate the memory went down after we hipFree() all of them
 */
TEST_CASE("Unit_hipMalloc3D_Basic") {
  CHECK_IMAGE_SUPPORT

  static constexpr int ChunkSize = 64;  // (in megabytes)
  static constexpr int NumAllocations = 3;

  size_t width{(ChunkSize * 1024) / NumAllocations}, height{1024}, depth{1};
  hipPitchedPtr devPitchedPtr[NumAllocations];
  hipExtent extent = make_hipExtent(width, height, depth);
  size_t tot, avail, itot, iavail, ptot, pavail;
  HIP_CHECK(hipMemGetInfo(&pavail, &ptot));

  for (int i = 0; i < NumAllocations; i++) {
    REQUIRE(hipMalloc3D(&devPitchedPtr[i], extent) == hipSuccess);
  }

  HIP_CHECK(hipMemGetInfo(&iavail, &itot));

  if (iavail >= pavail)
    WARN(
        "hipMemGetInfo() did not report increased memory usage after calling hipMalloc3D(). "
        "Before: "
        << pavail << " after: " << iavail);

  for (int i = 0; i < NumAllocations; i++) {
    HIPCHECK(hipFree(devPitchedPtr[i].ptr));
  }

  HIP_CHECK(hipMemGetInfo(&avail, &tot));

  // as the runtime might cache some of the allocations and also it is difficult the
  // available amount returned is the same as the one we got the first time because of
  // other processes running on the system; we consider a success if at least a size
  // equivalent to two of the allocations has become available
  // (give or take 4MB, as there is bookkeeping overhead)
  // This test was too brittle before, when it was expecting 'avail' to be equal to 'pavail'
  if (avail < pavail - height * width * depth - 4 * 1024 * 1024) {
    WARN("Memory leak of hipMalloc3D API in multithreaded scenario."
         << " Available memory before the hipMalloc3D() call (bytes): " << pavail
         << " Available memory after the call: " << iavail
         << " Available memory after hipFree(): " << avail);
    REQUIRE(false);
  }
}

/*
This testcase verifies the hipMalloc3D API by allocating
smaller and big chunk data.
*/
TEST_CASE("Unit_hipMalloc3D_SmallandBigChunks") {
  CHECK_IMAGE_SUPPORT

  MemoryAlloc3DDiffSizes(0);
}

/*
This testcase verifies the hipMalloc3D API in multithreaded
scenario by launching threads in parallel on multiple GPUs
and verifies the hipMalloc3D API with small and big chunks data
*/
TEST_CASE("Unit_hipMalloc3D_MultiThread") {
  CHECK_IMAGE_SUPPORT

  std::vector<std::thread> threadlist;
  int devCnt = 0;

  devCnt = HipTest::getDeviceCount();

  for (int i = 0; i < devCnt; i++) {
    threadlist.push_back(std::thread(Malloc3DThreadFunc, i));
  }

  for (auto& t : threadlist) {
    t.join();
  }
}
