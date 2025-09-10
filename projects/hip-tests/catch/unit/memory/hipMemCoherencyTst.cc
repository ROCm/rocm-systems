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

/* Test Case Description:
   Scenario 1: The  test validates if fine grain
   behavior is observed or not with memory allocated using hipHostMalloc()
   Scenario 2: The test validates if fine grain
   behavior is observed or not with memory allocated using hipMallocManaged()
   Scenario 3: The test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise
   Scenario 4: The test validates if memory access is fine
   with memory allocated using hipMalloc() and CoarseGrain Advise
   Scenario 5: The test validates if fine grain
   behavior is observed or not with memory allocated using
   hipExtMallocWithFlags()*/


#include <hip_test_common.hh>
#include <chrono>

__global__ void CoherentTst(int* ptr) {  // ptr was set to 1
  atomicAdd_system(ptr, 1);              // now ptr is 2
  while (atomicCAS_system(ptr, 3, 4) != 3) {
    // wait till ptr is updated to 3 in host, then change it to 4
  }
}

__global__ void SquareKrnl(int* ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}

// The variable below will work as signal to decide pass/fail
static bool YES_COHERENT = false;

enum class MemoryType { kHostMalloc, kManaged, kDeviceFineGrained };
// The function tests the coherency of allocated memory
// If this test hangs, means there is issue in coherency
static void TstCoherency(int* ptr, MemoryType type) {
  int* dptr = nullptr;
  hipStream_t stream{};
  HIP_CHECK(hipStreamCreate(&stream));
  int apu = 0;
  HIP_CHECK(hipDeviceGetAttribute(&apu, hipDeviceAttributeIntegrated, 0));
  fprintf(stderr, "Device 0 is %s\n", apu ? "apu" : "dgpu");
  // Host builtin atomcs cannot work on device fine grained mem on dgpu
  // Note: hipDeviceAttributeHostNativeAtomicSupported should return 1 for kHostMalloc here
  bool supportHostAtomic = type != MemoryType::kDeviceFineGrained || apu;
  // storing value 1 in the memory created above
  *ptr = 1;

  if (type == MemoryType::kHostMalloc) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dptr), ptr, 0));
    CoherentTst<<<1, 1, 0, stream>>>(dptr);
  } else {
    CoherentTst<<<1, 1, 0, stream>>>(ptr);
  }
  // To prevent Windows batch dispatching issue, run inspecting code in thread
  std::thread my_thread([ptr, supportHostAtomic] {
    int d = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    while (
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
            .count() <= 3) {
      d = supportHostAtomic ? __sync_fetch_and_add(ptr, 0) : *ptr;  // Retrieve *ptr
      if (d == 2) break;  // If kernel has updated *ptr to 2, exit
    }  // wait till ptr is updated to 2 from kernel or 3 seconds
    if (d != 2) {
      // 3 seconds should be long enough for kernel to update ptr
      fprintf(stderr, "d = %d hasn't been updated to 2 in 3s\n", d);
      return;
    }
    // increment it to 3
    if (supportHostAtomic) {
      __sync_fetch_and_add(ptr, 1);
    } else {
      *ptr += 1;
    }
  });

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipStreamDestroy(stream));
  my_thread.join();
  if (*ptr == 4) {
    YES_COHERENT = true;
  }
}

/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipHostMalloc()*/
TEST_CASE("Unit_hipHostMalloc_CoherentTst") {
  HIP_CHECK(hipSetDevice(0));
  CHECK_PCIE_ATOMIC_SUPPORT;

  int *Ptr = nullptr, SIZE = sizeof(int);
  YES_COHERENT = false;

  // Allocating hipHostMalloc() memory with hipHostMallocCoherent flag
  SECTION("hipHostMalloc with hipHostMallocCoherent flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocCoherent));
  }
  SECTION("hipHostMalloc with Default flag") { HIP_CHECK(hipHostMalloc(&Ptr, SIZE)); }
  SECTION("hipHostMalloc with hipHostMallocMapped flag") {
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocMapped));
  }

  TstCoherency(Ptr, MemoryType::kHostMalloc);
  HIP_CHECK(hipHostFree(Ptr));
  REQUIRE(YES_COHERENT);
}

/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using hipMallocManaged()*/
// The following tests are disabled for Nvidia as they are not consistently
// passing
#if HT_AMD
TEST_CASE("Unit_hipMallocManaged_CoherentTst") {
  HIP_CHECK(hipSetDevice(0));
  CHECK_PCIE_ATOMIC_SUPPORT;

  int *Ptr = nullptr, SIZE = sizeof(int), managed = 0;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed == 1) {
    // Allocating hipMallocManaged() memory
    SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
    }
    SECTION("hipMallocManaged with hipMemAttachHost flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
    }
    TstCoherency(Ptr, MemoryType::kManaged);
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED(
        "GPU 0 doesn't support ManagedMemory "
        "device attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif

/* Test case description: The following test validates if memory access is fine
   with memory allocated using hipMallocManaged() and CoarseGrain Advise*/
TEST_CASE("Unit_hipMallocManaged_CoherentTstWthAdvise") {
  HIP_CHECK(hipSetDevice(0));
  int *Ptr = nullptr, SIZE = sizeof(int), managed = 0;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);

  if (managed == 1) {
    // Allocating hipMallocManaged() memory
    SECTION("hipMallocManaged with hipMemAttachGlobal flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachGlobal));
    }
    SECTION("hipMallocManaged with hipMemAttachHost flag") {
      HIP_CHECK(hipMallocManaged(&Ptr, SIZE, hipMemAttachHost));
    }
#if HT_AMD
    HIP_CHECK(hipMemAdvise(Ptr, SIZE, hipMemAdviseSetCoarseGrain, 0));
#endif
    // Initializing Ptr memory with 9
    *Ptr = 9;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
    HIP_CHECK(hipStreamSynchronize(strm));
    if (*Ptr == 81) {
      YES_COHERENT = true;
    }
    HIP_CHECK(hipFree(Ptr));
    HIP_CHECK(hipStreamDestroy(strm));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the test with Pass result.\n");
  }
}


/* Test case description: The following test validates if memory allocated
   using hipMalloc() are of type Coarse Grain*/
// The following tests are disabled for Nvidia as they are not applicable
#if HT_AMD
TEST_CASE("Unit_hipMalloc_CoherentTst") {
  HIP_CHECK(hipSetDevice(0));
  int *Ptr = nullptr, SIZE = sizeof(int);
  uint32_t svm_attrib = 0;
  bool IfTstPassed = false;
  // Allocating hipMalloc() memory
  HIP_CHECK(hipMalloc(&Ptr, SIZE));
  HIP_CHECK(hipMemRangeGetAttribute(&svm_attrib, sizeof(svm_attrib),
                                    hipMemRangeAttributeCoherencyMode, Ptr, SIZE));
  if (svm_attrib == hipMemRangeCoherencyModeCoarseGrain) {
    IfTstPassed = true;
  }
  HIP_CHECK(hipFree(Ptr));
  REQUIRE(IfTstPassed);
}
#endif
/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using
   hipExtMallocWithFlags()*/
#if HT_AMD
TEST_CASE("Unit_hipExtMallocWithFlags_CoherentTst") {
  HIP_CHECK(hipSetDevice(0));
  int *Ptr = nullptr, SIZE = sizeof(int), InitVal = 9, Pageable = 0, managed = 0, finegrain = 0;
  bool FineGrain = true;
  YES_COHERENT = false;

  HIP_CHECK(hipDeviceGetAttribute(&Pageable, hipDeviceAttributePageableMemoryAccess, 0));
  INFO("hipDeviceAttributePageableMemoryAccess: " << Pageable);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed == 1 && Pageable == 1) {
    // Allocating hipExtMallocWithFlags() memory with flags
    HIP_CHECK(hipDeviceGetAttribute(&finegrain, hipDeviceAttributeFineGrainSupport, 0));
    if (finegrain == 1) {
      SECTION("hipExtMallocWithFlags with hipDeviceMallocFinegrained flag") {
        HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE * 2,
                                        hipDeviceMallocFinegrained));
      }
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocSignalMemory flag") {
      // for hipMallocSignalMemory flag the size of memory must be 8
      HIP_CHECK(
          hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE * 2, hipMallocSignalMemory));
    }
    SECTION("hipExtMallocWithFlags with hipDeviceMallocDefault flag") {
      /* hipExtMallocWithFlags() with flag
      hipDeviceMallocDefault allocates CoarseGrain memory */
      FineGrain = false;
      HIP_CHECK(
          hipExtMallocWithFlags(reinterpret_cast<void**>(&Ptr), SIZE * 2, hipDeviceMallocDefault));
    }
    if (FineGrain) {
      TstCoherency(Ptr, MemoryType::kDeviceFineGrained);
    } else {
      *Ptr = InitVal;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      if (*Ptr == (InitVal * InitVal)) {
        YES_COHERENT = true;
      }
    }
    HIP_CHECK(hipFree(Ptr));
    REQUIRE(YES_COHERENT);
  } else {
    SUCCEED(
        "GPU 0 doesn't support ManagedMemory or PageableMemoryAccess"
        "device attribute. Hence skipping the test with Pass result.\n");
  }
}
#endif
