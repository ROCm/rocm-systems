/*
   Copyright (c) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
   Scenario 3: The test validates if fine grain
   behavior is observed or not with memory allocated using malloc()
   Scenario 4: The test validates if coarse grain memory
   behavior is observed or not with memory allocated using malloc()
   Scenario 5: The test validates if fine memory
   behavior is observed or not with memory allocated using mmap()
   Scenario 6: The test validates if coarse grain memory
   behavior is observed or not with memory allocated using mmap()
   Scenario:7 Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0
   Scenario:8 Test Case Description: The following test checks if the memory
   exhibits fine grain behavior when HIP_HOST_COHERENT is set to 1
   */

#ifdef __linux__
#include <hip_test_common.hh>
#include <hip_test_features.hh>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <chrono>
#include "../unit/memory/hipSVMCommon.h"

__global__ void CoherentTst(int* ptr, volatile unsigned int* expired) {
  // Incrementing the value by 1
  atomicAdd_system(ptr, 1);
  // The following while loop checks the value until expiration.
  while (*expired == 0) {
    if (atomicCAS_system(ptr, 3, 4) == 3) break;
  }
}

__global__ void SquareKrnl(int* ptr) {
  // ptr value squared here
  *ptr = (*ptr) * (*ptr);
}

// The function tests the coherency of allocated memory
// Return false on failure, true on success.
bool static TstCoherency(int* Ptr, bool HmmMem) {
  using namespace std::chrono_literals;
  int* Dptr = nullptr;
  hipStream_t strm;
  HIP_CHECK(hipStreamCreate(&strm));
  // storing value 1 in the memory created above
  *Ptr = 1;

  unsigned int* expired = nullptr;
  HIP_CHECK(hipHostMalloc(&expired, sizeof(unsigned int)));  // hipHostMallocCoherent by defaut
  *expired = 0;

  if (!HmmMem) {
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&Dptr), Ptr, 0));
    CoherentTst<<<1, 1, 0, strm>>>(Dptr, expired);
  } else {
    CoherentTst<<<1, 1, 0, strm>>>(Ptr, expired);
  }
  // looping until the value is 2 for 3 seconds
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  while (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start)
             .count() < 3) {
    if (*Ptr == 2) {
      *Ptr += 1;
      std::this_thread::sleep_for(200ms);  // Make sure kernel gets updated Dptr
      break;
    }
  }
  *expired = 1;  // Notify kernel loop to exit
  HIP_CHECK(hipStreamSynchronize(strm));
  HIP_CHECK(hipStreamDestroy(strm));
  HIP_CHECK(hipHostFree(expired));

  if (*Ptr == 4) {
    return true;
  }
  fprintf(stderr, "TstCoherency: *Ptr=%u\b", *Ptr);
  return false;
}

/* Test case description: The following test validates if fine grain
   behavior is observed or not with memory allocated using malloc()*/
// The following test is failing on Nvidia platform hence disabled it for now
#if HT_AMD
TEST_CASE("Unit_malloc_CoherentTst") {
  CHECK_PCIE_ATOMIC_SUPPORT
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  char* p = NULL;
  p = strstr(prop.gcnArchName, "xnack+");
  if (p) {
    // Test Case execution begins from here
    int managed = 0;
    HIPCHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
    if (managed == 1) {
      int *Ptr = nullptr, SIZE = sizeof(int);
      bool HmmMem = true;

      // Allocating hipMallocManaged() memory
      Ptr = reinterpret_cast<int*>(malloc(SIZE));
      auto ret = TstCoherency(Ptr, HmmMem);
      free(Ptr);
      REQUIRE(ret);
    }
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test...\n");
  }
}
#endif


/* Test case description: The following test validates if coarse grain memory
   behavior is observed or not with memory allocated using malloc()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_malloc_CoherentTstWthAdvise") {
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  char* p = NULL;
  p = strstr(prop.gcnArchName, "xnack+");
  if (p) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
    if (managed == 1) {
      int *Ptr = nullptr, SIZE = sizeof(int);

      // Allocating hipMallocManaged() memory
      Ptr = reinterpret_cast<int*>(malloc(SIZE));
      *Ptr = 4;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      HIP_CHECK(hipStreamDestroy(strm));
      REQUIRE(*Ptr == 16);
    }
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test...\n");
  }
}
#endif

/* Test case description: The following test validates if fine memory
   behavior is observed or not with memory allocated using mmap()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_mmap_CoherentTst") {
  CHECK_PCIE_ATOMIC_SUPPORT
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  char* p = NULL;
  p = strstr(prop.gcnArchName, "xnack+");
  if (p) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
    if (managed == 1) {
      bool HmmMem = true;
      int* Ptr = reinterpret_cast<int*>(
          mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
      if (Ptr == MAP_FAILED) {
        WARN("Mapping Failed\n");
        REQUIRE(false);
      }
      auto ret = TstCoherency(Ptr, HmmMem);
      int err = munmap(Ptr, sizeof(int));
      if (err != 0) {
        WARN("munmap failed\n");
      }
      REQUIRE(ret);
    }
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test...\n");
  }
}
#endif

/* Test case description: The following test validates if coarse grain memory
   behavior is observed or not with memory allocated using mmap()*/
// The following test is failing on Nvidia platform hence disabling it for now
#if HT_AMD
TEST_CASE("Unit_mmap_CoherentTstWthAdvise") {
  hipDeviceProp_t prop;
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  char* p = NULL;
  p = strstr(prop.gcnArchName, "xnack+");
  if (p) {
    int managed = 0;
    HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
    if (managed == 1) {
      int SIZE = sizeof(int);
      int* Ptr = reinterpret_cast<int*>(
          mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0));
      if (Ptr == MAP_FAILED) {
        WARN("Mapping Failed\n");
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Ptr, SIZE, hipMemAdviseSetCoarseGrain, 0));
      // Initializing the value with 9
      *Ptr = 9;
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      SquareKrnl<<<1, 1, 0, strm>>>(Ptr);
      HIP_CHECK(hipStreamSynchronize(strm));
      bool IfTstPassed = false;
      if (*Ptr == 81) {
        IfTstPassed = true;
      }
      int err = munmap(Ptr, SIZE);
      if (err != 0) {
        WARN("munmap failed\n");
      }
      REQUIRE(IfTstPassed);
    }
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test...\n");
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg1") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocPortable));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg2") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocWriteCombined));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg3") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNumaUser));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory is
   accessible when HIP_HOST_COHERENT is set to 0*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv0Flg4") {
  if ((setenv("HIP_HOST_COHERENT", "0", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {
    int *Ptr = nullptr, *PtrD = nullptr, SIZE = sizeof(int);
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNonCoherent));
    *Ptr = 4;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&PtrD), Ptr, 0));
    SquareKrnl<<<1, 1, 0, strm>>>(PtrD);
    HIP_CHECK(hipStreamSynchronize(strm));
    HIP_CHECK(hipStreamDestroy(strm));
    if (*Ptr == 16) {
      // exit() with code 10 which indicates pass
      HIP_CHECK(hipHostFree(Ptr));
      exit(10);
    } else {
      // exit() with code 9 which indicates fail
      HIP_CHECK(hipHostFree(Ptr));
      exit(9);
    }
  } else {
    wait(&stat);
    int Result = WEXITSTATUS(stat);
    if (Result != 10) {
      REQUIRE(false);
    }
  }
}
#endif


/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {  // child process
    CHECK_PCIE_ATOMIC_SUPPORT;
    int *Ptr = nullptr, SIZE = sizeof(int);
    bool HmmMem = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE));
    auto ret = TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipHostFree(Ptr));
    exit(ret ? EXIT_SUCCESS : EXIT_FAILURE);
  } else {  // parent process
    wait(&stat);
    if (WEXITSTATUS(stat) != EXIT_SUCCESS) {
      REQUIRE(false);
    }
  }
}
#endif


/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg1") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {  // child process
    CHECK_PCIE_ATOMIC_SUPPORT
    int *Ptr = nullptr, SIZE = sizeof(int);
    bool HmmMem = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocPortable));
    auto ret = TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipHostFree(Ptr));
    exit(ret ? EXIT_SUCCESS : EXIT_FAILURE);
  } else {  // parent process
    wait(&stat);
    if (WEXITSTATUS(stat) != EXIT_SUCCESS) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg2") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {  // child process
    CHECK_PCIE_ATOMIC_SUPPORT
    int *Ptr = nullptr, SIZE = sizeof(int);
    bool HmmMem = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocWriteCombined));
    auto ret = TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipHostFree(Ptr));
    exit(ret ? EXIT_SUCCESS : EXIT_FAILURE);
  } else {  // parent process
    wait(&stat);
    if (WEXITSTATUS(stat) != EXIT_SUCCESS) {
      REQUIRE(false);
    }
  }
}
#endif

/* Test Case Description: The following test checks if the memory exhibits
   fine grain behavior when HIP_HOST_COHERENT is set to 1*/
// The following test is AMD specific test hence skipping for Nvidia
#if HT_AMD
TEST_CASE("Unit_hipHostMalloc_WthEnv1Flg3") {
  if ((setenv("HIP_HOST_COHERENT", "1", 1)) != 0) {
    WARN("Unable to turn on HIP_HOST_COHERENT, hence terminating the Test case!");
    REQUIRE(false);
  }
  int stat = 0;
  if (fork() == 0) {  // child process
    CHECK_PCIE_ATOMIC_SUPPORT
    int *Ptr = nullptr, SIZE = sizeof(int);
    bool HmmMem = false;
    // Allocating hipHostMalloc() memory
    HIP_CHECK(hipHostMalloc(&Ptr, SIZE, hipHostMallocNumaUser));
    auto ret = TstCoherency(Ptr, HmmMem);
    HIP_CHECK(hipHostFree(Ptr));
    exit(ret ? EXIT_SUCCESS : EXIT_FAILURE);
  } else {  // parent process
    wait(&stat);
    if (WEXITSTATUS(stat) != EXIT_SUCCESS) {
      REQUIRE(false);
    }
  }
}
#endif
#endif
