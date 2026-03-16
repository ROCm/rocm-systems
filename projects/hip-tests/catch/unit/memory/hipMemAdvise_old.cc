/*
Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* Test Case Description:
   Scenario-1: The following Function Tests the working of flags which can be
   assigned to HMM memory using hipMemAdvise() api
   Scenario-2: Negative tests on hipMemAdvise() api
   Scenario-3: The following function tests various scenarios around the flag
   'hipMemAdviseSetPreferredLocation' using HMM memory and hipMemAdvise() api
   Scenario-4: The following function tests various scenarios around the flag
   'hipMemAdviseSetReadMostly' using HMM memory and hipMemAdvise() api
   Scenario-5: The following function verifies if assigning of a flag
   invalidates the earlier flag which was assigned to the same memory region
   using hipMemAdvise()
   Scenario-6: The following function tests if peers can set
   hipMemAdviseSetAccessedBy flag
   on HMM memory prefetched on each of the other gpus
   Scenario-7: Set AccessedBy flag and check value returned by
   hipMemRangeGetAttribute() It should be -2(same is observed on cuda)
   Scenario-8: Set AccessedBy flag to device 0 on Hmm memory and prefetch the
   memory to device 1, then probe for AccessedBy flag using
   hipMemRangeGetAttribute() we should still see the said flag is set for
   device 0
   Scenario-9: 1) Set AccessedBy to device 0 followed by PreferredLocation to
   device 1 check for AccessedBy flag using hipMemRangeGetAttribute() it should
   return 0
   2) Unset AccessedBy to 0 and set it to device 1 followed by
   PreferredLocation to device 1, check for AccessedBy flag using
   hipMemRangeGetAttribute() it should return 1
   Scenario-10: Set AccessedBy flag to HMM memory launch a kernel and then unset
   AccessedBy, launch kernel. We should not have any access issues
   Scenario-11: Allocate memory using aligned_alloc(), assign PreferredLocation
   flag to the allocated memory and launch a kernel. Kernel should get executed
   successfully without hang or segfault
   Scenario-12: Allocate Hmm memory, set advise to PreferredLocation and then
   get attribute using the api hipMemRangeGetAttribute() for
   hipMemRangeAttributeLastPrefetchLocation the value returned should be -2
   Scenario-13: Allocate HMM memory, set PreferredLocation to device 0, Prfetch
   the mem to device1, probe for hipMemRangeAttributeLastPrefetchLocation using
   hipMemRangeGetAttribute(), we should get 1
   Scenario-14: Allocate HMM memory, set ReadMostly followed by
   PreferredLocation, probe for hipMemRangeAttributeReadMostly and
   hipMemRangeAttributePreferredLocation
   using hipMemRangeGetAttribute() we should observe 1 and 0 correspondingly.
   In other words setting of hipMemRangeAttributePreferredLocation should not
   impact hipMemRangeAttributeReadMostly advise to the memory
   Scenario-15: Allocate Hmm memory, advise it to ReadMostly for gpu: 0 and
   launch kernel on all other gpus except 0. This test case may discover any
   effect or access denial case arising due to setting ReadMostly only to a
   particular gpu
*/

#include <hip_test_common.hh>
#include <hip_test_features.hh>
#include <hip_test_process.hh>

#if __linux__
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#endif

// Kernel function
__global__ void MemAdvseKernel(int n, int* x) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) x[index] = x[index] * x[index];
}

// Kernel
__global__ void MemAdvise2(int* Hmm, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Hmm[i] = Hmm[i] + 10;
  }
}

// Kernel
__global__ void MemAdvise3(int* Hmm, int* Hmm1, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    Hmm1[i] = Hmm[i] + 10;
  }
}

static int HmmAttrPrint() {
  int managed = 0;
  WARN(
      "The following are the attribute values related to HMM for"
      " device 0:\n");
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeDirectManagedMemAccessFromHost, 0));
  WARN("hipDeviceAttributeDirectManagedMemAccessFromHost: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeConcurrentManagedAccess, 0));
  WARN("hipDeviceAttributeConcurrentManagedAccess: " << managed);
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributePageableMemoryAccess, 0));
  WARN("hipDeviceAttributePageableMemoryAccess: " << managed);
  HIP_CHECK(
      hipDeviceGetAttribute(&managed, hipDeviceAttributePageableMemoryAccessUsesHostPageTables, 0));
  WARN("hipDeviceAttributePageableMemoryAccessUsesHostPageTables:" << managed);

  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  WARN("hipDeviceAttributeManagedMemory: " << managed);
  return managed;
}

// The following function tests if peers can set hipMemAdviseSetAccessedBy flag
// on HMM memory prefetched on each of the other gpus
#if HT_AMD
TEST_CASE(Unit_hipMemAdvise_TstAccessedByPeer) {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
    bool IfTestPassed = true;
    int *Hmm = nullptr, MEM_SIZE = 4 * 4096, A_CONST = 9999;
    ;
    int NumDevs = 0, CanAccessPeer = A_CONST, flag = 0;

    HIP_CHECK(hipGetDeviceCount(&NumDevs));
    if (NumDevs < 2) {
      SUCCEED(
          "Test TestSetAccessedByPeer() need atleast two Gpus to test"
          " the scenario. This system has GPUs less than 2");
    }
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SIZE, hipMemAttachGlobal));
    for (int i = 0; i < NumDevs; ++i) {
      HIP_CHECK(hipMemPrefetchAsync(Hmm, MEM_SIZE, i, 0));
      for (int j = 0; j < NumDevs; ++j) {
        if (i == j) continue;
        HIP_CHECK(hipSetDevice(j));
        HIP_CHECK(hipDeviceCanAccessPeer(&CanAccessPeer, j, i));
        if (CanAccessPeer) {
          HIP_CHECK(hipMemAdvise(Hmm, MEM_SIZE, hipMemAdviseSetAccessedBy, j));
          for (uint64_t m = 0; m < (MEM_SIZE / sizeof(int)); ++m) {
            Hmm[m] = 4;
          }
          HIP_CHECK(hipDeviceEnablePeerAccess(i, 0));
          MemAdvseKernel<<<(MEM_SIZE / sizeof(int) / 32), 32>>>((MEM_SIZE / sizeof(int)), Hmm);
          HIP_CHECK(hipDeviceSynchronize());
          // Verifying the result
          for (uint64_t m = 0; m < (MEM_SIZE / sizeof(int)); ++m) {
            if (Hmm[m] != 16) {
              flag = 1;
            }
          }
          if (flag) {
            WARN("Didnt get Expected results with device: " << j);
            WARN("line no.: " << __LINE__);
            IfTestPassed = false;
            flag = 0;
          }
        }
      }
    }
    HIP_CHECK(hipFree(Hmm));
    REQUIRE(IfTestPassed);
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}
#endif

/* Set AccessedBy flag to device 0 on Hmm memory and prefetch the memory to
   device 1, then probe for AccessedBy flag using hipMemRangeGetAttribute()
   we should still see the said flag is set for device 0*/
TEST_CASE(Unit_hipMemAdvise_TstAccessedByFlg2) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999, Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus >= 2) {
      hipStream_t strm;
      HIP_CHECK(hipStreamCreate(&strm));
      HIP_CHECK(hipMallocManaged(&Hmm, 2 * 4096));
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(hipMemPrefetchAsync(Hmm, 2 * 4096, 1, strm));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int), hipMemRangeAttributeAccessedBy, Hmm,
                                        2 * 4096));
      if (data != 0) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseUnsetAccessedBy, 0));
      HIP_CHECK(hipStreamDestroy(strm));
      HIP_CHECK(hipFree(Hmm));
    }
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* 1) Set AccessedBy to device 0 followed by PreferredLocation to device 1
   check for AccessedBy flag using hipMemRangeGetAttribute() it should
   return 0
   2) Unset AccessedBy to 0 and set it to device 1 followed by
   PreferredLocation to device 1, check for AccessedBy flag using
   hipMemRangeGetAttribute() it should return 1*/

TEST_CASE(Unit_hipMemAdvise_TstAccessedByFlg3) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, data = 999, Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus >= 2) {
      HIP_CHECK(hipMallocManaged(&Hmm, 2 * 4096));
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseSetPreferredLocation, 1));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int), hipMemRangeAttributeAccessedBy, Hmm,
                                        2 * 4096));
      if (data != 0) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseUnsetAccessedBy, 0));
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseSetAccessedBy, 1));
      HIP_CHECK(hipMemAdvise(Hmm, 2 * 4096, hipMemAdviseSetPreferredLocation, 0));
      HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int), hipMemRangeAttributeAccessedBy, Hmm,
                                        2 * 4096));
      if (data != 1) {
        WARN("Didnt get expected behavior at line: " << __LINE__);
        REQUIRE(false);
      }
      HIP_CHECK(hipFree(Hmm));
    }
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}


/* Set AccessedBy flag to HMM memory launch a kernel and then unset
   AccessedBy, launch kernel. We should not have any access issues*/

TEST_CASE(Unit_hipMemAdvise_TstAccessedByFlg4) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, NumElms = (1024 * 1024), InitVal = 123, blockSize = 1024;
    hipStream_t strm;
    HIP_CHECK(hipStreamCreate(&strm));
    HIP_CHECK(hipMallocManaged(&Hmm, (NumElms * sizeof(int))));
    // Initializing memory
    for (int i = 0; i < NumElms; ++i) {
      Hmm[i] = InitVal;
    }
    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)), hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(hipMemPrefetchAsync(Hmm, (NumElms * sizeof(int)), 0, strm));
    HIP_CHECK(hipDeviceSynchronize());
    // launching kernel from each one of the gpus
    MemAdvise2<<<1024, 1024, 0, strm>>>(Hmm, NumElms);
    HIP_CHECK(hipDeviceSynchronize());

    // verifying the final result
    for (int i = 0; i < NumElms; ++i) {
      INFO("index: " << i << " Hmm[i]: " << Hmm[i] << " Expected: " << (InitVal + 10));
      REQUIRE(Hmm[i] == (InitVal + 10));
    }

    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)), hipMemAdviseUnsetAccessedBy, 0));
    HIP_CHECK(hipDeviceSynchronize());
    MemAdvise2<<<1024, 1024, 0, strm>>>(Hmm, NumElms);
    HIP_CHECK(hipDeviceSynchronize());
    // verifying the final result
    for (int i = 0; i < NumElms; ++i) {
      INFO("index: " << i << " Hmm[i]: " << Hmm[i] << " Expected: " << (InitVal + 20));
      REQUIRE(Hmm[i] == (InitVal + 20));
    }

    HIP_CHECK(hipFree(Hmm));
    HIP_CHECK(hipStreamDestroy(strm));
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* Allocate memory using aligned_alloc(), assign PreferredLocation flag to
   the allocated memory and launch a kernel. Kernel should get executed
   successfully without hang or segfault*/
#if __linux__ && HT_AMD
TEST_CASE(Unit_hipMemAdvise_TstAlignedAllocMem) {
  // The following code block checks for xnack+
  // so as to skip if the device is not xnack+
  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);

  if (gfxName.find("xnack+") != std::string::npos) {
    int managedMem = 0, pageMemAccess = 0;
    HIP_CHECK(hipDeviceGetAttribute(&pageMemAccess, hipDeviceAttributePageableMemoryAccess, 0));
    WARN("hipDeviceAttributePageableMemoryAccess:" << pageMemAccess);
    HIP_CHECK(hipDeviceGetAttribute(&managedMem, hipDeviceAttributeManagedMemory, 0));
    WARN("hipDeviceAttributeManagedMemory: " << managedMem);
    if ((managedMem == 1) && (pageMemAccess == 1)) {
      int *Mllc = nullptr, MemSz = 4096 * 4, NumElms = 4096, InitVal = 123;
      // Mllc = reinterpret_cast<(int *)>(aligned_alloc(4096, MemSz));
      Mllc = reinterpret_cast<int*>(aligned_alloc(4096, 4096 * 4));
      for (int i = 0; i < NumElms; ++i) {
        Mllc[i] = InitVal;
      }
      hipStream_t strm;
      int DataMismatch = 0;
      HIP_CHECK(hipStreamCreate(&strm));
      // The following hipMemAdvise() call is made to know if advise on
      // aligned_alloc() is causing any issue
      HIP_CHECK(hipMemAdvise(Mllc, MemSz, hipMemAdviseSetPreferredLocation, 0));
      HIP_CHECK(hipMemPrefetchAsync(Mllc, MemSz, 0, strm));
      HIP_CHECK(hipStreamSynchronize(strm));
      MemAdvise2<<<4, 1024, 0, strm>>>(Mllc, NumElms);
      HIP_CHECK(hipStreamSynchronize(strm));
      for (int i = 0; i < NumElms; ++i) {
        if (Mllc[i] != (InitVal + 10)) {
          DataMismatch++;
        }
      }
      REQUIRE(DataMismatch == 0);
      free(Mllc);
      HIP_CHECK(hipStreamDestroy(strm));
    }
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test");
  }
}

TEST_CASE(Unit_hipMemAdvise_TstAlignedAllocMem_XNACK) {
  if (setenv("HSA_XNACK", "1", 1) != 0) {
    HipTest::HIP_SKIP_TEST("Unable to set xnack on environment variable.");
    return;
  }

  hipDeviceProp_t prop;
  int device;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&prop, device));
  std::string gfxName(prop.gcnArchName);

  if (gfxName.find("xnack+") != std::string::npos) {
    hip::SpawnProc proc("hipMemAdviseTstAlignedAllocMem", true);
    REQUIRE(proc.run() == 0);
  } else {
    HipTest::HIP_SKIP_TEST("GPU is not xnack enabled hence skipping the test");
  }
}
#endif


/*Allocate Hmm memory, advise it to ReadMostly for gpu: 0 and launch kernel
  on all other gpus except 0. This test case may discover any effect or
  access denial case arising due to setting ReadMostly only to a particular
  gpu*/

TEST_CASE(Unit_hipMemAdvise_ReadMosltyMgpuTst) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    if (Ngpus < 2) {
      SUCCEED(
          "This test needs atleast two gpus to run."
          "Hence skipping the test.\n");
    }
    int *Hmm = NULL, NumElms = (1024 * 1024), InitVal = 123;
    int *Hmm1 = NULL, DataMismatch = 0;
    hipStream_t strm;
    HIP_CHECK(hipMallocManaged(&Hmm, (NumElms * sizeof(int))));
    // Initializing memory
    for (int i = 0; i < NumElms; ++i) {
      Hmm[i] = InitVal;
    }
    HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)), hipMemAdviseSetReadMostly, 0));
#if HT_AMD
    SECTION("Launch Kernel on all other gpus") {
      // launching kernel from each one of the gpus
      for (int i = 1; i < Ngpus; ++i) {
        DataMismatch = 0;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&strm));
        HIP_CHECK(hipMallocManaged(&Hmm1, (NumElms * sizeof(int))));
        MemAdvise3<<<1024, 1024, 0, strm>>>(Hmm, Hmm1, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        // verifying the results
        for (int j = 0; j < NumElms; ++j) {
          if (Hmm1[j] != (InitVal + 10)) {
            DataMismatch++;
          }
        }
        if (DataMismatch != 0) {
          WARN("DataMismatch is observed with the gpu: " << i);
          REQUIRE(false);
        }
        HIP_CHECK(hipStreamDestroy(strm));
        HIP_CHECK(hipFree(Hmm1));
      }
    }

    SECTION("Launch Kernel on all other gpus and manipulate the content") {
      for (int i = 0; i < Ngpus; ++i) {
        DataMismatch = 0;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&strm));
        HIP_CHECK(hipMemAdvise(Hmm, (NumElms * sizeof(int)), hipMemAdviseSetReadMostly, i));
        MemAdvise2<<<1024, 1024, 0, strm>>>(Hmm, NumElms);
        HIP_CHECK(hipStreamSynchronize(strm));
        HIP_CHECK(hipStreamDestroy(strm));
      }
      // verifying the final result
      for (int i = 0; i < NumElms; ++i) {
        if (Hmm[i] != (InitVal + Ngpus * 10)) {
          DataMismatch++;
        }
      }

      if (DataMismatch != 0) {
        WARN("DataMismatch is observed at line: " << __LINE__);
        REQUIRE(false);
      }
    }
#endif
    HIP_CHECK(hipFree(Hmm));

  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}
