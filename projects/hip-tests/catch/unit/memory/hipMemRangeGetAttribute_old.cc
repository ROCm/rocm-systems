/*
Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/* Test Case Description:
   Scenario-1: The following function tests the count parameter(last param) to
   hipMemRangeGetAttribute api by passing possible extreme values.

   Scenario-2: This test case checks behavior of hipMemRangeGetAttribute() with
   AccessedBy when the output buffer size is smaller than the number of devices
   potentially returned (should not crash).
*/

#include <hip_test_common.hh>

#include <vector>

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

// The following function tests the count parameter(last param) to
// hipMemRangeGetAttribute api by passing possible extreme values.
// Curently the only way to test if count param working properly is to verify
// the first parameter of hipMemRangeGetAttribute() api has value 1 stored
TEST_CASE(Unit_hipMemRangeGetAttribute_TstCountParam) {
  int MangdMem = HmmAttrPrint();
  if (MangdMem == 1) {
#if HT_AMD
    int isPageableHMM = 0;
    HIP_CHECK(hipDeviceGetAttribute(&isPageableHMM, hipDeviceAttributePageableMemoryAccess, 0));
    if (!isPageableHMM) {
      SUCCEED(
          "Running on a system  where all the memory requested in hipMallocManaged "
          "is allocated on the host.\nThis can cause instability because of out-of-memory "
          "failures.\n"
          "Hence skipping the test with Pass result.\n");
      return;
    }
#endif

    int MEM_SIZE = 4096, RND_NUM = 9999, FLG_READMOSTLY_ENBLD = 1;
    bool IfTestPassed = true;
    int data = RND_NUM, *devPtr = nullptr;
    size_t TotGpuMem, TotGpuFreeMem;
    HIP_CHECK(hipMemGetInfo(&TotGpuFreeMem, &TotGpuMem));

    HIP_CHECK(hipMallocManaged(&devPtr, MEM_SIZE, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, MEM_SIZE, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(reinterpret_cast<void*>(&data), sizeof(int),
                                      hipMemRangeAttributeReadMostly, devPtr, MEM_SIZE));
    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));
    HIP_CHECK(hipMallocManaged(&devPtr, TotGpuFreeMem, hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, TotGpuFreeMem, hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int), hipMemRangeAttributeReadMostly, devPtr,
                                      TotGpuFreeMem));

    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));
    HIP_CHECK(hipMallocManaged(&devPtr, (TotGpuFreeMem - 1), hipMemAttachGlobal));
    HIP_CHECK(hipMemAdvise(devPtr, (TotGpuFreeMem - 1), hipMemAdviseSetReadMostly, 0));
    HIP_CHECK(hipMemRangeGetAttribute(&data, sizeof(int), hipMemRangeAttributeReadMostly, devPtr,
                                      (TotGpuFreeMem - 1)));

    if (data != FLG_READMOSTLY_ENBLD) {
      WARN("hipMemRangeGetAttribute() api didnt return expected value!\n");
      IfTestPassed = false;
    }
    HIP_CHECK(hipFree(devPtr));

    REQUIRE(IfTestPassed);
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* This test case checks the behavior of hipMemRangeGetAttribute() with
   AccessedBy flag is consistent with cuda's counter part*/
TEST_CASE(Unit_hipMemRangeGetAttribute_AccessedBy1) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int Ngpus = 0, *Hmm = NULL, MEM_SZ = 4096, RND_NUM = 999;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    std::vector<int> OutData;
    for (int i = 0; i < Ngpus; ++i) {
      OutData.push_back(RND_NUM);
    }
    HIP_CHECK(hipMallocManaged(&Hmm, MEM_SZ));
    HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ, hipMemAdviseSetAccessedBy, 0));
    HIP_CHECK(hipMemRangeGetAttribute(OutData.data(), sizeof(int) * OutData.size(),
                                      hipMemRangeAttributeAccessedBy, Hmm, MEM_SZ));
    if (OutData[0] != 0) {
      WARN("Didn't receive expected value at line: " << __LINE__);
      REQUIRE(false);
    }
    for (int i = 1; i < Ngpus; ++i) {
      if (OutData[i] != -2) {
        WARN("Didn't receive expected value at line: " << __LINE__);
        REQUIRE(false);
      }
    }
    if (Ngpus >= 2) {
      for (int i = 0; i < Ngpus; ++i) {
        HIP_CHECK(hipMemAdvise(Hmm, MEM_SZ, hipMemAdviseSetAccessedBy, i));
      }
      // checking the behavior with dataSize less than the number of gpus
      // This should not result in segfault.
      HIP_CHECK(hipMemRangeGetAttribute(OutData.data(), sizeof(int) * (OutData.size() - 1),
                                        hipMemRangeAttributeAccessedBy, Hmm, MEM_SZ));
      // OutData should have stored the gpu ordinals for which AccessedBy is
      // assigned except for the last element which should have -2 stored
      // so as to be consistent with cuda's behavior
      for (int i = 0; i < (Ngpus - 1); ++i) {
        if (OutData[i] != i) {
          WARN("Didn't receive expected value at line: " << __LINE__);
          REQUIRE(false);
        }
      }
      if (OutData[Ngpus - 1] != -2) {
        WARN("Didn't receive expected value at line: " << __LINE__);
        REQUIRE(false);
      }
    }
    HIP_CHECK(hipFree(Hmm));
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}

/* The following scenarios tests that probing the attributes which are not set
   by hipMemAdvise() but being probed using hipMemRangeGetAttribute() should
   not result in a crash*/

TEST_CASE(Unit_hipMemRangeGetAttribute_4) {
  int managed = HmmAttrPrint();
  if (managed == 1) {
    int *Hmm = NULL, PageSz = 4096, Ngpus, RND_NUM = 999;
    HIP_CHECK(hipGetDeviceCount(&Ngpus));
    int* OutData = new int[Ngpus];
    for (int i = 0; i < Ngpus; ++i) {
      OutData[i] = RND_NUM;
    }
    HIP_CHECK(hipMallocManaged(&Hmm, 4 * PageSz));
    SECTION("Set ReadMostly & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseSetReadMostly, 0));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4 * Ngpus, hipMemRangeAttributeAccessedBy, Hmm,
                                        4 * PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4, hipMemRangeAttributePreferredLocation, Hmm,
                                        4 * PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseUnsetReadMostly, 0));
    }
    SECTION("Set AccessedBy & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseSetAccessedBy, 0));
      HIP_CHECK(
          hipMemRangeGetAttribute(OutData, 4, hipMemRangeAttributeReadMostly, Hmm, 4 * PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4, hipMemRangeAttributePreferredLocation, Hmm,
                                        4 * PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseUnsetAccessedBy, 0));
    }
    SECTION("Set AccessedBy & probe other flags") {
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseSetPreferredLocation, 0));
      HIP_CHECK(
          hipMemRangeGetAttribute(OutData, 4, hipMemRangeAttributeReadMostly, Hmm, 4 * PageSz));
      HIP_CHECK(hipMemRangeGetAttribute(OutData, 4 * Ngpus, hipMemRangeAttributeAccessedBy, Hmm,
                                        4 * PageSz));
      HIP_CHECK(hipMemAdvise(Hmm, 4 * PageSz, hipMemAdviseUnsetPreferredLocation, 0));
    }
    HIP_CHECK(hipFree(Hmm));
    delete[] OutData;
  } else {
    SUCCEED(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory "
        "attribute. Hence skipping the testing with Pass result.\n");
  }
}
