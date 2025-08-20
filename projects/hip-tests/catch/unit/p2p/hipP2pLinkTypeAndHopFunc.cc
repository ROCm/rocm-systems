/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hipP2pLinkTypeAndHopFunc.h"
#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#include <dlfcn.h>
#endif
#include <vector>
#define MAX_SIZE 30
#define VISIBLE_DEVICE 0

/**
 * Fetches Gpu device count
 */
#ifdef __linux__
void getDeviceCount(int* pdevCnt) {
  int fd[2], val = 0;
  pid_t childpid;
  // create pipe descriptors
  pipe(fd);
  // disable visible_devices env from shell
  unsetenv("ROCR_VISIBLE_DEVICES");
  unsetenv("HIP_VISIBLE_DEVICES");

  childpid = fork();
  if (childpid > 0) {  // Parent
    close(fd[1]);
    // parent will wait to read the device cnt
    read(fd[0], &val, sizeof(val));
    // close the read-descriptor
    close(fd[0]);
    // wait for child exit
    wait(NULL);
    *pdevCnt = val;
  } else if (!childpid) {  // Child
    int devCnt = 1;
    // writing only, no need for read-descriptor
    close(fd[0]);
    HIP_CHECK(hipGetDeviceCount(&devCnt));
    // send the value on the write-descriptor:
    write(fd[1], &devCnt, sizeof(devCnt));
    // close the write descriptor:
    close(fd[1]);
    exit(0);
  } else {  // failure
    *pdevCnt = 1;
    return;
  }
}

bool testMaskedDevice(int actualNumGPUs) {
  bool testResult = true;
  int fd[2];
  pipe(fd);

  pid_t cPid;
  cPid = fork();
  if (cPid == 0) {  // child
    hipError_t err;
    char visibleDeviceString[MAX_SIZE] = {};
    snprintf(visibleDeviceString, MAX_SIZE, "%d", VISIBLE_DEVICE);
    // disable visible_devices env from shell
    unsetenv("ROCR_VISIBLE_DEVICES");
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("ROCR_VISIBLE_DEVICES", visibleDeviceString, 1);
    setenv("HIP_VISIBLE_DEVICES", visibleDeviceString, 1);
    uint32_t linktype;
    uint32_t hopcount;
    for (int count = 1; count < actualNumGPUs; count++) {
      err = hipExtGetLinkTypeAndHopCount(VISIBLE_DEVICE, VISIBLE_DEVICE + count, &linktype,
                                         &hopcount);
      REQUIRE(err == hipSuccess);
    }
    close(fd[0]);
    write(fd[1], &testResult, sizeof(testResult));
    close(fd[1]);
    exit(0);

  } else if (cPid > 0) {  // parent
    close(fd[1]);
    read(fd[0], &testResult, sizeof(testResult));
    close(fd[0]);
    wait(NULL);

  } else {
    printf("Info:fork() failed\n");
    testResult = false;
  }
  return testResult;
}
#endif

bool testhipInvalidDevice(int numDevices) {
  hipError_t ret;
  uint32_t linktype;
  uint32_t hopcount;
  SECTION("Invalid device number case 1") {
    ret = hipExtGetLinkTypeAndHopCount(-1, 0, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  SECTION("Invalid device number case 2") {
    ret = hipExtGetLinkTypeAndHopCount(numDevices, 0, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  SECTION("Invalid device number case 3") {
    ret = hipExtGetLinkTypeAndHopCount(0, -1, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  SECTION("Invalid device number case 4") {
    ret = hipExtGetLinkTypeAndHopCount(0, numDevices, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  SECTION("Invalid device number case 5") {
    ret = hipExtGetLinkTypeAndHopCount(-1, numDevices, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  return true;
}

#ifdef __linux__
bool testhipInvalidLinkType() {
  uint32_t hopcount;
  REQUIRE(hipSuccess != hipExtGetLinkTypeAndHopCount(0, 1, nullptr, &hopcount));
  return true;
}

bool testhipInvalidHopcount() {
  uint32_t linktype;
  REQUIRE(hipSuccess != hipExtGetLinkTypeAndHopCount(0, 1, &linktype, nullptr));
  return true;
}

bool testhipSameDevice(int numGPUs) {
  hipError_t ret;
  uint32_t linktype = 0;
  uint32_t hopcount = 0;
  for (int gpuId = 0; gpuId < numGPUs; gpuId++) {
    ret = hipExtGetLinkTypeAndHopCount(gpuId, gpuId, &linktype, &hopcount);
    REQUIRE(ret != hipSuccess);
  }
  return true;
}

bool testhipLinkTypeHopcountDeviceOrderRev(int numDevices) {
  bool TestPassed = true;
  // Get the unique pair of devices
  for (int x = 0; x < numDevices; x++) {
    for (int y = x + 1; y < numDevices; y++) {
      uint32_t linktype1 = 0, linktype2 = 0;
      uint32_t hopcount1 = 0, hopcount2 = 0;
      HIP_CHECK(hipExtGetLinkTypeAndHopCount(x, y, &linktype1, &hopcount1));
      HIP_CHECK(hipExtGetLinkTypeAndHopCount(y, x, &linktype2, &hopcount2));
      if (hopcount1 != hopcount2) {
        TestPassed = false;
        break;
      }
    }
  }
  return TestPassed;
}

/**
 * Internal Function
 */
bool validateLinkType(uint32_t linktype_Hip, RSMI_IO_LINK_TYPE linktype_RocmSmi) {
  bool TestPassed = false;

  if ((linktype_Hip == HSA_AMD_LINK_INFO_TYPE_PCIE) &&
      (linktype_RocmSmi == RSMI_IOLINK_TYPE_PCIEXPRESS)) {
    TestPassed = true;
  } else if ((linktype_Hip == HSA_AMD_LINK_INFO_TYPE_XGMI) &&
             (linktype_RocmSmi == RSMI_IOLINK_TYPE_XGMI)) {
    TestPassed = true;
  } else {
    printf("linktype Hip = %u, linktype RocmSmi = %u\n", linktype_Hip, linktype_RocmSmi);
    TestPassed = false;
  }
  return TestPassed;
}

bool testhipLinkTypeHopcountDevice(int numDevices) {
  bool TestPassed = true;
  // Opening and initializing rocm-smi library
  void* lib_rocm_smi_hdl;
  rsmi_status_t (*fntopo_get_link_type)(uint32_t, uint32_t, uint64_t*, RSMI_IO_LINK_TYPE*);
  rsmi_status_t (*fntopo_init)(uint64_t);
  rsmi_status_t (*fntopo_shut_down)();

  lib_rocm_smi_hdl = dlopen("/opt/rocm/lib/librocm_smi64.so", RTLD_LAZY);
  REQUIRE(lib_rocm_smi_hdl);

  void* fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_topo_get_link_type");
  REQUIRE(fnsym);

  fntopo_get_link_type =
      reinterpret_cast<rsmi_status_t (*)(uint32_t, uint32_t, uint64_t*, RSMI_IO_LINK_TYPE*)>(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_init");
  REQUIRE(fnsym);
  fntopo_init = reinterpret_cast<rsmi_status_t (*)(uint64_t)>(fnsym);

  fnsym = dlsym(lib_rocm_smi_hdl, "rsmi_shut_down");
  REQUIRE(fnsym);
  fntopo_shut_down = reinterpret_cast<rsmi_status_t (*)()>(fnsym);

  uint64_t init_flags = 0;
  rsmi_status_t retsmi_init;
  retsmi_init = fntopo_init(init_flags);
  REQUIRE(RSMI_STATUS_SUCCESS == retsmi_init);

  // Use rocm-smi API rsmi_topo_get_link_type() to validate
  struct devicePair {
    int device1;
    int device2;
  };
  std::vector<struct devicePair> devicePairList;
  // Get the unique pair of devices
  for (int x = 0; x < numDevices; x++) {
    for (int y = x + 1; y < numDevices; y++) {
      devicePairList.push_back({x, y});
    }
  }
  for (auto pos = devicePairList.begin(); pos != devicePairList.end(); pos++) {
    int can_access_peer = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&can_access_peer, (*pos).device1, (*pos).device2));
    if (!can_access_peer) {
      continue;
    }
    uint32_t linktype1 = 0;
    uint32_t hopcount1 = 0;
    RSMI_IO_LINK_TYPE linktype2 = RSMI_IOLINK_TYPE_UNDEFINED;
    uint64_t hopcount2 = 0;
    rsmi_status_t retsmi;
    HIPCHECK(hipExtGetLinkTypeAndHopCount((*pos).device1, (*pos).device2, &linktype1, &hopcount1));
    retsmi = fntopo_get_link_type((*pos).device1, (*pos).device2, &hopcount2, &linktype2);
    REQUIRE(RSMI_STATUS_SUCCESS == retsmi);

    // Validate linktype
    TestPassed = validateLinkType(linktype1, linktype2);
  }
  fntopo_shut_down();
  dlclose(lib_rocm_smi_hdl);
  return TestPassed;
}
#endif

/**
 * @addtogroup hipExtGetLinkTypeAndHopCount hipExtGetLinkTypeAndHopCount
 * @{
 * @ingroup p2pTest
 * `hipError_t hipExtGetLinkTypeAndHopCount(int device1, int device2, uint32_t* linktype, uint32_t*
 * hopcount)` - Returns the link type and hop count between two devices
 * @}
 */

/**
 * Test Description
 * ------------------------
 *    - Validates negative scenarios for hipExtGetLinkTypeAndHopCount
 * 1)Test Scenario to verify when device1 is visible and device2 is masked
 * 2)Test Scenario to verify Invalid Device Number(s)
 * 3)Test Scenario to verify when linktype = NULL
 * 4)Test Scenario to verify when hopcount = NULL
 * 5)Test Scenario to verify when device1 = device2
 * 6)Test Scenario: Verify (hopcount, linktype) values for (src= device1, dest = device2)
 * and (src = device2, dest = device1), where device1 and device2 are valid device numbers.
 * 7)Test Scenario: Verify (hopcount, linktype) values for all combination of
 * GPUs with the output of rocm_smi tool.

 * Test source
 * ------------------------
 *    - catch/unit/p2p/hipExtGetLinkTypeAndHopCount.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

TEST_CASE("Unit_hipP2pLinkTypeAndHopFunc") {
  int numDevices = 0;
  bool TestPassed = true;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  SECTION("Test running for testhipInvalidDevice") {
    TestPassed = testhipInvalidDevice(numDevices);
    REQUIRE(TestPassed == true);
  }
#ifdef __linux__
  getDeviceCount(&numDevices);
  if (numDevices < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  SECTION("Test running for testMaskedDevice") {
    TestPassed = testMaskedDevice(numDevices);
    REQUIRE(TestPassed == true);
  }
  SECTION("Test running for testhipInvalidLinkType") {
    TestPassed = testhipInvalidLinkType();
    REQUIRE(TestPassed == true);
  }
  SECTION("Test running for testhipInvalidHopcount") {
    TestPassed = testhipInvalidHopcount();
    REQUIRE(TestPassed == true);
  }
  SECTION("Test running for testhipSameDevice") {
    TestPassed = testhipSameDevice(numDevices);
    REQUIRE(TestPassed == true);
  }
  SECTION("Test running for testhipLinkTypeHopcountDeviceOrderRev") {
    TestPassed = testhipLinkTypeHopcountDeviceOrderRev(numDevices);
    REQUIRE(TestPassed == true);
  }
  SECTION("Test running for testhipLinkTypeHopcountDevice") {
    TestPassed = testhipLinkTypeHopcountDevice(numDevices);
    REQUIRE(TestPassed == true);
  }
#else
  printf("This test is skipped due to non linux environment.\n");
#endif
}

/**
 * End doxygen group p2pTest.
 * @}
 */
