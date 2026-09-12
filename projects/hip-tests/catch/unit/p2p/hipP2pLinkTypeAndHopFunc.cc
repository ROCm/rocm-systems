/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_kernels.hh>
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

#ifdef __linux__
#include <unistd.h>
#include <sys/wait.h>
#endif
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
    _exit(0);
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
    _exit(0);

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

 * Test source
 * ------------------------
 *    - catch/unit/p2p/hipExtGetLinkTypeAndHopCount.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.5
 */

HIP_TEST_CASE(Unit_hipP2pLinkTypeAndHopFunc) {
  int numDevices = 0;
  bool TestPassed = true;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }
  SECTION("Test running for testhipInvalidDevice") {
    TestPassed = testhipInvalidDevice(numDevices);
    REQUIRE(TestPassed == true);
  }
#ifdef __linux__
  getDeviceCount(&numDevices);
  if (numDevices < 2) {
    WARN("Skipping Linux-only P2P sections: " << HipTest::SkipReason::kFewerThanTwoGpus);
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
#else
  WARN("Skipping Linux-only P2P link scenarios: " << HipTest::SkipReason::kRequiresLinux);
#endif
}

/**
 * End doxygen group p2pTest.
 * @}
 */
