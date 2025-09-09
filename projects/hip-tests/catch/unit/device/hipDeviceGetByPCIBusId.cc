/*
 * Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */


#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceGetByPCIBusId hipDeviceGetByPCIBusId
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetByPCIBusId(int* device, const char* pciBusId)` -
 * Returns a handle to a compute device.
 */

#define SIZE 20

/**
 * Test Description
 * ------------------------
 *  - Check that PCI bus ID is the same as the one returned from attributes.
 *  - Perform for each device.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetByPCIBusId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetByPCIBusId_Functional") {
  char pciBusId[SIZE]{};
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  HIP_ASSERT(deviceCount != 0);
  for (int i = 0; i < deviceCount; i++) {
    int pciBusID = -1;
    int pciDeviceID = -1;
    int pciDomainID = -1;
    int tempPciBusId = -1;
    int tempDeviceId = -1;
    int pciFunction = -1;
    HIP_CHECK(hipDeviceGetPCIBusId(&pciBusId[0], SIZE, i));
    REQUIRE(sscanf(pciBusId, "%x:%x:%x.%x", &pciDomainID, &pciBusID, &pciDeviceID, &pciFunction) ==
            4);
    HIP_CHECK(hipDeviceGetAttribute(&tempPciBusId, hipDeviceAttributePciBusId, i));

    REQUIRE(pciBusID == tempPciBusId);
    HIP_CHECK(hipDeviceGetByPCIBusId(&tempDeviceId, pciBusId));
    REQUIRE(tempDeviceId == i);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of `nullptr` arguments:
 *    -# When the output pointer to the device is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When the PCI bus ID pointer is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetByPCIBusId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetByPCIBusId_NegativeNullChk") {
  int device = -1;
  hipError_t ret;
  char pciBusIdstr[SIZE]{};
  ret = hipDeviceGetByPCIBusId(nullptr, pciBusIdstr);
  CHECK(ret != hipSuccess);

  ret = hipDeviceGetByPCIBusId(&device, nullptr);
  CHECK(ret != hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid PCI bus ID strings:
 *    -# When PCI bus ID is an empty string
 *      - Expected output: do not return `hipSuccess`
 *    -# When PCI bus ID is a short string
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetByPCIBusId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetByPCIBusId_NegativeInputString") {
  int device = -1;
  hipError_t ret;
  ret = hipDeviceGetByPCIBusId(&device, "");
  CHECK(ret != hipSuccess);

  ret = hipDeviceGetByPCIBusId(&device, "0000:");
  CHECK(ret != hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid PCI bus ID values:
 *    -# Passes non-existing PCI bus ID
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetByPCIBusId.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetByPCIBusId_WrongBusID") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  HIP_ASSERT(deviceCount != 0);
  constexpr int MaxIter = 256;
  constexpr int MaxBusIdLen = 12;
  std::vector<int> pciBusId;
  std::vector<int> pciDeviceID;
  std::vector<int> pciDomainID;

  // get bus id of all the devices
  for (int i = 0; i < deviceCount; i++) {
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, i));
    pciBusId.push_back(prop.pciBusID);
    pciDeviceID.push_back(prop.pciDeviceID);
    pciDomainID.push_back(prop.pciDomainID);
    printf("device %d: pciDomainID=%x, pciBusID=%x, pciDeviceID=%x \n", i, prop.pciDomainID,
           prop.pciBusID, prop.pciDomainID);
  }
  // get a non existing bus id
  int id = 0;
  for (; id < MaxIter; id++) {
    bool bFound = false;
    // check if id is the pci busid of any existing device
    for (int j = 0; j < deviceCount; j++) {
      if (id == pciBusId[j]) {
        bFound = true;
        break;
      }
    }
    if (!bFound) break;
  }
  // now pass the non existing bus id as string
  char pciBusIdstr[MaxBusIdLen];
  int device = -1;
  hipError_t ret;
  snprintf(pciBusIdstr, sizeof(pciBusIdstr), "%04x:%02x:%02x", pciDomainID[0], id, pciDeviceID[0]);
  ret = hipDeviceGetByPCIBusId(&device, pciBusIdstr);
  REQUIRE(ret != hipSuccess);
}


/**
 * End doxygen group DriverTest.
 * @}
 */
