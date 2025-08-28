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

#include <hip_test_common.hh>
#include <hip_test_process.hh>

/**
 * @addtogroup hipGetDeviceCount hipGetDeviceCount
 * @{
 * @ingroup DeviceTest
 * `hipGetDeviceCount(int* count)` -
 * Return number of compute-capable devices.
 */

/**
 * Test Description
 * ------------------------
 *  - Passes invalid pointer as output parameter for device count - `nullptr`
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetDeviceCount_NegTst") {
  REQUIRE_FALSE(hipGetDeviceCount(nullptr) == hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Validates correct functionality when the device visibility
 *    environment variables are set. Uses unit/device/hipDeviceCount_exe.cc
 *    to set visibility.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetDeviceCount_HideDevices") {
  int deviceCount = HipTest::getDeviceCount();
  if (deviceCount < 2) {
    HipTest::HIP_SKIP_TEST("This test requires more than 2 GPUs. Skipping.");
    return;
  }

  for (int i = deviceCount; i >= 1; i--) {
    std::string visibleStr;
    for (int j = 0; j < i; j++) {  // Generate a string which has first i devices
      visibleStr += std::to_string(j);
      if (j != (i - 1)) {
        visibleStr += ",";
      }
    }

    hip::SpawnProc proc("getDeviceCount", true);
    INFO("Output from process : " << proc.getOutput());
    REQUIRE(proc.run(visibleStr) == i);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Prints device count to the standard output.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceCount.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Print_Out_Device_Count") {
  std::cout << "Device Count: " << HipTest::getDeviceCount() << std::endl;
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
