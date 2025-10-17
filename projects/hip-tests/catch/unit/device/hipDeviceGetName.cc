/*
Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <array>
#include <algorithm>
#include <iterator>
#include <map>

/**
 * @addtogroup hipDeviceGetName hipDeviceGetName
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetName(char* name, int len, hipDevice_t device)` -
 * Returns an identifer string for the device.
 */

constexpr size_t LEN = 256;

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# Valid devices and output pointer to the name is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Valid devices and output name buffer length is 0
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Valid devices and output name buffer has length -1
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# Invalid devices, device ordinal is out of bounds
 *      - Expected output: return `hipErrorInvalidDevice`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetName_NegTst") {
  std::array<char, LEN> name;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  std::vector<hipDevice_t> devices(numDevices);
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&devices[i], i));
  }

  SECTION("Valid Device") {
    const auto device = GENERATE_COPY(from_range(std::begin(devices), std::end(devices)));

    SECTION("Nullptr for name argument") {
      // Scenario2
      HIP_CHECK_ERROR(hipDeviceGetName(nullptr, name.size(), device), hipErrorInvalidValue);
    }
#if HT_AMD
    // These test scenarios fail on NVIDIA.
    SECTION("Zero name length") {
      // Scenario3
      HIP_CHECK_ERROR(hipDeviceGetName(name.data(), 0, device), hipErrorInvalidValue);
    }

    SECTION("Negative name length") {
      // Scenario4
      HIP_CHECK_ERROR(hipDeviceGetName(name.data(), -1, device), hipErrorInvalidValue);
    }
#endif
  }
  SECTION("Invalid Device") {
    hipDevice_t badDevice = devices.back() + 1;

    constexpr size_t timeout = 100;
    size_t timeoutCount = 0;
    while (std::find(std::begin(devices), std::end(devices), badDevice) != std::end(devices)) {
      badDevice += 1;
      timeoutCount += 1;
      REQUIRE(timeoutCount < timeout);  // give up after a while
    }

    // Scenario5
    HIP_CHECK_ERROR(hipDeviceGetName(name.data(), name.size(), badDevice), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Get the device name for each device.
 *  - Compare the name with the name returned from device properties.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetName_CheckPropName") {
  int numDevices = 0;
  std::array<char, LEN> name;
  hipDevice_t device;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  for (int i = 0; i < numDevices; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(hipDeviceGetName(name.data(), name.size(), device));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));

    // Scenario1
    REQUIRE(strncmp(name.data(), prop.name, name.size()) == 0);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Set name buffer length to the half of the name length.
 *  - Check that device name is successfuly returned.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetName_PartialFill") {
#if HT_AMD
  HipTest::HIP_SKIP_TEST("EXSWCPHIPT-108");
  return;
#endif
  std::array<char, LEN> name;

  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  auto ordinal = GENERATE_COPY(range(0, numDevices));
  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, ordinal));
  HIP_CHECK(hipDeviceGetName(name.data(), name.size(), device));

  auto start = std::begin(name);
  auto end = std::end(name);
  const auto len = std::distance(start, std::find(start, end, 0));

  // fill up only half of the length
  const auto fillLen = len / 2;
  constexpr char fillValue = 1;
  std::fill(start, end, fillValue);

  // Scenario6
  HIP_CHECK(hipDeviceGetName(name.data(), fillLen, device));

  const auto strEnd = start + fillLen - 1;
  REQUIRE(std::all_of(start, strEnd, [](char& c) { return c != 0; }));
  REQUIRE(*strEnd == 0);
  REQUIRE(std::all_of(strEnd + 1, end, [](char& c) { return c == fillValue; }));
}

#ifdef __linux__
#if HT_AMD
#define BUFFER_LEN 512

static inline std::vector<int> parseVisibleDevices() {
  std::vector<int> res;
  auto env_res = std::getenv("HIP_VISIBLE_DEVICES");
  if (env_res == nullptr) {
    env_res = std::getenv("ROCR_VISIBLE_DEVICES");
    if (env_res == nullptr) {
      return res;
    }
  }

  std::stringstream ss(std::string{env_res});
  std::string item;
  while (std::getline(ss, item, ',')) {
    res.push_back(std::stoi(item));
  }

  return res;
}

/**
 * Test Description
 * ------------------------
 *  - Get the GPU name from rocm_agent_enumerator and
      compare it with the value from hipGetDeviceProperties gcnArchName
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetName.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipDeviceName_gcnArchName_And_rocm_agent_enumerator") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  if (deviceCount <= 0) {
    HipTest::HIP_SKIP_TEST("No device found, skipping the test.");
    return;
  }

  FILE* fpipe;
  fpipe = popen("rocm_agent_enumerator", "r");
  if (fpipe == nullptr) {
    HipTest::HIP_SKIP_TEST("Unable to create command file.\n");
    return;
  }
  char command_op[BUFFER_LEN];
  const char* defCpu = "gfx000";
  int j = 0;
  std::map<int, std::vector<char>> dNameMap;
  while (fgets(command_op, BUFFER_LEN, fpipe)) {
    command_op[strcspn(command_op, "\n")] = '\0';
    std::string rocmCommand_line(command_op);
    int dNameLen = strlen(rocmCommand_line.c_str());
    if (std::string::npos != rocmCommand_line.find(defCpu)) {  // ignore CPU
      continue;
    } else {
      std::vector<char> dName(dNameLen + 1, 0);
      std::memcpy(dName.data(), &rocmCommand_line[0], dNameLen);
      dNameMap[j] = dName;
    }
    j++;
  }

  auto visible_devices = parseVisibleDevices();
  if (visible_devices.size() > 0) {
    // We have visible devices set, basically parse the visible devices and remove the entries
    size_t start = 0;  // The devices will be reported from 0..
    std::map<int, std::vector<char>> dNameMapCopy;
    for (auto device : visible_devices) {
      dNameMapCopy[start] = dNameMap[device];
    }
    dNameMap = dNameMapCopy;
  }

  for (const auto& i : dNameMap) {
    if (i.second.size() == 0) {
      continue;
    }
    auto dev = i.first;
    HIP_CHECK(hipSetDevice(dev));
    hipDevice_t device;
    hipDeviceProp_t prop;
    HIP_CHECK(hipDeviceGet(&device, dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, device));
    REQUIRE(strncmp(i.second.data(), prop.gcnArchName, strlen(i.second.data())) == 0);
  }
}
#endif
#endif

/**
 * End doxygen group DriverTest.
 * @}
 */
