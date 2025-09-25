/*
Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip_test_process.hh>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN64
#include <windows.h>
#endif
#include <cstring>
#include <cstdio>
#include <map>
#include <sstream>
#include <vector>
#include <thread>  // NOLINT
#include <mutex>   //NOLINT

#ifdef _WIN64
#define setenv(x, y, z) _putenv_s(x, y)
#define unsetenv(x) _putenv(x)
#endif
#define COMMAND_LEN 256
#define BUFFER_LEN 512

std::atomic<int> tState{1};  // 0:fail, 1:pass, 2:skip

/**
 * @addtogroup hipDeviceGetUuid hipDeviceGetUuid
 * @{
 * @ingroup DriverTest
 * `hipDeviceGetUuid(hipUUID* uuid, hipDevice_t device)` -
 * Returns an UUID for the device.[BETA]
 */

/**
 * Test Description
 * ------------------------
 *  - Check that non-empty UUID is returned for each available device.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetUuid_Positive") {
  hipDevice_t device;
  hipUUID uuid{0};
  bool uuidValid = false;

  const int deviceId = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipDeviceGet(&device, deviceId));

  // Scenario 1
  HIP_CHECK(hipDeviceGetUuid(&uuid, device));
  // Atleast one non zero value
  size_t uuidSize = sizeof(uuid.bytes) / sizeof(uuid.bytes[0]);
  for (size_t i = 0; i < uuidSize; i++) {
    if (uuid.bytes[i] != 0) {
      uuidValid = true;
      break;
    }
  }
  REQUIRE(uuidValid == true);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the UUID is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is negative
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ordinal is out of bounds
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetUuid_Negative") {
  int numDevices = 0;
  hipDevice_t device;
  hipUUID uuid;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    HIP_CHECK(hipDeviceGet(&device, 0));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(nullptr, device));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, -1));
    REQUIRE_FALSE(hipSuccess == hipDeviceGetUuid(&uuid, numDevices));
  }
}
#ifdef __linux__
#if HT_AMD

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
 *  - Get the UUID from rocminfo and compare it with the value from hipDeviceGetUuid API
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
TEST_CASE("Unit_hipDeviceGetUuid_From_RocmInfo") {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  assert(deviceCount > 0);

  FILE* fpipe;
  char command[COMMAND_LEN] = "";
  const char* rocmInfo = "rocminfo";

  snprintf(command, COMMAND_LEN, "%s", rocmInfo);
  strncat(command, " | grep -i Uuid:", COMMAND_LEN);
  // Execute the rocminfo command and extract the UUID info
  fpipe = popen(command, "r");
  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    return;
  }
  char command_op[BUFFER_LEN];
  int j = 0;
  std::map<int, std::vector<char>> uuid_map;
  while (fgets(command_op, BUFFER_LEN, fpipe)) {
    std::string rocminfo_line(command_op);
    if (std::string::npos != rocminfo_line.find("CPU-")) {
      continue;
    } else if (auto loc = rocminfo_line.find("GPU-"); loc != std::string::npos) {
      if (std::string::npos ==
          rocminfo_line.find(
              "GPU-XX")) {  // Only make an entry if the device is not an iGPU  // NOLINT
        std::vector<char> t_uuid(16, 0);
        std::memcpy(t_uuid.data(), &rocminfo_line[loc + 4], 16);
        uuid_map[j] = t_uuid;
      }
    }
    j++;
  }

  auto visible_devices = parseVisibleDevices();
  if (visible_devices.size() > 0) {
    // We have visible devices set, basically parse the visible devices and remove the entries
    size_t start = 0;  // The devices will be reported from 0..
    std::map<int, std::vector<char>> uuid_map_copy;
    for (auto device : visible_devices) {
      uuid_map_copy[start] = uuid_map[device];
    }
    uuid_map = uuid_map_copy;
  }

  for (const auto& i : uuid_map) {
    if (i.second.size() == 0) {
      continue;
    }
    auto dev = i.first;
    HIP_CHECK(hipSetDevice(dev));
    hipDevice_t device;
    HIP_CHECK(hipDeviceGet(&device, dev));
    hipUUID d_uuid{0};
    HIP_CHECK(hipDeviceGetUuid(&d_uuid, device));
    REQUIRE(memcmp(d_uuid.bytes, i.second.data(), 16) == 0);
  }
}
#endif
/**
 * Test Description
 * ------------------------
 *  - Get the UUID from hipGetDeviceProperties and compare it with value from hipDeviceGetUuid API
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.7
 */
// Guarding it against NVIDIA as this test is faling on it.
#if HT_AMD
TEST_CASE("Unit_hipDeviceGetUuid_VerifyUuidFrm_hipGetDeviceProperties") {
  int deviceCount = 0;
  hipDevice_t device;
  hipDeviceProp_t prop;
  hipUUID uuid{0};
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  assert(deviceCount > 0);
  for (int dev = 0; dev < deviceCount; dev++) {
    HIP_CHECK(hipSetDevice(dev));
    HIP_CHECK(hipDeviceGet(&device, dev));
    HIP_CHECK(hipDeviceGetUuid(&uuid, device));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    REQUIRE(memcmp(uuid.bytes, prop.uuid.bytes, 16) == 0);
  }
}
#endif
#endif

#if HT_AMD
#ifdef __linux__
auto getUUIDlistFromRocmInfo() {
  FILE* fpipe;
  char command[COMMAND_LEN] = "";
  const char* rocmInfo = "rocminfo";

  snprintf(command, COMMAND_LEN, "%s", rocmInfo);
  strncat(command, " | grep -i Uuid:", COMMAND_LEN);
  // Execute the rocminfo command and extract the UUID info
  fpipe = popen(command, "r");
  if (fpipe == nullptr) {
    printf("Unable to create command file\n");
    assert(false);
  }
  char command_op[BUFFER_LEN];
  int j = 0;
  std::map<int, std::vector<char>> uuid_map;
  while (fgets(command_op, BUFFER_LEN, fpipe)) {
    std::string rocminfo_line(command_op);
    if (std::string::npos != rocminfo_line.find("CPU-")) {
      continue;
    } else if (auto loc = rocminfo_line.find("GPU-"); loc != std::string::npos) {
      if (std::string::npos == rocminfo_line.find("GPU-XX")) {
        std::vector<char> t_uuid(20, 0);
        std::memcpy(t_uuid.data(), &rocminfo_line[loc], 20);
        uuid_map[j] = t_uuid;
      }
    }
    j++;
  }
  return uuid_map;
}
#endif

auto getUUIDlistWithoutRocmInfo() {
  hip::SpawnProc proc("uuidList", true);
  REQUIRE(proc.run() == 0);
  std::string strList = proc.getOutput();
  std::string delimiter = ",";
  size_t pos = 0;
  int first = 0;
  std::map<int, std::vector<char>> uuid_map;
  while ((pos = strList.find(delimiter)) != std::string::npos) {
    std::vector<char> t_uuid(20, 0);
    std::memcpy(t_uuid.data(), &strList[0], 20);
    uuid_map[first] = t_uuid;
    first++;
    strList.erase(0, pos + delimiter.length());
  }
  std::vector<char> tmp_uuid(20, 0);
  std::memcpy(tmp_uuid.data(), &strList[0], 20);
  uuid_map[first] = tmp_uuid;
  return uuid_map;
}

/**
 * Test Description
 * ------------------------
 *  - All the tests contains the various cases of setting the Env
 *  - var HIP_VISIBLE_DEVICES in different process (parent/child)
 *  - and verifies the Env functionality by checking UUID'S in
 *  - different process.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_Uuid_FntlTstsFor_SetEnv_HIP_VISIBLE_DEVICES") {
  std::map<int, std::vector<char>> uuid_map;
#ifdef __linux__
  uuid_map = getUUIDlistFromRocmInfo();
#else
  uuid_map = getUUIDlistWithoutRocmInfo();
#endif
  if (uuid_map.size() > 0) {
    SECTION("Set Env in parent and verify UUID in child ") {
      // Set Env Var with first GPU
      std::string uuid = uuid_map[0].data();
      std::string uuidEnv = uuid.substr(0, 20);
      unsetenv("HIP_VISIBLE_DEVICES");
      setenv("HIP_VISIBLE_DEVICES", uuidEnv.c_str(), 1);
      hip::SpawnProc proc("chkUUIDFrmChildProc_Exe", true);
      std::string t_uuid = uuidEnv.substr(4, 19);
      REQUIRE(proc.run(t_uuid) == 1);
      unsetenv("HIP_VISIBLE_DEVICES");
    }
#if 0  // Disabling below 2 tests due to the defect SWDEV-467665
    SECTION("Set Env in parent and verify UUID in Grand child") {
      std::string uuid = uuid_map[0].data();
      std::string uuidEnv = uuid.substr(0, 20);
      unsetenv("HIP_VISIBLE_DEVICES");
      setenv("HIP_VISIBLE_DEVICES", uuidEnv.c_str(), 1);
      hip::SpawnProc proc("passUUIDToGrandChild_Exe", true);
      REQUIRE(proc.run(uuidEnv)== 1);
      unsetenv("HIP_VISIBLE_DEVICES");
    }

    SECTION("Reset Env in child and verify UUID in Grand child") {
      if (uuid_map.size() >= 2) {
        std::string uuid = uuid_map[1].data();
        std::string uuidEnv = uuid.substr(0, 20);
        unsetenv("HIP_VISIBLE_DEVICES");
        setenv("HIP_VISIBLE_DEVICES", uuidEnv.c_str(), 1);
        hip::SpawnProc proc("ResetUUIDInChild_Exe", true);
        REQUIRE(proc.run()== 1);
        unsetenv("HIP_VISIBLE_DEVICES");
      } else {
        HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");  // NOLINT
      }
    }
#endif
    SECTION("Get Dev Count from Child") {
      if (uuid_map.size() >= 2) {
        std::string uuid = uuid_map[0].data();
        std::string uuidEnv = uuid.substr(0, 20);
        std::string uuid1 = uuid_map[1].data();
        std::string uuidEnv1 = uuid1.substr(0, 20);
        std::string totalString = uuidEnv + "," + uuidEnv1;
        unsetenv("HIP_VISIBLE_DEVICES");
        setenv("HIP_VISIBLE_DEVICES", totalString.c_str(), 1);
        hip::SpawnProc proc("setuuidGetDevCount", true);
        REQUIRE(proc.run() == 2);
        unsetenv("HIP_VISIBLE_DEVICES");
      } else {
        HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");  // NOLINT
      }
    }
#ifdef __linux__
    SECTION("Get UUID from Child proc rocminfo") {
      std::string setUuid = uuid_map[0].data();
      std::string uuidEnv = setUuid.substr(0, 20);
      unsetenv("HIP_VISIBLE_DEVICES");
      setenv("HIP_VISIBLE_DEVICES", uuidEnv.c_str(), 1);
      std::string wholeString;
      std::string uuid;
      std::string finalUuid = "";
      for (auto it = uuid_map.cbegin(); it != uuid_map.cend(); ++it) {
        if (it->second.size() == 0) {
          continue;
        } else {
          wholeString = it->second.data();
          uuid = wholeString.substr(0, 20);
          if (it->first == uuid_map.size() - 1) {
            finalUuid = finalUuid + uuid;
          } else {
            finalUuid = finalUuid + uuid + ",";
          }
        }
      }
      hip::SpawnProc proc("getUUIDfrmRocinfo", true);
      REQUIRE(proc.run(finalUuid) == 1);
      unsetenv("HIP_VISIBLE_DEVICES");
    }
#endif
    SECTION("Set multiple uuid") {
      std::string wholeString;
      std::string uuid;
      std::string finalUuid = "";
      for (auto it = uuid_map.cbegin(); it != uuid_map.cend(); ++it) {
        if (it->second.size() == 0) {
          continue;
        } else {
          wholeString = it->second.data();
          uuid = wholeString.substr(0, 20);
          if (it->first == uuid_map.size() - 1) {
            finalUuid = finalUuid + uuid;
          } else {
            finalUuid = finalUuid + uuid + ",";
          }
        }
      }
      unsetenv("HIP_VISIBLE_DEVICES");
      setenv("HIP_VISIBLE_DEVICES", finalUuid.c_str(), 1);
      hip::SpawnProc proc("multipleUUID", true);
      REQUIRE(proc.run(finalUuid) == 1);
      unsetenv("HIP_VISIBLE_DEVICES");
    }
    SECTION("Set mix Env variables") {
      if (uuid_map.size() >= 2) {
        std::string uuid = "0";
        std::string uuid1 = uuid_map[1].data();
        std::string uuidEnv1 = uuid1.substr(0, 20);
        std::string totalString = uuid + "," + uuidEnv1;
        unsetenv("HIP_VISIBLE_DEVICES");
        setenv("HIP_VISIBLE_DEVICES", totalString.c_str(), 1);
        hip::SpawnProc proc("setuuidGetDevCount", true);
        REQUIRE(proc.run() == 2);
        unsetenv("HIP_VISIBLE_DEVICES");
      } else {
        HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");  // NOLINT
      }
    }
    SECTION("Set Same UUID/Device ordinal more than once ") {
      if (uuid_map.size() >= 2) {
        std::string uuid = "0";
        std::string uuid1 = uuid_map[0].data();
        std::string uuidEnv1 = uuid1.substr(0, 20);
        std::string uuid2 = uuid_map[1].data();
        std::string uuidEnv2 = uuid2.substr(0, 20);
        std::string totalString = uuid + "," + uuidEnv2 + "," + uuidEnv1 + "," + uuid;
        unsetenv("HIP_VISIBLE_DEVICES");
        setenv("HIP_VISIBLE_DEVICES", totalString.c_str(), 1);
        hip::SpawnProc proc("setuuidGetDevCount", true);
        REQUIRE(proc.run() == 2);
        unsetenv("HIP_VISIBLE_DEVICES");
      } else {
        HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");  // NOLINT
      }
    }
    SECTION("Set Env Variable in child process") {
      std::string uuid = uuid_map[0].data();
      std::string uuidEnv = uuid.substr(0, 20);
      hip::SpawnProc proc("setEnvInChildProc", true);
      REQUIRE(proc.run(uuidEnv) == 1);
      int devCount = 0;
      HIP_CHECK(hipGetDeviceCount(&devCount));
      REQUIRE(devCount == uuid_map.size());
    }
#ifdef __linux__
    SECTION("Chk RocmInfo Uuid list before and after set Env") {
      std::map<int, std::vector<char>> uuid_map;
      uuid_map = getUUIDlistFromRocmInfo();
      std::string uuid = uuid_map[0].data();
      unsetenv("HIP_VISIBLE_DEVICES");
      setenv("HIP_VISIBLE_DEVICES", uuid.c_str(), 1);
      std::map<int, std::vector<char>> uuid_map1;
      uuid_map1 = getUUIDlistFromRocmInfo();
      REQUIRE(uuid_map.size() == uuid_map1.size());
    }
#endif
  } else {
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 1");  // NOLINT
  }
}

void ChkUUID() {
  if (tState == 2) {
    return;
  }
  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  if (devCount != 1) {
    tState = 0;
    return;
  }
  hipDevice_t device;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipDeviceGet(&device, 0));
  hipUUID d_uuid{0};
  HIP_CHECK(hipDeviceGetUuid(&d_uuid, device));
  std::map<int, std::vector<char>> uuid_map;
#ifdef __linux__
  uuid_map = getUUIDlistFromRocmInfo();
#else
  uuid_map = getUUIDlistWithoutRocmInfo();
#endif
  if (!uuid_map.empty()) {
    std::string uuid = uuid_map[0].data();
    std::string t_uuid = uuid.substr(4, 19);
    if (memcmp(d_uuid.bytes, t_uuid.c_str(), 16) == 0) {
      tState = 1;
    }
  }
}

void setEnv() {
  std::map<int, std::vector<char>> uuid_map;
#ifdef __linux__
  uuid_map = getUUIDlistFromRocmInfo();
#else
  uuid_map = getUUIDlistWithoutRocmInfo();
#endif
  if (uuid_map.size() >= 2) {
    std::string uuid = uuid_map[1].data();
    std::string uuidEnv = uuid.substr(0, 20);
    unsetenv("HIP_VISIBLE_DEVICES");
    setenv("HIP_VISIBLE_DEVICES", uuidEnv.c_str(), 1);
  } else {
    tState = 2;
    HipTest::HIP_SKIP_TEST("Skipping because this machine has total GPUs < 2");  // NOLINT
  }
}
/**
 * Test Description
 * ------------------------
 *  - Multi thread scenario.
 *  - Set Env var HIP_VISIBLE_DEVICES in one thread
 *  - Verify the followed by functionality in another thread
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceGetUuid.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_UUID_setEnv_Thread") {
  // Create Thread one
  std::thread t1(setEnv);
  t1.join();
  // Create Thread two
  std::thread t2(ChkUUID);
  t2.join();
  REQUIRE(tState != 0);
}
#endif
/**
 * End doxygen group DriverTest.
 * @}
 */
