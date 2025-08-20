/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_helper.hh>
#include <hip_test_defgroups.hh>
#include <hip/hip_deprecated.h>

/**
 * @addtogroup hipGetProcAddress hipGetProcAddress
 * @{
 * @ingroup DeviceTest
 * `hipGetProcAddress(const char* symbol, void** pfn, int  hipVersion, uint64_t flags,
                      hipDriverProcAddressQueryResult* symbolStatus);` -
 * Gets the symbol's function address.
 */
constexpr int size = 20;
constexpr size_t len = 256;
void CreateMemPool(int device, hipMemPool_t& mem_pool) {
  hipMemPoolProps kPoolProps;
  kPoolProps.allocType = hipMemAllocationTypePinned;
  kPoolProps.location.type = hipMemLocationTypeDevice;
  kPoolProps.location.id = device;
  HIP_CHECK(hipMemPoolCreate(&mem_pool, &kPoolProps));
}
/**
 * Test Description
 * ------------------------
 *  - Validate different device Api's basic functionality
 *  - with the function pointer from the API hipGetProcAddress
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.CC
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_ValidateDeviceApis") {
  void* hipGetDeviceCount_ptr = nullptr;
  void* hipRuntimeGetVersion_ptr = nullptr;
  void* hipDeviceGetLimit_ptr = nullptr;
  void* hipDeviceSetLimit_ptr = nullptr;
  void* hipDeviceComputeCapability_ptr = nullptr;
  void* hipDeviceGet_ptr = nullptr;
  void* hipDeviceGetPCIBusId_ptr = nullptr;
  void* hipDeviceGetByPCIBusId_ptr = nullptr;
  void* hipDeviceGetDefaultMemPool_ptr = nullptr;
  void* hipDeviceGetName_ptr = nullptr;
  void* hipDeviceGetUuid_ptr = nullptr;
  void* hipGetDeviceFlags_ptr = nullptr;
  void* hipSetDeviceFlags_ptr = nullptr;
  void* hipDeviceReset_ptr = nullptr;
  void* hipDriverGetVersion_ptr = nullptr;
  void* hipDeviceGetCacheConfig_ptr = nullptr;
  void* hipDeviceSetCacheConfig_ptr = nullptr;
  void* hipDeviceTotalMem_ptr = nullptr;
  void* hipGetDeviceProperties_ptr = nullptr;
  void* hipGetDevicePropertiesR0000_ptr = nullptr;
  void* hipGetDevicePropertiesR0600_ptr = nullptr;
  void* hipChooseDevice_ptr = nullptr;
  void* hipChooseDeviceR0000_ptr = nullptr;
  void* hipChooseDeviceR0600_ptr = nullptr;
  void* hipDeviceSetSharedMemConfig_ptr = nullptr;
  void* hipDeviceGetSharedMemConfig_ptr = nullptr;
  void* hipDeviceGetAttribute_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGetDeviceCount", &hipGetDeviceCount_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipRuntimeGetVersion", &hipRuntimeGetVersion_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetLimit", &hipDeviceGetLimit_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceSetLimit", &hipDeviceSetLimit_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceComputeCapability", &hipDeviceComputeCapability_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGet", &hipDeviceGet_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetPCIBusId", &hipDeviceGetPCIBusId_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetByPCIBusId", &hipDeviceGetByPCIBusId_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetDefaultMemPool", &hipDeviceGetDefaultMemPool_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipDeviceGetName", &hipDeviceGetName_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipDeviceGetUuid", &hipDeviceGetUuid_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetDeviceFlags", &hipGetDeviceFlags_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipSetDeviceFlags", &hipSetDeviceFlags_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipDeviceReset", &hipDeviceReset_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDriverGetVersion", &hipDriverGetVersion_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetCacheConfig", &hipDeviceGetCacheConfig_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceSetCacheConfig", &hipDeviceSetCacheConfig_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceTotalMem", &hipDeviceTotalMem_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetDeviceProperties", &hipGetDeviceProperties_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetDevicePropertiesR0000", &hipGetDevicePropertiesR0000_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetDevicePropertiesR0600", &hipGetDevicePropertiesR0600_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(
      hipGetProcAddress("hipChooseDevice", &hipChooseDevice_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipChooseDeviceR0000", &hipChooseDeviceR0000_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipChooseDeviceR0600", &hipChooseDeviceR0600_ptr, currentHipVersion,
                              0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceSetSharedMemConfig", &hipDeviceSetSharedMemConfig_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetSharedMemConfig", &hipDeviceGetSharedMemConfig_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetAttribute", &hipDeviceGetAttribute_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipGetDeviceCount_ptr)(int*) =
      reinterpret_cast<hipError_t (*)(int*)>(hipGetDeviceCount_ptr);
  hipError_t (*dyn_hipRuntimeGetVersion_ptr)(int*) =
      reinterpret_cast<hipError_t (*)(int*)>(hipRuntimeGetVersion_ptr);
  hipError_t (*dyn_hipDeviceGetLimit_ptr)(size_t*, hipLimit_t) =
      reinterpret_cast<hipError_t (*)(size_t*, hipLimit_t)>(hipDeviceGetLimit_ptr);
  hipError_t (*dyn_hipDeviceSetLimit_ptr)(hipLimit_t, size_t) =
      reinterpret_cast<hipError_t (*)(hipLimit_t, size_t)>(hipDeviceSetLimit_ptr);
  hipError_t (*dyn_hipDeviceComputeCapability_ptr)(int*, int*, hipDevice_t) =
      reinterpret_cast<hipError_t (*)(int*, int*, hipDevice_t)>(hipDeviceComputeCapability_ptr);
  hipError_t (*dyn_hipDeviceGet_ptr)(hipDevice_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDevice_t*, int)>(hipDeviceGet_ptr);
  hipError_t (*dyn_hipDeviceGetPCIBusId_ptr)(char*, int, int) =
      reinterpret_cast<hipError_t (*)(char*, int, int)>(hipDeviceGetPCIBusId_ptr);
  hipError_t (*dyn_hipDeviceGetByPCIBusId_ptr)(int*, const char*) =
      reinterpret_cast<hipError_t (*)(int*, const char*)>(hipDeviceGetByPCIBusId_ptr);
  hipError_t (*dyn_hipDeviceGetDefaultMemPool_ptr)(hipMemPool_t*, int) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t*, int)>(hipDeviceGetDefaultMemPool_ptr);
  hipError_t (*dyn_hipDeviceGetName_ptr)(char*, int, hipDevice_t) =
      reinterpret_cast<hipError_t (*)(char*, int, hipDevice_t)>(hipDeviceGetName_ptr);
  hipError_t (*dyn_hipDeviceGetUuid_ptr)(hipUUID*, hipDevice_t) =
      reinterpret_cast<hipError_t (*)(hipUUID*, hipDevice_t)>(hipDeviceGetUuid_ptr);
  hipError_t (*dyn_hipGetDeviceFlags_ptr)(unsigned int*) =
      reinterpret_cast<hipError_t (*)(unsigned int*)>(hipGetDeviceFlags_ptr);
  hipError_t (*dyn_hipSetDeviceFlags_ptr)(unsigned int) =
      reinterpret_cast<hipError_t (*)(unsigned int)>(hipSetDeviceFlags_ptr);
  hipError_t (*dyn_hipDeviceReset_ptr)() = reinterpret_cast<hipError_t (*)()>(hipDeviceReset_ptr);
  hipError_t (*dyn_hipDriverGetVersion_ptr)(int*) =
      reinterpret_cast<hipError_t (*)(int*)>(hipDriverGetVersion_ptr);
  hipError_t (*dyn_hipDeviceGetCacheConfig_ptr)(hipFuncCache_t*) =
      reinterpret_cast<hipError_t (*)(hipFuncCache_t*)>(hipDeviceGetCacheConfig_ptr);
  hipError_t (*dyn_hipDeviceSetCacheConfig_ptr)(hipFuncCache_t) =
      reinterpret_cast<hipError_t (*)(hipFuncCache_t)>(hipDeviceSetCacheConfig_ptr);
  hipError_t (*dyn_hipDeviceTotalMem_ptr)(size_t*, hipDevice_t) =
      reinterpret_cast<hipError_t (*)(size_t*, hipDevice_t)>(hipDeviceTotalMem_ptr);
  hipError_t (*dyn_hipGetDeviceProperties_ptr)(hipDeviceProp_t*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_t*, int)>(hipGetDeviceProperties_ptr);
  hipError_t (*dyn_hipGetDevicePropertiesR0000_ptr)(hipDeviceProp_tR0000*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_tR0000*, int)>(hipGetDevicePropertiesR0000_ptr);
  hipError_t (*dyn_hipGetDevicePropertiesR0600_ptr)(hipDeviceProp_tR0600*, int) =
      reinterpret_cast<hipError_t (*)(hipDeviceProp_tR0600*, int)>(hipGetDevicePropertiesR0600_ptr);
  hipError_t (*dyn_hipChooseDevice_ptr)(int*, hipDeviceProp_t*) =
      reinterpret_cast<hipError_t (*)(int*, hipDeviceProp_t*)>(hipChooseDevice_ptr);
  hipError_t (*dyn_hipChooseDeviceR0000_ptr)(int*, hipDeviceProp_tR0000*) =
      reinterpret_cast<hipError_t (*)(int*, hipDeviceProp_tR0000*)>(hipChooseDevice_ptr);
  hipError_t (*dyn_hipChooseDeviceR0600_ptr)(int*, hipDeviceProp_tR0600*) =
      reinterpret_cast<hipError_t (*)(int*, hipDeviceProp_tR0600*)>(hipChooseDevice_ptr);
  hipError_t (*dyn_hipDeviceSetSharedMemConfig_ptr)(hipSharedMemConfig) =
      reinterpret_cast<hipError_t (*)(hipSharedMemConfig)>(hipDeviceSetSharedMemConfig_ptr);
  hipError_t (*dyn_hipDeviceGetSharedMemConfig_ptr)(hipSharedMemConfig*) =
      reinterpret_cast<hipError_t (*)(hipSharedMemConfig*)>(hipDeviceGetSharedMemConfig_ptr);
  hipError_t (*dyn_hipDeviceGetAttribute_ptr)(int*, hipDeviceAttribute_t, int) =  // NOLINT
      reinterpret_cast<hipError_t (*)(int*, hipDeviceAttribute_t, int)>(hipDeviceGetAttribute_ptr);

  // hipGetDeviceCount API
  int devCount_ptr = 0, devCount = 0;
  HIP_CHECK(dyn_hipGetDeviceCount_ptr(&devCount_ptr));
  HIP_CHECK(hipGetDeviceCount(&devCount));
  REQUIRE(devCount_ptr >= 0);
  REQUIRE(devCount >= 0);
  REQUIRE(devCount_ptr == devCount);

  // hipRuntimeGetVersion API
  int runtimeVersion = -1, runtimeVersion_ptr = -1;
  HIP_CHECK(hipRuntimeGetVersion(&runtimeVersion));
  HIP_CHECK(dyn_hipRuntimeGetVersion_ptr(&runtimeVersion_ptr));
  REQUIRE(runtimeVersion == runtimeVersion_ptr);

  // hipDeviceGetLimit API
  // hipDeviceSetLimit API
  size_t oldVal, oldVal_ptr;
  HIP_CHECK(hipDeviceGetLimit(&oldVal, hipLimitStackSize));
  HIP_CHECK(dyn_hipDeviceGetLimit_ptr(&oldVal_ptr, hipLimitStackSize));
  REQUIRE(oldVal == oldVal_ptr);
  HIP_CHECK(hipDeviceSetLimit(hipLimitStackSize, oldVal + 8));
  HIP_CHECK(dyn_hipDeviceSetLimit_ptr(hipLimitStackSize, oldVal_ptr + 8));
  REQUIRE((oldVal + 8) == (oldVal_ptr + 8));
  size_t new_val;
  HIP_CHECK(dyn_hipDeviceGetLimit_ptr(&new_val, hipLimitStackSize));
  REQUIRE(new_val >= oldVal + 8);

  // hipDeviceGet API
  // hipDeviceComputeCapability API
  int major, minor, major_ptr, minor_ptr;
  hipDevice_t device, device_ptr;
  for (int i = 0; i < devCount; i++) {
    HIP_CHECK(hipDeviceGet(&device, i));
    HIP_CHECK(dyn_hipDeviceGet_ptr(&device_ptr, i));
    REQUIRE(device == device_ptr);
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    HIP_CHECK(dyn_hipDeviceComputeCapability_ptr(&major_ptr, &minor_ptr, device));
    REQUIRE(major == major_ptr);
    REQUIRE(minor == minor_ptr);
  }
  // hipDeviceGetPCIBusId API
  char pciBusId[size]{};
  char pciBusId_ptr[size]{};
  HIP_CHECK(hipDeviceGetPCIBusId(&pciBusId[0], size, 0));
  HIP_CHECK(dyn_hipDeviceGetPCIBusId_ptr(&pciBusId_ptr[0], size, 0));
  REQUIRE(*pciBusId == *pciBusId_ptr);

  // hipDeviceGetByPCIBusId API
  int tempDeviceId1, tempDeviceId2;
  HIP_CHECK(hipDeviceGetByPCIBusId(&tempDeviceId1, pciBusId));
  HIP_CHECK(dyn_hipDeviceGetByPCIBusId_ptr(&tempDeviceId2, pciBusId));
  REQUIRE(tempDeviceId1 == tempDeviceId2);

  // hipDeviceGetDefaultMemPool API
  hipMemPool_t mem_pool, mem_pool_ptr;
  HIP_CHECK(hipDeviceGetDefaultMemPool(&mem_pool, 0));
  HIP_CHECK(dyn_hipDeviceGetDefaultMemPool_ptr(&mem_pool_ptr, 0));
  REQUIRE(mem_pool != nullptr);
  REQUIRE(mem_pool == mem_pool_ptr);

  // hipDeviceGetName API
  std::array<char, len> name, name_ptr;
  HIP_CHECK(hipDeviceGetName(name.data(), name.size(), 0));
  HIP_CHECK(dyn_hipDeviceGetName_ptr(name_ptr.data(), name_ptr.size(), 0));
  REQUIRE(strncmp(name.data(), name_ptr.data(), name.size()) == 0);

  // hipDeviceGetUuid API
  hipUUID uuid{0}, uuid_ptr{0};
  HIP_CHECK(hipDeviceGetUuid(&uuid, 0));
  HIP_CHECK(dyn_hipDeviceGetUuid_ptr(&uuid_ptr, 0));
  size_t uuidSize = sizeof(uuid.bytes) / sizeof(uuid.bytes[0]);
  REQUIRE(strncmp(uuid.bytes, uuid_ptr.bytes, uuidSize) == 0);

  // hipGetDeviceFlags API
  unsigned int flags = 0u, flags_ptr = 0u;
  HIP_CHECK(hipGetDeviceFlags(&flags));
  HIP_CHECK(dyn_hipGetDeviceFlags_ptr(&flags_ptr));
  REQUIRE(flags == flags_ptr);

  // hipSetDeviceFlags API
  unsigned int flagsVar = 0u, flagsVarStore = 0u, flagsVar_ptr = 0u;
  HIP_CHECK(hipSetDeviceFlags(hipDeviceScheduleBlockingSync));
  HIP_CHECK(hipGetDeviceFlags(&flagsVar));
  flagsVarStore = flagsVar;
  // hipDeviceReset API
  HIP_CHECK(dyn_hipDeviceReset_ptr());
  HIP_CHECK(dyn_hipSetDeviceFlags_ptr(hipDeviceScheduleBlockingSync));
  HIP_CHECK(hipGetDeviceFlags(&flagsVar_ptr));
  REQUIRE(flagsVarStore == flagsVar_ptr);

  // hipDriverGetVersion API
  int driverVersion, driverVersion_ptr;
  HIP_CHECK(hipDriverGetVersion(&driverVersion));
  HIP_CHECK(dyn_hipDriverGetVersion_ptr(&driverVersion_ptr));
  REQUIRE(driverVersion == driverVersion_ptr);

  // hipDeviceSetCacheConfig API
  HIP_CHECK(hipSetDevice(0));
  auto cache_config = hipFuncCachePreferL1;
  auto cache_config1 = hipFuncCachePreferNone;
  HIP_CHECK(hipDeviceSetCacheConfig(cache_config));
  HIP_CHECK(dyn_hipDeviceSetCacheConfig_ptr(cache_config1));

  // hipDeviceGetCacheConfig API
  hipFuncCache_t cacheConfig, cacheConfig_ptr;
  HIP_CHECK(hipDeviceGetCacheConfig(&cacheConfig));
  HIP_CHECK(dyn_hipDeviceGetCacheConfig_ptr(&cacheConfig_ptr));
  REQUIRE(cacheConfig == hipFuncCachePreferNone);
  REQUIRE(cacheConfig == cacheConfig_ptr);

  // hipDeviceTotalMem API
  size_t totMem, totMem_ptr;
  HIP_CHECK(hipDeviceTotalMem(&totMem, 0));
  HIP_CHECK(dyn_hipDeviceTotalMem_ptr(&totMem_ptr, 0));
  REQUIRE(totMem == totMem_ptr);

  // hipGetDeviceProperties API
  hipDeviceProp_t prop, prop_ptr;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  HIP_CHECK(dyn_hipGetDeviceProperties_ptr(&prop_ptr, 0));
  REQUIRE(prop.major == prop_ptr.major);

  // hipGetDevicePropertiesR0000 API
  hipDeviceProp_tR0000 propR0000, propR0000_ptr;
  HIP_CHECK(hipGetDevicePropertiesR0000(&propR0000, 0));
  HIP_CHECK(dyn_hipGetDevicePropertiesR0000_ptr(&propR0000_ptr, 0));
  REQUIRE(propR0000.major == propR0000_ptr.major);

  // hipGetDevicePropertiesR0600 API
  hipDeviceProp_tR0600 propR0600, propR0600_ptr;
  HIP_CHECK(hipGetDevicePropertiesR0600(&propR0600, 0));
  HIP_CHECK(dyn_hipGetDevicePropertiesR0600_ptr(&propR0600_ptr, 0));
  REQUIRE(propR0600.major == propR0600_ptr.major);

  // hipChooseDevice API
  int dev, dev_ptr;
  HIP_CHECK(hipChooseDevice(&dev, &prop));
  HIP_CHECK(dyn_hipChooseDevice_ptr(&dev_ptr, &prop));
  REQUIRE(dev == dev_ptr);

  // hipChooseDeviceR0000 API
  int devR0000, devR0000_ptr;
  HIP_CHECK(hipChooseDeviceR0000(&devR0000, &propR0000));
  HIP_CHECK(dyn_hipChooseDeviceR0000_ptr(&devR0000_ptr, &propR0000));
  REQUIRE(devR0000 == devR0000_ptr);

  // hipChooseDeviceR0600 API
  int devR0600, devR0600_ptr;
  HIP_CHECK(hipChooseDeviceR0600(&devR0600, &propR0600));
  HIP_CHECK(dyn_hipChooseDeviceR0600_ptr(&devR0600_ptr, &propR0600));
  REQUIRE(devR0600 == devR0600_ptr);

  // hipDeviceSetSharedMemConfig API
  HIP_CHECK(hipDeviceSetSharedMemConfig(hipSharedMemBankSizeFourByte));
  HIP_CHECK(dyn_hipDeviceSetSharedMemConfig_ptr(hipSharedMemBankSizeFourByte));

  // hipDeviceGetSharedMemConfig API
  hipSharedMemConfig memConfig, memConfig_ptr;
  HIP_CHECK(hipDeviceGetSharedMemConfig(&memConfig));
  HIP_CHECK(dyn_hipDeviceGetSharedMemConfig_ptr(&memConfig_ptr));
  REQUIRE(memConfig == hipSharedMemBankSizeFourByte);
  REQUIRE(memConfig == memConfig_ptr);

  // hipDeviceGetAttribute API
  int value, value_ptr;
  HIP_CHECK(hipDeviceGetAttribute(&value, hipDeviceAttributeMaxThreadsPerBlock, 0));
  HIP_CHECK(dyn_hipDeviceGetAttribute_ptr(&value_ptr, hipDeviceAttributeMaxThreadsPerBlock, 0));
  REQUIRE(value == value_ptr);
}
/**
 * Test Description
 * ------------------------
 *  - Validate device Api's basic functionality
 *  - with the function pointer from the API hipGetProcAddress
 * Test source
 * ------------------------
 *  - unit/device/hipGetProcAddress.CC
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */

TEST_CASE("Unit_hipGetProcAddress_PeerDeviceAccessAPIs") {
  void* hipDeviceCanAccessPeer_ptr = nullptr;
  void* hipSetDevice_ptr = nullptr;
  void* hipGetDevice_ptr = nullptr;
  void* hipDeviceEnablePeerAccess_ptr = nullptr;
  void* hipDeviceDisablePeerAccess_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipDeviceCanAccessPeer", &hipDeviceCanAccessPeer_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipSetDevice", &hipSetDevice_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipGetDevice", &hipGetDevice_ptr, currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceEnablePeerAccess", &hipDeviceEnablePeerAccess_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceDisablePeerAccess", &hipDeviceDisablePeerAccess_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipDeviceCanAccessPeer_ptr)(int*, int, int) =
      reinterpret_cast<hipError_t (*)(int*, int, int)>(hipDeviceCanAccessPeer_ptr);
  hipError_t (*dyn_hipSetDevice_ptr)(int) = reinterpret_cast<hipError_t (*)(int)>(hipSetDevice_ptr);
  hipError_t (*dyn_hipGetDevice_ptr)(int*) =
      reinterpret_cast<hipError_t (*)(int*)>(hipGetDevice_ptr);
  hipError_t (*dyn_hipDeviceEnablePeerAccess_ptr)(int, unsigned int) =
      reinterpret_cast<hipError_t (*)(int, unsigned int)>(hipDeviceEnablePeerAccess_ptr);
  hipError_t (*dyn_hipDeviceDisablePeerAccess_ptr)(int) =
      reinterpret_cast<hipError_t (*)(int)>(hipDeviceDisablePeerAccess_ptr);

  SECTION("Test Peer Device Access API's") {
    int canAccessPeer_ptr = 0, canAccessPeer = 0, devCount = 0;
    HIP_CHECK(hipGetDeviceCount(&devCount));
    if (devCount < 2) {
      HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
      return;
    }
    // hipDeviceCanAccessPeer API
    int devId{};
    int dev = GENERATE(range(0, HipTest::getGeviceCount()));
    int peerDev = GENERATE(range(0, HipTest::getGeviceCount()));
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, dev, peerDev));
    HIP_CHECK(dyn_hipDeviceCanAccessPeer_ptr(&canAccessPeer_ptr, dev, peerDev));
    REQUIRE(canAccessPeer == canAccessPeer_ptr);
    HIP_CHECK(hipSetDevice(dev));
    // hipGetDevice API
    HIP_CHECK(dyn_hipGetDevice_ptr(&devId));
    REQUIRE(dev == devId);
    // hipSetDevice API
    HIP_CHECK(dyn_hipSetDevice_ptr(peerDev));
    HIP_CHECK(hipGetDevice(&devId));
    REQUIRE(devId == peerDev);
    // hipDeviceEnablePeerAccess API
    // hipDeviceDisablePeerAccess API
    SECTION("hipDeviceEnableDisablePeerAccess API's") {
      if (dev != peerDev) {
        HIP_CHECK(hipSetDevice(dev));
        HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer, dev, peerDev));
        if (canAccessPeer == 0) {
          HipTest::HIP_SKIP_TEST("Skipping because no P2P support");
          return;
        }
        HIP_CHECK(hipDeviceEnablePeerAccess(peerDev, 0));
        HIP_CHECK_ERROR(dyn_hipDeviceEnablePeerAccess_ptr(peerDev, 0),
                        hipErrorPeerAccessAlreadyEnabled);
        HIP_CHECK(hipDeviceDisablePeerAccess(peerDev));
        HIP_CHECK_ERROR(dyn_hipDeviceDisablePeerAccess_ptr(peerDev), hipErrorPeerAccessNotEnabled);
      }
    }
  }
}
bool CheckMemPoolSupport(const int device) {
  int mem_pool_support = 0;
  HIP_CHECK(
      hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, device));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST("Device doest have memory pool support");
    return false;
  }
  return true;
}

TEST_CASE("Unit_hipGetProcAddress_SetGetMemPoolAPIs") {
  void* hipDeviceSetMemPool_ptr = nullptr;
  void* hipDeviceGetMemPool_ptr = nullptr;
  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));
  HIP_CHECK(hipGetProcAddress("hipDeviceSetMemPool", &hipDeviceSetMemPool_ptr, currentHipVersion, 0,
                              nullptr));
  HIP_CHECK(hipGetProcAddress("hipDeviceGetMemPool", &hipDeviceGetMemPool_ptr, currentHipVersion, 0,
                              nullptr));
  hipError_t (*dyn_hipDeviceSetMemPool_ptr)(int, hipMemPool_t) =
      reinterpret_cast<hipError_t (*)(int, hipMemPool_t)>(hipDeviceSetMemPool_ptr);
  hipError_t (*dyn_hipDeviceGetMemPool_ptr)(hipMemPool_t*, int) =
      reinterpret_cast<hipError_t (*)(hipMemPool_t*, int)>(hipDeviceGetMemPool_ptr);

  int devCount = 0;
  HIP_CHECK(hipGetDeviceCount(&devCount));
  if (devCount < 2) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 2");
    return;
  }
  // hipDeviceSetMemPool API
  hipMemPool_t getMemPool = nullptr, getMemPool_ptr = nullptr;
  HIP_CHECK(hipSetDevice(0));
  if (!CheckMemPoolSupport(0)) {
    return;
  } else {
    CreateMemPool(0, getMemPool);
  }
  HIP_CHECK(hipSetDevice(1));
  if (!CheckMemPoolSupport(1)) {
    return;
  } else {
    CreateMemPool(1, getMemPool_ptr);
  }
  HIP_CHECK(hipDeviceSetMemPool(0, getMemPool));
  HIP_CHECK(dyn_hipDeviceSetMemPool_ptr(1, getMemPool_ptr));
  REQUIRE(getMemPool != nullptr);
  REQUIRE(getMemPool_ptr != nullptr);
  HIP_CHECK(hipMemPoolDestroy(getMemPool));
  HIP_CHECK(hipMemPoolDestroy(getMemPool_ptr));

  // hipDeviceGetMemPool API
  hipMemPool_t memPool, memPoolPtr;
  HIP_CHECK(hipDeviceGetMemPool(&memPool, 0));
  HIP_CHECK(dyn_hipDeviceGetMemPool_ptr(&memPoolPtr, 0));
  REQUIRE(memPool == memPoolPtr);
  HIP_CHECK(hipSetDevice(0));
}
