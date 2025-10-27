/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string.h>
#ifdef __linux__
#include <unistd.h>
#endif

#include <array>
#include <iostream>

#include <hip_test_common.hh>

/**
 * @addtogroup hipDeviceGetAttribute hipDeviceGetAttribute
 * @{
 * @ingroup DeviceTest
 * `hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)` -
 * Query for a specific device attribute.
 */

static hipError_t test_hipDeviceGetAttribute(int deviceId, hipDeviceAttribute_t attr,
                                             int expectedValue = -1) {
  int value = 0;
  std::cout << "Test hipDeviceGetAttribute attribute " << attr;
  if (expectedValue != -1) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != -1) && value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

static hipError_t test_hipDeviceGetHdpAddress(int deviceId, hipDeviceAttribute_t attr,
                                              uint32_t* expectedValue) {
  uint32_t* value = 0;
  std::cout << "Test hipDeviceGetHdpAddress attribute " << attr;
  if (expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) {
    std::cout << " expected value " << expectedValue;
  }
  HIP_CHECK(hipDeviceGetAttribute(reinterpret_cast<int*>(&value), attr, deviceId));
  std::cout << " actual value " << value << std::endl;
  if ((expectedValue != reinterpret_cast<uint32_t*>(0xdeadbeef)) && value != expectedValue) {
    std::cout << "fail" << std::endl;
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

/**
 * Test Description
 * ------------------------
 *  - Validate various device attributes against device properties.
 *  - Matching attribute and property value shall be equal.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetDeviceAttribute_CheckAttrValues") {
  int deviceId;
  HIP_CHECK(hipGetDevice(&deviceId));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  printf("info: running on device #%d %s\n", deviceId, props.name);

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerBlock,
                                       props.maxThreadsPerBlock));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimX, props.maxThreadsDim[0]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimY, props.maxThreadsDim[1]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxBlockDimZ, props.maxThreadsDim[2]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimX, props.maxGridSize[0]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimY, props.maxGridSize[1]));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxGridDimZ, props.maxGridSize[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerBlock,
                                       props.sharedMemPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTotalConstantMemory,
                                       props.totalConstMem));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeWarpSize, props.warpSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxRegistersPerBlock,
                                       props.regsPerBlock));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeClockRate, props.clockRate));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryClockRate,
                                       props.memoryClockRate));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMemoryBusWidth, props.memoryBusWidth));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMultiprocessorCount,
                                       props.multiProcessorCount));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIsMultiGpuBoard,
                                       props.isMultiGpuBoard));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeMode, props.computeMode));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeL2CacheSize, props.l2CacheSize));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxThreadsPerMultiProcessor,
                                       props.maxThreadsPerMultiProcessor));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMajor, props.major));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeComputeCapabilityMinor, props.minor));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeConcurrentKernels,
                                       props.concurrentKernels));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciBusId, props.pciBusID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePciDeviceId, props.pciDeviceID));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeIntegrated, props.integrated));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture1DWidth,
                                       props.maxTexture1D));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DWidth,
                                       props.maxTexture2D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture2DHeight,
                                       props.maxTexture2D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DWidth,
                                       props.maxTexture3D[0]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DHeight,
                                       props.maxTexture3D[1]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxTexture3DDepth,
                                       props.maxTexture3D[2]));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeLaunch,
                                       props.cooperativeLaunch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCooperativeMultiDeviceLaunch,
                                       props.cooperativeMultiDeviceLaunch));

#if HT_AMD
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
                                       props.maxSharedMemoryPerMultiProcessor));
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpMemFlushCntl,
                                        props.hdpMemFlushCntl));
  HIP_CHECK(test_hipDeviceGetHdpAddress(deviceId, hipDeviceAttributeHdpRegFlushCntl,
                                        props.hdpRegFlushCntl));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeDirectManagedMemAccessFromHost,
                                       props.directManagedMemAccessFromHost));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeConcurrentManagedAccess,
                                       props.concurrentManagedAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributePageableMemoryAccess,
                                       props.pageableMemoryAccess));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
                                       props.pageableMemoryAccessUsesHostPageTables));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
                                       props.cooperativeMultiDeviceUnmatchedFunc));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
                                       props.cooperativeMultiDeviceUnmatchedGridDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
                                       props.cooperativeMultiDeviceUnmatchedBlockDim));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId,
                                       hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
                                       props.cooperativeMultiDeviceUnmatchedSharedMem));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeAsicRevision, props.asicRevision));
  HIP_CHECK(
      test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeManagedMemory, props.managedMemory));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeNumberOfXccs));
#endif

  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeMaxPitch, props.memPitch));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTextureAlignment,
                                       props.textureAlignment));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeKernelExecTimeout,
                                       props.kernelExecTimeoutEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeCanMapHostMemory,
                                       props.canMapHostMemory));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeEccEnabled, props.ECCEnabled));
  HIP_CHECK(test_hipDeviceGetAttribute(deviceId, hipDeviceAttributeTexturePitchAlignment,
                                       props.texturePitchAlignment));
}

/**
 * Test Description
 * ------------------------
 *  - Validates negative scenarios:
 *    -# When pointer to value is `nullptr`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ID is `-1`
 *      - Expected output: do not return `hipSuccess`
 *    -# When device ID is out of bounds
 *      - Expected output: do not return `hipSuccess`
 *    -# When attribute is invalid (-1)
 *      - Expected output: do not return `hipSuccess`
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDeviceGetAttribute_NegTst") {
  int deviceCount = 0;
  int pi = -1;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));
  REQUIRE(deviceCount != 0);
  printf("No.of gpus in the system: %d\n", deviceCount);

  int device;
  HIP_CHECK(hipGetDevice(&device));

  // pi is nullptr
  SECTION("pi is nullptr") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(nullptr, hipDeviceAttributePciBusId, device));
  }

  // device is -1
  SECTION("device is -1") {
    REQUIRE_FALSE(hipSuccess == hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, -1));
  }

  // device is Non Existing Device
  SECTION("device is Non Existing Device") {
    REQUIRE_FALSE(hipSuccess ==
                  hipDeviceGetAttribute(&pi, hipDeviceAttributePciBusId, deviceCount));
  }

  // attr is Invalid Attribute
  SECTION("attr is invalid") {
    REQUIRE_FALSE(hipSuccess ==
                  hipDeviceGetAttribute(&pi, static_cast<hipDeviceAttribute_t>(-1), device));
  }
}

template <size_t n> using AttributeToStringMap =
    std::array<std::pair<hipDeviceAttribute_t, const char*>, n>;

namespace {

constexpr AttributeToStringMap<58> kCommonAttributes{
    {{hipDeviceAttributeEccEnabled, "hipDeviceAttributeEccEnabled"},
     {hipDeviceAttributeCanMapHostMemory, "hipDeviceAttributeCanMapHostMemory"},
     {hipDeviceAttributeClockRate, "hipDeviceAttributeClockRate"},
     {hipDeviceAttributeComputeMode, "hipDeviceAttributeComputeMode"},
     {hipDeviceAttributeConcurrentKernels, "hipDeviceAttributeConcurrentKernels"},
     {hipDeviceAttributeConcurrentManagedAccess, "hipDeviceAttributeConcurrentManagedAccess"},
     {hipDeviceAttributeCooperativeLaunch, "hipDeviceAttributeCooperativeLaunch"},
     {hipDeviceAttributeCooperativeMultiDeviceLaunch,
      "hipDeviceAttributeCooperativeMultiDeviceLaunch"},
     {hipDeviceAttributeDirectManagedMemAccessFromHost,
      "hipDeviceAttributeDirectManagedMemAccessFromHost"},
     {hipDeviceAttributeIntegrated, "hipDeviceAttributeIntegrated"},
     {hipDeviceAttributeIsMultiGpuBoard, "hipDeviceAttributeIsMultiGpuBoard"},
     {hipDeviceAttributeKernelExecTimeout, "hipDeviceAttributeKernelExecTimeout"},
     {hipDeviceAttributeL2CacheSize, "hipDeviceAttributeL2CacheSize"},
     {hipDeviceAttributeLocalL1CacheSupported, "hipDeviceAttributeLocalL1CacheSupported"},
     {hipDeviceAttributeComputeCapabilityMajor, "hipDeviceAttributeComputeCapabilityMajor"},
     {hipDeviceAttributeManagedMemory, "hipDeviceAttributeManagedMemory"},
     {hipDeviceAttributeMaxBlockDimX, "hipDeviceAttributeMaxBlockDimX"},
     {hipDeviceAttributeMaxBlockDimY, "hipDeviceAttributeMaxBlockDimY"},
     {hipDeviceAttributeMaxBlockDimZ, "hipDeviceAttributeMaxBlockDimZ"},
     {hipDeviceAttributeMaxGridDimX, "hipDeviceAttributeMaxGridDimX"},
     {hipDeviceAttributeMaxGridDimY, "hipDeviceAttributeMaxGridDimY"},
     {hipDeviceAttributeMaxGridDimZ, "hipDeviceAttributeMaxGridDimZ"},
     {hipDeviceAttributeMaxSurface1D, "hipDeviceAttributeMaxSurface1D"},
     {hipDeviceAttributeMaxSurface2D, "hipDeviceAttributeMaxSurface2D"},
     {hipDeviceAttributeMaxSurface3D, "hipDeviceAttributeMaxSurface3D"},
     {hipDeviceAttributeMaxTexture1DWidth, "hipDeviceAttributeMaxTexture1DWidth"},
     {hipDeviceAttributeMaxTexture1DLinear, "hipDeviceAttributeMaxTexture1DLinear"},
     {hipDeviceAttributeMaxTexture2DWidth, "hipDeviceAttributeMaxTexture2DWidth"},
     {hipDeviceAttributeMaxTexture2DHeight, "hipDeviceAttributeMaxTexture2DHeight"},
     {hipDeviceAttributeMaxTexture3DWidth, "hipDeviceAttributeMaxTexture3DWidth"},
     {hipDeviceAttributeMaxTexture3DHeight, "hipDeviceAttributeMaxTexture3DHeight"},
     {hipDeviceAttributeMaxTexture3DDepth, "hipDeviceAttributeMaxTexture3DDepth"},
     {hipDeviceAttributeMaxThreadsDim, "hipDeviceAttributeMaxThreadsDim"},
     {hipDeviceAttributeMaxThreadsPerBlock, "hipDeviceAttributeMaxThreadsPerBlock"},
     {hipDeviceAttributeMaxThreadsPerMultiProcessor,
      "hipDeviceAttributeMaxThreadsPerMultiProcessor"},
     {hipDeviceAttributeMaxPitch, "hipDeviceAttributeMaxPitch"},
     {hipDeviceAttributeMemoryBusWidth, "hipDeviceAttributeMemoryBusWidth"},
     {hipDeviceAttributeMemoryClockRate, "hipDeviceAttributeMemoryClockRate"},
     {hipDeviceAttributeComputeCapabilityMinor, "hipDeviceAttributeComputeCapabilityMinor"},
     {hipDeviceAttributeMultiprocessorCount, "hipDeviceAttributeMultiprocessorCount"},
     {hipDeviceAttributeUnused1, "hipDeviceAttributeUnused1"},
     {hipDeviceAttributePageableMemoryAccess, "hipDeviceAttributePageableMemoryAccess"},
     {hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
      "hipDeviceAttributePageableMemoryAccessUsesHostPageTables"},
     {hipDeviceAttributePciBusId, "hipDeviceAttributePciBusId"},
     {hipDeviceAttributePciDeviceId, "hipDeviceAttributePciDeviceId"},
     {hipDeviceAttributePciDomainID, "hipDeviceAttributePciDomainID"},
     {hipDeviceAttributeMaxRegistersPerBlock, "hipDeviceAttributeMaxRegistersPerBlock"},
     {hipDeviceAttributeMaxRegistersPerMultiprocessor,
      "hipDeviceAttributeMaxRegistersPerMultiprocessor"},
     {hipDeviceAttributeMaxSharedMemoryPerBlock, "hipDeviceAttributeMaxSharedMemoryPerBlock"},
     {hipDeviceAttributeTextureAlignment, "hipDeviceAttributeTextureAlignment"},
     {hipDeviceAttributeTexturePitchAlignment, "hipDeviceAttributeTexturePitchAlignment"},
     {hipDeviceAttributeTotalConstantMemory, "hipDeviceAttributeTotalConstantMemory"},
     {hipDeviceAttributeTotalGlobalMem, "hipDeviceAttributeTotalGlobalMem"},
     {hipDeviceAttributeWarpSize, "hipDeviceAttributeWarpSize"},
     {hipDeviceAttributeMemoryPoolsSupported, "hipDeviceAttributeMemoryPoolsSupported"},
     {hipDeviceAttributeUnifiedAddressing, "hipDeviceAttributeUnifiedAddressing"},
     {hipDeviceAttributeVirtualMemoryManagementSupported,
      "hipDeviceAttributeVirtualMemoryManagementSupported"},
     {hipDeviceAttributeHostRegisterSupported, "hipDeviceAttributeHostRegisterSupported"}}};

#if HT_NVIDIA
constexpr AttributeToStringMap<33> kCudaOnlyAttributes{
    {{hipDeviceAttributeAccessPolicyMaxWindowSize, "hipDeviceAttributeAccessPolicyMaxWindowSize"},
     {hipDeviceAttributeAsyncEngineCount, "hipDeviceAttributeAsyncEngineCount"},
     {hipDeviceAttributeCanUseHostPointerForRegisteredMem,
      "hipDeviceAttributeCanUseHostPointerForRegisteredMem"},
     {hipDeviceAttributeComputePreemptionSupported, "hipDeviceAttributeComputePreemptionSupported"},
     {hipDeviceAttributeDeviceOverlap, "hipDeviceAttributeDeviceOverlap"},
     {hipDeviceAttributeGlobalL1CacheSupported, "hipDeviceAttributeGlobalL1CacheSupported"},
     {hipDeviceAttributeHostNativeAtomicSupported, "hipDeviceAttributeHostNativeAtomicSupported"},
     {hipDeviceAttributeLuid, "hipDeviceAttributeLuid"},
     {hipDeviceAttributeLuidDeviceNodeMask, "hipDeviceAttributeLuidDeviceNodeMask"},
     {hipDeviceAttributeMaxBlocksPerMultiProcessor, "hipDeviceAttributeMaxBlocksPerMultiProcessor"},
     {hipDeviceAttributeMaxSurface1DLayered, "hipDeviceAttributeMaxSurface1DLayered"},
     {hipDeviceAttributeMaxSurface2DLayered, "hipDeviceAttributeMaxSurface2DLayered"},
     {hipDeviceAttributeMaxSurfaceCubemap, "hipDeviceAttributeMaxSurfaceCubemap"},
     {hipDeviceAttributeMaxSurfaceCubemapLayered, "hipDeviceAttributeMaxSurfaceCubemapLayered"},
     {hipDeviceAttributeMaxTexture1DLayered, "hipDeviceAttributeMaxTexture1DLayered"},
     {hipDeviceAttributeMaxTexture1DMipmap, "hipDeviceAttributeMaxTexture1DMipmap"},
     {hipDeviceAttributeMaxTexture2DGather, "hipDeviceAttributeMaxTexture2DGather"},
     {hipDeviceAttributeMaxTexture2DLayered, "hipDeviceAttributeMaxTexture2DLayered"},
     {hipDeviceAttributeMaxTexture2DLinear, "hipDeviceAttributeMaxTexture2DLinear"},
     {hipDeviceAttributeMaxTexture2DMipmap, "hipDeviceAttributeMaxTexture2DMipmap"},
     {hipDeviceAttributeMaxTexture3DAlt, "hipDeviceAttributeMaxTexture3DAlt"},
     {hipDeviceAttributeMaxTextureCubemap, "hipDeviceAttributeMaxTextureCubemap"},
     {hipDeviceAttributeMaxTextureCubemapLayered, "hipDeviceAttributeMaxTextureCubemapLayered"},
     {hipDeviceAttributeMultiGpuBoardGroupID, "hipDeviceAttributeMultiGpuBoardGroupID"},
     {hipDeviceAttributePersistingL2CacheMaxSize, "hipDeviceAttributePersistingL2CacheMaxSize"},
     {hipDeviceAttributeReservedSharedMemPerBlock, "hipDeviceAttributeReservedSharedMemPerBlock"},
     {hipDeviceAttributeSharedMemPerBlockOptin, "hipDeviceAttributeSharedMemPerBlockOptin"},
     {hipDeviceAttributeSharedMemPerMultiprocessor, "hipDeviceAttributeSharedMemPerMultiprocessor"},
     {hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
      "hipDeviceAttributeSingleToDoublePrecisionPerfRatio"},
     {hipDeviceAttributeStreamPrioritiesSupported, "hipDeviceAttributeStreamPrioritiesSupported"},
     {hipDeviceAttributeSurfaceAlignment, "hipDeviceAttributeSurfaceAlignment"},
     {hipDeviceAttributeTccDriver, "hipDeviceAttributeTccDriver"},
     {hipDeviceAttributeUnused2, "hipDeviceAttributeUnused2"}}};
#endif

#if HT_AMD
constexpr AttributeToStringMap<18> kAmdOnlyAttributes{{
    {hipDeviceAttributeClockInstructionRate, "hipDeviceAttributeClockInstructionRate"},
    {hipDeviceAttributeUnused3, "hipDeviceAttributeUnused3"},
    {hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
     "hipDeviceAttributeMaxSharedMemoryPerMultiprocessor"},
    {hipDeviceAttributeUnused4, "hipDeviceAttributeUnused4"},
    {hipDeviceAttributeUnused5, "hipDeviceAttributeUnused5"},
    {hipDeviceAttributeHdpMemFlushCntl, "hipDeviceAttributeHdpMemFlushCntl"},
    {hipDeviceAttributeHdpRegFlushCntl, "hipDeviceAttributeHdpRegFlushCntl"},
    {hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
     "hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc"},
    {hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
     "hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim"},
    {hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
     "hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim"},
    {hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
     "hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem"},
    {hipDeviceAttributeIsLargeBar, "hipDeviceAttributeIsLargeBar"},
    {hipDeviceAttributeAsicRevision, "hipDeviceAttributeAsicRevision"},
    {hipDeviceAttributeCanUseStreamWaitValue, "hipDeviceAttributeCanUseStreamWaitValue"},
    {hipDeviceAttributeImageSupport, "hipDeviceAttributeImageSupport"},
    {hipDeviceAttributePhysicalMultiProcessorCount,
     "hipDeviceAttributePhysicalMultiProcessorCount"},
    {hipDeviceAttributeFineGrainSupport, "hipDeviceAttributeFineGrainSupport"},
    {hipDeviceAttributeNumberOfXccs, "hipDeviceAttributeNumberOfXccs"}
    // {hipDeviceAttributeWallClockRate, "hipDeviceAttributeWallClockRate"}
}};
#endif

constexpr int kW = 60;

}  // anonymous namespace

template <size_t n>
void printAttributes(const AttributeToStringMap<n>& attributes, const int device) {
  hipError_t ret_val;
  for (const auto& attribute : attributes) {
    int64_t attribute_value = 0;
    ret_val =
        hipDeviceGetAttribute(reinterpret_cast<int*>(&attribute_value), attribute.first, device);
    std::cout << std::setw(kW) << std::string(attribute.second).append(": ");
    if (ret_val == hipSuccess)
      std::cout << attribute_value << "\n";
    else
      std::cout << "unsupported\n";
  }
  std::flush(std::cout);
}

/**
 * Test Description
 * ------------------------
 *  - Print out all device attributes in agreed upon format.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Print_Out_Attributes") {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  hipDeviceProp_t properties;
  HIP_CHECK(hipGetDeviceProperties(&properties, device));

  std::cout << std::left;
  std::cout << std::setw(kW) << "device#: " << device << "\n";
  std::cout << std::setw(kW) << "name: " << properties.name << "\n";

  printAttributes(kCommonAttributes, device);

#if HT_NVIDIA
  std::cout << "\nCUDA only\n";
  std::cout << std::setw(kW)
            << "--------------------------------------------------------------------------------"
            << "\n";
  printAttributes(kCudaOnlyAttributes, device);
#endif

#if HT_AMD
  std::cout << "\nAMD only\n";
  std::cout << std::setw(kW)
            << "--------------------------------------------------------------------------------"
            << "\n";
  printAttributes(kAmdOnlyAttributes, device);
#endif

  std::flush(std::cout);
}

/**
 * Test Description
 * ------------------------
 *  - verify hipDeviceAttributeHostRegisterSupported attribute.
 * Test source
 * ------------------------
 *  - unit/device/hipGetDeviceAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGetDeviceAttribute_hipDevAttrHostRegisterSupported") {
  hipError_t ret_val;
  int hipDevAttr = 0;
  ret_val = hipDeviceGetAttribute(&hipDevAttr, hipDeviceAttributeHostRegisterSupported, 0);
  INFO("hipDeviceAttributeHostRegisterSupported: " << hipDevAttr);

  if (ret_val == hipSuccess) {
    auto x = std::unique_ptr<int>(new int);
    HIP_CHECK(hipHostRegister(x.get(), sizeof(int), hipHostRegisterDefault));

    void* device_memory;
    HIP_CHECK(hipHostGetDevicePointer(&device_memory, x.get(), 0));

    HIP_CHECK(hipHostUnregister(x.get()));
    HIP_CHECK_ERROR(hipHostGetDevicePointer(&device_memory, x.get(), 0), hipErrorInvalidValue);
  } else {
    HipTest::HIP_SKIP_TEST(
        "Skipping the test as GPU 0 doesn't support "
        "hipDeviceAttributeHostRegisterSupported attribute.\n");
    return;
  }
}

/**
 * End doxygen group DeviceTest.
 * @}
 */
