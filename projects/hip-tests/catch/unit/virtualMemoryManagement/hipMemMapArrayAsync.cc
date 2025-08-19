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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipMemMapArrayAsync hipMemMapArrayAsync
 * @{
 * @ingroup VirtualMemoryManagementTest
 * `hipError_t hipMemMapArrayAsync(hipArrayMapInfo *mapInfoList,
 *                                 unsigned int count,
 *                                 hipStream_t stream)` -
 * 	Maps or unmaps subregions of sparse HIP arrays and sparse HIP mipmapped arrays.
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

#include "hip_vmm_common.hh"

/**
 * Test Description
 * ------------------------
 *    - Basic sanity test.
 * ------------------------
 *    - unit/virtualMemoryManagement/hipMemMapArrayAsync.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipMemMapArrayAsync_Positive_Basic") {
  HIP_CHECK(hipFree(0));

  hipDevice_t device;
  HIP_CHECK(hipDeviceGet(&device, 0));
  checkVMMSupported(device);

  CHECK_IMAGE_SUPPORT;

  hipmipmappedArray array;

  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<float>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 1;
  desc.Height = 1;
  desc.Flags = CUDA_ARRAY3D_SPARSE;

  unsigned int levels = 2;

  HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.allocFlags.usage = CU_MEM_CREATE_USAGE_TILE_POOL;

  size_t granularity;
  HIP_CHECK(
      hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityRecommended));

  hipMemGenericAllocationHandle_t handle;
  HIP_CHECK(hipMemCreate(&handle, granularity, &prop, 0));

  hipArrayMapInfo map_info_list = {};
  map_info_list.resourceType = HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;
  map_info_list.resource.mipmap = array;
  map_info_list.subresourceType = hipArraySparseSubresourceTypeSparseLevel;
  map_info_list.subresource.sparseLevel.extentWidth = 1;
  map_info_list.subresource.sparseLevel.extentHeight = 1;
  map_info_list.subresource.sparseLevel.extentDepth = 1;
  map_info_list.memOperationType = hipMemOperationTypeMap;
  map_info_list.memHandleType = hipMemHandleTypeGeneric;
  map_info_list.memHandle.memHandle = handle;
  map_info_list.deviceBitMask = 0x1;

  StreamGuard stream(Streams::created);

  HIP_CHECK(hipMemMapArrayAsync(&map_info_list, 1, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  map_info_list.memOperationType = hipMemOperationTypeUnmap;
  map_info_list.memHandle.memHandle = NULL;
  HIP_CHECK(hipMemMapArrayAsync(&map_info_list, 1, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  HIP_CHECK(hipMemRelease(handle));

  HIP_CHECK(hipMipmappedArrayDestroy(array));
}

/**
 * End doxygen group VirtualMemoryManagementTest.
 * @}
 */
