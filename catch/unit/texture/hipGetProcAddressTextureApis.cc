/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <utils.hh>

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of texture API's from the
 *  - hipGetProcAddress API and then validates the basic functionality of
 *  - those APIs using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/texture/hipGetProcAddressTextureAPIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_Texture_ChannelDesc") {
  CHECK_IMAGE_SUPPORT

  void *hipCreateChannelDesc_ptr = nullptr;
  void *hipGetChannelDesc_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipCreateChannelDesc", &hipCreateChannelDesc_ptr,
                              currentHipVersion, 0, nullptr));
  REQUIRE(hipCreateChannelDesc_ptr != nullptr);

  HIP_CHECK(hipGetProcAddress("hipGetChannelDesc", &hipGetChannelDesc_ptr,
                              currentHipVersion, 0, nullptr));
  REQUIRE(hipGetChannelDesc_ptr != nullptr);

  hipChannelFormatDesc (*dyn_hipCreateChannelDesc_ptr)(
      int x, int y, int z, int w, hipChannelFormatKind f) =
      reinterpret_cast<hipChannelFormatDesc (*)(int x, int y, int z, int w,
                                                hipChannelFormatKind f)>(
          hipCreateChannelDesc_ptr);
  REQUIRE(dyn_hipCreateChannelDesc_ptr != nullptr);

  hipError_t (*dyn_hipGetChannelDesc_ptr)(hipChannelFormatDesc * desc,
                                          hipArray_const_t array) =
      reinterpret_cast<hipError_t (*)(hipChannelFormatDesc * desc,
                                      hipArray_const_t array)>(
          hipGetChannelDesc_ptr);
  REQUIRE(dyn_hipGetChannelDesc_ptr != nullptr);

  // Validating hipCreateChannelDesc API
  hipChannelFormatDesc channelFormatDesc;
  int x = 32, y = 32, z = 32, w = 32;

  hipChannelFormatKind kind =
      GENERATE(hipChannelFormatKindSigned, hipChannelFormatKindUnsigned,
               hipChannelFormatKindFloat);

  channelFormatDesc = dyn_hipCreateChannelDesc_ptr(x, y, z, w, kind);

  REQUIRE(channelFormatDesc.x == 32);
  REQUIRE(channelFormatDesc.y == 32);
  REQUIRE(channelFormatDesc.z == 32);
  REQUIRE(channelFormatDesc.w == 32);
  REQUIRE(channelFormatDesc.f == kind);

  // Validating hipGetChannelDesc API
  hipArray_t array;
  HIP_CHECK(hipMallocArray(&array, &channelFormatDesc, 8, 8, 0));

  hipChannelFormatDesc desc;
  HIP_CHECK(dyn_hipGetChannelDesc_ptr(&desc, array));

  REQUIRE(desc.x == 32);
  REQUIRE(desc.y == 32);
  REQUIRE(desc.z == 32);
  REQUIRE(desc.w == 32);
  REQUIRE(desc.f == kind);

  HIP_CHECK(hipFreeArray(array));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of texture API's from the
 *  - hipGetProcAddress API and then validates the basic functionality of
 *  - those APIs using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/texture/hipGetProcAddressTextureAPIs.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.2
 */
TEST_CASE("Unit_hipGetProcAddress_Texture_TextureObject") {
  CHECK_IMAGE_SUPPORT

  void *hipGetTextureObjectResourceDesc_ptr = nullptr;
  void *hipGetTextureObjectTextureDesc_ptr = nullptr;
  void *hipGetTextureObjectResourceViewDesc_ptr = nullptr;
  void *hipDestroyTextureObject_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipGetTextureObjectResourceDesc",
                              &hipGetTextureObjectResourceDesc_ptr,
                              currentHipVersion, 0, nullptr));
  REQUIRE(hipGetTextureObjectResourceDesc_ptr != nullptr);

  HIP_CHECK(hipGetProcAddress("hipGetTextureObjectTextureDesc",
                              &hipGetTextureObjectTextureDesc_ptr,
                              currentHipVersion, 0, nullptr));
  REQUIRE(hipGetTextureObjectTextureDesc_ptr != nullptr);

  HIP_CHECK(hipGetProcAddress("hipGetTextureObjectResourceViewDesc",
                              &hipGetTextureObjectResourceViewDesc_ptr,
                              currentHipVersion, 0, nullptr));
  REQUIRE(hipGetTextureObjectResourceViewDesc_ptr != nullptr);

  HIP_CHECK(hipGetProcAddress("hipDestroyTextureObject",
                              &hipDestroyTextureObject_ptr, currentHipVersion,
                              0, nullptr));
  REQUIRE(hipDestroyTextureObject_ptr != nullptr);

  hipError_t (*dyn_hipGetTextureObjectResourceDesc_ptr)(hipResourceDesc *,
                                                        hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(hipResourceDesc *, hipTextureObject_t)>(
          hipGetTextureObjectResourceDesc_ptr);
  REQUIRE(dyn_hipGetTextureObjectResourceDesc_ptr != nullptr);

  hipError_t (*dyn_hipGetTextureObjectTextureDesc_ptr)(hipTextureDesc *,
                                                       hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(hipTextureDesc *, hipTextureObject_t)>(
          hipGetTextureObjectTextureDesc_ptr);
  REQUIRE(dyn_hipGetTextureObjectTextureDesc_ptr != nullptr);

  hipError_t (*dyn_hipGetTextureObjectResourceViewDesc_ptr)(
      struct hipResourceViewDesc *, hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(struct hipResourceViewDesc *,
                                      hipTextureObject_t)>(
          hipGetTextureObjectResourceViewDesc_ptr);
  REQUIRE(dyn_hipGetTextureObjectResourceViewDesc_ptr != nullptr);

  hipError_t (*dyn_hipDestroyTextureObject_ptr)(hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(hipTextureObject_t)>(
          hipDestroyTextureObject_ptr);
  REQUIRE(dyn_hipDestroyTextureObject_ptr != nullptr);

  int width = 128;
  size_t size = width * sizeof(float);

  float *hostData = reinterpret_cast<float *>(malloc(size));
  memset(hostData, 0, size);
  for (int i = 0; i < width; i++) {
    hostData[i] = i;
  }

  // Create Channel Format Descriptor
  hipChannelFormatDesc channelFormatDesc;
  channelFormatDesc = hipCreateChannelDesc(32, 0, 0, 0,
                                           hipChannelFormatKindFloat);

  // Create array
  hipArray_t array;
  HIP_CHECK(hipMallocArray(&array, &channelFormatDesc, width));
  HIP_CHECK(hipMemcpy2DToArray(array, 0, 0, hostData, size, size, 1,
                               hipMemcpyHostToDevice));

  // Create Resource Descriptor
  hipResourceDesc resourceDesc;
  memset(&resourceDesc, 0, sizeof(resourceDesc));
  resourceDesc.resType = hipResourceTypeArray;
  resourceDesc.res.array.array = array;

  // Create Texture Descriptor
  hipTextureDesc textureDesc;
  memset(&textureDesc, 0, sizeof(textureDesc));
  textureDesc.addressMode[0] = hipAddressModeClamp;
  textureDesc.filterMode = hipFilterModePoint;
  textureDesc.readMode = hipReadModeElementType;
  textureDesc.normalizedCoords = false;

  // Create Resource View Descriptor
  hipResourceViewDesc resourceViewDesc;
  memset(&resourceViewDesc, 0, sizeof(resourceViewDesc));
  resourceViewDesc.format = hipResViewFormatFloat1;
  resourceViewDesc.width = size;

  // Create Texture Object
  hipTextureObject_t textureObject = nullptr;
  HIP_CHECK(hipCreateTextureObject(&textureObject, &resourceDesc, &textureDesc,
                                   &resourceViewDesc));
  REQUIRE(textureObject != nullptr);

  // Validating hipGetTextureObjectResourceDesc
  SECTION("hipGetTextureObjectResourceDesc") {
    hipResourceDesc resourceDescToCheck;
    HIP_CHECK(dyn_hipGetTextureObjectResourceDesc_ptr(&resourceDescToCheck,
                                                      textureObject));

    REQUIRE(resourceDescToCheck.resType == hipResourceTypeArray);
    REQUIRE(resourceDescToCheck.res.array.array == array);
  }

  // Validating hipGetTextureObjectTextureDesc
  SECTION("hipGetTextureObjectTextureDesc") {
    hipTextureDesc textureDescToCheck;
    HIP_CHECK(dyn_hipGetTextureObjectTextureDesc_ptr(&textureDescToCheck,
                                                     textureObject));

    REQUIRE(textureDescToCheck.addressMode[0] == hipAddressModeClamp);
    REQUIRE(textureDescToCheck.filterMode == hipFilterModePoint);
    REQUIRE(textureDescToCheck.readMode == hipReadModeElementType);
    REQUIRE(textureDescToCheck.normalizedCoords == false);
  }

  // Validating hipGetTextureObjectResourceViewDesc
  SECTION("hipGetTextureObjectResourceViewDesc") {
    hipResourceViewDesc resourceViewDescToCheck;
    HIP_CHECK(dyn_hipGetTextureObjectResourceViewDesc_ptr(
        &resourceViewDescToCheck, textureObject));

    REQUIRE(resourceViewDescToCheck.format == hipResViewFormatFloat1);
    REQUIRE(resourceViewDescToCheck.width == size);
  }

  // Destroy Texture Object
  // Validating hipDestroyTextureObject
  HIP_CHECK(dyn_hipDestroyTextureObject_ptr(textureObject));

  HIP_CHECK(hipFreeArray(array));
}

/**
 * Test Description
 * ------------------------
 *  - This test will get the function pointer of hipTexObjectCreate
 *  - hipTexObjectDestroy, hipTexObjectGetResourceDesc,
 *  - hipTexObjectGetResourceViewDesc, hipTexObjectGetTextureDesc
 *  - API's from the hipGetProcAddress API and then validates the basic
 *  - functionality of all those API's using the funtion pointer.
 * Test source
 * ------------------------
 *  - unit/texture/hipTexObjectTests_hipGetProcAddress.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGetProcAddress_Texture_TexObjectRelated") {
  CHECK_IMAGE_SUPPORT

  void *hipTexObjectCreate_ptr = nullptr;
  void *hipTexObjectDestroy_ptr = nullptr;
  void *hipGetTexObjectResourceDesc_ptr = nullptr;
  void *hipTexObjectGetResourceViewDesc_ptr = nullptr;
  void *hipTexObjectGetTextureDesc_ptr = nullptr;

  int currentHipVersion = 0;
  HIP_CHECK(hipRuntimeGetVersion(&currentHipVersion));

  HIP_CHECK(hipGetProcAddress("hipTexObjectCreate", &hipTexObjectCreate_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipTexObjectDestroy", &hipTexObjectDestroy_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipTexObjectGetResourceDesc",
                              &hipGetTexObjectResourceDesc_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipTexObjectGetResourceViewDesc",
                              &hipTexObjectGetResourceViewDesc_ptr,
                              currentHipVersion, 0, nullptr));
  HIP_CHECK(hipGetProcAddress("hipTexObjectGetTextureDesc",
                              &hipTexObjectGetTextureDesc_ptr,
                              currentHipVersion, 0, nullptr));

  hipError_t (*dyn_hipTexObjectCreate_ptr)(
      hipTextureObject_t *, const HIP_RESOURCE_DESC *, const HIP_TEXTURE_DESC *,
      const HIP_RESOURCE_VIEW_DESC *) =
      reinterpret_cast<hipError_t (*)(hipTextureObject_t *,
          const HIP_RESOURCE_DESC *, const HIP_TEXTURE_DESC *,
          const HIP_RESOURCE_VIEW_DESC *)>(
          hipTexObjectCreate_ptr);

  hipError_t (*dyn_hipTexObjectDestroy_ptr)(hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(hipTextureObject_t)>(
          hipTexObjectDestroy_ptr);

  hipError_t (*dyn_hipGetTexObjectResourceDesc_ptr)(HIP_RESOURCE_DESC *,
                                                    hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(HIP_RESOURCE_DESC *, hipTextureObject_t)>(
          hipGetTexObjectResourceDesc_ptr);

  hipError_t (*dyn_hipTexObjectGetResourceViewDesc_ptr)(
      HIP_RESOURCE_VIEW_DESC *, hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(HIP_RESOURCE_VIEW_DESC *,
                                      hipTextureObject_t)>(
          hipTexObjectGetResourceViewDesc_ptr);

  hipError_t (*dyn_hipTexObjectGetTextureDesc_ptr)(HIP_TEXTURE_DESC *,
                                                   hipTextureObject_t) =
      reinterpret_cast<hipError_t (*)(HIP_TEXTURE_DESC *, hipTextureObject_t)>(
          hipTexObjectGetTextureDesc_ptr);

  float *hostData;
  hipTextureObject_t textureObject = 0;
  HIP_RESOURCE_DESC resDesc;
  HIP_TEXTURE_DESC texDesc;
  HIP_RESOURCE_VIEW_DESC resViewDesc;
  HIP_ARRAY_DESCRIPTOR arrayDesc;
  hipArray_t array_member;
  size_t size;
  int width = 128;
  size = width * sizeof(float);
  hostData = reinterpret_cast<float *>(malloc(size));
  memset(hostData, 0, size);
  for (int i = 0; i < width; i++) {
    hostData[i] = i;
  }
  memset(&arrayDesc, 0, sizeof(arrayDesc));
  arrayDesc.Format = HIP_AD_FORMAT_FLOAT;
  arrayDesc.NumChannels = 1;
  arrayDesc.Width = width;
  arrayDesc.Height = 0;
  HIP_CHECK(hipArrayCreate(&array_member, &arrayDesc));
  HIP_CHECK(hipMemcpyHtoA(reinterpret_cast<hipArray_t>(array_member), 0,
                          hostData, size));

  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = HIP_RESOURCE_TYPE_ARRAY;
  resDesc.res.array.hArray = array_member;
  resDesc.flags = 0;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.filterMode = HIP_TR_FILTER_MODE_POINT;
  texDesc.flags = 0;
  memset(&resViewDesc, 0, sizeof(resViewDesc));
  resViewDesc.format = HIP_RES_VIEW_FORMAT_FLOAT_1X32;
  resViewDesc.width = size;

  // Validating hipTexObjectCreate API
  HIP_CHECK(dyn_hipTexObjectCreate_ptr(&textureObject, &resDesc, &texDesc,
                                       &resViewDesc));

  // Validating hipGetTexObjectResourceDesc API
  HIP_RESOURCE_DESC checkResDesc;
  memset(&checkResDesc, 0, sizeof(checkResDesc));

  HIP_CHECK(dyn_hipGetTexObjectResourceDesc_ptr(&checkResDesc, textureObject));

  REQUIRE(checkResDesc.resType == HIP_RESOURCE_TYPE_ARRAY);
  REQUIRE(checkResDesc.res.array.hArray == array_member);

  // Validating hipTexObjectGetResourceViewDesc API
  HIP_RESOURCE_VIEW_DESC checkResViewDesc;
  memset(&checkResViewDesc, 0, sizeof(checkResViewDesc));

  HIP_CHECK(dyn_hipTexObjectGetResourceViewDesc_ptr(&checkResViewDesc,
                                                    textureObject));

  REQUIRE(checkResViewDesc.format == HIP_RES_VIEW_FORMAT_FLOAT_1X32);
  REQUIRE(checkResViewDesc.width == size);

  // Validating hipTexObjectGetTextureDesc API
  HIP_TEXTURE_DESC checkTexDesc;
  memset(&checkTexDesc, 0, sizeof(checkTexDesc));

  HIP_CHECK(dyn_hipTexObjectGetTextureDesc_ptr(&checkTexDesc,
                                               textureObject));

  REQUIRE(checkTexDesc.filterMode == HIP_TR_FILTER_MODE_POINT);
  REQUIRE(checkTexDesc.flags == 0);

  // Validating hipTexObjectDestroy API
  HIP_CHECK(dyn_hipTexObjectDestroy_ptr(textureObject));
}
