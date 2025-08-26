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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>


class TextureObjectTestWrapper {
 private:
  float* host_data_;
  bool ommit_destroy_;

 public:
  hipTextureObject_t texture_object = 0;
  hipResourceDesc res_desc;
  hipTextureDesc tex_desc;
  hipChannelFormatDesc channel_desc;
  hipResourceViewDesc res_vew_desc;
  hipArray_t array_member;
  size_t size; /* size in bytes*/
  int width;   /* width in elements */

  TextureObjectTestWrapper(bool useResourceViewDescriptor, bool ommitDestroy = false)
      : ommit_destroy_(ommitDestroy), width(128) {
    int i;
    size = width * sizeof(float);

    host_data_ = (float*)malloc(size);
    memset(host_data_, 0, size);

    for (i = 0; i < width; i++) {
      host_data_[i] = i;
    }

    channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    HIP_CHECK(hipMallocArray(&array_member, &channel_desc, width));

    HIP_CHECK(
        hipMemcpy2DToArray(array_member, 0, 0, host_data_, size, size, 1, hipMemcpyHostToDevice));

    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = hipResourceTypeArray;
    res_desc.res.array.array = array_member;

    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = hipAddressModeClamp;
    tex_desc.filterMode = hipFilterModePoint;
    tex_desc.readMode = hipReadModeElementType;
    tex_desc.normalizedCoords = false;

    memset(&res_vew_desc, 0, sizeof(res_vew_desc));

    if (useResourceViewDescriptor) {
#if HT_AMD
      res_vew_desc.format = hipResViewFormatFloat1;
      res_vew_desc.width = size;
#else
      std::cout << "Resource View Descriptors are not supported on NVIDIA currently" << std::endl;
      useResourceViewDescriptor = false;
#endif
    }


    HIP_CHECK(hipCreateTextureObject(&texture_object, &res_desc, &tex_desc,
                                     useResourceViewDescriptor ? &res_vew_desc : nullptr));
  }

  ~TextureObjectTestWrapper() {
    if (!ommit_destroy_) {
      HIP_CHECK(hipDestroyTextureObject(texture_object));
    }
    HIP_CHECK(hipFreeArray(array_member));
    free(host_data_);
  }
};

/**
 * @addtogroup hipGetTextureObjectResourceDesc hipGetTextureObjectResourceDesc
 * @{
 * @ingroup TextureTest
 * `hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
 * hipTextureObject_t textureObject)` -
 * Gets resource descriptor for the texture object.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates new texture object and an empty resource descriptor.
 *  - Gets resource descriptor from the texture.
 *  - Compares it to the empty created one.
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetTextureObjectResourceDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(false);

  hipResourceDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipGetTextureObjectResourceDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.resType == tex_obj_wrapper.res_desc.resType);
  REQUIRE(check_desc.res.array.array == tex_obj_wrapper.res_desc.res.array.array);
}

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the resource descriptor is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the texture is not valid (0)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGetTextureObjectResourceDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(false);

  hipResourceDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectResourceDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(
        hipGetTextureObjectResourceDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
        hipErrorInvalidValue);
  }
}

/**
 * End doxygen group hipGetTextureObjectResourceDesc.
 * @}
 */

/**
 * @addtogroup hipGetTextureObjectResourceViewDesc hipGetTextureObjectResourceViewDesc
 * @{
 * @ingroup TextureTest
 * `hipGetTextureObjectResourceViewDesc(struct hipResourceViewDesc* pResViewDesc,
 * hipTextureObject_t textureObject)` -
 * Gets resource view descriptor for the texture object.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates new texture object and an empty resource view descriptor.
 *  - Gets resource view descriptor from the texture.
 *  - Compares it to the empty created one.
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEST_CASE("Unit_hipGetTextureObjectResourceViewDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(true);

  hipResourceViewDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipGetTextureObjectResourceViewDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.format == tex_obj_wrapper.res_vew_desc.format);
  REQUIRE(check_desc.width == tex_obj_wrapper.res_vew_desc.width);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the resource view descriptor is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the texture is not valid (0)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEST_CASE("Unit_hipGetTextureObjectResourceViewDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(true);

  hipResourceViewDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectResourceViewDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(
        hipGetTextureObjectResourceViewDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
        hipErrorInvalidValue);
  }

  HipTest::HIP_SKIP_TEST("Skipping on NVIDIA platform");
}
#endif

/**
 * End doxygen group hipGetTextureObjectResourceViewDesc.
 * @}
 */

/**
 * @addtogroup hipGetTextureObjectTextureDesc hipGetTextureObjectTextureDesc
 * @{
 * @ingroup TextureTest
 * `hipGetTextureObjectTextureDesc(hipTextureDesc* pTexDesc,
 * hipTextureObject_t textureObject)` -
 * Gets texture descriptor for the texture object.
 */

/**
 * Test Description
 * ------------------------
 *  - Creates new texture object and an empty texture descriptor.
 *  - Gets texture descriptor from the texture.
 *  - Compares it to the empty created one.
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEST_CASE("Unit_hipGetTextureObjectTextureDesc_positive") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(false);

  hipTextureDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  HIP_CHECK(hipGetTextureObjectTextureDesc(&check_desc, tex_obj_wrapper.texture_object));

  REQUIRE(check_desc.addressMode[0] == tex_obj_wrapper.tex_desc.addressMode[0]);
  REQUIRE(check_desc.filterMode == tex_obj_wrapper.tex_desc.filterMode);
  REQUIRE(check_desc.readMode == tex_obj_wrapper.tex_desc.readMode);
  REQUIRE(check_desc.normalizedCoords == tex_obj_wrapper.tex_desc.normalizedCoords);
}
#endif

/**
 * Test Description
 * ------------------------
 *  - Validates handling of invalid arguments:
 *    -# When output pointer to the texture descriptor is `nullptr`
 *      - Expected output: return `hipErrorInvalidValue`
 *    -# When the texture is not valid (0)
 *      - Expected output: return `hipErrorInvalidValue`
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - Platform specific (AMD)
 *  - HIP_VERSION >= 5.2
 */
#if HT_AMD
TEST_CASE("Unit_hipGetTextureObjectTextureDesc_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(false);

  hipTextureDesc check_desc;
  memset(&check_desc, 0, sizeof(check_desc));

  SECTION("desc is nullptr") {
    HIP_CHECK_ERROR(hipGetTextureObjectTextureDesc(nullptr, tex_obj_wrapper.texture_object),
                    hipErrorInvalidValue);
  }

  SECTION("texture is invalid") {
    HIP_CHECK_ERROR(hipGetTextureObjectTextureDesc(&check_desc, static_cast<hipTextureObject_t>(0)),
                    hipErrorInvalidValue);
  }
}
#endif

/**
 * End doxygen group hipGetTextureObjectTextureDesc.
 * @}
 */

/**
 * @addtogroup hipDestroyTextureObject hipDestroyTextureObject
 * @{
 * @ingroup TextureTest
 * `hipDestroyTextureObject(hipTextureObject_t textureObject)` -
 * Destroys a texture object.
 * ________________________
 * Test cases from other modules:
 *  - @ref Unit_hipCreateTextureObject_ArgValidation
 */

/**
 * Test Description
 * ------------------------
 *  - Successfully destroys regular texture object.
 * Test source
 * ------------------------
 *  - unit/texture/hipTextureObjectTests.cc
 * Test requirements
 * ------------------------
 *  - Textures supported on device
 *  - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipDestroyTextureObject_positive") {
  CHECK_IMAGE_SUPPORT;

  TextureObjectTestWrapper tex_obj_wrapper(false, true);
  REQUIRE(hipDestroyTextureObject(tex_obj_wrapper.texture_object) == hipSuccess);
}
