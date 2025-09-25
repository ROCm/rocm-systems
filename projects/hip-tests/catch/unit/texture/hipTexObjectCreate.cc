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

#define N 256

TEST_CASE("Unit_TexObjectCreate_NullptrParams") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  HIP_RESOURCE_VIEW_DESC res_view_desc{};

  SECTION("tex_object can't be nullptr") {
    HIP_CHECK_ERROR(hipTexObjectCreate(nullptr, &res_desc, &tex_desc, &res_view_desc),
                    hipErrorInvalidValue);
  }

  SECTION("res_desc can't be nullptr") {
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, nullptr, &tex_desc, &res_view_desc),
                    hipErrorInvalidValue);
  }

  SECTION("tex_desc can't be nullptr") {
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, nullptr, &res_view_desc),
                    hipErrorInvalidValue);
  }

  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeLinear") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  size_t array_size = N * sizeof(float);

  HIP_CHECK(hipMalloc(&tex_buffer, array_size));

  res_desc.resType = HIP_RESOURCE_TYPE_LINEAR;

  res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);

  res_desc.res.linear.format =
      GENERATE(HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16,
               HIP_AD_FORMAT_UNSIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT8, HIP_AD_FORMAT_SIGNED_INT16,
               HIP_AD_FORMAT_SIGNED_INT32, HIP_AD_FORMAT_HALF, HIP_AD_FORMAT_FLOAT);

  res_desc.res.linear.numChannels = GENERATE(1, 2, 4);
  res_desc.res.linear.sizeInBytes = array_size;

  SECTION("regular setup") {
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  }

  SECTION("size_in_bytes set to 0") {
    res_desc.res.linear.sizeInBytes = 0;
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  }

  HIP_CHECK(hipTexObjectDestroy(tex_object));
  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeLinear_IncompleteInit") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  size_t array_size = N * sizeof(float);

  res_desc.resType = HIP_RESOURCE_TYPE_LINEAR;

  HIP_CHECK(hipMalloc(&tex_buffer, array_size));

  auto formats =
      GENERATE(HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16,
               HIP_AD_FORMAT_UNSIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT8, HIP_AD_FORMAT_SIGNED_INT16,
               HIP_AD_FORMAT_SIGNED_INT32, HIP_AD_FORMAT_HALF, HIP_AD_FORMAT_FLOAT);

  auto num_channels = GENERATE(1, 2, 4);

  SECTION("Only devPtr initialized") {
    res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Only format initialized") {
    res_desc.res.linear.format = formats;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Only num channels initialized") {
    res_desc.res.linear.numChannels = num_channels;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Only size in bytes initialized") {
    res_desc.res.linear.sizeInBytes = array_size;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Missing devPtr") {
    res_desc.res.linear.format = formats;
    res_desc.res.linear.numChannels = num_channels;
    res_desc.res.linear.sizeInBytes = array_size;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Missing format") {
    res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    res_desc.res.linear.numChannels = num_channels;
    res_desc.res.linear.sizeInBytes = array_size;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Missing num channels") {
    res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    res_desc.res.linear.format = formats;
    res_desc.res.linear.sizeInBytes = array_size;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeLinear_EdgeCases") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  size_t array_size = N * sizeof(float);

  HIP_CHECK(hipMalloc(&tex_buffer, array_size));

  res_desc.resType = HIP_RESOURCE_TYPE_LINEAR;

  res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);

  res_desc.res.linear.format =
      GENERATE(HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16,
               HIP_AD_FORMAT_UNSIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT8, HIP_AD_FORMAT_SIGNED_INT16,
               HIP_AD_FORMAT_SIGNED_INT32, HIP_AD_FORMAT_HALF, HIP_AD_FORMAT_FLOAT);

  res_desc.res.linear.numChannels = GENERATE(1, 2, 4);
  res_desc.res.linear.sizeInBytes = array_size;

  SECTION("Invalid number of channels") {
    res_desc.res.linear.numChannels = 8;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  SECTION("Providing fake device pointer") {
    char handle;
    res_desc.res.linear.devPtr = reinterpret_cast<hipDeviceptr_t>(&handle);
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidChannelDescriptor);
  }

  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeArray") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  hipArray_t array;
  hipChannelFormatDesc channel_desc;
  int width = 32;

  res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;

  auto channel_types = hipChannelFormatKindUnsigned;

  channel_desc = hipCreateChannelDesc(width, 0, 0, 0, channel_types);

  HIP_CHECK(hipMallocArray(&array, &channel_desc, width));

  res_desc.res.array.hArray = array;

  HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  HIP_CHECK(hipTexObjectDestroy(tex_object));
  HIP_CHECK(hipFreeArray(array));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeArray_NullptrArray") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};

  res_desc.resType = HIP_RESOURCE_TYPE_ARRAY;
  res_desc.res.array.hArray = nullptr;

  HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                  hipErrorInvalidValue);

  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypeMipmapped") {
#if __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  // __linux__
  CHECK_IMAGE_SUPPORT

  hipMipmappedArray_t mipmapped_array;
  hipTextureObject_t tex_object{};
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  hipExtent extent;
  hipChannelFormatDesc channel_desc;
  constexpr size_t s = 16;

  res_desc.resType = HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;

  channel_desc = hipCreateChannelDesc<float>();
  extent = make_hipExtent(s, s, s);

  HIP_CHECK(hipMallocMipmappedArray(&mipmapped_array, &channel_desc, extent, 1, hipArrayDefault));

  res_desc.res.mipmap.hMipmappedArray = mipmapped_array;
  tex_desc.flags = HIP_TRSF_NORMALIZED_COORDINATES;

  HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));

  HIP_CHECK(hipTexObjectDestroy(tex_object));
  HIP_CHECK(hipFreeMipmappedArray(mipmapped_array));
}

TEST_CASE("Unit_TexObjectCreate_TypeMipmaped_IncompleteInit") {
#if __linux__
  HipTest::HIP_SKIP_TEST("Mipmap APIs are not supported on Linux");
  return;
#endif  // __linux__
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipMipmappedArray_t mipmapped_array;
  hipTextureObject_t tex_object{};
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  hipExtent extent;
  hipChannelFormatDesc channel_desc;
  unsigned int width = 256, height = 256, mipmap_level = 2;

  res_desc.resType = HIP_RESOURCE_TYPE_MIPMAPPED_ARRAY;

  channel_desc = hipCreateChannelDesc<float>();
  extent = make_hipExtent(width, height, 0);

  HIP_CHECK(hipMallocMipmappedArray(&mipmapped_array, &channel_desc, extent, 2 * mipmap_level,
                                    hipArrayDefault));

  res_desc.res.mipmap.hMipmappedArray = mipmapped_array;
  tex_desc.flags = HIP_TRSF_NORMALIZED_COORDINATES;

  SECTION("Array is nullptr") {
    res_desc.res.mipmap.hMipmappedArray = nullptr;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFreeMipmappedArray(mipmapped_array));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypePitch2D") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  int width = 16;
  int height = 16;

  HIP_CHECK(hipMalloc(&tex_buffer, width * height * sizeof(float)));

  res_desc.resType = HIP_RESOURCE_TYPE_PITCH2D;

  res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);

  res_desc.res.pitch2D.format = HIP_AD_FORMAT_UNSIGNED_INT32;

  res_desc.res.pitch2D.numChannels = GENERATE(1, 2, 4);
  res_desc.res.pitch2D.pitchInBytes = width * height * sizeof(float);
  res_desc.res.pitch2D.width = width;
  res_desc.res.pitch2D.height = height;

  SECTION("regular setup") {
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  }

  SECTION("width set to 0") {
    res_desc.res.pitch2D.width = 0;
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  }

  SECTION("height set to 0") {
    res_desc.res.pitch2D.height = 0;
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
  }

  HIP_CHECK(hipTexObjectDestroy(tex_object));
  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypePitch2D_IncompleteInit") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  int width = 16;
  int height = 16;

  HIP_CHECK(hipMalloc(&tex_buffer, width * height * sizeof(float)));

  res_desc.resType = HIP_RESOURCE_TYPE_PITCH2D;

  auto formats =
      GENERATE(HIP_AD_FORMAT_UNSIGNED_INT8, HIP_AD_FORMAT_UNSIGNED_INT16,
               HIP_AD_FORMAT_UNSIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT8, HIP_AD_FORMAT_SIGNED_INT16,
               HIP_AD_FORMAT_SIGNED_INT32, HIP_AD_FORMAT_HALF, HIP_AD_FORMAT_FLOAT);

  auto num_channels = GENERATE(1, 2, 4);

  SECTION("Missing devPtr") {
    res_desc.res.pitch2D.format = formats;
    res_desc.res.pitch2D.numChannels = num_channels;
    res_desc.res.pitch2D.pitchInBytes = N;
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Missing formats") {
    res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    res_desc.res.pitch2D.numChannels = num_channels;
    res_desc.res.pitch2D.pitchInBytes = N;
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Missing num_channels") {
    res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    res_desc.res.pitch2D.format = formats;
    res_desc.res.pitch2D.pitchInBytes = N;
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Missing pitch_in_bytes") {
    res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);
    res_desc.res.pitch2D.format = formats;
    res_desc.res.pitch2D.numChannels = num_channels;
    res_desc.res.pitch2D.width = width;
    res_desc.res.pitch2D.height = height;
    HIP_CHECK(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));
    HIP_CHECK(hipTexObjectDestroy(tex_object));
  }

  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}

TEST_CASE("Unit_TexObjectCreate_TypePitch2D_EdgeCases") {
  CHECK_IMAGE_SUPPORT
  CTX_CREATE();

  hipTextureObject_t tex_object{};
  float* tex_buffer;
  HIP_RESOURCE_DESC res_desc{};
  HIP_TEXTURE_DESC tex_desc{};
  int width = 16;
  int height = 16;

  HIP_CHECK(hipMalloc(&tex_buffer, width * height * sizeof(float)));

  res_desc.resType = HIP_RESOURCE_TYPE_PITCH2D;

  res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(tex_buffer);

  res_desc.res.pitch2D.format = HIP_AD_FORMAT_UNSIGNED_INT32;

  res_desc.res.pitch2D.numChannels = GENERATE(1, 2, 4);
  res_desc.res.pitch2D.pitchInBytes = width * height * sizeof(float);
  res_desc.res.pitch2D.width = width;
  res_desc.res.pitch2D.height = height;

  SECTION("Invalid device pointer") {
    char handle;
    res_desc.res.pitch2D.devPtr = reinterpret_cast<hipDeviceptr_t>(&handle);
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Pitch size larger that the actual array") {
    res_desc.res.pitch2D.pitchInBytes += 1;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Pitch size lower that the actual array") {
    res_desc.res.pitch2D.pitchInBytes -= 1;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Invalid number of channels") {
    res_desc.res.pitch2D.numChannels = 8;
    HIP_CHECK_ERROR(hipTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(tex_buffer));
  CTX_DESTROY();
}
