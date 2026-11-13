/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipArrayGetPlane
 * @{
 * @ingroup MemoryTest
 * hipError_t hipArrayGetPlane(hipArray_t* pPlaneArray, hipArray_t hArray, unsigned int planeIdx) -
 * Gets a single format plane of a multiplanar HIP array.
 */

namespace {

constexpr size_t kWidth = 256;
constexpr size_t kHeight = 256;

HIP_ARRAY3D_DESCRIPTOR Nv12Desc(size_t width = kWidth, size_t height = kHeight) {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = width;
  desc.Height = height;
  desc.Depth = 0;
  desc.Format = HIP_AD_FORMAT_NV12;
  desc.NumChannels = 3;
  desc.Flags = hipArrayVideoEncodeDecode | hipArraySurfaceLoadStore;
  return desc;
}

// NV12 is unsupported on the PAL backend, so every test here is skipped rather than failed there.
bool CreateNv12OrSkip(hipArray_t* array, const HIP_ARRAY3D_DESCRIPTOR& desc) {
  const hipError_t status = hipArray3DCreate(array, &desc);
  if (status == hipErrorNotSupported) {
    WARN("NV12 arrays are not supported on this backend; skipping");
    return false;
  }
  HIP_CHECK(status);
  return true;
}

// Both planes carry W bytes per row: Y is W px x 1 channel, UV is W/2 px x 2 channels.
constexpr size_t kRowBytes = kWidth;

void FillPlane(hipArray_t plane, size_t rowBytes, size_t rows, unsigned char value) {
  std::vector<unsigned char> host(rowBytes * rows, value);
  HIP_CHECK(hipMemcpy2DToArray(plane, 0, 0, host.data(), rowBytes, rowBytes, rows,
                               hipMemcpyHostToDevice));
}

std::vector<unsigned char> ReadPlane(hipArray_t plane, size_t rowBytes, size_t rows) {
  std::vector<unsigned char> host(rowBytes * rows, 0);
  HIP_CHECK(hipMemcpy2DFromArray(host.data(), rowBytes, plane, 0, 0, rowBytes, rows,
                                 hipMemcpyDeviceToHost));
  return host;
}

bool AllEqual(const std::vector<unsigned char>& v, unsigned char value) {
  for (const unsigned char b : v) {
    if (b != value) return false;
  }
  return true;
}

__global__ void ReadTextureKernel(hipTextureObject_t tex, unsigned char* out, int width,
                                  int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x < width) && (y < height)) {
    out[(y * width) + x] = tex2D<unsigned char>(tex, x, y);
  }
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Creates an NV12 array and verifies both plane descriptors against the CUDA contract: plane 0
 *    is one-channel U8 at full size, plane 1 is two-channel U8 at half size, and both report the
 *    parent's flags.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Positive_PlaneGeometry") {
  hipArray_t parent = nullptr;
  const HIP_ARRAY3D_DESCRIPTOR desc = Nv12Desc();
  if (!CreateNv12OrSkip(&parent, desc)) return;

  hipArray_t yPlane = nullptr;
  hipArray_t uvPlane = nullptr;
  HIP_CHECK(hipArrayGetPlane(&yPlane, parent, 0));
  HIP_CHECK(hipArrayGetPlane(&uvPlane, parent, 1));
  REQUIRE(yPlane != nullptr);
  REQUIRE(uvPlane != nullptr);

  HIP_ARRAY3D_DESCRIPTOR got{};
  HIP_CHECK(hipArray3DGetDescriptor(&got, yPlane));
  REQUIRE(got.Width == kWidth);
  REQUIRE(got.Height == kHeight);
  REQUIRE(got.Format == HIP_AD_FORMAT_UNSIGNED_INT8);
  REQUIRE(got.NumChannels == 1);
  REQUIRE(got.Flags == desc.Flags);

  HIP_CHECK(hipArray3DGetDescriptor(&got, uvPlane));
  REQUIRE(got.Width == kWidth / 2);
  REQUIRE(got.Height == kHeight / 2);
  REQUIRE(got.Format == HIP_AD_FORMAT_UNSIGNED_INT8);
  REQUIRE(got.NumChannels == 2);
  REQUIRE(got.Flags == desc.Flags);

  // The parent keeps reporting NV12.
  HIP_CHECK(hipArray3DGetDescriptor(&got, parent));
  REQUIRE(got.Format == HIP_AD_FORMAT_NV12);
  REQUIRE(got.NumChannels == 3);

  HIP_CHECK(hipArrayDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - Writes distinct patterns to the luma and chroma planes and reads each back, in both orders.
 *    This is the test that distinguishes a correct chroma offset from a view that silently aliases
 *    the luma plane: if the UV view started at offset 0, writing UV would overwrite Y.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Positive_PlanesDoNotAlias") {
  hipArray_t parent = nullptr;
  if (!CreateNv12OrSkip(&parent, Nv12Desc())) return;

  hipArray_t yPlane = nullptr;
  hipArray_t uvPlane = nullptr;
  HIP_CHECK(hipArrayGetPlane(&yPlane, parent, 0));
  HIP_CHECK(hipArrayGetPlane(&uvPlane, parent, 1));

  // Write Y, then UV. A UV view aliased onto Y would corrupt the luma plane.
  FillPlane(yPlane, kRowBytes, kHeight, 0xAA);
  FillPlane(uvPlane, kRowBytes, kHeight / 2, 0x55);

  const std::vector<unsigned char> y = ReadPlane(yPlane, kRowBytes, kHeight);
  INFO("luma plane was clobbered by the chroma write: the chroma view is at the wrong offset");
  REQUIRE(AllEqual(y, 0xAA));

  const std::vector<unsigned char> uv = ReadPlane(uvPlane, kRowBytes, kHeight / 2);
  REQUIRE(AllEqual(uv, 0x55));

  // Now the other direction: rewriting Y must not disturb UV.
  FillPlane(yPlane, kRowBytes, kHeight, 0xBB);
  const std::vector<unsigned char> uvAfter = ReadPlane(uvPlane, kRowBytes, kHeight / 2);
  INFO("chroma plane was clobbered by the luma write: the planes overlap");
  REQUIRE(AllEqual(uvAfter, 0x55));

  const std::vector<unsigned char> yAfter = ReadPlane(yPlane, kRowBytes, kHeight);
  REQUIRE(AllEqual(yAfter, 0xBB));

  HIP_CHECK(hipArrayDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - Binds a texture object to the luma plane and samples every texel from a kernel. Copies only
 *    exercise pitch and offset metadata; this exercises the hardware image descriptor.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Positive_TextureSampling") {
  hipArray_t parent = nullptr;
  if (!CreateNv12OrSkip(&parent, Nv12Desc())) return;

  hipArray_t yPlane = nullptr;
  HIP_CHECK(hipArrayGetPlane(&yPlane, parent, 0));

  // A ramp, so a wrong base address or pitch shows up as shifted data rather than a uniform value.
  std::vector<unsigned char> pattern(kRowBytes * kHeight);
  for (size_t row = 0; row < kHeight; ++row) {
    for (size_t col = 0; col < kRowBytes; ++col) {
      pattern[(row * kRowBytes) + col] = static_cast<unsigned char>((row + col) & 0xFF);
    }
  }
  HIP_CHECK(hipMemcpy2DToArray(yPlane, 0, 0, pattern.data(), kRowBytes, kRowBytes, kHeight,
                               hipMemcpyHostToDevice));

  hipResourceDesc resDesc{};
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = yPlane;

  hipTextureDesc texDesc{};
  texDesc.addressMode[0] = hipAddressModeClamp;
  texDesc.addressMode[1] = hipAddressModeClamp;
  texDesc.filterMode = hipFilterModePoint;
  texDesc.readMode = hipReadModeElementType;
  texDesc.normalizedCoords = 0;

  hipTextureObject_t tex = 0;
  HIP_CHECK(hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

  unsigned char* out = nullptr;
  HIP_CHECK(hipMalloc(&out, kRowBytes * kHeight));
  const dim3 block(16, 16);
  const dim3 grid((kWidth + block.x - 1) / block.x, (kHeight + block.y - 1) / block.y);
  hipLaunchKernelGGL(ReadTextureKernel, grid, block, 0, 0, tex, out, static_cast<int>(kWidth),
                     static_cast<int>(kHeight));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned char> sampled(kRowBytes * kHeight, 0);
  HIP_CHECK(hipMemcpy(sampled.data(), out, sampled.size(), hipMemcpyDeviceToHost));
  REQUIRE(sampled == pattern);

  HIP_CHECK(hipFree(out));
  HIP_CHECK(hipDestroyTextureObject(tex));
  HIP_CHECK(hipArrayDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - Replays the lifetime sequence measured on CUDA: destroying a plane succeeds without releasing
 *    anything, the parent stays usable, the plane is re-queryable and the handle is stable.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Positive_Lifetime") {
  hipArray_t parent = nullptr;
  if (!CreateNv12OrSkip(&parent, Nv12Desc())) return;

  hipArray_t first = nullptr;
  hipArray_t second = nullptr;
  HIP_CHECK(hipArrayGetPlane(&first, parent, 1));
  HIP_CHECK(hipArrayGetPlane(&second, parent, 1));
  REQUIRE(first == second);  // CUDA caches plane children and returns the same handle

  // Destroying a plane is accepted and releases nothing; the parent still owns it.
  HIP_CHECK(hipArrayDestroy(first));

  HIP_ARRAY3D_DESCRIPTOR got{};
  HIP_CHECK(hipArray3DGetDescriptor(&got, parent));
  REQUIRE(got.Format == HIP_AD_FORMAT_NV12);

  hipArray_t requeried = nullptr;
  HIP_CHECK(hipArrayGetPlane(&requeried, parent, 1));
  HIP_CHECK(hipArray3DGetDescriptor(&got, requeried));
  REQUIRE(got.Width == kWidth / 2);
  REQUIRE(got.Height == kHeight / 2);

  HIP_CHECK(hipArrayDestroy(parent));

  // Parent destruction invalidates the plane handles.
  hipArray_t afterParent = nullptr;
  REQUIRE(hipArrayGetPlane(&afterParent, parent, 0) != hipSuccess);
}

/**
 * Test Description
 * ------------------------
 *  - Verifies that an NV12 parent is rejected, not crashed on, by every API that requires a
 *    single-plane image: copies, texture and surface creation.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_ParentRejectedByImageApis") {
  hipArray_t parent = nullptr;
  if (!CreateNv12OrSkip(&parent, Nv12Desc())) return;

  std::vector<unsigned char> host(kRowBytes * kHeight, 0);

  SECTION("hipMemcpy2DToArray") {
    REQUIRE(hipMemcpy2DToArray(parent, 0, 0, host.data(), kRowBytes, kRowBytes, kHeight,
                               hipMemcpyHostToDevice) != hipSuccess);
  }
  SECTION("hipMemcpy2DFromArray") {
    REQUIRE(hipMemcpy2DFromArray(host.data(), kRowBytes, parent, 0, 0, kRowBytes, kHeight,
                                 hipMemcpyDeviceToHost) != hipSuccess);
  }
  SECTION("hipMemcpyAtoH") {
    REQUIRE(hipMemcpyAtoH(host.data(), parent, 0, kRowBytes) != hipSuccess);
  }
  SECTION("hipMemcpyHtoA") {
    REQUIRE(hipMemcpyHtoA(parent, 0, host.data(), kRowBytes) != hipSuccess);
  }
  SECTION("hipCreateTextureObject") {
    hipResourceDesc resDesc{};
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = parent;
    hipTextureDesc texDesc{};
    texDesc.readMode = hipReadModeElementType;
    hipTextureObject_t tex = 0;
    REQUIRE(hipCreateTextureObject(&tex, &resDesc, &texDesc, nullptr) != hipSuccess);
  }
  SECTION("hipCreateSurfaceObject") {
    hipResourceDesc resDesc{};
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = parent;
    hipSurfaceObject_t surf = 0;
    REQUIRE(hipCreateSurfaceObject(&surf, &resDesc) != hipSuccess);
  }

  HIP_CHECK(hipArrayDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - Checks the argument contract, including CUDA's distinction between a null output pointer
 *    (invalid value) and a null array handle (invalid handle).
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_Parameters") {
  hipArray_t parent = nullptr;
  if (!CreateNv12OrSkip(&parent, Nv12Desc())) return;
  hipArray_t plane = nullptr;

  SECTION("null output pointer") {
    REQUIRE(hipArrayGetPlane(nullptr, parent, 0) == hipErrorInvalidValue);
  }
  SECTION("null array handle") {
    REQUIRE(hipArrayGetPlane(&plane, nullptr, 0) == hipErrorInvalidHandle);
  }
  SECTION("plane index out of range") {
    REQUIRE(hipArrayGetPlane(&plane, parent, 2) == hipErrorInvalidValue);
    REQUIRE(hipArrayGetPlane(&plane, parent, 0xFFFFFFFFu) == hipErrorInvalidValue);
  }
  SECTION("a plane is not itself multiplanar") {
    hipArray_t yPlane = nullptr;
    HIP_CHECK(hipArrayGetPlane(&yPlane, parent, 0));
    REQUIRE(hipArrayGetPlane(&plane, yPlane, 0) == hipErrorInvalidValue);
  }

  HIP_CHECK(hipArrayDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - An ordinary, non-multiplanar array has no planes.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_OrdinaryArray") {
  HIP_ARRAY3D_DESCRIPTOR desc{};
  desc.Width = kWidth;
  desc.Height = kHeight;
  desc.Depth = 0;
  desc.Format = HIP_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  desc.Flags = hipArrayDefault;

  hipArray_t array = nullptr;
  HIP_CHECK(hipArray3DCreate(&array, &desc));

  hipArray_t plane = nullptr;
  REQUIRE(hipArrayGetPlane(&plane, array, 0) == hipErrorInvalidValue);

  HIP_CHECK(hipArrayDestroy(array));
}

/**
 * Test Description
 * ------------------------
 *  - NV12 creation is strict on all four axes measured on CUDA: channel count, the video flag,
 *    even dimensions and 2D only.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetPlane.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.1
 */
TEST_CASE("Unit_hipArrayGetPlane_Negative_Nv12CreationValidation") {
  hipArray_t array = nullptr;

  // Establish that NV12 is supported here at all before asserting on rejections.
  {
    hipArray_t probe = nullptr;
    if (!CreateNv12OrSkip(&probe, Nv12Desc())) return;
    HIP_CHECK(hipArrayDestroy(probe));
  }

  SECTION("NumChannels must be 3") {
    for (const unsigned int channels : {1u, 2u, 4u}) {
      HIP_ARRAY3D_DESCRIPTOR desc = Nv12Desc();
      desc.NumChannels = channels;
      REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
    }
  }
  SECTION("video flag is required") {
    HIP_ARRAY3D_DESCRIPTOR desc = Nv12Desc();
    desc.Flags = hipArrayDefault;
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
    desc.Flags = hipArraySurfaceLoadStore;
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
  }
  SECTION("dimensions must be even and non-zero") {
    HIP_ARRAY3D_DESCRIPTOR desc = Nv12Desc(kWidth - 1, kHeight);
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
    desc = Nv12Desc(kWidth, kHeight - 1);
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
    desc = Nv12Desc(kWidth, 0);
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
  }
  SECTION("2D only") {
    HIP_ARRAY3D_DESCRIPTOR desc = Nv12Desc();
    desc.Depth = 4;
    REQUIRE(hipArray3DCreate(&array, &desc) == hipErrorInvalidValue);
  }
}

/**
* End doxygen group MemoryTest.
* @}
*/
