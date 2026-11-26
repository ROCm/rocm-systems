/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
Round-trips data through 3D and layered arrays at the reported maximum extents.
Each side of the copy is either pinned host memory or device memory. On the PAL backend,
pinned host <-> array copies with packed pitch use SDMA; device <-> array copies use blit kernels.
*/

#include <hip_test_common.hh>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <vector>

namespace {

constexpr size_t kSmall = 64;
constexpr size_t kThin = 4;

enum class Side { PinnedHost, Device };

const char* SideName(Side side) { return side == Side::PinnedHost ? "pinned host" : "device"; }

uint32_t Pattern(size_t x, size_t y, size_t z) {
  return static_cast<uint32_t>((x * 0x9E3779B1u) ^ (y * 0x85EBCA77u) ^ (z * 0xC2B2AE3Du)) ^
      0x5BD1E995u;
}

void* Alloc(Side side, size_t bytes) {
  void* ptr = nullptr;
  if (side == Side::PinnedHost) {
    HIP_CHECK(hipHostMalloc(&ptr, bytes));
  } else {
    HIP_CHECK(hipMalloc(&ptr, bytes));
  }
  return ptr;
}

void Free(Side side, void* ptr) {
  if (side == Side::PinnedHost) {
    HIP_CHECK(hipHostFree(ptr));
  } else {
    HIP_CHECK(hipFree(ptr));
  }
}

hipMemcpyKind Kind(Side side, bool toArray) {
  if (side == Side::Device) return hipMemcpyDeviceToDevice;
  return toArray ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
}

void CopyToArray(hipArray_t array, void* src, Side side, const hipExtent& extent) {
  hipMemcpy3DParms p{};
  p.srcPtr = make_hipPitchedPtr(src, extent.width * sizeof(uint32_t), extent.width, extent.height);
  p.dstArray = array;
  p.extent = extent;
  p.kind = Kind(side, true);
  HIP_CHECK(hipMemcpy3D(&p));
}

void CopyFromArray(hipArray_t array, void* dst, Side side, const hipExtent& extent) {
  hipMemcpy3DParms p{};
  p.srcArray = array;
  p.dstPtr = make_hipPitchedPtr(dst, extent.width * sizeof(uint32_t), extent.width, extent.height);
  p.extent = extent;
  p.kind = Kind(side, false);
  HIP_CHECK(hipMemcpy3D(&p));
}

void RoundTrip(const hipExtent& extent, unsigned int flags, Side writeSide, Side readSide) {
  CAPTURE(extent.width, extent.height, extent.depth, flags, SideName(writeSide),
          SideName(readSide));

  const size_t count = extent.width * extent.height * extent.depth;
  const size_t bytes = count * sizeof(uint32_t);

  size_t freeMem = 0, totalMem = 0;
  HIP_CHECK(hipMemGetInfo(&freeMem, &totalMem));
  // Array, up to two device buffers and tiling overhead.
  if (4 * bytes > freeMem) {
    HIP_SKIP_TEST("Not enough device memory for max-extent round trip");
  }

  hipArray_t array = nullptr;
  const hipChannelFormatDesc desc = hipCreateChannelDesc<uint32_t>();
  HIP_CHECK(hipMalloc3DArray(&array, &desc, extent, flags));

  std::vector<uint32_t> expected(count);
  for (size_t z = 0; z < extent.depth; ++z) {
    for (size_t y = 0; y < extent.height; ++y) {
      for (size_t x = 0; x < extent.width; ++x) {
        expected[(z * extent.height + y) * extent.width + x] = Pattern(x, y, z);
      }
    }
  }

  void* src = Alloc(writeSide, bytes);
  if (writeSide == Side::PinnedHost) {
    std::memcpy(src, expected.data(), bytes);
  } else {
    HIP_CHECK(hipMemcpy(src, expected.data(), bytes, hipMemcpyHostToDevice));
  }
  CopyToArray(array, src, writeSide, extent);

  void* dst = Alloc(readSide, bytes);
  if (readSide == Side::PinnedHost) {
    std::memset(dst, 0, bytes);
  } else {
    HIP_CHECK(hipMemset(dst, 0, bytes));
  }
  CopyFromArray(array, dst, readSide, extent);

  std::vector<uint32_t> actual(count);
  if (readSide == Side::PinnedHost) {
    std::memcpy(actual.data(), dst, bytes);
  } else {
    HIP_CHECK(hipMemcpy(actual.data(), dst, bytes, hipMemcpyDeviceToHost));
  }

  size_t mismatches = 0;
  size_t first = 0;
  for (size_t i = 0; i < count; ++i) {
    if (actual[i] != expected[i]) {
      if (mismatches == 0) first = i;
      ++mismatches;
    }
  }

  Free(writeSide, src);
  Free(readSide, dst);
  HIP_CHECK(hipFreeArray(array));

  std::ostringstream where;
  if (mismatches != 0) {
    const size_t plane = extent.width * extent.height;
    where << "(" << first % extent.width << ", " << (first % plane) / extent.width << ", "
          << first / plane << "): expected " << expected[first] << ", got " << actual[first];
  }
  INFO("first mismatch: " << where.str());
  REQUIRE(mismatches == 0);
}

}  // namespace

/*
Writes and reads back a 3D array at its maximum width, height and depth, through every
combination of pinned host and device memory.
*/
HIP_TEST_CASE(Unit_hipMalloc3DArray_MaxExtent_RoundTrip) {
  CHECK_IMAGE_SUPPORT
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  const hipExtent extent = GENERATE_COPY(
      make_hipExtent(prop.maxTexture3D[0], kThin, kThin),
      make_hipExtent(kThin, prop.maxTexture3D[1], kThin),
      make_hipExtent(kSmall, kSmall, prop.maxTexture3D[2]));
  const Side writeSide = GENERATE(Side::PinnedHost, Side::Device);
  const Side readSide = GENERATE(Side::PinnedHost, Side::Device);

  RoundTrip(extent, hipArrayDefault, writeSide, readSide);
}

/*
Writes and reads back a layered 2D array at its maximum width, height and layer count, through
every combination of pinned host and device memory.
*/
HIP_TEST_CASE(Unit_hipMalloc3DArray_Layered_MaxExtent_RoundTrip) {
  CHECK_IMAGE_SUPPORT
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  const hipExtent extent = GENERATE_COPY(
      make_hipExtent(prop.maxTexture2DLayered[0], kThin, kThin),
      make_hipExtent(kThin, prop.maxTexture2DLayered[1], kThin),
      make_hipExtent(kSmall, kSmall, prop.maxTexture2DLayered[2]));
  const Side writeSide = GENERATE(Side::PinnedHost, Side::Device);
  const Side readSide = GENERATE(Side::PinnedHost, Side::Device);

  RoundTrip(extent, hipArrayLayered, writeSide, readSide);
}
