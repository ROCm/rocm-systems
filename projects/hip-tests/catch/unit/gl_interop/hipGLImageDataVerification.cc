/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Image data verification tests for GL-HIP interop.
 *
 * These tests verify that pixel data written on one side of the interop
 * boundary (GL or HIP) is faithfully readable on the other side.  The
 * pattern for every test is:
 *
 *   1. Populate a GL texture on the CPU side via glTexImage2D.
 *   2. Register + map the texture for HIP access.
 *   3. Read or write the hipArray backing the texture (via surface object
 *      or hipMemcpy2D*Array).
 *   4. Unmap + unregister, then read back via glGetTexImage.
 *   5. Compare host-side expected values against the readback.
 *
 * All pixel formats use GL_RGBA8UI (4 × uint8) to keep the arithmetic
 * simple and to match the GLImageObject fixture defined in
 * gl_interop_common.hh.
 *
 * Mipmap tests use a base level of 64×64 with two additional mip levels
 * (32×32 and 16×16) so that mipLevel indices 0, 1, and 2 are all valid.
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <hip/hip_gl_interop.h>

#include "gl_interop_common.hh"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

/* Read every texel of a surface and copy it to a flat device buffer. */
__global__ void kernel_read_surface(hipSurfaceObject_t surf, uint32_t* out,
                                    unsigned int width, unsigned int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  uchar4 px;
  surf2Dread(&px, surf, x * sizeof(uchar4), y);

  out[y * width + x] = (static_cast<uint32_t>(px.w) << 24) |
                       (static_cast<uint32_t>(px.z) << 16) |
                       (static_cast<uint32_t>(px.y) << 8) |
                        static_cast<uint32_t>(px.x);
#endif
}

/* Write a solid color to every texel. */
__global__ void kernel_fill_surface(hipSurfaceObject_t surf, uint8_t r,
                                    uint8_t g, uint8_t b, uint8_t a,
                                    unsigned int width, unsigned int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  surf2Dwrite(make_uchar4(r, g, b, a), surf, x * sizeof(uchar4), y);
#endif
}

/* Invert all four channels (255 - value) of every texel. */
__global__ void kernel_invert_surface(hipSurfaceObject_t surf,
                                      unsigned int width, unsigned int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  uchar4 px;
  surf2Dread(&px, surf, x * sizeof(uchar4), y);
  px.x = 255u - px.x;
  px.y = 255u - px.y;
  px.z = 255u - px.z;
  px.w = 255u - px.w;
  surf2Dwrite(px, surf, x * sizeof(uchar4), y);
#endif
}

/* Write R=x, G=y, B=x^y, A=0xFF unique-per-pixel to every texel. */
__global__ void kernel_write_unique_surface(hipSurfaceObject_t surf,
                                            unsigned int width,
                                            unsigned int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  surf2Dwrite(make_uchar4(static_cast<uint8_t>(x & 0xFF),
                          static_cast<uint8_t>(y & 0xFF),
                          static_cast<uint8_t>((x ^ y) & 0xFF), 0xFF),
              surf, x * sizeof(uchar4), y);
#endif
}

/* Add (x+y) mod 256 to the R, G, B channels; leave alpha unchanged. */
__global__ void kernel_transform_surface(hipSurfaceObject_t surf,
                                         unsigned int width,
                                         unsigned int height) {
#if !__HIP_NO_IMAGE_SUPPORT
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  uchar4 px;
  surf2Dread(&px, surf, x * sizeof(uchar4), y);
  uint8_t delta = static_cast<uint8_t>((x + y) & 0xFF);
  px.x = static_cast<uint8_t>(px.x + delta);
  px.y = static_cast<uint8_t>(px.y + delta);
  px.z = static_cast<uint8_t>(px.z + delta);
  surf2Dwrite(px, surf, x * sizeof(uchar4), y);
#endif
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr unsigned int kWidth    = 64;
static constexpr unsigned int kHeight   = 64;
static constexpr unsigned int kNumPixels = kWidth * kHeight;

/* Number of mip levels used by the mipmap tests (base + 2 reduced levels). */
static constexpr int kMipLevels = 3;

/* Width/height of mip level i relative to the base (kWidth, kHeight). */
static unsigned int mip_dim(unsigned int base, int level) {
  unsigned int d = base >> level;
  return d < 1u ? 1u : d;
}

/* Pack four uint8 RGBA channels into a uint32 (R in lowest byte). */
static uint32_t pack_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  return (static_cast<uint32_t>(a) << 24) | (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(g) << 8)  |  static_cast<uint32_t>(r);
}

/*
 * Create a GL_RGBA8UI texture with the given base-level pixel data.
 * min_filter is set to GL_NEAREST_MIPMAP_NEAREST when num_levels > 1 so
 * that GL accepts multiple mip levels.  Caller owns the returned name.
 */
static GLuint make_gl_texture(unsigned int width, unsigned int height,
                               const std::vector<uint32_t>& pixels,
                               int num_levels = 1) {
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  GLint min_filter = (num_levels > 1) ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, num_levels - 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI_EXT,
               static_cast<GLsizei>(width), static_cast<GLsizei>(height),
               0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, pixels.data());
  REQUIRE(glGetError() == GL_NO_ERROR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return tex;
}

/*
 * Upload pixel data for one mip level of an existing GL texture.
 * level 0 = base, level 1 = half size, etc.
 */
static void upload_mip_level(GLuint tex, int level,
                              unsigned int width, unsigned int height,
                              const std::vector<uint32_t>& pixels) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA8UI_EXT,
               static_cast<GLsizei>(width), static_cast<GLsizei>(height),
               0, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_BYTE, pixels.data());
  REQUIRE(glGetError() == GL_NO_ERROR);
  glBindTexture(GL_TEXTURE_2D, 0);
}

/* Download one mip level of a GL texture into a flat packed-RGBA vector. */
static std::vector<uint32_t> read_gl_mip_level(GLuint tex, int level,
                                                unsigned int width,
                                                unsigned int height) {
  std::vector<uint32_t> result(width * height, 0);
  glBindTexture(GL_TEXTURE_2D, tex);
  glGetTexImage(GL_TEXTURE_2D, level, GL_RGBA_INTEGER_EXT,
                GL_UNSIGNED_BYTE, result.data());
  REQUIRE(glGetError() == GL_NO_ERROR);
  glBindTexture(GL_TEXTURE_2D, 0);
  return result;
}

/* Convenience wrapper for level 0. */
static std::vector<uint32_t> read_gl_texture(GLuint tex,
                                              unsigned int width,
                                              unsigned int height) {
  return read_gl_mip_level(tex, 0, width, height);
}

/* Build a hipSurfaceObject_t from a mapped hipArray_t.
 * Requires hipGraphicsRegisterFlagsSurfaceLoadStore at registration time. */
static hipSurfaceObject_t make_surface(hipArray_t array) {
  hipResourceDesc res_desc{};
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = array;
  hipSurfaceObject_t surf = 0;
  HIP_CHECK(hipCreateSurfaceObject(&surf, &res_desc));
  return surf;
}

/* Compute a 2-D grid/block covering w × h threads with 16×16 blocks. */
static void dispatch2d(dim3& grid, dim3& block,
                        unsigned int w, unsigned int h) {
  block = dim3(16, 16, 1);
  grid  = dim3((w + 15u) / 16u, (h + 15u) / 16u, 1u);
}

// ---------------------------------------------------------------------------
// Tests: GL → HIP (read texture written by GL, verify in HIP)
// ---------------------------------------------------------------------------

/*
 * Upload a solid color to a GL texture from the CPU, map it for HIP, read
 * every pixel via a surface kernel into a device buffer, copy to host,
 * and verify all pixels match the expected color.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_GLWrite_HIPRead_SolidColor) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;
  hipGetLastError(); // reset previous err
  const uint8_t R = 0xDE, G = 0xAD, B = 0xBE, A = 0xEF;
  const uint32_t expected_pixel = pack_rgba(R, G, B, A);

  std::vector<uint32_t> host_pixels(kNumPixels, expected_pixel);
  GLuint tex = make_gl_texture(kWidth, kHeight, host_pixels);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  hipSurfaceObject_t surf = make_surface(array);

  uint32_t* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kNumPixels * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_out, 0, kNumPixels * sizeof(uint32_t)));

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_read_surface<<<grid, block>>>(surf, d_out, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hip_pixels(kNumPixels, 0);
  HIP_CHECK(hipMemcpy(hip_pixels.data(), d_out,
                      kNumPixels * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < kNumPixels; ++i) {
    REQUIRE(hip_pixels[i] == expected_pixel);
  }
}

/*
 * Upload an (x, y) coordinate-encoded pattern to a GL texture from the CPU
 * (R=x, G=y, B=x^y, A=0xFF), map it for HIP, read every pixel via a surface
 * kernel, and verify the values match the original encoding.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_GLWrite_HIPRead_Gradient) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  std::vector<uint32_t> host_pixels(kNumPixels);
  for (unsigned int y = 0; y < kHeight; ++y) {
    for (unsigned int x = 0; x < kWidth; ++x) {
      host_pixels[y * kWidth + x] =
          pack_rgba(static_cast<uint8_t>(x & 0xFF),
                    static_cast<uint8_t>(y & 0xFF),
                    static_cast<uint8_t>((x ^ y) & 0xFF), 0xFF);
    }
  }

  GLuint tex = make_gl_texture(kWidth, kHeight, host_pixels);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  hipSurfaceObject_t surf = make_surface(array);

  uint32_t* d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, kNumPixels * sizeof(uint32_t)));
  HIP_CHECK(hipMemset(d_out, 0, kNumPixels * sizeof(uint32_t)));

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_read_surface<<<grid, block>>>(surf, d_out, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint32_t> hip_pixels(kNumPixels, 0);
  HIP_CHECK(hipMemcpy(hip_pixels.data(), d_out,
                      kNumPixels * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < kNumPixels; ++i) {
    REQUIRE(hip_pixels[i] == host_pixels[i]);
  }
}

// ---------------------------------------------------------------------------
// Tests: HIP → GL (write from HIP, read back via GL)
// ---------------------------------------------------------------------------

/*
 * Start with a zeroed GL texture, map it for HIP with SurfaceLoadStore,
 * fill every pixel with a known solid color from a HIP kernel, unmap,
 * read back via glGetTexImage, and verify every pixel matches.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_HIPWrite_GLRead_SolidColor) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  const uint8_t R = 0x42, G = 0x13, B = 0x57, A = 0xFF;
  const uint32_t expected_pixel = pack_rgba(R, G, B, A);

  std::vector<uint32_t> zeroes(kNumPixels, 0);
  GLuint tex = make_gl_texture(kWidth, kHeight, zeroes);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  hipSurfaceObject_t surf = make_surface(array);

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_fill_surface<<<grid, block>>>(surf, R, G, B, A, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));

  std::vector<uint32_t> gl_pixels = read_gl_texture(tex, kWidth, kHeight);
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < kNumPixels; ++i) {
    REQUIRE(gl_pixels[i] == expected_pixel);
  }
}

/*
 * Build a per-pixel unique image on the host (R=x, G=y, B=x^y, A=0xFF),
 * upload it to the hipArray backing a GL texture via a surface write kernel,
 * unmap, read back from GL via glGetTexImage, and verify every pixel.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_HIPWrite_GLRead_UniquePerPixel) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  std::vector<uint32_t> zeroes(kNumPixels, 0);
  GLuint tex = make_gl_texture(kWidth, kHeight, zeroes);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  std::vector<uint32_t> expected(kNumPixels);
  for (unsigned int y = 0; y < kHeight; ++y) {
    for (unsigned int x = 0; x < kWidth; ++x) {
      expected[y * kWidth + x] =
          pack_rgba(static_cast<uint8_t>(x & 0xFF),
                    static_cast<uint8_t>(y & 0xFF),
                    static_cast<uint8_t>((x ^ y) & 0xFF), 0xFF);
    }
  }

  hipSurfaceObject_t surf = make_surface(array);

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_write_unique_surface<<<grid, block>>>(surf, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));

  std::vector<uint32_t> gl_pixels = read_gl_texture(tex, kWidth, kHeight);
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < kNumPixels; ++i) {
    REQUIRE(gl_pixels[i] == expected[i]);
  }
}

// ---------------------------------------------------------------------------
// Tests: round-trip (GL write → HIP modify → GL read)
// ---------------------------------------------------------------------------

/*
 * Upload a known image from the CPU to GL, map it in HIP with
 * SurfaceLoadStore, run the invert kernel (all four channels: 255 - value),
 * unmap, read back from GL, and verify every pixel equals the per-channel
 * bitwise complement of the original.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_RoundTrip_InvertChannels) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  std::vector<uint32_t> original(kNumPixels);
  for (unsigned int i = 0; i < kNumPixels; ++i) {
    uint8_t v = static_cast<uint8_t>(i & 0xFF);
    original[i] = pack_rgba(v,
                             static_cast<uint8_t>(v + 32),
                             static_cast<uint8_t>(v + 64),
                             static_cast<uint8_t>(v + 96));
  }

  GLuint tex = make_gl_texture(kWidth, kHeight, original);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  hipSurfaceObject_t surf = make_surface(array);

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_invert_surface<<<grid, block>>>(surf, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));

  std::vector<uint32_t> gl_pixels = read_gl_texture(tex, kWidth, kHeight);
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < kNumPixels; ++i) {
    REQUIRE(gl_pixels[i] == (original[i] ^ 0xFFFFFFFFu));
  }
}

/*
 * Upload a uniform base color (all RGB channels = 10, A = 0xAA) from the
 * CPU to GL, map it in HIP, apply the coordinate transform kernel
 * (RGB += (x+y) mod 256, alpha unchanged), unmap, read back from GL,
 * and verify each pixel.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_RoundTrip_CoordTransform) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  const uint8_t base = 10;
  std::vector<uint32_t> original(kNumPixels,
                                  pack_rgba(base, base, base, 0xAA));

  GLuint tex = make_gl_texture(kWidth, kHeight, original);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  hipSurfaceObject_t surf = make_surface(array);

  dim3 grid, block;
  dispatch2d(grid, block, kWidth, kHeight);
  kernel_transform_surface<<<grid, block>>>(surf, kWidth, kHeight);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));

  std::vector<uint32_t> gl_pixels = read_gl_texture(tex, kWidth, kHeight);
  glDeleteTextures(1, &tex);

  for (unsigned int y = 0; y < kHeight; ++y) {
    for (unsigned int x = 0; x < kWidth; ++x) {
      uint8_t delta        = static_cast<uint8_t>((x + y) & 0xFF);
      uint8_t expected_rgb = static_cast<uint8_t>(base + delta);
      REQUIRE(gl_pixels[y * kWidth + x] ==
              pack_rgba(expected_rgb, expected_rgb, expected_rgb, 0xAA));
    }
  }
}

// ---------------------------------------------------------------------------
// Tests: mipmap levels > 0
//
// The texture is created with kMipLevels levels (base 64×64, mip1 32×32,
// mip2 16×16).  Each mip level is uploaded separately so that
// hipGraphicsSubResourceGetMappedArray can be called with mipLevel > 0.
// ---------------------------------------------------------------------------

/*
 * Upload distinct solid colors to each mip level from the CPU, register the
 * texture, and for each mip level: map → GetMappedArray(mipLevel) → read via
 * surface kernel → unmap → verify the color matches the level-specific value.
 *
 * This confirms that mipLevel indexing selects the correct sub-resource.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_Mipmap_GLWrite_HIPRead_PerLevel) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  /* One distinct color per mip level so cross-level aliasing is detectable. */
  const uint32_t level_color[kMipLevels] = {
    pack_rgba(0xFF, 0x00, 0x00, 0xFF),  /* level 0 — red   */
    pack_rgba(0x00, 0xFF, 0x00, 0xFF),  /* level 1 — green */
    pack_rgba(0x00, 0x00, 0xFF, 0xFF),  /* level 2 — blue  */
  };

  /* Create the base level and set the mip count. */
  std::vector<uint32_t> base_pixels(kNumPixels, level_color[0]);
  GLuint tex = make_gl_texture(kWidth, kHeight, base_pixels, kMipLevels);

  /* Upload levels 1 and 2. */
  for (int lvl = 1; lvl < kMipLevels; ++lvl) {
    unsigned int w = mip_dim(kWidth, lvl);
    unsigned int h = mip_dim(kHeight, lvl);
    std::vector<uint32_t> pixels(w * h, level_color[lvl]);
    upload_mip_level(tex, lvl, w, h, pixels);
  }

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));

  for (int lvl = 0; lvl < kMipLevels; ++lvl) {
    unsigned int w = mip_dim(kWidth, lvl);
    unsigned int h = mip_dim(kHeight, lvl);

    HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

    hipArray_t array = nullptr;
    HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, lvl));

    hipSurfaceObject_t surf = make_surface(array);

    uint32_t* d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, w * h * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_out, 0, w * h * sizeof(uint32_t)));

    dim3 grid, block;
    dispatch2d(grid, block, w, h);
    kernel_read_surface<<<grid, block>>>(surf, d_out, w, h);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<uint32_t> hip_pixels(w * h, 0);
    HIP_CHECK(hipMemcpy(hip_pixels.data(), d_out,
                        w * h * sizeof(uint32_t), hipMemcpyDeviceToHost));

    HIP_CHECK(hipDestroySurfaceObject(surf));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));

    for (unsigned int i = 0; i < w * h; ++i) {
      REQUIRE(hip_pixels[i] == level_color[lvl]);
    }
  }

  HIP_CHECK(hipGraphicsUnregisterResource(resource));
  glDeleteTextures(1, &tex);
}

/*
 * For each mip level > 0 (32×32 and 16×16): start with a zeroed GL texture,
 * map the specific mip level in HIP, fill it with a level-specific color via
 * a surface kernel, unmap, read back via glGetTexImage at that mip level,
 * and verify the written color.  The base level (0) is left zeroed to confirm
 * the kernel only touches the targeted level.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_Mipmap_HIPWrite_GLRead_HigherLevels) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;
  unsigned int w = 0;
  unsigned int h = 0;

  const uint32_t fill_color[kMipLevels] = {
    pack_rgba(0x11, 0x22, 0x33, 0xFF),  /* level 0 (base, left as-is) */
    pack_rgba(0xAA, 0xBB, 0xCC, 0xFF),  /* level 1 — written by HIP   */
    pack_rgba(0xDD, 0xEE, 0xFF, 0xFF),  /* level 2 — written by HIP   */
  };

  /* Create the texture with all levels zeroed initially. */
  std::vector<uint32_t> base_pixels(kNumPixels, fill_color[0]);
  GLuint tex = make_gl_texture(kWidth, kHeight, base_pixels, kMipLevels);

  for (int lvl = 1; lvl < kMipLevels; ++lvl) {
    w = mip_dim(kWidth, lvl);
    h = mip_dim(kHeight, lvl);
    std::vector<uint32_t> zeroes(w * h, 0);
    upload_mip_level(tex, lvl, w, h, zeroes);
  }

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  /* Write each non-base mip level from HIP and verify via GL. */
  for (int lvl = 1; lvl < kMipLevels; ++lvl) {
    w = mip_dim(kWidth, lvl);
    h = mip_dim(kHeight, lvl);

    hipArray_t array = nullptr;
    HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0, lvl));

    hipSurfaceObject_t surf = make_surface(array);

    uint8_t r = static_cast<uint8_t>(fill_color[lvl] & 0xFF);
    uint8_t g = static_cast<uint8_t>((fill_color[lvl] >>  8) & 0xFF);
    uint8_t b = static_cast<uint8_t>((fill_color[lvl] >> 16) & 0xFF);
    uint8_t a = static_cast<uint8_t>((fill_color[lvl] >> 24) & 0xFF);

    dim3 grid, block;
    dispatch2d(grid, block, w, h);
    kernel_fill_surface<<<grid, block>>>(surf, r, g, b, a, w, h);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipDestroySurfaceObject(surf));
  }

  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));

  /* Verify */
  for (int lvl = 0; lvl < kMipLevels; ++lvl) {
    w = mip_dim(kWidth, lvl);
    h = mip_dim(kHeight, lvl);
    /* Read back this mip level from GL and verify. */
    std::vector<uint32_t> gl_pixels = read_gl_mip_level(tex, lvl, w, h);
    for (unsigned int i = 0, size = w * h; i < size; ++i) {
      REQUIRE(gl_pixels[i] == fill_color[lvl]);
    }
  }

  HIP_CHECK(hipGraphicsUnregisterResource(resource));
  glDeleteTextures(1, &tex);
}

/*
 * Round-trip for mip level 2 (16×16): upload a gradient to GL at level 2,
 * map it in HIP, run the invert kernel (all four channels: 255 - value),
 * unmap, read back from GL at level 2, and verify the per-channel complement.
 */
HIP_TEST_CASE(Unit_GLHIPImageData_Positive_Mipmap_RoundTrip_InvertLevel2) {
  CHECK_IMAGE_SUPPORT;
  GLContextScopeGuard gl_context;

  const int kTargetLevel = 2;
  const unsigned int w = mip_dim(kWidth, kTargetLevel);
  const unsigned int h = mip_dim(kHeight, kTargetLevel);

  /* Base level: arbitrary solid color (not touched by the test). */
  std::vector<uint32_t> base_pixels(kNumPixels,
                                     pack_rgba(0x10, 0x20, 0x30, 0xFF));
  GLuint tex = make_gl_texture(kWidth, kHeight, base_pixels, kMipLevels);

  /* Level 2: per-pixel gradient. */
  std::vector<uint32_t> mip1_original(w * h);
  for (unsigned int y = 0; y < h; ++y) {
    for (unsigned int x = 0; x < w; ++x) {
      mip1_original[y * w + x] =
          pack_rgba(static_cast<uint8_t>(x * 255u / (w - 1)),
                    static_cast<uint8_t>(y * 255u / (h - 1)),
                    static_cast<uint8_t>((x + y) & 0xFF), 0x80);
    }
  }
  upload_mip_level(tex, kTargetLevel, w, h, mip1_original);

  /* Level 1: arbitrary filler so the texture is complete. */
  unsigned int w2 = mip_dim(kWidth, 1), h2 = mip_dim(kHeight, 1);
  std::vector<uint32_t> mip2(w2 * h2, pack_rgba(0xAA, 0xBB, 0xCC, 0xFF));
  upload_mip_level(tex, 1, w2, h2, mip2);

  hipGraphicsResource* resource = nullptr;
  HIP_CHECK(hipGraphicsGLRegisterImage(&resource, tex, GL_TEXTURE_2D,
                                       hipGraphicsRegisterFlagsSurfaceLoadStore));
  HIP_CHECK(hipGraphicsMapResources(1, &resource, 0));

  hipArray_t array = nullptr;
  HIP_CHECK(hipGraphicsSubResourceGetMappedArray(&array, resource, 0,
                                                 kTargetLevel));

  hipSurfaceObject_t surf = make_surface(array);

  dim3 grid, block;
  dispatch2d(grid, block, w, h);
  kernel_invert_surface<<<grid, block>>>(surf, w, h);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipDestroySurfaceObject(surf));
  HIP_CHECK(hipGraphicsUnmapResources(1, &resource, 0));
  HIP_CHECK(hipGraphicsUnregisterResource(resource));

  std::vector<uint32_t> gl_mip1 = read_gl_mip_level(tex, kTargetLevel, w, h);
  glDeleteTextures(1, &tex);

  for (unsigned int i = 0; i < w * h; ++i) {
    REQUIRE(gl_mip1[i] == (mip1_original[i] ^ 0xFFFFFFFFu));
  }
}
