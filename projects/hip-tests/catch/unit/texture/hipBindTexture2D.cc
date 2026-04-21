/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

#define SIZE_H 8
#define SIZE_W 12

texture<float, 2, hipReadModeElementType> tex;

// texture object is a kernel argument
static __global__ void texture2dCopyKernel(float* dst) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if ((x < SIZE_W) && (y < SIZE_H)) {
    dst[SIZE_W * y + x] = tex2D(tex, x, y);
  }
#endif
}

HIP_TEST_CASE(Unit_hipBindTexture2D_Positive) {
  CHECK_IMAGE_SUPPORT
  float* device_ptr;
  size_t device_pitch, texture_offset;
  HIP_CHECK(hipMallocPitch(&device_ptr, &device_pitch, SIZE_W, SIZE_H));
  HIP_CHECK(hipBindTexture2D(&texture_offset, &tex, device_ptr, &tex.channelDesc, SIZE_W, SIZE_H,
                             device_pitch));
  HIP_CHECK(hipUnbindTexture(tex));
  HIP_CHECK(hipFree((void*)device_ptr));
}

HIP_TEST_CASE(Unit_hipBindTexture2D_Pitch) {
  CHECK_IMAGE_SUPPORT
  (void) hipGetLastError();  // Prevent negative tests affecting this

  float* b;
  float* a;
  float* dev_ptr_b;
  float* dev_ptr_a;

  b = new float[SIZE_H * SIZE_W];
  a = new float[SIZE_H * SIZE_W];
  for (size_t i = 1; i <= (SIZE_H * SIZE_W); i++) {
    a[i - 1] = i;
  }

  size_t dev_pitch_a, tex_ofs;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&dev_ptr_a), &dev_pitch_a,
                           SIZE_W * sizeof(float), SIZE_H));
  HIP_CHECK(hipMemcpy2D(dev_ptr_a, dev_pitch_a, a, SIZE_W * sizeof(float), SIZE_W * sizeof(float),
                        SIZE_H, hipMemcpyHostToDevice));

  tex.normalized = false;
  HIP_CHECK(
      hipBindTexture2D(&tex_ofs, &tex, dev_ptr_a, &tex.channelDesc, SIZE_W, SIZE_H, dev_pitch_a));
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_ptr_b), SIZE_W * sizeof(float) * SIZE_H));

  hipLaunchKernelGGL(texture2dCopyKernel, dim3(4, 4, 1), dim3(32, 32, 1), 0, 0, dev_ptr_b);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy2D(b, SIZE_W * sizeof(float), dev_ptr_b, SIZE_W * sizeof(float),
                        SIZE_W * sizeof(float), SIZE_H, hipMemcpyDeviceToHost));
  HipTest::checkArray(a, b, SIZE_H, SIZE_W);
  delete[] a;
  delete[] b;
  HIP_CHECK(hipFree(dev_ptr_a));
  HIP_CHECK(hipFree(dev_ptr_b));
  HIP_CHECK(hipUnbindTexture(tex));
}

HIP_TEST_CASE(Unit_hipBindTexture2D_Negative) {
  CHECK_IMAGE_SUPPORT
  float* device_ptr;
  size_t device_pitch, texture_offset;
  HIP_CHECK(hipMallocPitch(&device_ptr, &device_pitch, SIZE_W, SIZE_H / 2));

  SECTION("Texture is nullptr") {
#if HT_AMD
    HIP_CHECK_ERROR(hipBindTexture2D(&texture_offset, nullptr, device_ptr, &tex.channelDesc, SIZE_W,
                                     SIZE_H, device_pitch),
                    hipErrorUnknown);
#else
    HIP_CHECK_ERROR(hipBindTexture2D(&texture_offset, nullptr, device_ptr, &tex.channelDesc, SIZE_W,
                                     SIZE_H, device_pitch),
                    hipErrorInvalidTexture);
#endif
  }

  SECTION("Device ptr is nullptr") {
    HIP_CHECK_ERROR(hipBindTexture2D(&texture_offset, &tex, nullptr, &tex.channelDesc, SIZE_W,
                                     SIZE_H, device_pitch),
                    hipErrorNotFound);
  }

  SECTION("Pitch is 0") {
    HIP_CHECK_ERROR(
        hipBindTexture2D(&texture_offset, &tex, device_ptr, &tex.channelDesc, SIZE_W, SIZE_H, 0),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(device_ptr));
}

#endif  // __HIP_PLATFORM_AMD__ || CUDA_VERSION < CUDA_12000
