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

#pragma clang diagnostic ignored "-Wunused-parameter"
#include <hip_test_common.hh>

#define N 512

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex_ref;

static __global__ void kernel(float* out) {
#if !__HIP_NO_IMAGE_SUPPORT
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < N) {
    out[x] = tex1Dfetch(tex_ref, x);
  }
#endif
}

TEST_CASE("Unit_hipBindTexture_Positive") {
  CHECK_IMAGE_SUPPORT
  size_t offset = 0;
  float* tex_buf;

  hipChannelFormatDesc channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, N * sizeof(float)));

  SECTION("With Channel Descriptor") {
    HIP_CHECK(hipBindTexture(&offset, tex_ref, reinterpret_cast<void*>(tex_buf), channel_desc,
                             N * sizeof(float)));
  }

  SECTION("Without Channel Descriptor") {
    HIP_CHECK(
        hipBindTexture(&offset, tex_ref, reinterpret_cast<void*>(tex_buf), N * sizeof(float)));
  }

  HIP_CHECK(hipUnbindTexture(&tex_ref));
  HIP_CHECK(hipFree(tex_buf));
}

TEST_CASE("Unit_hipBindTexture_1DfetchVerification") {
  CHECK_IMAGE_SUPPORT

#if __HIP_NO_IMAGE_SUPPORT
  HipTest::HIP_SKIP_TEST("__HIP_NO_IMAGE_SUPPORT is set");
  return;
#endif

  float* tex_buf;
  float val[N], output[N];
  size_t offset = 0;
  float* dev_buf;
  for (int i = 0; i < N; i++) {
    val[i] = i;
    output[i] = 0.0;
  }
  hipChannelFormatDesc channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dev_buf, N * sizeof(float)));
  HIP_CHECK(hipMemcpy(tex_buf, val, N * sizeof(float), hipMemcpyHostToDevice));

  tex_ref.addressMode[0] = hipAddressModeClamp;
  tex_ref.addressMode[1] = hipAddressModeClamp;
  tex_ref.filterMode = hipFilterModePoint;
  tex_ref.normalized = 0;

  HIP_CHECK(hipBindTexture(&offset, tex_ref, reinterpret_cast<void*>(tex_buf), channel_desc,
                           N * sizeof(float)));
  HIP_CHECK(hipGetTextureAlignmentOffset(&offset, &tex_ref));

  dim3 dimBlock(64, 1, 1);
  dim3 dimGrid(N / dimBlock.x, 1, 1);

  hipLaunchKernelGGL(kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_buf);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(output, dev_buf, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    if (output[i] != val[i]) {
      INFO("Mismatch at index : " << i << ", output[i] " << output[i] << ", val[i] " << val[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipUnbindTexture(&tex_ref));
  HIP_CHECK(hipFree(tex_buf));
  HIP_CHECK(hipFree(dev_buf));
}

TEST_CASE("Unit_hipBindTexture_Negative") {
  CHECK_IMAGE_SUPPORT
  size_t offset = 0;
  float* tex_buf;

  hipChannelFormatDesc channel_desc = hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);

  HIP_CHECK(hipMalloc(&tex_buf, N * sizeof(float)));

  SECTION("Invalid texture reference") {
#if HT_AMD
    HIP_CHECK_ERROR(hipBindTexture(&offset, nullptr, reinterpret_cast<void*>(tex_buf),
                                   &channel_desc, N * sizeof(float)),
                    hipErrorInvalidSymbol);
#else
    HIP_CHECK_ERROR(hipBindTexture(&offset, nullptr, reinterpret_cast<void*>(tex_buf),
                                   &channel_desc, N * sizeof(float)),
                    hipErrorInvalidTexture);
#endif
  }

  SECTION("Device memory is nullptr") {
    HIP_CHECK_ERROR(hipBindTexture(&offset, tex_ref, nullptr, channel_desc, N * sizeof(float)),
                    hipErrorNotFound);
  }

  SECTION("Invalid hipChannelFormatDesc") {
    hipChannelFormatDesc invalid_channel_desc{-1, -1, -1, -1, hipChannelFormatKindSigned};
    HIP_CHECK_ERROR(hipBindTexture(&offset, tex_ref, reinterpret_cast<void*>(tex_buf),
                                   invalid_channel_desc, N * sizeof(float)),
                    hipErrorInvalidChannelDescriptor);
  }

  if (tex_buf) {
    HIP_CHECK(hipFree(tex_buf));
  }
}

#endif
