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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>

/*
 * Square each element in the array A and write to array C.
 */
template <typename T> static __global__ void vector_square_generic(T* C_d, const T* A_d, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

#ifdef GENERIC_COMPRESSED
TEST_CASE("Unit_test_generic_target_in_compressed_fatbin") {
#else
TEST_CASE("Unit_test_generic_target_in_regular_fatbin") {
#endif
  if (!isGenericTargetSupported()) {
    fprintf(stderr, "Generic target test is skipped\n");
    return;
  }
  float *A_d, *C_d;
  float *A_h, *C_h;
  size_t N = 1000000;
  size_t Nbytes = N * sizeof(float);
  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
  printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_AMD__
  printf("info: architecture on AMD GPU device is: %s\n", props.gcnArchName);
// check the scope of supportted types
#endif
  printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  A_h = (float*)malloc(Nbytes);
  HIP_CHECK(A_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  C_h = (float*)malloc(Nbytes);
  HIP_CHECK(C_h == 0 ? hipErrorOutOfMemory : hipSuccess);
  // Fill with Phi + i
  for (size_t i = 0; i < N; i++) {
    A_h[i] = 1.618f + i;
  }
  printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipMalloc(&C_d, Nbytes));

  printf("info: copy Host2Device\n");
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

  const unsigned blocks = 512;
  const unsigned threadsPerBlock = 256;

  printf("info: launch 'vector_square' kernel\n");
  hipLaunchKernelGGL(vector_square_generic, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

  printf("info: copy Device2Host\n");
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  printf("info: check result\n");
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] * A_h[i]) {
      HIP_CHECK(hipErrorUnknown);
    }
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(C_d));
  free(A_h);
  free(C_h);
  printf("PASSED!\n");
  REQUIRE(true);
}

#ifndef NO_GENERIC_TARGET_ONLY_TEST
#ifdef GENERIC_COMPRESSED
TEST_CASE("Unit_test_generic_target_only_in_compressed_fatbin") {
#ifdef __linux__
  char* cmd =
      "chmod  u+x ./hipSquareGenericTargetOnlyCompressed && ./hipSquareGenericTargetOnlyCompressed";
#else
  char* cmd = "hipSquareGenericTargetOnlyCompressed.exe";
#endif
#else  // else GENERIC_COMPRESSED
TEST_CASE("Unit_test_generic_target_only_in_regular_fatbin ") {
#ifdef __linux__
  char* cmd = "chmod  u+x ./hipSquareGenericTargetOnly && ./hipSquareGenericTargetOnly";
#else
  char* cmd = "hipSquareGenericTargetOnly.exe";
#endif
#endif  // GENERIC_COMPRESSED

  printf("Run %s\n", cmd);
  REQUIRE(std::system(cmd) == 0);
  printf("PASSED!\n");
}
#endif  // NO_GENERIC_TARGET_ONLY_TEST
