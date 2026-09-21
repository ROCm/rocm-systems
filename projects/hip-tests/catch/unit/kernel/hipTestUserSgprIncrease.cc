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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>

//---------------------------------------------------------------------
// Kernel that takes 32 integer arguments.
// This helps validate that all 32 kernel arguments are
// passed correctly via the user SGPRs when the compiler option is enabled.
__attribute__((amdgpu_num_sgpr(32)))
__global__ void kernel_validate_args(int* result,
                                     int arg1,  int arg2,  int arg3,  int arg4,
                                     int arg5,  int arg6,  int arg7,  int arg8,
                                     int arg9,  int arg10, int arg11, int arg12,
                                     int arg13, int arg14, int arg15, int arg16,
                                     int arg17, int arg18, int arg19, int arg20,
                                     int arg21, int arg22, int arg23, int arg24,
                                     int arg25, int arg26, int arg27, int arg28,
                                     int arg29, int arg30, int arg31, int arg32) {
    int sum = arg1  + arg2  + arg3  + arg4  + arg5  + arg6  + arg7  + arg8 +
              arg9  + arg10 + arg11 + arg12 + arg13 + arg14 + arg15 + arg16 +
              arg17 + arg18 + arg19 + arg20 + arg21 + arg22 + arg23 + arg24 +
              arg25 + arg26 + arg27 + arg28 + arg29 + arg30 + arg31 + arg32;
    *result = sum;
}

HIP_TEST_CASE(Unit_Validate_User_Sgpr_Count) {
    // ------------------------------------------------------
    // Test: Validate that a kernel with 32 arguments runs correctly.
    // ------------------------------------------------------
    const int expected = (32 * 33) / 2;
    int h_result = 0;
    int* d_result = nullptr;

    HIP_CHECK(hipMalloc(&d_result, sizeof(int)));

    hipLaunchKernelGGL(kernel_validate_args,
                       dim3(1),
                       dim3(1),
                       0,
                       0,
                       d_result,
                       1,  2,  3,  4,
                       5,  6,  7,  8,
                       9,  10, 11, 12,
                       13, 14, 15, 16,
                       17, 18, 19, 20,
                       21, 22, 23, 24,
                       25, 26, 27, 28,
                       29, 30, 31, 32);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(h_result == expected);
    std::cout << "Kernel computed sum = " << h_result
              << ", expected sum = " << expected << std::endl;
    HIP_CHECK(hipFree(d_result));
}