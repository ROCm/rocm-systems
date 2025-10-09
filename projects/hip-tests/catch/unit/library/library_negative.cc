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

TEST_CASE("Unit_library_negative") {
  SECTION("load negative") {
    HIP_CHECK_ERROR(hipLibraryLoadData(nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
                    hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipLibraryLoadFromFile(nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipLibraryUnload(nullptr), hipErrorInvalidResourceHandle);
    HIP_CHECK_ERROR(hipLibraryGetKernel(nullptr, nullptr, nullptr), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipLibraryGetKernelCount(nullptr, nullptr), hipErrorInvalidValue);
  }

  SECTION("Load random code") {
    const char* code = "call me ishmael";  // definitely not compile-able
    hipLibrary_t lib;
    hipKernel_t kernel;
    // Default behavior is lazy load, so if we pass anything to it, it should pass
    HIP_CHECK(hipLibraryLoadData(&lib, code, nullptr, nullptr, 0, nullptr, nullptr, 0));
    // But this check will fail
    HIP_CHECK_ERROR(hipLibraryGetKernel(&kernel, lib, "moby"), hipErrorInvalidImage);
    HIP_CHECK(hipLibraryUnload(lib));
  }
}
