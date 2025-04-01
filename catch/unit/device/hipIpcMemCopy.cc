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
#include <hip_test_common.hh>
#include <hip_test_process.hh>

#include <string>
#include <cstdint>

#include <sstream>
#include <iomanip>
#include <iostream>

std::string hipHandleToHex(const hipIpcMemHandle_t &h) {
    std::ostringstream oss;
    auto bytes = reinterpret_cast<const unsigned char*>(&h);
    for (size_t i = 0; i < sizeof(h); ++i) {
        oss << std::hex << std::setw(2) << std::setfill('0') << int(bytes[i]);
    }
    return oss.str();
}

/**
 * Test Description
 * ------------------------
 *  - Verifies IPC copy with hipIpcGetMemHandle() and hipIpcOpenMemHandle()
 *    by copying data between two processes and verifying result.
 *  - Spawns child process and waits for it to finish.
 *  - Child process reads the data and check copy result.
 * Test source
 * ------------------------
 *  - unit/multiproc/hipIpcMemCopyTest.cc
 *  - unit/multiproc/hipIpcMemCopyTest_child.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipIpcMemCopyTest_validation") {
  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int *A_h{nullptr};
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), Nbytes, hipHostMallocDefault));
  int *A_d{nullptr};
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes));

  for (int i = 0; i < N; i ++){
    A_h[i] = 6;
  }

  hipIpcMemHandle_t memHandle;

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  HIP_CHECK(hipIpcGetMemHandle(&memHandle,
                               A_d));
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

  std::string hex = hipHandleToHex(memHandle);

  hip::SpawnProc proc("hipIpcMemCopyChildProc_Exe", false);
  REQUIRE(proc.run(hex) == 1);
}
