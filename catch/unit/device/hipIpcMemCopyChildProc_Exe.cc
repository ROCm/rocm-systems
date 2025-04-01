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

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <sstream>

#define HIP_CHECK(error)\
{\
  hipError_t localError = error;\
  if (localError != hipSuccess) {\
    printf("error: '%s'(%d) from %s at %s:%d\n", \
           hipGetErrorString(localError), \
            localError, #error, __FUNCTION__, __LINE__);\
    exit(0);\
  }\
}

hipIpcMemHandle_t hexToHipHandle(const std::string &hex) {
  if (hex.size() != sizeof(hipIpcMemHandle_t)*2) {
    printf("Invalid hex string length\n");
  }

  hipIpcMemHandle_t h{};
  auto bytes = reinterpret_cast<unsigned char*>(&h);
  for (size_t i = 0; i < sizeof(h); i++) {
      unsigned int byte;
      std::stringstream ss(hex.substr(i*2, 2));
      ss >> std::hex >> byte;
      bytes[i] = static_cast<unsigned char>(byte);
  }
  return h;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    return -1;
  }

  hipIpcMemHandle_t memHandle = hexToHipHandle(argv[1]);
  bool IfTestPassed = true;

  size_t N = 1024;
  size_t Nbytes = N * sizeof(int);
  int *B_d{nullptr}, *C_d{nullptr};
  int *C_h{nullptr};

  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&C_h), Nbytes, hipHostMallocDefault));
  memset(reinterpret_cast<void*>(C_h), 0, Nbytes);

  HIP_CHECK(hipMalloc(&C_d, Nbytes));
  HIP_CHECK(hipIpcOpenMemHandle(reinterpret_cast<void **>(&B_d),
                                memHandle,
                                hipIpcMemLazyEnablePeerAccess));

  HIP_CHECK(hipMemcpy(C_d, B_d, Nbytes, hipMemcpyDeviceToDevice));
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i ++) {
    if (C_h[i] != 6) {
      printf("mismatch at index: %zu with %d", i, C_h[i]);
      IfTestPassed = false;
      break;
    }
  }

  // Checking if the data obtained from IPC shared memory is consistent
  memset(reinterpret_cast<void*>(C_h), 0, Nbytes);
  HIP_CHECK(hipMemcpy(C_h, B_d, Nbytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i ++) {
    if (C_h[i] != 6) {
      printf("mismatch at index: %zu with %d", i, C_h[i]);
      IfTestPassed = false;
    }
  }

  HIP_CHECK(hipIpcCloseMemHandle(reinterpret_cast<void*>(B_d)));
  HIP_CHECK(hipFree(C_d));

  return (IfTestPassed == true);
}