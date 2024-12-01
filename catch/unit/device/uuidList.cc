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
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip/hip_runtime.h>
#include<iostream>
#include<string>
int main() {
  hipDevice_t device;
  int devCount;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  std::string uuid;
  for (int i = 0; i < devCount; i++) {
    localError = hipSetDevice(i);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    localError = hipDeviceGet(&device, i);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    hipUUID d_uuid{0};
    localError = hipDeviceGetUuid(&d_uuid, device);
    if (localError == hipSuccess) {
      printf("HIP Api returned hipSuccess");
    }
    char preStr[16] = "GPU-";
    strcat(preStr, d_uuid.bytes);
    if (i == devCount - 1) {
      uuid = uuid + preStr;
    } else {
      uuid = uuid + preStr + ",";
    }
  }
  std::cout<< uuid;
  return 0;
}

