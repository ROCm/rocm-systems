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
#include <cstring>
#include <string>
#include <vector>
int main(int argc, char** argv) {
  if (argc < 0) {
    return -1;
  }
  int testPassed = 0;
  std::string s = argv[1];
  std::string delimiter = ",";

  size_t pos = 0;
  std::vector<std::string> token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token.push_back(s.substr(4, 16));
    s.erase(0, pos + delimiter.length());
  }
  token.push_back(s.substr(4, 16));
  int devCount = 0;
  hipError_t localError;
  localError = hipGetDeviceCount(&devCount);
  if (localError == hipSuccess) {
    printf("HIP Api returned hipSuccess");
  }
  hipDevice_t device;
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
    std::string uuid = token[i];
    if (memcmp(d_uuid.bytes, uuid.c_str(), 16) == 0) {
      testPassed += 1;
    }
  }
  if (testPassed == devCount) {
    return 1;
  } else {
    return 0;
  }
  return testPassed;
}
