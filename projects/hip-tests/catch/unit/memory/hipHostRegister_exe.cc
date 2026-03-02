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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <stdlib.h>
#include <iostream>
#include <chrono>  // NOLINT
#include "hip/hip_runtime_api.h"

#pragma clang diagnostic ignored "-Wunused-parameter"

#define ITERATION 1000
#define SIZE (64 * 1024 * 1024)
#define ARRAY_SIZE 20

static bool UNSETENV(std::string var) {
  int result = -1;
#ifdef __unix__
  result = unsetenv(var.c_str());
#else
  result = _putenv((var + '=').c_str());
#endif
  return (result == 0) ? true : false;
}

static bool SETENV(std::string var, std::string value, int overwrite) {
  int result = -1;
#ifdef __unix__
  result = setenv(var.c_str(), value.c_str(), overwrite);
#else
  result = _putenv((var + '=' + value).c_str());
#endif
  return (result == 0) ? true : false;
}

/**
 Expects 2 command line arg, first command is flag svm_enable = 1/0
 and second command is test number: 0 = Register/Unregister different
 chunks of host memory, 1 = Register/Unregister the same chunk of host
 memory repeatedly, 2 = Register/Unregister the same chunk of host
 memory repeatedly on multiple GPUs.
*/
int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Invalid number of args passed.\n"
              << "argc : " << argc << std::endl;
    return -1;
  }
  std::string env_flag = argv[1];
  int test = std::stoi(argv[2]);
  // disable SVM feature using HSA_USE_SVM=0 env from shell
  UNSETENV("HSA_USE_SVM");
  if (env_flag == "svm_enable") {
    SETENV("HSA_USE_SVM", "1", 1);
  } else {
    SETENV("HSA_USE_SVM", "0", 1);
  }
  if (test == 0) {
    uint8_t* A[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
      A[i] = reinterpret_cast<uint8_t*>(malloc(SIZE));
      if (A[i] == nullptr) {
        return -1;
      }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int count = 0; count < ITERATION; count++) {
      // Register the host pointer
      if (hipSuccess != hipHostRegister(A[count % ARRAY_SIZE], SIZE, 0)) {
        return -1;
      }
      // Unregister the host pointer
      if (hipSuccess != hipHostUnregister(A[count % ARRAY_SIZE])) {
        return -1;
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ARRAY_SIZE; i++) {
      free(A[i]);
    }
    std::chrono::duration<float, std::milli> fp_ms = t2 - t1;
    std::cout << fp_ms.count() << std::endl;
  } else if (test == 1) {
    uint8_t* A;
    A = reinterpret_cast<uint8_t*>(malloc(SIZE));
    if (A == nullptr) {
      return -1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int count = 0; count < ITERATION; count++) {
      // Register the host pointer
      if (hipSuccess != hipHostRegister(A, SIZE, 0)) {
        return -1;
      }
      // Unregister the host pointer
      if (hipSuccess != hipHostUnregister(A)) {
        return -1;
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    free(A);
    std::chrono::duration<float, std::milli> fp_ms = t2 - t1;
    std::cout << fp_ms.count() << std::endl;
  } else if (test == 2) {
    uint8_t* A;
    A = reinterpret_cast<uint8_t*>(malloc(SIZE));
    if (A == nullptr) {
      return -1;
    }
    int dev_count = 0;
    if (hipSuccess != hipGetDeviceCount(&dev_count)) {
      return -1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int dev = 0; dev < dev_count; dev++) {
      if (hipSuccess != hipSetDevice(dev)) {
        return -1;
      }
      for (int count = 0; count < ITERATION; count++) {
        // Register the host pointer
        if (hipSuccess != hipHostRegister(A, SIZE, 0)) {
          return -1;
        }
        // Unregister the host pointer
        if (hipSuccess != hipHostUnregister(A)) {
          return -1;
        }
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    free(A);
    std::chrono::duration<float, std::milli> fp_ms = t2 - t1;
    std::cout << fp_ms.count() << std::endl;
  } else {
    // Undefined test
  }
  UNSETENV("HSA_USE_SVM");
  return 0;
}
