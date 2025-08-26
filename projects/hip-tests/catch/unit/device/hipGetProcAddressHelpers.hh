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

#ifndef CATCH_UNIT_DEVICE_HIPGETPROCADDRESSHELPERS_HH_
#define CATCH_UNIT_DEVICE_HIPGETPROCADDRESSHELPERS_HH_

#include <hip_test_common.hh>

/**
 * Local Function to fill the array with given value
 */
void fillHostArray(int *arr, int size, int value) {
  for ( int i = 0; i < size; i++ ) {
    arr[i] = value;
  }
}

/**
 * Local Function to validate the array with given reference value
 */
bool validateHostArray(int *arr, int size, int refValue) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != refValue ) {
      return false;
    }
  }
  return true;
}

/**
 * Local Function to fill the character array with given value
 */
void fillCharHostArray(char *arr, int size, int value) {
  for ( int i = 0; i < size; i++ ) {
    arr[i] = value;
  }
}

/**
 * Local Function to validate the array with given reference value
 */
bool validateCharHostArray(char *arr, int size, int refValue) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != refValue ) {
      return false;
    }
  }
  return true;
}

/**
 * Kernel to validate the array with given reference value
 */
__global__ void verifyArray(int *arr, int size, int refValue, int* status) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != refValue ) {
      *status = 0;
      return;
    }
  }
  *status = 1;
  return;
}

/**
 * Local Function to validate the device array with given reference value
 */
bool validateDeviceArray(int *arr, int size, int refValue) {
  int *devStatus = nullptr;
  HIP_CHECK(hipMalloc(&devStatus, sizeof(int)));
  REQUIRE(devStatus != nullptr);

  verifyArray<<<1, 1>>>(arr, size, refValue, devStatus);
  int status;
  HIP_CHECK(hipMemcpy(&status, devStatus, sizeof(int), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(devStatus));

  if ( status == 1 ) {
    return true;
  } else {
    return false;
  }
}

/**
 * Kernel to validate the character array with given reference value
 */
__global__ void verifyCharArray(char *arr, int size,
                                int refValue, int* status) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != refValue ) {
      *status = 0;
      return;
    }
  }
  *status = 1;
  return;
}

/**
 * Local Function to validate the character device array with
 * given reference value
 */
bool validateCharDeviceArray(char *arr, int size, int refValue) {
  int *devStatus = nullptr;
  HIP_CHECK(hipMalloc(&devStatus, sizeof(int)));
  REQUIRE(devStatus != nullptr);

  verifyCharArray<<< 1, 1 >>>(arr, size, refValue, devStatus);
  int status;
  HIP_CHECK(hipMemcpy(&status, devStatus, sizeof(int), hipMemcpyDeviceToHost));

  HIP_CHECK(hipFree(devStatus));

  if ( status == 1 ) {
    return true;
  } else {
    return false;
  }
}

/**
 * Kernel to fill the array with given value
 */
__global__ void fillArray(int *arr, int size, int value) {
  for ( int i = 0; i < size; i++ ) {
    arr[i] = value;
  }
}

/**
 * Local Function to fill the device array with given value
 */
void fillDeviceArray(int *arr, int size, int value) {
  fillArray<<<1, 1>>>(arr, size, value);
}

/**
 * Local Function to validate the array of different types
 */
template<class T>
bool validateArrayT(T *arr, int size, T value) {
  for ( int i = 0; i < size; i++ ) {
    if ( arr[i] != value ) {
      return false;
    }
  }
  return true;
}

/**
 * Kernel to increment 1 value to all the elements in array
 */
__global__ void addOneKernel(int *a, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i+= stride) {
    a[i] += 1;
  }
}

/**
 * Kernel to increment 2 value to all the elements in array
 */
__global__ void addTwoKernel(int *a, int size) {
  int offset = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = offset; i < size; i+= stride) {
    a[i] += 2;
  }
}

/**
 * A Simple Kernel useful for validating some kernel related scenarios
 */
__global__ void simpleKernel() {
}

/**
 * In addTenKernel, integer a value increased by 10
 */
__global__ void addTenKernel(int *a) {
  *a = *a + 10;
}

/**
 * Host function adds 10 to the integer a
 */
void addTen(void* data) {
  int *a = reinterpret_cast<int *>(data);
  *a = *a + 10;
}

/**
 * Host function adds 20 to the integer a
 */
void addTwenty(void* data) {
  int *a = reinterpret_cast<int *>(data);
  *a = *a + 20;
}

/**
 * Local function used in call back scenarios
 */
void callBackFunction(hipStream_t stream, hipError_t status, void* userData) {
  REQUIRE(stream != nullptr);
  REQUIRE(status == hipSuccess);
  int *a = reinterpret_cast<int *>(userData);
  *a += 100;
}

#endif  // CATCH_UNIT_DEVICE_HIPGETPROCADDRESSHELPERS_HH_
