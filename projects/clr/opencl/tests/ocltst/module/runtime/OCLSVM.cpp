/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "OCLSVM.h"

#include <stdio.h>

#include <algorithm>
#include <cstdlib>
#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#endif
#include <iostream>

#define NUM_SIZES 6

#define OCL_CHECK(error)                                                                           \
  if (error != CL_SUCCESS) {                                                                       \
    fprintf(stderr, "OpenCL API invocation failed at %s:%d\n", __FILE__, __LINE__);                \
    exit(-1);                                                                                      \
  }

#define STR(__macro__) #__macro__

#ifdef _WIN32
size_t getTotalSystemMemory() {
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
}
#endif

template <typename T, unsigned N> static unsigned countOf(const T (&)[N]) { return N; }

const static char* sources[] = {
    STR(__kernel void test(__global int* ptr) { ptr[get_global_id(0)] = 0xDEADBEEF; }),
    STR(__kernel void test(__global int* ptr, __global int* ptr2) {
      ptr[get_global_id(0)] = 0xDEADBEEF;
      ptr2[get_global_id(0)] = 0xDEADF00D;
    }),
    STR(__kernel void test(__global long* ptr) { ptr[get_global_id(0) * 1024] = 0xBAADF00D; }),
    STR(__kernel void test(__global ulong* ptr) {
      while (ptr) {
        *ptr = 0xDEADBEEF;
        ptr = *((__global ulong * __global*)(ptr + 1));
      }
    }),
    STR(__kernel void test(__global volatile int* ptr, int numIterations) {
      for (int i = 0; i < numIterations; i++) {
        // This should be:
        // atomic_fetch_add_explicit(ptr, 1, memory_order_relaxed,
        //                           memory_scope_all_svm_devices);
        // But using device atomics is mapped to the same ISA and compiles
        // in OpenCL 1.2
        atomic_inc(ptr);
      }
    }),
    STR(__kernel void test(){
        // dummy
    }),
    STR(__kernel void test(int8 arg0, __global int* arg1, int arg2, __global int* arg3,
                           __global float* arg4){
        // dummy
    }),
    STR(__kernel void test(__global int* ptr, int to) {
      // dummy kernel that takes a long time to complete
      for (int i = 0; i < to; ++i) {
        // avoid compiler optimizations
        if (ptr[get_global_id(0)] != 17) {
          ptr[get_global_id(0)]++;
        } else {
          ptr[get_global_id(0)] += 2;
        }
      }
    }),
    STR(__kernel void test(){
        // dummy
    })};

OCLSVM::OCLSVM() { _numSubTests = countOf(sources); }

OCLSVM::~OCLSVM() {}

void OCLSVM::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  CHECK_ERROR(error_, "Error opening test");
  _openTest = test;

  if (!isOpenClSvmAvailable(devices_[_deviceId])) {
    printf("Device does not support any SVM features, skipping...\n");
    return;
  }

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, sources + _openTest, NULL, &error_);
  CHECK_ERROR(error_, "clCreateProgramWithSource()  failed");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], "-cl-std=CL2.0", NULL, NULL);
  CHECK_ERROR(error_, "clBuildProgram() failed");

  kernel_ = _wrapper->clCreateKernel(program_, "test", &error_);
  CHECK_ERROR(error_, "clCreateKernel() failed");
}

#ifndef CL_VERSION_2_0
// make sure the tests compile in OpenCL <= 1.2
void OCLSVM::runFineGrainedBuffer() {}
void OCLSVM::runFineGrainedSystem() {}
void OCLSVM::runFineGrainedSystemLargeAllocations() {}
void OCLSVM::runLinkedListSearchUsingFineGrainedSystem() {}
void OCLSVM::runPlatformAtomics() {}
void OCLSVM::runEnqueueOperations() {}
void OCLSVM::runSvmArgumentsAreRecognized() {}
void OCLSVM::runSvmCommandsExecutedInOrder() {}
void OCLSVM::runIdentifySvmBuffers() {}
#else

void OCLSVM::runFineGrainedBuffer() {
  if (!(svmCaps_ & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)) {
    printf(
        "Device does not support fined-grained buffer sharing, skipping "
        "test...\n");
    return;
  }
  const size_t numElements = 256;
  int* ptr = (int*)clSVMAlloc(context_, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                              numElements * sizeof(int), 0);
  CHECK_RESULT(!ptr, "clSVMAlloc() failed");

  error_ = clSetKernelArgSVMPointer(kernel_, 0, ptr);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  size_t gws[1] = {numElements};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");

  size_t matchingElements = std::count(ptr, ptr + numElements, (int)0xDEADBEEF);
  CHECK_RESULT(matchingElements != numElements, "Expected: %zd, found:%zd", numElements,
               matchingElements);
  clSVMFree(context_, ptr);
}

void OCLSVM::runFineGrainedSystem() {
  if (!(svmCaps_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
    printf(
        "Device does not support fined-grained system sharing, skipping "
        "test...\n");
    return;
  }

  const size_t numElements = 256;
  int* ptr = new int[numElements];
  int* ptr2 = new int[numElements];
  error_ = clSetKernelArgSVMPointer(kernel_, 0, ptr);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  error_ = clSetKernelArgSVMPointer(kernel_, 1, ptr2);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  size_t gws[1] = {numElements};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");

  size_t matchingElements = std::count(ptr, ptr + numElements, (int)0xDEADBEEF);
  size_t matchingElements2 = std::count(ptr2, ptr2 + numElements, (int)0xDEADF00D);
  CHECK_RESULT(matchingElements + matchingElements2 != 2 * numElements, "Expected: %zd, found:%zd",
               numElements * 2, matchingElements + matchingElements2);
  delete[] ptr;
  delete[] ptr2;
}

void OCLSVM::runFineGrainedSystemLargeAllocations() {
#ifdef _WIN32
  if (!(svmCaps_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
    printf(
        "Device does not support fined-grained system sharing on Lnx, skipping "
        "test...\n");
    return;
  }

  // Max allowed multiplier for malloc
  size_t allowedMemSize = getTotalSystemMemory() >> 12;

  size_t numElements = 256;

  char* s = getenv("OCLSVM_MALLOC_GB_SIZE");
  char* s2 = getenv("OCLSVM_MEMSET_ALLOC");

  for (int j = 1; j <= NUM_SIZES; j++) {
    numElements = 131072 * j;

    if (s != NULL) numElements = 131072 * atoi(s);

    if (numElements > allowedMemSize) break;

    void* ptr = malloc(numElements * 1024 * sizeof(uint64_t));
    CHECK_ERROR(ptr == NULL, "malloc failure");

    if (s2 != NULL) memset(ptr, 0, numElements * 1024 * sizeof(uint64_t));

    error_ = clSetKernelArgSVMPointer(kernel_, 0, ptr);
    CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

    size_t gws[1] = {numElements};
    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                              NULL, NULL);
    CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");

    error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
    CHECK_ERROR(error_, "Queue::finish() failed");

    uint64_t* ptr64 = reinterpret_cast<uint64_t*>(ptr);
    // Do a check
    for (int i = 0; i < numElements; i++) {
      if ((int)ptr64[i * 1024] != 0xBAADF00D) {
        uint64_t temp = ptr64[i * 1024];
        delete[] ptr;
        CHECK_RESULT(temp != 0xBAADF00D, "Found: %d, Expected:%d", temp, 0xBAADF00D);
      }
    }
    delete[] ptr;
  }
#endif
}

void OCLSVM::runLinkedListSearchUsingFineGrainedSystem() {
  if (!(svmCaps_ & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
    printf(
        "Device does not support fined-grained system sharing, skipping "
        "test...\n");
    return;
  }

  uint64_t input[] = {34, 6, 0, 11, 89, 34, 6, 6, 6, 0xDEADBEEF};
  int inputSize = countOf(input);
  Node* ptr = NULL;
  for (int i = 0; i < inputSize; i++) {
    ptr = new Node(input[i], ptr);
  }
  error_ = clSetKernelArgSVMPointer(kernel_, 0, ptr);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  size_t gws[1] = {1};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");

  int matchingElements = 0;
  // verify result while deallocating resources at the same time
  while (ptr) {
    if (ptr->value_ == 0xDEADBEEF) {
      matchingElements++;
    }
    Node* tmp = ptr;
    ptr = (Node*)ptr->next_;
    delete tmp;
  }
  CHECK_RESULT(matchingElements != inputSize, "Expected: %d, found:%d", inputSize,
               matchingElements);
}

static int atomicIncrement(volatile int* loc) {
#if defined(_MSC_VER)
  return _InterlockedIncrement((volatile long*)loc);
#elif defined(__GNUC__)
  return __sync_fetch_and_add(loc, 1);
#endif
  printf("Atomic increment not supported, aborting...");
  std::abort();
  return 0;
}

void OCLSVM::runPlatformAtomics() {
  if (!(svmCaps_ & CL_DEVICE_SVM_ATOMICS)) {
    printf("SVM atomics not supported, skipping test...\n");
    return;
  }

  volatile int* value = (volatile int*)clSVMAlloc(
      context_, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS, sizeof(int), 0);
  CHECK_RESULT(!value, "clSVMAlloc() failed");
  *value = 0;
  const int numIterations = 1000000;
  error_ = clSetKernelArgSVMPointer(kernel_, 0, (const void*)value);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  error_ = clSetKernelArg(kernel_, 1, sizeof(numIterations), &numIterations);
  CHECK_ERROR(error_, "clSetKernelArg() failed");

  size_t gws[1] = {1};
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");

  for (int i = 0; i < numIterations; i++) {
    atomicIncrement(value);
  }

  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");

  int expected = numIterations * 2;
  CHECK_RESULT(*value != expected, "Expected: %d, found:%d", expected, *value);
  clSVMFree(context_, (void*)value);
}

void OCLSVM::runEnqueueOperations() {
  size_t numElements = 32;
  size_t size = numElements * 4;
  int* ptr0 = (int*)clSVMAlloc(context_, 0, size, 0);
  CHECK_RESULT(!ptr0, "clSVMAlloc() failed");
  int* ptr1 = (int*)clSVMAlloc(context_, 0, size, 0);
  CHECK_RESULT(!ptr1, "clSVMAlloc() failed");
  cl_event userEvent = clCreateUserEvent(context_, &error_);
  CHECK_ERROR(error_, "clCreateUserEvent() failed");

  cl_command_queue queue = cmdQueues_[_deviceId];
  // coarse-grained buffer semantics: the SVM pointer needs to be mapped
  // before the pointer can write to it
  error_ = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, ptr0, size, 0, NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueSVMMap() failed");
  std::fill(ptr0, ptr0 + numElements, 1);
  error_ = clEnqueueSVMUnmap(queue, ptr0, 0, NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueSVMUnmap() failed");

  // we copy the 1st buffer into the 2nd buffer
  error_ = clEnqueueSVMMemcpy(queue, true, ptr1, ptr0, size, 0, NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueSVMMemcpy() failed");

  // verification: the 2nd buffer should be identical to the 1st
  error_ = clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, ptr1, size, 0, NULL, &userEvent);
  CHECK_ERROR(error_, "clEnqueueSVMMap() failed");

  error_ = clWaitForEvents(1, &userEvent);
  CHECK_ERROR(error_, "clWaitForEvents() failed");

  size_t observed = std::count(ptr1, ptr1 + numElements, 1);
  size_t expected = numElements;
  CHECK_RESULT(observed != expected, "Expected: %zd, found:%zd", expected, observed);

  void* ptrs[2] = {ptr0, ptr1};
  error_ = clEnqueueSVMFree(queue, countOf(ptrs), ptrs, NULL, NULL, 0, NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueSVMFree() failed");
  error_ = clFinish(queue);
  CHECK_ERROR(error_, "clFinish() failed");
}

/**
 * Simple test to ensure that SVM pointer arguments are identified properly in
 * the runtime, since kernel arguments of pointer type can be bound to either
 * SVM pointers or cl_mem objects.
 */
void OCLSVM::runSvmArgumentsAreRecognized() {
  cl_int8 arg0;
  error_ = clSetKernelArg(kernel_, 0, sizeof(arg0), &arg0);
  CHECK_ERROR(error_, "clSetKernelArg() failed");

  error_ = clSetKernelArgSVMPointer(kernel_, 1, NULL);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  cl_int arg2;
  error_ = clSetKernelArg(kernel_, 2, sizeof(arg2), &arg2);
  CHECK_ERROR(error_, "clSetKernelArg() failed");

  error_ = clSetKernelArgSVMPointer(kernel_, 3, NULL);
  CHECK_ERROR(error_, "clSetKernelArgSVMPointer() failed");

  cl_mem arg4 = NULL;
  error_ = clSetKernelArg(kernel_, 4, sizeof(arg4), &arg4);
  CHECK_ERROR(error_, "clSetKernelArg() failed");

  size_t gws[1] = {1};

  // run dummy kernel
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");
  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");

  // now we bind a pointer argument to a standard buffer instead of a SVM one
  cl_mem buffer = NULL;
  error_ = clSetKernelArg(kernel_, 1, sizeof(buffer), &buffer);
  CHECK_ERROR(error_, "clSetKernelArg() failed");

  // re-execute the dummy kernel using different actual parameters
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, gws, NULL, 0,
                                            NULL, NULL);
  CHECK_ERROR(error_, "clEnqueueNDRangeKernel() failed");
  error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "Queue::finish() failed");
}

void OCLSVM::runSvmCommandsExecutedInOrder() {
#if EMU_ENV
  // Small number is enough to verify functionality in Emu environment
  const int numElements = 5000;
#else
  const int numElements = 100000;
#endif  // EMU_ENV
  size_t size = numElements * sizeof(int);
  // allocate SVM memory
  int* data = (int*)clSVMAlloc(context_, CL_MEM_READ_WRITE, size, 0);
  CHECK_RESULT(!data, "clSVMAlloc failed");

  // map the SVM buffer to host
  cl_int status =
      clEnqueueSVMMap(cmdQueues_[_deviceId], CL_TRUE, CL_MAP_WRITE, data, size, 0, NULL, NULL);
  CHECK_ERROR(status, "Error when mapping SVM buffer");

  // fill buffer with 0s
  std::fill(data, data + numElements, 0);

  // unmap the SVM buffer to host
  status = clEnqueueSVMUnmap(cmdQueues_[_deviceId], data, 0, NULL, NULL);
  CHECK_ERROR(status, "Error when unmapping SVM buffer");

  // enqueue kernel
  status = clSetKernelArgSVMPointer(kernel_, 0, data);
  CHECK_ERROR(status, "Error when setting kernel argument");
  status = clSetKernelArg(kernel_, 1, sizeof(int), &numElements);
  CHECK_ERROR(status, "clSetKernelArg() failed");

  cl_event event;
  size_t overallSize = (size_t)numElements;
  status = clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, &overallSize, NULL, 0,
                                  NULL, &event);
  CHECK_ERROR(status, "Error when enqueuing kernel");
  error_ = clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(status, "clFinish()");

  // map the SVM buffer to host
  status = clEnqueueSVMMap(cmdQueues_[_deviceId], CL_TRUE, CL_MAP_READ, data, size, 0, NULL, NULL);
  CHECK_ERROR(status, "Error when mapping SVM buffer");

  bool pass = true;
  // verify the data. Using descending order might increase the chance of
  // finding an error since the GPU (when used) might not have finished
  // updating the data array by the time we do the verification
  for (int i = numElements - 1; i >= 0; i--) {
    if (data[i] != numElements + 1) {
      pass = false;
      break;
    }
  }

  // unmap the SVM buffer to host
  status = clEnqueueSVMUnmap(cmdQueues_[_deviceId], data, 0, NULL, NULL);
  CHECK_ERROR(status, "Error when unmapping SVM buffer");

  // free the SVM buffer
  status = clEnqueueSVMFree(cmdQueues_[_deviceId], 1, (void**)&data, NULL, NULL, 0, NULL, NULL);
  CHECK_ERROR(status, "Error when freeing the SVM buffer");
  error_ = clFinish(cmdQueues_[_deviceId]);
  CHECK_ERROR(error_, "clFinish() failed");
  CHECK_RESULT(!pass, "Wrong result");
}

void OCLSVM::runIdentifySvmBuffers() {
  size_t size = 1024 * 1024;

  // dummy allocation to force the runtime to track several SVM buffers
  clSVMAlloc(context_, CL_MEM_READ_WRITE, size * 10, 0);

  void* ptr = clSVMAlloc(context_, CL_MEM_READ_WRITE, size, 0);
  cl_int status;
  cl_bool usesSVMpointer = CL_FALSE;

  // dummy allocation to force the runtime to track several SVM buffers
  clSVMAlloc(context_, CL_MEM_READ_WRITE, size * 4, 0);

  // buffer using the entire SVM region should be identified as such
  cl_mem buf1 = clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, size, ptr, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  size_t paramSize = 0;
  status = clGetMemObjectInfo(buf1, CL_MEM_USES_SVM_POINTER, 0, 0, &paramSize);
  CHECK_ERROR(status, "clGetMemObjectInfo failed");
  CHECK_RESULT(paramSize != sizeof(cl_bool),
               "clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) "
               "returned wrong size.");

  status = clGetMemObjectInfo(buf1, CL_MEM_USES_SVM_POINTER, sizeof(cl_bool), &usesSVMpointer, 0);
  CHECK_ERROR(status, "clGetMemObjectInfo failed");
  CHECK_RESULT(usesSVMpointer != CL_TRUE,
               "clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) "
               "returned CL_FALSE for buffer created from SVM pointer.");

  // Buffer that uses random region within SVM buffers
  cl_mem buf2 =
      clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, 256, (char*)ptr + size - 256, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  status = clGetMemObjectInfo(buf2, CL_MEM_USES_SVM_POINTER, sizeof(cl_bool), &usesSVMpointer, 0);
  CHECK_ERROR(status, "clGetMemObjectInfo failed");
  CHECK_RESULT(usesSVMpointer != CL_TRUE,
               "clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) "
               "returned CL_FALSE for buffer created from SVM pointer.");

  // for any other pointer the query should return false
  void* randomPtr = malloc(size);
  cl_mem buf3 = clCreateBuffer(context_, CL_MEM_USE_HOST_PTR, size, randomPtr, &status);
  CHECK_ERROR(status, "clCreateBuffer failed.");

  status = clGetMemObjectInfo(buf3, CL_MEM_USES_SVM_POINTER, sizeof(cl_bool), &usesSVMpointer, 0);
  CHECK_ERROR(status, "clGetMemObjectInfo failed");
  CHECK_RESULT(usesSVMpointer == CL_TRUE,
               "clGetMemObjectInfo(CL_MEM_USES_SVM_POINTER) "
               "returned CL_TRUE for buffer not created from SVM pointer.");

  clReleaseMemObject(buf3);
  clReleaseMemObject(buf2);
  clReleaseMemObject(buf1);
  clSVMFree(context_, ptr);
}
#endif

cl_bool OCLSVM::isOpenClSvmAvailable(cl_device_id device_id) {
#ifdef CL_VERSION_2_0
  error_ = clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_SVM_CAPABILITIES, sizeof(svmCaps_),
                           &svmCaps_, NULL);
  CHECK_ERROR_NO_RETURN(error_, "clGetDeviceInfo() failed");
  if (!(svmCaps_ & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
    return CL_FALSE;
  } else {
    return CL_TRUE;
  }
#endif
  // -Device does not support OpenCL >= 2.0
  // -Device supports OpenCL >= 2.0, but available headers are <= 1.2
  return CL_FALSE;
}

void OCLSVM::run() {
  if (!isOpenClSvmAvailable(devices_[_deviceId])) {
    printf("Device does not support any SVM features, skipping...\n");
    return;
  }

  if (_openTest == 0) {
    runFineGrainedBuffer();
  } else if (_openTest == 1) {
    runFineGrainedSystem();
  } else if (_openTest == 2) {
    runFineGrainedSystemLargeAllocations();
  } else if (_openTest == 3) {
    runLinkedListSearchUsingFineGrainedSystem();
  } else if (_openTest == 4) {
    runPlatformAtomics();
  } else if (_openTest == 5) {
    runEnqueueOperations();
  } else if (_openTest == 6) {
    runSvmArgumentsAreRecognized();
  } else if (_openTest == 7) {
    runSvmCommandsExecutedInOrder();
  } else if (_openTest == 8) {
    runIdentifySvmBuffers();
  }
}

unsigned int OCLSVM::close(void) { return OCLTestImp::close(); }
