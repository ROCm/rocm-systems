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

#include "OCLSDI.h"

#include "Timer.h"
#define NUM_TESTS 6

#include <cmath>

typedef struct _threadInfo {
  int threadID_;
  OCLSDI* testObj_;
} ThreadInfo;
const char* kernel_str_ =
    "__kernel void test_kernel(global unsigned int * A) \
			   { \
					int id = get_global_id(0);  \
                    A[id] = id + 2;\
			   } ";
const char* testNames[NUM_TESTS] = {
    "WriteBuffer", "CopyBuffer",      "NDRangeKernel",
    "MapBuffer",   "WriteBufferRect", "CopyImageToBuffer",
};

void* ThreadMain(void* data) {
  if (data == NULL) {
    return 0;
  }
  ThreadInfo* threadData = (ThreadInfo*)data;
  threadData->testObj_->threadEntry(threadData->threadID_);
  return NULL;
}

OCLSDI::OCLSDI() {
  // If there are two different gpus in the system,
  // we have to test each of them as sender and receiver
  _numSubTests = 2 * NUM_TESTS;
}

OCLSDI::~OCLSDI() {}

void OCLSDI::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  cl_uint numPlatforms = 0;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  _crcword = 0;
  conversion = 1.0f;
  program_ = 0;
  kernel_ = 0;
  srcBuff_ = 0;
  _openTest = test % NUM_TESTS;
  bufSize_ = 0x10000;
  error_ = 0;
  markerValue_ = 0x12345;
  inputArr_ = 0;
  outputArr_ = 0;
  success_ = true;
  extPhysicalBuff_ = 0;
  silentFailure = false;
  busAddressableBuff_ = 0;
  devices_[0] = devices_[1] = 0;
  contexts_[0] = contexts_[1] = 0;
  cmd_queues_[0] = cmd_queues_[1] = 0;
  image_ = 0;

  inputArr_ = (cl_uint*)malloc(bufSize_);
  outputArr_ = (cl_uint*)malloc(bufSize_);
  for (unsigned int i = 0; i < (bufSize_ / sizeof(cl_uint)); ++i) {
    inputArr_[i] = i + 1;
    outputArr_[i] = 0;
  }
  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(numPlatforms == 0, "clGetPlatformIDs failed");
  error_ = _wrapper->clGetPlatformIDs(1, &platform, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  error_ = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (num_devices < 2) {
    printf("\nSilent Failure: Two GPUs are required to run OCLSdi test\n");
    silentFailure = true;
    return;
  }
  error_ = _wrapper->clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, devices_, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
  if (test >= NUM_TESTS) {
    cl_device_id temp = devices_[0];
    devices_[0] = devices_[1];
    devices_[1] = temp;
  }
  size_t param_size = 0;
  char* strExtensions = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_EXTENSIONS, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strExtensions = (char*)malloc(param_size);
  error_ =
      _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_EXTENSIONS, param_size, strExtensions, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strstr(strExtensions, "cl_amd_bus_addressable_memory") == 0) {
    printf(
        "\nSilent Failure: cl_amd_bus_addressable_memory extension is not "
        "enabled on GPU 0\n");
    silentFailure = true;
    free(strExtensions);
    return;
  }
  free(strExtensions);
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_EXTENSIONS, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strExtensions = (char*)malloc(param_size);
  error_ =
      _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_EXTENSIONS, param_size, strExtensions, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  if (strstr(strExtensions, "cl_amd_bus_addressable_memory") == 0) {
    printf(
        "\nSilent Failure: cl_amd_bus_addressable_memory extension is not "
        "enabled on GPU 1\n");
    silentFailure = true;
    free(strExtensions);
    return;
  }
  free(strExtensions);
  deviceNames_ = " [";
  param_size = 0;
  char* strDeviceName = 0;
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_NAME, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strDeviceName = (char*)malloc(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[1], CL_DEVICE_NAME, param_size, strDeviceName, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  deviceNames_ = deviceNames_ + strDeviceName;
  free(strDeviceName);
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_NAME, 0, 0, &param_size);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  strDeviceName = (char*)malloc(param_size);
  error_ = _wrapper->clGetDeviceInfo(devices_[0], CL_DEVICE_NAME, param_size, strDeviceName, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");
  deviceNames_ = deviceNames_ + "->";
  deviceNames_ = deviceNames_ + strDeviceName;
  free(strDeviceName);
  deviceNames_ = deviceNames_ + "]";
  cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
  contexts_[0] = _wrapper->clCreateContext(props, 1, &devices_[0], 0, 0, &error_);
  CHECK_RESULT(contexts_[0] == 0, "clCreateContext failed");
  contexts_[1] = _wrapper->clCreateContext(props, 1, &devices_[1], 0, 0, &error_);
  CHECK_RESULT(contexts_[1] == 0, "clCreateContext failed");
  cmd_queues_[0] = _wrapper->clCreateCommandQueue(contexts_[0], devices_[0], 0, NULL);
  CHECK_RESULT(cmd_queues_[0] == 0, "clCreateCommandQueue failed");
  cmd_queues_[1] = _wrapper->clCreateCommandQueue(contexts_[1], devices_[1], 0, NULL);
  CHECK_RESULT(cmd_queues_[1] == 0, "clCreateCommandQueue failed");
  busAddressableBuff_ =
      _wrapper->clCreateBuffer(contexts_[0], CL_MEM_BUS_ADDRESSABLE_AMD, bufSize_, 0, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  error_ = _wrapper->clEnqueueMakeBuffersResidentAMD(cmd_queues_[0], 1, &busAddressableBuff_, true,
                                                     &busAddr_, 0, 0, 0);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMakeBuffersResidentAMD failed");
  extPhysicalBuff_ = _wrapper->clCreateBuffer(contexts_[1], CL_MEM_EXTERNAL_PHYSICAL_AMD, bufSize_,
                                              &busAddr_, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateBuffer failed");
  error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, 0, 0, 0, 0, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
  error_ = _wrapper->clFinish(cmd_queues_[1]);
  CHECK_RESULT(error_, "clFinish failed");
  srcBuff_ = _wrapper->clCreateBuffer(contexts_[1], CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      bufSize_, inputArr_, &error_);
  CHECK_RESULT(error_ != CL_SUCCESS, "clCreateBuffer failed");
  error_ = _wrapper->clEnqueueMigrateMemObjects(cmd_queues_[1], 1, &extPhysicalBuff_, 0, 0, 0, 0);
  CHECK_RESULT(error_, "clEnqueueMigrateMemObjects failed");
  error_ = _wrapper->clFinish(cmd_queues_[1]);
  CHECK_RESULT(error_, "clFinish failed");
  error_ = _wrapper->clEnqueueMigrateMemObjects(cmd_queues_[1], 1, &srcBuff_, 0, 0, 0, 0);
  CHECK_RESULT(error_, "clEnqueueMigrateMemObjects failed");
  error_ = _wrapper->clFinish(cmd_queues_[1]);
  CHECK_RESULT(error_, "clFinish failed");
  if (_openTest == 2) {
    program_ = _wrapper->clCreateProgramWithSource(contexts_[1], 1, &kernel_str_, NULL, &error_);
    CHECK_RESULT(error_, "clCreateProgramWithSource failed");
    error_ = _wrapper->clBuildProgram(program_, 1, &devices_[1], NULL, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      char* errorstr;
      size_t size;
      _wrapper->clGetProgramBuildInfo(program_, devices_[1], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
      errorstr = new char[size];
      _wrapper->clGetProgramBuildInfo(program_, devices_[1], CL_PROGRAM_BUILD_LOG, size, errorstr,
                                      &size);
      printf("\n%s\n", errorstr);
      delete[] errorstr;
    }
    CHECK_RESULT(error_, "clBuildProgram failed");

    kernel_ = _wrapper->clCreateKernel(program_, "test_kernel", &error_);
    CHECK_RESULT(error_, "clCreateKernel failed");
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&extPhysicalBuff_);
    CHECK_RESULT(error_, "clSetKernelArg failed");
  }
  if (_openTest == 5) {
    cl_image_format format = {CL_R, CL_UNSIGNED_INT32};
    cl_image_desc desc;
    desc.image_type = CL_MEM_OBJECT_IMAGE1D;
    desc.image_width = bufSize_ / sizeof(cl_uint);
    desc.image_height = 0;
    desc.image_depth = 0;
    desc.image_array_size = 0;
    desc.image_row_pitch = 0;
    desc.image_slice_pitch = 0;
    desc.num_mip_levels = 0;
    desc.num_samples = 0;
    desc.buffer = (cl_mem)NULL;
    image_ = _wrapper->clCreateImage(contexts_[1], CL_MEM_READ_ONLY, &format, &desc, 0, &error_);
    CHECK_RESULT(error_, "clCreateImage failed");
  }
}

void OCLSDI::run(void) {
  if (silentFailure) {
    return;
  }
  ++markerValue_;
  OCLutil::Thread threads[2];
  ThreadInfo threadInfo[2];
  threadInfo[0].testObj_ = threadInfo[1].testObj_ = this;
  threadInfo[0].threadID_ = 0;
  threadInfo[1].threadID_ = 1;
  threads[0].create(ThreadMain, &threadInfo[0]);
  threads[1].create(ThreadMain, &threadInfo[1]);
  threads[0].join();
  threads[1].join();
  char* descString = (char*)malloc(25 + deviceNames_.size());
  sprintf(descString, "%-20s%s", testNames[_openTest], deviceNames_.c_str());
  testDescString = descString;
  free(descString);
  if (!success_) {
    _errorFlag = true;
    _crcword += 1;
  }
}

void OCLSDI::threadEntry(int threadID) {
  if (silentFailure) {
    return;
  }
  switch (_openTest) {
    case 0:
      testEnqueueWriteBuffer(threadID);
      break;
    case 1:
      testEnqueueCopyBuffer(threadID);
      break;
    case 2:
      testEnqueueNDRangeKernel(threadID);
      break;
    case 3:
      testEnqueueMapBuffer(threadID);
      break;
    case 4:
      testEnqueueWriteBufferRect(threadID);
      break;
    case 5:
      testEnqueueCopyImageToBuffer(threadID);
      break;
  }
}

unsigned int OCLSDI::close(void) {
  if (srcBuff_) {
    error_ = _wrapper->clReleaseMemObject(srcBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (extPhysicalBuff_) {
    error_ = _wrapper->clReleaseMemObject(extPhysicalBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (busAddressableBuff_) {
    error_ = _wrapper->clReleaseMemObject(busAddressableBuff_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (cmd_queues_[0]) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queues_[0]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (cmd_queues_[1]) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queues_[1]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (contexts_[0]) {
    error_ = _wrapper->clReleaseContext(contexts_[0]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }
  if (contexts_[1]) {
    error_ = _wrapper->clReleaseContext(contexts_[1]);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (image_) {
    error_ = _wrapper->clReleaseMemObject(image_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject failed");
  }
  if (inputArr_) {
    free(inputArr_);
  }
  if (outputArr_) {
    free(outputArr_);
  }
  return _crcword;
}

void OCLSDI::readAndVerifyResult() {
  memset(outputArr_, 0, bufSize_);
  error_ =
      _wrapper->clEnqueueWaitSignalAMD(cmd_queues_[0], busAddressableBuff_, markerValue_, 0, 0, 0);
  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWaitSignalAMD failed");
  error_ = _wrapper->clEnqueueReadBuffer(cmd_queues_[0], busAddressableBuff_, CL_TRUE, 0, bufSize_,
                                         outputArr_, 0, 0, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueReadBuffer failed");
  success_ = (memcmp(inputArr_, outputArr_, bufSize_) == 0);
}

void OCLSDI::testEnqueueCopyImageToBuffer(int threadID) {
  if (threadID == 0) {
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {bufSize_ / sizeof(cl_uint), 1, 1};
    memset(inputArr_, (_openTest + 1), bufSize_);
    error_ = _wrapper->clEnqueueWriteImage(cmd_queues_[1], image_, CL_TRUE, origin, region, 0, 0,
                                           inputArr_, 0, 0, 0);
    CHECK_RESULT(error_, "clEnqueueWriteImage failed");
    _wrapper->clFinish(cmd_queues_[1]);
    error_ = _wrapper->clEnqueueCopyImageToBuffer(cmd_queues_[1], image_, extPhysicalBuff_, origin,
                                                  region, 0, 0, 0, 0);
    CHECK_RESULT(error_, "clEnqueueCopyImageToBuffer failed");
    _wrapper->clFinish(cmd_queues_[1]);
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    readAndVerifyResult();
  }
}

void OCLSDI::testEnqueueWriteBufferRect(int threadID) {
  size_t width = (size_t)sqrt((float)bufSize_);
  size_t bufOrigin[3] = {0, 0, 0};
  size_t hostOrigin[3] = {0, 0, 0};
  size_t region[3] = {width, width, 1};
  if (threadID == 0) {
    memset(inputArr_, (_openTest + 1), bufSize_);
    error_ = _wrapper->clEnqueueWriteBufferRect(cmd_queues_[1], extPhysicalBuff_, CL_TRUE,
                                                bufOrigin, hostOrigin, region, width, 0, width, 0,
                                                inputArr_, 0, 0, 0);
    CHECK_RESULT(error_, "clEnqueueWriteBufferRect failed");
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    memset(outputArr_, 0, bufSize_);
    error_ = _wrapper->clEnqueueWaitSignalAMD(cmd_queues_[0], busAddressableBuff_, markerValue_, 0,
                                              0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWaitSignalAMD failed");
    error_ = _wrapper->clEnqueueReadBufferRect(cmd_queues_[0], busAddressableBuff_, CL_TRUE,
                                               bufOrigin, hostOrigin, region, width, 0, width, 0,
                                               outputArr_, 0, 0, 0);
    CHECK_RESULT(error_, "clEnqueueReadBufferRect failed");
    success_ = (memcmp(inputArr_, outputArr_, bufSize_) == 0);
  }
}

void OCLSDI::testEnqueueMapBuffer(int threadID) {
  if (threadID == 0) {
    memset(inputArr_, (_openTest + 1), bufSize_);
    error_ = _wrapper->clEnqueueWriteBuffer(cmd_queues_[1], extPhysicalBuff_, CL_TRUE, 0, bufSize_,
                                            inputArr_, 0, 0, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    error_ = _wrapper->clEnqueueWaitSignalAMD(cmd_queues_[0], busAddressableBuff_, markerValue_, 0,
                                              0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWaitSignalAMD failed");
    void* ptr = _wrapper->clEnqueueMapBuffer(cmd_queues_[0], busAddressableBuff_, CL_TRUE,
                                             CL_MAP_READ, 0, bufSize_, 0, 0, 0, &error_);
    CHECK_RESULT(error_, "clEnqueueMapBuffer failed");
    success_ = (memcmp(inputArr_, ptr, bufSize_) == 0);
    error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queues_[0], busAddressableBuff_, ptr, 0, 0, 0);
    CHECK_RESULT(error_, "clEnqueueUnmapMemObject failed");
    error_ = _wrapper->clFinish(cmd_queues_[0]);
    CHECK_RESULT(error_, "clFinish failed");
  }
}

void OCLSDI::testEnqueueNDRangeKernel(int threadID) {
  if (threadID == 0) {
    size_t global_work_size = bufSize_ / sizeof(cl_uint);
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queues_[1], kernel_, 1, NULL, &global_work_size,
                                              NULL, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    memset(outputArr_, 0, bufSize_);
    error_ = _wrapper->clEnqueueWaitSignalAMD(cmd_queues_[0], busAddressableBuff_, markerValue_, 0,
                                              0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWaitSignalAMD failed");
    error_ = _wrapper->clEnqueueReadBuffer(cmd_queues_[0], busAddressableBuff_, CL_TRUE, 0,
                                           bufSize_, outputArr_, 0, 0, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
    success_ = true;
    for (cl_uint i = 0; i < bufSize_ / sizeof(cl_uint); ++i) {
      success_ &= (outputArr_[i] == i + 2);
    }
  }
}

void OCLSDI::testEnqueueCopyBuffer(int threadID) {
  if (threadID == 0) {
    memset(inputArr_, (_openTest + 1), bufSize_);
    error_ = _wrapper->clEnqueueWriteBuffer(cmd_queues_[1], srcBuff_, CL_TRUE, 0, bufSize_,
                                            inputArr_, 0, 0, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
    error_ = _wrapper->clEnqueueCopyBuffer(cmd_queues_[1], srcBuff_, extPhysicalBuff_, 0, 0,
                                           bufSize_, 0, NULL, NULL);
    CHECK_RESULT(error_, "clEnqueueCopyBuffer failed");
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    readAndVerifyResult();
  }
}

void OCLSDI::testEnqueueWriteBuffer(int threadID) {
  if (threadID == 0) {
    memset(inputArr_, (_openTest + 1), bufSize_);
    error_ = _wrapper->clEnqueueWriteBuffer(cmd_queues_[1], extPhysicalBuff_, CL_TRUE, 0, bufSize_,
                                            inputArr_, 0, 0, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteBuffer failed");
    error_ = _wrapper->clEnqueueWriteSignalAMD(cmd_queues_[1], extPhysicalBuff_, markerValue_, 0, 0,
                                               0, 0);
    CHECK_RESULT(error_ != CL_SUCCESS, "clEnqueueWriteSignalAMD failed");
    error_ = _wrapper->clFinish(cmd_queues_[1]);
    CHECK_RESULT(error_, "clFinish failed");
  } else {
    readAndVerifyResult();
  }
}
