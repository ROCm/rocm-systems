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

#include "OCLBufferFromImage.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define GROUP_SIZE 256

const static char strKernel[] =
    "__kernel void buffer2bufferCopy(                                          "
    "         \n"
    "    __global char* input,                                                 "
    "          \n"
    "    __global char* output)                                                "
    "          \n"
    "{                                                                         "
    "         \n"
    "    int coord = (int)(get_global_id(0));                                  "
    "          \n"
    "    output[coord] = input[coord];                                         "
    "          \n"
    "}                                                                         "
    "         \n";

typedef CL_API_ENTRY cl_mem(CL_API_CALL* clCreateBufferFromImageAMD_fn)(cl_context context,
                                                                        cl_mem image,
                                                                        cl_int* errcode_ret);
clCreateBufferFromImageAMD_fn clCreateBufferFromImageAMD;

OCLBufferFromImage::OCLBufferFromImage() : OCLTestImp() {
  _numSubTests = 2;
  blockSizeX = GROUP_SIZE;
  blockSizeY = 1;
}

OCLBufferFromImage::~OCLBufferFromImage() {}

void OCLBufferFromImage::open(unsigned int test, char* units, double& conversion,
                              unsigned int deviceId) {
  buffer = bufferImage = clImage2D = bufferOut = NULL;
  done = false;
  pitchAlignment = 0;
  bufferSize = 0;

  _openTest = test;
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLTestImp::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  cl_device_type deviceType;
  error_ = _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_TYPE, sizeof(deviceType),
                                     &deviceType, NULL);
  CHECK_RESULT((error_ != CL_SUCCESS), "CL_DEVICE_TYPE failed");

  if (!(deviceType & CL_DEVICE_TYPE_GPU)) {
    testDescString = "GPU device is required for this test!\n";
    done = true;
    return;
  }

  cl_bool imageSupport;
  size_t size;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport),
                            &imageSupport, &size);
  if (!imageSupport) {
    testDescString = "Image not supported, skipping this test! ";
    done = true;
    return;
  }

  clCreateBufferFromImageAMD =
      (clCreateBufferFromImageAMD_fn)clGetExtensionFunctionAddressForPlatform(
          platform_, "clCreateBufferFromImageAMD");
  if (clCreateBufferFromImageAMD == NULL) {
    testDescString = "clCreateBufferFromImageAMD not found!\n";
    done = true;
    return;
  }

  CompileKernel();
  AllocateOpenCLBuffer();
}

void OCLBufferFromImage::run(void) {
  if (_errorFlag || done) {
    return;
  }

  if ((_openTest % 2) == 0) {
    testReadBuffer(bufferImage);
  } else {
    testKernel();
  }
}

void OCLBufferFromImage::AllocateOpenCLBuffer() {
  cl_int status = 0;

  size_t size = 0;
  pitchAlignment = 0;
  status = _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DEVICE_IMAGE_PITCH_ALIGNMENT,
                                     sizeof(cl_uint), &pitchAlignment, &size);
  pitchAlignment--;

  const unsigned int requiredPitch = ((imageWidth + pitchAlignment) & ~pitchAlignment);
  const unsigned int pitch = requiredPitch;
  bufferSize = pitch * imageHeight;

  unsigned char* sourceData = new unsigned char[bufferSize];

  // init data
  for (unsigned int y = 0; y < bufferSize; y++) {
    *(sourceData + y) = y;
  }
  buffer = _wrapper->clCreateBuffer(context_, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, bufferSize,
                                    sourceData, &status);

  delete[] sourceData;

  const cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};
#if defined(CL_VERSION_2_0)
  const cl_image_desc desc = {
      CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, pitch, 0, 0, 0, {buffer}};
#else
  const cl_image_desc desc = {
      CL_MEM_OBJECT_IMAGE2D, imageWidth / 4, imageHeight, 0, 0, pitch, 0, 0, 0, buffer};
#endif
  clImage2D = _wrapper->clCreateImage(context_, CL_MEM_READ_WRITE, &format, &desc, NULL, &status);
  CHECK_RESULT(clImage2D == NULL || status != CL_SUCCESS, "AllocateOpenCLImage() failed");

  bufferImage = clCreateBufferFromImageAMD(context_, clImage2D, &status);
  char c[1024];
  _wrapper->clGetDeviceInfo(devices_[_deviceId], CL_DRIVER_VERSION, sizeof(c), &c, NULL);
  if (status == CL_INVALID_OPERATION) {
    testDescString = "clCreateBufferFromImageAMD not supported on this device!\n";
    done = true;
    return;
  }
  CHECK_RESULT(bufferImage == NULL || status != CL_SUCCESS,
               "clCreateBufferFromImage(bufferOut) failed");

  bufferOut = _wrapper->clCreateBuffer(context_, CL_MEM_READ_WRITE, bufferSize, NULL, &status);
  CHECK_RESULT(bufferOut == NULL || status != CL_SUCCESS, "clCreateBuffer(bufferOut) failed");
}

void OCLBufferFromImage::testReadBuffer(cl_mem buffer) {
  cl_int status = 0;
  unsigned char* dstData = new unsigned char[bufferSize];

  status = clEnqueueReadBuffer(cmdQueues_[_deviceId], buffer, 1, 0, bufferSize, dstData, 0, 0, 0);

  ::clFinish(cmdQueues_[_deviceId]);

  for (unsigned int y = 0; y < bufferSize; y++) {
    if (*(dstData + y) != (unsigned char)y) {
      CHECK_RESULT_NO_RETURN(true, "CheckCLBuffer: *(dstData+y)!=y => %i != %i", *(dstData + y), y);
      goto cleanup;
    }
  }
cleanup:

  delete[] dstData;
}

void OCLBufferFromImage::testKernel() {
  CopyOpenCLBuffer(bufferImage);

  testReadBuffer(bufferOut);
}

unsigned int OCLBufferFromImage::close(void) {
  if (bufferImage != NULL) clReleaseMemObject(bufferImage);
  if (clImage2D != NULL) clReleaseMemObject(clImage2D);
  if (buffer != NULL) clReleaseMemObject(buffer);
  if (bufferOut != NULL) clReleaseMemObject(bufferOut);
  return OCLTestImp::close();
}

void OCLBufferFromImage::CopyOpenCLBuffer(cl_mem buffer) {
  cl_int status = 0;

  // Set appropriate arguments to the kernel2D

  // input buffer image
  status = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buffer);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLBuffer() failed at "
               "clSetKernelArg(kernel_,0,sizeof(cl_mem),&buffer)");
  status = clSetKernelArg(kernel_, 1, sizeof(cl_mem), &bufferOut);
  CHECK_RESULT((status != CL_SUCCESS),
               "CopyOpenCLBuffer() failed at "
               "clSetKernelArg(kernel_,1,sizeof(cl_mem),&bufferOut)");

  // Enqueue a kernel run call.
  size_t global_work_offset[] = {0};
  size_t globalThreads[] = {bufferSize};
  size_t localThreads[] = {blockSizeX};

  status = clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 1, NULL, globalThreads, NULL, 0,
                                  NULL, 0);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLBuffer() failed at clEnqueueNDRangeKernel");

  status = clFinish(cmdQueues_[_deviceId]);
  CHECK_RESULT((status != CL_SUCCESS), "CopyOpenCLBuffer() failed at clFinish");
}

void OCLBufferFromImage::CompileKernel() {
  cl_int status = 0;

  size_t kernelSize = sizeof(strKernel);
  const char* strs = (const char*)&strKernel[0];

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strs, &kernelSize, &status);

  status = _wrapper->clBuildProgram(program_, 1, &devices_[_deviceId], NULL, NULL, NULL);
  if (status != CL_SUCCESS) {
    if (status == CL_BUILD_PROGRAM_FAILURE) {
      cl_int logStatus;
      size_t buildLogSize = 0;
      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, NULL, &buildLogSize);
      std::string buildLog;
      buildLog.resize(buildLogSize);

      logStatus = clGetProgramBuildInfo(program_, devices_[_deviceId], CL_PROGRAM_BUILD_LOG,
                                        buildLogSize, &buildLog[0], NULL);
      printf("%s", buildLog.c_str());
    }
    return;
  }
  // get a kernel object handle for a kernel with the given name
  kernel_ = _wrapper->clCreateKernel(program_, "buffer2bufferCopy", &status);

  size_t kernel2DWorkGroupSize = 0;
  status = clGetKernelWorkGroupInfo(kernel_, devices_[_deviceId], CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(size_t), &kernel2DWorkGroupSize, 0);

  if ((blockSizeX * blockSizeY) > kernel2DWorkGroupSize) {
    if (blockSizeX > kernel2DWorkGroupSize) {
      blockSizeX = kernel2DWorkGroupSize;
      blockSizeY = 1;
    }
  }
}
