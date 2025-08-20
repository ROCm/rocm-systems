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

#include "OCLPersistent.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "__kernel void persistentImage( write_only image2d_t source){   \n"
    "    int  tidX = get_global_id(0);\n"
    "    int  tidY = get_global_id(1);\n"
    "    write_imagei( source, (int2)( tidX, tidY ),(int4)( tidX, tidY,0,0 ) "
    ");\n"
    "}\n";

OCLPersistent::OCLPersistent() : clImage_(0) { _numSubTests = 1; }

OCLPersistent::~OCLPersistent() {}

void OCLPersistent::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  OCLTestImp::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  // Build the kernel
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed!");

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed!");

  kernel_ = _wrapper->clCreateKernel(program_, "persistentImage", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed!");
  cl_image_format format;
  format.image_channel_data_type = CL_SIGNED_INT32;
  format.image_channel_order = CL_RG;
  cl_image_desc desc = {0};
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = c_dimSize;
  desc.image_height = c_dimSize;
  desc.image_depth = 1;
  desc.image_array_size = 1;
  // CL_MEM_USE_PERSISTENT_MEM_AMD
  clImage_ = clCreateImage(context_, CL_MEM_USE_PERSISTENT_MEM_AMD | CL_MEM_WRITE_ONLY, &format,
                           &desc, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateImage() failed");
}

void OCLPersistent::run(void) {
  _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clImage_);

  size_t dimSizes[] = {c_dimSize, c_dimSize};

  size_t origin[] = {0, 0, 0};
  size_t region[] = {c_dimSize, c_dimSize, 1};
  size_t pitch, slice;
  cl_event event;
  error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, dimSizes, NULL,
                                            0, NULL, NULL);
  error_ = _wrapper->clEnqueueMarkerWithWaitList(cmdQueues_[_deviceId], 0, NULL, &event);

  _wrapper->clFlush(cmdQueues_[_deviceId]);

  cl_uint status;
  _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_uint), &status,
                           NULL);
  while (status != CL_COMPLETE) {
    _wrapper->clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_uint), &status,
                             NULL);
  }

  unsigned int* image = (unsigned int*)_wrapper->clEnqueueMapImage(
      cmdQueues_[_deviceId], clImage_, CL_TRUE, CL_MAP_READ, origin, region, &pitch, &slice, 0,
      NULL, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clEnqueueMapImage() failed");

  bool result = validateImage(image, pitch, c_dimSize);
  CHECK_RESULT(!result, "Validation failed!");

  _wrapper->clEnqueueUnmapMemObject(cmdQueues_[_deviceId], clImage_, image, 0, NULL, NULL);
}

unsigned int OCLPersistent::close(void) {
  _wrapper->clReleaseMemObject(clImage_);

  return OCLTestImp::close();
}

bool OCLPersistent::validateImage(unsigned int* image, size_t pitch, unsigned int dimSize) {
  unsigned int x, y;
  int idx = 0;
  for (y = 0; y < dimSize; y++) {
    for (x = 0; x < dimSize; x++) {
      if ((image[idx] != x) || (image[idx + 1] != y)) {
        printf("Failed at coordinate (%5d, %5d) - R:%d, G:%d value\n", x, y, image[idx],
               image[idx + 1]);
        return false;
      }
      idx += 2;
    }
    image += pitch / sizeof(int);
    idx = 0;
  }
  return true;
}
