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

#include "OCLGLTexture.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

const static char* strKernelui =
    "__kernel void gltexture_test(read_only image2d_t source, write_only "
    "image2d_t dest)    \n"
    "{                                                                         "
    "             \n"
    "    int  tidX = get_global_id(0);                                         "
    "             \n"
    "    int  tidY = get_global_id(1);                                         "
    "             \n"
    "    uint4 pixel = read_imageui(source, (int2)(tidX, tidY));               "
    "             \n"
    "    write_imageui(dest, (int2)(tidX, tidY), pixel);                       "
    "             \n"
    "}";

const static char* strKernelf =
    "__kernel void gltexture_test(read_only image2d_t source, write_only "
    "image2d_t dest)    \n"
    "{                                                                         "
    "             \n"
    "    int  tidX = get_global_id(0);                                         "
    "             \n"
    "    int  tidY = get_global_id(1);                                         "
    "             \n"
    "    float4 pixel = read_imagef(source, (int2)(tidX, tidY));               "
    "             \n"
    "    write_imagef(dest, (int2)(tidX, tidY), pixel);                        "
    "            \n"
    "}                                                                         "
    "             \n";

OCLGLTexture::OCLGLTexture()
    : inDataGL_(NULL), outDataGL_(NULL), inGLTexture_(0), outGLTexture_(0) {
  _numSubTests = 4 * 2;
}

OCLGLTexture::~OCLGLTexture() {}

void OCLGLTexture::open(unsigned int test, char* units, double& conversion, unsigned int deviceId) {
  // Initialize random number seed
  srand((unsigned int)time(NULL));

  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  currentTest_ = test % 4;
  testRender_ = ((test / 4) >= 1) ? true : false;

  // Build the kernel
  if (0 == currentTest_) {
    program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernelui, NULL, &error_);

  } else {
    program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernelf, NULL, &error_);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], NULL, NULL, NULL);
  if (error_ != CL_SUCCESS) {
    char programLog[1024];
    _wrapper->clGetProgramBuildInfo(program_, devices_[deviceId], CL_PROGRAM_BUILD_LOG, 1024,
                                    programLog, 0);
    printf("\n%s\n", programLog);
    fflush(stdout);
  }
  CHECK_RESULT((error_ != CL_SUCCESS), "clBuildProgram() failed (%d)", error_);

  kernel_ = _wrapper->clCreateKernel(program_, "gltexture_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
}

void OCLGLTexture::run(void) {
  bool retVal = false;
  switch (currentTest_) {
    case 0:
      retVal = runTextureTest<unsigned int>(GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT);
      break;
    case 1:
      retVal = runTextureTest<unsigned char>(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
      break;
    case 2:
      retVal = runTextureTest<short>(GL_RGBA16, GL_RGBA, GL_SHORT);
      break;
    case 3:
      retVal = runTextureTest<float>(GL_RGBA32F, GL_RGBA, GL_FLOAT);
      break;
    default:
      CHECK_RESULT(true, "unsupported test number\n");
  }
  CHECK_RESULT((retVal != true), "cl-gl texture interop test failed ");
}

unsigned int OCLGLTexture::close(void) {
  clReleaseMemObject(buffers_[0]);
  clReleaseMemObject(buffers_[1]);
  buffers_.clear();
  // Delete GL in & out buffers
  glFinish();
  glBindTexture(GL_TEXTURE_2D, 0);
  glDeleteTextures(1, &inGLTexture_);
  inGLTexture_ = 0;
  glDeleteTextures(1, &outGLTexture_);
  outGLTexture_ = 0;

  free(inDataGL_);
  inDataGL_ = NULL;
  free(outDataGL_);
  outDataGL_ = NULL;
  return OCLGLCommon::close();
}
