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

#include "OCLGLDepthBuffer.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
    "__kernel void gldepths_test( __global float *output, read_only  image2d_t "
    "source, sampler_t sampler){   \n"
    "    int  tidX = get_global_id(0);\n"
    "    int  tidY = get_global_id(1);\n"
    "    float4 value = read_imagef( source, sampler, (int2)( tidX, tidY ) );\n"
    "    output[ tidY * get_image_width( source ) + tidX ] =  value.z;\n"
    "}\n";

OCLGLDepthBuffer::OCLGLDepthBuffer()
    : glDepthBuffer_(0),
      frameBufferOBJ_(0),
      colorBuffer_(0),
      clOutputBuffer_(0),
      clDepth_(0),
      clSampler_(0),
      pGLOutput_(0),
      pCLOutput_(0),
      extensionSupported_(false) {
  _numSubTests = 2;
  _currentTest = 0;
}

OCLGLDepthBuffer::~OCLGLDepthBuffer() {}

void OCLGLDepthBuffer::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  char* pExtensions = (char*)malloc(8192);
  size_t returnSize;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 8192, pExtensions,
                            &returnSize);

  // if extension if not supported
  if (!strstr(pExtensions, "cl_khr_gl_depth_images")) {
    printf("skipping test depth interop not supported\n");
    free(pExtensions);
    return;
  }
  free(pExtensions);
  extensionSupported_ = true;

  _currentTest = test;

  // Build the kernel
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
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

  kernel_ = _wrapper->clCreateKernel(program_, "gldepths_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
}

void OCLGLDepthBuffer::run(void) {
  if (_errorFlag || !extensionSupported_) {
    return;
  }
  bool retVal;
  switch (_currentTest) {
    case 0:
      retVal = testDepthRead(GL_DEPTH_COMPONENT32F, GL_DEPTH_ATTACHMENT);
      break;
    case 1:
      retVal = testDepthRead(GL_DEPTH_COMPONENT16, GL_DEPTH_ATTACHMENT);
      break;
    case 2:
      retVal = testDepthRead(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT);
      break;
    case 3:
      retVal = testDepthRead(GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL_ATTACHMENT);
      break;
    default:
      CHECK_RESULT(true, "unsupported test number\n");
  }
  CHECK_RESULT((retVal != true), "cl-gl depth test failed ");
}

bool OCLGLDepthBuffer::testDepthRead(GLint internalFormat, GLenum attachmentType) {
  cl_int error;
  size_t dimSizes[] = {c_dimSize, c_dimSize};

  unsigned int bufferSize = c_dimSize * c_dimSize * 4;
  bool retVal = false;

  pGLOutput_ = (float*)malloc(bufferSize);
  pCLOutput_ = (float*)malloc(bufferSize);
  // create Frame buffer object
  glGenFramebuffers(1, &frameBufferOBJ_);

  // create   textures
  glGenTextures(1, &colorBuffer_);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, colorBuffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, c_dimSize, c_dimSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  // create a renderbuffer for the depth/stencil buffer
  glGenRenderbuffers(1, &glDepthBuffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, glDepthBuffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, internalFormat, c_dimSize, c_dimSize);

  //
  glBindFramebuffer(GL_FRAMEBUFFER, frameBufferOBJ_);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorBuffer_, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachmentType, GL_RENDERBUFFER, glDepthBuffer_);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (GL_FRAMEBUFFER_COMPLETE != status) {
    return false;
  }
  // set up gl state machine
  glViewport(0, 0, c_dimSize, c_dimSize);  // Reset The Current Viewport
  glMatrixMode(GL_PROJECTION);             // Select The Projection Matrix
  glLoadIdentity();                        // Reset The Projection Matrix
  gluPerspective(30.0f, (GLfloat)c_dimSize / (GLfloat)c_dimSize, 0.1f, 100.0f);
  glMatrixMode(GL_MODELVIEW);  // Select The Modelview Matrix
  glLoadIdentity();
  glEnable(GL_DEPTH_TEST);
  // The Type Of Depth Testing To Do
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear Screen And Depth Buffer
  glBegin(GL_QUADS);                                   // Draw A Quad
  glVertex3f(-1.0f, 1.0f, -6.0f);                      // Top Left
  glVertex3f(1.0f, 1.0f, -6.0f);                       // Top Right
  glVertex3f(1.0f, -1.0f, -3.0f);                      // Bottom Right
  glVertex3f(-1.0f, -1.0f, -3.0f);                     // Bottom Left
  glEnd();

  glFinish();

  clDepth_ =
      _wrapper->clCreateFromGLRenderbuffer(context_, CL_MEM_READ_WRITE, glDepthBuffer_, &error);
  if (CL_SUCCESS != error) {
    printf("clCreateFromGLRenderbuffer failed\n");
    return false;
  }

  clOutputBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
  if (CL_SUCCESS != error) return false;

  clSampler_ =
      _wrapper->clCreateSampler(context_, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
  if (CL_SUCCESS != error) return false;

  error = _wrapper->clEnqueueAcquireGLObjects(cmdQueues_[_deviceId], 1, &clDepth_, 0, NULL, NULL);

  _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clOutputBuffer_);

  _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), &clDepth_);

  _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_sampler), &clSampler_);

  _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, dimSizes, NULL, 0, NULL,
                                   NULL);

  _wrapper->clEnqueueReleaseGLObjects(cmdQueues_[_deviceId], 1, &clDepth_, 0, NULL, NULL);

  _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], clOutputBuffer_, CL_TRUE, 0, bufferSize,
                                pCLOutput_, 0, NULL, NULL);

  glReadPixels(0, 0, c_dimSize, c_dimSize, GL_DEPTH_COMPONENT, GL_FLOAT, pGLOutput_);

  // test that both resources are identical.
  if (0 == memcmp(pGLOutput_, pCLOutput_, bufferSize)) {
    retVal = true;  // test successful
  } else {
    printf("expected results is different from actual results\n");
    dumpBuffer(pGLOutput_, "GLDepth.csv", c_dimSize);
    dumpBuffer(pCLOutput_, "CLDepth.csv", c_dimSize);
  }

  return retVal;
}

unsigned int OCLGLDepthBuffer::close(void) {
  if (pGLOutput_) {
    free(pGLOutput_);
    pGLOutput_ = NULL;
  }

  if (pCLOutput_) {
    free(pCLOutput_);
    pCLOutput_ = NULL;
  }

  clReleaseMemObject(clDepth_);
  clReleaseMemObject(clOutputBuffer_);
  clReleaseSampler(clSampler_);
  // unbind the texture and frame buffer.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  // clean gl resources
  glDeleteFramebuffers(1, &frameBufferOBJ_);
  frameBufferOBJ_ = 0;
  glDeleteTextures(1, &colorBuffer_);
  colorBuffer_ = 0;
  glDeleteTextures(1, &glDepthBuffer_);
  glDepthBuffer_ = 0;

  return OCLGLCommon::close();
}

// helper functions
unsigned int OCLGLDepthBuffer::formatToSize(GLint internalFormat) {
  switch (internalFormat) {
    case GL_DEPTH_COMPONENT32F:
      return 4;
      break;
    case GL_DEPTH_COMPONENT16:
      return 2;
      break;
    case GL_DEPTH24_STENCIL8:
      return 4;
      break;
    case GL_DEPTH32F_STENCIL8:
      return 8;
      break;
    default:
      return 0;
  }
}
