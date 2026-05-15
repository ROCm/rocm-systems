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

#include "OCLGLDepthTex.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "__kernel void gldepths_test( __global float *output, read_only image2d_t "
    "source, sampler_t sampler){   \n"
    "    int  tidX = get_global_id(0);\n"
    "    int  tidY = get_global_id(1);\n"
    "    float4 value = read_imagef( source, sampler, (int2)( tidX, tidY ) );\n"
    "    output[ tidY * get_image_width( source ) + tidX ] =  value.z;\n"
    "}\n";

OCLGLDepthTex::OCLGLDepthTex()
    : glDepthBuffer_(0),
      frameBufferOBJ_(0),
      colorBuffer_(0),
      clOutputBuffer_(0),
      clDepth_(0),
      clSampler_(0),
      pGLOutput_(0),
      pCLOutput_(0),
      extensionSupported_(false) {
  _numSubTests = 8;
  _currentTest = 0;
}

OCLGLDepthTex::~OCLGLDepthTex() {}

void OCLGLDepthTex::open(unsigned int test, char* units, double& conversion,
                         unsigned int deviceId) {
  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

  char* pExtensions = (char*)malloc(8192);
  size_t returnSize;
  _wrapper->clGetDeviceInfo(devices_[deviceId], CL_DEVICE_EXTENSIONS, 8192, pExtensions,
                            &returnSize);

  // if extension if not supported
  if (!strstr(pExtensions, "cl_khr_gl_depth_images")) {
    free(pExtensions);
    printf("skipping test depth interop not supported\n");
    return;
  }
  free(pExtensions);
  extensionSupported_ = true;

  static const char* OpenCL20Kernel = "-cl-std=CL2.0";
  const char* options = OpenCL20Kernel;
  if (test < 4) {
    options = NULL;
  }
  _currentTest = test % 4;

  // Build the kernel
  program_ = _wrapper->clCreateProgramWithSource(context_, 1, &strKernel, NULL, &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateProgramWithSource()  failed (%d)", error_);

  error_ = _wrapper->clBuildProgram(program_, 1, &devices_[deviceId], options, NULL, NULL);
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

void OCLGLDepthTex::run(void) {
  if (_errorFlag || !extensionSupported_) {
    return;
  }
  bool retVal;
  switch (_currentTest) {
    case 0:
      retVal = testDepthRead(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8);
      break;
    case 1:
      retVal = testDepthRead(GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_FLOAT);
      break;
    case 2:
      retVal = testDepthRead(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT);
      break;
    case 3:
      retVal =
          testDepthRead(GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV);
      break;
    default:
      CHECK_RESULT(true, "unsupported test number\n");
  }
  CHECK_RESULT((retVal != true), "cl-gl depth test failed ");
}

bool OCLGLDepthTex::testDepthRead(GLint internalFormat, GLenum format, GLenum type) {
  const unsigned int bufferSize = c_dimSize * c_dimSize * 4;

  pGLOutput_ = (float*)malloc(bufferSize);
  pCLOutput_ = (float*)malloc(bufferSize);
  size_t dimSizes[] = {c_dimSize, c_dimSize};

  bool retVal = false;
  // create Frame buffer object
  glGenFramebuffers(1, &frameBufferOBJ_);
  glBindFramebuffer(GL_FRAMEBUFFER, frameBufferOBJ_);

  // create   textures
  glGenTextures(1, &colorBuffer_);
  glBindTexture(GL_TEXTURE_2D, colorBuffer_);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, c_dimSize, c_dimSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glGenTextures(1, &glDepthBuffer_);
  glBindTexture(GL_TEXTURE_2D, glDepthBuffer_);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, c_dimSize, c_dimSize, 0, format, type, 0);
  GLint glError = glGetError();
  //
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorBuffer_, 0);

  if (GL_DEPTH_COMPONENT == format) {
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, glDepthBuffer_, 0);
  } else {
    glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, glDepthBuffer_, 0);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, frameBufferOBJ_);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (GL_FRAMEBUFFER_COMPLETE != status) {
    printf("frame buffer incomplete!\n");
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
  glBindFramebuffer(GL_FRAMEBUFFER, frameBufferOBJ_);

  cl_int error;

  clOutputBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
  if (CL_SUCCESS != error) return false;

  clSampler_ =
      _wrapper->clCreateSampler(context_, CL_FALSE, CL_ADDRESS_NONE, CL_FILTER_NEAREST, &error);
  if (CL_SUCCESS != error) return false;

  clDepth_ = _wrapper->clCreateFromGLTexture(context_, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0,
                                             glDepthBuffer_, &error);
  if (CL_SUCCESS != error) return false;

  for (int i = 0; i < 3; ++i) {
    // The Type Of Depth Testing To Do
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  // Clear Screen And Depth Buffer

    const float zValues[3][2] = {
        {-6.f, -3.f},
        {-5.f, -2.f},
        {-4.f, -1.f},
    };

    glBegin(GL_QUADS);                        // Draw A Quad
    glVertex3f(-1.0f, 1.0f, zValues[i][0]);   // Top Left
    glVertex3f(1.0f, 1.0f, zValues[i][0]);    // Top Right
    glVertex3f(1.0f, -1.0f, zValues[i][1]);   // Bottom Right
    glVertex3f(-1.0f, -1.0f, zValues[i][1]);  // Bottom Left
    glEnd();

    glFinish();

    error = _wrapper->clEnqueueAcquireGLObjects(cmdQueues_[_deviceId], 1, &clDepth_, 0, NULL, NULL);

    _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clOutputBuffer_);

    _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), &clDepth_);

    _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_sampler), &clSampler_);

    _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, dimSizes, NULL, 0,
                                     NULL, NULL);

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
      dumpBuffer(pCLOutput_, "clDepth_.csv", c_dimSize);
    }
  }

  return retVal;
}

unsigned int OCLGLDepthTex::close(void) {
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
