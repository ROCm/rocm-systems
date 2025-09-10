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

#include "OCLGLMsaaTexture.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

const static char* strKernel =
    "__kernel void gl_msaa_test( __global uint4 *output, read_only "
    "image2d_msaa_t source, unsigned int numSamples){   \n"
    "    int  tidX = get_global_id(0);\n"
    "    int  tidY = get_global_id(1);\n"
    "    for (int i = 0 ; i < numSamples ; i++) {\n"
    "       uint4 value = read_imageui( source, (int2)( tidX, tidY ) ,i);\n"
    "       int index = (tidY * get_image_width( source ) + tidX)*numSamples + "
    "i;\n"
    "       output[ index ] =  value;\n"
    "   }\n"
    "}\n";

const static char* glDownSampleShader =
    "uniform sampler2DMS MsaaTex;\n"
    "uniform int numSamples;\n"
    "uniform ivec2 resolution;\n"
    "\n"
    "varying vec4  gl_TexCoord[ ];  \n"
    "\n"
    "void main(void)\n"
    "{\n"
    "    vec4 accum = vec4(0.0,0.0,0.0,0.0);\n"
    "    ivec2 coord = ivec2(resolution * gl_TexCoord[0].xy) ;\n"
    "    for ( int i = 0 ; i < numSamples ; i++)\n"
    "    {\n"
    "        accum += texelFetch(MsaaTex,coord,i);\n"
    "    }\n"
    "    accum /= numSamples;\n"
    "    \n"
    "  \n"
    "        \n"
    "    gl_FragColor = accum;\n"
    "}";

OCLGLMsaaTexture::OCLGLMsaaTexture()
    : msaaDepthBuffer_(0),
      msaaFrameBufferOBJ_(0),
      msaaColorBuffer_(0),
      glShader_(0),
      glprogram_(0),
      clOutputBuffer_(0),
      clMsaa_(0),
      pGLOutput_(0),
      pCLOutput_(0) {
  _numSubTests = 1;
  _currentTest = 0;
}

OCLGLMsaaTexture::~OCLGLMsaaTexture() {}

void OCLGLMsaaTexture::open(unsigned int test, char* units, double& conversion,
                            unsigned int deviceId) {
  OCLGLCommon::open(test, units, conversion, deviceId);
  if (_errorFlag) return;

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

  kernel_ = _wrapper->clCreateKernel(program_, "gl_msaa_test", &error_);
  CHECK_RESULT((error_ != CL_SUCCESS), "clCreateKernel() failed (%d)", error_);
}

void OCLGLMsaaTexture::run(void) {
  if (_errorFlag) {
    return;
  }
  bool retVal;
  switch (_currentTest) {
    case 0:
      retVal = testMsaaRead(GL_RGBA, 2);
      break;
    default:
      CHECK_RESULT(true, "unsupported test number\n");
  }
  CHECK_RESULT((retVal != true), "cl-gl depth test failed ");
}

unsigned int OCLGLMsaaTexture::close(void) {
  if (pGLOutput_) {
    free(pGLOutput_);
    pGLOutput_ = NULL;
  }

  if (pCLOutput_) {
    free(pCLOutput_);
    pCLOutput_ = NULL;
  }

  clReleaseMemObject(clMsaa_);
  clReleaseMemObject(clOutputBuffer_);

  glFinish();
  // unbind the texture and frame buffer.
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, 0, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);

  // clean gl resources
  glDeleteFramebuffers(1, &msaaFrameBufferOBJ_);
  msaaFrameBufferOBJ_ = 0;
  glDeleteTextures(1, &msaaColorBuffer_);
  msaaColorBuffer_ = 0;
  glDeleteTextures(1, &msaaDepthBuffer_);
  msaaDepthBuffer_ = 0;

  glDeleteProgram(glprogram_);
  glDeleteShader(glShader_);

  return OCLGLCommon::close();
}

bool OCLGLMsaaTexture::testMsaaRead(GLint internalFormat, unsigned int numSamples) {
  size_t dimSizes[] = {c_dimSize, c_dimSize};

  unsigned int bufferSize = c_dimSize * c_dimSize * 4;
  bool retVal = false;
  createGLFragmentProgramFromSource(glDownSampleShader, glShader_, glprogram_);

  /////////////////////
  // create msaa FBO //
  /////////////////////
  glGenFramebuffers(1, &msaaFrameBufferOBJ_);
  glBindFramebuffer(GL_FRAMEBUFFER, msaaFrameBufferOBJ_);

  // create   textures
  glGenTextures(1, &msaaColorBuffer_);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaaColorBuffer_);
  glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, numSamples, GL_RGBA8, c_dimSize, c_dimSize,
                          GL_TRUE);

  glGenTextures(1, &msaaDepthBuffer_);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaaDepthBuffer_);
  glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, numSamples, GL_DEPTH_COMPONENT24, c_dimSize,
                          c_dimSize, GL_TRUE);

  //
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, msaaColorBuffer_, 0);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, msaaDepthBuffer_, 0);

  // verify all resource allocations are well.
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
  cl_int error;
  clOutputBuffer_ = _wrapper->clCreateBuffer(context_, CL_MEM_WRITE_ONLY, bufferSize, NULL, &error);
  if (CL_SUCCESS != error) return false;

  clMsaa_ = _wrapper->clCreateFromGLTexture(context_, CL_MEM_READ_WRITE, GL_TEXTURE_2D_MULTISAMPLE,
                                            0, msaaColorBuffer_, &error);
  if (CL_SUCCESS != error) return false;

  GLsizei samples;
  error = _wrapper->clGetGLTextureInfo(clMsaa_, CL_GL_NUM_SAMPLES, sizeof(samples), &samples, NULL);

  error = _wrapper->clEnqueueAcquireGLObjects(cmdQueues_[_deviceId], 1, &clMsaa_, 0, NULL, NULL);
  if (CL_SUCCESS != error) return false;

  _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), &clOutputBuffer_);

  _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_mem), &clMsaa_);

  _wrapper->clSetKernelArg(kernel_, 2, sizeof(unsigned int), &numSamples);

  _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, dimSizes, NULL, 0, NULL,
                                   NULL);

  _wrapper->clEnqueueReleaseGLObjects(cmdQueues_[_deviceId], 1, &clMsaa_, 0, NULL, NULL);

  pGLOutput_ = (unsigned int*)malloc(bufferSize);
  pCLOutput_ = (unsigned int*)malloc(bufferSize);

  _wrapper->clEnqueueReadBuffer(cmdQueues_[_deviceId], clOutputBuffer_, CL_TRUE, 0, bufferSize,
                                pCLOutput_, 0, NULL, NULL);

  // down sample
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaaColorBuffer_);
  glUseProgram(glprogram_);

  glUniform1i(glGetUniformLocation(glprogram_, "numSamples"), numSamples);
  glUniform2i(glGetUniformLocation(glprogram_, "resolution"), c_dimSize, c_dimSize);
  glUniform1i(glGetUniformLocation(glprogram_, "MsaaTex"), 0);

  // printOpenGLError();

  glBegin(GL_QUADS);
  glVertex2f(-1.0f, 1.0f);
  glTexCoord2f(1.0f, 0.0f);
  glVertex2f(1.0f, 1.0f);
  glTexCoord2f(1.0f, 1.0f);
  glVertex2f(1.0f, -1.0f);
  glTexCoord2f(0.0f, 1.0f);
  glVertex2f(-1.0f, -1.0f);
  glTexCoord2f(0.0f, 0.0f);
  glEnd();

  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
  glUseProgram(0);

  glReadPixels(0, 0, c_dimSize, c_dimSize, GL_BGRA, GL_UNSIGNED_BYTE, pGLOutput_);

  if (absDiff(pGLOutput_, pCLOutput_, c_dimSize)) retVal = true;

  return retVal;
}

bool OCLGLMsaaTexture::absDiff(unsigned int* pGLBuffer, unsigned int* pCLBuffer,
                               const unsigned int c_dimSize) {
  bool retVal = true;
  for (unsigned int i = 0; i < c_dimSize * c_dimSize; i++) {
    char clPixel[4];
    char glPixel[4];
    char diff[4] = {0};
    memcpy(clPixel, &(pCLBuffer[i]), sizeof(clPixel));
    memcpy(glPixel, &(pGLBuffer[i]), sizeof(glPixel));

    for (int j = 0; j < 4; j++) {
      diff[j] = abs(clPixel[j] - glPixel[i]);
      if (diff[j] > 10) retVal = false;
    }
  }
  return retVal;
}
