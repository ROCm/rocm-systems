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

#ifndef _OCL_GL_DEPTH_TEX_H_
#define _OCL_GL_DEPTH_TEX_H_

#include "OCLGLCommon.h"

class OCLGLDepthTex : public OCLGLCommon {
 public:
  OCLGLDepthTex();
  virtual ~OCLGLDepthTex();
  static const unsigned int c_dimSize = 128;
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  ////////////////////
  // test functions //
  ////////////////////
  bool testDepthRead(GLint internalFormat, GLenum format, GLenum type);
  unsigned int _currentTest;

  /////////////////////
  // private members //
  /////////////////////
  // GL resource identifiers
  GLuint glDepthBuffer_;
  GLuint frameBufferOBJ_;
  GLuint colorBuffer_;

  // CL identifiers
  cl_mem clOutputBuffer_;
  cl_mem clDepth_;
  cl_sampler clSampler_;

  // pointers to buffers
  float* pGLOutput_;
  float* pCLOutput_;
  bool extensionSupported_;
};

#endif  // _OCL_GL_BUFFER_H_
