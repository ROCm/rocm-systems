/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GL_MSAA_TEXTURE_H_
#define _OCL_GL_MSAA_TEXTURE_H_

#include "OCLGLCommon.h"

class OCLGLMsaaTexture : public OCLGLCommon {
 public:
  OCLGLMsaaTexture();
  virtual ~OCLGLMsaaTexture();
  static const unsigned int c_dimSize = 128;
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  ////////////////////
  // test functions //
  ////////////////////
  bool testMsaaRead(GLint internalFormat, unsigned int NumSamples);
  unsigned int _currentTest;

  //////////////////////////////
  // private helper functions //
  //////////////////////////////

  // returns element size in bytes.
  static bool absDiff(unsigned int* pGLBuffer, unsigned int* pCLBuffer, const unsigned int dimSize);

  /////////////////////
  // private members //
  /////////////////////
  // GL resource identifiers
  GLuint msaaDepthBuffer_;
  GLuint msaaFrameBufferOBJ_;
  GLuint msaaColorBuffer_;
  GLuint glShader_;
  GLuint glprogram_;
  // CL identifiers
  cl_mem clOutputBuffer_;
  cl_mem clMsaa_;

  unsigned int* pGLOutput_;
  unsigned int* pCLOutput_;
};

#endif  // _OCL_GL_BUFFER_H_
