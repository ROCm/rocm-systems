/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCL_GL_DEPTH_BUFFER_H_
#define _OCL_GL_DEPTH_BUFFER_H_

#include "OCLGLCommon.h"

class OCLGLDepthBuffer : public OCLGLCommon {
 public:
  OCLGLDepthBuffer();
  virtual ~OCLGLDepthBuffer();
  static const unsigned int c_dimSize = 128;
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  ////////////////////
  // test functions //
  ////////////////////
  bool testDepthRead(GLint internalFormat, GLenum attachmentType);
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
  //////////////////////////////
  // private helper functions //
  //////////////////////////////
  // returns element size in bytes.
  static unsigned int formatToSize(GLint internalFormat);
};

#endif  // _OCL_GL_BUFFER_H_
