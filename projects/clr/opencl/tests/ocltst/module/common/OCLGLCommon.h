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

#ifndef _OCL_GL_COMMON_H_
#define _OCL_GL_COMMON_H_

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include "OCLTestImp.h"

typedef struct OCLGLHandle_* OCLGLHandle;

#define printOpenGLError() OCLGLCommon::printOglError(__FILE__, __LINE__)

class OCLGLCommon : public OCLTestImp {
 public:
  /////////////////////////////////////////
  // private initialization and clean-up //
  /////////////////////////////////////////
  OCLGLCommon();
  virtual ~OCLGLCommon();
  ///////////////////////
  // virtual interface //
  ///////////////////////
  virtual void open(unsigned int test, char* units, double& conversion, unsigned int& deviceId);
  virtual unsigned int close(void);
  static void gluPerspective(double fovy, double aspect, double zNear, double zFar);
  static void dumpBuffer(float* pBuffer, const char fileName[], unsigned int dimSize);
  static int printOglError(char* file, int line);
  static bool createGLFragmentProgramFromSource(const char* source, GLuint& shader,
                                                GLuint& program);
  static void printShaderInfoLog(GLuint shader);
  static void printProgramInfoLog(GLuint program);

 protected:
  const OCLGLHandle getGLHandle() { return hGL_; }
  void makeCurrent(const OCLGLHandle hGL);
  void getCLContextPropertiesFromGLContext(const OCLGLHandle hGL,
                                           cl_context_properties properties[7]);
  bool createGLContext(OCLGLHandle& hGL);
  void destroyGLContext(OCLGLHandle& hGL);
  bool IsGLEnabled(unsigned int test, char* units, double& conversion, unsigned int deviceId);

 private:
  bool initializeGLContext(OCLGLHandle& hGL);
  void deleteGLContext(OCLGLHandle& hGL);
  bool checkAssociationDeviceWithGLContext(OCLGLHandle& hGL);
  void createCLContextFromGLContext(OCLGLHandle& hGL);

  OCLGLHandle hGL_;
};

#endif  // _OCL_GL_COMMON_H_
