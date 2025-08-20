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

#ifndef _OCL_GL_TEXTURE_H_
#define _OCL_GL_TEXTURE_H_

#include <iostream>

#include "OCLGLCommon.h"

class OCLGLTexture : public OCLGLCommon {
 public:
  static const unsigned int c_imageWidth = 512;
  static const unsigned int c_imageHeight = 512;
  static const unsigned int c_elementsPerPixel = 4;

  OCLGLTexture();
  virtual ~OCLGLTexture();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 private:
  unsigned int currentTest_;
  void* inDataGL_;
  void* outDataGL_;
  GLuint inGLTexture_;
  GLuint outGLTexture_;
  bool testRender_;
  template <typename T> bool runTextureTest(GLint internalFormat, GLenum format, GLenum type);
};

template <typename T>
bool OCLGLTexture::runTextureTest(GLint internalFormat, GLenum format, GLenum type) {
  cl_mem image;
  inDataGL_ = malloc(c_imageWidth * c_imageHeight * c_elementsPerPixel * sizeof(T));
  outDataGL_ = malloc(c_imageWidth * c_imageHeight * c_elementsPerPixel * sizeof(T));

  // Initialize input data with random values
  T* inputIterator = (T*)inDataGL_;
  for (unsigned int i = 0; i < c_imageWidth * c_imageHeight * c_elementsPerPixel; i++) {
    inputIterator[i] = (T)(rand() % 255);
  }
  // Initialize output data with zeros
  memset(outDataGL_, 0, c_imageWidth * c_imageHeight * c_elementsPerPixel * sizeof(T));

  // Generate and Bind in & out OpenGL textures
  glGenTextures(1, &inGLTexture_);
  glGenTextures(1, &outGLTexture_);

  glBindTexture(GL_TEXTURE_2D, inGLTexture_);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, (GLsizei)c_imageWidth, (GLsizei)c_imageHeight, 0,
               format, type, inDataGL_);

  glBindTexture(GL_TEXTURE_2D, outGLTexture_);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, (GLsizei)c_imageWidth, (GLsizei)c_imageHeight, 0,
               format, type, outDataGL_);

  glFinish();

  // Create input buffer from GL input texture
  image = _wrapper->clCreateFromGLTexture(context_, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0,
                                          inGLTexture_, &error_);
  if (error_ != CL_SUCCESS) {
    printf("Unable to create input buffer from GL texture (%d)", error_);
    return false;
  }
  buffers_.push_back(image);

  // Create output buffer from GL output texture
  image = _wrapper->clCreateFromGLTexture(context_, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0,
                                          outGLTexture_, &error_);
  if (error_ != CL_SUCCESS) {
    printf("Unable to create output buffer from GL texture (%d)", error_);
    return false;
  }
  buffers_.push_back(image);
  size_t gws[2] = {c_imageWidth, c_imageHeight};

  // Assign args
  for (unsigned int i = 0; i < buffers_.size(); i++) {
    error_ = _wrapper->clSetKernelArg(kernel_, i, sizeof(cl_mem), &buffers()[i]);
    if (error_ != CL_SUCCESS) {
      printf("clSetKernelArg() failed (%d)", error_);
      return false;
    }
  }

  int loop = (testRender_) ? 2 : 1;
  for (int l = 0; l < loop; ++l) {
    if (testRender_ && (l == 0)) {
      GLuint FrameBufferName = 0;
      glGenFramebuffers(1, &FrameBufferName);
      glBindFramebuffer(GL_FRAMEBUFFER, FrameBufferName);
      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, inGLTexture_, 0);
      glClearColor(.5f, 1.f, 1.0f, 0);
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
    }

    error_ =
        _wrapper->clEnqueueAcquireGLObjects(cmdQueues_[_deviceId], 2, &buffers()[0], 0, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      printf("Unable to acquire GL objects (%d)", error_);
      return false;
    }

    error_ = _wrapper->clEnqueueNDRangeKernel(cmdQueues_[_deviceId], kernel_, 2, NULL, gws, NULL, 0,
                                              NULL, NULL);
    if (error_ != CL_SUCCESS) {
      printf("clEnqueueNDRangeKernel() failed (%d)", error_);
      return false;
    }

    error_ =
        _wrapper->clEnqueueReleaseGLObjects(cmdQueues_[_deviceId], 2, &buffers()[0], 0, NULL, NULL);
    if (error_ != CL_SUCCESS) {
      printf("clEnqueueReleaseGLObjects failed (%d)", error_);
      return false;
    }

    error_ = _wrapper->clFinish(cmdQueues_[_deviceId]);
    if (error_ != CL_SUCCESS) {
      printf("clFinish() failed (%d)", error_);
      return false;
    }

    if (testRender_ && (l == 0)) {
      glClearColor(1.f, 1.f, 1.f, 1.f);
      glClear(GL_COLOR_BUFFER_BIT);
      glFinish();
    }
  }

  // Get the results from GL texture
  glBindTexture(GL_TEXTURE_2D, outGLTexture_);
  glActiveTexture(GL_TEXTURE0);
  glGetTexImage(GL_TEXTURE_2D, 0, format, type, outDataGL_);

  // Check output texture data
  inputIterator = (T*)inDataGL_;
  T* outputIterator = (T*)outDataGL_;
  T color;
  switch (type) {
    case GL_UNSIGNED_INT:
      color = (T)0x3f800000;
      break;
    case GL_UNSIGNED_BYTE:
      color = (T)0xff;
      break;
    case GL_SHORT:
      color = (T)0x7fff;
      break;
    case GL_FLOAT:
      color = (T)1.f;
      break;
    default:
      return false;
  }
  for (unsigned int i = 0; i < c_imageWidth * c_imageHeight * c_elementsPerPixel; i++) {
    if (testRender_) {
      if (outputIterator[i] != color) {
        std::cout << "Element " << i
                  << " in output texture is incorrect! (internal format = " << internalFormat
                  << "\n\t expected:" << inputIterator[i]
                  << " differs from actual clear color:" << color << std::endl;
        return false;
      }
    } else if (inputIterator[i] != outputIterator[i]) {
      std::cout << "Element " << i
                << " in output texture is incorrect! (internal format = " << internalFormat
                << "\n\t expected:" << inputIterator[i]
                << " differs from actual: " << outputIterator[i] << std::endl;
      return false;
    }
  }
  return true;
}

#endif  // _OCL_GL_TEXTURE_H_
