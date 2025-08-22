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

#ifndef _OCLBufferFromImage_H_
#define _OCLBufferFromImage_H_

#include "OCLTestImp.h"

class OCLBufferFromImage : public OCLTestImp {
 public:
  OCLBufferFromImage();
  virtual ~OCLBufferFromImage();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int imageWidth = 1920;
  static const unsigned int imageHeight = 1080;

  void testReadBuffer(cl_mem buffer);
  void testKernel();
  void AllocateOpenCLBuffer();
  void CopyOpenCLBuffer(cl_mem buffer);
  void CompileKernel();

  bool done;
  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  size_t bufferSize;
  cl_mem buffer;
  cl_mem clImage2D;
  cl_mem bufferImage;
  cl_mem bufferOut;
  cl_uint pitchAlignment;
};

#endif  // _OCLBufferFromImage_H_
