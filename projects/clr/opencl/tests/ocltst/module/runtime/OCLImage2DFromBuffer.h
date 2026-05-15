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

#ifndef _OCLImage2DFromBuffer_H_
#define _OCLImage2DFromBuffer_H_

#include "OCLTestImp.h"

class OCLImage2DFromBuffer : public OCLTestImp {
 public:
  OCLImage2DFromBuffer();
  virtual ~OCLImage2DFromBuffer();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int imageWidth;
  static const unsigned int imageHeight;

  void testReadImage(cl_mem image);
  void testKernel();
  void AllocateOpenCLImage();
  void CopyOpenCLImage(cl_mem clImageSrc);
  void CompileKernel();

  bool done_;
  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  cl_mem buffer;
  cl_mem clImage2DOriginal;
  cl_mem clImage2D;
  cl_mem clImage2DOut;
  cl_uint pitchAlignment;
};

#endif  // _OCLImage2DFromBuffer_H_
