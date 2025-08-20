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

#ifndef _OCL_DX11_YUY2_H_
#define _OCL_DX11_YUY2_H_

#include "OCLDX11Common.h"

class OCLDX11YUY2 : public OCLDX11Common {
 public:
  OCLDX11YUY2();
  virtual ~OCLDX11YUY2();

  virtual void open(unsigned int test, char* units, double& conversion, unsigned int deviceId);
  virtual void run(void);
  virtual unsigned int close(void);

 protected:
  static const unsigned int WIDTH = 1280;
  static const unsigned int HEIGHT = 720;

  void testInterop();
  void AllocateOpenCLImage();
  bool CheckCLImage(cl_mem clImage);
  bool CheckCLImageY(cl_mem clImage);
  bool CheckCLImageUV(cl_mem clImage);
  void CopyOpenCLImage(cl_mem clImageSrc);
  void CompileKernel();
  bool formatSupported();
  void testFormat();

  size_t blockSizeX; /**< Work-group size in x-direction */
  size_t blockSizeY; /**< Work-group size in y-direction */
  cl_mem clImage2DOut;
  DXGI_FORMAT dxFormat;
};

#endif  // _OCL_DX11_YUY2_H_
